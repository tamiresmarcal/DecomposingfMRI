#!/usr/bin/env python3
"""Brain states x reference maps: the correlation TABLES. No figures.

    python notebooks/state_correlations.py --states simulate
    python notebooks/state_correlations.py --states HMM2_pca14_10 \\
        --window-s -1 --atlases harvardoxford networks_nodes --output-root /path/outputs

For each state atlas, the states' mean per parcel is correlated across parcels
with every reference map, parcellated into the same parcels:

    molecular            19 PET receptor/transporter maps (neuromaps / Hansen 2022)
    behavioural_domain   BrainMap's 5 domains over Brainnetome      } one table,
    behavioural          BrainMap's subcategories over Brainnetome  } two depths
    yeo7                 Yeo 2011's 7 networks
    networks14           this repo's 14 coordinate networks

Each run writes one directory per atlas under <out>/<states>/<atlas>/:

    state_means.csv     the states, in `fmri-decomp state-means --csv` layout
    correlations.csv    one row per state x map: r, p_null, q_fdr
    provenance.json     where the atlas, the states and every family came from

`state_figures.py` draws from exactly these files, so a figure never
recomputes a number.

States: `simulate` builds six states with a known answer (a signed mix of Yeo
networks and deep structures, sampled as noisy frames and averaged back per
label); a state COLUMN (e.g. HMM2_pca14_10) is read through
`fmri_decomposition.state_means` -- the same table `fmri-decomp state-means`
prints -- for the atlas and aperture given; a path to a CSV is taken as that
table already written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refmaps as rm  # noqa: E402

ATLASES = ["harvardoxford", "networks_nodes", "networks", "yeo7"]
FAMILIES = ["molecular", "behavioural_domain", "behavioural", "yeo7", "networks14"]
BN_LEVELS = {"domain": "behavioural_domain", "subcategory": "behavioural"}

# Ground truth of the simulation: state -> {component: weight}. Components are
# Yeo networks and Brainnetome's deep gyri.
SIM_STATES = {
    "Default": {"Default": 1.0, "Dorsal Attention": -0.6, "Somatomotor": -0.3, "Hipp": 0.5},
    "Visual": {"Visual": 1.0, "Default": -0.4, "Tha": 0.3},
    "Sensorimotor": {"Somatomotor": 1.0, "Ventral Attention": 0.4, "Frontoparietal": -0.4,
                     "BG": 0.6, "Tha": 0.4},
    "Task-positive": {"Dorsal Attention": 0.8, "Frontoparietal": 0.8, "Default": -0.6, "BG": 0.3},
    "Limbic": {"Limbic": 1.0, "Default": 0.3, "Visual": -0.4, "Amyg": 0.9, "Hipp": 0.6},
    "Salience": {"Ventral Attention": 1.0, "Somatomotor": 0.3, "Default": -0.5,
                 "Tha": 0.5, "Amyg": 0.3},
}


# ------------------------------------------------------------------ states ---
def simulate_state_means(parc: rm.Parcellator, components: dict, seed: int,
                         n_subjects: int = 30, n_frames: int = 400, frame_noise: float = 2.0,
                         subject_noise: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subjects -> frames -> labels -> per-subject and group state means.

    Each subject expresses each state as the group pattern plus its own
    spatially smooth deviation; frames are drawn from a sticky Markov chain
    and averaged back per label. Returns (group table in `state-means`' CSV
    layout, per-subject table indexed by (subject, state)).
    """
    rng = np.random.default_rng(seed)
    P = parc.n
    cen = parc.centroids
    d = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
    smooth = np.exp(-(d / 25.0) ** 2)                 # 25 mm Gaussian kernel
    smooth /= smooth.sum(1, keepdims=True)

    def z(x):
        return (x - x.mean()) / (x.std() or 1.0)

    group = np.array([z(sum(w * components[c] for c, w in mix.items()))
                      for mix in SIM_STATES.values()])
    K = len(group)
    land = rng.dirichlet(np.full(K, 3.0))
    rows, counts = [], np.zeros(K)
    for s_ in range(n_subjects):
        pat = np.array([z(g + subject_noise * z(smooth @ rng.normal(size=P))) for g in group])
        lab = np.empty(n_frames, dtype=int)
        lab[0] = rng.choice(K, p=land)
        for t in range(1, n_frames):
            lab[t] = lab[t - 1] if rng.random() < 0.9 else rng.choice(K, p=land)
        fr = pat[lab] + frame_noise * rng.normal(size=(n_frames, P))
        for k in range(K):
            if (lab == k).any():
                rows.append([f"sim{s_ + 1:02d}", k] + list(fr[lab == k].mean(0)))
        counts += np.bincount(lab, minlength=K)
    subj = pd.DataFrame(rows, columns=["subject", "state"] + parc.columns).set_index(["subject", "state"])
    out = subj.groupby(level="state").mean()
    out.insert(0, "n_rows", counts.astype(int))
    out.insert(1, "share", counts / counts.sum())
    out.insert(2, "name", list(SIM_STATES))
    return out, subj


def node_strength(t: pd.DataFrame, parc: rm.Parcellator) -> pd.DataFrame | None:
    """DFC states -> one value per parcel: its nodal strength.

    A state fitted on windowed connectivity has, as its mean, an edge vector
    (`A__B` columns, the atlas's upper triangle), not an activation map. The
    map that can be set beside receptor densities or networks is each parcel's
    mean connectivity with every other parcel in that state. Returns None when
    the table is not edges.
    """
    edges = parc.spec.edge_names()
    if not set(edges) <= set(t.columns):
        return None
    P = parc.n
    iu, ju = np.triu_indices(P, k=1)
    E = t[edges].to_numpy(float)
    M = np.zeros((len(t), P, P))
    M[:, iu, ju] = E
    M[:, ju, iu] = E
    out = pd.DataFrame(M.sum(2) / (P - 1), index=t.index, columns=parc.columns)
    for c in ("n_rows", "share", "name"):
        if c in t:
            out.insert(len([x for x in ("n_rows", "share", "name") if x in out]), c, t[c])
    out.attrs["measure"] = "node strength (mean connectivity with every other parcel)"
    return out


def checked(t: pd.DataFrame, parc: rm.Parcellator, what: str) -> pd.DataFrame:
    """Keep the atlas's columns. One the atlas has and the table lacks means
    the states came from another atlas: hard error, not a silent subset.
    Edge tables (DFC states) are first reduced to node strength."""
    strength = node_strength(t, parc)
    if strength is not None:
        print(f"[{parc.spec.name}] {what}: connectivity states -> node strength per parcel")
        t = strength
    missing = [c for c in parc.columns if c not in t.columns]
    if missing:
        raise SystemExit(f"{what}: {len(missing)} of {parc.n} `{parc.spec.name}` columns "
                         f"missing, e.g. {missing[:3]} -- states from another atlas?")
    measure = t.attrs.get("measure", "mean activation")
    t = t.copy()
    t.attrs["measure"] = measure
    if "name" not in t:
        t.insert(0, "name", [f"state {s}" for s in t.index])
    return t[["n_rows", "share", "name"] + parc.columns]


def load_states(states: str, atlas: str, parc, components, seed: int,
                output_root=None, window_s=None, cohorts=None):
    """-> (group table, per-subject table or None). The per-subject table is
    indexed by (subject, state), one column per parcel -- what the error bars
    (SD over subjects) are computed from."""
    if states == "simulate":
        return simulate_state_means(parc, components, seed)
    if Path(states).suffix == ".csv":
        return checked(pd.read_csv(states, index_col="state"), parc, states), None
    root = Path(output_root) if output_root else default_output_root()
    if window_s is None:
        window_s = find_window(root, atlas, states)
    if cohorts == "train":                 # the cohorts decompose fitted on, from the model
        from fmri_decomposition.state_means import load_models

        cohorts = list(load_models(root, atlas, window_s)["train_cohorts"])
        print(f"[{atlas}] COHORTS='train' -> {cohorts}")
    grp, subj = state_means_any_layout(root, atlas, window_s, states, cohorts)
    what = f"{states} at atlas={atlas} window_s={window_s}"
    return checked(grp, parc, what), subject_maps(subj, parc, what)


def subject_maps(subj: pd.DataFrame, parc: rm.Parcellator, what: str) -> pd.DataFrame:
    """Per-subject state means in parcel units: edges -> node strength, as for the group."""
    strength = node_strength(subj, parc)
    t = strength if strength is not None else subj
    missing = [c for c in parc.columns if c not in t.columns]
    if missing:
        raise SystemExit(f"{what} (per subject): {len(missing)} columns missing, e.g. {missing[:3]}")
    return t[parc.columns]


def _has_column(path: Path, col: str) -> bool:
    import pyarrow.parquet as pq

    return col in pq.ParquetFile(path).schema_arrow.names


def latent_files(root: Path, atlas: str, window_s, cohorts=None) -> list:
    """Every latents file of one (atlas, aperture) cell, in either layout:
    `cohort=<c>/data.parquet` or `cohort=<c>/task=<t>/data.parquet`."""
    from fmri_decomposition.io import latents_root

    cell = latents_root(root, atlas, window_s)
    out = []
    for p in sorted(cell.rglob("data.parquet")):
        cohort = next((x.split("=", 1)[1] for x in p.parts if x.startswith("cohort=")), None)
        if cohorts is None or cohort in cohorts:
            out.append(p)
    return out


def find_window(root: Path, atlas: str, state_col: str):
    """The aperture whose latents carry `state_col`, when WINDOW_S is not given."""
    from fmri_decomposition.io import latents_root

    found = []
    for wdir in sorted(latents_root(root, atlas).glob("window_s=*")):
        w = wdir.name.split("=", 1)[1]
        if any(_has_column(p, state_col) for p in latent_files(root, atlas, w)):
            found.append(w)
    if len(found) != 1:
        raise SystemExit(f"{state_col!r} is in {found or 'no'} window_s of atlas={atlas} "
                         f"under {root / 'latents'}: set WINDOW_S"
                         + ("" if found else " and check ATLASES / OUTPUT_ROOT"))
    print(f"[{atlas}] {state_col}: window_s={found[0]} (found in the latents)")
    return found[0]


def state_means_any_layout(root: Path, atlas: str, window_s, state_col: str,
                           cohorts=None):
    """Group and per-subject state means, in ACTIVATION, for any state set.

    Same route for PCA, raw and UMAP state sets: no inverse transform. The
    labels are joined back to the per-TR parcel activations in
    outputs/activation/, so a state's map is always the mean activation over
    the time it covers:

      window_s = -1   each label is a frame: joined on (task, sub, window_id),
                      frames rebuilt with the fit's band-pass and z-scoring
                      (read from the saved model)
      window_s > 0    each label is a DFC window; a window covers the frames
                      with stimulus_time_s in [stimulus_start_s,
                      stimulus_start_s + window_s) -- windows.py's own rule --
                      so a state's map is the mean over the frames its windows
                      cover (overlapping windows weight a frame once each).
                      Frames use frames.py's defaults (band-pass 0.01-0.1 Hz,
                      z-scored per run), the activation fit's defaults.

    Only the (task, sub) the latents have are read: the set decompose fitted
    on, after its censor policy. Returns (group table, per-subject table
    indexed by (subject, state)).
    """
    from fmri_decomposition import frames
    from fmri_decomposition.io import read_file

    files = [p for p in latent_files(root, atlas, window_s, cohorts) if _has_column(p, state_col)]
    if not files:
        raise SystemExit(f"no latents with {state_col!r} at atlas={atlas} window_s={window_s}"
                         f"{'' if cohorts is None else f' for cohorts {cohorts}'} under {root}")
    per_tr = str(window_s) == "-1"
    if per_tr:
        from fmri_decomposition.state_means import load_models

        models = load_models(root, atlas, window_s)
        # absent = the fit used frames' default band; None = "as extracted"
        band = models["match_bandpass"] if "match_bandpass" in models else frames.DEFAULT_BANDPASS
        zscore = bool(models.get("zscore_runs", True))
    else:
        band, zscore, width = frames.DEFAULT_BANDPASS, True, float(window_s)

    keys = ["task", "sub", "window_id"] + ([] if per_tr else ["stimulus_start_s"])
    by_cohort: dict = {}
    for p in files:
        cohort = next((x.split("=", 1)[1] for x in p.parts if x.startswith("cohort=")), "")
        t = read_file(p, columns=keys + [state_col]).to_pandas().dropna(subset=[state_col])
        by_cohort.setdefault(cohort, []).append(t)

    sums: dict = {}
    features = None
    skipped: list = []
    for cohort, parts in by_cohort.items():
        lab = pd.concat(parts, ignore_index=True)
        lab["task"], lab["sub"] = lab["task"].astype(str), lab["sub"].astype(str)
        lab["window_id"] = lab["window_id"].astype("int64")
        lab[state_col] = lab[state_col].astype(int)
        kept = set(zip(lab["task"], lab["sub"]))
        shards = frames.shard_paths(root, atlas, cohort)
        if not shards:
            raise SystemExit(f"no activation shards for cohort={cohort} at atlas={atlas}: "
                             f"the state maps are read from outputs/activation/")
        feats = frames.feature_columns(shards[0])
        if features is None:
            features = feats
        ident, X, dropped = _read_frames(root, atlas, cohort, feats, band, zscore, kept)
        skipped += dropped
        ident = ident.reset_index(drop=True)
        ident["task"], ident["sub"] = ident["task"].astype(str), ident["sub"].astype(str)
        for (task, sub), fi in ident.groupby(["task", "sub"], sort=False).groups.items():
            L = lab[(lab["task"] == task) & (lab["sub"] == sub)]
            if L.empty:
                continue
            fi = np.asarray(fi)
            if per_tr:
                pos = pd.Series(np.arange(len(fi)), index=ident.loc[fi, "window_id"].astype("int64"))
                hit = L["window_id"].map(pos)
                ok = hit.notna().to_numpy()
                rows, states = fi[hit[ok].astype(int).to_numpy()], L[state_col].to_numpy()[ok]
                W = None
            else:
                tt = ident.loc[fi, "stimulus_start_s"].to_numpy(float)   # frame stimulus time
                s0 = L["stimulus_start_s"].to_numpy(float)[:, None]
                W = (tt[None] >= s0 - 1e-6) & (tt[None] < s0 + width - 1e-6)   # windows x frames
                states = L[state_col].to_numpy()
            for k in np.unique(states):
                if W is None:
                    w_rows = rows[states == k]
                    tot, n = X[w_rows].astype(np.float64).sum(0), len(w_rows)
                else:
                    wts = W[states == k].sum(0).astype(np.float64)        # frame weights
                    n = wts.sum()
                    if n == 0:
                        continue
                    tot = wts @ X[fi].astype(np.float64)
                acc = sums.setdefault((f"{cohort}:{sub}", int(k)), [0.0, 0.0])
                acc[0] = acc[0] + tot
                acc[1] += n
    if not sums:
        raise SystemExit(f"{state_col}: no activation frames matched the latents at "
                         f"atlas={atlas} window_s={window_s}")
    idx = pd.MultiIndex.from_tuples(sorted(sums), names=["subject", "state"])
    S = np.array([sums[k][0] for k in idx])
    N = np.array([sums[k][1] for k in idx], dtype=float)
    subj = pd.DataFrame(S / N[:, None], index=idx, columns=features)
    states = idx.get_level_values("state")
    tot = pd.Series(N, index=states).groupby(level=0).sum()
    grp = pd.DataFrame(pd.DataFrame(S, index=states).groupby(level=0).sum().to_numpy()
                       / tot.to_numpy()[:, None], index=tot.index, columns=features)
    n_rows = pd.concat([pd.concat(v) for v in by_cohort.values()])[state_col] \
        .astype(int).value_counts().reindex(grp.index).fillna(0).to_numpy()
    grp.insert(0, "n_rows", n_rows.astype(int))
    grp.insert(1, "share", n_rows / n_rows.sum())
    grp.index.name = "state"
    grp.attrs["measure"] = "mean activation" + ("" if per_tr else
                                                f" over each state's {window_s} s windows")
    if skipped:
        print(f"[{atlas}] {state_col}: {len(skipped)} subject(s) left out -- their activation "
              f"frames cannot be assembled into one time series:")
        for who, why in skipped:
            print(f"    {who}: {why.splitlines()[0][:160]}")
    print(f"[{atlas}] {state_col}: {len(grp)} states, activation from "
          f"{subj.index.get_level_values(0).nunique()} subjects "
          f"({'frames' if per_tr else f'frames under {int(n_rows.sum())} windows'})")
    return grp, subj


def _read_frames(root, atlas, cohort, feats, band, zscore, kept):
    """Per-TR activation frames for the latents' subjects -> (ident, X, dropped).

    Two things make this cheap across many state sets and robust to bad data:

    * a cache: one cohort's frames do not depend on the state set, so they are
      read once per (atlas, cohort, band, z-score) into
      state_maps_data/cache/ and every later state set only filters them;
    * one refused subject does not sink the cohort. frames.read_cohort refuses
      a subject whose acquisitions overlap in stimulus time (two runs on one
      span) -- which the dfc stage may still have windowed, so it can be in the
      latents. The refused subject is named in the error; it is dropped and the
      cohort re-read in ONE pass, not subject by subject.
    """
    import re

    from fmri_decomposition import frames

    tag = f"{atlas}_{cohort}_band-{'none' if not band else '-'.join(map(str, band))}_z-{int(zscore)}"
    cache = Path(__file__).resolve().parent / "state_maps_data" / "cache" / f"frames_{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        ident = pd.DataFrame({c: z[f"ident_{c}"] for c in z["ident_cols"]})
        X, dropped = z["X"], [tuple(d) for d in z["dropped"]]
    else:
        everyone = {tuple(x) for x in _all_subjects(frames.shard_paths(root, atlas, cohort))}
        dropped, subjects = [], set(everyone)
        while True:
            try:
                ident, X = frames.read_cohort(
                    root, atlas, cohort, feats, band=band, zscore=zscore, log=lambda *_: None,
                    censor={"policy": "readable", "subjects": subjects, "windows": None})[:2]
                break
            except SystemExit as e:
                msg = str(e)
                bad = {x for x in subjects
                       if re.search(rf"sub-{re.escape(x[1])}(?![A-Za-z0-9])", msg)
                       and re.search(rf"task-{re.escape(x[0])}(?![A-Za-z0-9])", msg)}
                if not bad:
                    raise
                for x in sorted(bad):
                    dropped.append((f"{cohort}:{x[1]} ({x[0]})", msg.splitlines()[0][:160]))
                subjects -= bad
        cache.parent.mkdir(parents=True, exist_ok=True)
        ident = ident.reset_index(drop=True)
        np.savez(cache, X=X, ident_cols=np.array(list(ident.columns)),
                 dropped=np.array(dropped, dtype=object).reshape(-1, 2),
                 **{f"ident_{c}": ident[c].to_numpy() for c in ident.columns})
    keep = np.array([(str(t), str(s_)) in kept for t, s_ in zip(ident["task"], ident["sub"])])
    dropped = [d for d in dropped if any(f":{k[1]} ({k[0]})" in d[0] for k in kept)]
    return ident[keep].reset_index(drop=True), X[keep], dropped


def _all_subjects(paths) -> set:
    out = set()
    for p in paths:
        kv = dict(x.split("=", 1) for x in p.parts if "=" in x)
        out.add((kv["task"], kv["sub"]))
    return out


def default_output_root() -> Path:
    """Where `fmri-decomp state-means` looks when not told: $FMRIDECOMP_OUTPUTS,
    else `output_root` in config/camcan_movie.yaml."""
    import os

    if os.environ.get("FMRIDECOMP_OUTPUTS"):
        return Path(os.environ["FMRIDECOMP_OUTPUTS"])
    import yaml

    return Path(yaml.safe_load((rm.REPO / "config" / "camcan_movie.yaml").read_text())["output_root"])


# ------------------------------------------------------------- reference ---
def reference_families(data: Path, ref, families, behavioural_csv=None, seed: int = 7,
                       taxonomy: str = "brainmap") -> dict:
    """Voxel-level reference maps for the families asked for, built once and
    shared across atlases (the PET resampling is the slow part)."""
    out = {}
    if "molecular" in families:
        out["molecular"] = rm.receptor_maps(data, ref)
    if {"behavioural", "behavioural_domain"} & set(families):
        if behavioural_csv == "neurosynth":
            table = rm.neurosynth_behavioural_table(data, ref, taxonomy)
        elif behavioural_csv:
            table = rm.load_behavioural_table(behavioural_csv)
        else:
            table = None
        leaves = rm.behavioural_maps(data, ref, table, seed=seed)
        if "behavioural" in families:
            out["behavioural"] = leaves
        if "behavioural_domain" in families:
            out["behavioural_domain"] = rm.behavioural_domains(leaves)
    out["yeo7"] = rm.yeo7_maps(data, ref)          # always: the simulation is built from it
    if "networks14" in families:
        out["networks14"] = rm.network14_maps(ref)
    return out


# ------------------------------------------------------------------- run ---
def compute(states: str = "simulate", atlases=ATLASES, families=FAMILIES, *,
            out: Path = HERE / "state_maps_results", data: Path = HERE / "state_maps_data",
            output_root=None, window_s=None, cohorts=None, behavioural_csv="neurosynth",
            n_nulls: int = 10000, null: str = "moran", seed: int = 7,
            taxonomy: str = "brainmap",
            verbose: bool = True) -> dict:
    """Write the tables for every atlas; return {atlas: its directory}."""
    families = [f for f in FAMILIES if f in families]
    ref = rm.reference_grid(data)
    refs = reference_families(data, ref, families, behavioural_csv, seed, taxonomy)
    subcort = rm.subcortical_structures(data, ref)
    label = "simulated" if states == "simulate" else Path(states).stem
    if window_s is not None and states != "simulate" and Path(states).suffix != ".csv":
        label = f"{states}_window-{window_s}"
    if behavioural_csv == "neurosynth" and taxonomy != "brainmap" and \
            {"behavioural", "behavioural_domain"} & set(families):
        label += f"_{taxonomy}"
    dirs = {}
    for atlas in atlases:
        spec, src = rm.state_atlas(atlas, data)
        parc = rm.Parcellator.build(spec, ref)
        d = Path(out) / label / atlas
        d.mkdir(parents=True, exist_ok=True)

        parcel = {f: parc.parcellate(m.voxels) for f, m in refs.items()}
        comps = {n: parcel["yeo7"][k] for k, n in enumerate(rm.YEO7)}
        comps.update({g: parc.parcellate(v) for g, v in subcort.items()})
        sm, subj = load_states(states, atlas, parc, comps, seed, output_root, window_s, cohorts)
        measure = sm.attrs.get("measure", "mean activation")
        sm.to_csv(d / "state_means.csv")
        X = sm[parc.columns].to_numpy(float)

        rows = []
        for k, fam in enumerate(families):
            res = rm.null_test(X, parcel[fam], parc.centroids, n_nulls,
                               seed=seed + 1000 * k, method=null)
            for i, s in enumerate(sm.index):
                for j, n in enumerate(refs[fam].names):
                    rows.append({"state": s, "state_name": sm["name"].iloc[i], "family": fam,
                                 "map": n, "r": res["r"][i, j], "p_null": res["p"][i, j],
                                 "q_fdr": res["q"][i, j], "map_simulated": refs[fam].simulated})
            if verbose:
                q = res["q"]
                sig = int(np.nansum(q < 0.05)) if np.isfinite(q).any() else "-"
                print(f"[{atlas}] {fam}: {len(refs[fam].names)} maps, q<.05: {sig}")
        pd.DataFrame(rows).to_csv(d / "correlations.csv", index=False)
        if subj is not None:
            subject_correlations(subj, sm, parcel, refs, families).to_parquet(
                d / "subject_correlations.parquet", index=False)
        (d / "provenance.json").write_text(json.dumps({
            "atlas": atlas, "atlas_source": src, "n_parcels": parc.n,
            "states": states, "window_s": window_s, "cohorts": cohorts,
            "state_measure": measure,
            "simulation_truth": SIM_STATES if states == "simulate" else None,
            "families": {f: {"source": refs[f].source, "maps": refs[f].names,
                             "simulated": refs[f].simulated} for f in families},
            "null": (f"{null}, {n_nulls} surrogates" if parc.n >= rm.MIN_PARCELS_FOR_NULL
                     else f"none ({parc.n} parcels: a description, not a sample)"),
            "null_method": null, "seed": seed,
        }, indent=2, ensure_ascii=False))
        if verbose:
            print(f"[{atlas}] {parc.n} parcels ({src}) -> {d}")
        dirs[atlas] = d
    return dirs


def subject_correlations(subj: pd.DataFrame, sm: pd.DataFrame, parcel: dict, refs: dict,
                         families) -> pd.DataFrame:
    """r of every subject's state map with every reference map: the spread the
    figures draw as mean +- SD. A subject who never visited a state has no row."""
    X = subj.to_numpy(float)
    names = dict(zip(sm.index, sm["name"]))
    out = []
    for fam in families:
        R = rm.correlate(X, parcel[fam])                  # rows x maps
        t = pd.DataFrame(R, columns=refs[fam].names)
        t["subject"] = subj.index.get_level_values(0)
        t["state"] = subj.index.get_level_values(1)
        t = t.melt(id_vars=["subject", "state"], var_name="map", value_name="r")
        t["family"] = fam
        out.append(t)
    out = pd.concat(out, ignore_index=True)
    out["state_name"] = out["state"].map(names)
    return out[["subject", "state", "state_name", "family", "map", "r"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", default="simulate",
                   help="`simulate`, a state column (e.g. HMM2_pca14_10) or a state-means CSV")
    p.add_argument("--atlases", nargs="+", default=ATLASES, choices=ATLASES)
    p.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    p.add_argument("--window-s", help="aperture of the state column; -1 = per-TR")
    p.add_argument("--cohorts", nargs="*", default=None)
    p.add_argument("--output-root", help="the pipeline's outputs/ (for a state column)")
    p.add_argument("--behavioural", default="neurosynth",
                   help="`neurosynth` (default): profiles computed from Neurosynth; a CSV: "
                        "the real BrainMap table (README); `none`: the simulated stand-in")
    p.add_argument("--taxonomy", choices=sorted(rm.TAXONOMIES), default="brainmap",
                   help="behavioural categories for the Neurosynth profiles")
    p.add_argument("--n-nulls", type=int, default=10000,
                   help="surrogates per map; the smallest p is 1/(n+1), and BH over 114 "
                        "molecular tests needs it well under .05/114")
    p.add_argument("--null", choices=rm.NULLS, default="moran")
    p.add_argument("--data", type=Path, default=HERE / "state_maps_data")
    p.add_argument("--out", type=Path, default=HERE / "state_maps_results")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    compute(a.states, a.atlases, a.families, out=a.out, data=a.data,
            output_root=a.output_root, window_s=a.window_s, cohorts=a.cohorts,
            behavioural_csv=None if a.behavioural == "none" else a.behavioural, n_nulls=a.n_nulls, null=a.null, seed=a.seed,
            taxonomy=a.taxonomy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
