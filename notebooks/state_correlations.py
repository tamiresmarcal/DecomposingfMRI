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
                         n_frames: int = 6000, frame_noise: float = 2.0,
                         pattern_noise: float = 0.45) -> pd.DataFrame:
    """Frames -> labels -> per-state mean, in `state-means`' CSV layout."""
    rng = np.random.default_rng(seed)
    P = parc.n
    cen = parc.centroids
    d = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
    smooth = np.exp(-(d / 25.0) ** 2)                 # 25 mm Gaussian kernel
    smooth /= smooth.sum(1, keepdims=True)

    patterns = []
    for mix in SIM_STATES.values():
        x = sum(w * components[c] for c, w in mix.items())
        x = (x - x.mean()) / (x.std() or 1.0)
        noise = smooth @ rng.normal(size=P)
        x = x + pattern_noise * noise / (noise.std() or 1.0)
        patterns.append((x - x.mean()) / x.std())
    patterns = np.array(patterns)
    K = len(patterns)

    land = rng.dirichlet(np.full(K, 3.0))
    labels = np.empty(n_frames, dtype=int)
    labels[0] = rng.choice(K, p=land)
    for t in range(1, n_frames):
        labels[t] = labels[t - 1] if rng.random() < 0.9 else rng.choice(K, p=land)
    frames = patterns[labels] + frame_noise * rng.normal(size=(n_frames, P))

    df = pd.DataFrame(frames, columns=parc.columns)
    df["state"] = labels
    g = df.groupby("state")
    out = g[parc.columns].mean()
    out.insert(0, "n_rows", g.size())
    out.insert(1, "share", out["n_rows"] / n_frames)
    out.insert(2, "name", list(SIM_STATES))
    return out


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
                output_root=None, window_s=None, cohorts=None) -> pd.DataFrame:
    if states == "simulate":
        return simulate_state_means(parc, components, seed)
    if Path(states).suffix == ".csv":
        return checked(pd.read_csv(states, index_col="state"), parc, states)
    from fmri_decomposition.state_means import means_in_feature_units

    if window_s is None:
        raise SystemExit(f"STATES={states!r} is a state column: set WINDOW_S (--window-s), "
                         f"the aperture it was fitted on -- -1 for the per-TR activation states")
    root = Path(output_root) if output_root else default_output_root()
    t = means_in_feature_units(root, atlas, window_s, states, cohorts)
    return checked(t, parc, f"{states} at atlas={atlas} window_s={window_s}")


def default_output_root() -> Path:
    """Where `fmri-decomp state-means` looks when not told: $FMRIDECOMP_OUTPUTS,
    else `output_root` in config/camcan_movie.yaml."""
    import os

    if os.environ.get("FMRIDECOMP_OUTPUTS"):
        return Path(os.environ["FMRIDECOMP_OUTPUTS"])
    import yaml

    return Path(yaml.safe_load((rm.REPO / "config" / "camcan_movie.yaml").read_text())["output_root"])


# ------------------------------------------------------------- reference ---
def reference_families(data: Path, ref, families, behavioural_csv=None, seed: int = 7) -> dict:
    """Voxel-level reference maps for the families asked for, built once and
    shared across atlases (the PET resampling is the slow part)."""
    out = {}
    if "molecular" in families:
        out["molecular"] = rm.receptor_maps(data, ref)
    if {"behavioural", "behavioural_domain"} & set(families):
        table = rm.load_behavioural_table(behavioural_csv) if behavioural_csv else None
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
            output_root=None, window_s=None, cohorts=None, behavioural_csv=None,
            n_nulls: int = 10000, null: str = "moran", seed: int = 7,
            verbose: bool = True) -> dict:
    """Write the tables for every atlas; return {atlas: its directory}."""
    families = [f for f in FAMILIES if f in families]
    ref = rm.reference_grid(data)
    refs = reference_families(data, ref, families, behavioural_csv, seed)
    subcort = rm.subcortical_structures(data, ref)
    label = "simulated" if states == "simulate" else Path(states).stem
    if window_s is not None and states != "simulate" and Path(states).suffix != ".csv":
        label = f"{states}_window-{window_s}"
    dirs = {}
    for atlas in atlases:
        spec, src = rm.state_atlas(atlas, data)
        parc = rm.Parcellator.build(spec, ref)
        d = Path(out) / label / atlas
        d.mkdir(parents=True, exist_ok=True)

        parcel = {f: parc.parcellate(m.voxels) for f, m in refs.items()}
        comps = {n: parcel["yeo7"][k] for k, n in enumerate(rm.YEO7)}
        comps.update({g: parc.parcellate(v) for g, v in subcort.items()})
        sm = load_states(states, atlas, parc, comps, seed, output_root, window_s, cohorts)
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
    p.add_argument("--behavioural", help="real Brainnetome behavioural table (README.md)")
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
            behavioural_csv=a.behavioural, n_nulls=a.n_nulls, null=a.null, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
