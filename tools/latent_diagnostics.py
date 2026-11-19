"""Why does the 30-s dFC latent space look the way it does? Notebook 08's diagnostics, computed once.

Two parts, each its own SLURM array task (slurm/latent_diagnostics.sbatch), written where
notebook 08 reads them (cohort_tables.diagnostics_dir):

    <outputs>/modelling/version=<v>/unsupervised_model/latent_diagnostics/atlas=/window_s=/

  --part stats      one pass over every cohort's dFC edges, no fit:
      window_stats.parquet     per window: mean |r|, mean r, SD of r over the edges, volumes
                               (n_tr_effective). Does the Cam-CAN (~12 volumes per 30 s) differ?
      person_stats.parquet     per (cohort, task, person): each edge's variance over that person's
                               windows, averaged over edges, beside the volumes per window -- the
                               sampling noise of a short-window correlation grows as 1 / (n - 1)
      state_edges_<states>_train.csv           each state's mean edge vector (raw r), training
      state_edges_<states>_train_centred.csv   cohorts; and the same minus the mean of every
                                               training window. Give either to notebook 04 as
                                               STATES: an edge table becomes node strength
                                               (contextualization.correlations.node_strength).

  --part umap --variant V   one UMAP refit, then every cohort's sampled windows put through it:
      umap_<V>.parquet         cohort, task, sub, window_id, stimulus_start_s, n_tr_effective,
                               state (the official label), x, y, z
      umap_<V>.json            what was fitted

    base           as decompose: z-score per edge (fit on the training windows), UMAP on 30,000
                   random training windows. The reference the others are compared with.
    all_rows       UMAP on every training window (--umap-fit-rows 0)
    no_scaler      no z-score per edge before the UMAP (raw r)
    person_scaled  each edge z-scored within each PERSON before the UMAP (every window of that
                   person, all tasks): the "before" twin of STIM's per-person z-score, which they
                   apply AFTER the reduction (that one needs no refit; notebook 08 does it)
    nndb_only      fitted on NNDb alone, as the old version of the project was

The training cohorts, their tasks, the censor policy and the seed are read from the official fit
(the latents' own metadata), so `base` reproduces it up to the sample. The windows put through each
UMAP are the same in every variant: at most --max-windows per person, drawn with --seed.

    python tools/latent_diagnostics.py --atlas harvardoxford --window-s 30 --part stats
    python tools/latent_diagnostics.py --atlas harvardoxford --window-s 30 --part umap --variant all_rows
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fmri_decomposition.io import (default_output_root, latents_root,  # noqa: E402
                                   unsupervised_root)
from fmri_decomposition.unsupervised_model import decompose as dc  # noqa: E402

VARIANTS = ("base", "all_rows", "no_scaler", "person_scaled", "nndb_only")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def out_dir(root, atlas, window_s, version=None) -> Path:
    return (unsupervised_root(root, version) / "latent_diagnostics" / f"atlas={atlas}"
            / f"window_s={window_s}")


def official_fit(root, atlas, window_s, version=None) -> tuple[dict, list[str]]:
    """(the official fit's description, every cohort in the cell), from the latents' metadata."""
    base = latents_root(root, atlas, window_s, version=version)
    files = sorted(base.glob("cohort=*/task=*/data.parquet"))
    if not files:
        raise SystemExit(f"no latents under {base}: run decompose first")
    cohorts = sorted({p.parent.parent.name.split("=", 1)[1] for p in files})
    for p in files:
        stored, _ = dc.latents_fit_description(p)
        if stored.get("train_cohorts"):
            return stored, cohorts
    raise SystemExit(f"no fit description in any latents file under {base}")


def read(root, atlas, window_s, cohort, edges, policy):
    """One cohort's windows -> (ident, float32 edges), censored as the official fit was."""
    from fmri_decomposition.extractions.censor import load_censor
    cen = load_censor(root, policy, atlas, window_s, cohort)
    ident, X = dc.read_cohort(root, atlas, window_s, cohort, edges, censor=cen)
    ident, X = dc.drop_nan_rows(ident, X)
    ident = ident.reset_index(drop=True)
    ident["cohort"] = cohort
    ident["window_id"] = ident["window_id"].astype("int64")
    return ident, X


def official_states(root, atlas, window_s, cohort, states, version=None) -> pd.DataFrame | None:
    """(task, sub, window_id, state) of one cohort's official labels, or None without that column."""
    import pyarrow.parquet as pq
    parts = []
    for p in sorted(latents_root(root, atlas, window_s, cohort, version=version).glob("task=*/data.parquet")):
        if states not in pq.read_schema(p).names:
            continue
        t = pd.read_parquet(p, columns=["task", "sub", "window_id", states]).dropna(subset=[states])
        parts.append(t)
    if not parts:
        return None
    t = pd.concat(parts, ignore_index=True).rename(columns={states: "state"})
    t["task"], t["sub"] = t["task"].astype(str), t["sub"].astype(str)
    t["window_id"] = t["window_id"].astype("int64")
    t["state"] = t["state"].astype(int)
    return t


def with_states(ident, lab):
    """ident + a `state` column (-1 where the window has no official label)."""
    if lab is None:
        return ident.assign(state=-1)
    k = ["task", "sub", "window_id"]
    m = ident[k].astype({"task": str, "sub": str}).merge(lab, on=k, how="left")
    return ident.assign(state=m["state"].fillna(-1).astype(int).to_numpy())


def sample(ident, max_windows, seed) -> np.ndarray:
    """Row indices: at most max_windows per person, at random (the same rows in every variant)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ident))
    rank = ident.iloc[perm].groupby("sub").cumcount().to_numpy()
    return np.sort(perm[rank < max_windows])


def zscore_by(groups, X, eps=1e-6):
    """Each column z-scored within each group, in place."""
    for _, idx in pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).groups.items():
        idx = np.asarray(idx)
        B = X[idx].astype(np.float64)
        X[idx] = ((B - B.mean(0)) / np.maximum(B.std(0), eps)).astype(np.float32)


# --------------------------------------------------------------------------------- part: stats
def part_stats(a, root, fit, cohorts, edges, d):
    train = set(fit["train_cohorts"])
    train_tasks = set(map(str, fit.get("train_tasks") or []))
    win, per, sums = [], [], {}
    grand, n_grand = np.zeros(len(edges)), 0
    for c in cohorts:
        t0 = time.time()
        ident, X = read(root, a.atlas, a.window_s, c, edges, fit.get("censor_policy"))
        absx = np.abs(X)
        win.append(ident[["cohort", "task", "sub", "window_id", "stimulus_start_s", "n_tr_effective"]]
                   .assign(mean_abs_r=absx.mean(1), mean_r=X.mean(1), sd_r=X.std(1)))
        del absx
        for (task, sub), idx in ident.groupby(["task", "sub"]).groups.items():
            B = X[np.asarray(idx)].astype(np.float64)
            if len(B) < 3:
                continue
            per.append(dict(cohort=c, task=task, sub=sub, n_windows=len(B),
                            n_tr=float(ident.loc[idx, "n_tr_effective"].median()),
                            mean_r=float(B.mean()), edge_var=float(B.var(0, ddof=1).mean())))
        if c in train:
            lab = official_states(root, a.atlas, a.window_s, c, a.states, a.modelling_version)
            if lab is None:
                log(f"  {c}: no {a.states} column in the latents -- no state edges from it")
            else:
                s = with_states(ident, lab)["state"].to_numpy()
                keep = s >= 0
                if train_tasks:
                    keep &= ident["task"].astype(str).isin(train_tasks).to_numpy()
                for k in np.unique(s[keep]):
                    m = keep & (s == k)
                    acc = sums.setdefault(int(k), [np.zeros(len(edges)), 0])
                    acc[0] += X[m].astype(np.float64).sum(0)
                    acc[1] += int(m.sum())
                grand += X[keep].astype(np.float64).sum(0)
                n_grand += int(keep.sum())
        log(f"  {c}: {len(ident):,} windows ({time.time() - t0:.0f} s)")
        del ident, X
        gc.collect()
    pd.concat(win, ignore_index=True).to_parquet(d / "window_stats.parquet", index=False)
    pd.DataFrame(per).to_parquet(d / "person_stats.parquet", index=False)
    log(f"-> {d / 'window_stats.parquet'}, {d / 'person_stats.parquet'}")
    if sums:
        ks = sorted(sums)
        n = np.array([sums[k][1] for k in ks], float)
        M = np.array([sums[k][0] for k in ks]) / n[:, None]
        mean_all = grand / n_grand
        for name, T in (("train", M), ("train_centred", M - mean_all)):
            t = pd.DataFrame(T, index=pd.Index(ks, name="state"), columns=edges)
            t.insert(0, "n_rows", n.astype(int))
            t.insert(1, "share", n / n.sum())
            f = d / f"state_edges_{a.states}_{name}.csv"
            t.to_csv(f)
            log(f"-> {f}")


# ---------------------------------------------------------------------------------- part: umap
def part_umap(a, root, fit, cohorts, edges, d):
    import umap
    from sklearn.preprocessing import StandardScaler
    v = a.variant
    seed = int(fit.get("seed", 42))
    train = ["ds002837"] if v == "nndb_only" else list(fit["train_cohorts"])
    train_tasks = set(map(str, fit.get("train_tasks") or []))
    fit_rows = 0 if v == "all_rows" else int(fit.get("umap_fit_rows") or 30_000)
    fit_parts, ev_id, ev_X = [], [], []
    for c in cohorts:
        t0 = time.time()
        ident, X = read(root, a.atlas, a.window_s, c, edges, fit.get("censor_policy"))
        if v == "person_scaled":
            zscore_by(ident["sub"].astype(str).to_numpy(), X)
        if c in train:
            m = (ident["task"].astype(str).isin(train_tasks).to_numpy() if train_tasks
                 else np.ones(len(ident), bool))
            fit_parts.append(X[m] if not m.all() else X)
        s = sample(ident, a.max_windows, a.seed)
        lab = official_states(root, a.atlas, a.window_s, c, a.states, a.modelling_version)
        ev_id.append(with_states(ident.iloc[s].reset_index(drop=True), lab))
        ev_X.append(X[s].copy())
        log(f"  {c}: {len(ident):,} windows, {len(s):,} put through the UMAP ({time.time() - t0:.0f} s)")
        if c not in train:
            del X
        del ident
        gc.collect()
    Xf = np.vstack(fit_parts) if len(fit_parts) > 1 else fit_parts[0]
    del fit_parts
    Xe = np.vstack(ev_X)
    del ev_X
    gc.collect()
    if v in ("base", "all_rows", "nndb_only"):
        sc = StandardScaler(copy=False).fit(Xf)
        Xf = sc.transform(Xf)
        Xe = sc.transform(Xe)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(Xf), fit_rows, replace=False) if fit_rows and len(Xf) > fit_rows
           else np.arange(len(Xf)))
    log(f"{v}: fitting UMAP on {len(idx):,} of {len(Xf):,} training windows ({', '.join(train)})")
    t0 = time.time()
    model = umap.UMAP(n_components=3, random_state=seed).fit(Xf[idx])
    log(f"  fitted in {time.time() - t0:.0f} s; transforming {len(Xe):,} windows")
    n_train = len(Xf)
    del Xf
    gc.collect()
    Y = model.transform(Xe)
    E = pd.concat(ev_id, ignore_index=True)[["cohort", "task", "sub", "window_id", "stimulus_start_s",
                                             "n_tr_effective", "state"]]
    E[["x", "y", "z"]] = Y
    E["role"] = np.where(E["cohort"].isin(train), "train", "projected")
    E.to_parquet(d / f"umap_{v}.parquet", index=False)
    (d / f"umap_{v}.json").write_text(json.dumps(dict(
        variant=v, train_cohorts=train, train_tasks=sorted(train_tasks) or None, fit_rows=int(len(idx)),
        n_train_windows=int(n_train),
        scaler={"base": "per edge, training windows", "all_rows": "per edge, training windows",
                "nndb_only": "per edge, NNDb windows", "no_scaler": None,
                "person_scaled": "per edge within each person"}[v],
        censor_policy=fit.get("censor_policy"), seed=seed, sample_seed=a.seed, max_windows=a.max_windows,
        states=a.states, umap="umap.UMAP(n_components=3, random_state=seed), defaults otherwise"), indent=1))
    log(f"-> {d / f'umap_{v}.parquet'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas", default="harvardoxford")
    ap.add_argument("--window-s", default="30", help="as in the dfc/latents paths, e.g. 30")
    ap.add_argument("--part", choices=["stats", "umap"], required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="base")
    ap.add_argument("--states", default="HMM2_umap3_8", help="the official state set (labels, state edges)")
    ap.add_argument("--max-windows", type=int, default=60, help="windows per person put through each UMAP")
    ap.add_argument("--seed", type=int, default=0, help="which windows are put through the UMAP")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--modelling-version", default=None)
    a = ap.parse_args(argv)
    if str(a.window_s) == "-1":
        raise SystemExit("this is for the dFC windows; the per-TR space (-1) is read by notebook 08 directly")
    root = Path(a.output_root) if a.output_root else default_output_root()
    fit, cohorts = official_fit(root, a.atlas, a.window_s, a.modelling_version)
    log(f"official fit: train {fit['train_cohorts']}, censor {fit.get('censor_policy')}, "
        f"umap_fit_rows {fit.get('umap_fit_rows')}; cohorts in the cell: {cohorts}")
    first = sorted(dc.shard_paths(root, a.atlas, a.window_s, cohorts[0]))
    if not first:
        raise SystemExit(f"no dfc shards for {cohorts[0]} at {a.atlas} / {a.window_s}")
    edges = dc.edge_columns(first[0])
    d = out_dir(root, a.atlas, a.window_s, a.modelling_version)
    d.mkdir(parents=True, exist_ok=True)
    log(f"{len(edges)} edges -> {d}")
    (part_stats if a.part == "stats" else part_umap)(a, root, fit, cohorts, edges, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
