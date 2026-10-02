#!/usr/bin/env python3
"""Stage 5b -- which state set predicts a phenotype column best.

    fmri-decomp select --target additional_HADS_anx_category
    fmri-decomp select --target additional_HADS_anx_category \\
        --models ridge hgb lgbm --n-jobs 16

writes, per target,

    outputs/selection/target=<t>/scores.parquet     every (state set, model, seed)
    outputs/selection/target=<t>/summary.csv        the ranking, readable
    outputs/selection/target=<t>/figures/*.png      the comparison plots
    outputs/selection/target=<t>/models/*.joblib    refit artifacts for the top N
    outputs/meta/selection/target=<t>.json          manifest

WHY A SCRIPT AND NOT THE NOTEBOOK
---------------------------------
The grid is state sets x models x fold seeds, and a boosted model is ~800 fits
at 16 state sets and 5 seeds -- tens of minutes in a kernel, and it blocks the
notebook while it runs. It is also embarrassingly parallel: every cell of the
grid is independent, so it belongs in a batch job with `--n-jobs`.

The notebook keeps the volcano, which is seconds.

THE GRID
--------
    state set   atlas x window_s x states (K)   discovered from outputs/
    features    cells | summary | cells+summary
    model       ridge | hgb | lgbm | logistic
    fold seed   --seeds

`cells` is the hypothesis: the K*K transition probabilities. `summary` is the
CONTROL -- occupancy, switch rate, dwell, entropy rate, dispersion, everything
about a subject that is not a transition cell. If `summary` predicts as well as
`cells`, the signal is where someone SITS and how long they stay, not how they
move, and the transition claim does not hold. `figures/cells_vs_summary.png`
puts the two on one axis.

Every feature set is prepended with the covariates, so each is measured against
the same covariates-only baseline.

WHAT THE SCORE IS FOR
---------------------
camcan is the DISCOVERY cohort. The score ranks candidate state sets; it is not
an effect size. Choosing the best of ~16-24 biases the winner's score upward
even when nothing is real, so the number to quote is "state set X ranked first",
never "transitions explain Y% of anxiety". Confirmation is a different dataset.

That is also why the ranking is repeated over fold seeds and the spread is
reported: a winner that changes with the seed has not been selected, it has been
sampled.

THE BASELINE
------------
Covariates alone, which every state set has to beat. It depends on `window_s`
(through `n_transitions`) but NOT on atlas or K -- the same subjects and the
same transition counts serve every atlas at a given aperture. So it is computed
once per (window_s, model, seed) and reused, rather than recomputed identically
for every state set.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import meta_dir

# The per-subject features that are NOT transition cells. `occ_*` is added to
# this per state set, since its width is K.
#
# n_windows, n_transitions and n_transitions_independent are deliberately NOT
# here: all three are the same quantity (n_transitions_independent is
# n_windows // n_overlaps - 1, and n_transitions is n_windows minus the
# boundary drops), and n_transitions is already a nuisance COVARIATE. Putting
# them in as predictors would hand the model its own covariate back, nearly
# collinear with it.
#
# switch_rate and self_transition_rate sum to 1 by construction. Both are kept
# -- ridge is untroubled by it and dropping one would make this block something
# other than "everything else".
SUMMARY_PREDICTORS = [
    "n_states_visited", "n_distinct_transitions", "switch_rate",
    "self_transition_rate", "switches_per_min", "mean_dwell_s",
    "entropy_rate_bits", "dispersion",
]

FEATURE_SETS = ["cells", "summary", "cells+summary"]

# How a transition cell is expressed.
#   joint  n(i->j) / all of this subject's transitions. As stored. The whole
#          table sums to 1, and the denominator is the same for every subject.
#   cond   n(i->j) / n(i->.), the Markov transition probability -- each ROW
#          sums to 1. Derived here by multiplying back to counts and
#          renormalising, so it costs nothing and needs no re-run of
#          `transitions`.
# `summary` holds no cells, so it is never expanded over this axis -- doing so
# would run the identical arm twice.
P_NORMS = ["joint", "cond"]

CAMCAN = Path("/project/6008063/tamires/cohorts/camcan/dataman/useraccess/"
              "opendata/paule_toussaint_camcan01870")
DEFAULT_PHENO = [f"{CAMCAN / 'approved_data.tsv'}:\t",
                 f"{CAMCAN / 'standard_data.csv'}:,"]
ORDINAL_LEVELS = ["Normal", "Mild", "Moderate", "Severe"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------- estimators ---
def make_model(name: str, seed: int):
    """(estimator, kind, label). `kind` is 'reg' or 'clf', which picks the metric.

    Every estimator is single-threaded on purpose: the parallelism is over grid
    cells via joblib, and a threaded estimator inside a joblib worker
    oversubscribes the node and runs slower than either alone.

    Boosting hyperparameters are deliberately small. Library defaults target
    tens of thousands of rows; here it is ~600 subjects against 64-729
    correlated, compositional features, where a default-sized ensemble
    memorises the training fold and the ranking turns to noise.
    """
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "ridge":
        return (make_pipeline(StandardScaler(),
                              RidgeCV(alphas=np.logspace(-2, 4, 25))),
                "reg", "ridge")

    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return (HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.05, max_leaf_nodes=7,
                    min_samples_leaf=30, l2_regularization=1.0,
                    early_stopping=False, random_state=seed),
                "reg", "hgb (sklearn histogram boosting)")

    if name == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise SystemExit(
                "lightgbm is not installed in this environment.\n"
                "  * use --models hgb, which is sklearn's histogram boosting "
                "and the same algorithm family, or\n"
                "  * rebuild the container: containers/stage45.def pins "
                "lightgbm==4.5.0")
        return (LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7,
                              min_child_samples=30, colsample_bytree=0.5,
                              subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                              n_jobs=1, verbose=-1, random_state=seed),
                "reg", "lgbm")

    if name == "logistic":
        return (make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=0.1,
                                                 class_weight="balanced")),
                "clf", "logistic (Normal vs above)")

    raise SystemExit(f"unknown model {name!r}; choose from "
                     f"ridge, hgb, lgbm, logistic")


def score_once(X: np.ndarray, y: np.ndarray, model: str, seed: int) -> float:
    """One out-of-fold score. Never accuracy -- with the upper ordinal levels
    this thin, predicting Normal for everyone scores well and means nothing."""
    from scipy import stats
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold

    est, kind, _ = make_model(model, seed)
    if kind == "clf":
        yb = (y > 0).astype(int)
        if yb.sum() < 10 or (1 - yb).sum() < 10:
            return np.nan
        pred = np.empty(len(yb), float)
        for tr, te in StratifiedKFold(5, shuffle=True,
                                      random_state=seed).split(X, yb):
            pred[te] = clone(est).fit(X[tr], yb[tr]).predict_proba(X[te])[:, 1]
        return float(roc_auc_score(yb, pred))

    pred = np.empty(len(y), float)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        pred[te] = clone(est).fit(X[tr], y[tr]).predict(X[te])
    return float(stats.spearmanr(pred, y).statistic)


def metric_name(model: str) -> str:
    return "AUC" if model == "logistic" else "Spearman"


# --------------------------------------------------------------- the data ---
def read_phenotype(specs: list[str], id_col: str, target: str,
                   covariates: list[str], categorical: list[str]) -> pd.DataFrame:
    """Merge the release tables on the id column, code the ordinal, coerce types.

    Cam-CAN splits what is needed: the HADS categories are in one table and Age
    and Sex in another, both keyed by CCID. A missing covariate is therefore a
    missing FILE, not a missing column, which is worth saying in the error.
    """
    frames = []
    for spec in specs:
        path_s, _, sep = spec.rpartition(":")
        path = Path(path_s or spec)
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
        f = pd.read_csv(path, sep=sep or ",", dtype={id_col: str})
        if id_col not in f.columns:
            raise SystemExit(f"{id_col!r} not in {path.name}: "
                             f"{list(f.columns)[:20]}")
        f[id_col] = f[id_col].astype(str).str.strip().str.upper()
        log(f"  {path.name:<24} {len(f):>6,} row(s) {len(f.columns):>4} col(s)")
        frames.append(f)

    pheno = frames[0]
    for f in frames[1:]:
        dup = [c for c in f.columns if c in pheno.columns and c != id_col]
        pheno = pheno.merge(f.drop(columns=dup), on=id_col, how="outer")
    pheno = pheno.rename(columns={id_col: "sub"})

    lower = {c.strip().lower(): c for c in pheno.columns}
    want = [target] + [c for c in covariates if c != "n_transitions"]
    missing = [w for w in want if w.strip().lower() not in lower]
    if missing:
        near = [c for c in pheno.columns
                if any(k in c.lower() for k in ("hads", "age", "sex"))]
        raise SystemExit(f"not in any phenotype file: {missing}\n"
                         f"close names: {near[:14]}")
    for w in want:
        got = lower[w.strip().lower()]
        if got != w:
            pheno[w] = pheno[got]
            log(f"  matched {w!r} -> column {got!r}")

    raw = pheno[target].astype("string").str.strip()
    codes = pd.to_numeric(raw, errors="coerce")
    codes = codes.where(codes.notna(),
                        raw.str.lower().map({v.lower(): i for i, v
                                             in enumerate(ORDINAL_LEVELS)}))
    unknown = sorted(set(raw.dropna()) - set(ORDINAL_LEVELS)
                     - set(str(i) for i in range(len(ORDINAL_LEVELS))))
    if unknown:
        log(f"  values not in ORDINAL_LEVELS, now NaN: {unknown[:6]}")
    pheno["y"] = codes

    for c in covariates:
        if c in pheno.columns and c not in categorical:
            pheno[c] = pd.to_numeric(pheno[c], errors="coerce")

    keep = ["sub", "y"] + [c for c in covariates if c != "n_transitions"]
    pheno = pheno[keep].dropna()
    vc = pheno["y"].astype(int).value_counts().sort_index()
    log(f"  usable labels: {len(pheno):,}  "
        + "  ".join(f"{ORDINAL_LEVELS[int(k)]}={v}" for k, v in vc.items()
                    if int(k) < len(ORDINAL_LEVELS)))
    return pheno


def discover(root: Path) -> pd.DataFrame:
    rows = [{"atlas": p.parts[-5].split("=")[1],
             "window_s": p.parts[-4].split("=")[1],
             "states": p.parts[-3].split("=")[1],
             "cohort": p.parts[-2].split("=")[1], "path": p}
            for p in (root / "transitions").rglob("subjects.parquet")]
    if not rows:
        raise SystemExit(f"no transition tables under {root / 'transitions'} -- "
                         f"run `fmri-decomp transitions` first")
    d = pd.DataFrame(rows)
    d["K"] = d["states"].str.rsplit("_", n=1).str[1].astype(int)
    d["window_s_num"] = d["window_s"].astype(float)
    return d


def design(cov: pd.DataFrame) -> np.ndarray:
    X = pd.get_dummies(cov, drop_first=True, dummy_na=False).astype(float)
    return X.fillna(X.mean()).to_numpy()


def to_cond(block: np.ndarray, n_transitions: np.ndarray, K: int) -> np.ndarray:
    """p_joint -> p_cond. Rows of an unvisited from-state stay at exact zero."""
    M = (block * n_transitions[:, None]).reshape(len(block), K, K)   # counts
    rs = M.sum(axis=2, keepdims=True)
    return np.divide(M, rs, out=np.zeros_like(M),
                     where=rs > 0).reshape(len(block), K * K)


def build(table: Path, pheno: pd.DataFrame, covariates: list[str],
          feature_sets: list[str], p_norms: list[str]):
    """One state set -> ({feature set: matrix}, covariates-only matrix, y, sizes).

    Covariates are prepended to every feature set, so each one has to beat the
    covariates-only baseline on the same footing.

      cells          the K*K transition probabilities -- the hypothesis
      summary        everything else per subject: occupancy, switch rate, dwell,
                     entropy rate, dispersion. The CONTROL -- if this predicts
                     as well as `cells`, the signal is not in the transitions.
      cells+summary  both, to see whether the summaries add anything
    """
    t = pd.read_parquet(table)
    t["sub"] = t["sub"].astype(str).str.strip().str.upper()
    d = t.merge(pheno, on="sub", how="inner", suffixes=("", "_pheno"))
    if d.empty:
        raise SystemExit(
            f"{table}: the phenotype join matched nothing.\n"
            f"  transitions sub e.g. {t['sub'].iloc[:3].tolist()}\n"
            f"  phenotype   sub e.g. {pheno['sub'].iloc[:3].tolist()}")

    cells = sorted((c for c in t.columns if "->" in c),
                   key=lambda c: tuple(int(x) for x in c.split("->")))
    occ = sorted((c for c in t.columns if c.startswith("occ_")),
                 key=lambda c: int(c.split("_")[1]))
    summary = [c for c in SUMMARY_PREDICTORS if c in t.columns] + occ
    blocks = {"cells": cells, "summary": summary,
              "cells+summary": cells + summary}

    C = design(d[covariates])
    K = int(t["n_states"].iloc[0])
    n_tr = d["n_transitions"].to_numpy(float)
    joint = d[cells].to_numpy(float)
    cond = to_cond(joint, n_tr, K) if cells else joint
    summ = d[summary].to_numpy(float) if summary else np.empty((len(d), 0))

    mats, widths = {}, {}
    for fs in feature_sets:
        # summary carries no cells, so the normalisation axis does not apply
        norms = ["-"] if fs == "summary" else p_norms
        for nm in norms:
            cellblock = joint if nm == "joint" else cond
            parts = [C]
            if "cells" in fs:
                parts.append(cellblock)
            if "summary" in fs:
                parts.append(summ)
            cols = blocks[fs]
            if not cols:
                continue
            mats[(fs, nm)] = np.column_stack(parts)
            widths[(fs, nm)] = len(cols)
    return mats, C, d["y"].to_numpy(float), widths, len(d)


# ------------------------------------------------------------------- run ---
def run(args) -> int:
    from joblib import Parallel, delayed

    root = Path(args.output_root) if args.output_root else _default_root()
    sets = discover(root)
    sets = sets[sets["cohort"] == args.cohort]
    if args.atlas:
        sets = sets[sets["atlas"].isin(args.atlas)]
    if args.window_s:
        sets = sets[sets["window_s"].isin([str(w) for w in args.window_s])]
    if sets.empty:
        raise SystemExit("no state set matches --atlas / --window-s / --cohort")
    sets = sets.sort_values(["atlas", "K", "window_s_num"]).reset_index(drop=True)

    log(f"phenotype for target={args.target!r}")
    pheno = read_phenotype(args.pheno, args.id_col, args.target,
                           args.covariates, args.categorical)

    log(f"{len(sets)} state set(s) x {len(args.models)} model(s) x "
        f"{len(args.seeds)} seed(s)")

    # Load once. Each table is ~100 KB, so the whole grid is a few MB and
    # re-reading it inside every worker would dominate the runtime.
    data, meta = {}, []
    for t in sets.itertuples():
        key = (t.atlas, t.window_s, t.states)
        mats, C, y, widths, n = build(t.path, pheno, args.covariates,
                                      args.features, args.p_norm)
        data[key] = (mats, C, y)
        for (fs, nm), w in widths.items():
            meta.append({"atlas": t.atlas, "window_s": t.window_s_num,
                         "states": t.states, "K": t.K, "features": fs,
                         "p_norm": nm, "n": n, "n_features": w})
    meta = pd.DataFrame(meta)
    log(f"  subjects per state set: {meta['n'].min()}-{meta['n'].max()}")
    for (fs, nm), g in meta.groupby(["features", "p_norm"]):
        log(f"    {fs + ('' if nm == '-' else f' [{nm}]'):<22} "
            f"{sorted(g['n_features'].unique())} predictor(s) "
            f"+ {len(args.covariates)} covariate(s)")

    # The baseline depends on window_s only -- same subjects, same
    # n_transitions, every atlas, K and feature set. One per
    # (window_s, model, seed), not one per state set.
    jobs = [("set", k, arm, m, s)
            for k in data for arm in data[k][0]
            for m in args.models for s in args.seeds]
    base_keys = {}
    for k in data:
        base_keys.setdefault(k[1], k)          # first state set at this window
    n_base = len(base_keys) * len(args.models) * len(args.seeds)
    jobs += [("base", base_keys[w], ("(covariates only)", "-"), m, s)
             for w in base_keys for m in args.models for s in args.seeds]
    log(f"  {len(jobs)} fit job(s) ({len(jobs) - n_base} state-set "
        f"+ {n_base} baseline)")

    def one(kind, key, arm, model, seed):
        mats, C, y = data[key]
        return score_once(mats[arm] if kind == "set" else C, y, model, seed)

    t0 = time.time()
    scores = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(one)(kind, key, arm, m, s) for kind, key, arm, m, s in jobs)
    log(f"  fitted in {time.time() - t0:.0f}s")

    rows = []
    for (kind, key, arm, model, seed), sc in zip(jobs, scores):
        atlas, window_s, states = key
        fs, nm = arm
        rows.append({"atlas": atlas if kind == "set" else "(baseline)",
                     "window_s": float(window_s),
                     "states": states if kind == "set" else "(covariates only)",
                     "K": int(states.rsplit("_", 1)[1]) if kind == "set" else 0,
                     "features": fs, "p_norm": nm, "model": model,
                     "metric": metric_name(model),
                     "seed": seed, "kind": kind, "score": sc})
    scores_df = pd.DataFrame(rows).merge(
        meta, on=["atlas", "window_s", "states", "K", "features", "p_norm"],
        how="left")
    # one readable label per arm, used by every figure and the ranking
    scores_df["arm"] = (scores_df["features"]
                        + np.where(scores_df["p_norm"] == "-", "",
                                   " [" + scores_df["p_norm"] + "]"))

    out = root / "selection" / f"target={args.target}"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(out / "scores.parquet", index=False)

    summary = (scores_df.groupby(["model", "arm", "atlas", "window_s", "K",
                                  "states"], dropna=False)["score"]
               .agg(["mean", "std", "min", "max", "count"]).reset_index()
               .sort_values(["model", "mean"], ascending=[True, False]))
    summary.to_csv(out / "summary.csv", index=False)
    print()
    for model, g in summary.groupby("model"):
        print(f"=== {model}  ({metric_name(model)}) ===")
        print(g.drop(columns=["model", "states"])
               .head(args.show).to_string(index=False))
        print()

    note = design_note(args, sets, meta)
    (out / "DESIGN.md").write_text(note)
    print(note)

    _figures(scores_df, summary, out / "figures", args.target)
    saved = _save_models(scores_df, data, out / "models", args.save_top,
                         args.covariates)

    mf = meta_dir(root) / "selection" / f"target={args.target}.json"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(
        {"target": args.target, "cohort": args.cohort, "models": args.models,
         "seeds": args.seeds, "features": args.features,
         "p_norm": args.p_norm, "covariates": args.covariates,
         "fixed_not_compared": {
             "method": "quantile-binned PCA-3 (ThresholdCluster)",
             "feature": "windowed DFC edges, not per-TR activation",
             "pca_components": 3, "n_overlaps": 5,
             "boost_hyperparameters": "fixed, not tuned (ridge's alpha is)"},
         "n_state_sets": int(len(sets)), "n_jobs_run": len(jobs),
         "saved_models": saved, "output": str(out.relative_to(root)),
         "note": "camcan is the discovery cohort; these scores RANK state sets "
                 "and are not effect sizes",
         "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, default=str))
    log(f"-> {out.relative_to(root)}  (scores, summary, figures, models)")
    log(f"-> {mf.relative_to(root)}")
    return 0


def design_note(args, sets, meta) -> str:
    """What this run varied and what it held fixed, in the output folder.

    Written beside the scores because six months on, "which state set won" is
    useless without "won against what, holding what constant" -- and the fixed
    list is the part nobody writes down.
    """
    n_arms = meta[["features", "p_norm"]].drop_duplicates().shape[0]
    n_sets = sets[["atlas", "window_s", "states"]].drop_duplicates().shape[0]
    combos = n_sets * n_arms * len(args.models)
    lines = [
        f"# Model selection design — target = {args.target}",
        "",
        f"Written {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}.",
        "",
        "## Compared",
        "",
        "| axis | n | values |",
        "|---|---|---|",
        f"| atlas | {sets['atlas'].nunique()} | "
        f"{', '.join(sorted(sets['atlas'].unique()))} |",
        f"| window_s (aperture) | {sets['window_s_num'].nunique()} | "
        f"{', '.join(f'{w:g}' for w in sorted(sets['window_s_num'].unique()))} |",
        f"| state definition (K) | {sets['states'].nunique()} | "
        f"{', '.join(sorted(sets['states'].unique()))} |",
        f"| feature set | {len(args.features)} | {', '.join(args.features)} |",
        f"| cell normalisation | {len(args.p_norm)} | "
        f"{', '.join(args.p_norm)}  (not applied to `summary`) |",
        f"| regressor | {len(args.models)} | {', '.join(args.models)} |",
        "",
        f"**{combos} configurations**, each repeated over "
        f"{len(args.seeds)} fold seeds {args.seeds} — the seeds are a "
        f"reliability check, not a choice.",
        "",
        "## Held fixed — NOT compared",
        "",
        "A state set is `(method, feature, atlas, aperture, K)`. This run varies",
        "atlas, aperture and K. The other two are fixed:",
        "",
        "| | fixed at | the alternative not tested |",
        "|---|---|---|",
        "| **method** | quantile-binned PCA-3 (`ThresholdCluster`) | k-means, HMM, GMM |",
        "| **feature** | windowed DFC edges | per-TR activation |",
        "| PCA components | 3 | 2, 5, 10 |",
        "| window overlap | `n_overlaps=5` (80%) | less overlap |",
        "| censor policy | the one the latents were built under | other gates |",
        f"| covariates | {', '.join(args.covariates)} | + education, + handedness |",
        "| cross-validation | 5-fold | repeated / nested |",
        "",
        "## One asymmetry between the regressors",
        "",
        "`ridge` has its penalty tuned inside each fit (RidgeCV over 25 alphas).",
        "The boosted models do **not** — their hyperparameters are fixed, chosen",
        "conservatively for n in the hundreds. So a boosted model losing is partly",
        "a statement about those settings; a boosted model winning is stronger,",
        "because it won while handicapped.",
        "",
        "## What the scores are for",
        "",
        "camcan is the **discovery** cohort. These scores RANK candidates; they",
        "are not effect sizes. Choosing the best of many biases the winner's score",
        "upward even when nothing is real, so quote \"state set X ranked first\",",
        "never \"transitions explain Y% of the variance\". Confirmation is a",
        "different dataset.",
        "",
        "`figures/cells_vs_summary.png` is the control: if the non-transition",
        "summaries predict as well as the cells, the signal is where someone sits",
        "and how long they stay, not how they move.",
        "",
    ]
    return "\n".join(lines)


def _figures(scores, summary, fig_dir: Path, target: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = scores[scores["kind"] == "set"]
    base = scores[scores["kind"] == "base"]
    models = sorted(sets["model"].unique())

    # 1. score vs aperture, one panel per model, with the baseline as a floor
    featsets = sorted(sets["arm"].unique())
    fig, axes = plt.subplots(len(featsets), len(models),
                             figsize=(5.2 * len(models), 4.3 * len(featsets)),
                             squeeze=False)
    for r, fs in enumerate(featsets):
      for ax, model in zip(axes[r], models):
        sub = sets[(sets["model"] == model) & (sets["arm"] == fs)]
        for (atlas, K), g in sub.groupby(["atlas", "K"]):
            m = g.groupby("window_s")["score"].agg(["mean", "std"])
            ax.errorbar(m.index, m["mean"], yerr=m["std"], marker="o",
                        capsize=3, label=f"{atlas} K={K}")
        b = base[base["model"] == model].groupby("window_s")["score"].mean()
        ax.plot(b.index, b.values, "k--", lw=1.3, label="covariates only")
        ax.axhline(0 if model != "logistic" else .5, lw=.6, c="grey")
        ax.set_xscale("log")
        ax.set_xticks(sorted(sets["window_s"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("window_s")
        ax.set_ylabel(f"out-of-fold {metric_name(model)}")
        ax.set_title(f"{model} — features: {fs}", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"{target} — state set vs aperture", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "score_by_window.png", dpi=150)
    plt.close(fig)

    # 2. rank stability: a winner that moves with the fold seed is not a winner
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6),
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        g = sets[sets["model"] == model].assign(
            label=lambda d: d["arm"] + " | " + d["atlas"] + " "
            + d["window_s"].astype(str) + "s K" + d["K"].astype(str))
        rank = g.pivot_table(index="label", columns="seed",
                             values="score").rank(ascending=False)
        order = rank.mean(axis=1).sort_values().index
        ax.imshow(rank.loc[order], aspect="auto", cmap="viridis_r")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=6)
        ax.set_xticks(range(rank.shape[1]))
        ax.set_xticklabels(rank.columns)
        ax.set_xlabel("fold seed")
        ax.set_title(f"{model}: rank by seed (1 = best)", fontsize=9)
        for i, lab in enumerate(order):
            for j, c in enumerate(rank.columns):
                ax.text(j, i, int(rank.loc[lab, c]), ha="center", va="center",
                        fontsize=5, color="w")
    fig.tight_layout()
    fig.savefig(fig_dir / "rank_stability.png", dpi=150)
    plt.close(fig)

    # 3. do the models agree on the winner?
    piv = (sets.groupby(["model", "arm", "atlas", "window_s", "K"])["score"]
               .mean().reset_index()
               .assign(label=lambda d: d["arm"] + " | " + d["atlas"] + " "
                       + d["window_s"].astype(str) + "s K" + d["K"].astype(str))
               .pivot(index="label", columns="model", values="score"))
    if piv.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(6, 1 + .3 * len(piv)))
        r = piv.rank(ascending=False)
        ax.imshow(r, aspect="auto", cmap="viridis_r")
        ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index, fontsize=6)
        ax.set_xticks(range(r.shape[1])); ax.set_xticklabels(r.columns, fontsize=8)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                ax.text(j, i, int(r.iloc[i, j]), ha="center", va="center",
                        fontsize=6, color="w")
        ax.set_title("rank by model — agreement means the winner is robust",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "model_agreement.png", dpi=150)
        plt.close(fig)


    # 4. cells vs summary, head to head. This is the control: if `summary`
    #    tracks the phenotype as well as `cells`, the signal is occupancy and
    #    dwell rather than the transitions themselves.
    if len(featsets) > 1:
        pair = (sets.groupby(["model", "arm", "atlas", "window_s", "K"])
                    ["score"].mean().reset_index()
                    .pivot_table(index=["model", "atlas", "window_s", "K"],
                                 columns="arm", values="score").dropna())
        cellarm = next((c for c in pair.columns if c.startswith("cells [")), None)
        if cellarm and "summary" in pair.columns and len(pair):
            pair = pair.rename(columns={cellarm: "cells"})
            fig, ax = plt.subplots(figsize=(5.4, 5.2))
            for model, g in pair.reset_index().groupby("model"):
                ax.scatter(g["summary"], g["cells"], s=34, alpha=.8, label=model)
            lo = float(min(pair["summary"].min(), pair["cells"].min()))
            hi = float(max(pair["summary"].max(), pair["cells"].max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            bl = base["score"].mean()
            ax.axhline(bl, color="grey", lw=.7, ls=":")
            ax.axvline(bl, color="grey", lw=.7, ls=":")
            ax.set_xlabel("summary features (occupancy, dwell, entropy, ...)")
            ax.set_ylabel("transition cells")
            ax.set_title("above the diagonal = the transitions carry something\n"
                         "the summaries do not  (dotted = covariates-only)",
                         fontsize=9)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "cells_vs_summary.png", dpi=150)
            plt.close(fig)


def _save_models(scores, data, model_dir: Path, top_n: int,
                 covariates: list[str]) -> list[str]:
    """Refit the top state sets on ALL subjects and store the fitted object.

    Refit on everything on purpose: these are artifacts to carry to the
    validation dataset, not the thing the score came from. The score came from
    held-out folds and is recorded separately.
    """
    import joblib

    sets = scores[scores["kind"] == "set"]
    saved = []
    for (model, fs), g in sets.groupby(["model", "arm"]):
        best = (g.groupby(["atlas", "window_s", "states"])["score"].mean()
                 .sort_values(ascending=False).head(top_n))
        for (atlas, window_s, states), mean_score in best.items():
            key = (atlas, str(int(window_s)) if float(window_s).is_integer()
                   else str(window_s), states)
            if key not in data:
                continue
            mats, C, y = data[key]
            base, _, nm = fs.partition(" [")
            arm = (base, nm.rstrip("]") or "-")
            if arm not in mats:
                continue
            est, kind, label = make_model(model, 0)
            est.fit(mats[arm], (y > 0).astype(int) if kind == "clf" else y)
            tag = fs.replace("+", "-").replace(" [", "-").replace("]", "")
            name = f"{model}__{tag}__{atlas}__w{key[1]}__{states}.joblib"
            joblib.dump({"estimator": est, "model": model, "label": label,
                         "features": arm[0], "p_norm": arm[1],
                         "atlas": atlas, "window_s": key[1], "states": states,
                         "covariates": covariates, "n": len(y),
                         "mean_cv_score": float(mean_score),
                         "note": "refit on all subjects; the score came from "
                                 "held-out folds, not from this fit"},
                        model_dir / name)
            saved.append(name)
    return saved


def _default_root() -> Path:
    if os.environ.get("FMRIDECOMP_OUTPUTS"):
        return Path(os.environ["FMRIDECOMP_OUTPUTS"])
    import yaml
    repo = Path(__file__).resolve().parent.parent
    return Path(yaml.safe_load(
        (repo / "config" / "camcan_movie.yaml").read_text())["output_root"])


def add_arguments(p) -> None:
    p.add_argument("--target", required=True,
                   help="phenotype column, e.g. additional_HADS_anx_category")
    p.add_argument("--models", nargs="+", default=["ridge", "hgb"],
                   choices=["ridge", "hgb", "lgbm", "logistic"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--p-norm", nargs="+", default=P_NORMS, choices=P_NORMS,
                   help="how a transition cell is expressed: joint (share of "
                        "all this subject's transitions, as stored) or cond "
                        "(the Markov probability, each row summing to 1). "
                        "Ignored for the `summary` feature set, which has no "
                        "cells.")
    p.add_argument("--features", nargs="+",
                   default=["cells", "summary", "cells+summary"],
                   choices=FEATURE_SETS,
                   help="cells = the K*K transition probabilities (the "
                        "hypothesis); summary = everything else per subject "
                        "(occupancy, switch rate, dwell, entropy, dispersion) "
                        "-- the control that asks whether the signal is in the "
                        "transitions at all")
    p.add_argument("--pheno", nargs="+", default=DEFAULT_PHENO,
                   metavar="PATH:SEP",
                   help="phenotype tables as path:separator, merged on --id-col")
    p.add_argument("--id-col", default="CCID")
    p.add_argument("--covariates", nargs="+",
                   default=["Age", "Sex", "n_transitions"])
    p.add_argument("--categorical", nargs="+", default=["Sex"])
    p.add_argument("--cohort", default="camcan")
    p.add_argument("--atlas", nargs="*", default=None)
    p.add_argument("--window-s", nargs="*", default=None)
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="parallel fits; match --cpus-per-task")
    p.add_argument("--save-top", type=int, default=3,
                   help="refit and store this many state sets per model")
    p.add_argument("--show", type=int, default=12,
                   help="rows of the ranking to print per model")
    p.add_argument("--output-root")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
