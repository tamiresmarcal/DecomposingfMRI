#!/usr/bin/env python3
"""Stage 5b' -- how well does a STATIC connectivity matrix predict a phenotype?

    fmri-decomp fcm-select --target additional_HADS_anx_category \\
        --cohorts camcan camcan_rest

writes, per target,

    outputs/fcm_selection/target=<t>/DESIGN.md       what was compared and fixed
    outputs/fcm_selection/target=<t>/scores.parquet  every (cell, arm, model, seed)
    outputs/fcm_selection/target=<t>/summary.csv     the ranking, readable
    outputs/fcm_selection/target=<t>/figures/*.png
    outputs/meta/fcm_selection/target=<t>.json       manifest

THE THIRD TREE
--------------
`bstm_selection` ranks movie state sets, `resting_bstm_selection` ranks rest
state sets -- the same script under two `--output-name` values -- and this
ranks the thing both of them have to beat. Same `summary.csv` shape, so the
three are read side by side.

Static FC is the default in this literature: flatten one correlation matrix
per subject, hand the vector to a regressor. If it predicts HADS as well as a
transition matrix does, then the dynamics are not the biomarker, and saying so
requires measuring it rather than citing it.

WHY IT IS A FAIR COMPARISON
---------------------------
`static-fc` computes these edges from the SAME FRAMES the `window_s = -1`
state arm is fitted on -- `frames.read_cohort`, same band-pass, same per-run
z-score, same `good_frame` gate. So a gap between this tree and
`bstm_selection` is a gap between MODELS, not between preprocessing.

WHAT THE GRID IS
----------------
    cohort      camcan | camcan_rest | ...     movie and rest in one table,
                                               as one `cohort` column
    atlas       whichever have a static-FC table
    arm         edges | edges+global           the hypothesis and its control
    model       ridge | hgb | lgbm | logistic
    fold seed   --seeds

`edges` is the hypothesis: every pairwise correlation. `global` is the control
it has to beat -- the mean and SD of a subject's correlations, two numbers that
describe overall connectivity strength and spread with no topology in them at
all. If two numbers predict as well as 6,105 do, the pattern is not what is
being measured; the amount is.

COVARIATES
----------
Age and Sex from the phenotype, `frac_good_frames` and `n_tr_used` from the
static-FC table itself. That last pair is the one `bstm_selection` has no
equivalent of and it matters more here: scrubbing removes more data from
subjects who move more, fewer frames give noisier edges, and noise attenuates
prediction. Uncontrolled, "less connectivity structure" and "moved more" are
the same column.

Every arm is prepended with them, so each is measured against the same
covariates-only baseline -- one per (cohort, model, seed), since the quality
columns differ by cohort while Age and Sex do not.

READING IT BESIDE THE OTHER TWO
-------------------------------
`n` is in `summary.csv` for exactly this reason: these trees are separate runs
on whoever each one has, so two tables can differ in sample as well as in
score. Check `n` before reading a gap as a result, and if it differs, pass
`--restrict-subjects` the same subject list to all three.

No table here subtracts one row from another. The arms share their folds, so a
difference between two scores is dependent and has no standard error the usual
tests supply.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import static_fc as _sfc
from .bstm_selection import (ORDINAL_LEVELS, _wipe, log, metric_name,
                             read_phenotype, read_subject_list, score_once)
from .io import meta_dir

ARMS = ["edges", "global"]

# Supplied by the static-FC table rather than by the phenotype.
QUALITY_COVARIATES = ["frac_good_frames", "n_tr_used"]

FC_TRANSFORMS = ("z", "r")


def discover(root: Path, cohorts: list[str], atlases: list[str] | None
             ) -> list[tuple[str, str]]:
    """(cohort, atlas) pairs that have a stored static-FC table."""
    cells = []
    for cohort in cohorts:
        found = sorted(p.parts[-3].split("=")[1] for p in
                       (root / "static_fc").glob(
                           f"atlas=*/cohort={cohort}/subjects.parquet"))
        if atlases is not None:
            found = [a for a in found if a in atlases]
        if not found:
            raise SystemExit(
                f"no static FC for cohort={cohort}"
                + (f" at atlas(es) {atlases}" if atlases else "") + ".\n"
                f"  run: fmri-decomp static-fc --cohorts {cohort} "
                f"--atlas harvardoxford yeo7 networks")
        cells += [(cohort, a) for a in found]
    return cells


def one_row_per_subject(d: pd.DataFrame, what: str,
                        tasks: list[str] | None = None) -> pd.DataFrame:
    """Refuse a table with several rows for one subject rather than guess.

    A cohort where a subject watches several films has one row per (task, sub).
    Silently keeping the first would make this a ranking of whichever task
    sorted first.
    """
    if tasks:
        before = sorted(d["task"].unique())
        d = d[d["task"].isin(tasks)]
        if d.empty:
            raise SystemExit(f"--tasks {tasks} matches no row of {what}; "
                             f"it has {before}")
    dup = d["sub"].duplicated(keep=False)
    if not dup.any():
        return d
    raise SystemExit(
        f"{what} has {int(dup.sum())} row(s) for the same subject across "
        f"task(s) {sorted(d.loc[dup, 'task'].unique())[:6]}.\n"
        f"  This stage scores one row per subject and will not pick one or "
        f"average them.\n"
        f"  * --tasks <one task> to choose, or\n"
        f"  * re-run `static-fc --pool subject`, which concatenates a "
        f"subject's tasks into a single correlation.")


def blocks(d: pd.DataFrame, transform: str, max_missing: float
           ) -> tuple[dict[str, np.ndarray], dict]:
    """One cell's table -> {arm: matrix}, plus what was dropped or filled.

    Fisher z by default. The stored value is raw r, which is bounded and whose
    sampling variance depends on the true value; arctanh makes that variance
    roughly constant, which is what a linear model's penalty assumes.
    """
    cols = _sfc.fc_columns(d)
    if not cols:
        raise SystemExit("this static-FC table has no edge columns -- re-run "
                         "`fmri-decomp static-fc`")
    X = d[cols].to_numpy(float)
    if transform == "z":
        # r = +-1 is reachable; arctanh of it is infinite, and one infinity
        # survives mean imputation to poison the whole column. Clipped at the
        # precision of float32 storage rather than at 1.
        X = np.arctanh(np.clip(X, -0.999999, 0.999999))

    keep = np.isnan(X).mean(axis=0) <= max_missing
    dropped = int((~keep).sum())
    X = X[:, keep]
    if not X.shape[1]:
        raise SystemExit(
            f"every one of {dropped} edge(s) is NaN in more than "
            f"{max_missing:.0%} of subjects. An edge is NaN when a parcel is "
            f"empty under that subject's brain mask; check per-parcel "
            f"coverage before raising --fc-max-missing.")
    n_imputed = int(np.isnan(X).sum())
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)

    # The control: overall strength and spread, with every bit of topology
    # thrown away. Computed from the SAME matrix the hypothesis uses, after the
    # same transform and imputation, so the two differ only in what is kept.
    g = np.column_stack([X.mean(axis=1), X.std(axis=1)])
    return ({"edges": X, "global": g},
            {"n_edges_dropped": dropped, "n_values_imputed": n_imputed,
             "n_edges_used": X.shape[1]})


def covariate_block(pheno: pd.DataFrame, quality: pd.DataFrame,
                    subs: list[str], covariates: list[str],
                    categorical: list[str]) -> np.ndarray:
    """Age, Sex and this cohort's data-quality columns, as one design matrix."""
    ph = pheno.set_index("sub").reindex(subs)
    q = quality.set_index("sub").reindex(subs)
    cols = []
    for c in covariates:
        src = q if (c in q.columns and c not in ph.columns) else ph
        if c not in src.columns:
            raise SystemExit(
                f"covariate {c!r} is in neither the phenotype nor the "
                f"static-FC table.\n"
                f"  phenotype: {[x for x in ph.columns if x != 'y']}\n"
                f"  static FC: {sorted(set(q.columns) & set(_sfc.QC_COLUMNS))}")
        cols.append(src[c].rename(c))
    X = pd.concat(cols, axis=1)
    X = pd.get_dummies(X, columns=[c for c in categorical if c in X.columns],
                       drop_first=True, dummy_na=False).astype(float)
    return X.fillna(X.mean()).to_numpy()


def build(root: Path, cells: list[tuple[str, str]], pheno: pd.DataFrame,
          args) -> tuple[dict, pd.DataFrame]:
    """Every (cohort, atlas, arm) matrix, plus one baseline per cohort."""
    # Read once, not once per cell: it is the same file every time, and a
    # per-cell read would let a mid-run edit give two cells different samples.
    keep = (read_subject_list(args.restrict_subjects)
            if args.restrict_subjects else None)
    data, meta = {}, []
    seen_baseline = set()
    for cohort, atlas in cells:
        d = _sfc.read_cohort_table(root, atlas, cohort)
        d["sub"] = d["sub"].astype(str).str.strip().str.upper()
        d = one_row_per_subject(d, f"static_fc atlas={atlas} cohort={cohort}",
                                args.tasks)
        if keep is not None:
            d = d[d["sub"].isin(keep)]
        d = d[d["sub"].isin(set(pheno["sub"]))].sort_values("sub")
        if d.empty:
            raise SystemExit(
                f"the phenotype join matched nothing for cohort={cohort} "
                f"atlas={atlas}.\n"
                f"  static FC sub e.g. {_sfc.read_cohort_table(root, atlas, cohort, ['sub'])['sub'].head(3).tolist()}\n"
                f"  phenotype sub e.g. {pheno['sub'].head(3).tolist()}")
        subs = d["sub"].tolist()

        C = covariate_block(pheno, d, subs, args.covariates, args.categorical)
        y = pheno.set_index("sub").reindex(subs)["y"].to_numpy(float)
        mats, info = blocks(d, args.fc_transform, args.fc_max_missing)

        if cohort not in seen_baseline:
            # Once per cohort, not once per atlas: the covariates are the same
            # subjects and the same quality columns at every atlas, so a second
            # baseline would be the identical fit under another name.
            seen_baseline.add(cohort)
            data[(cohort, "(covariates only)", "-")] = (C, y)
            meta.append({"cohort": cohort, "arm": "(covariates only)",
                         "atlas": "-", "kind": "base", "n": len(subs),
                         "n_features": 0, "n_covariates": C.shape[1]})
        for arm in args.arms:
            data[(cohort, arm, atlas)] = (np.column_stack([C, mats[arm]]), y)
            meta.append({"cohort": cohort, "arm": arm, "atlas": atlas,
                         "kind": "set", "n": len(subs),
                         "n_features": mats[arm].shape[1],
                         "n_covariates": C.shape[1],
                         **(info if arm == "edges" else {})})
    return data, pd.DataFrame(meta)


def design_note(args, meta: pd.DataFrame, n_jobs: int) -> str:
    lines = [
        f"# fcm_selection -- {args.target}", "",
        f"Written {time.strftime('%Y-%m-%d %H:%M:%S')} UTC", "",
        "## What was compared", "", "| axis | values |", "|---|---|",
        f"| cohort | {', '.join(sorted(meta['cohort'].unique()))} |",
        f"| atlas | {', '.join(sorted(a for a in meta['atlas'].unique() if a != '-'))} |",
        f"| arm | {', '.join(sorted(meta['arm'].unique()))} |",
        f"| model | {', '.join(args.models)} |",
        f"| fold seed | {', '.join(str(s) for s in args.seeds)} |",
        "", "## What was held fixed", "", "| held fixed | value |", "|---|---|",
        f"| subjects | {int(meta['n'].min())}"
        + ("" if meta["n"].nunique() == 1 else f"-{int(meta['n'].max())}")
        + " |",
        f"| covariates | {', '.join(args.covariates)} |",
        f"| tasks | {', '.join(args.tasks) if args.tasks else 'every task in each cohort'} |",
        f"| edges expressed as | {'Fisher z' if args.fc_transform == 'z' else 'raw r'} |",
        f"| edges dropped above | {args.fc_max_missing:.0%} missing |",
        "| cross-validation | 5-fold, repeated over the fold seeds |",
        "| boosting hyperparameters | fixed, not tuned (ridge's alpha is) |",
        "", "## Feature widths", "",
        "| cohort | arm | atlas | predictors | + covariates | n |",
        "|---|---|---|---|---|---|",
    ]
    for r in meta.sort_values(["cohort", "arm", "atlas"]).itertuples():
        lines.append(f"| {r.cohort} | {r.arm} | {r.atlas} | {r.n_features} | "
                     f"{r.n_covariates} | {r.n} |")
    lines += [
        "", "## Reading this", "",
        "* Read `n` before reading a gap. This tree, `bstm_selection` and "
        "`resting_bstm_selection` are separate runs on whoever each one has, "
        "so two tables can differ in sample as well as in score. "
        "`--restrict-subjects` pins all three to one list.",
        "* `global` is the control: a subject's mean and SD over every edge, "
        "with no topology in them. If it matches `edges`, the pattern is not "
        "what is being measured -- the amount is.",
        "* The scores are out-of-fold and the arms share their folds, so a "
        "difference between two rows is dependent and is NOT a test. None is "
        "computed here on purpose.",
        "* The best of many cells is biased upward. Quote the order and the "
        "spread across fold seeds, never the number.",
        f"* {n_jobs} fit job(s) ran.",
    ]
    return "\n".join(lines) + "\n"


def run(args) -> int:
    from joblib import Parallel, delayed

    from .bstm_selection import DEFAULT_PHENO

    root = Path(args.output_root) if args.output_root else _sfc._default_root()
    if not args.pheno:
        args.pheno = list(DEFAULT_PHENO)

    cells = discover(root, args.cohorts, args.atlas)
    log(f"phenotype for target={args.target!r}")
    pheno = read_phenotype(
        args.pheno, args.id_col, args.target,
        [c for c in args.covariates if c not in QUALITY_COVARIATES],
        args.categorical, levels=args.ordinal_levels)

    log(f"{len(cells)} (cohort, atlas) cell(s): "
        + "  ".join(f"{c}/{a}" for c, a in cells))
    data, meta = build(root, cells, pheno, args)
    for r in meta.sort_values(["cohort", "arm", "atlas"]).itertuples():
        log(f"    {r.cohort:<14} {r.arm:<18} {r.atlas:<14} "
            f"{r.n_features:>6} predictor(s) + {r.n_covariates} "
            f"covariate(s)  n={r.n}")

    jobs = [(k, m, s) for k in data for m in args.models for s in args.seeds]
    log(f"  {len(jobs)} fit job(s)")
    t0 = time.time()
    scores = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(score_once)(data[k][0], data[k][1], m, s) for k, m, s in jobs)
    log(f"  fitted in {time.time() - t0:.0f}s")

    rows = [{"cohort": k[0], "arm": k[1], "atlas": k[2], "model": m,
             "metric": metric_name(m), "seed": s, "score": sc}
            for (k, m, s), sc in zip(jobs, scores)]
    scores_df = pd.DataFrame(rows).merge(meta, on=["cohort", "arm", "atlas"],
                                         how="left")

    out = root / args.output_name / f"target={args.target}"
    _wipe(out, parent=args.output_name)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(out / "scores.parquet", index=False)

    summary = (scores_df.groupby(["model", "cohort", "arm", "atlas", "n",
                                  "n_features"], dropna=False)["score"]
               .agg(["mean", "std", "min", "max", "count"]).reset_index()
               .sort_values(["model", "mean"], ascending=[True, False]))
    summary.to_csv(out / "summary.csv", index=False)

    print()
    for model, g in summary.groupby("model"):
        print(f"=== {model}  ({metric_name(model)}) ===")
        print(g.drop(columns=["model"]).head(args.show).to_string(index=False))
        print()

    note = design_note(args, meta, len(jobs))
    (out / "DESIGN.md").write_text(note)
    print(note)
    _figures(scores_df, summary, out / "figures", args.target)

    mf = meta_dir(root) / args.output_name / f"target={args.target}.json"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(
        {"target": args.target, "cohorts": args.cohorts, "arms": args.arms,
         "models": args.models, "seeds": args.seeds,
         "covariates": args.covariates, "tasks": args.tasks,
         "fc_transform": args.fc_transform,
         "fc_max_missing": args.fc_max_missing,
         "restrict_subjects": args.restrict_subjects,
         "cells": [{"cohort": c, "atlas": a} for c, a in cells],
         "n_subjects": {str(k): int(v) for k, v in
                        meta.groupby("cohort")["n"].max().items()},
         "n_jobs_run": len(jobs),
         "output": str(out.relative_to(root)),
         "note": "out-of-fold scores; the ORDER is the result, the numbers are "
                 "not effect sizes. Read `n` before reading a gap against "
                 "bstm_selection or resting_bstm_selection.",
         "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, default=str))
    log(f"-> {out.relative_to(root)}")
    log(f"-> {mf.relative_to(root)}")
    return 0


def _figures(scores: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path,
             target: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(scores["model"].unique())
    cohorts = sorted(scores["cohort"].unique())
    arms = ["(covariates only)"] + sorted(
        a for a in scores["arm"].unique() if a != "(covariates only)")

    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.6),
                             squeeze=False)
    width = 0.8 / max(len(cohorts), 1)
    for ax, model in zip(axes[0], models):
        g = summary[summary["model"] == model]
        for ci, cohort in enumerate(cohorts):
            h = g[g["cohort"] == cohort]
            best, err = [], []
            for a in arms:
                k = h[h["arm"] == a]
                best.append(k["mean"].max() if len(k) else np.nan)
                err.append(k.loc[k["mean"].idxmax(), "std"] if len(k)
                           else np.nan)
            ax.bar(np.arange(len(arms)) + ci * width, best, width, yerr=err,
                   capsize=3, label=cohort)
        ax.axhline(0 if model != "logistic" else 0.5, lw=0.6, c="grey")
        ax.set_xticks(np.arange(len(arms)) + width * (len(cohorts) - 1) / 2)
        ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(f"out-of-fold {metric_name(model)} (best atlas per arm)")
        ax.set_title(model, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"{target} -- static connectivity", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "arms_by_cohort.png", dpi=150)
    plt.close(fig)

    # The spread across fold seeds, which is what says whether an order is
    # real. A bar chart of means hides exactly the thing that matters.
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.6),
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        s = scores[scores["model"] == model]
        labels, series = [], []
        for cohort in cohorts:
            for a in arms:
                h = s[(s["cohort"] == cohort) & (s["arm"] == a)]
                if h.empty:
                    continue
                best = h.groupby("atlas")["score"].mean().idxmax()
                labels.append(f"{cohort}\n{a}")
                series.append(h[h["atlas"] == best]["score"].to_numpy())
        if series:
            # `labels=` was renamed `tick_labels=` in matplotlib 3.9. Ask the
            # signature rather than the version string.
            import inspect
            kw = ("tick_labels" if "tick_labels" in
                  inspect.signature(ax.boxplot).parameters else "labels")
            ax.boxplot(series, **{kw: labels})
        ax.axhline(0 if model != "logistic" else 0.5, lw=0.6, c="grey")
        ax.set_ylabel(f"out-of-fold {metric_name(model)} per fold seed")
        ax.set_title(f"{model} -- spread over {scores['seed'].nunique()} seeds",
                     fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
    fig.suptitle(f"{target} -- is the ordering stable across fold seeds?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "seed_spread.png", dpi=150)
    plt.close(fig)


def add_arguments(p) -> None:
    p.add_argument("--target", required=True,
                   help="phenotype column, e.g. additional_HADS_anx_category")
    p.add_argument("--cohorts", nargs="+", default=["camcan"],
                   help="movie and rest go in ONE table, as a `cohort` column: "
                        "--cohorts camcan camcan_rest")
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS,
                   help="edges = every pairwise correlation (the hypothesis); "
                        "global = a subject's mean and SD over those edges, "
                        "two numbers with no topology in them (the control)")
    p.add_argument("--models", nargs="+", default=["ridge"],
                   choices=["ridge", "hgb", "lgbm", "logistic"],
                   help="ridge only by default: harvardoxford is 6,105 "
                        "features against ~600 subjects, where a tree ensemble "
                        "is both slow and badly matched")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--fc-transform", default="z", choices=FC_TRANSFORMS,
                   help="z = Fisher (arctanh), which stabilises the variance a "
                        "linear penalty assumes is constant")
    p.add_argument("--fc-max-missing", type=float, default=0.05,
                   help="drop an edge NaN in more than this fraction of "
                        "subjects; mean-impute what is left")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="keep only these task labels, where a subject has "
                        "several runs of a cohort")
    p.add_argument("--restrict-subjects", default=None, metavar="FILE",
                   help="one subject id per line, or a CSV with a `sub` "
                        "column. Pass the SAME file here and to `select` to "
                        "put all three trees on one sample, so a gap between "
                        "their tables cannot be a difference in who was "
                        "scored.")
    p.add_argument("--pheno", nargs="+", default=None, metavar="PATH:SEP",
                   help="phenotype tables as path:separator (default: the "
                        "same ones `select` uses)")
    p.add_argument("--id-col", default="CCID")
    p.add_argument("--ordinal-levels", nargs="+", default=list(ORDINAL_LEVELS),
                   metavar="LEVEL",
                   help="the --target column's labels, LOWEST FIRST")
    p.add_argument("--covariates", nargs="+",
                   default=["Age", "Sex"] + QUALITY_COVARIATES,
                   help=f"{QUALITY_COVARIATES} are read from the static-FC "
                        f"table; the rest from the phenotype")
    p.add_argument("--categorical", nargs="+", default=["Sex"])
    p.add_argument("--atlas", nargs="*", default=None)
    p.add_argument("--output-name", default="fcm_selection", metavar="FOLDER")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--show", type=int, default=16)
    p.add_argument("--output-root")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
