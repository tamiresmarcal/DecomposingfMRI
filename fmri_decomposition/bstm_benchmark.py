#!/usr/bin/env python3
"""Stage 5c -- does the brain-state transition model beat the obvious baselines?

    fmri-decomp benchmark --target additional_HADS_anx_category \\
        --conditions movie=camcan rest=camcan_rest

writes, per target,

    outputs/bstm_benchmark/target=<t>/DESIGN.md       what was scored
    outputs/bstm_benchmark/target=<t>/scores.parquet  every (arm, condition, model, seed)
    outputs/bstm_benchmark/target=<t>/summary.csv     ONE TABLE, every arm in it
    outputs/bstm_benchmark/target=<t>/figures/*.png
    outputs/meta/bstm_benchmark/target=<t>.json       manifest

`summary.csv` is the deliverable: the same shape as `select`'s, with `arm` and
`condition` as two extra columns, so a static-FC row and a transition-matrix
row sit side by side and are read off against each other by eye. This stage
computes no differences between rows -- see "WHY NO DELTA COLUMN" below.

HOW THIS DIFFERS FROM `select`
------------------------------
`select` asks WHICH state set is best, and varies atlas x aperture x K inside
one model family. It is a model-selection tool, run on the discovery cohort,
and its scores rank -- they are not effect sizes.

`benchmark` asks whether the winning idea beats the alternatives it is supposed
to improve on. Two axes `select` does not have:

    condition   the SAME subjects, scanned under movie and at rest
    arm         transition cells | occupancy | dynamics | static FC | covariates

so the rows the one table has to carry side by side are the ones that decide
whether the project has a result:

    bstm vs fc      within a condition. If one static correlation matrix
                    predicts as well as a transition matrix, the dynamics are
                    not the biomarker and the extra machinery bought nothing.
    movie vs rest   within an arm. If rest does as well, the naturalistic
                    stimulus is not load-bearing and a 10-minute rest scan --
                    which every clinical site already acquires -- is the
                    cheaper instrument.
    either vs the covariates-only floor. Age and sex predict HADS on their own.

THE THREE THINGS THAT MAKE IT A BENCHMARK AND NOT FOUR SEPARATE RUNS
--------------------------------------------------------------------
1. ONE SUBJECT SET. Every arm in every condition is scored on the SAME
   subjects -- the intersection across arms and conditions, taken before any
   model is fitted. Without it, rest could win by being measured on the 480
   subjects who have a usable rest scan while movie is measured on 610, and the
   difference would be the sample, not the condition. `--no-intersect-subjects`
   turns it off and the DESIGN file then says so.

2. ONE COVARIATE BLOCK PER CONDITION, shared by every arm in it. Age and sex
   come from the phenotype; `frac_good_frames` and `n_tr_used` come from that
   condition's static-FC table, which is the one place in the pipeline that
   measures them per subject. So every arm is adjusted for the same nuisances
   with the same numbers, and the covariates-only floor is literally the same
   fit for all of them.

   This deliberately differs from `select`'s default of `n_transitions`. That
   covariate exists only for a transition matrix, so using it here would adjust
   the BSTM arms for their own data quantity and the FC arms for nothing.

3. ONE SCORING FUNCTION. `bstm_selection.score_once` -- same folds, same seeds,
   same metric, same estimators. Imported, not reimplemented.

WHAT A WIN HERE DOES AND DOES NOT MEAN
--------------------------------------
Same caveat as `select`, and it bites harder because there are now more arms:
the best of N arms is biased upward even when nothing is real. The honest
reading is the ORDER and the spread across fold seeds. An arm whose rank
changes with the seed has not won; it has been sampled.

WHY NO DELTA COLUMN
-------------------
This stage deliberately stops at the table. It does not subtract one arm's
score from another's, because the arms SHARE their folds: the two scores are
dependent, so a difference between them has no standard error that any of the
usual tests supply, and a `delta` column invites being read as one. Reading two
rows and their seed spreads off the same table is the honest version of the
same comparison, and it is what the table is laid out for.

A real test -- a corrected resampled t-test, or a nested comparison -- is a
separate piece of work and belongs in whatever writes it up, with the
dependence handled explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import static_fc as _sfc
from .bstm_selection import (DYNAMICS_PREDICTORS, ORDINAL_LEVELS, _wipe,
                             log, metric_name, read_phenotype, score_once,
                             to_cond)
from .io import meta_dir, read_file

# Which blocks of per-subject features each arm hands the regressor, on top of
# the covariates. `fc` is the new one; the rest are `select`'s, kept under the
# same names so a reader can line the two outputs up.
BSTM_ARMS = ["cells", "occupancy", "dynamics", "all"]
FC_ARM = "fc"
ARM_CHOICES = BSTM_ARMS + [FC_ARM]

# Supplied by the condition's static-FC table rather than by the phenotype, and
# identical for every arm in that condition. See point 2 of the docstring.
QUALITY_COVARIATES = ["frac_good_frames", "n_tr_used"]

FC_TRANSFORMS = ("z", "r")


# --------------------------------------------------------------- discovery ---
def parse_conditions(specs: list[str]) -> dict[str, str]:
    """`LABEL=COHORT` pairs -> {label: cohort}, order preserved.

    A label, not the cohort name, because `camcan` and `camcan_rest` are an
    implementation fact -- the axis being compared is movie against rest, and a
    figure legend should say so. Rest has to be its own cohort because its TR
    differs and a cohort config carries one `tr`.
    """
    out: dict[str, str] = {}
    for spec in specs:
        label, sep, cohort = spec.partition("=")
        if not sep or not label or not cohort:
            raise SystemExit(
                f"--conditions {spec!r}: write each one as LABEL=COHORT, "
                f"e.g. movie=camcan rest=camcan_rest")
        if label in out:
            raise SystemExit(f"--conditions: {label!r} given twice")
        out[label] = cohort
    return out


def discover_bstm(root: Path, cohort: str) -> pd.DataFrame:
    """Every transition table for one cohort. Empty frame if there are none."""
    rows = [{"atlas": p.parts[-5].split("=")[1],
             "window_s": p.parts[-4].split("=")[1],
             "states": p.parts[-3].split("=")[1], "path": p}
            for p in (root / "transitions").rglob(f"*/cohort={cohort}/"
                                                  f"subjects.parquet")]
    if not rows:
        return pd.DataFrame(columns=["atlas", "window_s", "states", "path",
                                     "K", "window_s_num"])
    d = pd.DataFrame(rows)
    d["K"] = d["states"].str.rsplit("_", n=1).str[-1].astype(int)
    d["window_s_num"] = d["window_s"].astype(float)
    return d.sort_values(["atlas", "window_s_num", "K"]).reset_index(drop=True)


def discover_fc(root: Path, cohort: str) -> list[str]:
    """Atlases with a stored static-FC table for one cohort."""
    return sorted(p.parts[-3].split("=")[1] for p in
                  (root / "static_fc").glob(f"atlas=*/cohort={cohort}/"
                                            f"subjects.parquet"))


def one_row_per_subject(d: pd.DataFrame, what: str, cohort: str,
                        tasks: list[str] | None = None) -> pd.DataFrame:
    """Refuse a table with several rows for one subject rather than guess.

    A cohort where a subject watches several films has one row per (task, sub),
    and silently keeping the first would make the benchmark a comparison of
    whichever task sorted first. Averaging transition matrices across different
    stimuli is a modelling decision, not a cleanup.

    `tasks` narrows the table first, which is the deliberate way to choose. A
    filter that matches nothing is refused here rather than reported four steps
    later as "the phenotype join matched nothing".
    """
    if tasks:
        before = sorted(d["task"].unique())
        d = d[d["task"].isin(tasks)]
        if d.empty:
            raise SystemExit(
                f"--tasks {tasks} matches no row of {what} for "
                f"cohort={cohort}; it has {before}")
    dup = d["sub"].duplicated(keep=False)
    if not dup.any():
        return d
    tasks = sorted(d.loc[dup, "task"].unique())
    raise SystemExit(
        f"{what} for cohort={cohort} has {int(dup.sum())} row(s) across "
        f"{len(tasks)} task(s) {tasks[:6]} for the same subject.\n"
        f"  This stage scores one row per subject, so it will not pick one or "
        f"average them.\n"
        f"  * pass --tasks <one task> to choose, or\n"
        f"  * for the FC arm, re-run `static-fc --pool subject`, which "
        f"concatenates a subject's tasks into a single correlation.")


# ------------------------------------------------------------------ blocks ---
def fc_block(root: Path, atlas: str, cohort: str, subs: list[str],
             transform: str, max_missing: float,
             tasks: list[str] | None = None
             ) -> tuple[np.ndarray, list[str], dict]:
    """The flattened connectivity matrix, as the model sees it.

    Fisher z by default. The stored value is raw r, which is bounded and whose
    sampling variance depends on the true value; arctanh makes the variance
    roughly constant, which is what a linear model's penalty assumes. It is
    monotone, so it cannot change a rank-based score through the ordering of a
    single edge -- it changes how edges COMBINE.
    """
    d = _sfc.read_cohort_table(root, atlas, cohort)
    d["sub"] = d["sub"].astype(str).str.strip().str.upper()
    d = one_row_per_subject(d, f"static_fc atlas={atlas}", cohort, tasks)
    d = d.set_index("sub").reindex(subs)
    cols = _sfc.fc_columns(d)
    if not cols:
        raise SystemExit(f"static_fc atlas={atlas} cohort={cohort} has no edge "
                         f"columns -- re-run `fmri-decomp static-fc`")
    X = d[cols].to_numpy(float)
    if transform == "z":
        # r = +-1 is reachable on a short window; arctanh of it is infinite, and
        # one infinity poisons the whole column after mean imputation. Clipped
        # at the precision of float32 storage rather than at 1.
        X = np.arctanh(np.clip(X, -0.999999, 0.999999))

    missing = np.isnan(X).mean(axis=0)
    keep = missing <= max_missing
    dropped = [c for c, k in zip(cols, keep) if not k]
    X, cols = X[:, keep], [c for c, k in zip(cols, keep) if k]
    if not cols:
        raise SystemExit(
            f"every one of {len(dropped)} edge(s) at atlas={atlas} "
            f"cohort={cohort} is NaN in more than "
            f"{max_missing:.0%} of subjects.\n"
            f"  An edge is NaN when a parcel is empty under that subject's "
            f"brain mask. Check per-parcel coverage before raising "
            f"--fc-max-missing.")
    # Mean-imputed, and the count is recorded. The alternative -- dropping any
    # subject with one missing edge -- would shrink the shared subject set for
    # every other arm too, which is a worse trade for 111 parcels.
    n_imputed = int(np.isnan(X).sum())
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    return X, cols, {"n_edges_dropped": len(dropped),
                     "n_values_imputed": n_imputed,
                     "dropped_examples": dropped[:6]}


def bstm_blocks(path: Path, subs: list[str], p_norm: str,
                tasks: list[str] | None = None
                ) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """One transition table -> {arm: matrix}, reindexed onto the shared subjects."""
    t = read_file(path).to_pandas()
    t["sub"] = t["sub"].astype(str).str.strip().str.upper()
    cohort = str(t["cohort"].iloc[0]) if "cohort" in t.columns else "?"
    t = one_row_per_subject(t, f"transitions {path.parent.parent.name}", cohort,
                            tasks)
    t = t.set_index("sub").reindex(subs)

    cells = sorted((c for c in t.columns if "->" in c),
                   key=lambda c: tuple(int(x) for x in c.split("->")))
    occ = sorted((c for c in t.columns if c.startswith("occ_")),
                 key=lambda c: int(c.split("_")[1]))
    dyn = [c for c in DYNAMICS_PREDICTORS if c in t.columns]

    cell_mat = t[cells].to_numpy(float) if cells else np.empty((len(t), 0))
    if cells and p_norm == "cond":
        K = int(t["n_states"].iloc[0])
        cell_mat = to_cond(cell_mat, t["n_transitions"].to_numpy(float), K)
    blocks = {"cells": cell_mat,
              "occupancy": t[occ].to_numpy(float),
              "dynamics": t[dyn].to_numpy(float)}
    blocks["all"] = np.column_stack([blocks[k] for k in
                                     ("cells", "occupancy", "dynamics")])
    names = {"cells": cells, "occupancy": occ, "dynamics": dyn,
             "all": cells + occ + dyn}
    return blocks, names


def covariate_block(pheno: pd.DataFrame, quality: pd.DataFrame,
                    subs: list[str], covariates: list[str],
                    categorical: list[str]) -> np.ndarray:
    """Age, sex and the condition's data-quality columns, as one design matrix."""
    d = pheno.set_index("sub").reindex(subs)
    q = quality.set_index("sub").reindex(subs)
    cols = []
    for c in covariates:
        src = q if c in q.columns and c not in d.columns else d
        if c not in src.columns:
            raise SystemExit(
                f"covariate {c!r} is in neither the phenotype nor the "
                f"condition's static-FC table.\n"
                f"  phenotype: {[x for x in d.columns if x != 'y']}\n"
                f"  static FC: {sorted(set(q.columns) & set(_sfc.QC_COLUMNS))}")
        cols.append(src[c].rename(c))
    X = pd.concat(cols, axis=1)
    X = pd.get_dummies(X, columns=[c for c in categorical if c in X.columns],
                       drop_first=True, dummy_na=False).astype(float)
    return X.fillna(X.mean()).to_numpy()


# --------------------------------------------------------------------- run ---
def assemble(root: Path, conditions: dict[str, str], pheno: pd.DataFrame,
             args) -> tuple[dict, pd.DataFrame, dict]:
    """Every (condition, arm) matrix, on one shared subject set.

    Two passes on purpose. The first collects which subjects each condition and
    each arm can actually serve; the second builds the matrices once the shared
    set is known. Building first and intersecting afterwards would mean reading
    Harvard-Oxford's 6,105 edges for subjects that are about to be discarded.
    """
    found, per_cond = {}, {}
    for label, cohort in conditions.items():
        bstm = discover_bstm(root, cohort)
        if args.atlas is not None:
            bstm = bstm[bstm["atlas"].isin(args.atlas)]
        if args.window_s is not None:
            bstm = bstm[bstm["window_s"].isin([str(w) for w in args.window_s])]
        fc_atlases = [a for a in discover_fc(root, cohort)
                      if args.atlas is None or a in args.atlas]
        if bstm.empty and not fc_atlases:
            raise SystemExit(
                f"condition {label!r} (cohort={cohort}) has neither a "
                f"transition table nor a static-FC table under {root}.\n"
                f"  transitions: fmri-decomp transitions --cohorts {cohort}\n"
                f"  static FC:   fmri-decomp static-fc --cohorts {cohort} "
                f"--atlas <a>")

        # The quality covariates come from ONE place per condition. Any atlas's
        # table carries the same values -- they are counts of frames, not of
        # parcels -- so the first one sorted is as good as any, and saying which
        # one was read keeps that checkable.
        if not fc_atlases:
            raise SystemExit(
                f"condition {label!r} (cohort={cohort}) has no static-FC table, "
                f"so {QUALITY_COVARIATES} cannot be read and the arms would "
                f"not share a covariate block.\n"
                f"  run: fmri-decomp static-fc --cohorts {cohort} --atlas "
                f"<atlas>\n"
                f"  or drop them: --covariates "
                f"{' '.join(c for c in args.covariates if c not in QUALITY_COVARIATES)}")
        q = _sfc.read_cohort_table(root, fc_atlases[0], cohort,
                                   columns=["task", "sub"] + _sfc.QC_COLUMNS)
        q["sub"] = q["sub"].astype(str).str.strip().str.upper()
        q = one_row_per_subject(q, f"static_fc atlas={fc_atlases[0]}", cohort,
                                args.tasks)
        quality = q[["sub"] + [c for c in _sfc.QC_COLUMNS if c in q.columns]]

        # Surveyed under the SAME --tasks filter the matrices will be built
        # with. Counting every row here and filtering later would intersect
        # onto subjects that no arm ends up carrying.
        def _subs(d: pd.DataFrame) -> set[str]:
            if args.tasks:
                d = d[d["task"].isin(args.tasks)]
            return set(d["sub"].astype(str).str.strip().str.upper())

        subs_by_arm = {}
        for t in bstm.itertuples():
            subs_by_arm[("bstm", t.atlas, t.window_s, t.states)] = _subs(
                read_file(t.path, ["sub", "task"]).to_pandas())
        for a in fc_atlases:
            subs_by_arm[("fc", a, "static", "fc_edges")] = _subs(
                _sfc.read_cohort_table(root, a, cohort,
                                       columns=["task", "sub"]))
        per_cond[label] = {"cohort": cohort, "bstm": bstm,
                           "fc_atlases": fc_atlases, "quality": quality,
                           "quality_from": fc_atlases[0],
                           "subs_by_arm": subs_by_arm}
        found[label] = set.intersection(*subs_by_arm.values())

    shared = set(pheno["sub"])
    if args.intersect_subjects:
        shared &= set.intersection(*found.values())
    else:
        shared &= set.union(*found.values())
    subs = sorted(shared)
    if len(subs) < args.min_subjects:
        sizes = {f"{lab}:{'/'.join(str(x) for x in k[:2])}": len(v)
                 for lab, c in per_cond.items()
                 for k, v in c["subs_by_arm"].items()}
        raise SystemExit(
            f"only {len(subs)} subject(s) are shared by every arm and have a "
            f"usable {args.target}, below --min-subjects {args.min_subjects}.\n"
            f"  phenotype: {len(pheno)}\n"
            f"  per arm:   {sizes}\n"
            f"  Drop the thin arm with --atlas / --window-s, or run with "
            f"--no-intersect-subjects and read the per-arm n in summary.csv.")

    data, meta = {}, []
    for label, c in per_cond.items():
        subs_here = ([s for s in subs if s in set.intersection(
            *c["subs_by_arm"].values())] if not args.intersect_subjects
            else subs)
        C = covariate_block(pheno, c["quality"], subs_here, args.covariates,
                            args.categorical)
        y = pheno.set_index("sub").reindex(subs_here)["y"].to_numpy(float)
        data[(label, "(covariates only)", "-", "-", "-")] = (C, y)
        meta.append({"condition": label, "arm": "(covariates only)",
                     "atlas": "-", "window_s": "-", "states": "-", "K": 0,
                     "kind": "base", "n": len(subs_here),
                     "n_features": 0, "n_covariates": C.shape[1]})

        for t in c["bstm"].itertuples():
            blocks, names = bstm_blocks(t.path, subs_here, args.p_norm,
                                        args.tasks)
            for arm in args.arms:
                if arm == FC_ARM or blocks[arm].shape[1] == 0:
                    continue
                key = (label, f"bstm:{arm}", t.atlas, t.window_s, t.states)
                data[key] = (np.column_stack([C, blocks[arm]]), y)
                meta.append({"condition": label, "arm": f"bstm:{arm}",
                             "atlas": t.atlas, "window_s": t.window_s,
                             "states": t.states, "K": t.K, "kind": "set",
                             "n": len(subs_here),
                             "n_features": blocks[arm].shape[1],
                             "n_covariates": C.shape[1]})
        if FC_ARM in args.arms:
            for a in c["fc_atlases"]:
                Xfc, cols, info = fc_block(root, a, c["cohort"], subs_here,
                                           args.fc_transform,
                                           args.fc_max_missing, args.tasks)
                key = (label, "fc:edges", a, "static", "fc_edges")
                data[key] = (np.column_stack([C, Xfc]), y)
                meta.append({"condition": label, "arm": "fc:edges",
                             "atlas": a, "window_s": "static",
                             "states": "fc_edges", "K": 0, "kind": "set",
                             "n": len(subs_here), "n_features": len(cols),
                             "n_covariates": C.shape[1], **info})
    return data, pd.DataFrame(meta), per_cond


def design_note(args, meta: pd.DataFrame, per_cond: dict, n_jobs: int) -> str:
    lines = [
        f"# benchmark -- {args.target}", "",
        f"Written {time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        f"{'' if args.intersect_subjects else '  --  SUBJECT SETS NOT INTERSECTED'}",
        "", "## What was compared", "",
        "| axis | values |", "|---|---|",
        "| condition | " + ", ".join(
            f"{label} (cohort={c['cohort']})" for label, c in per_cond.items())
        + " |",
        f"| arm | {', '.join(sorted(meta['arm'].unique()))} |",
        f"| atlas | {', '.join(sorted(a for a in meta['atlas'].unique() if a != '-'))} |",
        f"| aperture | {', '.join(sorted(str(w) for w in meta['window_s'].unique() if w != '-'))} |",
        f"| model | {', '.join(args.models)} |",
        f"| fold seed | {', '.join(str(s) for s in args.seeds)} |",
        "", "## What was held fixed", "",
        "| held fixed | value |", "|---|---|",
        f"| subjects | {int(meta['n'].min())}"
        + (f" (one shared set)" if args.intersect_subjects
           else f"-{int(meta['n'].max())} (per condition)") + " |",
        f"| covariates | {', '.join(args.covariates)} |",
        f"| tasks | {', '.join(args.tasks) if args.tasks else 'every task in each condition'} |",
        f"| transition cells expressed as | p_{args.p_norm} |",
        f"| FC edges expressed as | {'Fisher z' if args.fc_transform == 'z' else 'raw r'} |",
        f"| FC edges dropped above | {args.fc_max_missing:.0%} missing |",
        "| cross-validation | 5-fold, repeated over the fold seeds |",
        "| boosting hyperparameters | fixed, not tuned (ridge's alpha is) |",
        "", "## Feature widths", "",
        "| condition | arm | atlas | aperture | predictors | + covariates | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in meta.sort_values(["condition", "arm", "atlas"]).itertuples():
        lines.append(f"| {r.condition} | {r.arm} | {r.atlas} | {r.window_s} | "
                     f"{r.n_features} | {r.n_covariates} | {r.n} |")
    lines += [
        "", "## Reading this", "",
        "* The scores are out-of-fold and the arms share their folds, so a "
        "difference between two rows is dependent and is NOT a test. This "
        "stage computes none on purpose; read the rows and their spreads.",
        "* The best of many arms is biased upward. Quote the ORDER and the "
        "spread across fold seeds, never the number.",
        f"* Quality covariates were read from the static-FC table of atlas "
        f"{', '.join(sorted({c['quality_from'] for c in per_cond.values()}))} "
        f"-- they count frames, not parcels, so every atlas carries the same "
        f"values.",
        f"* {n_jobs} fit job(s) ran.",
    ]
    return "\n".join(lines) + "\n"


def run(args) -> int:
    from joblib import Parallel, delayed

    from .bstm_selection import DEFAULT_PHENO

    root = Path(args.output_root) if args.output_root else _sfc._default_root()
    conditions = parse_conditions(args.conditions)
    # Resolved here, not in argparse: `benchmark` is reached through the shared
    # CLI as well as through this module's own main(), and a default built at
    # import time would be baked into --help output for a path that only exists
    # on the cluster.
    if not args.pheno:
        args.pheno = list(DEFAULT_PHENO)

    log(f"phenotype for target={args.target!r}")
    pheno_cov = [c for c in args.covariates if c not in QUALITY_COVARIATES]
    pheno = read_phenotype(args.pheno, args.id_col, args.target, pheno_cov,
                           args.categorical, levels=args.ordinal_levels)

    log(f"conditions: " + "  ".join(f"{k}={v}" for k, v in conditions.items()))
    data, meta, per_cond = assemble(root, conditions, pheno, args)
    log(f"  {len(data)} (condition, arm) cell(s) on "
        f"{int(meta['n'].min())}-{int(meta['n'].max())} subject(s)")
    for r in meta.sort_values(["condition", "arm"]).itertuples():
        log(f"    {r.condition:<8} {r.arm:<18} {r.atlas:<14} "
            f"{str(r.window_s):>8}  {r.n_features:>6} predictor(s) "
            f"+ {r.n_covariates} covariate(s)  n={r.n}")

    jobs = [(key, m, s) for key in data for m in args.models for s in args.seeds]
    log(f"  {len(jobs)} fit job(s)")
    t0 = time.time()
    scores = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(score_once)(data[key][0], data[key][1], m, s)
        for key, m, s in jobs)
    log(f"  fitted in {time.time() - t0:.0f}s")

    rows = [{"condition": key[0], "arm": key[1], "atlas": key[2],
             "window_s": key[3], "states": key[4], "model": m,
             "metric": metric_name(m), "seed": s, "score": sc}
            for (key, m, s), sc in zip(jobs, scores)]
    scores_df = pd.DataFrame(rows).merge(
        meta, on=["condition", "arm", "atlas", "window_s", "states"],
        how="left")

    out = root / "bstm_benchmark" / f"target={args.target}"
    _wipe(out, parent="bstm_benchmark")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(out / "scores.parquet", index=False)

    summary = (scores_df.groupby(["model", "condition", "arm", "atlas",
                                  "window_s", "states", "K", "n",
                                  "n_features"], dropna=False)["score"]
               .agg(["mean", "std", "min", "max", "count"]).reset_index()
               .sort_values(["model", "mean"], ascending=[True, False]))
    summary.to_csv(out / "summary.csv", index=False)

    print()
    for model, g in summary.groupby("model"):
        print(f"=== {model}  ({metric_name(model)}) ===")
        print(g.drop(columns=["model", "states"]).head(args.show)
               .to_string(index=False))
        print()

    note = design_note(args, meta, per_cond, len(jobs))
    (out / "DESIGN.md").write_text(note)
    print(note)
    _figures(scores_df, summary, out / "figures", args.target,
             list(conditions))

    mf = meta_dir(root) / "bstm_benchmark" / f"target={args.target}.json"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(
        {"target": args.target,
         "conditions": {k: v["cohort"] for k, v in per_cond.items()},
         "arms": args.arms, "models": args.models, "seeds": args.seeds,
         "covariates": args.covariates, "p_norm": args.p_norm,
         "tasks": args.tasks,
         "fc_transform": args.fc_transform,
         "fc_max_missing": args.fc_max_missing,
         "intersect_subjects": bool(args.intersect_subjects),
         "n_subjects": int(meta["n"].min()),
         "n_cells": len(data), "n_jobs_run": len(jobs),
         "quality_covariates_from": {k: v["quality_from"]
                                     for k, v in per_cond.items()},
         "output": str(out.relative_to(root)),
         "note": "out-of-fold scores on shared folds: the ORDER is the result, "
                 "the numbers are not effect sizes and the deltas are not tests",
         "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, default=str))
    log(f"-> {out.relative_to(root)}")
    log(f"-> {mf.relative_to(root)}")
    return 0


def _figures(scores: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path,
             target: str, conditions: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(scores["model"].unique())
    arms = ["(covariates only)"] + sorted(
        a for a in scores["arm"].unique() if a != "(covariates only)")

    # 1. the headline: every arm's best cell, grouped by condition
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.6),
                             squeeze=False)
    width = 0.8 / max(len(conditions), 1)
    for ax, model in zip(axes[0], models):
        g = summary[summary["model"] == model]
        for ci, cond in enumerate(conditions):
            h = g[g["condition"] == cond]
            best = [h[h["arm"] == a]["mean"].max() if (h["arm"] == a).any()
                    else np.nan for a in arms]
            err = [h.loc[h[h["arm"] == a]["mean"].idxmax(), "std"]
                   if (h["arm"] == a).any() else np.nan for a in arms]
            ax.bar(np.arange(len(arms)) + ci * width, best, width,
                   yerr=err, capsize=3, label=cond)
        base = g[g["arm"] == "(covariates only)"]["mean"].max()
        if np.isfinite(base):
            ax.axhline(base, ls="--", lw=1.1, c="k",
                       label="covariates floor (best condition)")
        ax.axhline(0 if model != "logistic" else 0.5, lw=0.6, c="grey")
        ax.set_xticks(np.arange(len(arms)) + width * (len(conditions) - 1) / 2)
        ax.set_xticklabels(arms, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(f"out-of-fold {metric_name(model)} (best cell per arm)")
        ax.set_title(model, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"{target} -- transitions vs static FC, movie vs rest",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "arms_by_condition.png", dpi=150)
    plt.close(fig)

    # 2. the spread across fold seeds, which is what says whether an order is
    #    real. A bar chart of means hides exactly the thing that matters.
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.6),
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        s = scores[scores["model"] == model]
        labels, series = [], []
        for cond in conditions:
            for a in arms:
                h = s[(s["condition"] == cond) & (s["arm"] == a)]
                if h.empty:
                    continue
                bestcell = (h.groupby(["atlas", "states"])["score"].mean()
                            .idxmax())
                h = h[(h["atlas"] == bestcell[0]) & (h["states"] == bestcell[1])]
                labels.append(f"{cond}\n{a}")
                series.append(h["score"].to_numpy())
        if series:
            # `labels=` was renamed `tick_labels=` in matplotlib 3.9 and
            # deprecated-then-removed. The container's version is not pinned
            # here, so ask the signature rather than the version string.
            import inspect
            kw = ("tick_labels" if "tick_labels" in
                  inspect.signature(ax.boxplot).parameters else "labels")
            ax.boxplot(series, **{kw: labels})
        ax.axhline(0 if model != "logistic" else 0.5, lw=0.6, c="grey")
        ax.set_ylabel(f"out-of-fold {metric_name(model)} per fold seed")
        ax.set_title(f"{model} -- spread over {scores['seed'].nunique()} seeds",
                     fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=60)
    fig.suptitle(f"{target} -- is the ordering stable across fold seeds?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "seed_spread.png", dpi=150)
    plt.close(fig)


def add_arguments(p) -> None:
    p.add_argument("--target", required=True,
                   help="phenotype column, e.g. additional_HADS_anx_category")
    p.add_argument("--conditions", nargs="+", required=True,
                   metavar="LABEL=COHORT",
                   help="e.g. movie=camcan rest=camcan_rest. Rest is its own "
                        "cohort because its TR differs.")
    p.add_argument("--arms", nargs="+", default=["cells", "occupancy", FC_ARM],
                   choices=ARM_CHOICES,
                   help="cells = the K*K transition probabilities (the "
                        "hypothesis); occupancy = share of time per state; "
                        "dynamics = switch rate, dwell, entropy; all = every "
                        "BSTM block; fc = the flattened static connectivity "
                        "matrix (the literature's default)")
    p.add_argument("--models", nargs="+", default=["ridge"],
                   choices=["ridge", "hgb", "lgbm", "logistic"],
                   help="ridge only by default: the FC arm is 6,105 features "
                        "against ~600 subjects, where a tree ensemble is both "
                        "slow and badly matched")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--p-norm", default="cond", choices=["joint", "cond"],
                   help="how a transition cell is expressed. One value, not a "
                        "list: this stage compares arms, and expanding the "
                        "BSTM arm over both would give it two entries against "
                        "the FC arm's one")
    p.add_argument("--fc-transform", default="z", choices=FC_TRANSFORMS,
                   help="z = Fisher (arctanh), which stabilises the variance "
                        "a linear penalty assumes is constant")
    p.add_argument("--fc-max-missing", type=float, default=0.05,
                   help="drop an edge NaN in more than this fraction of "
                        "subjects; mean-impute what is left")
    p.add_argument("--intersect-subjects", action="store_true", default=True,
                   help="score every arm on the same subjects (default)")
    p.add_argument("--no-intersect-subjects", dest="intersect_subjects",
                   action="store_false",
                   help="let each condition use every subject it has -- then a "
                        "difference between conditions may be the sample")
    p.add_argument("--min-subjects", type=int, default=50,
                   help="refuse rather than report a score from a shared set "
                        "this thin")
    p.add_argument("--pheno", nargs="+",
                   default=None, metavar="PATH:SEP",
                   help="phenotype tables as path:separator (default: the same "
                        "ones `select` uses)")
    p.add_argument("--id-col", default="CCID")
    p.add_argument("--ordinal-levels", nargs="+", default=list(ORDINAL_LEVELS),
                   metavar="LEVEL",
                   help="the --target column's labels, LOWEST FIRST")
    p.add_argument("--covariates", nargs="+",
                   default=["Age", "Sex"] + QUALITY_COVARIATES,
                   help=f"{QUALITY_COVARIATES} are read from the condition's "
                        f"static-FC table; the rest from the phenotype")
    p.add_argument("--categorical", nargs="+", default=["Sex"])
    p.add_argument("--tasks", nargs="*", default=None,
                   help="keep only these task labels. Needed where a subject "
                        "has several runs of a condition: this stage scores "
                        "one row per subject and refuses to pick for you. For "
                        "the FC arm, `static-fc --pool subject` is the other "
                        "answer -- it concatenates them into one correlation.")
    p.add_argument("--atlas", nargs="*", default=None)
    p.add_argument("--window-s", nargs="*", default=None)
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
