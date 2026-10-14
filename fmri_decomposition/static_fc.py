#!/usr/bin/env python3
"""STAGE 3b -- one connectivity matrix per subject. What fcm_selection ranks.

    fmri-decomp static-fc --atlas harvardoxford yeo7 networks \\
        --cohorts camcan camcan_rest

writes, per (atlas, cohort),

    outputs/static_fc/atlas=<a>/cohort=<c>/subjects.parquet

one row per (task, sub): the upper triangle of the Pearson correlation between
parcels, over every good frame of every run, plus the QC that says how much
data went into it.

WHY THIS STAGE EXISTS
---------------------
It is the thing the brain-state transition model has to beat. Static FC is the
default in this literature -- flatten one correlation matrix per subject, hand
the vector to a regressor -- and if it predicts HADS as well as a transition
matrix does, then the dynamics are not the biomarker. Saying so requires
measuring it, not citing it.

THE ONE PROPERTY THAT MAKES THE COMPARISON FAIR
-----------------------------------------------
These edges are computed from the SAME FRAMES the `window_s = -1` state arm is
fitted on -- `frames.read_cohort`, same band-pass, same per-run z-score, same
`good_frame` gate, same censor table. Not a reimplementation that agrees in
spirit. So a difference between `fc:edges` and `bstm:cells` at the frame
aperture is a difference between the MODELS, which is the question, and not a
difference in preprocessing, which would be an artifact.

That is also why this does not go through stage 3. A 476-second window through
`dfc` would be close, but the stimulus grid is per-window and per-cohort, the
emission policy drops incomplete windows, and nothing would guarantee that the
frames entering the correlation are the frames entering the HMM. Reading the
same function is the guarantee.

CONCATENATION, AND WHY THE Z-SCORE IS LOAD-BEARING
--------------------------------------------------
`--pool subject` concatenates every task a subject has -- her "overall FC over
the concatenation of all the movies they watch". Concatenating raw timeseries
would be wrong: two runs with different parcel means produce a step at the
join, and a step shared across parcels is correlation that no neural process
put there. `frames.read_cohort` centres and scales each parcel WITHIN each run
before anything is joined, so the step is removed before the concatenation
happens rather than modelled out afterwards.

For Cam-CAN movie it changes nothing -- one task, one run -- so `--pool task`
(the default, one row per (task, sub), matching stage 5a's granularity) and
`--pool subject` give the same numbers there. It matters for ds002837 and
cneuromod, where a subject watches several films.

WHAT IS STORED, AND WHAT IS NOT
-------------------------------
Raw r, not Fisher z, exactly as stage 3 stores it: arctanh is invertible and
costs nothing at load time, so the file keeps the measurement and the model
step chooses the transform. `select-fcm` applies Fisher z by default.

An edge is NaN when either parcel is empty under that subject's brain mask, or
constant. NaN is kept, not filled: which subject lost which parcel is a
coverage fact, and the decision about how many missing edges a model tolerates
belongs to the model step, where it can be recorded and varied. `pearson_upper`
makes that per-edge rather than poisoning the row.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dfc import pearson_upper
from .io import edge_storage_mode, hive_key, meta_dir, static_fc_root

POOL_CHOICES = ("task", "subject")

# What travels beside the edges. `n_tr_used` is the n that every edge in the row
# was computed over, so it is the reliability column AND the data-quantity
# covariate fcm_selection controls on -- a subject who lost half their frames
# to motion has noisier edges, and noise attenuates prediction.
QC_COLUMNS = ["n_tasks", "n_tr_total", "n_tr_good", "n_tr_used",
              "frac_good_frames", "n_edges_nan"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def edge_names(features: list[str]) -> list[str]:
    """`<i>__<j>` over the upper triangle, row-major -- stage 3's convention.

    Built from the activation file's own column order rather than from the atlas
    registry, because that order is what the correlation was computed in. Taking
    it from the registry instead would silently relabel every edge the day a
    label table is reordered.
    """
    iu, ju = np.triu_indices(len(features), k=1)
    return [f"{features[i]}__{features[j]}" for i, j in zip(iu, ju)]


def frame_coverage(paths: list[Path]) -> dict[tuple[str, str], tuple[int, int]]:
    """(task, sub) -> (frames acquired, frames that passed the motion gate).

    Read here rather than taken from `read_cohort`, which drops bad frames
    before it returns and so cannot report how many there were. The ratio is
    the covariate that matters most after mean FD: scrubbing removes more data
    from subjects who move more, fewer frames give noisier edges, and noise
    attenuates prediction. Left uncontrolled, "less connectivity structure"
    and "moved more" are the same column.

    Only `good_frame` is read, so this costs one small column group per shard.
    """
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for path in paths:
        keys = dict(s.split("=", 1) for s in path.parts if "=" in s)
        key = (keys["task"], keys["sub"])
        g = pd.read_parquet(path, columns=["good_frame"])["good_frame"]
        prev = out.get(key, (0, 0))
        out[key] = (prev[0] + len(g), prev[1] + int(g.to_numpy(bool).sum()))
    return out


#: The `task=` key for a `--pool subject` table, whose rows span every task a
#: subject has. The COLUMN keeps `(pooled)` -- the parentheses are there so a
#: pooled row can never be mistaken for one that measured a single condition --
#: while the path uses the bare word, because `task=(pooled)` needs quoting in
#: every shell glob anyone writes against the tree.
POOLED_TASK_KEY = "pooled"
POOLED_TASK_LABEL = "(pooled)"


def task_key(pool: str, task: str) -> str:
    """The `task=` path key for a row's task label."""
    return POOLED_TASK_KEY if pool != "task" else task


def fc_for_cohort(root: Path, atlas: str, cohort: str, pool: str = "task",
                  censor: dict | None = None, band=None, zscore: bool = True,
                  min_tr: int = 30, log=log,
                  task: str | None = None) -> tuple[pd.DataFrame, dict]:
    """One cohort -> (table, stats). Edges as columns, QC beside them.

    `band` is passed through to `frames.read_cohort` untouched, including None,
    which means "as extracted". The default is resolved by the caller so this
    function has no opinion the CLI cannot override.
    """
    from . import frames

    paths = frames.shard_paths(root, atlas, cohort, task)
    if not paths:
        raise SystemExit(
            f"no activation shards for cohort={cohort}"
            + (f" task={task}" if task else "")
            + f" at atlas={atlas}.\n"
            f"  `static-fc` reads stage 2, so run `fmri-decomp extract "
            f"config/<cohort>.yaml` first.")
    features = frames.feature_columns(paths[0])
    # `task=None` is only reachable under --pool subject, and frames.cohort_tr
    # REFUSES it when the cohort holds two TRs -- which is right: a correlation
    # that concatenated a 2.47s movie run with a 1.97s rest run is not a
    # measurement of anything.
    ident, X, tr = frames.read_cohort(root, atlas, cohort, features,
                                      censor=censor, band=band, zscore=zscore,
                                      log=log, task=task)

    # The group key IS the pooling policy. `(task, sub)` matches stage 5a's
    # granularity so the two arms line up row for row; `sub` is the pooled
    # variant, and its `task` is a literal label rather than one of the real
    # task names -- a row that pooled Movie and Rest must not be mistakable for
    # a row that measured one of them.
    keys = ["task", "sub"] if pool == "task" else ["sub"]
    names = edge_names(features)
    cover = frame_coverage(paths)
    rows, stats = [], {"skipped_thin": [], "all_nan": []}
    for key, g in ident.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        sub = key[-1]
        task = key[0] if pool == "task" else POOLED_TASK_LABEL
        sel = g.index.to_numpy()
        if len(sel) < min_tr:
            stats["skipped_thin"].append((task, sub, len(sel)))
            continue
        r = pearson_upper(X[sel])
        n_nan = int(np.isnan(r).sum())
        if n_nan == len(r):
            stats["all_nan"].append((task, sub))
            continue
        # Summed over whichever tasks this row pooled, so the ratio describes
        # the data behind THIS correlation and not one arbitrary run of it.
        tasks = sorted(g["task"].unique()) if pool != "task" else [task]
        total = sum(cover.get((t, sub), (0, 0))[0] for t in tasks)
        good = sum(cover.get((t, sub), (0, 0))[1] for t in tasks)
        rows.append({"task": task, "sub": sub, "n_tasks": len(tasks),
                     "n_tr_total": total, "n_tr_good": good,
                     "frac_good_frames": (good / total if total else np.nan),
                     "n_tr_used": int(len(sel)), "n_edges_nan": n_nan,
                     "_r": r})

    if not rows:
        raise SystemExit(
            f"no subject in cohort={cohort} at atlas={atlas} reached "
            f"--min-tr {min_tr} usable frames.\n"
            f"  {len(stats['skipped_thin'])} were below it"
            + (f" (worst {min(n for _, _, n in stats['skipped_thin'])} "
               f"frame(s))" if stats["skipped_thin"] else "")
            + f", {len(stats['all_nan'])} had no computable edge.")

    stacked = np.vstack([row.pop("_r") for row in rows])
    out = pd.DataFrame(rows)
    # `columns` vs `list` on the same threshold stage 3 uses, for the same
    # reason: pyarrow's per-column metadata degrades in the low tens of
    # thousands of columns. Harvard-Oxford's 6,105 stay columns, so an edge
    # keeps its name and a loading stays interpretable.
    mode = edge_storage_mode(len(names))
    if mode == "columns":
        out = pd.concat(
            [out, pd.DataFrame(stacked, columns=names, index=out.index)],
            axis=1)
    else:
        out["edges"] = list(stacked)
    stats.update({"n_subjects": len(out), "n_edges": len(names),
                  "edge_names": names,
                  "edge_storage": mode, "tr": tr, "n_nodes": len(features),
                  "median_tr_used": float(out["n_tr_used"].median()),
                  "min_tr_used": int(out["n_tr_used"].min()),
                  "frac_edges_nan": float(out["n_edges_nan"].sum()
                                          / (len(out) * len(names)))})
    return out, stats


def write_cohort(root: Path, atlas: str, cohort: str, table: pd.DataFrame,
                 prov: dict, task: str) -> Path:
    out_dir = static_fc_root(root, atlas, cohort, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "subjects.parquet"
    for k, v in prov.items():
        table[k] = v
    tmp = path.with_name(f"{path.name}.tmp")
    table.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return path


def read_cohort_table(root: Path, atlas: str, cohort: str,
                      columns: list[str] | None = None,
                      task: str | None = None) -> pd.DataFrame:
    """One stored table, with `edges` unpacked back into named columns.

    The two storage modes are an on-disk detail; a reader asking for edges
    should not have to branch on the atlas's width.

    `columns` is a projection, for a caller that wants the keys and not the
    edges -- listing which subjects a cell holds should not read 6,105 float
    columns. Parquet is columnar, so this physically reads fewer bytes. The
    edges are unpacked only when they were asked for.
    """
    if task is None:
        # Every task of this cohort, concatenated -- which is what the
        # selection stage wants, since it filters rows with --keep-tasks.
        found = sorted(static_fc_root(root, atlas, cohort)
                       .glob("task=*/subjects.parquet"))
        if not found:
            raise SystemExit(
                f"no static FC for atlas={atlas} cohort={cohort}.\n"
                f"  expected {static_fc_root(root, atlas, cohort)}"
                f"/task=*/subjects.parquet\n"
                f"  run: fmri-decomp static-fc --atlas {atlas} "
                f"--cohorts {cohort}")
        if len(found) > 1:
            return pd.concat(
                [read_cohort_table(root, atlas, cohort, columns,
                                   hive_key(f, "task")) for f in found],
                ignore_index=True)
        path = found[0]
    else:
        path = static_fc_root(root, atlas, cohort, task) / "subjects.parquet"
    if not path.exists():
        raise SystemExit(
            f"no static FC for atlas={atlas} cohort={cohort} task={task}.\n"
            f"  expected {path}\n"
            f"  run: fmri-decomp static-fc --atlas {atlas} --cohorts {cohort}")
    if columns is not None:
        import pyarrow.parquet as pq

        have = set(pq.ParquetFile(path).schema_arrow.names)
        # A projection names what it wants; a column this cell does not have
        # is dropped rather than raising, because the width of the table
        # depends on the storage mode and on which QC columns the writing
        # version carried.
        return pd.read_parquet(path, columns=[c for c in columns if c in have])
    d = pd.read_parquet(path)
    if "edges" in d.columns:
        names = json.loads(d["edge_names"].iloc[0])
        wide = pd.DataFrame(np.vstack(d["edges"].to_numpy()), columns=names,
                            index=d.index)
        d = pd.concat([d.drop(columns=["edges"]), wide], axis=1)
    return d


def fc_columns(d: pd.DataFrame) -> list[str]:
    """The edge columns of a table read back by `read_cohort_table`."""
    return [c for c in d.columns if "__" in c]


# ------------------------------------------------------------------- run ---
def run(args) -> int:
    root = Path(args.output_root) if args.output_root else _default_root()
    band = tuple(args.match_bandpass) if args.match_bandpass else None
    written = []
    for atlas in args.atlas:
        for cohort in args.cohorts:
            censor = None
            if args.censor_policy:
                from .decompose import load_censor
                censor = load_censor(root, args.censor_policy, atlas, -1,
                                     cohort)
            # ONE FILE PER TASK under --pool task, which is both what the path
            # now says and what the TR requires: a cohort holding two TRs is
            # read one condition at a time (see frames.cohort_tr). --pool
            # subject is deliberately ONE read of the whole cohort, because
            # that is what pooling means -- and the same guard refuses it when
            # the TRs differ, since concatenating a 2.47s run with a 1.97s one
            # is not a measurement.
            from . import frames
            tasks = (frames.shard_tasks(root, atlas, cohort)
                     if args.pool == "task" else [None])
            if not tasks:
                raise SystemExit(
                    f"no activation shards for cohort={cohort} at "
                    f"atlas={atlas} -- `static-fc` reads stage 2, so run "
                    f"`fmri-decomp extract config/<cohort>.yaml` first.")
            for task in tasks:
                log(f"atlas={atlas} cohort={cohort}"
                    + (f" task={task}" if task else "")
                    + f" pool={args.pool}")
                written.append(_one(root, atlas, cohort, task, args, band,
                                    censor, log=log))
    return _finish(root, args, written)


def _one(root: Path, atlas: str, cohort: str, task, args, band, censor,
         log=log) -> dict:
    """One (atlas, cohort, task) -> one stored table, and its manifest row."""
    table, stats = fc_for_cohort(
        root, atlas, cohort, pool=args.pool, censor=censor, band=band,
        zscore=args.zscore_runs, min_tr=args.min_tr, log=log, task=task)
    prov = {"atlas": atlas, "cohort": cohort,
            "task_key": task_key(args.pool, task or ""),
            "pool": args.pool,
            "estimator": "pearson_over_good_frames",
            "frame_source": "frames.read_cohort",
            "match_bandpass": json.dumps(list(band) if band else None),
            "zscore_runs": bool(args.zscore_runs),
            "censor_policy": args.censor_policy or "",
            "min_tr": int(args.min_tr),
            "tr": stats["tr"], "n_nodes": stats["n_nodes"],
            "n_edges": stats["n_edges"]}
    # The name list is carried ONLY when the edges are packed into one list
    # column, because that is the only case where the names are not already the
    # column names. Writing it in the wide case would repeat a 6,105-name JSON
    # string on every row to say what the schema already says.
    if stats["edge_storage"] == "list":
        prov["edge_names"] = json.dumps(stats["edge_names"])
    path = write_cohort(root, atlas, cohort, table, prov,
                        task_key(args.pool, task or ""))
    log(f"  {stats['n_subjects']} subject(s), {stats['n_edges']} edge(s)"
        f" [{stats['edge_storage']}], frames used "
        f"median {stats['median_tr_used']:.0f} / min {stats['min_tr_used']}, "
        f"{stats['frac_edges_nan'] * 100:.3f}% of edges NaN")
    if stats["skipped_thin"]:
        log(f"  WARNING: {len(stats['skipped_thin'])} subject-task(s) "
            f"below --min-tr {args.min_tr}, not written: "
            f"{stats['skipped_thin'][:4]}")
    if stats["all_nan"]:
        log(f"  WARNING: {len(stats['all_nan'])} subject-task(s) had no "
            f"computable edge at all: {stats['all_nan'][:4]}")
    return {"atlas": atlas, "cohort": cohort,
            "task": task_key(args.pool, task or ""),
            "path": str(path.relative_to(root)),
            **{k: v for k, v in stats.items()
               if k not in ("skipped_thin", "all_nan", "edge_names")},
            "n_skipped_thin": len(stats["skipped_thin"]),
            "n_all_nan": len(stats["all_nan"])}


def _finish(root: Path, args, written: list[dict]) -> int:
    band = tuple(args.match_bandpass) if args.match_bandpass else None
    mf = meta_dir(root) / "static_fc.json"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(
        {"pool": args.pool, "min_tr": args.min_tr,
         "match_bandpass": list(band) if band else None,
         "zscore_runs": bool(args.zscore_runs),
         "censor_policy": args.censor_policy or None,
         "cells": written,
         "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, default=str))
    log(f"-> {mf.relative_to(root)}")
    return 0

def _default_root() -> Path:
    """Delegates to io.default_output_root -- see it for why this is not
    eight copies of a cohort filename any more."""
    from .io import default_output_root

    return default_output_root()


def add_arguments(p) -> None:
    from .frames import DEFAULT_BANDPASS

    p.add_argument("--atlas", nargs="+", required=True)
    p.add_argument("--cohorts", nargs="+", required=True,
                   help="one table per (atlas, cohort). A resting-state cohort "
                        "is its own cohort here, because its TR differs.")
    p.add_argument("--pool", choices=POOL_CHOICES, default="task",
                   help="`task`: one row per (task, sub), matching stage 5a. "
                        "`subject`: concatenate every task a subject has into "
                        "one FC, labelled task=(pooled). Identical for a "
                        "one-task cohort such as camcan Movie.")
    p.add_argument("--min-tr", type=int, default=30,
                   help="a subject-task with fewer usable frames is not "
                        "written. 30 frames over 100+ edges is already a poor "
                        "correlation; below that it is noise with a name.")
    p.add_argument("--censor-policy", default=None,
                   help="a stage 3.5 policy name, applied exactly as "
                        "`decompose` applies it at window_s=-1")
    p.add_argument("--match-bandpass", nargs=2, type=float,
                   metavar=("LOW", "HIGH"), default=list(DEFAULT_BANDPASS),
                   help="must match what the state arm was fitted with, or the "
                        "two arms are not comparable")
    p.add_argument("--no-match-bandpass", dest="match_bandpass",
                   action="store_const", const=None)
    p.add_argument("--no-zscore-runs", dest="zscore_runs", action="store_false")
    p.set_defaults(zscore_runs=True)
    p.add_argument("--output-root")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
