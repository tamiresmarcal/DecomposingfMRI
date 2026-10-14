#!/usr/bin/env python3
"""Move one cohort's output under another cohort's key, and split it by task.

    # 1. merge two cohorts into one (look first -- that is the default)
    python tools/migrate_cohort.py --from camcan_rest --to camcan
    python tools/migrate_cohort.py --from camcan_rest --to camcan --apply

    # 2. add the task= level to the trees that did not have it
    python tools/migrate_cohort.py --split-tasks
    python tools/migrate_cohort.py --split-tasks --apply

TWO HALVES OF ONE MIGRATION, which is why they live together. The first moves
`cohort=camcan_rest` under `cohort=camcan`, because they are the same people.
The second gives `latents/`, `transitions/`, `static_fc/` and `censor/` the
`task=` key that `activation/` and `dfc/` always had -- without which a single
cohort holding two conditions is not merely untidy but WRONG: a latents file
records one `stride_s`, which is the TR of its rows, and `transitions` reads it
for every dwell time in seconds. Run the move first, then the split.

WHY THE SPLIT IS A REWRITE AND NOT A MOVE
-----------------------------------------
For `activation/` and `dfc/`, `task=` was always a directory, so moving one is
a rename. In these four trees it was a COLUMN, so the split reads each file,
partitions its rows by that column, and writes one file per task. The schema
metadata copies verbatim to every part, because all of it describes the FIT and
not the rows: `model_hash`, `censor_policy`, `stride_s`, and the per-column
`clusterers` block carrying each state set's `fit_hash`, `params` and
`train_cohorts`. Dropping that would leave the HMM2 label columns in place
while stripping their provenance, which is worse than losing them.

WHY A MOVE IS ENOUGH, AND WHERE IT IS NOT
-----------------------------------------
`cohort` is a PARTITION KEY, not a column (activation.py: "cohort / atlas /
task / sub are partition keys, carried by the path"), and `frames.read_cohort`
assigns it from the path at read time. So for the immutable per-subject shards
-- stage 1 `activation/` and stage 2 `dfc/` -- renaming the directory is the
whole migration. Nothing inside the files has to change, and the expensive
part of the pipeline is not re-run.

Two things are NOT moved, because moving them would be wrong rather than slow:

  * the DERIVED per-cohort tables -- manifest_*.json, participants_qc.csv,
    coverage.parquet, isc_alignment.csv -- and everything under `censor/`.
    These are keyed by cohort and hold one row per (sub, task). The source and
    the destination each have a complete copy covering THEIR OWN tasks, so a
    move would overwrite one task's table with the other's. They are cheap to
    rebuild and `finalize` rewrites them from scratch, so this script moves the
    per-task `shards/` manifests it needs and leaves the rest to be
    regenerated. It prints the commands.

  * the shards' own schema metadata, which still records the OLD cohort name
    and the old `config_hash`. Nothing reads either -- `cohort` comes from the
    path and `check_cohort.py` compares only `tr`, which is per run and
    unchanged -- so rewriting 1,953 files to correct a string no consumer opens
    would be churn. It is why `--verify` reports them rather than fixing them.

WHAT IT REFUSES
---------------
A DATA destination that already exists. Two cohorts merging into one is only
safe when their `task=` directories are disjoint -- which is the case this
exists for, Cam-CAN's Movie and Rest -- and a collision means two files claim
the same (cohort, task, sub) leaf. Overwriting one with the other would
silently keep whichever moved last.

A clashing shard MANIFEST name is not that case and is renamed, not refused:
those are named by array-task index, so two runs of one stage both write
`shard-0000-of-0008`, while the files describe different runs and
`merge-manifests` reads every file matching the glob. The suffix goes before
`.json` so the renamed file stays inside that glob.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

#: Stages whose leaves are immutable per-subject shards: safe to move.
MOVABLE = ("activation", "dfc")

#: Derived per-cohort tables. Regenerated, never moved -- see the docstring.
REGENERATE = ("censor",)

#: Inside meta/cohorts/cohort=<c>/, only the per-array-task shard manifests
#: are per-run and therefore movable; the merged tables beside them are not.
META_MOVABLE = ("shards",)


def cohort_dirs(root: Path, stage: str, cohort: str) -> list[Path]:
    return sorted(p for p in (root / stage).rglob(f"cohort={cohort}")
                  if p.is_dir())


def plan(root: Path, src: str, dst: str):
    """(source, destination) pairs, plus whatever would collide."""
    moves, collisions = [], []
    for stage in MOVABLE:
        for d in cohort_dirs(root, stage, src):
            # Move each task= directory, not the cohort= directory, so an
            # existing destination cohort keeps the tasks it already has.
            target_cohort = d.parent / f"cohort={dst}"
            for task in sorted(p for p in d.iterdir() if p.is_dir()):
                to = target_cohort / task.name
                (collisions if to.exists() else moves).append((task, to))
            leftovers = sorted(p for p in d.iterdir() if not p.is_dir())
            for f in leftovers:
                to = target_cohort / f.name
                (collisions if to.exists() else moves).append((f, to))
    meta_src = root / "meta" / "cohorts" / f"cohort={src}"
    for name in META_MOVABLE:
        d = meta_src / name
        if not d.exists():
            continue
        to = root / "meta" / "cohorts" / f"cohort={dst}" / name
        if not to.exists():
            moves.append((d, to))
            continue
        # Shard manifests are named by ARRAY TASK INDEX, not by task label, so
        # two runs of the same stage both produce shard-0000-of-0008. That is a
        # name clash and NOT the dangerous case: the two files describe
        # different runs and `merge-manifests` concatenates the entries of
        # every file matching `manifest_<stage>_shard-*.json`. So the name is
        # disambiguated instead of refused -- with the suffix before `.json`,
        # which keeps the file inside that glob and therefore inside the merge.
        # Refusing here would block the migration on provenance files while the
        # data itself was perfectly disjoint.
        for f in sorted(d.iterdir()):
            t = to / f.name
            if t.exists():
                t = to / f"{f.stem}-from-{src}{f.suffix}"
            (collisions if t.exists() else moves).append((f, t))
    return moves, collisions


# --------------------------------------------------- split by task ---------
#: The leaf file name in each tree that keys on cohort=.
LEAVES = ("data.parquet", "subjects.parquet", "windows.parquet")


def unsplit_leaves(root: Path) -> list[Path]:
    """Every leaf still written at the pre-`task=` depth.

    The rule is uniform across all five shapes -- latents, transitions,
    static_fc, and censor's subject AND window tables -- because they all end
    in `cohort=<c>/<leaf>`: a leaf whose immediate parent is a `cohort=` key
    has not been split, and one whose parent is a `task=` key has. No per-tree
    pattern list to keep in step with io.py.
    """
    out = []
    for tree in ("latents", "transitions", "static_fc", "censor"):
        base = root / tree
        if not base.is_dir():
            continue
        for leaf in LEAVES:
            out += [p for p in base.rglob(leaf)
                    if p.parent.name.startswith("cohort=")]
    return sorted(out)


def split_plan(root: Path) -> tuple[list[tuple[Path, dict]], list[str]]:
    """[(file, {task: destination})], plus warnings worth reading first."""
    import pyarrow.parquet as pq

    plan, warn = [], []
    for f in unsplit_leaves(root):
        try:
            tasks = sorted({str(t) for t in pq.ParquetFile(f)
                            .read(columns=["task"]).column("task").to_pylist()})
        except Exception as exc:                                 # noqa: BLE001
            warn.append(f"  SKIPPED {f.relative_to(root)}: no readable `task` "
                        f"column ({type(exc).__name__})")
            continue
        if not tasks:
            warn.append(f"  SKIPPED {f.relative_to(root)}: empty")
            continue
        dest = {t: f.parent / f"task={t}" / f.name for t in tasks}
        clash = [str(d.relative_to(root)) for d in dest.values() if d.exists()]
        if clash:
            warn.append(f"  SKIPPED {f.relative_to(root)}: already split "
                        f"({clash[0]} exists)")
            continue
        # A latents file carries ONE stride_s, and every part inherits it. That
        # is right exactly when the file held one TR -- which it did if it holds
        # one task, or if the cohort has a single TR across them. It is the
        # reason the task level exists, so a file about to be split into
        # SEVERAL tasks gets said out loud rather than assumed.
        if len(tasks) > 1 and f.name == "data.parquet":
            warn.append(
                f"  NOTE {f.relative_to(root)} holds {len(tasks)} tasks "
                f"{tasks}; each part inherits the one stride_s this file "
                f"records, which is correct only if those tasks share a TR. "
                f"If they do not, re-run `decompose` for this cell instead of "
                f"trusting the split.")
        plan.append((f, dest))
    return plan, warn


def split_file(f: Path, dest: dict) -> None:
    """One file -> one per task, carrying the schema metadata verbatim.

    The metadata is what makes a latents file self-identifying: `model_hash`,
    `censor_policy`, `stride_s`, and the per-column `clusterers` block holding
    each state set's `fit_hash`, `params` and `train_cohorts`. All of it
    describes the FIT, not the rows, so it copies unchanged to every part --
    and dropping it would strip the HMM2 labels of their provenance while
    leaving the columns in place, which is the worst of both.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(f)
    md = table.schema.metadata
    col = table.column("task").to_pylist()
    for task, to in dest.items():
        mask = pa.array([str(v) == task for v in col])
        part = table.filter(mask).replace_schema_metadata(md)
        to.parent.mkdir(parents=True, exist_ok=True)
        tmp = to.with_name(f"{to.name}.tmp")
        pq.write_table(part, tmp, compression="zstd")
        tmp.replace(to)


def do_split(root: Path, apply: bool) -> int:
    plan, warn = split_plan(root)
    for w in warn:
        print(w)
    if not plan:
        print(f"\nnothing to split under {root} -- every leaf is already "
              f"under a task= key")
        return 0
    for f, dest in plan:
        print(f"  {f.relative_to(root)}")
        for t, to in dest.items():
            print(f"    -> {to.relative_to(root)}")
    print(f"\n{len(plan)} file(s) to split into "
          f"{sum(len(d) for _, d in plan)} part(s)"
          + ("" if apply else "  (--apply to do it)"))
    if not apply:
        return 0
    for f, dest in plan:
        split_file(f, dest)
        f.unlink()
    print("split.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="src", metavar="COHORT")
    p.add_argument("--to", dest="dst", metavar="COHORT")
    p.add_argument("--split-tasks", action="store_true",
                   help="the OTHER half of this migration: split each leaf "
                        "still written at cohort=<c>/<leaf> into "
                        "cohort=<c>/task=<t>/<leaf>, by its own `task` column. "
                        "Carries the schema metadata, including the per-column "
                        "clusterers block, so state labels keep their "
                        "provenance. Dry-run unless --apply.")
    p.add_argument("--output-root", default=None,
                   help="default: $FMRIDECOMP_OUTPUTS, else output_root from "
                        "the cohort configs in config/")
    p.add_argument("--apply", action="store_true",
                   help="actually move. Without it this only reports.")
    a = p.parse_args(argv)

    if not a.split_tasks and not (a.src and a.dst):
        raise SystemExit("either --from COHORT --to COHORT, or --split-tasks")
    if a.src and a.src == a.dst:
        raise SystemExit("--from and --to are the same cohort")

    if a.output_root:
        root = Path(a.output_root)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from fmri_decomposition.io import default_output_root
        root = default_output_root()
    if not root.is_dir():
        raise SystemExit(f"output_root does not exist: {root}")
    print(f"output_root {root}")

    if a.split_tasks:
        return do_split(root, a.apply)

    moves, collisions = plan(root, a.src, a.dst)
    if collisions:
        print(f"\nREFUSING: {len(collisions)} destination(s) already exist:")
        for frm, to in collisions[:10]:
            print(f"  {to.relative_to(root)}")
        print("  Two files would claim the same (cohort, task, sub) leaf, and "
              "whichever moved last would silently win. Merging two cohorts "
              "is only safe when their task= directories are disjoint.")
        return 1
    if not moves:
        print(f"\nnothing to move: no cohort={a.src} under {MOVABLE} or "
              f"meta/cohorts/")
    for frm, to in moves:
        print(f"  {frm.relative_to(root)}\n    -> {to.relative_to(root)}")
    print(f"\n{len(moves)} move(s)" + ("" if a.apply else "  (--apply to do it)"))

    if a.apply:
        for frm, to in moves:
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(frm), str(to))
        # Leave no empty cohort= shells behind to be mistaken for a cohort.
        for stage in MOVABLE:
            for d in cohort_dirs(root, stage, a.src):
                if not any(d.iterdir()):
                    d.rmdir()
        print("moved.")

    stale = [root / stage for stage in REGENERATE
             if cohort_dirs(root, stage, a.src) or cohort_dirs(root, stage, a.dst)]
    print(f"\nNOT MOVED -- regenerate these, they are per (sub, task) and each "
          f"cohort has a complete copy of its own:")
    print(f"  meta/cohorts/cohort={a.dst}/  manifest_*.json, participants_qc.csv,")
    print(f"                                coverage.parquet, isc_alignment.csv")
    for s in stale:
        print(f"  {s.relative_to(root)}/")
    print(f"\n  sbatch slurm/finalize.sbatch config/{a.dst}.yaml activation")
    print(f"  sbatch slurm/finalize.sbatch config/{a.dst}.yaml dfc")
    print(f"  sbatch slurm/censor.sbatch   config/{a.dst}.yaml "
          f"config/censor/motion.yaml")
    print(f"\n  Then: fmri-decomp status, and check cohort={a.dst} carries "
          f"every task.")
    print(f"\nShard metadata still records cohort={a.src!r} and the old "
          f"config_hash. Nothing reads either -- cohort comes from the path, "
          f"and check_cohort compares only `tr`, which is per run and "
          f"unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
