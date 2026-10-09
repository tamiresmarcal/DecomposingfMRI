#!/usr/bin/env python3
"""Move one cohort's output under another cohort's key.

    # look first -- this is the default
    python tools/migrate_cohort.py --from camcan_rest --to camcan

    # then do it
    python tools/migrate_cohort.py --from camcan_rest --to camcan --apply

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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="src", required=True, metavar="COHORT")
    p.add_argument("--to", dest="dst", required=True, metavar="COHORT")
    p.add_argument("--output-root", default=None,
                   help="default: $FMRIDECOMP_OUTPUTS, else output_root from "
                        "the cohort configs in config/")
    p.add_argument("--apply", action="store_true",
                   help="actually move. Without it this only reports.")
    a = p.parse_args(argv)

    if a.src == a.dst:
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
