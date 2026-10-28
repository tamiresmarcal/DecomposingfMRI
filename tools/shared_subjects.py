#!/usr/bin/env python3
"""The subjects every selection run has, so they all score the same sample.

    # one cohort, two conditions -- the Cam-CAN case
    python tools/shared_subjects.py --cohorts camcan --tasks Movie Rest \\
        -o shared_subjects.txt

    # several cohorts, one condition each
    python tools/shared_subjects.py --cohorts camcan ds002837 \\
        -o shared_subjects.txt

    fmri-decomp select-bstm ... --restrict-subjects shared_subjects.txt
    fmri-decomp select-fcm  ... --restrict-subjects shared_subjects.txt

WHY THIS EXISTS
---------------
`select-bstm` and `select-fcm`, each under `--task movie` and `--task rest`,
are four separate RUNS, each on whoever it has. Movie and rest being ONE cohort
does not change that -- they are still separate runs, split by `--keep-tasks`,
so their `n` can still differ. That is the right structure --
they answer different questions and one of them can be re-run without the
others -- but it means two of their tables can differ in SAMPLE as well as in
score, and a gap read as "rest does worse" could be "rest was measured on the
480 subjects with a usable rest scan while movie had 610".

`n` is in every summary.csv so that is visible. This is the fix: one id list,
passed to every run, which each applies to the phenotype before any fit.

WHAT IT INTERSECTS
------------------
One table per (stage, cohort), because within a cohort every atlas and every
state set holds the same subjects -- they come from the same extraction. So the
first match of each glob is enough, and the intersection is over stages and
cohorts rather than over the whole grid. What it reads is the INPUT to each
selection run -- transitions/ for bstm, static_fc/ for fcm -- not the
summary.csv they write, so the list can be built before any of them has run.
The paths are printed with their counts, so a source that silently contributed
nothing is visible rather than inferred.

Ids are upper-cased and stripped, the same normalisation every join in the
pipeline applies, so a list written by this tool matches regardless of how the
source tables spell them.
"""

from __future__ import annotations

import argparse
import functools
import glob
import sys
from pathlib import Path


def sources(root: Path, cohorts: list[str], window_s: str,
            tasks: list[str] | None = None) -> list[tuple[str, str, str | None]]:
    """(label, glob, task) per stage per cohort per task, in report order.

    The task dimension matters because a cohort can hold more than one
    condition. Cam-CAN's movie and rest are one cohort with two tasks, so
    without splitting on task this would read a table containing BOTH and
    return the subjects with EITHER -- a union dressed up as an intersection,
    which is the opposite of what this tool is for.
    """
    out = []
    for t in (tasks or [None]):
        for c in cohorts:
            out.append((f"transitions/{c}" + (f"/{t}" if t else ""),
                        f"{root}/transitions/atlas=*/window_s={window_s}/"
                        f"states=*/cohort={c}/subjects.parquet", t))
    for t in (tasks or [None]):
        for c in cohorts:
            out.append((f"static_fc/{c}" + (f"/{t}" if t else ""),
                        f"{root}/static_fc/atlas=*/cohort={c}/"
                        f"subjects.parquet", t))
    return out


def subjects_in(pattern: str, task: str | None = None) -> tuple[set[str], str]:
    import pandas as pd

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(
            f"nothing matched {pattern}\n"
            f"  That stage has not been built for that cohort yet. Build it, "
            f"or drop its cohort from --cohorts -- a list that silently skips "
            f"a source is worse than no list, because every run would then "
            f"still be scored on its own sample.")
    cols = ["sub"] if task is None else ["sub", "task"]
    d = pd.read_parquet(files[0], columns=cols)
    if task is not None:
        have = sorted(d["task"].astype(str).unique())
        d = d[d["task"].astype(str) == task]
        if d.empty:
            raise SystemExit(
                f"--tasks names {task!r}, which no row of {files[0]} has; "
                f"it holds {have}. A task that matches nothing would "
                f"intersect to the empty set and look like no shared "
                f"subjects.")
    s = set(d["sub"].astype(str).str.strip().str.upper())
    return s, files[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # No default: a cohort name baked into this file only works for one
    # project, and the cohorts to intersect are exactly the ones the selection
    # runs being compared were given.
    p.add_argument("--cohorts", nargs="+", required=True,
                   help="the cohorts whose subjects must all be present, e.g. "
                        "the --cohort of each run being compared")
    p.add_argument("--tasks", nargs="*", default=None, metavar="TASK",
                   help="intersect per TASK as well as per cohort, using the "
                        "BIDS labels the data spells (Movie, Rest). Needed "
                        "where a cohort holds more than one condition: without "
                        "it a table containing both yields the subjects with "
                        "EITHER, which is a union dressed up as an "
                        "intersection. Pass the same labels you pass each run "
                        "as --keep-tasks.")
    p.add_argument("--window-s", default="-1",
                   help="the transitions aperture to read (default -1)")
    p.add_argument("--output-root", default=None,
                   help="default: $FMRIDECOMP_OUTPUTS, else output_root from "
                        "the cohort configs in config/")
    p.add_argument("-o", "--out", default="shared_subjects.txt")
    a = p.parse_args(argv)

    if a.output_root:
        root = Path(a.output_root)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from fmri_decomposition.io import default_output_root

        root = default_output_root()

    sets = []
    for label, pattern, task in sources(root, a.cohorts, a.window_s, a.tasks):
        s, path = subjects_in(pattern, task)
        print(f"{len(s):>6}  {label:<24} {path}")
        sets.append(s)

    shared = functools.reduce(set.intersection, sets)
    print(f"{len(shared):>6}  SHARED by all {len(sets)} source(s)")
    if not shared:
        raise SystemExit(
            "no subject is in every tree. Check the id spelling in each table "
            "-- they are compared upper-cased and stripped, so a real mismatch "
            "is a different id SCHEME (CC110033 against sub-CC110033), not "
            "whitespace.")
    Path(a.out).write_text("\n".join(sorted(shared)) + "\n")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
