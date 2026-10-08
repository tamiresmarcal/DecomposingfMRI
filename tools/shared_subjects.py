#!/usr/bin/env python3
"""The subjects every selection tree has, so all three score the same sample.

    python tools/shared_subjects.py --cohorts camcan camcan_rest \\
        -o shared_subjects.txt

    fmri-decomp select     ... --restrict-subjects shared_subjects.txt
    fmri-decomp fcm-select ... --restrict-subjects shared_subjects.txt

WHY THIS EXISTS
---------------
`bstm_selection`, `resting_bstm_selection` and `fcm_selection` are separate
RUNS, each on whoever it has. That is the right structure -- they answer
different questions and one of them can be re-run without the others -- but it
means two of their tables can differ in SAMPLE as well as in score, and a gap
read as "rest does worse" could be "rest was measured on the 480 subjects with
a usable rest scan while movie had 610".

`n` is in every summary.csv so that is visible. This is the fix: one id list,
passed to every tree, which each applies to the phenotype before any fit.

WHAT IT INTERSECTS
------------------
One table per (tree, cohort), because within a cohort every atlas and every
state set holds the same subjects -- they come from the same extraction. So the
first match of each glob is enough, and the intersection is over trees and
cohorts rather than over the whole grid. The paths are printed with their
counts, so a tree that silently contributed nothing is visible rather than
inferred.

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


def sources(root: Path, cohorts: list[str], window_s: str) -> list[tuple[str, str]]:
    """(label, glob) per tree per cohort, in the order they are reported."""
    out = []
    for c in cohorts:
        out.append((f"transitions/{c}",
                    f"{root}/transitions/atlas=*/window_s={window_s}/"
                    f"states=*/cohort={c}/subjects.parquet"))
    for c in cohorts:
        out.append((f"static_fc/{c}",
                    f"{root}/static_fc/atlas=*/cohort={c}/subjects.parquet"))
    return out


def subjects_in(pattern: str) -> tuple[set[str], str]:
    import pandas as pd

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(
            f"nothing matched {pattern}\n"
            f"  That tree has not been built yet. Build it, or drop its cohort "
            f"from --cohorts -- a list that silently skips a tree is worse "
            f"than no list, because every tree would then still be scored on "
            f"its own sample.")
    s = set(pd.read_parquet(files[0], columns=["sub"])["sub"]
            .astype(str).str.strip().str.upper())
    return s, files[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohorts", nargs="+", default=["camcan", "camcan_rest"])
    p.add_argument("--window-s", default="-1",
                   help="the transitions aperture to read (default -1)")
    p.add_argument("--output-root", default=None,
                   help="default: output_root from config/camcan_movie.yaml, "
                        "or $FMRIDECOMP_OUTPUTS")
    p.add_argument("-o", "--out", default="shared_subjects.txt")
    a = p.parse_args(argv)

    if a.output_root:
        root = Path(a.output_root)
    else:
        import os

        if os.environ.get("FMRIDECOMP_OUTPUTS"):
            root = Path(os.environ["FMRIDECOMP_OUTPUTS"])
        else:
            import yaml

            repo = Path(__file__).resolve().parent.parent
            root = Path(yaml.safe_load(
                (repo / "config" / "camcan_movie.yaml").read_text())
                ["output_root"])

    sets = []
    for label, pattern in sources(root, a.cohorts, a.window_s):
        s, path = subjects_in(pattern)
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
