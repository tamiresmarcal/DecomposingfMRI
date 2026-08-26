#!/usr/bin/env python3
"""Generate a participants CSV from what discovery actually finds on disk.

    python tools/make_participants.py config/cneuromod_friends.yaml \
        -o config/cneuromod_friends_participants.csv

Why this exists rather than a checked-in file: for a cohort like ds002837 the
(sub, task) mapping is a published constant, so the CSV can ship with the
repo. For CNeuroMod it is not -- which episodes exist is a property of YOUR
copy of an incrementally released dataset, so the file has to be built where
the data is.

IMPORTANT -- what this does NOT do:

Every row is written excluded=False. This tool cannot know which runs you
mean to drop; it only reports what is on disk. Real exclusions must be edited
in by hand, and RECORDED as excluded=True with a reason -- never by deleting
the row. `validate_cohort` treats a run on disk with no table row as a
problem precisely so that "is this excluded, or did someone forget?" stays
answerable. Deleting rows destroys that distinction.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

COLUMNS = ["participant_id", "sub", "cohort", "task", "excluded", "exclusion_reason"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("-o", "--out", required=True, help="path to write the CSV to")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing CSV (refused by default: it may "
                        "carry exclusions you edited in by hand)")
    args = p.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out}\n"
              f"  It may contain exclusions you edited by hand, which this tool "
              f"cannot reproduce.\n  Pass --force if you are sure.", file=sys.stderr)
        return 1

    from fmri_decomposition.config import load_config
    from fmri_decomposition.cohort import discover_runs

    cfg = load_config(args.config)
    if not Path(cfg.derivatives_root).exists():
        print(f"derivatives_root does not exist: {cfg.derivatives_root}\n"
              f"  Edit the config first -- this tool reads real files.", file=sys.stderr)
        return 1

    refs = discover_runs(cfg)
    if not refs:
        print(f"discovery found 0 runs under {cfg.derivatives_root}\n"
              f"  Check discovery.bold_glob and discovery.include_tasks.", file=sys.stderr)
        return 1

    # One row per (sub, task). A subject with several sessions or runs of the
    # same task gets ONE row: the participants table is keyed (sub, task), and
    # attach_participants joins on that pair.
    pairs = sorted({(str(r.sub), r.task) for r in refs})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for sub, task in pairs:
            w.writerow([f"sub-{sub}", sub, cfg.cohort, task, False, ""])

    subs = sorted({s for s, _ in pairs})
    tasks = sorted({t for _, t in pairs})
    print(f"wrote {out}")
    print(f"  {len(pairs)} row(s): {len(subs)} subject(s) x {len(tasks)} task(s)")
    print(f"  runs on disk: {len(refs)}"
          + (f"  ({len(refs) - len(pairs)} extra from multiple ses/run per pair)"
             if len(refs) != len(pairs) else ""))
    print(f"  subjects: {subs[:8]}{' ...' if len(subs) > 8 else ''}")
    print(f"  tasks:    {tasks[:8]}{' ...' if len(tasks) > 8 else ''}")
    print("\n  All rows are excluded=False. Edit in your real exclusions by hand,")
    print("  setting excluded=True AND an exclusion_reason -- do not delete rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
