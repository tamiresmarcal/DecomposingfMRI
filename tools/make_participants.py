#!/usr/bin/env python3
"""Generate or update the participants CSV from what discovery finds on disk.

    # build the table
    python tools/make_participants.py config/cneuromod_friends.yaml \
        -o config/cneuromod_friends_participants.csv

    # add rows for newly released runs, keeping every existing row and every
    # exclusion someone edited in by hand
    python tools/make_participants.py config/ds002837.yaml \
        -o config/ds002837_participants.csv --update

    # build from the subject list the cohort was MEANT to have, so a subject
    # whose preprocessing failed gets a row instead of vanishing
    python tools/make_participants.py config/camcan_ccfrail_movie.yaml \
        -o config/camcan_ccfrail_movie_participants.csv \
        --from-list /path/to/subjects.txt

Why this exists rather than a checked-in file: for a cohort like ds002837 the
(sub, task) mapping is a published constant, so the CSV can ship with the
repo. For CNeuroMod it is not -- which episodes exist is a property of YOUR
copy of an incrementally released dataset, so the file has to be built where
the data is.

THIS FILE IS HUMAN-OWNED, AND CARRIES CURATION ONLY
---------------------------------------------------
`participants.csv` records who is in the cohort and who was deliberately
removed from it, with the reason: "corrupted run", "consent withdrawn",
"wrong stimulus presented". It is load-bearing at stage 2 -- `attach_participants`
drops excluded rows before extraction -- and nothing writes it automatically.

QC METRICS ARE NOT HERE. They live in `participants_qc.csv`, written by
`fmri-decomp diagnose` (which `03_finalize.sbatch` already runs), and the
thresholds that turn them into exclusions live with the models. The split is
by ownership: this file is edited by a person and read by the pipeline;
that one is written by the pipeline and never hand-edited.

    participants.csv       human-owned    curation      -> honoured at stage 2
    participants_qc.csv    machine-owned  measurement   -> thresholded at the models

IMPORTANT -- what this does NOT do:

Every generated row is written excluded=False. This tool cannot know which
runs you mean to drop; it only reports what is on disk. Real exclusions must
be edited in by hand, and RECORDED as excluded=True with a reason -- never by
deleting the row. `validate_cohort` treats a run on disk with no table row as
a problem precisely so that "is this excluded, or did someone forget?" stays
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


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if value != value else f"{value:.6g}"     # NaN -> empty cell
    return str(value)


def read_subject_list(path: Path) -> list[str]:
    """Subject ids from a plain text list, one per line.

    The list of subjects a cohort was MEANT to contain is not derivable from
    what is on disk -- that is the whole reason it exists as a separate file.
    Deriving the table from discovery instead makes a failed subject invisible:
    it has no bold file, so nothing reports it missing, and 54 silently becomes
    the cohort size.
    """
    out, seen = [], set()
    for line in path.read_text().splitlines():
        sub = line.strip()
        if not sub or sub.startswith("#"):
            continue
        sub = sub.removeprefix("sub-")
        if sub not in seen:
            seen.add(sub)
            out.append(sub)
    return out


def read_existing(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or COLUMNS)


def write_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write then rename, so a killed run cannot leave a half-written table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: _fmt(row.get(c)) for c in columns})
    tmp.replace(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("-o", "--out", required=True, help="path to write the CSV to")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing CSV from scratch (refused by default: "
                        "it may carry exclusions you edited in by hand)")
    p.add_argument("--update", action="store_true",
                   help="read the existing CSV, keep every row, every exclusion and "
                        "every extra column, and add newly discovered runs")
    p.add_argument("--from-list", metavar="PATH",
                   help="a text file of the subject ids the cohort was MEANT to "
                        "contain, one per line ('sub-' optional, blank and #-lines "
                        "skipped). Rows are written for all of them, including any "
                        "with nothing on disk -- see the note below")
    p.add_argument("--task", help="task to pair with --from-list ids; needed only "
                                  "when discovery finds more than one")
    args = p.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not (args.force or args.update):
        print(f"refusing to overwrite {out}\n"
              f"  It may contain exclusions you edited by hand, which this tool "
              f"cannot reproduce.\n"
              f"  --update keeps them and adds new rows; --force rebuilds from scratch.",
              file=sys.stderr)
        return 1
    if args.update and not out.exists():
        print(f"--update needs an existing {out}; drop the flag to create it.",
              file=sys.stderr)
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

    absent: list[str] = []
    if args.from_list:
        listed = read_subject_list(Path(args.from_list))
        if not listed:
            print(f"{args.from_list} lists no subject ids", file=sys.stderr)
            return 1
        tasks = [args.task] if args.task else sorted({t for _, t in pairs})
        if len(tasks) != 1:
            print(f"--from-list needs one task, but discovery found {tasks}.\n"
                  f"  Pass --task to say which of them these ids belong to.",
                  file=sys.stderr)
            return 1
        task = tasks[0]
        on_disk = {sub for sub, _ in pairs}
        absent = [s for s in listed if s not in on_disk]
        pairs = sorted(set(pairs) | {(s, task) for s in listed})

    columns = list(COLUMNS)
    if args.update:
        rows, columns = read_existing(out)
        for row in rows:
            row["excluded"] = str(row.get("excluded", "")).strip().lower() in ("true", "1", "yes")
        known = {(str(r.get("sub")), str(r.get("task"))) for r in rows}
        added = [pair for pair in pairs if pair not in known]
        for sub, task in added:
            rows.append({"participant_id": f"sub-{sub}", "sub": sub, "cohort": cfg.cohort,
                         "task": task, "excluded": False, "exclusion_reason": ""})
        stale = sorted(known - set(pairs))
        print(f"updating {out}: {len(rows) - len(added)} existing row(s), "
              f"{len(added)} added")
        if stale:
            # Kept, not deleted -- for the same reason exclusions are recorded.
            print(f"  {len(stale)} row(s) in the CSV with no run on disk, kept as-is: "
                  f"{stale[:5]}")
    else:
        rows = [{"participant_id": f"sub-{sub}", "sub": sub, "cohort": cfg.cohort,
                 "task": task, "excluded": False, "exclusion_reason": ""}
                for sub, task in pairs]

    write_table(out, rows, columns)

    if absent:
        # These are the whole point of --from-list. A subject whose
        # preprocessing failed has no bold file, so discovery cannot see it and
        # it would otherwise be ABSENT rather than EXCLUDED -- indistinguishable
        # from one nobody meant to include. The row makes it visible;
        # `validate_cohort` will now report it as an analysable row with no run
        # on disk until a person writes down why.
        print(f"\n  {len(absent)} listed subject(s) with NO run on disk:")
        for sub in absent[:10]:
            print(f"    sub-{sub}")
        if len(absent) > 10:
            print(f"    ... and {len(absent) - 10} more")
        print("  Each has a row, excluded=False. Edit in the real status by hand --")
        print("  excluded=True WITH a reason -- or `validate` will keep reporting them.")

    subs = sorted({str(r.get("sub")) for r in rows})
    tasks = sorted({str(r.get("task")) for r in rows})
    n_excluded = sum(1 for r in rows if r.get("excluded") in (True, "True"))
    print(f"\nwrote {out}")
    print(f"  {len(rows)} row(s): {len(subs)} subject(s) x {len(tasks)} task(s), "
          f"{n_excluded} excluded")
    # Only the surplus direction is a ses/run remark. With --from-list the
    # table can legitimately hold MORE pairs than there are runs, and those are
    # already reported above by name -- "-1 extra" was arithmetic, not English.
    extra = len(refs) - len(pairs)
    print(f"  runs on disk: {len(refs)}"
          + (f"  ({extra} extra from multiple ses/run per pair)" if extra > 0 else ""))
    print(f"  subjects: {subs[:8]}{' ...' if len(subs) > 8 else ''}")
    print(f"  tasks:    {tasks[:8]}{' ...' if len(tasks) > 8 else ''}")
    print("\n  Exclusions are unchanged -- nothing here writes one. Edit real ones in")
    print("  by hand as excluded=True WITH a reason; do not delete rows, or the")
    print("  validator can no longer tell a curation decision from an oversight.")
    print("  QC metrics are not here: see participants_qc.csv, written by")
    print("  `fmri-decomp diagnose`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
