#!/usr/bin/env python3
"""Generate or update a participants CSV from what discovery finds on disk.

    # build the table
    python tools/make_participants.py config/cneuromod_friends.yaml \
        -o config/cneuromod_friends_participants.csv

    # add subject-level motion (mean FD) to a table that already exists,
    # without touching the exclusions someone edited in by hand
    python tools/make_participants.py config/ds002837.yaml \
        -o config/ds002837_participants.csv --update --fd

    # ... and let the motion decide the exclusions, recorded with a reason
    python tools/make_participants.py config/ds002837.yaml \
        -o config/ds002837_participants.csv --update --fd --exclude-mean-fd 0.5

Why this exists rather than a checked-in file: for a cohort like ds002837 the
(sub, task) mapping is a published constant, so the CSV can ship with the
repo. For CNeuroMod it is not -- which episodes exist is a property of YOUR
copy of an incrementally released dataset, so the file has to be built where
the data is. `--fd` has the same property for both: the numbers come from
files that are not in this repo.

WHAT `--fd` DOES AND DOES NOT CLAIM
-----------------------------------
It reads `confounds.motion_glob` and writes SUBJECT-LEVEL motion columns:
mean_fd, median_fd, max_fd, frac_fd_gt_0p2, frac_fd_gt_0p5.

On ds002837 the motion regressors are on the acquisition clock and the
derivative images are on the stimulus clock, 15..28 frames apart with no
recoverable mapping -- which is why `confounds.censor_glob` is disabled and
`good_frame` is all true. That misalignment kills frame-level censoring and
does not touch a subject-level mean: shifting which 5,470 of ~5,490 frames you
average over moves mean FD by well under a percent. So these columns are for
excluding subjects and for use as a stage-5 covariate, and for nothing that is
indexed by frame.

EXCLUSIONS
----------
Without `--exclude-mean-fd`, every generated row is excluded=False and this
tool never changes an exclusion. Real exclusions are edited in by hand and
RECORDED as excluded=True with a reason -- never by deleting the row.
`validate_cohort` treats a run on disk with no table row as a problem
precisely so that "is this excluded, or did someone forget?" stays answerable.

`--exclude-mean-fd X` is the one exception, and it stays inside that rule: it
writes excluded=True with the reason `auto:mean_fd>X`. Rows carrying that
marker are owned by this tool and are recomputed on the next run (so lowering
the threshold cannot leave a stale exclusion behind); any row with a
hand-written reason is left exactly as it is, threshold or no threshold.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

COLUMNS = ["participant_id", "sub", "cohort", "task", "excluded", "exclusion_reason"]

# Written by --fd, in this order, after the six required columns.
FD_COLUMNS = ["mean_fd", "median_fd", "max_fd", "frac_fd_gt_0p2", "frac_fd_gt_0p5",
              "n_fd_frames", "n_motion_runs", "fd_source", "fd_columns", "fd_note"]

AUTO_PREFIX = "auto:mean_fd>"


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if value != value else f"{value:.6g}"     # NaN -> empty cell
    return str(value)


# ------------------------------------------------------------------ motion ---
def collect_fd(cfg, refs, order: str, columns: list[int] | None) -> dict[tuple[str, str], dict]:
    """(sub, task) -> motion QC row. Missing or unreadable files are reported."""
    from fmri_decomposition.cli import _attach_confounds, _attach_motion
    from fmri_decomposition.motion import (MotionError, MotionSummary,
                                           summarize_fd_column, summarize_motion_file)

    # Two cohort shapes, two sources. An AFNI cohort ships motion parameters
    # and no FD; an fMRIPrep cohort ships FD already computed and recomputing
    # it would only add a chance to get the radian/degree convention wrong.
    if cfg.confounds.motion_glob:
        _attach_motion(cfg, refs)
        source, attr = "motion_glob", "motion"
    elif cfg.confounds.format == "fmriprep_tsv" and cfg.confounds.confounds_glob:
        _attach_confounds(cfg, refs)
        source, attr = "confounds_glob", "confounds"
    else:
        raise SystemExit(
            "--fd has nothing to read. Set one of:\n"
            "  confounds.motion_glob   -- an AFNI motion regressor .1D, e.g.\n"
            '      motion_glob: "../regressors/sub-*_task-*_polort_bandpass_vent_wm_motion.1D"\n'
            "  confounds.confounds_glob with format: fmriprep_tsv -- fMRIPrep already\n"
            "      computes framewise_displacement; the column is read as-is."
        )

    out: dict[tuple[str, str], dict] = {}
    for ref in refs:
        key = (str(ref.sub), ref.task)
        if key in out:
            continue          # one row per (sub, task); first run wins, as elsewhere
        path = getattr(ref, attr)
        if path is None:
            out[key] = MotionSummary(fd_note=f"no file matched {source}").as_row()
            continue
        try:
            if attr == "motion":
                summary = summarize_motion_file(path, order=order, columns=columns)
            else:
                summary = summarize_fd_column(path, cfg.confounds.fd_column)
            out[key] = summary.as_row()
        except (MotionError, OSError, ValueError) as exc:
            out[key] = MotionSummary(
                fd_source=Path(path).name,
                fd_note=f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300],
            ).as_row()
    return out


def report_fd(rows: list[dict]) -> None:
    """Print the distribution, worst-first. The point is to be looked at."""
    have = [r for r in rows if _fmt(r.get("mean_fd"))]
    print(f"\n  motion: {len(have)}/{len(rows)} row(s) have a mean FD")
    failed = [r for r in rows if not _fmt(r.get("mean_fd"))]
    for r in failed[:5]:
        print(f"    NO FD  sub-{r['sub']}/{r['task']}: {r.get('fd_note', '')}")
    if len(failed) > 5:
        print(f"    ... and {len(failed) - 5} more without FD")
    if not have:
        return
    have.sort(key=lambda r: -float(r["mean_fd"]))
    vals = sorted(float(r["mean_fd"]) for r in have)
    mid = vals[len(vals) // 2]
    print(f"    mean_fd  min={vals[0]:.3f}  median={mid:.3f}  max={vals[-1]:.3f} mm")
    print("    worst 8 by mean_fd:")
    for r in have[:8]:
        print(f"      sub-{r['sub']:<4} {r['task']:<20} mean={float(r['mean_fd']):.3f} "
              f"max={float(r['max_fd']):.2f}  >0.5mm in "
              f"{100 * float(r['frac_fd_gt_0p5']):.1f}% of frames")


# ------------------------------------------------------------- exclusions ---
def apply_fd_exclusions(rows: list[dict], threshold: float) -> tuple[int, int]:
    """Set excluded=True/False for rows this tool owns. Returns (added, cleared).

    A row whose exclusion_reason was written by a person is never touched --
    not to exclude it, and above all not to un-exclude it. The marker is what
    makes an automatic decision reversible without also making a human one so.
    """
    added = cleared = 0
    marker = f"{AUTO_PREFIX}{threshold:g}"
    for row in rows:
        reason = str(row.get("exclusion_reason") or "")
        owned = reason.startswith(AUTO_PREFIX)
        if reason and not owned:
            continue
        mean_fd = row.get("mean_fd")
        try:
            over = mean_fd is not None and float(mean_fd) > threshold
        except (TypeError, ValueError):
            over = False
        if over:
            if not owned:
                added += 1        # a marker refresh at a new threshold is not new
            row["excluded"] = True
            row["exclusion_reason"] = marker
        elif owned:
            row["excluded"] = False
            row["exclusion_reason"] = ""
            cleared += 1
    return added, cleared


# ------------------------------------------------------------------- table ---
def read_existing(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or COLUMNS)


def write_table(path: Path, rows: list[dict], columns: list[str]) -> None:
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
                   help="read the existing CSV, keep every row and every exclusion, "
                        "and add newly discovered runs plus any --fd columns")
    p.add_argument("--fd", action="store_true",
                   help="add subject-level motion columns from confounds.motion_glob")
    p.add_argument("--motion-order", default="afni", choices=["afni", "fsl", "spm"],
                   help="parameter order when the file carries no ColumnLabels; "
                        "afni = rotations first (deg), fsl = rotations first (rad), "
                        "spm = translations first (default: afni)")
    p.add_argument("--motion-columns", type=int, nargs=6, default=None, metavar="J",
                   help="six 0-based column indices (negatives count from the end) "
                        "when the motion columns cannot be identified from the header")
    p.add_argument("--exclude-mean-fd", type=float, default=None, metavar="MM",
                   help="mark rows with mean_fd above this as excluded, with the "
                        f"reason {AUTO_PREFIX}MM. Hand-written reasons are never touched.")
    args = p.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not (args.force or args.update):
        print(f"refusing to overwrite {out}\n"
              f"  It may contain exclusions you edited by hand, which this tool "
              f"cannot reproduce.\n"
              f"  --update keeps them and adds columns; --force rebuilds from scratch.",
              file=sys.stderr)
        return 1
    if args.update and not out.exists():
        print(f"--update needs an existing {out}; drop the flag to create it.",
              file=sys.stderr)
        return 1
    if args.exclude_mean_fd is not None and not args.fd:
        print("--exclude-mean-fd needs --fd: there is nothing to threshold otherwise.",
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

    if args.fd:
        from fmri_decomposition.motion import MotionSummary

        fd = collect_fd(cfg, refs, args.motion_order, args.motion_columns)
        # A row with no run on disk gets an explicit empty summary rather than
        # keeping whatever a previous --fd run left behind: stale is worse than
        # absent when the number decides an exclusion.
        absent = MotionSummary(fd_note="no run on disk").as_row()
        for row in rows:
            row.update(fd.get((str(row.get("sub")), str(row.get("task"))), absent))
        columns = columns + [c for c in FD_COLUMNS if c not in columns]

    if args.exclude_mean_fd is not None:
        n_added, n_cleared = apply_fd_exclusions(rows, args.exclude_mean_fd)
        print(f"\n  mean_fd > {args.exclude_mean_fd:g} mm: {n_added} row(s) newly "
              f"excluded, {n_cleared} auto-exclusion(s) cleared")

    write_table(out, rows, columns)

    subs = sorted({str(r.get("sub")) for r in rows})
    tasks = sorted({str(r.get("task")) for r in rows})
    n_excluded = sum(1 for r in rows if r.get("excluded") in (True, "True"))
    print(f"\nwrote {out}")
    print(f"  {len(rows)} row(s): {len(subs)} subject(s) x {len(tasks)} task(s), "
          f"{n_excluded} excluded")
    print(f"  runs on disk: {len(refs)}"
          + (f"  ({len(refs) - len(pairs)} extra from multiple ses/run per pair)"
             if len(refs) != len(pairs) else ""))
    print(f"  subjects: {subs[:8]}{' ...' if len(subs) > 8 else ''}")
    print(f"  tasks:    {tasks[:8]}{' ...' if len(tasks) > 8 else ''}")

    if args.fd:
        report_fd(rows)
    if args.exclude_mean_fd is None:
        print("\n  Exclusions are unchanged. Edit real ones in by hand as "
              "excluded=True WITH a\n  reason -- do not delete rows, or the validator "
              "can no longer tell a curation\n  decision from an oversight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
