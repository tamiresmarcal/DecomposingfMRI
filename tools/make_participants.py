#!/usr/bin/env python3
"""Generate or update a participants CSV from what discovery finds on disk.

    # build the table
    python tools/make_participants.py config/cneuromod_friends.yaml \
        -o config/cneuromod_friends_participants.csv

    # add subject-level motion and QC to a table that already exists,
    # without touching the exclusions someone edited in by hand
    python tools/make_participants.py config/ds002837.yaml \
        -o config/ds002837_participants.csv --update --fd --qc

    # ... and let the four criteria decide, each recorded with its own reason
    python tools/make_participants.py config/ds002837.yaml \
        -o config/ds002837_participants.csv --update --fd --qc \
        --exclude-mean-fd 0.5 --exclude-lag-tr 1 \
        --exclude-stimulus-covered 0.95 --exclude-parcels-empty 0.05

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

WHAT `--qc` ADDS
----------------
Per-subject QC read out of the stage-2 parquet (see fmri_decomposition/qc.py).
Four criteria, chosen to be about four DIFFERENT failures, because a long list
of correlated ones costs sample size while only looking rigorous:

    mean_fd                 head movement           (--fd, not --qc)
    best_lag_tr             wrong stimulus timing
    frac_stimulus_covered   scan stopped early
    frac_parcels_empty      registration failure

plus frac_good_frames, which measures scrubbing survival and is meaningful only
where censoring is on -- it is identically 1.0 on ds002837.

EXCLUSIONS
----------
With no --exclude-* flag, every generated row is excluded=False and this tool
never changes an exclusion. Real exclusions are edited in by hand and RECORDED
as excluded=True with a reason -- never by deleting the row. `validate_cohort`
treats a run on disk with no table row as a problem precisely so that "is this
excluded, or did someone forget?" stays answerable.

The --exclude-* flags stay inside that rule. Each writes a machine-readable
reason (`auto:mean_fd>0.5`, `auto:abs(best_lag_tr)>1`, ...) and several are
joined with "; ". Rows whose reason consists only of such markers are owned by
this tool and are RECOMPUTED from scratch every run, so raising a threshold
releases the rows it used to catch. A row with a hand-written reason -- or one
mixing a human's words with a marker -- is left exactly as it is, in both
directions. A row whose metric is missing is never excluded: no motion file is
not evidence of good motion.

Nothing here deletes a row, and nothing here deletes a shard. An exclusion
written after `extract` does not retroactively stop stage 3, which walks the
filesystem rather than this table -- `fmri-decomp dfc` warns when it finds
shards for excluded subjects.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

COLUMNS = ["participant_id", "sub", "cohort", "task", "excluded", "exclusion_reason"]

# Written by --fd, in this order, after the six required columns.
FD_COLUMNS = ["mean_fd", "median_fd", "max_fd", "frac_fd_gt_0p2", "frac_fd_gt_0p5",
              "n_fd_frames", "n_motion_runs", "fd_source", "fd_columns", "fd_note"]

AUTO_PREFIX = "auto:"


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


def report_qc(rows: list[dict]) -> None:
    """Print each QC metric's distribution and its worst subjects.

    Thresholds are meant to be fixed BEFORE this is read -- but the
    distribution still has to be visible, because a metric whose spread is
    invisible cannot be sanity-checked. Reported per task where the reference
    is per task.
    """
    metrics = [("frac_stimulus_covered", "low", "coverage"),
               ("frac_parcels_empty", "high", "registration"),
               ("frac_good_frames", "low", "scrubbing"),
               ("peak_isc", "low", "stimulus-drivenness"),
               ("best_lag_tr", "abs", "timing")]
    print("\n  per-subject QC:")
    for col, worst, what in metrics:
        vals = []
        for r in rows:
            try:
                v = float(r.get(col))
            except (TypeError, ValueError):
                continue
            if v == v:
                vals.append((v, r))
        if not vals:
            print(f"    {col:<24} not computed")
            continue
        xs = sorted(v for v, _ in vals)
        mid = xs[len(xs) // 2]
        if worst == "high":
            vals.sort(key=lambda t: -t[0])
        elif worst == "abs":
            vals.sort(key=lambda t: -abs(t[0]))
        else:
            vals.sort(key=lambda t: t[0])
        tail = ", ".join(f"sub-{r.get('sub')}/{r.get('task')}={v:.3g}"
                         for v, r in vals[:3])
        print(f"    {col:<24} n={len(xs):<4} min={xs[0]:.3g} median={mid:.3g} "
              f"max={xs[-1]:.3g}   [{what}]")
        print(f"    {'':<24} worst: {tail}")


# ------------------------------------------------------------- exclusions ---
@dataclass
class Rule:
    """One automatic exclusion criterion.

    `column` is a participants.csv column, `op` is 'gt' or 'lt', and `abs_`
    thresholds |value| (which is what a timing lag needs -- lag -12 is as wrong
    as +12). A row is excluded when the comparison is TRUE.
    """

    column: str
    op: str                       # gt | lt
    threshold: float
    abs_: bool = False

    @property
    def marker(self) -> str:
        col = f"abs({self.column})" if self.abs_ else self.column
        return f"{AUTO_PREFIX}{col}{'>' if self.op == 'gt' else '<'}{self.threshold:g}"

    def fires(self, row: dict) -> bool:
        """Missing or unparseable is NEVER a failure.

        No motion file is not evidence of good motion, and an uncomputable ISC
        is not evidence of a bad subject. A row that cannot be judged is left
        for a person to judge.
        """
        try:
            v = float(row.get(self.column))
        except (TypeError, ValueError):
            return False
        if v != v:                                        # NaN
            return False
        if self.abs_:
            v = abs(v)
        return v > self.threshold if self.op == "gt" else v < self.threshold


def _is_owned(reason: str) -> bool:
    """True when every part of the reason was written by this tool.

    A reason mixing a human's words with an auto marker counts as the human's:
    the safe direction is to leave it alone.
    """
    parts = [p.strip() for p in str(reason or "").split(";") if p.strip()]
    return bool(parts) and all(p.startswith(AUTO_PREFIX) for p in parts)


def apply_auto_exclusions(rows: list[dict], rules: list[Rule]) -> dict[str, int]:
    """Recompute every automatic exclusion from scratch. Returns per-rule counts.

    Recomputing rather than accumulating is what makes a threshold reversible:
    raise a cutoff and the rows it used to catch are released, because their
    reason is rebuilt from the rules currently in force. A row whose
    exclusion_reason a person wrote is never touched -- not to exclude it, and
    above all not to un-exclude it.
    """
    counts = {r.marker: 0 for r in rules}
    counts["_cleared"] = 0
    counts["_protected"] = 0
    for row in rows:
        reason = str(row.get("exclusion_reason") or "")
        was_excluded = row.get("excluded") in (True, "True", "true", 1, "1")
        if reason and not _is_owned(reason):
            counts["_protected"] += 1
            continue
        if reason and not rules:
            continue                      # owned, but no rules given: leave as-is
        fired = [r.marker for r in rules if r.fires(row)]
        for m in fired:
            counts[m] += 1
        if fired:
            row["excluded"] = True
            row["exclusion_reason"] = "; ".join(fired)
        else:
            row["excluded"] = False
            row["exclusion_reason"] = ""
            if was_excluded:
                counts["_cleared"] += 1
    return counts


def report_exclusions(rows: list[dict], rules: list[Rule], counts: dict) -> None:
    """Who failed what, and where the criteria overlap.

    The overlap matters: "excluded 9" is not interpretable, and if one
    criterion accounts for all nine the other four are decoration.
    """
    print(f"\n  automatic exclusions ({len(rules)} rule(s)):")
    for r in rules:
        print(f"    {counts.get(r.marker, 0):>4}  {r.marker}")
    if counts.get("_cleared"):
        print(f"    {counts['_cleared']:>4}  released (no rule fires at these thresholds)")
    if counts.get("_protected"):
        print(f"    {counts['_protected']:>4}  left alone (hand-written reason)")

    excluded = [r for r in rows if _is_owned(r.get("exclusion_reason", ""))]
    if not excluded:
        return
    n_multi = sum(1 for r in excluded if ";" in str(r["exclusion_reason"]))
    print(f"    {len(excluded)} row(s) excluded in total, "
          f"{n_multi} by more than one rule")
    for row in excluded[:10]:
        print(f"      sub-{row.get('sub')}/{row.get('task')}: {row['exclusion_reason']}")
    if len(excluded) > 10:
        print(f"      ... and {len(excluded) - 10} more")


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
    p.add_argument("--qc", action="store_true",
                   help="add per-subject QC columns from the stage-2 output: "
                        "frac_good_frames, frac_stimulus_covered, frac_parcels_empty, "
                        "peak_isc and best_lag_tr (requires `extract` to have run)")
    p.add_argument("--isc-parcels", nargs="*", default=None,
                   help="parcel column(s) to seed ISC with; default is an auditory "
                        "parcel of the QC atlas, or a visual one")
    p.add_argument("--isc-max-lag-tr", type=int, default=30,
                   help="lag range searched by ISC, in TRs (default 30)")

    g = p.add_argument_group(
        "automatic exclusions",
        "Each writes excluded=True with a machine-readable reason. All of them are "
        "RECOMPUTED on every run, so raising a threshold releases the rows it used "
        "to catch; a hand-written exclusion_reason is never touched in either "
        "direction. A row whose metric is missing is never excluded -- absence of "
        "evidence is not evidence.")
    g.add_argument("--exclude-mean-fd", type=float, default=None, metavar="MM",
                   help="motion: exclude mean_fd > MM (e.g. 0.5)")
    g.add_argument("--exclude-lag-tr", type=float, default=None, metavar="TR",
                   help="timing: exclude |best_lag_tr| > TR (e.g. 1). Needs --qc. "
                        "This one is a correctness failure, not a quality gradient: "
                        "the subject's stimulus_time_s is wrong.")
    g.add_argument("--exclude-stimulus-covered", type=float, default=None, metavar="FRAC",
                   help="coverage: exclude frac_stimulus_covered < FRAC (e.g. 0.95). "
                        "Needs --qc.")
    g.add_argument("--exclude-parcels-empty", type=float, default=None, metavar="FRAC",
                   help="registration: exclude frac_parcels_empty > FRAC (e.g. 0.05). "
                        "Needs --qc.")
    g.add_argument("--exclude-good-frames", type=float, default=None, metavar="FRAC",
                   help="scrubbing: exclude frac_good_frames < FRAC (e.g. 0.5). Needs "
                        "--qc, and is meaningless on a cohort with censoring disabled, "
                        "where it is identically 1.0.")
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
    rules: list[Rule] = []
    if args.exclude_mean_fd is not None:
        rules.append(Rule("mean_fd", "gt", args.exclude_mean_fd))
    if args.exclude_lag_tr is not None:
        rules.append(Rule("best_lag_tr", "gt", args.exclude_lag_tr, abs_=True))
    if args.exclude_stimulus_covered is not None:
        rules.append(Rule("frac_stimulus_covered", "lt", args.exclude_stimulus_covered))
    if args.exclude_parcels_empty is not None:
        rules.append(Rule("frac_parcels_empty", "gt", args.exclude_parcels_empty))
    if args.exclude_good_frames is not None:
        rules.append(Rule("frac_good_frames", "lt", args.exclude_good_frames))

    if args.exclude_mean_fd is not None and not args.fd:
        print("--exclude-mean-fd needs --fd: there is nothing to threshold otherwise.",
              file=sys.stderr)
        return 1
    needs_qc = [f"--exclude-{n}" for n, v in (
        ("lag-tr", args.exclude_lag_tr),
        ("stimulus-covered", args.exclude_stimulus_covered),
        ("parcels-empty", args.exclude_parcels_empty),
        ("good-frames", args.exclude_good_frames)) if v is not None]
    if needs_qc and not args.qc:
        print(f"{', '.join(needs_qc)} need(s) --qc: there is nothing to threshold "
              f"otherwise.", file=sys.stderr)
        return 1

    from fmri_decomposition.config import load_config
    from fmri_decomposition.cohort import discover_runs
    from fmri_decomposition.atlases.registry import get_atlas

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

    if args.qc:
        from fmri_decomposition.qc import QC_COLUMNS, SubjectQC, collect_qc

        atlases = [get_atlas(n, **cfg.atlas_params.get(n, {})) for n in cfg.atlases]
        qc, messages = collect_qc(cfg, atlases, args.isc_parcels, args.isc_max_lag_tr)
        for m in messages:
            print(f"  {m}")
        absent = SubjectQC(qc_note="no activation shard on disk").as_row()
        for row in rows:
            row.update(qc.get((str(row.get("sub")), str(row.get("task"))), absent))
        columns = columns + [c for c in QC_COLUMNS if c not in columns]

    if rules:
        counts = apply_auto_exclusions(rows, rules)
        report_exclusions(rows, rules, counts)

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
    if args.qc:
        report_qc(rows)
    if not rules:
        print("\n  Exclusions are unchanged. Edit real ones in by hand as "
              "excluded=True WITH a\n  reason -- do not delete rows, or the validator "
              "can no longer tell a curation\n  decision from an oversight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
