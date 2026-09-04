"""Per-subject QC metrics, computed from stage-2 output.

Four measurements, chosen to be about four DIFFERENT failures. A long list of
correlated criteria costs sample size while only looking rigorous; these are
the axes that catch things the others cannot:

    mean_fd                 head movement            -> motion.py
    best_lag_tr             wrong stimulus timing    -> validate.isc_alignment
    frac_stimulus_covered   incomplete scan          -> here
    frac_parcels_empty      registration failure     -> here
    frac_good_frames        scrubbing survival       -> here (cohort-dependent)

MEASUREMENT HERE, DECISION AT THE MODELS
----------------------------------------
This module measures. It never thresholds and never excludes, and no threshold
appears anywhere in the pipeline.

    mean_fd = 0.52          a measurement: deterministic, reproducible
    0.52 > 0.5 -> exclude   a decision: arguable, and part of the claim

The first belongs in the pipeline and runs automatically; the second belongs
with the analysis that rests on it, so a sensitivity analysis can move a cutoff
without re-running anything upstream.

That is also why the output is `participants_qc.csv` and not a column added to
`participants.csv`. The two files differ in who owns them:

    participants.csv      HUMAN-owned. Curation only -- "corrupted run",
                          "consent withdrawn". Load-bearing at stage 2:
                          `attach_participants` drops these rows before
                          extraction. Hand-edited, and nothing writes it
                          automatically.
    participants_qc.csv   MACHINE-owned. Rewritten from scratch by every
                          `diagnose` run. Never hand-edited; an edit here is
                          lost on the next finalize.

Written by `fmri-decomp diagnose`, which `03_finalize.sbatch` already runs
after the extract array -- so the metrics appear without a separate step, and
ISC is computed once rather than twice.

WHY frac_stimulus_covered EXISTS
--------------------------------
Neither cohort configures `stimulus.durations_s`, so `dfc.py` builds each
subject's window grid from THAT SUBJECT'S OWN file length. Same origin, same
stride -- so window ids still line up -- but a subject whose scan stopped early
simply has no rows in the late windows. No error, no flag, no manifest entry.
The group mean for late-movie windows is then computed over fewer people than
for early ones, and nothing in the outputs says so. This is the one metric here
that is about correctness rather than quality.

WHY frac_parcels_empty IS THE REGISTRATION CHECK
------------------------------------------------
`extract_parcels` writes NaN for any parcel with no voxels under the subject's
brain mask. A handful is normal at the edges. Thirty out of Harvard-Oxford's
111 means the brain is not where MNI space says it should be -- which is what a
failed normalisation looks like, measured by its consequence rather than by
re-inspecting a transform we did not compute.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_shard

# Auditory cortex tracks a film's soundtrack more reliably than anything else,
# which is what makes it the conventional ISC seed. Matched case-insensitively
# against the QC atlas's column names, in order, first hit wins.
ISC_PARCEL_PATTERNS = (r"heschl", r"transverse.*temporal", r"\bauditory\b",
                       r"sommot|somatomotor", r"\bvisual\b|\bvis\b")


@dataclass
class SubjectQC:
    """One (sub, task). Scalars only, so it joins straight onto a CSV row."""

    n_tr_total: int = 0
    n_tr_good: int = 0
    frac_good_frames: float = float("nan")
    stimulus_duration_s: float = float("nan")
    frac_stimulus_covered: float = float("nan")
    n_parcels: int = 0
    n_parcels_empty: int = 0
    frac_parcels_empty: float = float("nan")
    qc_atlas: str = ""
    peak_isc: float = float("nan")
    best_lag_tr: float = float("nan")
    n_isc_subjects: int = 0
    qc_note: str = ""

    def as_row(self) -> dict:
        return asdict(self)


# A note exists to be acted on. The ds002837 motion error names the exact flag
# that resolves it, and at 300 characters that instruction was being cut
# mid-word -- leaving a cell that says there is a problem but not what to do.
QC_NOTE_MAX = 600


def _clip(text, limit: int = QC_NOTE_MAX) -> str:
    """Collapse a multi-line message into one readable, actionable cell.

    Exception text spans several lines; a CSV cell is one. Whitespace is
    collapsed rather than newlines merely replaced, so indentation does not
    survive as runs of spaces, and the cut lands on a word boundary and says
    that it happened.
    """
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + " [...]"


def _note(existing: str, addition: str) -> str:
    return "; ".join(x for x in (existing, addition) if x)


# ------------------------------------------------------------ activation ---
def activation_qc(files, atlas, tr: float) -> dict[tuple[str, str], dict]:
    """Walk one atlas's activation shards -> per (sub, task) QC.

    A (sub, task) with several ses/run shards is aggregated: frames sum, the
    duration is the longest shard's, and the empty-parcel count is the WORST
    shard's. A single badly-registered session is a finding, not something to
    average away.
    """
    cols = list(atlas.columns)
    acc: dict[tuple[str, str], dict] = {}
    for path in files:
        df = read_shard(path)
        key = (str(df["sub"].iloc[0]), str(df["task"].iloc[0]))
        present = [c for c in cols if c in df.columns]
        good = (df["good_frame"].to_numpy(bool) if "good_frame" in df
                else np.ones(len(df), bool))
        duration = (float(df["stimulus_time_s"].max()) + float(tr)
                    if "stimulus_time_s" in df and len(df) else len(df) * float(tr))
        # A parcel is empty exactly when extract_parcels wrote it as all-NaN,
        # which it does for `counts == 0` -- no voxel of this brain inside it.
        empty = int(sum(1 for c in present
                        if not np.isfinite(df[c].to_numpy(dtype=float)).any()))

        cur = acc.setdefault(key, {"n_tr_total": 0, "n_tr_good": 0,
                                   "stimulus_duration_s": 0.0, "n_parcels": len(present),
                                   "n_parcels_empty": 0, "qc_atlas": atlas.name,
                                   "qc_note": ""})
        cur["n_tr_total"] += int(len(df))
        cur["n_tr_good"] += int(good.sum())
        cur["stimulus_duration_s"] = max(cur["stimulus_duration_s"], duration)
        cur["n_parcels_empty"] = max(cur["n_parcels_empty"], empty)
        if len(present) != len(cols):
            cur["qc_note"] = _note(cur["qc_note"],
                                   f"{len(cols) - len(present)} parcel column(s) absent "
                                   f"from the shard")
    for cur in acc.values():
        cur["frac_good_frames"] = (cur["n_tr_good"] / cur["n_tr_total"]
                                   if cur["n_tr_total"] else float("nan"))
        cur["frac_parcels_empty"] = (cur["n_parcels_empty"] / cur["n_parcels"]
                                     if cur["n_parcels"] else float("nan"))
    return acc


def add_stimulus_coverage(qc: dict[tuple[str, str], dict]) -> None:
    """frac_stimulus_covered, relative to the longest scan of the SAME stimulus.

    Per task, not per cohort: ds002837's films run 95 to 154 minutes, so a
    cohort-wide reference would mark every Pulp Fiction viewer as complete and
    every 500 Days viewer as 60% covered. The reference is the longest observed
    scan rather than a configured duration because no duration is configured --
    it is the same fallback `dfc.py` uses to build the grid, so the two agree
    by construction.
    """
    longest: dict[str, float] = {}
    for (_, task), cur in qc.items():
        d = float(cur.get("stimulus_duration_s") or 0.0)
        longest[task] = max(longest.get(task, 0.0), d)
    for (_, task), cur in qc.items():
        ref = longest.get(task, 0.0)
        d = float(cur.get("stimulus_duration_s") or 0.0)
        cur["frac_stimulus_covered"] = (d / ref) if ref > 0 else float("nan")


# ---------------------------------------------------------------- motion ---
def motion_qc(cfg, order: str = "afni", columns=None):
    """(sub, task) -> subject-level motion. See motion.py for what it claims.

    Two cohort shapes, two sources: an AFNI cohort ships motion parameters and
    no FD, an fMRIPrep cohort ships FD already computed -- recomputing that from
    rot_*/trans_* would only add a chance to get the radian convention wrong.
    """
    from .cli import _attach_confounds, _attach_motion
    from .cohort import discover_runs
    from .motion import (MotionError, MotionSummary, summarize_fd_column,
                         summarize_motion_file)

    if cfg.confounds.motion_glob:
        source, attr = "confounds.motion_glob", "motion"
    elif cfg.confounds.format == "fmriprep_tsv" and cfg.confounds.confounds_glob:
        source, attr = "confounds.confounds_glob", "confounds"
    else:
        return {}, ("motion skipped: set confounds.motion_glob (an AFNI motion .1D) "
                    "or confounds.confounds_glob with format: fmriprep_tsv")

    refs = discover_runs(cfg)
    _attach_motion(cfg, refs)
    _attach_confounds(cfg, refs)

    out: dict[tuple[str, str], dict] = {}
    for ref in refs:
        key = (str(ref.sub), ref.task)
        if key in out:
            continue          # one row per (sub, task); first run wins, as elsewhere
        path = getattr(ref, attr)
        if path is None:
            out[key] = MotionSummary(fd_note=f"no file matched {source}").as_row()
            continue
        if Path(path).is_symlink() and not Path(path).exists():
            # A datalad/git-annex dataset stores unfetched content as a dangling
            # symlink. `glob` returns it because the link exists; opening it
            # raises FileNotFoundError for a path that is plainly there, which
            # reads as a broken pipeline rather than as absent data.
            out[key] = MotionSummary(
                fd_source=Path(path).name,
                fd_note="unfetched git-annex pointer -- `datalad get` this file "
                        "from OUTSIDE the container, then re-run diagnose",
            ).as_row()
            continue
        try:
            summary = (summarize_motion_file(path, order=order, columns=columns)
                       if attr == "motion"
                       else summarize_fd_column(path, cfg.confounds.fd_column))
            out[key] = summary.as_row()
        except (MotionError, OSError, ValueError) as exc:        # noqa: BLE001
            out[key] = MotionSummary(
                fd_source=Path(path).name,
                fd_note=_clip(f"{type(exc).__name__}: {exc}"),
            ).as_row()
    have = sum(1 for r in out.values() if r["mean_fd"] == r["mean_fd"])
    return out, f"motion: {have}/{len(out)} run(s) from {source}"


# ------------------------------------------------------------------- ISC ---
def default_isc_parcels(atlas) -> tuple[list[str], str]:
    """Pick an ISC seed from the QC atlas. Returns (columns, how)."""
    for pattern in ISC_PARCEL_PATTERNS:
        hits = [c for c in atlas.columns if re.search(pattern, c, flags=re.I)]
        if hits:
            return hits, f"matched /{pattern}/ in {atlas.name}"
    return list(atlas.columns[:1]), (
        f"NO auditory or visual parcel in {atlas.name}; fell back to "
        f"{atlas.columns[0]!r}, which is arbitrary -- pass --isc-parcels"
    )


def isc_qc(files, atlas, parcels=None, max_lag_tr: int = 30):
    """(sub, task) -> peak_isc / best_lag_tr, a message, and the raw frame."""
    from .validate import isc_alignment

    how = "supplied by the caller"
    if not parcels:
        parcels, how = default_isc_parcels(atlas)
    try:
        isc = isc_alignment(files, parcels=tuple(parcels), max_lag_tr=max_lag_tr)
    except (ValueError, KeyError) as exc:                        # noqa: BLE001
        return {}, f"ISC skipped: {type(exc).__name__}: {exc}", None
    out = {}
    for row in isc.to_dict("records"):
        out[(str(row["sub"]), str(row["movie"]))] = {
            "peak_isc": float(row["peak_isc"]),
            "best_lag_tr": float(row["best_lag_tr"]),
            "n_isc_subjects": int(row.get("n_subjects") or 0),
            "qc_note": str(row.get("isc_note") or ""),
        }
    return out, f"ISC seed: {', '.join(parcels[:3])} ({how})", isc


# ------------------------------------------------------------ collection ---
def pick_qc_atlas(cfg, atlases):
    """The atlas the QC is measured on: the finest one that has shards.

    Empty parcels are the point. Yeo's 7 networks are so large that a
    normalisation would have to fail catastrophically to empty one, while
    Harvard-Oxford's 111 register the same failure as a count you can threshold.
    """
    from .io import activation_root

    with_shards = [a for a in atlases
                   if any(activation_root(cfg.output_root, a.name, cfg.cohort)
                          .rglob("*.parquet"))]
    if not with_shards:
        return None
    return max(with_shards, key=lambda a: a.n_nodes)


def collect_qc(cfg, atlases, isc_parcels=None, max_lag_tr: int = 30,
               motion_order: str = "afni", motion_columns=None):
    """Everything above, for one cohort.

    Returns `(rows_by_key, messages, isc_frame)`. The ISC frame is handed back
    so `diagnose` can write it and run the gate on the same computation rather
    than doing it twice.
    """
    from .io import activation_root

    messages: list[str] = []
    atlas = pick_qc_atlas(cfg, atlases)
    if atlas is None:
        return {}, ["no activation shards found -- run `extract` first"], None
    files = sorted(activation_root(cfg.output_root, atlas.name, cfg.cohort)
                   .rglob("*.parquet"))
    messages.append(f"QC atlas: {atlas.name} ({atlas.n_nodes} nodes), {len(files)} shard(s)")

    qc = activation_qc(files, atlas, cfg.tr)
    add_stimulus_coverage(qc)

    isc, isc_msg, isc_frame = isc_qc(files, atlas, isc_parcels, max_lag_tr)
    messages.append(isc_msg)
    for key, row in isc.items():
        cur = qc.setdefault(key, {})
        note = row.pop("qc_note", "")
        cur.update(row)
        if note:
            cur["qc_note"] = _note(cur.get("qc_note", ""), note)

    # Discovery finds every run on disk; only the extracted ones have shards.
    # A row for a run that was never extracted is honest, but must say so --
    # otherwise it reads as a metric that failed rather than one never computed.
    extracted = set(qc)

    motion, motion_msg = motion_qc(cfg, motion_order, motion_columns)
    messages.append(motion_msg)
    for key, row in motion.items():
        note = row.pop("fd_note", "")
        cur = qc.setdefault(key, {})
        cur.update(row)
        if note:
            cur["qc_note"] = _note(cur.get("qc_note", ""), note)

    for key in set(qc) - extracted:
        qc[key]["qc_note"] = _note(qc[key].get("qc_note", ""),
                                   "no activation shard on disk -- this run was "
                                   "discovered but never extracted")

    from .motion import MotionSummary

    blank = {**MotionSummary().as_row(), **SubjectQC().as_row()}
    blank.pop("fd_note", None)
    return {k: {**blank, **v} for k, v in qc.items()}, messages, isc_frame


def qc_frame(cfg, qc: dict[tuple[str, str], dict]):
    """The per-subject QC rows as a DataFrame, keyed and column-ordered.

    Keyed the same way `participants.csv` is -- (cohort, sub, task) -- because
    a join on that pair is the only thing the models have to do to use it.
    """
    rows = []
    for (sub, task), row in sorted(qc.items()):
        rows.append({"participant_id": f"sub-{sub}", "sub": sub,
                     "cohort": cfg.cohort, "task": task, **row})
    df = pd.DataFrame(rows, columns=QC_COLUMNS)
    return df.sort_values(["task", "sub"], ignore_index=True) if len(df) else df


# Column order of participants_qc.csv: keys, then one block per failure mode,
# then provenance. `qc_note` is last because it is prose.
QC_COLUMNS = [
    "participant_id", "sub", "cohort", "task",
    # motion
    "mean_fd", "median_fd", "max_fd", "frac_fd_gt_0p2", "frac_fd_gt_0p5",
    "n_fd_frames", "n_motion_runs",
    # timing
    "best_lag_tr", "peak_isc", "n_isc_subjects",
    # coverage
    "frac_stimulus_covered", "stimulus_duration_s", "n_tr_total",
    # scrubbing
    "frac_good_frames", "n_tr_good",
    # registration
    "frac_parcels_empty", "n_parcels", "n_parcels_empty", "qc_atlas",
    # provenance
    "fd_source", "fd_columns", "qc_note",
]
