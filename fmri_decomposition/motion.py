"""Framewise displacement from a motion regressor file.

This module produces a SUBJECT-LEVEL motion summary, and deliberately not a
frame-level censor. On ds002837 that distinction is the whole point.

NNDb's regressors sit on the acquisition clock (`sum(raw run volumes) -
8 * n_runs`, 5469..5476 depending on run structure) while every derivative
BOLD is exactly 5470 volumes, standardised to the stimulus. The offset is
15..28 frames with no recoverable pattern, which is why `confounds.censor_glob`
is disabled in `config/ds002837.yaml`: labelling frame k with frame k+20's
verdict is worst exactly where censoring matters most.

A mean does not care. Averaging |FD| over ~5,470 frames, a 15..28 frame
misalignment at the ends moves the mean by well under a percent -- it is the
same movie, the same head, the same session, just indexed differently. So the
number this module returns is usable for exactly two things:

  1. a subject-level exclusion criterion (`mean_fd > x mm`), applied at the
     models, and
  2. a subject-level covariate in those same models.

Either way it reaches them as a column of `participants_qc.csv`, written by
`fmri-decomp diagnose`. Nothing here thresholds anything.

It is NOT usable for anything indexed by frame. Nothing here writes
`good_frame`, and nothing here is imported by stages 2 or 3.

Column identification is by AFNI header label wherever the file carries one.
Guessing the axis order matters: AFNI writes rotations FIRST (roll, pitch,
yaw, dS, dL, dP) where FSL and SPM do not, and getting it backwards silently
rescales the rotation contribution by 0.87 and the translation one by 1.15.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Power et al. (2012): rotations are converted to displacement on the surface
# of a sphere of this radius before being summed with the translations.
FD_RADIUS_MM = 50.0

# AFNI's 3dvolreg parameter order and units. `roll/pitch/yaw` are degrees,
# `dS/dL/dP` (superior / left / posterior) are millimetres.
AFNI_ROTATIONS = ("roll", "pitch", "yaw")
AFNI_TRANSLATIONS = ("ds", "dl", "dp")

# Label -> canonical slot. AFNI names only, on purpose. fMRIPrep's `rot_x` /
# `trans_x` are deliberately NOT accepted here: they carry rotations in
# RADIANS, and a lookup table that silently mixes both unit conventions is
# exactly the bug this module is trying not to have. An fMRIPrep cohort does
# not need this code path anyway -- its confounds TSV ships
# `framewise_displacement` already computed.
_ALIASES = {
    "roll": "roll", "pitch": "pitch", "yaw": "yaw",
    "ds": "ds", "dl": "dl", "dp": "dp",
}

# (rotation column indices, translation column indices, rotation unit).
PARAM_ORDERS = {
    "afni": ((0, 1, 2), (3, 4, 5), "deg"),
    "fsl":  ((0, 1, 2), (3, 4, 5), "rad"),
    "spm":  ((3, 4, 5), (0, 1, 2), "deg"),
}

FD_THRESHOLDS_MM = (0.2, 0.5)   # reported as frac_fd_gt_*; conventional, not law


class MotionError(RuntimeError):
    """Raised when a motion file cannot be read *unambiguously*."""


@dataclass
class MotionSummary:
    """One subject x task. Every field is a scalar so it joins onto a CSV row."""

    n_fd_frames: int = 0            # frames with a defined FD (excludes run starts)
    n_motion_runs: int = 1
    mean_fd: float = float("nan")
    median_fd: float = float("nan")
    max_fd: float = float("nan")
    frac_fd_gt_0p2: float = float("nan")
    frac_fd_gt_0p5: float = float("nan")
    fd_source: str = ""             # basename of the file the numbers came from
    fd_columns: str = ""            # which columns were used, and how they were found
    fd_note: str = ""               # anything a reader must know to trust the row

    def as_row(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 1D I/O ---
def read_1d(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read an AFNI .1D file into (values, header).

    AFNI writes its metadata as `# key = "value"` comment lines above the
    matrix, and `np.loadtxt` throws them away. They are the difference between
    knowing which columns are motion and guessing, so they are parsed here.
    """
    path = Path(path)
    header: dict[str, str] = {}
    kv = re.compile(r"^\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
    with open(path) as fh:
        for line in fh:
            if not line.lstrip().startswith("#"):
                continue
            m = kv.match(line)
            if m:
                header[m.group(1)] = m.group(2).strip().strip('"').strip()
    values = np.atleast_2d(np.loadtxt(path, comments="#", ndmin=2))
    if values.size == 0:
        raise MotionError(f"{path.name}: no numeric rows")
    return values.astype(np.float64), header


def header_column_labels(header: dict[str, str]) -> list[str]:
    """`# ColumnLabels = "Run#1Pol#0 ; ... ; roll_01 ; ..."` -> a list."""
    raw = header.get("ColumnLabels")
    if not raw:
        return []
    return [lab.strip() for lab in raw.split(";")]


# -------------------------------------------------------- column selection ---
def _base_label(label: str) -> str:
    """`roll_01` -> `roll`. AFNI appends a stimulus index to every label."""
    return re.sub(r"[_#]\d+$", "", label.strip()).lower()


def select_motion_columns(
    values: np.ndarray,
    header: dict[str, str] | None = None,
    order: str = "afni",
    columns: list[int] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], tuple[int, ...], str], str]:
    """Pull the six motion parameters out of a regressor matrix.

    Returns `(params, (rot_idx, trans_idx, rot_unit), provenance)` where
    `params` is (n_frames, 6) in the column order of the file's own six slots.

    Three paths, in decreasing order of trustworthiness:

      1. explicit `columns` -- the caller has read the file and knows;
      2. AFNI `ColumnLabels` -- the file says which column is `roll`;
      3. a 6- or 12-column file with no labels -- assume `order`.

    Anything else raises. A 130-column design matrix with no labels has no
    safe default: the motion block is at the end *by convention*, and a
    convention is not a guarantee worth a silently wrong covariate.
    """
    header = header or {}
    n_cols = values.shape[1]

    if columns is not None:
        if len(columns) != 6:
            raise MotionError(f"--motion-columns needs exactly 6 indices, got {len(columns)}")
        idx = [c if c >= 0 else n_cols + c for c in columns]
        bad = [c for c in idx if not 0 <= c < n_cols]
        if bad:
            raise MotionError(f"motion column index out of range for a {n_cols}-column file: {bad}")
        return values[:, idx], PARAM_ORDERS[order], f"explicit columns {list(columns)} ({order} order)"

    labels = header_column_labels(header)
    if labels and len(labels) == n_cols:
        groups: dict[str, list[int]] = {}
        for j, lab in enumerate(labels):
            slot = _ALIASES.get(_base_label(lab))
            if slot:
                groups.setdefault(slot, []).append(j)
        wanted = list(AFNI_ROTATIONS) + list(AFNI_TRANSLATIONS)
        if all(slot in groups for slot in wanted):
            # Per-run motion regressors (`-regress_motion_per_run`) give one
            # zero-padded column per run per axis. They are disjoint in time by
            # construction, so summing them rebuilds the concatenated series.
            params = np.column_stack([values[:, groups[slot]].sum(axis=1) for slot in wanted])
            per_run = max(len(groups[s]) for s in wanted)
            how = "AFNI ColumnLabels" + (f", {per_run} per-run block(s) summed" if per_run > 1 else "")
            return params, PARAM_ORDERS["afni"], how

    if n_cols in (6, 12):
        # 12 columns is AFNI's demean+deriv pair; the parameters are the first six.
        return (values[:, :6], PARAM_ORDERS[order],
                f"first 6 of {n_cols} columns, {order} order assumed (no ColumnLabels)")

    tail = values[:, -6:]
    ranges = ", ".join(f"[{lo:+.3f},{hi:+.3f}]" for lo, hi in zip(tail.min(0), tail.max(0)))
    raise MotionError(
        f"cannot identify the motion columns: {n_cols} columns and no usable "
        f"ColumnLabels header.\n"
        f"  The last 6 columns span {ranges}.\n"
        f"  If those are the motion parameters, say so explicitly:\n"
        f"    --motion-columns -6 -5 -4 -3 -2 -1 --motion-order {order}\n"
        f"  Guessing is not offered: AFNI writes rotations first, FSL and SPM "
        f"do not, and the wrong order is silently wrong rather than loud."
    )


def run_starts_from_matrix(values: np.ndarray, header: dict[str, str] | None = None) -> list[int]:
    """Frame indices at which a new acquisition run begins.

    Three sources, again in decreasing order of trustworthiness: AFNI's
    `RunStart` header; the block structure of a concatenated design matrix
    (per-run polort columns are zero outside their own run, so their nonzero
    support tiles the timeline); otherwise one run.

    This matters more than its size suggests. Differencing straight across a
    run boundary reports the head's position between two separate acquisitions
    as a single frame's movement -- typically a several-millimetre spike, and
    the largest FD in the file.
    """
    header = header or {}
    n = values.shape[0]
    raw = header.get("RunStart")
    if raw:
        try:
            starts = sorted({int(float(x)) for x in raw.replace(",", " ").split()})
            if starts and starts[0] == 0 and all(0 <= s < n for s in starts):
                return starts
        except ValueError:
            pass

    spans: dict[tuple[int, int], int] = {}
    for j in range(values.shape[1]):
        nz = np.flatnonzero(values[:, j])
        if nz.size == 0:
            continue
        spans[(int(nz[0]), int(nz[-1]))] = spans.get((int(nz[0]), int(nz[-1])), 0) + 1
    blocks = sorted(s for s in spans if s[1] - s[0] + 1 < n)
    if len(blocks) >= 2:
        # Accept only a clean tiling: gaps or overlaps mean these are not runs.
        tiles, pos = [], 0
        for lo, hi in blocks:
            if lo != pos:
                tiles = []
                break
            tiles.append(lo)
            pos = hi + 1
        if tiles and pos == n:
            return tiles
    return [0]


# --------------------------------------------------------------------- FD ---
def framewise_displacement(
    params: np.ndarray,
    rot_idx: tuple[int, ...] = (0, 1, 2),
    trans_idx: tuple[int, ...] = (3, 4, 5),
    rot_unit: str = "deg",
    run_starts: list[int] | None = None,
    radius_mm: float = FD_RADIUS_MM,
) -> np.ndarray:
    """Power et al. (2012) FD, in mm, length `n_frames`.

    The first frame of every run is NaN, not zero. Zero is a measurement --
    "the head did not move" -- and there is no measurement to make: there is no
    preceding volume to difference against. A NaN is dropped from the mean;
    a zero would drag it down, by more for a subject scanned in six runs than
    one scanned in two, which is a motion-correlated bias in a motion metric.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] < 6:
        raise MotionError(f"expected an (n_frames, 6) parameter array, got {params.shape}")
    n = params.shape[0]

    rot = params[:, list(rot_idx)]
    if rot_unit == "deg":
        rot = np.deg2rad(rot)
    elif rot_unit != "rad":
        raise MotionError(f"rot_unit must be 'deg' or 'rad', got {rot_unit!r}")
    disp = np.column_stack([params[:, list(trans_idx)], rot * float(radius_mm)])

    fd = np.full(n, np.nan)
    starts = sorted(set(run_starts or [0]) | {0})
    bounds = list(zip(starts, starts[1:] + [n]))
    for lo, hi in bounds:
        if hi - lo < 2:
            continue
        fd[lo + 1:hi] = np.abs(np.diff(disp[lo:hi], axis=0)).sum(axis=1)
    return fd


def summarize_fd(fd: np.ndarray, n_runs: int = 1) -> MotionSummary:
    finite = fd[np.isfinite(fd)]
    if finite.size == 0:
        return MotionSummary(n_motion_runs=n_runs, fd_note="no defined FD frames")
    return MotionSummary(
        n_fd_frames=int(finite.size),
        n_motion_runs=int(n_runs),
        mean_fd=float(finite.mean()),
        median_fd=float(np.median(finite)),
        max_fd=float(finite.max()),
        frac_fd_gt_0p2=float((finite > FD_THRESHOLDS_MM[0]).mean()),
        frac_fd_gt_0p5=float((finite > FD_THRESHOLDS_MM[1]).mean()),
    )


def summarize_fd_column(path: str | Path, column: str = "framewise_displacement") -> MotionSummary:
    """Summarise an FD column that a preprocessor already computed.

    fMRIPrep ships `framewise_displacement` in its confounds TSV, computed
    from the same Power formula. Recomputing it from `rot_*`/`trans_*` would
    only add a chance to get the radian/degree convention wrong, so this reads
    what is there. The first sample is NaN by fMRIPrep's own convention --
    there is no preceding volume -- and is dropped from the mean, exactly as
    a run start is above.
    """
    import pandas as pd

    path = Path(path)
    df = pd.read_csv(path, sep="\t")
    if column not in df.columns:
        raise MotionError(
            f"{path.name} has no {column!r} column; set confounds.fd_column. "
            f"Available example columns: {list(df.columns)[:6]}"
        )
    summary = summarize_fd(df[column].to_numpy(dtype=float), n_runs=1)
    summary.fd_source = path.name
    summary.fd_columns = f"{column!r}, as computed upstream"
    return summary


def summarize_motion_file(
    path: str | Path,
    order: str = "afni",
    columns: list[int] | None = None,
    radius_mm: float = FD_RADIUS_MM,
) -> MotionSummary:
    """One motion regressor file -> one row of subject-level motion QC."""
    if order not in PARAM_ORDERS:
        raise MotionError(f"motion order must be one of {sorted(PARAM_ORDERS)}, got {order!r}")
    path = Path(path)
    values, header = read_1d(path)
    params, (rot_idx, trans_idx, rot_unit), how = select_motion_columns(
        values, header, order=order, columns=columns)
    starts = run_starts_from_matrix(values, header)
    fd = framewise_displacement(params, rot_idx, trans_idx, rot_unit,
                                run_starts=starts, radius_mm=radius_mm)
    summary = summarize_fd(fd, n_runs=len(starts))
    summary.fd_source = path.name
    summary.fd_columns = how
    if len(starts) == 1 and values.shape[1] > 6:
        summary.fd_note = ("run boundaries not recoverable from this file; FD "
                           "differenced as one continuous run")
    return summary
