"""Time axes and censoring.

Two time columns exist because they are not the same thing. `time_s` is
acquisition time in the file, `t * TR`. `stimulus_time_s` is position in the
stimulus, and it is not derivable from the file alone: the scanner starts
before the stimulus and stops after it, and the link is a TTL trigger at the
first volume. The fallback chain (addendum §5) is, in order of preference:

    from_events -> from_log -> from_paper -> from_scans -> from_isc

Whichever was used is recorded per run so a downstream reader can tell an
alignment that was measured from one that was assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TIMING_SOURCES = ("identity", "from_events", "from_log", "from_paper", "from_scans", "from_isc")


@dataclass
class RunSegment:
    """One acquisition segment inside a (possibly concatenated) derivative."""

    n_vols: int
    stimulus_offset_s: float = 0.0   # stimulus time of this segment's first volume


@dataclass
class TimeAxis:
    t: np.ndarray                 # int32, TR index within the file
    time_s: np.ndarray            # float32, acquisition time
    stimulus_time_s: np.ndarray   # float32, position in the stimulus
    run_idx: np.ndarray           # int8
    source: str = "identity"
    segments: list[RunSegment] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t)


def build_time_axis(n_tr: int, tr: float, segments: list[RunSegment] | None = None,
                    source: str = "identity") -> TimeAxis:
    """Assemble the two time columns and the run index.

    With no segments the axis is the degenerate case: one run, offset zero,
    `stimulus_time_s == time_s`. That is a claim about the data, so it is
    tagged `identity` rather than silently presented as measured alignment.
    """
    if source not in TIMING_SOURCES:
        raise ValueError(f"unknown timing source {source!r}; expected one of {TIMING_SOURCES}")
    t = np.arange(n_tr, dtype=np.int32)
    time_s = (t * float(tr)).astype(np.float32)

    if not segments:
        return TimeAxis(t, time_s, time_s.copy(), np.zeros(n_tr, dtype=np.int8),
                        source, [RunSegment(n_tr, 0.0)])

    total = sum(s.n_vols for s in segments)
    if total != n_tr:
        raise ValueError(
            f"segments sum to {total} volumes but the file has {n_tr}. "
            "A mismatch here silently shifts stimulus time for every window."
        )
    stim = np.empty(n_tr, dtype=np.float32)
    run_idx = np.empty(n_tr, dtype=np.int8)
    pos = 0
    for i, seg in enumerate(segments):
        sl = slice(pos, pos + seg.n_vols)
        stim[sl] = seg.stimulus_offset_s + np.arange(seg.n_vols) * float(tr)
        run_idx[sl] = i
        pos += seg.n_vols
    return TimeAxis(t, time_s, stim, run_idx, source, list(segments))


def segments_from_scans(acq_times_s: list[float], n_vols: list[int], tr: float,
                        drop_after_restart: int = 0) -> list[RunSegment]:
    """scans.tsv acq_time + per-run volume counts -> segments (`from_scans`).

    The stimulus clock advances only during acquisition: a between-run gap is
    wall-clock time in which the movie was paused, so it contributes nothing to
    stimulus time. A negative gap means the arithmetic is wrong -- raise rather
    than emit a plausible-looking axis.
    """
    if len(acq_times_s) != len(n_vols):
        raise ValueError("acq_times_s and n_vols must be the same length")
    segments, stim_pos = [], 0.0
    for i, (nv, acq) in enumerate(zip(n_vols, acq_times_s)):
        if i > 0:
            duration = n_vols[i - 1] * tr
            gap = acq - acq_times_s[i - 1] - duration
            if gap < -1e-6:
                raise ValueError(
                    f"negative inter-run gap ({gap:.1f}s) between runs {i-1} and {i}: "
                    "acq_time, volume counts, or TR disagree"
                )
        usable = nv - (drop_after_restart if i > 0 else 0)
        segments.append(RunSegment(n_vols=usable, stimulus_offset_s=stim_pos))
        stim_pos += usable * tr
    return segments


def censor_mask(n_tr: int, censored: np.ndarray | list[int] | None = None,
                good: np.ndarray | None = None, dilate_tr: int = 1) -> np.ndarray:
    """Boolean `good_frame` array, with the censor mask dilated by +/-`dilate_tr`.

    Dilation is not conservatism for its own sake: scrubbing runs after
    temporal filtering, so motion artifact has already smeared into
    neighbouring volumes by the time censoring removes the epicentre.
    """
    mask = np.ones(n_tr, dtype=bool)
    if good is not None:
        g = np.asarray(good)
        if g.size != n_tr:
            raise ValueError(f"good has {g.size} entries, expected {n_tr}")
        mask &= g.astype(bool)
    if censored is not None:
        idx = np.asarray(censored, dtype=int)
        idx = idx[(idx >= 0) & (idx < n_tr)]
        mask[idx] = False
    if dilate_tr > 0:
        bad = ~mask
        dilated = bad.copy()
        for shift in range(1, dilate_tr + 1):
            dilated[shift:] |= bad[:-shift]
            dilated[:-shift] |= bad[shift:]
        mask = ~dilated
    return mask


def load_afni_censor(path: str, n_tr: int, dilate_tr: int = 1) -> np.ndarray:
    """AFNI 1D censor file (1 = keep, 0 = censor) -> good_frame."""
    vals = np.loadtxt(path).ravel()
    if vals.size != n_tr:
        raise ValueError(f"censor file has {vals.size} rows, expected {n_tr}")
    return censor_mask(n_tr, good=vals > 0.5, dilate_tr=dilate_tr)
