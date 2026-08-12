"""Windowing.

The grid is defined **once per stimulus**, in stimulus seconds (addendum §4).
Window k covers the half-open interval [k*stride_s, k*stride_s + window_s).
A subject's frames are selected by `stimulus_time_s` falling inside it, so
`window_id` is globally meaningful within a movie and cross-subject
comparison is correct by construction.

Nothing in this module touches a file, a TR count, or an atlas. It is pure
arithmetic and is the single place window boundaries are defined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

# Float slack for grid arithmetic. Window edges are exact multiples of
# stride_s; without slack, 3 * (30/3) can land at 29.999999999999996.
_EPS = 1e-9


@dataclass(frozen=True)
class Window:
    """One window on the stimulus grid. Times are stimulus seconds."""

    window_id: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def contains(self, stimulus_time_s: np.ndarray) -> np.ndarray:
        """Half-open membership test: start <= t < end.

        Half-open matters: with 80% overlap, a closed interval would put the
        frame at a boundary into two adjacent windows, double-counting it.
        """
        t = np.asarray(stimulus_time_s, dtype=np.float64)
        return (t >= self.start_s - _EPS) & (t < self.end_s - _EPS)


def window_tr_from_seconds(window_s: float, tr: float) -> int:
    """Nominal window length in TRs. `window_s` is the label; this is derived.

    Rounding policy is fixed here (round-half-up) rather than left to the
    caller, so two cohorts with the same TR can never disagree.
    """
    if tr <= 0:
        raise ValueError(f"tr must be positive, got {tr}")
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    return int(math.floor(window_s / tr + 0.5))


def stride_seconds(window_s: float, n_overlaps: int) -> float:
    """Stride in seconds. n_overlaps=5 gives 80% overlap."""
    if n_overlaps < 1:
        raise ValueError(f"n_overlaps must be >= 1, got {n_overlaps}")
    return window_s / n_overlaps


def n_windows(stimulus_duration_s: float, window_s: float, n_overlaps: int,
              drop_incomplete: bool = True) -> int:
    """How many windows the grid holds. Zero if the stimulus is too short."""
    if stimulus_duration_s <= 0:
        return 0
    stride = stride_seconds(window_s, n_overlaps)
    if drop_incomplete:
        if stimulus_duration_s + _EPS < window_s:
            return 0
        return int(math.floor((stimulus_duration_s - window_s) / stride + _EPS)) + 1
    return int(math.ceil(stimulus_duration_s / stride - _EPS))


def make_stimulus_grid(
    stimulus_duration_s: float,
    window_s: float,
    n_overlaps: int = 5,
    drop_incomplete: bool = True,
) -> list[Window]:
    """Build the window grid for one stimulus.

    Deterministic given (duration, window_s, n_overlaps): every subject who
    saw this stimulus gets the identical grid, which is the whole point.

    A run shorter than the window yields an empty grid rather than a partial
    window -- stage 4 must tolerate an empty partition, and the manifest
    records the coverage (handoff §8).
    """
    stride = stride_seconds(window_s, n_overlaps)
    k_max = n_windows(stimulus_duration_s, window_s, n_overlaps, drop_incomplete)
    return [Window(k, k * stride, k * stride + window_s) for k in range(k_max)]


def iter_stimulus_grid(*args, **kwargs) -> Iterator[Window]:
    """Generator form, for grids too large to materialise."""
    yield from make_stimulus_grid(*args, **kwargs)


def make_index_windows(
    n_tr: int,
    window_tr: int,
    n_overlaps: int = 5,
    drop_incomplete: bool = True,
) -> list[tuple[int, int]]:
    """Legacy-style index windows over file rows: [(start_tr, stop_tr), ...].

    Retained only for the equivalence check against the old pipeline and for
    cohorts with no stimulus timing at all. Prefer `make_stimulus_grid`.

    Note the contrast with the legacy loop, which rebound the loop variable
    inside the inner overlap loop and so emitted duplicated and unevenly
    spaced starts (handoff §11, bug 1). Here the stride is uniform and every
    start appears exactly once.
    """
    if window_tr < 2:
        raise ValueError(f"window_tr must be >= 2, got {window_tr}")
    stride = max(1, window_tr // n_overlaps)
    if drop_incomplete:
        starts = range(0, max(0, n_tr - window_tr + 1), stride)
        return [(s, s + window_tr) for s in starts]
    starts = range(0, n_tr, stride)
    return [(s, min(s + window_tr, n_tr)) for s in starts]


def assign_window_ids(stimulus_time_s: Sequence[float], window_s: float,
                      n_overlaps: int = 5) -> list[np.ndarray]:
    """For each frame, the ids of every window that contains it.

    Diagnostic helper: with n_overlaps=k an interior frame belongs to exactly
    k windows, which is a cheap way to spot a broken grid.
    """
    stride = stride_seconds(window_s, n_overlaps)
    t = np.asarray(stimulus_time_s, dtype=np.float64)
    out = []
    for ti in t:
        hi = int(math.floor(ti / stride + _EPS))
        # strict: k*stride + window_s > ti, so equality is excluded
        lo = int(math.floor((ti - window_s) / stride + _EPS)) + 1
        out.append(np.arange(max(0, lo), hi + 1, dtype=np.int64))
    return out
