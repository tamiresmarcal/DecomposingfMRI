#!/usr/bin/env python3
"""Per-TR activation frames as decomposition input -- the `window_s = -1` source.

WHY -1, AND NOT A TREE OF ITS OWN
---------------------------------
A state defined on a single frame is the same KIND of object as one defined on a
60 s window: one label per time point per subject, from a fit shared across
cohorts. Writing it to `latents/atlas=<a>/window_s=-1/` therefore lets `cluster`,
`transitions` and `bstm_selection` read it with no new code path, and makes the
aperture one more axis of the selection grid instead of a fork in the pipeline.

-1 does NOT mean "no window". It means "the window is one TR", and the TR is not
the same in every cohort -- 1.00 s in ds002837, 1.49 s in cneuromod, 2.47 s in
camcan. That is why it cannot be written as a number of seconds, and it is the
one row of the grid whose aperture is NOT constant across cohorts. Every other
window size is the same duration everywhere; this one is 2.5x wider in camcan
than in ds002837. Dwell times and switch rates are reported in seconds, which
absorbs part of it, but a frame is a different amount of smoothing in each
cohort and no column can hide that. It is a caveat of this aperture, not a bug.

WHAT THIS DOES TO THE FRAMES, AND WHY DFC NEEDED NONE OF IT
-----------------------------------------------------------
A correlation is invariant to the scale and the mean of each parcel's
timeseries, which is why all three cohort configs say `standardize: false` with
the comment "correlation is invariant to it". A per-TR activation PATTERN is
invariant to neither. Two transforms are therefore applied here that stage 3
rightly never needed, both uniform across cohorts, both in `model_hash`:

  1. BAND-PASS (--match-bandpass, default 0.01-0.1 Hz). ds002837 arrives
     band-passed 0.01-1.0 Hz by AFNI; cneuromod and camcan are band-passed
     0.01-0.1 Hz by this pipeline. Left alone, ds002837 frames carry an octave
     of power the other two do not, so the state sequence flickers faster in a
     TRAINING cohort than in the PROJECTED one for a purely preprocessing
     reason -- and `switch_rate` is a feature the selection reads. The filter is
     applied to every cohort, not only to the one that is out of band, so that
     no cohort is singled out for a correction; for the two already in band it
     is close to a no-op (a cascaded Butterworth is flat in band and rolls off
     harder at the edges). Pass `--no-match-bandpass` to leave the frequency
     content as extracted, which is a different analysis.

  2. PER-RUN Z-SCORE (--no-zscore-runs to disable). Each parcel is centred and
     scaled within each run, so a frame is a spatial pattern in units of that
     run's own variability. Without it the fit is dominated by between-scanner
     signal scale -- the projected cohort can land in a region of the embedding
     the training cohorts never visit, and every state label for it is then an
     extrapolation.

Order matters: filter first (it needs a contiguous series), z-score second, drop
bad frames last.

TIME, RUNS AND GAPS
-------------------
`window_id` is `round(stimulus_time_s / tr)`: the frame's position in the
stimulus, in TRs. Same meaning as stage 3's window_id -- an index into the
stimulus, comparable across subjects -- and contiguous within a run, because
stimulus time advances by exactly `tr` per frame.

It is NOT discontinuous between runs: the stimulus clock advances only during
acquisition (timing.segments_from_scans), so the last frame of one run and the
first of the next are adjacent in stimulus time while being minutes apart in
wall-clock time. The pair across them is not a transition. So the first frame of
every run EXCEPT the earliest within a (task, sub) is marked
`crosses_run_boundary`, which is exactly what stage 5a already drops --
one frame per boundary, the same accounting dfc uses for a window that straddles
one. Dropping it leaves a gap in window_id, and `subject_transitions` splits the
sequence at gaps, so no transition is ever counted across a discontinuity.

Bad frames (`good_frame == False`) are dropped for the same reason and by the
same mechanism: the gap they leave splits the sequence. This is frame-level
scrubbing, which the dfc path could not do -- there, a bad frame could only be
summarised as a window's `frac_good_frames`.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .io import activation_root

# Everything an activation shard carries that is not a parcel. Fixed by
# activation.build_activation_table; a parcel column is anything else.
NON_FEATURE = ("t", "time_s", "stimulus_time_s", "good_frame", "run_idx",
               "ses", "run", "acq", "run_key")

# Read from the shard, used here, and not carried into the latents: `t` and
# `run_idx` build window_id and the boundary flag, `good_frame` gates rows.
_META_COLS = ("t", "time_s", "stimulus_time_s", "good_frame", "run_idx", "run_key")

# The band two of the three cohorts already carry, so harmonising onto it moves
# one cohort rather than all three. See the module docstring.
DEFAULT_BANDPASS = (0.01, 0.1)


def shard_paths(root: Path, atlas: str, cohort: str) -> list[Path]:
    """Every activation leaf for one cohort. Unlike dfc there can be SEVERAL
    per (task, sub): a leaf is one acquisition, named by ses/run/acq."""
    return sorted(activation_root(root, atlas, cohort).glob(
        "task=*/sub=*/*.parquet"))


def feature_columns(path: Path) -> list[str]:
    """Parcel columns from one shard's schema, without reading a row group."""
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    cols = [n for n in names if n not in NON_FEATURE]
    if not cols:
        raise SystemExit(
            f"{path} has no parcel columns -- only {sorted(NON_FEATURE)}. "
            f"Re-run `fmri-decomp extract` for this atlas.")
    return cols


def shard_tr(path: Path) -> float:
    """The cohort's TR, from the shard that was written with it.

    Taken from the shard rather than from a cohort YAML on purpose: this stage
    owns no cohort config (see decompose's module docstring), and the claim
    being made is "these frames were acquired at that TR", which is a property
    of the file.
    """
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = md.get(b"tr")
    if raw is None:
        raise SystemExit(f"{path} has no `tr` in its schema metadata -- it "
                         f"predates activation.build_activation_table's "
                         f"provenance block. Re-extract it.")
    return float(raw.decode())


# ----------------------------------------------------------- per-run clean ---
_FILTER_ORDER = 5
_MIN_SAMPLES = 32


def _bandpass(ts: np.ndarray, tr: float, band) -> tuple[np.ndarray, bool]:
    """Band-pass one run's parcel block. Returns (block, whether it filtered).

    scipy rather than `nilearn.signal.butterworth`, which does exactly this: the
    analysis containers carry scipy and not always nilearn, and a stage that
    cannot run for want of an optional dependency is worse than 15 lines.

    A band edge at or above Nyquist is dropped rather than clipped. At camcan's
    TR of 2.47 s Nyquist is 0.202 Hz, so 0.1 Hz is a genuine low-pass; at a TR
    above 5 s it would not be, and silently turning the band-pass into a
    high-pass would make that cohort's frequency content differ from every other
    cohort's -- which is the exact thing this filter exists to prevent.

    A run shorter than `_MIN_SAMPLES` passes through: filtfilt's edge padding
    needs more samples than the filter order, and a run that short cannot carry
    a meaningful 0.01 Hz component anyway. The caller counts them.
    """
    low, high = band
    if len(ts) < _MIN_SAMPLES:
        return ts, False
    from scipy.signal import butter, filtfilt

    nyq = 0.5 / float(tr)
    lo = low / nyq if low and 0 < low / nyq < 1 else None
    hi = high / nyq if high and 0 < high / nyq < 1 else None
    if lo and hi:
        b, a = butter(_FILTER_ORDER, (lo, hi), btype="bandpass")
    elif hi:
        b, a = butter(_FILTER_ORDER, hi, btype="lowpass")
    elif lo:
        b, a = butter(_FILTER_ORDER, lo, btype="highpass")
    else:
        return ts, False
    # axis=0 is time. filtfilt is zero-phase, so no frame is shifted relative to
    # the stimulus -- a lag here would misalign every state label.
    out = filtfilt(b, a, ts.astype(np.float64), axis=0)
    return np.ascontiguousarray(out, dtype=np.float32), True


def _zscore(ts: np.ndarray) -> np.ndarray:
    """Centre and scale each parcel within this run.

    Two cases that look alike in the arithmetic and must not be treated alike:

    * SD exactly 0 -- a real parcel that did not move over this run. 0 is the
      right answer: it is everywhere at its own mean.
    * SD NaN -- a parcel with no voxels inside this subject's mask, which
      `activation.extract_parcels` writes as an all-NaN column. It stays NaN, so
      `decompose.drop_nan_rows` drops those rows. Letting `where=sd > 0` fall
      through to the 0 would hand the fit a FABRICATED value -- "this parcel sat
      at its own mean for every frame" -- for a parcel that was never measured,
      and nothing downstream could tell the two apart. The dfc path reaches the
      same outcome by a different route: every edge touching a missing parcel is
      NaN, so that subject's windows are dropped.
    """
    mu = ts.mean(axis=0, keepdims=True)
    sd = ts.std(axis=0, keepdims=True)
    out = np.zeros_like(ts)
    np.divide(ts - mu, sd, out=out, where=sd > 0)
    out[:, np.isnan(sd).ravel()] = np.nan
    return out


def _prepare_leaf(df: pd.DataFrame, features: list[str], tr: float,
                  band, zscore: bool, stats: dict) -> pd.DataFrame:
    """One leaf -> frames, cleaned run by run, with window_id and the flag.

    Returns the leaf's rows; `stats` accumulates counts for the caller's log.
    """
    df = df.sort_values("t", kind="stable")
    blocks, metas = [], []
    for run_idx, g in df.groupby("run_idx", sort=True):
        X = g[features].to_numpy(dtype=np.float32)
        if band is not None:
            X, filtered = _bandpass(X, tr, band)
            stats["runs_filtered" if filtered else "runs_too_short"] += 1
        if zscore:
            X = _zscore(X)
        blocks.append(X)
        metas.append(g[list(_META_COLS)])
        stats["runs"] += 1

    meta = pd.concat(metas, ignore_index=True)
    out = pd.DataFrame(np.vstack(blocks), columns=features)
    out[list(_META_COLS)] = meta.reset_index(drop=True)
    return out


def _finalise_subject(df: pd.DataFrame, tr: float, stats: dict) -> pd.DataFrame:
    """All leaves of one (task, sub) -> window_id, the boundary flag, the gate.

    A (task, sub) can span several leaves, each with its own run_idx starting at
    0, so a segment is (run_key, run_idx) and not run_idx alone.
    """
    df = df.copy()
    df["window_id"] = np.rint(
        df["stimulus_time_s"].to_numpy(dtype=np.float64) / tr).astype("int64")

    seg = list(zip(df["run_key"].astype(str), df["run_idx"].astype(int)))
    df["_seg"] = pd.factorize(pd.Series(seg, index=df.index))[0]

    # Mark the first frame of a segment ONLY when its window_id continues the
    # previous segment's last one. That is the case the flag exists for: the
    # stimulus clock makes the two frames look adjacent while the scanner stopped
    # between them, so stage 5a has nothing to split on and would count a
    # transition across the gap. Where the window_ids already jump -- a stretch
    # of the stimulus that was never acquired, or a frame already dropped as bad
    # -- the sequence splits by itself and marking a frame would throw away a
    # usable one for nothing. The earliest segment is never marked: it has no
    # predecessor, so there is no spurious pair to break.
    bounds = (df.groupby("_seg")["window_id"]
              .agg(first="min", last="max", first_i="idxmin")
              .sort_values("first"))
    mark = []
    prev_last = None
    for row in bounds.itertuples():
        if prev_last is not None and row.first == prev_last + 1:
            mark.append(row.first_i)
        prev_last = row.last
    df["crosses_run_boundary"] = df.index.isin(mark)
    stats["boundary_frames"] += len(mark)

    # A duplicated window_id within a (task, sub) means two acquisitions cover
    # the same stimulus span. Interleaving them would invent transitions between
    # repeats of the same moment, so this is a hard error, not a dedup.
    dup = df.loc[df.duplicated("window_id", keep=False)]
    if len(dup):
        keys = sorted(dup["run_key"].astype(str).unique())
        raise SystemExit(
            f"duplicate window_id within one (task, sub): {len(dup)} frame(s) "
            f"across run_key(s) {keys}. Two acquisitions cover the same "
            f"stimulus span, so their frames are not a single time series. "
            f"Exclude one in the cohort config, or split them into separate "
            f"tasks -- this stage will not guess an order for them.")

    n_before = len(df)
    df = df.loc[df["good_frame"].to_numpy(dtype=bool)]
    stats["bad_frames"] += n_before - len(df)

    # The three QC columns a dfc-sourced latents file carries, so a reader does
    # not have to know which source wrote the file. Their values here are facts,
    # not placeholders: a frame IS one TR, only good frames survived the gate
    # above, and nothing in this path inverts a matrix, so rank deficiency cannot
    # arise the way it does for a short window's correlation matrix.
    df["n_tr_effective"] = np.int32(1)
    df["frac_good_frames"] = np.float32(1.0)
    df["rank_deficient"] = False
    df["start_s"] = df["time_s"].astype(np.float32)
    df["stimulus_start_s"] = df["stimulus_time_s"].astype(np.float32)
    return df.sort_values("window_id", kind="stable").drop(columns=["_seg"])


# --------------------------------------------------------------- read path ---
def read_cohort(root: Path, atlas: str, cohort: str, features: list[str],
                censor: dict | None = None, band=DEFAULT_BANDPASS,
                zscore: bool = True, want_features: bool = True, log=print):
    """One cohort -> (identity frame, float32 frame matrix or None).

    Same contract as decompose.read_cohort, including that a censored subject is
    skipped before its file is opened. Grouped by (task, sub) rather than by
    leaf, because window_id and the run-boundary flag are only well defined once
    every acquisition for that subject is in hand.
    """
    from .decompose import IDENT, _DERIVED

    paths = shard_paths(root, atlas, cohort)
    if not paths:
        raise SystemExit(f"no activation shards for cohort={cohort} at "
                         f"atlas={atlas} -- run `fmri-decomp extract` first")
    tr = shard_tr(paths[0])

    by_sub: dict[tuple[str, str], list] = {}
    n_dropped_subs = 0
    for p in paths:
        keys = dict(s.split("=", 1) for s in p.parts if "=" in s)
        if censor is not None and (keys["task"], keys["sub"]) not in censor["subjects"]:
            n_dropped_subs += 1
            continue
        by_sub.setdefault((keys["task"], keys["sub"]), []).append(p)

    ident_cols = [c for c in IDENT if c not in _DERIVED]
    stats = {"runs": 0, "runs_filtered": 0, "runs_too_short": 0,
             "boundary_frames": 0, "bad_frames": 0, "censored_windows": 0}
    idents, blocks = [], []
    for (task, sub), leaves in sorted(by_sub.items()):
        parts = []
        for p in leaves:
            df = pd.read_parquet(p, columns=list(_META_COLS) + features)
            parts.append(_prepare_leaf(df, features, tr, band, zscore, stats))
        frames = _finalise_subject(
            pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0],
            tr, stats)

        if censor is not None and censor["windows"] is not None:
            wid = frames["window_id"].to_numpy(dtype="int64")
            mask = np.array([(task, sub, w) in censor["windows"] for w in wid],
                            dtype=bool)
            stats["censored_windows"] += int((~mask).sum())
            frames = frames.loc[mask]
        if frames.empty:
            continue

        block = frames[features].to_numpy(dtype=np.float32)
        if np.isnan(block).any():
            # Reported here rather than left to drop_nan_rows' silent filter: a
            # subject lost to one unmeasured parcel is a coverage problem, and it
            # should be legible in this stage's log instead of showing up only as
            # a smaller n much later.
            bad = [features[j] for j in np.unique(np.where(np.isnan(block))[1])]
            stats.setdefault("nan_subjects", []).append((task, sub, bad))

        ident = frames[[c for c in ident_cols if c in frames.columns]].copy()
        ident["cohort"], ident["task"], ident["sub"] = cohort, task, sub
        idents.append(ident)
        if want_features:
            blocks.append(block)
        del frames, parts
    gc.collect()

    if not idents:
        raise SystemExit(
            f"nothing survived for cohort={cohort} at atlas={atlas}: "
            f"{n_dropped_subs} subject-shard(s) censored, "
            f"{stats['bad_frames']:,} frame(s) dropped as not good. Check the "
            f"censor thresholds before running the fit.")

    log(f"  {cohort}: tr={tr} {stats['runs']} run(s), "
        f"{stats['runs_filtered']} filtered"
        + (f" ({stats['runs_too_short']} too short to filter)"
           if stats["runs_too_short"] else "")
        # MARKED, not dropped: the flag is written to the latents and stage 5a
        # is what drops them, exactly as it does for a dfc window that straddles
        # a boundary. Saying "dropped" here would not match the row count.
        + f", marked {stats['boundary_frames']} run-boundary frame(s), "
        f"dropped {stats['bad_frames']:,} bad frame(s)"
        + (f", {stats['censored_windows']:,} censored" if censor else "")
        + (f", {n_dropped_subs} censored subject-shard(s)" if n_dropped_subs else ""))
    nan_subs = stats.get("nan_subjects") or []
    if nan_subs:
        parcels = sorted({c for _, _, cs in nan_subs for c in cs})
        log(f"  WARNING: {cohort}: {len(nan_subs)} subject-task(s) have an "
            f"all-NaN parcel and will be dropped entirely by drop_nan_rows "
            f"(a frame is only usable with every parcel). Parcel(s): "
            f"{parcels[:6]}{' ...' if len(parcels) > 6 else ''}. "
            f"First: {nan_subs[0][0]}/{nan_subs[0][1]}")

    ident = pd.concat(idents, ignore_index=True)
    X = np.vstack(blocks) if want_features else None
    del idents, blocks
    gc.collect()
    return ident, X, tr
