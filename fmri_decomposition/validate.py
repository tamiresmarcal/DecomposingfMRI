"""Diagnostics that the design calls for as artifacts, not as afterthoughts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .atlases.registry import AtlasSpec
from .dfc import pearson_upper
from .io import read_shard


def equivalence_check(bold_img, atlas: AtlasSpec, windows: list[tuple[int, int]],
                      mask=None, tol: float = 1e-6) -> pd.DataFrame:
    """Recompute windows both ways and report the max absolute difference.

    Way A: slice the extracted parcel timeseries (what stage 3 does).
    Way B: slice the 4D image first, then extract parcels from the slice
           (what the legacy pipeline did, ~600 mask fits per subject).

    They agree to float noise because parcel averaging is a fixed linear
    operation and correlation is invariant to per-column affine rescaling.
    Cheap to run, and it documents the equivalence rather than asserting it.
    """
    import nibabel as nib

    from .activation import extract_parcels

    full_ts, _ = extract_parcels(bold_img, atlas, mask)
    rows = []
    for start, stop in windows:
        r_a = pearson_upper(full_ts[start:stop])
        sliced = nib.Nifti1Image(
            np.asarray(bold_img.dataobj[..., start:stop]), bold_img.affine, bold_img.header
        )
        ts_b, _ = extract_parcels(sliced, atlas, mask)
        r_b = pearson_upper(ts_b)
        diff = np.nanmax(np.abs(r_a - r_b)) if r_a.size else 0.0
        rows.append({"start_tr": start, "stop_tr": stop,
                     "max_abs_diff": float(diff), "passes": bool(diff < tol)})
    return pd.DataFrame(rows)


def lr_correlation_diagnostic(activation_files, atlas: AtlasSpec,
                              threshold: float = 0.95) -> pd.DataFrame:
    """L-R correlation for every symmetric_split pair, across subjects.

    nilearn splits at the x-midline, which cuts genuinely midline structures
    (cingulate, precuneus, SMA, frontal medial cortex) into near-duplicate L/R
    pairs that load together in PCA. Pairs above ~0.95 have empirical grounds
    for merging -- an argument from data rather than from anatomy.
    """
    labels = atlas.labels.copy()
    labels["column"] = atlas.columns
    base = labels["name"].str.replace(r"^(Left|Right)[\s_]+", "", regex=True)
    labels["base"] = base
    pairs = []
    for stem, grp in labels.groupby("base"):
        hemis = dict(zip(grp["hemi"], grp["column"]))
        if "L" in hemis and "R" in hemis:
            pairs.append((stem, hemis["L"], hemis["R"]))

    acc: dict[str, list[float]] = {stem: [] for stem, _, _ in pairs}
    for path in activation_files:
        df = pd.read_parquet(path)
        good = df["good_frame"].to_numpy(bool) if "good_frame" in df else np.ones(len(df), bool)
        for stem, lcol, rcol in pairs:
            if lcol not in df or rcol not in df:
                continue
            a, b = df.loc[good, lcol].to_numpy(), df.loc[good, rcol].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0:
                acc[stem].append(float(np.corrcoef(a[ok], b[ok])[0, 1]))

    out = pd.DataFrame(
        [
            {
                "region": stem, "left": lcol, "right": rcol,
                "n_subjects": len(acc[stem]),
                "mean_lr_r": float(np.mean(acc[stem])) if acc[stem] else np.nan,
                "merge_candidate": bool(acc[stem] and np.mean(acc[stem]) >= threshold),
            }
            for stem, lcol, rcol in pairs
        ],
        columns=["region", "left", "right", "n_subjects", "mean_lr_r", "merge_candidate"],
    )
    if out.empty:
        return out            # no symmetric_split pairs: nothing to diagnose
    return out.sort_values("mean_lr_r", ascending=False, ignore_index=True)


ISC_MIN_SUBJECTS = 3


def isc_alignment(activation_files, parcels=("Heschls_Gyrus_includes_H1_and_H2",),
                  max_lag_tr: int = 30) -> pd.DataFrame:
    """Cross-correlate each subject against the leave-one-out group mean.

    For cohorts with no timing record this is not a validation, it is the only
    way to obtain the offset. Gate stage 3 on it: refuse if median |best_lag|
    exceeds one TR.

    **Grouped by task, always.** The leave-one-out reference is only meaningful
    among people who saw the SAME stimulus: pooling ds002837's ten films would
    build a "group mean" out of ten unrelated soundtracks, truncate everyone to
    the shortest film, and report a lag for each subject against noise. That is
    not a conservative approximation, it is a different quantity. This was
    latent while `include_tasks` named a single film and became live when the
    cohort opened to all ten.

    A task with fewer than ISC_MIN_SUBJECTS subjects gets rows with NaN and a
    reason rather than being dropped: on ds002837 eight of the ten films have
    six subjects, so the reference is a mean of five, and `n_subjects` travels
    with every row so that is visible rather than implied.
    """
    by_task: dict[str, list[tuple[str, np.ndarray]]] = {}
    for path in activation_files:
        df = read_shard(path)
        cols = [c for c in parcels if c in df.columns]
        if not cols:
            continue
        x = df[cols].to_numpy(dtype=float).mean(axis=1)
        by_task.setdefault(str(df["task"].iloc[0]), []).append((str(df["sub"].iloc[0]), x))
    if not by_task:
        raise ValueError(
            f"no activation shard carries any of the ISC parcel(s) {list(parcels)}"
        )

    rows = []
    for task in sorted(by_task):
        entries = by_task[task]
        if len(entries) < ISC_MIN_SUBJECTS:
            rows += [{"sub": sub, "movie": task, "best_lag_tr": np.nan,
                      "peak_isc": np.nan, "n_subjects": len(entries),
                      "isc_note": f"only {len(entries)} subject(s) saw this stimulus; "
                                  f"ISC needs {ISC_MIN_SUBJECTS}"}
                     for sub, _ in entries]
            continue
        rows += _isc_one_task(task, entries, max_lag_tr)
    if not any(np.isfinite(r["peak_isc"]) for r in rows):
        raise ValueError(
            f"no task has {ISC_MIN_SUBJECTS} or more subjects; ISC is not computable"
        )
    return pd.DataFrame(rows)


def _isc_one_task(task: str, entries, max_lag_tr: int) -> list[dict]:
    """One stimulus. Everyone here saw the same thing, so a group mean exists."""
    n = min(len(x) for _, x in entries)
    X = np.vstack([_z(x[:n]) for _, x in entries])
    total = X.sum(axis=0)
    out = []
    for i, (sub, _) in enumerate(entries):
        loo = _z((total - X[i]) / (len(entries) - 1))
        best_lag, best_r = 0, -np.inf
        for lag in range(-max_lag_tr, max_lag_tr + 1):
            a, b = _shift(X[i], lag), loo
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 10:
                continue
            r = float(np.corrcoef(a[ok], b[ok])[0, 1])
            if r > best_r:
                best_lag, best_r = lag, r
        out.append({"sub": sub, "movie": task, "best_lag_tr": best_lag,
                    "peak_isc": best_r, "n_subjects": len(entries), "isc_note": ""})
    return out


def isc_gate(isc: pd.DataFrame, max_median_lag_tr: float = 1.0) -> tuple[bool, str]:
    """Cohort-level PASS/FAIL on the median |lag|. Excludes no individual."""
    lags = np.abs(pd.to_numeric(isc["best_lag_tr"], errors="coerce").to_numpy(dtype=float))
    lags = lags[np.isfinite(lags)]
    if lags.size == 0:
        return False, "no subject has a computable ISC lag"
    med = float(np.median(lags))
    ok = med <= max_median_lag_tr
    n_missing = len(isc) - lags.size
    tail = f"; {n_missing} row(s) had no computable lag" if n_missing else ""
    return ok, (f"median |best_lag| = {med:.2f} TR over {lags.size} subject(s) "
                f"({'within' if ok else 'exceeds'} the {max_median_lag_tr} TR gate){tail}")


def coverage_table(activation_files) -> pd.DataFrame:
    """meta/cohort-*_coverage.parquet: n_good / n_total per stimulus TR.

    Motion is not random with respect to the stimulus -- people move at
    startling scenes and during boring stretches, so censored frames cluster
    at particular movie moments across subjects simultaneously. This cannot be
    fixed, only measured, so that stage 4 can test whether conclusions survive
    restriction to well-sampled windows.
    """
    frames = []
    for path in activation_files:
        df = read_shard(path)
        good = df["good_frame"].to_numpy(bool) if "good_frame" in df else np.ones(len(df), bool)
        frames.append(pd.DataFrame({
            "movie": df["task"].to_numpy(),
            "stimulus_time_s": df["stimulus_time_s"].to_numpy(),
            "n_good": good.astype(int),
            "n_total": 1,
        }))
    if not frames:
        return pd.DataFrame(columns=["movie", "stimulus_time_s", "n_good", "n_total"])
    allf = pd.concat(frames, ignore_index=True)
    return (allf.groupby(["movie", "stimulus_time_s"], as_index=False)
            .agg(n_good=("n_good", "sum"), n_total=("n_total", "sum")))


def _z(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else x * np.nan


def _shift(x, lag):
    out = np.full_like(x, np.nan, dtype=float)
    if lag == 0:
        return x.copy()
    if lag > 0:
        out[lag:] = x[:-lag]
    else:
        out[:lag] = x[-lag:]
    return out


def write_diagnostic(df: pd.DataFrame, output_root: str | Path, name: str,
                     cohort: str | None = None) -> Path:
    """Diagnostics are cohort-scoped: coverage and ISC describe one cohort's data."""
    from .io import cohort_meta_dir, meta_dir

    d = meta_dir(output_root) if cohort is None else cohort_meta_dir(output_root, cohort)
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path
