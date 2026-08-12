"""STAGE 2 -- activation: NIfTI + atlas -> parcel timeseries.

Expensive, run once. Everything that must happen before windowing happens
here: detrending, confound regression, filtering. That ordering is not a
preference -- it is the precondition for stage 3 reading parquet instead of
re-fitting a masker per window. Per-window filtering is not invariant to
windowing; pre-window filtering is.

Cleaning is applied to the parcel timeseries rather than to voxels. For every
operation that matters this is exactly equivalent and vastly cheaper:
detrending, confound regression and band-pass filtering are all linear in
time and applied identically to every voxel column, so they commute with the
fixed linear operation of parcel averaging. Standardisation does not commute,
but correlation is blind to per-column affine rescaling, so it cannot change
any stage-3 output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from .atlases.registry import AtlasSpec
from .config import CohortConfig
from .io import ManifestEntry, activation_path, should_skip, write_table_atomic
from .timing import RunSegment, TimeAxis, build_time_axis, censor_mask

# Non-parcel columns, in output order. Stage 3 relies on this contract.
META_COLUMNS = ["t", "time_s", "stimulus_time_s", "good_frame", "run_idx"]
ENTITY_COLUMNS = ["cohort", "sub", "ses", "task", "run", "acq", "run_key"]

TIME_CHUNK = 64   # volumes loaded at once; bounds peak memory on 4D files


@dataclass
class RunRef:
    """One unit of stage-2 work. The only object that knows a filesystem path."""

    cohort: str
    sub: str
    task: str
    bold: Path
    mask: Path | None = None
    ses: str | None = None
    run: str | None = None
    acq: str | None = None
    confounds: Path | None = None
    censor: Path | None = None
    segments: list[RunSegment] = field(default_factory=list)
    timing_source: str = "identity"
    trim_end_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def run_key(self) -> str:
        bits = [f"cohort-{self.cohort}", f"sub-{self.sub}", f"task-{self.task}"]
        for k, v in (("ses", self.ses), ("run", self.run), ("acq", self.acq)):
            if v:
                bits.append(f"{k}-{v}")
        return "_".join(bits)


# ------------------------------------------------------------ extraction ---
def extract_parcels(bold_img, atlas: AtlasSpec, mask: np.ndarray | None = None,
                    time_chunk: int = TIME_CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """4D image -> (n_tr, n_nodes) parcel means, plus voxel count per parcel.

    Parcels with no voxels inside the brain mask come back as NaN columns, not
    as missing columns. One file with 110 columns among 84 with 111 fails at
    read time with an unhelpful error; a NaN column is honest and joins fine.
    """
    if len(bold_img.shape) != 4:
        raise ValueError(f"expected a 4D image, got shape {bold_img.shape}")
    M, counts = atlas.membership(bold_img, mask)
    n_tr = bold_img.shape[3]
    n_vox_mask = M.shape[1]
    flat_mask = (np.ones(bold_img.shape[:3], dtype=bool) if mask is None
                 else np.asarray(mask, dtype=bool)).reshape(-1)

    out = np.empty((n_tr, atlas.n_nodes), dtype=np.float64)
    for start in range(0, n_tr, time_chunk):
        stop = min(start + time_chunk, n_tr)
        block = np.asarray(bold_img.dataobj[..., start:stop], dtype=np.float64)
        flat = block.reshape(-1, stop - start)[flat_mask]
        assert flat.shape[0] == n_vox_mask
        out[start:stop] = (M @ flat).T

    out[:, counts == 0] = np.nan
    return out.astype(np.float32), counts.astype(np.int32)


def clean_timeseries(ts: np.ndarray, tr: float, cfg: CohortConfig,
                     confounds: np.ndarray | None = None) -> np.ndarray:
    """Detrend / regress / band-pass, applied after parcel averaging."""
    f = cfg.filtering
    need_filter = (not f.already_applied) and f.bandpass is not None
    if not (f.detrend or need_filter or f.standardize or confounds is not None):
        return ts
    from nilearn.signal import clean as nl_clean

    low, high = (f.bandpass if need_filter else (None, None))
    finite = ~np.isnan(ts).any(axis=0)
    cleaned = ts.copy()
    cleaned[:, finite] = nl_clean(
        ts[:, finite], t_r=tr, detrend=f.detrend,
        standardize="zscore_sample" if f.standardize else False,
        confounds=confounds, low_pass=high, high_pass=low,
    )
    return cleaned.astype(np.float32)


def build_activation_table(ref: RunRef, atlas: AtlasSpec, cfg: CohortConfig,
                           ts: np.ndarray, axis: TimeAxis,
                           good: np.ndarray) -> pa.Table:
    """Assemble contract B. Column order and dtypes are fixed here."""
    n_tr = ts.shape[0]
    arrays: dict[str, pa.Array] = {
        "t": pa.array(axis.t, type=pa.int32()),
        "time_s": pa.array(axis.time_s, type=pa.float32()),
        "stimulus_time_s": pa.array(axis.stimulus_time_s, type=pa.float32()),
        "good_frame": pa.array(good, type=pa.bool_()),
        "run_idx": pa.array(axis.run_idx, type=pa.int8()),
        # cohort / atlas / task / sub are partition keys, carried by the path.
        # ses / run / acq live in the filename, so they stay as columns.
        "ses": pa.array([ref.ses] * n_tr, type=pa.string()),
        "run": pa.array([ref.run] * n_tr, type=pa.string()),
        "acq": pa.array([ref.acq] * n_tr, type=pa.string()),
        "run_key": pa.array([ref.run_key] * n_tr, type=pa.string()),
    }
    for j, col in enumerate(atlas.columns):
        arrays[col] = pa.array(ts[:, j], type=pa.float32())

    meta = {
        b"atlas": atlas.name.encode(),
        b"cohort": ref.cohort.encode(),
        b"task": ref.task.encode(),
        b"sub": str(ref.sub).encode(),
        b"n_nodes": str(atlas.n_nodes).encode(),
        b"tr": str(cfg.tr).encode(),
        b"timing_source": axis.source.encode(),
        b"config_hash": cfg.hash().encode(),
        b"stage": b"activation",
    }
    return pa.table(arrays).replace_schema_metadata(meta)


def process_run(ref: RunRef, atlas: AtlasSpec, cfg: CohortConfig,
                overwrite: bool = False) -> ManifestEntry:
    """Extract one run for one atlas and write its shard."""
    import nibabel as nib

    out = activation_path(cfg.output_root, ref.cohort, atlas.name, ref.task,
                          ref.sub, ref.ses, ref.run, ref.acq)
    if should_skip(out, overwrite):
        return ManifestEntry("activation", ref.cohort, atlas.name, ref.task,
                             ref.sub, str(out), "skipped")
    try:
        img = nib.load(str(ref.bold))
        n_tr_file = img.shape[3]
        mask = None
        if ref.mask is not None:
            mask_img = nib.load(str(ref.mask))
            mask = np.asarray(mask_img.dataobj) > 0
            if mask.shape != img.shape[:3]:
                raise ValueError(f"mask shape {mask.shape} != bold {img.shape[:3]}")

        ts, counts = extract_parcels(img, atlas, mask)

        confounds = None
        if ref.confounds is not None and cfg.confounds.strategy != "none":
            confounds = load_confounds(ref.confounds, cfg, n_tr_file)
        ts = clean_timeseries(ts, cfg.tr, cfg, confounds)

        axis = build_time_axis(n_tr_file, cfg.tr, ref.segments or None, ref.timing_source)
        good = _good_frames(ref, cfg, n_tr_file)

        keep = _trim_mask(ref, cfg, axis, n_tr_file)
        if keep is not None:
            ts, good = ts[keep], good[keep]
            axis = TimeAxis(axis.t[keep], axis.time_s[keep], axis.stimulus_time_s[keep],
                            axis.run_idx[keep], axis.source, axis.segments)

        table = build_activation_table(ref, atlas, cfg, ts, axis, good)
        write_table_atomic(table, out)
        detail = ""
        if (counts == 0).any():
            empty = [atlas.columns[i] for i in np.flatnonzero(counts == 0)]
            detail = f"{len(empty)} empty parcel(s): {empty[:5]}"
        return ManifestEntry("activation", ref.cohort, atlas.name, ref.task, ref.sub,
                             str(out), "ok", n_rows=table.num_rows, detail=detail)
    except Exception as exc:                                     # noqa: BLE001
        return ManifestEntry("activation", ref.cohort, atlas.name, ref.task, ref.sub,
                             str(out), "error", detail=f"{type(exc).__name__}: {exc}")


def _good_frames(ref: RunRef, cfg: CohortConfig, n_tr: int) -> np.ndarray:
    if ref.censor is not None:
        from .timing import load_afni_censor

        return load_afni_censor(str(ref.censor), n_tr, cfg.confounds.dilate_tr)
    return censor_mask(n_tr, dilate_tr=0)


def _trim_mask(ref: RunRef, cfg: CohortConfig, axis: TimeAxis, n_tr: int):
    """Apply the configured trim. Fails loudly rather than truncating silently."""
    if ref.trim_end_s is None or cfg.trim.column is None:
        return None
    end_s = float(ref.trim_end_s)
    if cfg.trim.unit == "tr":
        end_s *= cfg.tr
    n_needed = int(round(end_s / cfg.tr))
    if n_needed > n_tr:
        raise ValueError(
            f"{ref.run_key}: trim asks for {n_needed} volumes ({end_s}s at TR={cfg.tr}) "
            f"but the file has {n_tr}. Check trim.unit -- it is declared as "
            f"{cfg.trim.unit!r}."
        )
    keep = np.zeros(n_tr, dtype=bool)
    if cfg.trim.mode in ("end", "both"):
        keep[:n_needed] = True
    else:
        keep[:] = True
    return keep


def load_confounds(path: Path, cfg: CohortConfig, n_tr: int) -> np.ndarray:
    fmt = cfg.confounds.format
    if fmt == "afni_1D":
        arr = np.loadtxt(path)
    elif fmt == "fmriprep_tsv":
        df = pd.read_csv(path, sep="\t")
        cols = cfg.confounds.columns or list(df.columns)
        arr = df[cols].to_numpy()
    else:
        raise ValueError(f"unsupported confounds format {fmt!r}")
    arr = np.atleast_2d(arr)
    if arr.shape[0] != n_tr:
        arr = arr.T
    if arr.shape[0] != n_tr:
        raise ValueError(f"confounds have {arr.shape[0]} rows, expected {n_tr}")
    return np.nan_to_num(arr)
