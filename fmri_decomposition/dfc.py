"""STAGE 3 -- DFC: parcel timeseries -> windowed connectivity.

Cheap, re-run often. Reads the stage-2 parquet, never the NIfTI.

Two things define this stage:

1. **The window grid is per-stimulus, not per-file.** Window k covers
   [k*stride_s, k*stride_s + window_s) in stimulus seconds and a subject's
   frames are selected by `stimulus_time_s`. Subject A's window 47 is subject
   B's window 47 by construction rather than by luck.

2. **Pairwise deletion, not zero-filling.** A censored TR is zero in *every*
   parcel simultaneously; inside a short window that is a synchronous event
   that inflates every edge in the same direction. That is bias, not noise.
   Selecting only `good_frame` rows removes it, and `n_tr_effective` -- not
   the nominal window length -- becomes the reliability column.

Correlations are stored as raw r, not Fisher z: invertible, and arctanh is
cheap at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .atlases.registry import AtlasSpec
from .config import CohortConfig
from .io import (ManifestEntry, dfc_path, edge_storage_mode, parse_hive_keys,
                 should_skip, write_table_atomic)
from .windows import Window, make_stimulus_grid, window_tr_from_seconds

QC_COLUMNS = [
    "window_id", "start_tr", "start_s", "stimulus_start_s", "stimulus_end_s",
    "n_tr_nominal", "n_tr_available", "n_tr_effective", "frac_good_frames",
    "crosses_run_boundary", "crosses_clip_boundary", "rank_deficient",
]


# ---------------------------------------------------------- estimation ---
def pearson_upper(X: np.ndarray, clip: bool = True) -> np.ndarray:
    """Upper-triangle (k=1) Pearson correlations of the columns of X.

    NaN-safe by construction: a parcel that is all-NaN (empty under this
    subject's brain mask) or constant within the window yields NaN edges for
    that node rather than poisoning the whole matrix or raising. Returns NaN
    for every edge when fewer than 2 samples are available -- with one sample
    a correlation is not badly estimated, it is undefined.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    iu, ju = np.triu_indices(p, k=1)
    if n < 2:
        return np.full(iu.size, np.nan, dtype=np.float32)

    Xc = X - X.mean(axis=0, keepdims=True)
    ss = np.einsum("ij,ij->j", Xc, Xc)
    denom = np.sqrt(np.outer(ss, ss))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (Xc.T @ Xc) / denom
    R[~np.isfinite(R)] = np.nan
    r = R[iu, ju]
    if clip:
        r = np.clip(r, -1.0, 1.0)
    return r.astype(np.float32)


def full_matrix_from_upper(r: np.ndarray, n_nodes: int) -> np.ndarray:
    """Rebuild a symmetric matrix from the canonical upper-triangle vector."""
    M = np.full((n_nodes, n_nodes), np.nan)
    iu, ju = np.triu_indices(n_nodes, k=1)
    M[iu, ju] = r
    M[ju, iu] = r
    np.fill_diagonal(M, 1.0)
    return M


# -------------------------------------------------------------- windows ---
@dataclass
class WindowResult:
    window: Window
    r: np.ndarray
    qc: dict


def dfc_for_run(df: pd.DataFrame, atlas: AtlasSpec, cfg: CohortConfig,
                window_s: float, stimulus_duration_s: float) -> list[WindowResult]:
    """All windows for one run at one window size.

    Emission policy is permissive: a window is emitted whenever at least
    `min_n_tr_effective` good frames fall inside it, and every reason you might
    later want to exclude it travels with it as a flag. Extraction is the
    expensive irreversible step; flags are free, so changing your mind at stage
    4 must not require re-running stage 3.
    """
    cols = atlas.columns
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"activation table is missing {len(missing)} parcel column(s) for atlas "
            f"{atlas.name!r}, e.g. {missing[:3]}. Atlases cannot be mixed: atlas is "
            "the one key that changes column count."
        )
    values = df[cols].to_numpy(dtype=np.float64)
    stim = df["stimulus_time_s"].to_numpy(dtype=np.float64)
    good = (df["good_frame"].to_numpy(dtype=bool) if "good_frame" in df
            else np.ones(len(df), dtype=bool))
    run_idx = (df["run_idx"].to_numpy() if "run_idx" in df
               else np.zeros(len(df), dtype=np.int8))
    clip_idx = df["clip_idx"].to_numpy() if "clip_idx" in df else None
    t_index = df["t"].to_numpy() if "t" in df else np.arange(len(df))
    time_s = df["time_s"].to_numpy(dtype=np.float64) if "time_s" in df else stim

    n_nominal = window_tr_from_seconds(window_s, cfg.tr)
    grid = make_stimulus_grid(stimulus_duration_s, window_s,
                              cfg.windows.n_overlaps, cfg.windows.drop_incomplete)

    results: list[WindowResult] = []
    for w in grid:
        inside = w.contains(stim)
        n_available = int(inside.sum())
        sel = inside & good
        n_eff = int(sel.sum())
        if n_eff < max(2, cfg.windows.min_n_tr_effective):
            continue                      # below 2 a correlation is undefined
        r = pearson_upper(values[sel])
        qc = {
            "window_id": w.window_id,
            "start_tr": int(t_index[sel][0]),
            "start_s": float(time_s[sel][0]),
            "stimulus_start_s": float(w.start_s),
            "stimulus_end_s": float(w.end_s),
            "n_tr_nominal": n_nominal,
            "n_tr_available": n_available,
            "n_tr_effective": n_eff,
            "frac_good_frames": (n_eff / n_available) if n_available else 0.0,
            "crosses_run_boundary": bool(np.unique(run_idx[inside]).size > 1),
            "crosses_clip_boundary": bool(clip_idx is not None
                                          and np.unique(clip_idx[inside]).size > 1),
            "rank_deficient": bool(n_eff - 1 < atlas.n_nodes),
        }
        results.append(WindowResult(w, r, qc))
    return results


# ---------------------------------------------------------------- table ---
def build_dfc_table(results: list[WindowResult], atlas: AtlasSpec, cfg: CohortConfig,
                    window_s: float, entities: dict) -> pa.Table:
    n_edges = atlas.n_edges
    mode = edge_storage_mode(n_edges, cfg.windows.edge_column_threshold)
    n_rows = len(results)

    arrays: dict[str, pa.Array] = {}
    for col in QC_COLUMNS:
        vals = [res.qc[col] for res in results]
        if col in ("window_id", "start_tr"):
            arrays[col] = pa.array(vals, type=pa.int32())
        elif col in ("n_tr_nominal", "n_tr_available", "n_tr_effective"):
            arrays[col] = pa.array(vals, type=pa.int16())
        elif col.startswith(("crosses_", "rank_")):
            arrays[col] = pa.array(vals, type=pa.bool_())
        else:
            arrays[col] = pa.array(vals, type=pa.float32())

    # cohort / atlas / task / sub / window_s are partition keys, carried by the
    # path and restored at read time. Only the filename-level entities are columns.
    for key in ("ses", "run", "acq", "run_key"):
        arrays[key] = pa.array([entities.get(key)] * n_rows, type=pa.string())

    stacked = (np.vstack([res.r for res in results]) if n_rows
               else np.empty((0, n_edges), dtype=np.float32))
    if mode == "columns":
        for j, name in enumerate(atlas.edge_names()):
            arrays[name] = pa.array(stacked[:, j], type=pa.float32())
    else:
        flat = pa.array(stacked.reshape(-1), type=pa.float32())
        arrays["edges"] = pa.FixedSizeListArray.from_arrays(flat, n_edges)

    meta = {
        b"stage": b"dfc",
        b"atlas": atlas.name.encode(),
        b"cohort": str(entities.get("cohort")).encode(),
        b"task": str(entities.get("task")).encode(),
        b"sub": str(entities.get("sub")).encode(),
        b"n_nodes": str(atlas.n_nodes).encode(),
        b"n_edges": str(n_edges).encode(),
        b"edge_storage": mode.encode(),
        b"window_s": str(window_s).encode(),
        b"n_overlaps": str(cfg.windows.n_overlaps).encode(),
        b"tr": str(cfg.tr).encode(),
        b"estimator": b"pearson_pairwise_deletion",
        b"fisher_z_applied": b"false",
        b"config_hash": cfg.hash().encode(),
    }
    return pa.table(arrays).replace_schema_metadata(meta)


def process_activation_file(path: str | Path, atlas: AtlasSpec, cfg: CohortConfig,
                            window_s: float, stimulus_duration_s: float | None = None,
                            overwrite: bool = False) -> ManifestEntry:
    """One activation shard + one window size -> one DFC shard."""
    path = Path(path)
    try:
        df = pd.read_parquet(path)
        # Partition keys come from the path, which is authoritative; ses/run/acq
        # come from the columns, where the filename put them.
        keys = parse_hive_keys(path)
        ent = {k: keys.get(k) for k in ("cohort", "task", "sub")}
        for k in ("ses", "run", "acq", "run_key"):
            ent[k] = df[k].iloc[0] if k in df.columns and len(df) else None
        ent["cohort"] = ent["cohort"] or cfg.cohort
        for k, v in list(ent.items()):
            ent[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

        out = dfc_path(cfg.output_root, ent["cohort"], atlas.name, window_s,
                       ent["task"], ent["sub"], ent["ses"], ent["run"], ent["acq"])
        if should_skip(out, overwrite):
            return ManifestEntry("dfc", ent["cohort"], atlas.name, ent["task"],
                                 ent["sub"], str(out), "skipped", window_s=window_s)

        if stimulus_duration_s is None:
            observed = float(df["stimulus_time_s"].max()) + cfg.tr
            stimulus_duration_s = cfg.stimulus_duration_s(ent["task"], fallback=observed)

        results = dfc_for_run(df, atlas, cfg, window_s, stimulus_duration_s)
        if not results:
            return ManifestEntry(
                "dfc", ent["cohort"], atlas.name, ent["task"], ent["sub"], str(out),
                "empty", window_s=window_s,
                detail=f"no window of {window_s}s fits in {stimulus_duration_s:.1f}s "
                       "of usable stimulus",
            )
        table = build_dfc_table(results, atlas, cfg, window_s, ent)
        write_table_atomic(table, out)
        return ManifestEntry("dfc", ent["cohort"], atlas.name, ent["task"], ent["sub"],
                             str(out), "ok", n_rows=table.num_rows, window_s=window_s)
    except Exception as exc:                                     # noqa: BLE001
        return ManifestEntry("dfc", cfg.cohort, atlas.name, "?", "?", str(path), "error",
                             window_s=window_s, detail=f"{type(exc).__name__}: {exc}")


def read_edges(path: str | Path, atlas: AtlasSpec | None = None) -> np.ndarray:
    """Read a DFC shard's edges as (n_windows, n_edges) regardless of storage mode."""
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    mode = (meta.get(b"edge_storage") or b"columns").decode()
    if mode == "list":
        return np.vstack(table.column("edges").to_pylist()).astype(np.float32)
    names = (atlas.edge_names() if atlas is not None
             else [n for n in table.column_names if "__" in n])
    return np.column_stack([table.column(n).to_numpy() for n in names]).astype(np.float32)
