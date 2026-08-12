"""Storage layout, atomic writes, manifest.

Hive-partitioned parquet. A partitioned dataset is not a file that gets
appended to -- partition keys are directory names, and the dataset only
exists at read time when pyarrow walks the tree. Each worker owns a distinct
leaf and writes one file, so there are no locks and nothing to consolidate.

Directory depth is constant across cohorts by construction: leftover entities
(ses, run, acq) go in the filename, never in a directory.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

COMPRESSION = "zstd"


# ---------------------------------------------------------------- paths ---
def _key(value: Any) -> str:
    """Hive key values must be filesystem-safe and stable across platforms."""
    s = str(value)
    for bad in ("/", "\\", "=", " "):
        s = s.replace(bad, "-")
    return s


def leaf_filename(ses: str | None = None, run: str | None = None,
                  acq: str | None = None, suffix: str = ".parquet") -> str:
    """Deterministic leaf name so skip_if_exists can just stat() it.

    Deliberately not the Spark `part-0` convention: that exists for when one
    logical partition spans several physical files, which never happens here.
    """
    bits = [f"{k}-{_key(v)}" for k, v in (("ses", ses), ("run", run), ("acq", acq)) if v]
    return ("_".join(bits) if bits else "data") + suffix


def activation_root(output_root: str | Path, atlas: str,
                    cohort: str | None = None) -> Path:
    """The shallowest schema-homogeneous root: one atlas, any number of cohorts.

    Atlas is the outermost key because it is the only one that changes *column
    count* (111 vs 14 vs 7). Everything below a given `atlas=` directory shares
    a schema, so this path can be handed straight to `pyarrow.dataset`. Pass a
    cohort to narrow to one; omit it to pool across cohorts in a single read.
    """
    p = Path(output_root) / "activation" / f"atlas={_key(atlas)}"
    return p if cohort is None else p / f"cohort={_key(cohort)}"


def dfc_root(output_root: str | Path, atlas: str, window_s: float | None = None,
             cohort: str | None = None) -> Path:
    """Same idea for stage 3, with `window_s` between atlas and cohort.

    window_s sits above cohort so that "one atlas, one window size, every
    cohort" is a single directory -- the cross-cohort pooling query stage 4
    actually runs. It changes row count, never column count, so it is safe
    anywhere below atlas.
    """
    p = Path(output_root) / "dfc" / f"atlas={_key(atlas)}"
    if window_s is not None:
        p = p / f"window_s={_key(_fmt_window(window_s))}"
    return p if cohort is None else p / f"cohort={_key(cohort)}"


def activation_path(output_root: str | Path, cohort: str, atlas: str, task: str,
                    sub: str, ses: str | None = None, run: str | None = None,
                    acq: str | None = None) -> Path:
    return (
        activation_root(output_root, atlas, cohort)
        / f"task={_key(task)}"
        / f"sub={_key(sub)}"
        / leaf_filename(ses, run, acq)
    )


def dfc_path(output_root: str | Path, cohort: str, atlas: str, window_s: float,
             task: str, sub: str, ses: str | None = None, run: str | None = None,
             acq: str | None = None) -> Path:
    return (
        dfc_root(output_root, atlas, window_s, cohort)
        / f"task={_key(task)}"
        / f"sub={_key(sub)}"
        / leaf_filename(ses, run, acq)
    )


def _fmt_window(window_s: float) -> str:
    return str(int(window_s)) if float(window_s).is_integer() else str(window_s)


# Keys carried by the directory tree. They are deliberately NOT written as
# columns: a partition key duplicated as a column must match its inferred type
# exactly, and pyarrow reads `sub=01` as int32 -- which both collides with the
# string column and quietly destroys Cam-CAN's `CC110033` and NNDb's leading
# zeros. The path is the single source of truth; `read_shard` puts them back
# when a single file is opened on its own.
PARTITION_KEYS = {
    "activation": ["atlas", "cohort", "task", "sub"],
    "dfc": ["atlas", "window_s", "cohort", "task", "sub"],
}


def parse_hive_keys(path: str | Path) -> dict[str, str]:
    """Recover partition keys from a leaf path. Everything comes back as a string."""
    return dict(
        part.split("=", 1) for part in Path(path).parts if "=" in part
    )


def hive_partitioning(stage: str = "dfc"):
    """Explicit all-string partitioning, so no key is ever type-inferred."""
    import pyarrow.dataset as pads

    keys = PARTITION_KEYS[stage]
    return pads.partitioning(pa.schema([pa.field(k, pa.string()) for k in keys]),
                             flavor="hive")


def open_dataset(root: str | Path, stage: str = "dfc"):
    """Open a partitioned dataset with the correct key types.

    Always use this rather than `partitioning="hive"`: the latter infers types
    per key and will turn subject ids into integers.
    """
    import pyarrow.dataset as pads

    return pads.dataset(str(root), format="parquet", partitioning=hive_partitioning(stage))


def read_shard(path: str | Path):
    """Read one leaf file and restore its partition keys as columns."""
    import pandas as pd

    df = pd.read_parquet(path)
    for k, v in parse_hive_keys(path).items():
        if k not in df.columns:
            df[k] = v
    return df


def meta_dir(output_root: str | Path) -> Path:
    """Atlas- and model-level metadata: valid for every cohort."""
    return Path(output_root) / "meta"


def cohort_meta_dir(output_root: str | Path, cohort: str) -> Path:
    """Everything scoped to one cohort: manifests, coverage, ISC, participants.

    Now that atlas sits above cohort in the data tree, a cohort no longer has a
    subtree of its own. This is where its provenance lives instead, keyed the
    same hive way so it can be discovered by the same walk.
    """
    return meta_dir(output_root) / "cohorts" / f"cohort={_key(cohort)}"


def atlas_labels_path(output_root: str | Path, atlas: str) -> Path:
    """Atlas label tables are cohort-independent: one file, shared by all."""
    return meta_dir(output_root) / f"atlas-{_key(atlas)}_labels.csv"


# --------------------------------------------------------------- writes ---
def write_table_atomic(table: pa.Table, final_path: str | Path,
                       compression: str = COMPRESSION) -> Path:
    """Write then rename. A file at its final path is complete by construction.

    This is what makes a walltime kill mid-write harmless: the partial file is
    a .tmp that skip_if_exists ignores, not a truncated shard that reads as
    valid until it doesn't.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(table, tmp, compression=compression)
        os.replace(tmp, final_path)   # atomic within a filesystem
    finally:
        if tmp.exists():
            tmp.unlink()
    return final_path


def should_skip(path: str | Path, overwrite: bool = False) -> bool:
    return (not overwrite) and Path(path).exists()


def cleanup_stale_tmp(root: str | Path, older_than_s: float = 86_400) -> list[Path]:
    """Remove .tmp shards abandoned by killed workers."""
    now = time.time()
    removed = []
    for p in Path(root).rglob("*.tmp.*"):
        if now - p.stat().st_mtime > older_than_s:
            p.unlink()
            removed.append(p)
    return removed


# ------------------------------------------------------------- manifest ---
@dataclass
class ManifestEntry:
    stage: str
    cohort: str
    atlas: str
    task: str
    sub: str
    path: str
    status: str                 # ok | skipped | empty | error
    n_rows: int = 0
    window_s: float | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def write_manifest(output_root: str | Path, entries: Iterable[ManifestEntry],
                   config: dict[str, Any] | None = None,
                   name: str = "manifest.json", extra: dict | None = None,
                   cohort: str | None = None) -> Path:
    from . import __version__

    d = meta_dir(output_root) if cohort is None else cohort_meta_dir(output_root, cohort)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "package_version": __version__,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": _versions(),
        "config": config or {},
        "entries": [e.as_dict() for e in entries],
    }
    if extra:
        payload.update(extra)
    path = d / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)
    return path


def _versions() -> dict[str, str]:
    import numpy
    import pandas

    v = {"numpy": numpy.__version__, "pandas": pandas.__version__, "pyarrow": pa.__version__}
    for mod in ("nibabel", "nilearn"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = "absent"
    return v


# --------------------------------------------------------- edge storage ---
def edge_storage_mode(n_edges: int, threshold: int = 20_000) -> str:
    """'columns' below the threshold, 'list' above it.

    Parquet carries per-column metadata and per-column-per-row-group
    statistics, so pyarrow degrades in the low tens of thousands of columns.
    Harvard-Oxford-111 gives 6,105 edges (columns, interpretable loadings);
    Schaefer-1000 gives 499,500 (one fixed_size_list column).
    """
    return "columns" if n_edges <= threshold else "list"
