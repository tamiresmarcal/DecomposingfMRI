"""Loaders for the analysis notebooks: partition-pruned, column-projected reads.

The pipeline writes hive-partitioned parquet whose partition keys are directory
names. Nothing here re-implements that -- every path is built by
`fmri_decomposition.io`, and this module only adds what a notebook needs on top:
an inventory of what is actually on disk, filtered reads that never materialise
more than they were asked for, and the participants / QC / phenotype join.

Three things it exists to stop you doing.

**Reading a whole stage.** `pd.read_parquet("outputs/dfc")` is a full walk and a
full materialisation. Every loader here takes partition filters (`cohort`,
`window_s`, `task`, `sub`) that prune whole directories before a file is opened,
and a column projection that decides how much of each file is read.

**Opening a dataset that spans atlases.** `harvardoxford`, `networks` and `yeo7`
have 111, 14 and 7 parcel columns. One dataset object per atlas, always -- so
`dataset()` takes an atlas and refuses to be called without one.

**Trusting a partition key that is above the dataset root.** This one is quiet
and costs an afternoon. A hive key is recovered from the path *relative to the
root*, so opening at `dfc/atlas=yeo7/` leaves `atlas` as a column of nulls
rather than "yeo7" -- and a filter on it then matches nothing, silently. The
loaders here backfill every key the root swallowed, and `dataset()` returns the
keys it could not recover so a caller can see which ones were filled in.

Memory, in the shapes that actually appear:

    activation, yeo7, one cohort, all subjects      tens of MB
    activation, harvardoxford, one cohort           0.05-0.3 GB
    dfc QC columns only, any atlas, one window      a few MB
    dfc + edges, harvardoxford, one cohort, 30 s    1-2.5 GB   <- the cliff
    dfc + edges, yeo7 (21 edges), one cohort        tens of MB

so `with_edges=True` on a fine atlas is guarded by a footer-based estimate
rather than discovered by an OOM kill.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fmri_decomposition.dfc import QC_COLUMNS as DFC_QC_COLUMNS  # noqa: E402
from fmri_decomposition.dfc import full_matrix_from_upper, read_edges  # noqa: E402
from fmri_decomposition.io import (PARTITION_KEYS, _fmt_window,  # noqa: E402
                                   activation_root, cohort_meta_dir, dfc_root,
                                   meta_dir, open_dataset, parse_hive_keys,
                                   read_shard)

__all__ = [
    "REPO", "ACTIVATION_META_COLUMNS", "DFC_QC_COLUMNS", "DFC_ENTITY_COLUMNS",
    "output_root", "cohort_configs", "inventory", "inventory_summary",
    "dataset", "parcel_columns", "edge_columns", "load_activation", "load_dfc",
    "atlas_labels", "edge_names", "subject_shards", "read_subject_edges",
    "full_matrix_from_upper", "read_edges", "read_shard", "fisher_z",
    "load_participants", "load_participants_qc", "load_phenotype",
    "subject_table", "estimate_gb", "mem_mb",
]

# Stage 2 non-parcel columns. Everything else in an activation file is a parcel,
# which is how `parcel_columns` can work without knowing the atlas.
ACTIVATION_META_COLUMNS = ["t", "time_s", "stimulus_time_s", "good_frame",
                           "run_idx", "ses", "run", "acq", "run_key"]

# Stage 3 columns that are neither QC nor edges: the filename-level entities.
DFC_ENTITY_COLUMNS = ["ses", "run", "acq", "run_key"]

_STAGE_ROOT = {"activation": activation_root, "dfc": dfc_root}


# ------------------------------------------------------------------ paths ---
def output_root(explicit: str | Path | None = None) -> Path:
    """Where the pipeline wrote. Env override, then the configs, then ./outputs.

    Raises rather than returning a path that is not there: every loader below
    would otherwise report "0 shards" for a typo, which reads exactly like a
    stage that never ran.
    """
    import os

    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    elif os.environ.get("FMRIDECOMP_OUTPUTS"):
        candidates.append(Path(os.environ["FMRIDECOMP_OUTPUTS"]))
    else:
        for cfg in sorted((REPO / "config").glob("*.yaml")):
            import yaml
            root = (yaml.safe_load(cfg.read_text()) or {}).get("output_root")
            if root:
                candidates.append(Path(root))
        candidates.append(REPO / "outputs")

    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "no output_root found. Tried:\n  " + "\n  ".join(str(c) for c in candidates)
        + "\nSet FMRIDECOMP_OUTPUTS, or pass root=... explicitly."
    )


def cohort_configs() -> pd.DataFrame:
    """The four cohort YAMLs, reduced to what a notebook needs to reason about.

    `tr` and the window grid are here because they are the two things that make
    a cross-cohort comparison of edge noise not a comparison of edge noise:
    window length in *samples* is `window_s / tr`, and the Fisher-z sampling SD
    is `1/sqrt(n-3)`.
    """
    import yaml

    rows = []
    for path in sorted((REPO / "config").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        if "cohort" not in cfg:
            continue
        win = cfg.get("windows", {}) or {}
        rows.append({
            "cohort": cfg["cohort"],
            "config": path.name,
            "tr": cfg.get("tr"),
            "atlases": ",".join(cfg.get("atlases", [])),
            "window_sizes_s": ",".join(str(w) for w in win.get("sizes_s", [])),
            "n_overlaps": win.get("n_overlaps"),
            "bandpass": str((cfg.get("filtering", {}) or {}).get("bandpass")),
            "fd_threshold": (cfg.get("confounds", {}) or {}).get("fd_threshold"),
            "participants": cfg.get("participants"),
            "output_root": cfg.get("output_root"),
        })
    return pd.DataFrame(rows).sort_values("cohort", ignore_index=True)


# -------------------------------------------------------------- inventory ---
def inventory(stage: str = "dfc", root: str | Path | None = None) -> pd.DataFrame:
    """One row per leaf on disk, from the directory names only -- no file opens.

    This is the "never assume file counts" tool: it is the ground truth about
    which cohorts, atlases and window sizes actually exist, as opposed to which
    ones a config asked for. A glob at fixed depth does not stat anything, so
    this stays cheap even at ~13k leaves.

    Also reports abandoned `.tmp.<pid>` shards, which mean a writer was killed
    mid-write. They are invisible to `pyarrow.dataset` (wrong extension) and so
    would otherwise never show up as anything but a missing subject.
    """
    root = output_root(root)
    keys = PARTITION_KEYS[stage]
    pattern = "/".join(f"{k}=*" for k in keys)
    leaves = sorted((root / stage).glob(f"{pattern}/data.parquet"))

    rows = []
    for p in leaves:
        row = parse_hive_keys(p)
        row["path"] = str(p)
        rows.append(row)
    df = pd.DataFrame(rows, columns=keys + ["path"])

    stale = sorted((root / stage).glob(f"{pattern}/*.tmp.*"))
    if stale:
        print(f"WARNING: {len(stale)} abandoned .tmp shard(s) under {root/stage} "
              f"-- a writer was killed mid-write. e.g. {stale[0]}")
    if df.empty:
        print(f"WARNING: no leaves under {root/stage} matching {pattern}/data.parquet")
    return df


def inventory_summary(inv: pd.DataFrame, stage: str = "dfc") -> pd.DataFrame:
    """Shards and distinct subjects per (cohort, atlas[, window_s])."""
    if inv.empty:
        return inv
    by = ["cohort", "atlas"] + (["window_s"] if stage == "dfc" else [])
    out = (inv.groupby(by)
              .agg(n_shards=("path", "size"),
                   n_subs=("sub", "nunique"),
                   n_tasks=("task", "nunique"))
              .reset_index())
    if "window_s" in out:
        out["window_s"] = out["window_s"].astype(float)
    return out.sort_values(by, ignore_index=True)


# ---------------------------------------------------------------- reading ---
def dataset(stage: str, atlas: str, root: str | Path | None = None):
    """A dataset rooted at one atlas, plus the keys that root swallowed.

    Returns `(dataset, swallowed)` where `swallowed` maps key -> value for every
    partition key at or above the root -- always `atlas`, since the root IS the
    `atlas=` directory. Those columns come back null from pyarrow and must be
    backfilled by the caller; `load_activation` and `load_dfc` do it for you.
    """
    root = output_root(root)
    ds_root = _STAGE_ROOT[stage](root, atlas)
    if not ds_root.is_dir():
        raise FileNotFoundError(
            f"{ds_root} does not exist. Populated atlases for stage {stage!r}: "
            f"{sorted(inventory(stage, root)['atlas'].unique()) if (root/stage).is_dir() else '(stage absent)'}"
        )
    return open_dataset(ds_root, stage=stage), {"atlas": atlas}


def _filter(**kw):
    """AND of equality / isin predicates on partition keys, skipping Nones."""
    expr = None
    for key, val in kw.items():
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            clause = pads.field(key).isin([str(v) for v in val])
        else:
            clause = pads.field(key) == str(val)
        expr = clause if expr is None else (expr & clause)
    return expr


def parcel_columns(ds) -> list[str]:
    """Stage 2 parcel columns: everything that is not metadata or a key."""
    known = set(ACTIVATION_META_COLUMNS) | set(PARTITION_KEYS["activation"])
    return [n for n in ds.schema.names if n not in known]


def edge_columns(ds) -> list[str]:
    """Stage 3 edge columns, or `['edges']` when they are packed.

    Below ~20,000 edges they are one column per edge named `NodeA__NodeB`;
    above it a single `fixed_size_list<float32>`. Never assume which.
    """
    if "edges" in ds.schema.names:
        return ["edges"]
    return [n for n in ds.schema.names if "__" in n]


def _gb(x: float) -> str:
    """Bytes are easier to act on than three leading zeros."""
    return f"{x * 1000:.1f} MB" if x < 1 else f"{x:.2f} GB"


def _to_pandas(ds, columns, filt, swallowed, guard_gb, force):
    n_frag = None
    if guard_gb is not None:
        est, n_frag = estimate_gb(ds, columns=columns, filter=filt)
        if est > guard_gb and not force:
            raise MemoryError(
                f"this read is estimated at {_gb(est)} uncompressed across "
                f"{n_frag} shard(s), over the {_gb(guard_gb)} guard.\n"
                f"  Narrow it (cohort=, task=, sub=, a coarser atlas, one "
                f"window_s), drop the edge/parcel columns, or pass "
                f"guard_gb=None to override deliberately."
            )
    df = ds.to_table(columns=columns, filter=filt).to_pandas()
    for key, val in swallowed.items():
        # Null because the key sits at or above the dataset root. Filtering on
        # it would have matched nothing; putting it back makes the frame
        # self-describing and safe to concat with a read from another atlas.
        df[key] = val
    return df


def load_activation(atlas: str, cohort=None, task=None, sub=None, *,
                    with_parcels: bool = False, columns: list[str] | None = None,
                    root=None, guard_gb: float | None = 4.0,
                    force: bool = False) -> pd.DataFrame:
    """Stage 2 rows, pruned by partition key and projected to columns.

    `with_parcels=False` (the default) reads the nine metadata columns and no
    parcel data at all -- that is the first-look read, and it stays small on any
    atlas. Pass `columns=[...]` to name parcels explicitly; passing
    `with_parcels=True` takes all of them.
    """
    ds, swallowed = dataset("activation", atlas, root)
    if columns is None:
        columns = list(ACTIVATION_META_COLUMNS)
        columns += [k for k in PARTITION_KEYS["activation"] if k != "atlas"]
        if with_parcels:
            columns += parcel_columns(ds)
    filt = _filter(cohort=cohort, task=task, sub=sub)
    return _to_pandas(ds, columns, filt, swallowed, guard_gb, force)


def load_dfc(atlas: str, window_s=None, cohort=None, task=None, sub=None, *,
             with_edges: bool = False, columns: list[str] | None = None,
             root=None, guard_gb: float | None = 4.0,
             force: bool = False) -> pd.DataFrame:
    """Stage 3 rows. QC columns by default; edges only when asked for.

    The QC-only read is what you want first: twelve columns for any atlas, a few
    MB for a whole cohort, and it carries `n_tr_effective`, `frac_good_frames`
    and `rank_deficient` -- everything needed to decide which windows are worth
    loading edges for.
    """
    ds, swallowed = dataset("dfc", atlas, root)
    if columns is None:
        columns = list(DFC_QC_COLUMNS) + list(DFC_ENTITY_COLUMNS)
        columns += [k for k in PARTITION_KEYS["dfc"] if k != "atlas"]
        if with_edges:
            columns += edge_columns(ds)
    win = None if window_s is None else (
        [_fmt_window(float(w)) for w in window_s]
        if isinstance(window_s, (list, tuple, set)) else _fmt_window(float(window_s)))
    filt = _filter(window_s=win, cohort=cohort, task=task, sub=sub)
    return _to_pandas(ds, columns, filt, swallowed, guard_gb, force)


def estimate_gb(ds, columns: list[str] | None = None, filter=None,
                sample: int = 20) -> tuple[float, int]:
    """Uncompressed size of a projected read, from parquet footers.

    Footers carry per-column-chunk `total_uncompressed_size`, so the projection
    is priced properly: asking for 12 QC columns of a 6,105-edge file costs
    almost nothing, and the estimate says so. At most `sample` footers are read
    and the total is scaled by the fragment count, because on a parallel
    filesystem a few thousand small footer reads is itself the slow part.

    Returns `(gb, n_fragments)`. It prices what is on disk; pandas will hold
    somewhat more (object dtype for the string keys, and a copy during
    conversion), so treat it as a lower bound.
    """
    frags = list(ds.get_fragments(filter=filter) if filter is not None
                 else ds.get_fragments())
    if not frags:
        return 0.0, 0
    want = None if columns is None else set(columns)
    total, seen = 0, 0
    for frag in frags[:sample]:
        md = pq.ParquetFile(frag.path).metadata
        for rg in range(md.num_row_groups):
            group = md.row_group(rg)
            for col in range(group.num_columns):
                chunk = group.column(col)
                # `path_in_schema`, not `schema.names`: a packed
                # fixed_size_list is stored as the leaf `edges.list.element`,
                # so matching on the leaf name would price the edges at zero --
                # exactly the column the estimate exists to catch.
                top = chunk.path_in_schema.split(".", 1)[0]
                if want is None or top in want:
                    total += chunk.total_uncompressed_size
        seen += 1
    return (total / seen) * len(frags) / 1e9, len(frags)


def mem_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / 1e6


# ------------------------------------------------------------------ atlas ---
def atlas_labels(atlas: str, root=None) -> pd.DataFrame:
    """The label table the pipeline wrote next to the data.

    Read this rather than `get_atlas(...)`: the label table is what defines node
    order, it is what makes the packed `edges` column interpretable, and unlike
    the registry it needs neither nilearn nor network access on a compute node.
    """
    path = meta_dir(output_root(root)) / f"atlas-{atlas}_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- stage 2 writes it; did it run?")
    return pd.read_csv(path)


def edge_names(atlas: str, root=None) -> list[str]:
    """Upper-triangle (row-major, k=1) edge names, in the stored order.

    Rebuilt from the label table's `column` field, which is exactly what
    `AtlasSpec.edge_names()` builds them from -- so this is the same order the
    packed `edges` column uses, without importing the atlas registry.
    """
    cols = atlas_labels(atlas, root)["column"].tolist()
    iu, ju = np.triu_indices(len(cols), k=1)
    return [f"{cols[i]}__{cols[j]}" for i, j in zip(iu, ju)]


def subject_shards(atlas: str, window_s=None, cohort=None, task=None, sub=None,
                   stage: str = "dfc", root=None) -> pd.DataFrame:
    """Inventory rows narrowed to a selection -- the paths for a per-shard read."""
    inv = inventory(stage, root)
    if inv.empty:
        return inv
    sel = inv["atlas"] == atlas
    for key, val in (("window_s", None if window_s is None else _fmt_window(float(window_s))),
                     ("cohort", cohort), ("task", task), ("sub", sub)):
        if val is not None and key in inv.columns:
            sel &= inv[key].isin([str(v) for v in val]) if isinstance(val, (list, tuple, set)) \
                else inv[key] == str(val)
    return inv[sel].reset_index(drop=True)


def read_subject_edges(path: str | Path, atlas: str | None = None,
                       root=None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """One DFC leaf as `(qc, edges, names)`, whichever way the edges are stored.

    Per shard is how you touch edges on a fine atlas without the packed-list
    memory cliff: one subject at a time, reduce, discard. `read_edges` handles
    both storage modes; the names come from the label table.
    """
    path = Path(path)
    qc = pq.read_table(path, columns=[c for c in DFC_QC_COLUMNS]).to_pandas()
    for key, val in parse_hive_keys(path).items():
        qc[key] = val
    edges = read_edges(path)
    names = edge_names(atlas or parse_hive_keys(path)["atlas"], root)
    if edges.shape[1] != len(names):
        raise ValueError(f"{path}: {edges.shape[1]} edges but the label table "
                         f"implies {len(names)} -- wrong atlas?")
    return qc, edges, names


def fisher_z(r: np.ndarray | pd.Series) -> np.ndarray:
    """arctanh, with the |r| = 1 endpoints kept finite.

    Edges are stored as raw *r* (`fisher_z_applied: false` in the schema
    metadata), so this is never already done for you.
    """
    r = np.asarray(r, dtype=np.float64)
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


# ----------------------------------------------------------- participants ---
def load_participants(cohort: str) -> pd.DataFrame:
    """`config/*_participants.csv` -- human-owned curation, one row per (sub, task).

    Carries `excluded` / `exclusion_reason` and nothing else about the person:
    no age, no sex, no clinical score. That is by design -- see
    `config/phenotype/README.md` for where the phenotype comes from instead.
    """
    matches = [p for p in sorted((REPO / "config").glob("*_participants.csv"))
               if cohort in _peek_cohort(p)]
    if not matches:
        raise FileNotFoundError(f"no participants CSV in config/ for cohort={cohort!r}")
    df = pd.read_csv(matches[0], dtype={"sub": str})
    df["excluded"] = df["excluded"].astype(str).str.lower().isin(("true", "1", "yes"))
    return df


def _peek_cohort(path: Path) -> set[str]:
    df = pd.read_csv(path, usecols=["cohort"], dtype=str)
    return set(df["cohort"].dropna().unique())


def load_participants_qc(cohort: str, root=None) -> pd.DataFrame:
    """`meta/cohorts/cohort=<c>/participants_qc.csv` -- machine-owned measurement.

    Measurements only: no threshold has been applied to any column here, and
    none should be applied inside this file. Deciding that `mean_fd > 0.5` is an
    exclusion belongs to the analysis, so that a sensitivity analysis can move
    the cutoff without re-running the pipeline.
    """
    path = cohort_meta_dir(output_root(root), cohort) / "participants_qc.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- written by `fmri-decomp diagnose` "
            f"(03_finalize.sbatch). Run it for cohort={cohort!r}.")
    return pd.read_csv(path, dtype={"sub": str})


def load_phenotype(cohort: str, quiet: bool = False) -> pd.DataFrame | None:
    """`config/phenotype/<cohort>_phenotype.csv`, or None with a loud reason.

    Age, sex and clinical scores are NOT in any file this pipeline writes and
    not in `*_participants.csv` either. They are built from each dataset's own
    source table by `tools/make_phenotype.py`. Returning None rather than
    raising is deliberate: the notebooks are useful before the phenotype
    exists, and silently returning an empty frame would let a groupby on `sex`
    quietly produce nothing.
    """
    path = REPO / "config" / "phenotype" / f"{cohort}_phenotype.csv"
    if not path.exists():
        if not quiet:
            print(f"no phenotype for cohort={cohort!r} at {path.relative_to(REPO)}\n"
                  f"  Build it: python tools/make_phenotype.py --help "
                  f"(sources listed in config/phenotype/README.md)")
        return None
    return pd.read_csv(path, dtype={"sub": str})


def subject_table(cohort: str, root=None, with_qc: bool = True) -> pd.DataFrame:
    """participants + participants_qc + phenotype, joined on (sub, task) / sub.

    Columns that came from the phenotype are prefixed `pheno_` where they would
    collide, so it is always visible which file a value came from. `has_pheno`
    marks the rows the join actually found a person for -- a partial phenotype
    is normal (Cam-CAN's frailty tables do not cover every CC700 subject) and
    should be visible rather than inferred from NaNs.
    """
    out = load_participants(cohort)
    if with_qc:
        try:
            qc = load_participants_qc(cohort, root)
            drop = [c for c in ("participant_id", "cohort") if c in qc.columns]
            out = out.merge(qc.drop(columns=drop), on=["sub", "task"],
                            how="left", suffixes=("", "_qc"))
        except FileNotFoundError as exc:
            print(f"QC not joined: {exc}")
    ph = load_phenotype(cohort)
    if ph is not None:
        ph = ph.drop(columns=[c for c in ("cohort",) if c in ph.columns])
        ph = ph.rename(columns={c: f"pheno_{c}" for c in ph.columns
                                if c != "sub" and c in out.columns})
        out = out.merge(ph.assign(has_pheno=True), on="sub", how="left")
        out["has_pheno"] = out["has_pheno"].fillna(False)
    else:
        out["has_pheno"] = False
    return out
