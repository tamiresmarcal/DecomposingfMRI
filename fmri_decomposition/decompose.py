#!/usr/bin/env python3
"""Stage 4 -- fit a latent decomposition on some cohorts, project it onto others.

    fmri-decomp decompose --atlas harvardoxford --window-s 30 60 120 300

WHICH COHORT IS FIT AND WHICH IS PROJECTED
------------------------------------------
`--train` and `--project`, and nothing else. No cohort YAML owns this stage:
a decomposition spans cohorts by definition, so the split cannot live in a
per-cohort config the way `tr` or `atlases` does. Defaults are
`--train ds002837 cneuromod --project camcan`, which is the design the camcan
symptom analysis needs -- camcan is never seen by any `fit`, only by
`transform`.

The split is recorded in three places so a file can never be orphaned from it:
`role` and `model_hash` are COLUMNS in every latents file, the full fit
description is in that file's parquet schema metadata, and the fitted objects
plus a manifest sit under `meta/models/`.

`model_hash` is a short digest of everything that changes the fit -- atlas,
window, train cohorts, n_latents, bins, seed, umap rows, package version and
the edge list. Two files with the same hash came from the same fit; two with
different hashes are not comparable, whatever the filenames say. It is the
same idea as `config_hash` on the stage 2 and 3 shards.

writes, per window size,

    outputs/latents/atlas=<a>/window_s=<w>/cohort=<c>/data.parquet
    outputs/meta/models/decompose_atlas-<a>_window-<w>.joblib

This is `notebooks/05_decompose.ipynb` as a batch job. The notebook holds every
cohort's edges in memory at once, which is fine at yeo7's 21 edges and not fine
at Harvard-Oxford's 6,105: 124k training windows x 6,105 edges is 3.0 GB in
float32 and 6.1 GB in float64, and the notebook made several copies of it.

WHAT THIS DOES DIFFERENTLY, AND WHY IT FITS WHERE THE NOTEBOOK DID NOT
----------------------------------------------------------------------
* float32 end to end. The edges are written as float32 and nothing here needs
  more; it halves every array. sklearn preserves the dtype.
* The training matrix is built once and scaled IN PLACE (`StandardScaler(
  copy=False)`), instead of being converted to an array twice and copied again
  by `transform`. That alone was three simultaneous copies in the notebook.
* Two passes. Fitting needs the training cohorts in memory together; writing
  does not, so each cohort is re-read, transformed, written and freed one at a
  time. Re-reading costs IO and saves a whole cohort's worth of peak RSS.
* Each window size is an independent fit, so `--window-s` loops rather than
  pooling -- and a SLURM array can put one window size per task.

Peak memory is roughly `n_train_rows x n_edges x 4 bytes x 1.6`. Print it
before committing an allocation:

    python tools/decompose.py --atlas harvardoxford --window-s 30 --dry-run

WHAT IS STILL MISSING
---------------------
No tests, and the window grid comes from the command line rather than from
`windows.sizes_s`. Both are worth closing; neither is a reason to keep this
outside the package, since it is the stage that turns edges into the latents
every later analysis reads.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Identity and QC carried into the latents. Everything else in a dfc file is
# either an edge (dropped -- it is already in dfc/) or a partition key.
IDENT = ["cohort", "role", "model_hash", "task", "sub", "window_id", "start_s",
         "stimulus_start_s", "n_tr_effective", "frac_good_frames",
         "rank_deficient", "crosses_run_boundary"]

# Kept out of read_cohort's projection: they are added after the fit, not read
# from the dfc shard.
_DERIVED = ("cohort", "task", "sub", "role", "model_hash")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shard_paths(root: Path, atlas: str, window_s, cohort: str) -> list[Path]:
    return sorted(root.glob(f"dfc/atlas={atlas}/window_s={window_s}/"
                            f"cohort={cohort}/task=*/sub=*/data.parquet"))


def edge_columns(path: Path) -> list[str]:
    """Edge names from one shard's schema, without reading a row group."""
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    edges = [n for n in names if "__" in n]
    if not edges:
        raise SystemExit(
            f"{path} has no NodeA__NodeB columns. This atlas packs its edges "
            f"into a single `edges` list column, which this script does not "
            f"unpack -- all three configured atlases are below the packing "
            f"threshold, so check --atlas.")
    return edges


def read_cohort(root: Path, atlas: str, window_s, cohort: str, edges: list[str],
                want_edges: bool = True):
    """One cohort -> (identity frame, float32 edge matrix or None).

    Reads the identity columns and the edge columns in one pass but keeps them
    apart, so the edge block is a contiguous float32 array and never a
    DataFrame with 6,105 columns of object-dtype partition keys beside it.
    """
    paths = shard_paths(root, atlas, window_s, cohort)
    if not paths:
        raise SystemExit(f"no dfc shards for cohort={cohort} at "
                         f"atlas={atlas} window_s={window_s}")
    ident_cols = [c for c in IDENT if c not in _DERIVED]
    idents, blocks = [], []
    for p in paths:
        keys = dict(s.split("=", 1) for s in p.parts if "=" in s)
        cols = ident_cols + (edges if want_edges else [])
        df = pd.read_parquet(p, columns=cols)
        ident = df[ident_cols].copy()
        for k in ("cohort", "task", "sub"):
            ident[k] = keys[k]
        idents.append(ident)
        if want_edges:
            blocks.append(df[edges].to_numpy(dtype=np.float32))
        del df
    ident = pd.concat(idents, ignore_index=True)
    X = np.vstack(blocks) if want_edges else None
    del idents, blocks
    gc.collect()
    return ident, X


def drop_nan_rows(ident: pd.DataFrame, X: np.ndarray):
    """A window with any NaN edge cannot be scaled or decomposed.

    Windows flagged rank_deficient are KEPT: the correlation matrix is
    singular, but each edge in it is an ordinary two-variable correlation, and
    dropping them would delete whole window sizes for the coarse-TR cohort.
    """
    keep = ~np.isnan(X).any(axis=1)
    if keep.all():
        return ident, X
    return ident[keep].reset_index(drop=True), X[keep]


def model_hash(meta: dict, edges: list[str]) -> str:
    """Short digest of everything that changes the fit.

    Same role as `config_hash` on the stage 2 and 3 shards: two latents files
    with the same hash came from one fit and are comparable; two with different
    hashes are not, however alike the paths look. The edge list is included
    because a different atlas revision with the same name is a different model.
    """
    from . import __version__

    payload = json.dumps({**meta, "package_version": __version__,
                          "n_edges": len(edges),
                          "edges_digest": hashlib.sha256(
                              "\n".join(edges).encode()).hexdigest()[:16]},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def fit_meta(args, window_s, edges: list[str]) -> dict:
    """The fit description carried into every output of this run."""
    return {"stage": "latents", "atlas": args.atlas, "window_s": str(window_s),
            "train_cohorts": list(args.train), "project_cohorts": list(args.project),
            "n_latents": list(args.n_latents), "bins": list(args.bins),
            "umap_fit_rows": int(args.umap_fit_rows), "no_umap": bool(args.no_umap),
            "seed": int(args.seed)}


def fit_models(X_train: np.ndarray, edges: list[str], args, meta: dict) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

    # copy=False scales the training matrix in place -- there is no second
    # 3 GB array, and X_train is not needed in raw units again.
    scaler = StandardScaler(copy=False).fit(X_train)
    Z = scaler.transform(X_train)
    log(f"scaled in place: {Z.shape} {Z.dtype} "
        f"({Z.nbytes / 1e9:.2f} GB)")

    # `meta` first, working containers second: meta carries a `bins` LIST (the
    # bin counts asked for) and this dict needs a `bins` DICT (the fitted
    # discretizers). Spread the other way round and the list wins, and the
    # first assignment into it fails with an IndexError.
    models = {**meta, "scaler": scaler, "pca": {}, "umap": {}, "bins": {},
              "edges": edges}

    for n in args.n_latents:
        models["pca"][n] = PCA(n_components=n, random_state=args.seed,
                               svd_solver="randomized").fit(Z)
        evr = models["pca"][n].explained_variance_ratio_
        log(f"pca {n}: cumulative explained {evr.sum():.3f}  {evr.round(3)}")

    if not args.no_umap:
        try:
            import umap
        except ImportError:
            log("umap-learn not importable -- skipping UMAP, PCA still runs")
        else:
            rng = np.random.default_rng(args.seed)
            idx = (rng.choice(len(Z), args.umap_fit_rows, replace=False)
                   if args.umap_fit_rows and len(Z) > args.umap_fit_rows
                   else np.arange(len(Z)))
            log(f"fitting UMAP on {len(idx):,} of {len(Z):,} rows")
            for n in args.n_latents:
                t0 = time.time()
                models["umap"][n] = umap.UMAP(n_components=n,
                                              random_state=args.seed).fit(Z[idx])
                log(f"  umap {n}: {time.time() - t0:.0f}s")

    if 3 not in args.n_latents:
        raise SystemExit("the threshold grid is defined on pca3; "
                         "include 3 in --n-latents")
    P3 = models["pca"][3].transform(Z)
    for n in args.bins:
        kbd = KBinsDiscretizer(n_bins=n, encode="ordinal", strategy="quantile",
                               subsample=None).fit(P3)
        models["bins"][n] = kbd
        lab = np.ravel_multi_index(kbd.transform(P3).astype(int).T, (n, n, n))
        log(f"{n} bins -> {n ** 3} cells, {len(np.unique(lab))} occupied")
    del P3, Z
    gc.collect()
    return models


def latents_for(ident: pd.DataFrame, X: np.ndarray, models: dict,
                role: str) -> pd.DataFrame:
    """Identity + every latent + every cluster label, for one cohort.

    `role` and `model_hash` are COLUMNS, not just schema metadata, because
    pandas drops parquet key-value metadata on read -- a provenance field only
    a pyarrow user can see is one nobody checks.
    """
    Z = models["scaler"].transform(X)
    out = ident.copy()
    out["role"] = role
    out["model_hash"] = models["model_hash"]
    for kind in ("pca", "umap"):
        for n, model in models[kind].items():
            arr = model.transform(Z)
            for j in range(n):
                out[f"{kind}{j}/{n}"] = arr[:, j].astype(np.float32)
    P3 = models["pca"][3].transform(Z)
    for n, kbd in models["bins"].items():
        out[f"ThresholdCluster_pca3_{n ** 3}"] = np.ravel_multi_index(
            kbd.transform(P3).astype(int).T, (n, n, n))
    return out[[c for c in IDENT if c in out.columns]
               + [c for c in out.columns if c not in IDENT]]


def write_latents(lat: pd.DataFrame, path: Path, models: dict, cohort: str) -> None:
    """Write then rename, with the fit description in the schema metadata.

    Mirrors what stage 2 and 3 do to every shard, so a latents file opened on
    its own is still self-identifying.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from . import __version__

    table = pa.Table.from_pandas(lat, preserve_index=False)
    meta = {k.encode(): json.dumps(v, default=str).encode()
            for k, v in {**models["fit_meta"], "cohort": cohort,
                         "role": models["role_of"][cohort],
                         "model_hash": models["model_hash"],
                         "n_train_rows": models["n_train_rows"],
                         "package_version": __version__,
                         "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime())}.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(table.replace_schema_metadata(meta), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def dump_models(models: dict, path_stem: Path) -> Path:
    try:
        import joblib
        path = path_stem.with_suffix(".joblib")
        joblib.dump(models, path)
    except ImportError:
        import pickle
        path = path_stem.with_suffix(".pkl")
        path.write_bytes(pickle.dumps(models))
    return path


def run_one(root: Path, window_s, args) -> None:
    atlas = args.atlas
    out_dir = root / "latents" / f"atlas={atlas}" / f"window_s={window_s}"
    models_dir = root / "meta" / "models"
    stem = models_dir / f"decompose_atlas-{atlas}_window-{window_s}"

    cohorts = list(dict.fromkeys(args.train + args.project))
    if not args.overwrite and all(
            (out_dir / f"cohort={c}" / "data.parquet").exists() for c in cohorts):
        log(f"window_s={window_s}: latents already exist for every cohort, "
            f"skipping (--overwrite to redo)")
        return

    first = shard_paths(root, atlas, window_s, args.train[0])
    if not first:
        raise SystemExit(f"no shards for the first training cohort "
                         f"{args.train[0]!r} at atlas={atlas} window_s={window_s}")
    edges = edge_columns(first[0])

    counts = {c: len(shard_paths(root, atlas, window_s, c)) for c in cohorts}
    log(f"window_s={window_s}  {len(edges)} edges  shards: {counts}")

    log("loading training cohorts")
    idents, blocks = [], []
    for cohort in args.train:
        ident, X = read_cohort(root, atlas, window_s, cohort, edges)
        ident, X = drop_nan_rows(ident, X)
        log(f"  {cohort}: {len(X):,} windows, {ident['sub'].nunique()} subs, "
            f"{X.nbytes / 1e9:.2f} GB")
        idents.append(ident)
        blocks.append(X)
    X_train = np.vstack(blocks) if len(blocks) > 1 else blocks[0]
    n_train = len(X_train)
    del idents, blocks
    gc.collect()
    log(f"training matrix {X_train.shape} = {X_train.nbytes / 1e9:.2f} GB")

    meta = fit_meta(args, window_s, edges)
    mhash = model_hash(meta, edges)
    role_of = {c: ("train" if c in args.train else "projected") for c in cohorts}
    log(f"model_hash {mhash}  roles {role_of}")

    models = fit_models(X_train, edges, args, meta={
        **meta, "n_train_rows": int(n_train), "model_hash": mhash,
        "fit_meta": meta, "role_of": role_of})
    del X_train
    gc.collect()

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = dump_models(models, stem)
    log(f"models -> {model_path.relative_to(root)}")

    # A manifest beside the models, in the same spirit as io.write_manifest:
    # readable without unpickling anything, and the place to look when two
    # latents files disagree.
    from . import __version__
    (stem.parent / f"{stem.name}_manifest.json").write_text(json.dumps({
        **meta, "model_hash": mhash, "n_train_rows": int(n_train),
        "n_edges": len(edges), "role_of": role_of,
        "shards": counts, "models_file": model_path.name,
        "package_version": __version__,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

    # Second pass: one cohort in memory at a time.
    for cohort in cohorts:
        ident, X = read_cohort(root, atlas, window_s, cohort, edges)
        ident, X = drop_nan_rows(ident, X)
        lat = latents_for(ident, X, models, role_of[cohort])
        del ident, X
        gc.collect()
        path = out_dir / f"cohort={cohort}" / "data.parquet"
        write_latents(lat, path, models, cohort)
        log(f"  {cohort:<14} {role_of[cohort]:<10} {len(lat):>9,} rows "
            f"x {lat.shape[1]} cols "
            f"-> {path.relative_to(root)}")
        del lat
        gc.collect()


def dry_run(root: Path, args) -> None:
    print(f"{'window':>7}  {'edges':>6}  {'train rows':>11}  {'peak GB':>8}  shards")
    for w in args.window_s:
        paths = {c: shard_paths(root, args.atlas, w, c)
                 for c in dict.fromkeys(args.train + args.project)}
        first = next((p for c in args.train for p in paths[c]), None)
        if first is None:
            print(f"{w:>7}  (no shards)")
            continue
        n_edges = len(edge_columns(first))
        import pyarrow.parquet as pq
        rows = {c: sum(pq.ParquetFile(p).metadata.num_rows for p in ps)
                for c, ps in paths.items()}
        n_train = sum(rows[c] for c in args.train)
        peak = n_train * n_edges * 4 * 1.6 / 1e9
        print(f"{w:>7}  {n_edges:>6}  {n_train:>11,}  {peak:>8.2f}  "
              + "  ".join(f"{c}:{len(ps)}({rows[c]:,})" for c, ps in paths.items()))
    print("\npeak GB is the training matrix x1.6 for scaler and PCA workspace; "
          "ask for at least double.")


def add_arguments(p) -> None:
    """Shared by `fmri-decomp decompose` and `python -m ...decompose`."""
    p.add_argument("--atlas", required=True)
    p.add_argument("--window-s", nargs="+", required=True,
                   help="one fit per window size; they are independent")
    p.add_argument("--train", nargs="+", default=["ds002837", "cneuromod"])
    p.add_argument("--project", nargs="+", default=["camcan"])
    p.add_argument("--n-latents", nargs="+", type=int, default=[2, 3, 5],
                   help="must include 3: the threshold grid is defined on pca3")
    p.add_argument("--bins", nargs="+", type=int, default=[2, 3, 5, 8],
                   help="quantile bins per PCA axis -> n**3 cells")
    p.add_argument("--umap-fit-rows", type=int, default=30_000,
                   help="0 fits UMAP on every training row")
    p.add_argument("--no-umap", action="store_true")
    p.add_argument("--output-root",
                   help="default: output_root from config/camcan_movie.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print rows, edges and the memory estimate, fit nothing")


def run(args) -> int:
    if args.output_root:
        root = Path(args.output_root)
    elif os.environ.get("FMRIDECOMP_OUTPUTS"):
        root = Path(os.environ["FMRIDECOMP_OUTPUTS"])
    else:
        import yaml
        repo = Path(__file__).resolve().parent.parent
        root = Path(yaml.safe_load(
            (repo / "config" / "camcan_movie.yaml").read_text())["output_root"])
    if not root.is_dir():
        raise SystemExit(f"output_root does not exist: {root}")
    log(f"output_root {root}")

    if args.dry_run:
        dry_run(root, args)
        return 0

    for w in args.window_s:
        run_one(root, w, args)
    log("done")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
