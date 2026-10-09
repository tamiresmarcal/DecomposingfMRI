#!/usr/bin/env python3
"""Stage 4 -- fit a latent decomposition on some cohorts, project it onto others.

    fmri-decomp decompose --atlas harvardoxford --window-s 30 60 120 300

THIS STAGE PRODUCES EMBEDDINGS. IT DOES NOT DEFINE STATES
---------------------------------------------------------
PCA and UMAP are fitted here; every way of turning those coordinates into a
discrete state label lives in `cluster` (stage 4b), threshold included.

The quantile thresholding used to be here, and having it here while every other
clusterer was in `cluster` was the wrong seam in two concrete ways. One method
got re-fitted on every stage 4 re-run while the others did not, so the cheapest
state definition was the most expensive to change. And it wrote
`ThresholdCluster_pca3_*` columns with no entry in the schema's `clusterers`
provenance, because that block is written by stage 4b -- which is exactly why
`transitions.n_states_for` needs a fallback that reads K out of a column NAME.
One owner, one provenance record, one place to add the next method.

`--bins` is therefore gone; `cluster --methods threshold --k 8 27` is where it
went, and it reproduces the same columns (K = bins**3, quantile edges fitted on
the training rows).

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
window, train cohorts, n_latents, seed, umap rows, package version and
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
The window grid comes from the command line rather than from `windows.sizes_s`.
Worth closing; not a reason to keep this outside the package, since it is the
stage that turns features into the latents every later analysis reads.
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
from .io import PASSTHROUGH_MAX_FEATURES, RAW_PREFIX, latents_root

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


# --------------------------------------------------------- feature sources ---
# A source answers three questions and nothing else: where its shards are, what
# its feature columns are called, and how to turn a cohort into (identity,
# matrix). Everything after that -- scaling, PCA, UMAP, the threshold grid,
# model_hash, the two-pass write -- is shared, because the aperture is the only
# thing that differs. Adding a third source is an entry here, not a new stage.
ACTIVATION_WINDOW = "-1"


def source_shard_paths(root: Path, atlas: str, window_s, cohort: str,
                       source: str) -> list[Path]:
    if source == "activation":
        from . import frames
        return frames.shard_paths(root, atlas, cohort)
    return shard_paths(root, atlas, window_s, cohort)


def source_feature_columns(path: Path, source: str) -> list[str]:
    if source == "activation":
        from . import frames
        return frames.feature_columns(path)
    return edge_columns(path)


def source_read_cohort(root: Path, atlas: str, window_s, cohort: str,
                       features: list[str], args, censor: dict | None):
    """-> (identity frame, float32 matrix, stride_s, indep_factor).

    `stride_s` and `indep_factor` describe the time axis of the rows and are
    written to the latents for stage 5a: at this aperture the step between
    consecutive rows is the cohort's TR, which decompose cannot derive from
    `window_s` the way it can for a sliding window. They are None for dfc, where
    stage 5a's own `--n-overlaps` already gives it both.
    """
    if source_of(args) == "activation":
        from . import frames
        ident, X, tr = frames.read_cohort(
            root, atlas, cohort, features, censor=censor,
            band=args.match_bandpass, zscore=args.zscore_runs, log=log)
        # Frames do not overlap, so one row IS one independent sample.
        return ident, X, float(tr), 1
    ident, X = read_cohort(root, atlas, window_s, cohort, features,
                           censor=censor)
    return ident, X, None, None


def source_of(args) -> str:
    return getattr(args, "source", "dfc")


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


def load_censor(root: Path, policy: str | None, atlas: str, window_s, cohort: str):
    """The stage 3.5 decision for one cohort -> (kept subjects, kept windows).

    Returns None when no policy was named, which is how this stage stays
    runnable on a tree that predates `censor`. Everything else is a hard error:
    a policy that was asked for and not found must not silently become no
    censoring at all, because the run would look identical and be a different
    analysis.

    The window gate is optional even when the policy exists -- `censor` writes
    windows.parquet only for the atlas x aperture it was given `--stage dfc`
    for. A policy with subjects only is the normal first-pass state, and gates
    whole subjects rather than individual windows.
    """
    if not policy:
        return None

    base = root / "censor" / f"policy={policy}"
    subj_path = base / f"cohort={cohort}" / "subjects.parquet"
    if not subj_path.exists():
        raise SystemExit(
            f"--censor-policy {policy!r} but {subj_path} does not exist.\n"
            f"Run:  fmri-decomp censor --policy config/censor/{policy}.yaml "
            f"--cohorts {cohort}")

    subj = pd.read_parquet(subj_path, columns=["sub", "task", "keep"])
    keep_subs = set(zip(subj.loc[subj["keep"], "task"].astype(str),
                        subj.loc[subj["keep"], "sub"].astype(str)))

    win_path = (base / f"atlas={atlas}" / f"window_s={window_s}"
                / f"cohort={cohort}" / "windows.parquet")
    keep_windows = None
    if win_path.exists():
        win = pd.read_parquet(win_path, columns=["sub", "task", "window_id", "keep"])
        win = win.loc[win["keep"]]
        keep_windows = set(zip(win["task"].astype(str), win["sub"].astype(str),
                               win["window_id"].astype("int64")))
    return {"policy": policy, "subjects": keep_subs, "windows": keep_windows,
            "n_subject_rows": len(subj), "n_subject_kept": int(subj["keep"].sum())}


def read_cohort(root: Path, atlas: str, window_s, cohort: str, edges: list[str],
                want_edges: bool = True, censor: dict | None = None):
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
    n_dropped_subs = n_dropped_windows = 0
    for p in paths:
        keys = dict(s.split("=", 1) for s in p.parts if "=" in s)

        # A censored subject is skipped before the file is opened: the rows are
        # never read, never scaled and never counted toward peak RSS. That is
        # the whole reason this filter belongs here rather than after the
        # concat, where 6,105 float32 edge columns would already be resident.
        if censor is not None and (keys["task"], keys["sub"]) not in censor["subjects"]:
            n_dropped_subs += 1
            continue

        cols = ident_cols + (edges if want_edges else [])
        df = pd.read_parquet(p, columns=cols)

        if censor is not None and censor["windows"] is not None:
            wid = df["window_id"].astype("int64")
            mask = np.array([(keys["task"], keys["sub"], w) in censor["windows"]
                             for w in wid], dtype=bool)
            if not mask.all():
                n_dropped_windows += int((~mask).sum())
                df = df.loc[mask]
            if df.empty:
                del df
                continue

        ident = df[ident_cols].copy()
        for k in ("cohort", "task", "sub"):
            ident[k] = keys[k]
        idents.append(ident)
        if want_edges:
            blocks.append(df[edges].to_numpy(dtype=np.float32))
        del df

    if censor is not None and (n_dropped_subs or n_dropped_windows):
        log(f"  censor[{censor['policy']}] {cohort}: dropped {n_dropped_subs} "
            f"subject-shard(s), {n_dropped_windows:,} further window(s)")
    if not idents:
        raise SystemExit(
            f"censor policy {censor['policy']!r} kept nothing for cohort="
            f"{cohort} at atlas={atlas} window_s={window_s} -- check the "
            f"thresholds before running the fit")

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


def model_hash(meta: dict, features: list[str]) -> str:
    """Short digest of everything that changes the fit.

    Same role as `config_hash` on the stage 2 and 3 shards: two latents files
    with the same hash came from one fit and are comparable; two with different
    hashes are not, however alike the paths look. The feature list is included
    because a different atlas revision with the same name is a different model.

    The payload keys are still called `n_edges` and `edges_digest` under the
    activation source, where the features are parcels rather than edges. The
    names are wrong there and kept anyway: renaming them would change every dfc
    hash already on disk, and a digest's key names are not read by anything.
    `meta["source"]` is what distinguishes the two.
    """
    from . import __version__

    payload = json.dumps({**meta, "package_version": __version__,
                          "n_edges": len(features),
                          "edges_digest": hashlib.sha256(
                              "\n".join(features).encode()).hexdigest()[:16]},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _umap_latents(args) -> list[int]:
    """Which component counts UMAP is fitted at. Defaults to --n-latents.

    They were one list, and that became a trap once `--n-latents 14` existed to
    get a lossless rotation for the raw-dimension arm: UMAP was then fitted at
    14 components too, which costs most of the stage's walltime and is not an
    embedding anything asks for. UMAP's defaults are tuned for 2-3 components.
    """
    want = getattr(args, "umap_latents", None)
    return list(want) if want else list(args.n_latents)


def fit_meta(args, window_s, features: list[str]) -> dict:
    """The fit description carried into every output of this run.

    The `--source activation` keys are added ONLY for that source, and this is
    deliberate rather than tidy. Everything in here feeds `model_hash`, so
    writing `"source": "dfc"` into the default payload would change the hash of
    every dfc fit ever written -- turning latents, transitions and selection
    outputs that are perfectly valid into files whose hash no longer matches a
    re-run. Absent keys mean the default, which is the only way to extend a hash
    payload without invalidating what it has already stamped.
    """
    # `project_cohorts` is NOT here, deliberately, and this is the same rule
    # `stride_s` follows (see write_latents): a per-INVOCATION quantity does not
    # belong in a description of the fit. Only `--train` feeds the fit -- the
    # scaler, the PCA and the UMAP never see a projected cohort -- so two runs
    # that differ only in who else was projected produce the SAME model, and a
    # hash that said otherwise made them look incomparable. It is written
    # beside the hash instead, where it is still recoverable.
    #
    # This one removal DOES move every hash already on disk, which the
    # present-only-when-non-default rule above exists to avoid. It is accepted
    # once, because the alternative is permanent: while the project list is
    # hashed, adding a cohort to a cell always changes the hash, so no cohort
    # can ever be skipped and every addition rewrites and re-clusters the whole
    # cell.
    meta = {"stage": "latents", "atlas": args.atlas, "window_s": str(window_s),
            "train_cohorts": list(args.train),
            "n_latents": list(args.n_latents),
            "umap_fit_rows": int(args.umap_fit_rows), "no_umap": bool(args.no_umap),
            # In the fit description, therefore in model_hash: a PCA fit on
            # censored rows is not the same model as one fit on all of them,
            # and two latents files that differ only by policy must not be
            # able to claim the same hash.
            "censor_policy": args.censor_policy or None,
            "censor_policy_hash": censor_policy_hash(args),
            "seed": int(args.seed)}
    # Both of these follow the same rule as `source` above: present only when
    # they differ from the default, so adding them does NOT move the hash of any
    # fit already on disk. An absent key means the default.
    if getattr(args, "passthrough_features", False):
        meta["passthrough_features"] = True
    if _umap_latents(args) != list(args.n_latents):
        meta["umap_latents"] = _umap_latents(args)
    if getattr(args, "source", "dfc") != "dfc":
        meta.update({
            "source": args.source,
            # Both are transforms applied to the frames before the fit, so both
            # change the model -- see frames.py's docstring for why a per-TR
            # pattern needs them and a correlation does not.
            "match_bandpass": (list(args.match_bandpass)
                               if args.match_bandpass else None),
            "zscore_runs": bool(args.zscore_runs),
        })
    return meta


def _censor_policy(value: str):
    """`none` -> None, at PARSE time rather than in run().

    It has to be here and not in `run`, because `fit_meta` is reachable without
    going through `run` and the literal string "none" in its payload is a
    DIFFERENT model_hash from None -- so an uncensored fit would stop matching
    every uncensored latents file already on disk, for no reason but spelling.
    Normalising at the boundary makes `args.censor_policy is None` mean "no
    censoring" everywhere, with no caller able to see the other spelling.
    """
    return None if value.strip().lower() == "none" else value


def censor_policy_hash(args) -> str | None:
    """The hash `censor` stamped on its own output, read back from the summary.

    Taken from the written summary rather than re-hashing the YAML: the claim
    being recorded is "these rows were filtered by that run", and the YAML on
    disk may have been edited since.
    """
    if not args.censor_policy:
        return None
    root = Path(args.output_root) if args.output_root else None
    if root is None:
        return None
    summary = root / "meta" / "censor" / f"policy={args.censor_policy}.json"
    if not summary.exists():
        return None
    return json.loads(summary.read_text()).get("policy_hash")


def fit_models(X_train: np.ndarray, edges: list[str], args, meta: dict) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # copy=False scales the training matrix in place -- there is no second
    # 3 GB array, and X_train is not needed in raw units again.
    scaler = StandardScaler(copy=False).fit(X_train)
    Z = scaler.transform(X_train)
    log(f"scaled in place: {Z.shape} {Z.dtype} "
        f"({Z.nbytes / 1e9:.2f} GB)")

    # `meta` first, working containers second, so a key in both resolves to the
    # container rather than to the scalar `meta` recorded it as.
    models = {**meta, "scaler": scaler, "pca": {}, "umap": {}, "edges": edges}

    # Checked here, where `edges` is known, rather than in argparse, which
    # cannot see how wide a row is until the shards have been read.
    models["passthrough"] = bool(getattr(args, "passthrough_features", False))
    if models["passthrough"] and len(edges) > PASSTHROUGH_MAX_FEATURES:
        raise SystemExit(
            f"--passthrough-features on {len(edges):,} features, over the "
            f"{PASSTHROUGH_MAX_FEATURES} cap.\n"
            f"  It writes every input feature into the latents table as a "
            f"column, so this would add {len(edges):,} per row.\n"
            f"  It is meant for --source activation, where a row is 7, 14 or "
            f"111 parcels. Here a row is {len(edges):,} "
            f"{'edges' if getattr(args, 'source', 'dfc') == 'dfc' else 'features'}.\n"
            f"  Drop the flag, or use --source activation.")
    if models["passthrough"]:
        log(f"passthrough: {len(edges)} feature column(s) will be written as "
            f"{RAW_PREFIX}<name>, SCALED (post-StandardScaler) so that "
            f"raw{len(edges)} and pca{len(edges)} differ by a rotation only")

    for n in args.n_latents:
        if n > len(edges):
            # sklearn's own message names `min(n_samples, n_features)` without
            # saying which, and on the activation source "n_features" is the
            # atlas's parcel count -- the one thing the caller can act on.
            raise SystemExit(
                f"--n-latents {n} on an atlas with only {len(edges)} "
                f"feature(s).\n"
                f"  PCA cannot produce more components than the input has "
                f"columns. {len(edges)} components IS the whole space "
                f"(a lossless rotation), so {n} is asking for more "
                f"information than exists.\n"
                f"  Use --n-latents 3 {len(edges)} for this atlas.")
        models["pca"][n] = PCA(n_components=n, random_state=args.seed,
                               svd_solver="randomized").fit(Z)
        evr = models["pca"][n].explained_variance_ratio_
        log(f"pca {n}: cumulative explained {evr.sum():.3f}  {evr.round(3)}")

    models["umap_fitted"] = False
    if not args.no_umap:
        try:
            import umap
        except ImportError:
            # Used to warn and carry on. That was wrong: the latents file then
            # recorded `no_umap: false` -- the REQUEST -- while containing no
            # umap columns, and carried the same model_hash as a run that had
            # them. The notebook's only symptom was "umap0/3 not in the
            # latents", with nothing on disk saying why.
            raise SystemExit(
                "umap-learn is not importable, but --no-umap was not passed.\n"
                "Refusing to write latents that silently lack UMAP columns.\n"
                "  * PCA only:   add --no-umap\n"
                "  * with UMAP:  use a container that has umap-learn "
                "(containers/stage45.def pins 0.5.7)")
        else:
            models["umap_fitted"] = True
            rng = np.random.default_rng(args.seed)
            idx = (rng.choice(len(Z), args.umap_fit_rows, replace=False)
                   if args.umap_fit_rows and len(Z) > args.umap_fit_rows
                   else np.arange(len(Z)))
            log(f"fitting UMAP on {len(idx):,} of {len(Z):,} rows")
            for n in _umap_latents(args):
                t0 = time.time()
                models["umap"][n] = umap.UMAP(n_components=n,
                                              random_state=args.seed).fit(Z[idx])
                log(f"  umap {n}: {time.time() - t0:.0f}s")

    # Nothing here defines a STATE. This stage produces embeddings; `cluster`
    # turns them into labels -- see the module docstring on why the quantile
    # thresholding that used to live here moved there.
    if 3 not in args.n_latents:
        raise SystemExit("every clusterer in stage 4b is defined on a 3-D "
                         "embedding; include 3 in --n-latents")
    del Z
    gc.collect()
    return models


def latents_for(ident: pd.DataFrame, X: np.ndarray, models: dict,
                role: str) -> pd.DataFrame:
    """Identity + every embedding, for one cohort. No state labels.

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
    if models.get("passthrough"):
        # Z, not X: the SCALED features. PCA is fitted on Z, so writing Z here
        # makes `raw<N>` and `pca<N>` differ by exactly an orthonormal rotation
        # (measured round-trip error ~1e-14), which is what lets a
        # full-covariance HMM on one be the same model as on the other. Writing
        # X instead would leave the two arms differing by a scaling as well,
        # and the equivalence would no longer be exact.
        #
        # The scaler was fitted on the TRAIN cohorts and is applied to every
        # cohort, projected ones included -- one transform everywhere, which is
        # what makes a state mean comparable across cohorts at all.
        for j, name in enumerate(models["edges"]):
            out[f"{RAW_PREFIX}{name}"] = Z[:, j].astype(np.float32)
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
    # Written BESIDE fit_meta, not into it: `stride_s` is the cohort's TR under
    # the activation source, and a TR differs per cohort inside one fit. A
    # per-cohort quantity in the fit description would make model_hash differ
    # between two files that came from the same model -- exactly the thing the
    # hash exists to rule out. Stage 5a reads it from here.
    axis = {k: v for k, v in (("stride_s", models.get("stride_s")),
                              ("indep_factor", models.get("indep_factor")))
            if v is not None}
    meta = {k.encode(): json.dumps(v, default=str).encode()
            for k, v in {**models["fit_meta"], **axis, "cohort": cohort,
                         "role": models["role_of"][cohort],
                         "model_hash": models["model_hash"],
                         # What actually happened, beside what was asked for.
                         "umap_fitted": bool(models.get("umap_fitted")),
                         "n_umap_components": sorted(models["umap"]),
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


def latents_model_hash(path: Path) -> str | None:
    """The `model_hash` a latents file already carries, or None.

    Read from the schema metadata without touching a row group, so deciding
    whether a cohort needs rewriting costs one small read per cohort.
    """
    if not path.exists():
        return None
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = md.get(b"model_hash")
    if raw is None:
        return None
    # write_latents json.dumps() EVERY metadata value, so the hash is on disk
    # as `"abc123"` with the quotes, not as `abc123`. Comparing the raw bytes
    # therefore never matches and every cohort looks stale -- which silently
    # turns the per-cohort skip back into the cell-wide rewrite it replaced.
    # Decoded the same way transitions._meta_value does, falling back to the
    # bare string so a file written by some other path still reads.
    try:
        return json.loads(raw.decode())
    except ValueError:
        return raw.decode()


def latents_fit_description(path: Path) -> tuple[dict, list[str]]:
    """(the fit_meta a latents file was written with, its state columns).

    Both come from the schema metadata, so this costs one small read. The
    state columns are the `clusterers` block stage 4b writes -- what a rewrite
    of this file would destroy.
    """
    if not path.exists():
        return {}, []
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(path).schema_arrow
    md = schema.metadata or {}
    stored = {}
    for k, v in md.items():
        key = k.decode()
        if key in ("clusterers", "ARROW:schema", "pandas"):
            continue
        try:
            stored[key] = json.loads(v.decode())
        except ValueError:
            stored[key] = v.decode()
    cols = sorted(json.loads((md.get(b"clusterers") or b"{}").decode()))
    return stored, cols


# Written into a latents file BESIDE the fit description (see write_latents),
# so they are not part of it and must not appear in a diff of it. Without this
# the one useful line -- the flag that moved -- is buried under ten entries
# reading `-> '(absent)'` for things that were never in `fit_meta` to begin
# with. Keep in step with write_latents' metadata block.
_NOT_FIT_KEYS = frozenset({
    "cohort", "role", "model_hash", "umap_fitted", "n_umap_components",
    "n_train_rows", "package_version", "written_utc", "stride_s",
    "indep_factor",
})


def _fit_differences(stored: dict, meta: dict) -> list[str]:
    """Which keys of the fit description changed, as readable lines.

    The point is to name the FLAG that moved. A bare "the hash differs" sends
    you to diff two 16-character strings; "n_latents: [3, 7] -> [2, 3, 5]"
    says you forgot `--n-latents 3 7`.

    A key in `stored` but absent from `meta` is kept, not filtered: that is
    exactly the forgotten-flag case, because `fit_meta` writes a key only when
    it differs from its default. `passthrough_features: True -> (absent)` is
    the message someone needs most.
    """
    out = []
    for key in sorted((set(stored) | set(meta)) - _NOT_FIT_KEYS):
        was, now = stored.get(key, "(absent)"), meta.get(key, "(absent)")
        if was != now:
            out.append(f"      {key}: {was!r} -> {now!r}")
    # In the hash payload but not in fit_meta, so the loop above cannot see it.
    from . import __version__

    if stored.get("package_version") not in (None, __version__):
        out.append(f"      package_version: "
                   f"{stored['package_version']!r} -> {__version__!r}")
    return out


def refuse_to_destroy_states(out_dir: Path, stale: list[str], meta: dict,
                             window_s) -> None:
    """Stop before rewriting a cohort whose state columns would be lost.

    A rewrite here is not recoverable by re-running: `cluster` would have to
    refit, and HMM2 at K=27 measured hours per cell. The commonest cause is a
    FORGOTTEN FLAG -- omit `--passthrough-features` or `--n-latents 3 7` when
    adding a cohort and the model_hash moves, every existing cohort looks
    stale, and all of them are rewritten. So the refusal names the difference
    rather than the hash.
    """
    at_risk = {}
    for c in stale:
        stored, cols = latents_fit_description(
            out_dir / f"cohort={c}" / "data.parquet")
        if cols:
            at_risk[c] = (stored, cols)
    if not at_risk:
        return
    first = next(iter(at_risk.values()))[0]
    diff = _fit_differences(first, meta)
    lines = [
        f"window_s={window_s}: refusing to rewrite "
        f"{len(at_risk)} cohort(s) that carry brain-state columns.",
        "",
        "  Rewriting a latents file REMOVES those columns -- parquet cannot "
        "append, so stage 4b's labels live inside the file -- and getting them "
        "back means re-fitting, which for HMM2 is hours per cell.",
        "",
    ]
    for c, (_, cols) in at_risk.items():
        lines.append(f"    cohort={c}: {len(cols)} state set(s), e.g. "
                     f"{cols[:4]}")
    lines += ["", "  They are stale because this run describes a DIFFERENT "
                  "fit:"]
    lines += diff or ["      (no readable difference -- the stored fit "
                      "description is older than this provenance block)"]
    lines += [
        "",
        "  If a flag above is missing from this command, add it back: the fit "
        "is defined by the flags, so omitting one is a different model.",
        "  If the change is intended, pass --overwrite to accept losing the "
        "state columns and re-cluster afterwards.",
    ]
    raise SystemExit("\n".join(lines))


def reuse_models(stem: Path, mhash: str, log=print) -> dict | None:
    """The saved fit for this cell, if it is the SAME fit. Else None.

    The manifest beside the models is what makes this checkable without
    unpickling: it records the `model_hash` and the library versions the
    objects were pickled under. A hash match says the fit is the one wanted;
    the version check is separate because a pickled sklearn estimator is not
    guaranteed to behave across versions, and silently applying one that does
    not is worse than refitting.
    """
    manifest = stem.parent / f"{stem.name}_manifest.json"
    if not manifest.exists():
        return None
    try:
        man = json.loads(manifest.read_text())
    except (ValueError, OSError):
        return None
    if man.get("model_hash") != mhash:
        return None
    from .io import _versions

    want = {k: v for k, v in _versions().items() if k in ("numpy",)}
    try:
        import sklearn

        want["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    have = man.get("library_versions") or {}
    drift = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
    if drift:
        log(f"  saved fit matches model_hash {mhash} but was pickled under "
            f"different libraries {drift} -- refitting rather than trusting it")
        return None
    for suffix in (".joblib", ".pkl"):
        path = stem.with_suffix(suffix)
        if not path.exists():
            continue
        try:
            if suffix == ".joblib":
                import joblib

                return joblib.load(path)
            import pickle

            return pickle.loads(path.read_bytes())
        except Exception as e:                       # a truncated or stale file
            log(f"  could not load {path.name} ({type(e).__name__}) -- refitting")
            return None
    return None


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
    out_dir = latents_root(root, atlas, window_s)
    models_dir = root / "meta" / "models"
    # KEYED BY (atlas, window) AND NOT BY model_hash, unlike stage 4b's
    # clusterer cache. So one output root holds ONE fit per cell: changing
    # `--train` overwrites the previous fit's objects rather than sitting
    # beside them.
    #
    # That is safe, not silent -- `reuse_models` compares the manifest's
    # model_hash and refits on a mismatch, and every latents file carries the
    # hash it was written with, so nothing can read a model that did not make
    # it. What it costs is co-residency: two training splits at one (atlas,
    # aperture) cannot both keep their artefacts here, and each run evicts the
    # other's. Use a separate --output-root for a second split. Adding the hash
    # to this name would fix it and would also move where `state_means` looks
    # (state_means.load_models builds the same stem), so it is a change to make
    # deliberately rather than in passing.
    stem = models_dir / f"decompose_atlas-{atlas}_window-{window_s}"

    cohorts = list(dict.fromkeys(args.train + args.project))
    source = source_of(args)
    first = source_shard_paths(root, atlas, window_s, args.train[0], source)
    if not first:
        raise SystemExit(f"no {source} shards for the first training cohort "
                         f"{args.train[0]!r} at atlas={atlas} window_s={window_s}")
    features = source_feature_columns(first[0], source)

    # The hash is computed from the config and the feature list alone, so WHICH
    # cohorts need writing is known before a single row is read. That is what
    # makes adding one cohort to a cell cheap: the others are left exactly as
    # they are, rather than rewritten identically and stripped of the state
    # columns stage 4b put in them.
    meta = fit_meta(args, window_s, features)
    mhash = model_hash(meta, features)
    on_disk = {c: latents_model_hash(out_dir / f"cohort={c}" / "data.parquet")
               for c in cohorts}
    todo = [c for c in cohorts if args.overwrite or on_disk[c] != mhash]
    keep = [c for c in cohorts if c not in todo]
    # A cohort with NO file is new and free to write. One with a DIFFERENT hash
    # is the dangerous case: it exists, it may carry state columns, and a
    # rewrite deletes them.
    if not args.overwrite:
        refuse_to_destroy_states(
            out_dir, [c for c in todo if on_disk[c] is not None], meta,
            window_s)

    unit = "parcels" if source == "activation" else "edges"
    log(f"window_s={window_s}  source={source}  {len(features)} {unit}  "
        f"model_hash {mhash}")
    if keep:
        log(f"  already at this model_hash, left untouched: {', '.join(keep)}")
    if not todo:
        log(f"  nothing to write (--overwrite to redo)")
        return
    log(f"  to write: {', '.join(todo)}")

    # Refit only when the saved fit for this cell is not the one wanted. The
    # scaler/PCA/UMAP depend on `--train` and the settings, never on who is
    # projected, so projecting a new cohort onto an existing fit needs no fit
    # at all -- which is the difference between minutes and an hour at
    # harvardoxford, on top of not disturbing the other cohorts.
    models = None if args.overwrite else reuse_models(stem, mhash, log=log)
    read_cohorts = list(dict.fromkeys(
        todo if models is not None else list(args.train) + todo))

    counts = {c: len(source_shard_paths(root, atlas, window_s, c, source))
              for c in read_cohorts}
    log(f"  shards: {counts}")
    if source == "activation":
        log(f"  frames: bandpass="
            f"{tuple(args.match_bandpass) if args.match_bandpass else 'as extracted'}"
            f"  zscore_runs={args.zscore_runs}  (both are in model_hash)")

    # Acted on, not just printed. This used to be a log line only, so a cohort
    # with 0 shards was visible on line 2 and discovered on line 20 -- after a
    # full fit, and after the training cohorts had already been written. That
    # left window sizes half-done, which is worse than not starting.
    empty = [c for c, n in counts.items() if n == 0]
    if empty:
        fix = ("    fmri-decomp extract config/<cohort>.yaml --atlas "
               f"{atlas}\n" if source == "activation" else
               f"    fmri-decomp dfc config/<cohort>.yaml --window-s {window_s}\n")
        raise SystemExit(
            f"window_s={window_s}: no {source} shards for {', '.join(empty)} at "
            f"atlas={atlas}.\n"
            f"Nothing is written for this window size. Either run the stage "
            f"that writes them:\n" + fix +
            f"or drop {window_s} from --window-s.")

    censors = {c: load_censor(root, args.censor_policy, atlas, window_s, c)
               for c in read_cohorts}
    if args.censor_policy:
        for c, cen in censors.items():
            gate = "subjects+windows" if cen["windows"] is not None else "subjects only"
            log(f"censor[{args.censor_policy}] {c}: "
                f"{cen['n_subject_kept']}/{cen['n_subject_rows']} subjects kept "
                f"({gate})")
    else:
        log("WARNING: no --censor-policy. Every window in every shard enters "
            "the fit, including subjects `censor` would have dropped.")

    role_of = {c: ("train" if c in args.train else "projected") for c in cohorts}

    if models is not None:
        log(f"reusing the saved fit for model_hash {mhash} -- no refit, and "
            f"the training cohorts are not re-read")
        # role_of is per-invocation and the saved copy was written by the run
        # that fitted, which may not have known about today's cohorts. Refresh
        # it so a newly projected cohort is labelled `projected` rather than
        # raising a KeyError in write_latents.
        models["role_of"] = {**models.get("role_of", {}), **role_of}
        n_train = int(models.get("n_train_rows") or 0)
    else:
        log("loading training cohorts")
        idents, blocks = [], []
        stride_s = indep_factor = None
        for cohort in args.train:
            ident, X, stride_s, indep_factor = source_read_cohort(
                root, atlas, window_s, cohort, features, args, censors[cohort])
            ident, X = drop_nan_rows(ident, X)
            log(f"  {cohort}: {len(X):,} rows, {ident['sub'].nunique()} subs, "
                f"{X.nbytes / 1e9:.2f} GB")
            idents.append(ident)
            blocks.append(X)
        X_train = np.vstack(blocks) if len(blocks) > 1 else blocks[0]
        n_train = len(X_train)
        del idents, blocks
        gc.collect()
        log(f"training matrix {X_train.shape} = {X_train.nbytes / 1e9:.2f} GB")
        log(f"model_hash {mhash}  roles {role_of}")

        models = fit_models(X_train, features, args, meta={
            **meta, "n_train_rows": int(n_train), "model_hash": mhash,
            "fit_meta": meta, "role_of": role_of,
            "stride_s": stride_s, "indep_factor": indep_factor})
        del X_train
        gc.collect()

        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = dump_models(models, stem)
        log(f"models -> {model_path.relative_to(root)}")

        # A manifest beside the models, in the same spirit as io.write_manifest:
        # readable without unpickling anything, and the place to look when two
        # latents files disagree. `library_versions` is what lets a later run
        # decide whether the pickle beside it can be trusted.
        from . import __version__
        from .io import _versions

        libs = {"numpy": _versions()["numpy"]}
        try:
            import sklearn

            libs["sklearn"] = sklearn.__version__
        except ImportError:
            pass
        (stem.parent / f"{stem.name}_manifest.json").write_text(json.dumps({
            **meta, "model_hash": mhash, "n_train_rows": int(n_train),
            "project_cohorts": list(args.project),
            "umap_fitted": bool(models.get("umap_fitted")),
            "n_umap_components": sorted(models["umap"]),
            "n_features": len(features), "role_of": role_of,
            "shards": counts, "models_file": model_path.name,
            "package_version": __version__, "library_versions": libs,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))

    # Second pass: one cohort in memory at a time. Only the cohorts that need
    # it -- an untouched cohort keeps its file, and with it every state column
    # stage 4b wrote into it.
    for cohort in todo:
        ident, X, stride_s, indep_factor = source_read_cohort(
            root, atlas, window_s, cohort, features, args, censors[cohort])
        ident, X = drop_nan_rows(ident, X)
        # Per cohort, because the TR is: see write_latents on why this sits
        # outside fit_meta and therefore outside model_hash.
        models["stride_s"], models["indep_factor"] = stride_s, indep_factor
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
    source = source_of(args)
    unit = "parcels" if source == "activation" else "edges"
    print(f"{'window':>7}  {unit:>7}  {'train rows':>11}  {'peak GB':>8}  shards")
    for w in args.window_s:
        paths = {c: source_shard_paths(root, args.atlas, w, c, source)
                 for c in dict.fromkeys(args.train + args.project)}
        first = next((p for c in args.train for p in paths[c]), None)
        if first is None:
            print(f"{w:>7}  (no shards)")
            continue
        n_edges = len(source_feature_columns(first, source))
        import pyarrow.parquet as pq
        rows = {c: sum(pq.ParquetFile(p).metadata.num_rows for p in ps)
                for c, ps in paths.items()}
        n_train = sum(rows[c] for c in args.train)
        peak = n_train * n_edges * 4 * 1.6 / 1e9
        print(f"{w:>7}  {n_edges:>7}  {n_train:>11,}  {peak:>8.2f}  "
              + "  ".join(f"{c}:{len(ps)}({rows[c]:,})" for c, ps in paths.items()))
    print("\npeak GB is the training matrix x1.6 for scaler and PCA workspace; "
          "ask for at least double.")
    if source == "activation":
        print("rows are counted from the parquet footers, so they INCLUDE the "
              "frames that will be dropped as not good or as a run boundary -- "
              "an upper bound, which is the safe direction for an allocation.")
    if args.censor_policy:
        print(f"rows are UNCENSORED counts -- --censor-policy "
              f"{args.censor_policy} will remove some, so this is an upper "
              f"bound on memory, which is the safe direction for an allocation.")


def add_arguments(p) -> None:
    """Shared by `fmri-decomp decompose` and `python -m ...decompose`."""
    # One source of truth for the band, imported here because argparse evaluates
    # the default when the parser is built rather than when the module loads.
    from .frames import DEFAULT_BANDPASS
    p.add_argument("--atlas", required=True)
    p.add_argument("--source", choices=["dfc", "activation"], default="dfc",
                   help="dfc: windowed edges, one fit per --window-s. "
                        "activation: per-TR parcel patterns, written to "
                        f"window_s={ACTIVATION_WINDOW} -- one fit, because a "
                        "frame has no window to vary.")
    p.add_argument("--window-s", nargs="+", default=None,
                   help="one fit per window size; they are independent. Not "
                        "accepted with --source activation, which has exactly "
                        f"one aperture and names it {ACTIVATION_WINDOW}.")
    p.add_argument("--match-bandpass", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   default=list(DEFAULT_BANDPASS),
                   help="--source activation only: band-pass every cohort's "
                        "frames onto this band before the fit, so one cohort's "
                        "wider preprocessing band does not make its states "
                        "flicker faster than another's. Default %(default)s.")
    p.add_argument("--no-match-bandpass", dest="match_bandpass",
                   action="store_const", const=None,
                   help="leave the frequency content as extracted. A different "
                        "analysis, and recorded as one in model_hash.")
    p.add_argument("--no-zscore-runs", dest="zscore_runs", action="store_false",
                   help="--source activation only: do NOT centre and scale each "
                        "parcel within each run. Off-by-default because without "
                        "it the fit is dominated by between-scanner signal "
                        "scale -- see frames.py.")
    p.set_defaults(zscore_runs=True)
    p.add_argument("--train", nargs="+", default=["ds002837", "cneuromod"])
    p.add_argument("--project", nargs="*", default=["camcan"],
                   help="cohorts the fit is APPLIED to. `--project` with no "
                        "values means none of them: fit on --train and write "
                        "only those, which `nargs=+` made impossible to say -- "
                        "the nearest expressible thing was projecting a "
                        "training cohort onto itself, which works by accident "
                        "and reads as nonsense.")
    p.add_argument("--n-latents", nargs="+", type=int, default=[2, 3, 5],
                   help="must include 3: every stage 4b clusterer works on a "
                        "3-D embedding")
    p.add_argument("--umap-fit-rows", type=int, default=30_000,
                   help="0 fits UMAP on every training row")
    p.add_argument("--umap-latents", nargs="+", type=int, default=None,
                   metavar="N",
                   help="component counts to fit UMAP at. Default: the same as "
                        "--n-latents. Set it when --n-latents carries a count "
                        "that only PCA needs -- `--n-latents 3 14 "
                        "--umap-latents 3` fits UMAP once, at 3, instead of "
                        "also spending most of the stage on a 14-component "
                        "UMAP nothing reads.")
    p.add_argument("--no-umap", action="store_true")
    p.add_argument("--passthrough-features", action="store_true",
                   help="also write the input features themselves into the "
                        "latents table, as `raw/<name>` columns, scaled the "
                        "same way PCA sees them. Stage 4b can then fit a "
                        "clusterer on the NAMED parcels -- `raw/AM`, "
                        "`raw/Motor` -- so a state mean is readable as a "
                        "network pattern rather than a point in PC space. "
                        f"Refused above {PASSTHROUGH_MAX_FEATURES} features; "
                        "intended for --source activation.")
    p.add_argument("--censor-policy", required=True, metavar="NAME",
                   type=_censor_policy,
                   help="apply the stage 3.5 decision written by `fmri-decomp "
                        "censor --policy config/censor/NAME.yaml`. REQUIRED: "
                        "pass `none` to fit on every window, which is a "
                        "different analysis and is recorded as one. There is no "
                        "default, because the default was silently `none` and "
                        "one aperture got built that way while the others were "
                        "censored -- a warning in a 30-line log is not a guard.")
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
    # Write the resolved root back, so anything downstream of here (censor
    # lookup, fit_meta) sees a path rather than the None the user passed.
    args.output_root = str(root)
    log(f"output_root {root}")

    # The aperture is a property of the source, not a free parameter, so it is
    # resolved here once rather than trusted from the command line. Silently
    # overriding a --window-s the user typed would write the fit to a path they
    # did not ask for, so it is an error instead.
    if source_of(args) == "activation":
        if args.window_s and list(args.window_s) != [ACTIVATION_WINDOW]:
            raise SystemExit(
                f"--source activation has one aperture, written as "
                f"window_s={ACTIVATION_WINDOW} (a frame is one TR, and the TR "
                f"differs per cohort). Drop --window-s {' '.join(args.window_s)}.")
        args.window_s = [ACTIVATION_WINDOW]
    elif not args.window_s:
        raise SystemExit("--window-s is required with --source dfc")

    if args.dry_run:
        dry_run(root, args)
        return 0

    # Each window size is an independent fit, so one failing is not a reason to
    # abandon the others -- and a bare loop meant a missing cohort at the FIRST
    # size silently cancelled the remaining four.
    failed = {}
    for w in args.window_s:
        try:
            run_one(root, w, args)
        except SystemExit as exc:
            failed[w] = str(exc)
            log(f"window_s={w} FAILED, continuing with the remaining sizes")

    done = [w for w in args.window_s if w not in failed]
    log(f"done: {len(done)}/{len(args.window_s)} window size(s) written"
        + (f" {done}" if done else ""))
    if failed:
        print(f"\n{len(failed)} window size(s) did not run:", flush=True)
        for w, msg in failed.items():
            print(f"\n  --- window_s={w} ---\n  " + msg.replace("\n", "\n  "),
                  flush=True)
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
