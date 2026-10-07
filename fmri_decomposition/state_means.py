#!/usr/bin/env python3
"""What each brain state LOOKS LIKE, in named network units.

    fmri-decomp state-means --atlas networks --window-s -1 \\
        --states HMM2_pca14_10

A state set is a column of small integers, and nothing downstream of stage 4b
ever needs to know more than that: `transitions` counts them, `select` ranks
them. This is the one thing that cannot be read off the integers -- what the
state actually is.

WHY THIS IS NOT JUST A GROUPBY MEAN
-----------------------------------
It is a groupby mean, and then an inverse transform, and the second part is the
reason the module exists.

`decompose` z-scores the input features and rotates them with PCA, so a state
mean computed on the latents is a point in PC space: `pca0/14 = -1.8` says
nothing a reader can use. Both steps are invertible and both fitted objects are
on disk, so the mean can be carried back to the units that have names:

    scaler.inverse_transform(pca.inverse_transform(mean_in_pc_space))

That is exact to floating point -- a full-rank PCA is an orthonormal rotation,
measured at ~1e-14 round-trip -- so the figure this produces is the same figure
you would get by averaging the raw parcels, with no approximation in between.

It is the missing half of the comparison with van der Meer et al. 2020, whose
Fig. 1 is exactly this: a heat map per state over 14 named networks. They could
make it directly because they never reduced; we need the rotation undone first.

A `raw<N>` embedding needs only the scaler undone, since those columns ARE the
features. The output is identical in form either way, which is the point: a
state mean from `HMM2_pca14_10` and one from `HMM2_raw14_10` are directly
comparable, in the same units, on the same axes.

WHAT IT DOES NOT DO
-------------------
No covariance. An HMM state has one, it is a real part of the state's identity,
and `full` covariance is the main thing separating HMM2 from HMM1 -- but a K x N
x N array is not a table, and the thing people reach for first is the mean. The
fitted model is on disk for anyone who wants the rest.

Read-only. Nothing here writes to `outputs/latents/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import RAW_PREFIX, latents_root, read_file


def load_models(root: Path, atlas: str, window_s) -> dict:
    """The scaler/PCA/UMAP objects `decompose` saved for this cell."""
    stem = root / "meta" / "models" / f"decompose_atlas-{atlas}_window-{window_s}"
    for suffix, loader in ((".joblib", _joblib_load), (".pkl", _pickle_load)):
        path = stem.with_suffix(suffix)
        if path.exists():
            return loader(path)
    raise SystemExit(
        f"no saved models for atlas={atlas} window_s={window_s}.\n"
        f"  looked for {stem.with_suffix('.joblib')} and "
        f"{stem.with_suffix('.pkl')}\n"
        f"  `decompose` writes one per (atlas, aperture); without it the "
        f"rotation cannot be undone and a state mean stays in PC space.")


def _joblib_load(path: Path):
    import joblib

    return joblib.load(path)


def _pickle_load(path: Path):
    import pickle

    return pickle.loads(path.read_bytes())


def embedding_of(path: Path, state_col: str) -> tuple[str, list[str]]:
    """Which embedding a state column was fitted on, from its own provenance.

    Read from the `clusterers` block rather than parsed out of the column name.
    The name carries the embedding too, but as a label -- the provenance carries
    the actual column list, which is what has to be averaged. For a `raw<N>`
    column those names are the networks themselves and cannot be reconstructed
    from `raw14`.
    """
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    block = json.loads((md.get(b"clusterers") or b"{}").decode())
    entry = block.get(state_col)
    if entry is None:
        raise SystemExit(
            f"{state_col!r} has no entry in this file's `clusterers` "
            f"provenance.\n"
            f"  known: {sorted(block) or '(none)'}\n"
            f"  A column written by a stage 4b older than per-column "
            f"provenance cannot say which embedding it came from; re-run "
            f"`cluster` for this cell.")
    cols = entry.get("embedding_columns")
    if not cols:
        raise SystemExit(
            f"{state_col!r} records no `embedding_columns`, so the columns it "
            f"was fitted on are not known.\n"
            f"  Re-run `cluster` for this cell: it records them now.")
    return entry.get("embedding", "?"), list(cols)


def means_in_feature_units(root: Path, atlas: str, window_s, state_col: str,
                           cohorts: list[str] | None = None) -> pd.DataFrame:
    """One row per state, one column per named feature. Plus `n_rows`, `share`.

    Pooled over the cohorts asked for, weighted by how many rows each
    contributes -- the states are defined by ONE fit across cohorts, so their
    means are a property of that fit and not of any single cohort. Ask for one
    cohort to see that cohort's expression of them.
    """
    cell = latents_root(root, atlas, window_s)
    paths = {p.parent.name.split("=", 1)[1]: p
             for p in sorted(cell.glob("cohort=*/data.parquet"))}
    if cohorts:
        missing = [c for c in cohorts if c not in paths]
        if missing:
            raise SystemExit(f"no latents for cohort(s) {missing} at "
                             f"atlas={atlas} window_s={window_s}; "
                             f"found {sorted(paths)}")
        paths = {c: paths[c] for c in cohorts}
    if not paths:
        raise SystemExit(f"no latents at atlas={atlas} window_s={window_s}")

    emb, cols = embedding_of(next(iter(paths.values())), state_col)
    # Decided BEFORE anything is read. Checking after the groupby meant a UMAP
    # state set died inside pandas with `Columns not found: 'umap0/3'...`, which
    # names neither the real reason nor the fix.
    is_raw = all(c.startswith(RAW_PREFIX) for c in cols)
    if not is_raw and not all(c.startswith("pca") for c in cols):
        raise SystemExit(
            f"{state_col!r} was fitted on {emb!r}, which is not invertible.\n"
            f"  Recovering a state mean in network units means undoing the "
            f"transform, and UMAP has no inverse_transform that recovers its "
            f"input -- the embedding is not a rotation, it is a learned "
            f"non-linear map.\n"
            f"  Use a pca<N> or raw<N> state set for this. On the same cell "
            f"they label the same rows, so the states are comparable even "
            f"though this one's mean is not recoverable.")
    frames = []
    for cohort, path in paths.items():
        t = read_file(path, columns=cols + [state_col]).to_pandas()
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)

    g = df.groupby(state_col, sort=True)
    emb_means = g[cols].mean()
    counts = g.size().rename("n_rows")

    models = load_models(root, atlas, window_s)
    features = list(models["edges"])
    arr = emb_means.to_numpy(float)

    if is_raw:
        # Already the features, only scaled. Undo the scaler, in the feature
        # order the columns are in rather than the order `edges` happens to be:
        # `raw_columns` sorts, and the scaler's parameters are indexed by the
        # original order.
        names = [c[len(RAW_PREFIX):] for c in cols]
        idx = [features.index(n) for n in names]
        full = np.zeros((len(arr), len(features)))
        full[:, idx] = arr
        back = models["scaler"].inverse_transform(full)[:, idx]
        out = pd.DataFrame(back, index=emb_means.index, columns=names)
    else:
        n = int(cols[0].partition("/")[2])
        pca = models["pca"].get(n)
        if pca is None:
            raise SystemExit(f"the saved models for this cell have no {n}-"
                             f"component PCA; found {sorted(models['pca'])}")
        back = models["scaler"].inverse_transform(pca.inverse_transform(arr))
        out = pd.DataFrame(back, index=emb_means.index, columns=features)
        if n < len(features):
            # Honest rather than silent: a 3-component PCA inverted back to 111
            # parcels is the state mean PROJECTED ONTO a 3-D subspace, not the
            # state's mean over the parcels. The numbers are in parcel units and
            # the shape of the pattern is real, but it carries only the variance
            # those components kept.
            out.attrs["lossy"] = (
                f"{emb} keeps {n} of {len(features)} dimensions, so these means "
                f"are the state's position in that {n}-D subspace expressed in "
                f"feature units -- not its mean over all {len(features)} "
                f"features.")
    out.insert(0, "n_rows", counts)
    out.insert(1, "share", counts / counts.sum())
    out.index.name = "state"
    out.attrs["embedding"] = emb
    out.attrs["cohorts"] = sorted(paths)
    return out


def add_arguments(p) -> None:
    p.add_argument("--atlas", required=True)
    p.add_argument("--window-s", required=True,
                   help="-1 for the activation aperture")
    p.add_argument("--states", required=True, metavar="COLUMN",
                   help="a state column, e.g. HMM2_pca14_10")
    p.add_argument("--cohorts", nargs="*", default=None,
                   help="default: every cohort in the cell, pooled")
    p.add_argument("--output-root")
    p.add_argument("--csv", help="also write the table here")


def run(args) -> int:
    import os

    if args.output_root:
        root = Path(args.output_root)
    elif os.environ.get("FMRIDECOMP_OUTPUTS"):
        root = Path(os.environ["FMRIDECOMP_OUTPUTS"])
    else:
        import yaml
        repo = Path(__file__).resolve().parent.parent
        root = Path(yaml.safe_load(
            (repo / "config" / "camcan_movie.yaml").read_text())["output_root"])

    out = means_in_feature_units(root, args.atlas, args.window_s, args.states,
                                 args.cohorts)
    print(f"\n{args.states}  embedding={out.attrs['embedding']}  "
          f"cohorts={','.join(out.attrs['cohorts'])}\n")
    with pd.option_context("display.width", 200,
                           "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v:+.3f}"):
        print(out.to_string())
    if out.attrs.get("lossy"):
        print(f"\nNOTE: {out.attrs['lossy']}")
    if args.csv:
        out.to_csv(args.csv)
        print(f"\n-> {args.csv}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
