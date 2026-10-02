#!/usr/bin/env python3
"""Stage 4b -- add brain-state definitions to latents that already exist.

    fmri-decomp cluster --atlas yeo7 --window-s 30 60 120 300
    fmri-decomp cluster --methods threshold meanshift hmm \\
                        --embeddings pca3 umap3 --k 8 27

A latents file already holds `pca0/3..pca2/3` and `umap0/3..umap2/3`. A new way
of defining states needs neither PCA nor UMAP refitted -- it reads those columns
and appends label columns to the same file. That is the whole reason this is a
separate stage from `decompose`: adding a sixth state definition must not mean
redoing the five that already work.

ADDING A METHOD LATER
---------------------
Write a class with `fit(X, k)` and `labels(X)`, add one line to CLUSTERERS, and
nothing downstream changes: `transitions` discovers state columns from the file
schema, so a new method is picked up without being named anywhere.

EVERY METHOD MUST PROJECT
-------------------------
The whole design fits on the training cohorts and projects onto camcan, which
is never seen by any `fit`. That rules out DBSCAN, HDBSCAN and OPTICS, which in
scikit-learn expose `fit_predict` and nothing else -- there is no mechanism to
label a new subject at all. MeanShift is the density method that survives the
constraint: it discovers K from a bandwidth rather than being told it, and it
has a real `predict`.

WHAT EACH METHOD NEEDS
----------------------
    threshold   K = bins**3, so k=8 is 2 bins per axis and k=27 is 3. Quantile
                edges, fitted on the training rows.
    meanshift   K is DISCOVERED. `--meanshift-quantile` sets the bandwidth
                (smaller -> more states); the K it finds is recorded in the
                column name.
    hmm         K is specified. Fitted with per-(task, sub) sequence lengths so
                it never models a transition across a subject boundary.

An HMM is itself a transition model: it fits `transmat_` while defining the
states, and stage 5a then counts transitions per subject from its Viterbi path.
Those are different quantities -- group-level parameter against per-subject
counts -- but it is the first thing a reader will ask about, so it is recorded
in the provenance rather than left implicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import latents_root, meta_dir

EMBEDDINGS = {
    "pca3": ["pca0/3", "pca1/3", "pca2/3"],
    "umap3": ["umap0/3", "umap1/3", "umap2/3"],
}

# Which cohorts a clusterer may be fitted on. Kept here rather than read from
# the latents' own `role` column so that a mistake is visible in the command.
DEFAULT_TRAIN = ["ds002837", "cneuromod"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------ clusterers ---
class Threshold:
    """Quantile bins per axis, raveled into one label. K = bins**3.

    The same definition `decompose` already writes, lifted here so every method
    goes through one code path. Re-running it on pca3 reproduces the existing
    `ThresholdCluster_pca3_*` columns.
    """

    name = "ThresholdCluster"

    def fit(self, X, k, rng=None):
        from sklearn.preprocessing import KBinsDiscretizer

        bins = round(k ** (1 / 3))
        if bins ** 3 != k:
            raise SystemExit(f"threshold needs a perfect cube for k; {k} is not "
                             f"({bins}**3 = {bins ** 3}). Use 8, 27, 64, ...")
        self.bins = bins
        self.kbd = KBinsDiscretizer(n_bins=bins, encode="ordinal",
                                    strategy="quantile",
                                    subsample=None).fit(X)
        self.k_found = k
        return self

    def labels(self, X):
        b = self.kbd.transform(X).astype(int)
        return np.ravel_multi_index(b.T, (self.bins,) * 3)

    def params(self):
        return {"bins_per_axis": self.bins, "strategy": "quantile"}


class MeanShiftCluster:
    """Density clustering that DISCOVERS K and can still label new data.

    DBSCAN was the first choice and cannot be used: scikit-learn's DBSCAN,
    HDBSCAN and OPTICS all expose `fit_predict` only, so a held-out cohort can
    never be labelled. MeanShift keeps the "do not tell it K" property and has
    a real `predict`, which is what makes the train/project split possible.
    """

    name = "MeanShift"

    def __init__(self, quantile=0.2, fit_rows=50_000):
        self.quantile, self.fit_rows = quantile, fit_rows

    def fit(self, X, k=None, rng=None):
        from sklearn.cluster import MeanShift, estimate_bandwidth

        rng = rng or np.random.default_rng(0)
        idx = (rng.choice(len(X), self.fit_rows, replace=False)
               if self.fit_rows and len(X) > self.fit_rows else np.arange(len(X)))
        self.bandwidth = float(estimate_bandwidth(
            X[idx], quantile=self.quantile,
            n_samples=min(10_000, len(idx)), random_state=0))
        if not self.bandwidth > 0:
            raise SystemExit("estimate_bandwidth returned 0 -- the embedding is "
                             "degenerate, or --meanshift-quantile is too small")
        self.ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=True,
                            n_jobs=-1).fit(X[idx])
        self.k_found = int(len(self.ms.cluster_centers_))
        return self

    def labels(self, X):
        return self.ms.predict(X)

    def params(self):
        return {"quantile": self.quantile, "bandwidth": round(self.bandwidth, 5),
                "fit_rows": int(self.fit_rows)}


class HMMCluster:
    """Gaussian HMM. K is specified; sequences are per (task, sub).

    `lengths` is not optional. Without it hmmlearn treats the stacked rows as
    ONE sequence and learns a transition from the last window of each subject
    to the first window of the next -- a transition between two different
    people, which is not a thing.
    """

    name = "HMM"

    def __init__(self, n_iter=50, covariance_type="diag"):
        self.n_iter, self.covariance_type = n_iter, covariance_type

    def fit(self, X, k, rng=None, lengths=None):
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise SystemExit(
                "hmmlearn is not installed in this environment.\n"
                "containers/stage45.def pins hmmlearn==0.3.3 -- use that image, "
                "or drop `hmm` from --methods.")
        self.hmm = GaussianHMM(n_components=k, covariance_type=self.covariance_type,
                               n_iter=self.n_iter, random_state=0)
        self.hmm.fit(X, lengths=lengths)
        self.k_found = k
        return self

    def labels(self, X, lengths=None):
        return self.hmm.predict(X, lengths=lengths)

    def params(self):
        return {"n_iter": self.n_iter, "covariance_type": self.covariance_type,
                "note": "the HMM fits its own transmat_ while defining the "
                        "states; stage 5a counts per-subject transitions from "
                        "the Viterbi path, which is a different quantity"}


CLUSTERERS = {
    "threshold": Threshold,
    "meanshift": MeanShiftCluster,
    "hmm": HMMCluster,
}
# Methods that discover K rather than being given one.
K_FREE = {"meanshift"}
SEQUENTIAL = {"hmm"}          # need per-subject sequence lengths


def column_name(method: str, embedding: str, k: int) -> str:
    """`<Method>_<embedding>_<K>` -- the convention stage 5a parses."""
    return f"{CLUSTERERS[method].name}_{embedding}_{k}"


# ----------------------------------------------------------------- data ---
def cohort_paths(root: Path, atlas: str, window_s) -> dict[str, Path]:
    d = latents_root(root, atlas, window_s)
    if not d.is_dir():
        return {}
    return {p.name.split("=", 1)[1]: p / "data.parquet"
            for p in sorted(d.glob("cohort=*"))
            if (p / "data.parquet").exists()}


def _seq_lengths(df: pd.DataFrame) -> list[int]:
    """Row counts per (task, sub), in the order the frame is already sorted."""
    return (df.groupby(["task", "sub"], sort=False).size().tolist())


def read_embedding(path: Path, cols: list[str]) -> pd.DataFrame | None:
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(path).schema_arrow.names)
    need = cols + ["task", "sub", "window_id"]
    if not set(cols) <= have:
        return None
    return pd.read_parquet(path, columns=[c for c in need if c in have])


def fit_hash(method: str, embedding: str, k, params: dict, train: list[str],
             n_rows: int) -> str:
    payload = json.dumps({"method": method, "embedding": embedding, "k": k,
                          "params": params, "train": sorted(train),
                          "n_train_rows": n_rows}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def append_columns(path: Path, new: dict[str, np.ndarray],
                   provenance: dict) -> None:
    """Add label columns to a latents file, carrying per-column provenance.

    Parquet cannot append a column in place, so the file is rewritten. The
    existing schema metadata is preserved and a `clusterers` entry is merged
    into it: the file's own `model_hash` describes the `decompose` fit that made
    the embeddings, and says nothing about a clusterer added afterwards. Without
    per-column provenance there would be no way to tell which fit produced which
    label column.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    md = dict(table.schema.metadata or {})
    existing = json.loads(md.get(b"clusterers", b"{}").decode())
    existing.update(provenance)
    md[b"clusterers"] = json.dumps(existing, default=str).encode()

    for col, values in new.items():
        arr = pa.array(np.asarray(values, dtype=np.int32))
        if col in table.column_names:
            table = table.set_column(table.column_names.index(col), col, arr)
        else:
            table = table.append_column(col, arr)

    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(table.replace_schema_metadata(md), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ------------------------------------------------------------------ run ---
def run_one(root: Path, atlas: str, window_s, args) -> list[dict]:
    paths = cohort_paths(root, atlas, window_s)
    if not paths:
        log(f"  atlas={atlas} window_s={window_s}: no latents, skipped")
        return []
    train = [c for c in args.train if c in paths]
    if not train:
        raise SystemExit(f"none of --train {args.train} has latents at "
                         f"atlas={atlas} window_s={window_s}; found "
                         f"{sorted(paths)}")

    entries = []
    for emb in args.embeddings:
        cols = EMBEDDINGS[emb]
        frames = {c: read_embedding(p, cols) for c, p in paths.items()}
        if any(f is None for f in frames.values()):
            missing = [c for c, f in frames.items() if f is None]
            log(f"  {emb}: not in {missing} -- skipped "
                f"(was decompose run with the matching --n-latents?)")
            continue

        Xtr = np.vstack([frames[c][cols].to_numpy(float) for c in train])
        len_tr = [n for c in train for n in _seq_lengths(frames[c])]

        for method in args.methods:
            ks = [None] if method in K_FREE else args.k
            for k in ks:
                cl = CLUSTERERS[method](**_opts(method, args))
                t0 = time.time()
                if method in SEQUENTIAL:
                    cl.fit(Xtr, k, lengths=len_tr)
                else:
                    cl.fit(Xtr, k)
                col = column_name(method, emb, cl.k_found)
                h = fit_hash(method, emb, cl.k_found, cl.params(), train, len(Xtr))

                written = {}
                for cohort, f in frames.items():
                    X = f[cols].to_numpy(float)
                    lab = (cl.labels(X, lengths=_seq_lengths(f))
                           if method in SEQUENTIAL else cl.labels(X))
                    append_columns(paths[cohort], {col: lab},
                                   {col: {"method": method, "embedding": emb,
                                          "k": int(cl.k_found), "fit_hash": h,
                                          "params": cl.params(),
                                          "train_cohorts": train,
                                          "n_train_rows": int(len(Xtr)),
                                          "written_utc": time.strftime(
                                              "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}})
                    written[cohort] = int(len(np.unique(lab)))
                log(f"  {col:<28} K={cl.k_found:<4} {time.time()-t0:>5.1f}s  "
                    f"states used per cohort {written}")
                entries.append({"atlas": atlas, "window_s": str(window_s),
                                "column": col, "method": method,
                                "embedding": emb, "k": int(cl.k_found),
                                "fit_hash": h, "train_cohorts": train,
                                "states_used": written})
    return entries


def _opts(method: str, args) -> dict:
    if method == "meanshift":
        return {"quantile": args.meanshift_quantile,
                "fit_rows": args.meanshift_fit_rows}
    if method == "hmm":
        return {"n_iter": args.hmm_iter}
    return {}


def run(args) -> int:
    root = Path(args.output_root) if args.output_root else _default_root()
    if not root.is_dir():
        raise SystemExit(f"output_root does not exist: {root}")
    log(f"output_root {root}")
    log(f"methods {args.methods}  embeddings {args.embeddings}  k {args.k}")
    log(f"fitted on {args.train}, applied to every cohort present")

    entries = []
    for atlas in args.atlas:
        for w in args.window_s:
            log(f"atlas={atlas} window_s={w}")
            entries += run_one(root, atlas, w, args)

    if not entries:
        raise SystemExit("nothing was written -- check --atlas / --window-s "
                         "against what decompose produced")

    out = meta_dir(root) / "cluster" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"entries": entries, "methods": args.methods,
         "embeddings": args.embeddings, "k": args.k, "train": args.train,
         "n_state_columns": len(entries),
         "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, default=str))
    log(f"{len(entries)} state column(s) written -> {out.relative_to(root)}")
    return 0


def _default_root() -> Path:
    if os.environ.get("FMRIDECOMP_OUTPUTS"):
        return Path(os.environ["FMRIDECOMP_OUTPUTS"])
    import yaml
    repo = Path(__file__).resolve().parent.parent
    return Path(yaml.safe_load(
        (repo / "config" / "camcan_movie.yaml").read_text())["output_root"])


def add_arguments(p) -> None:
    p.add_argument("--atlas", nargs="+",
                   default=["harvardoxford", "yeo7", "networks"])
    p.add_argument("--window-s", nargs="+",
                   default=["30", "60", "120", "300"])
    p.add_argument("--methods", nargs="+", default=list(CLUSTERERS),
                   choices=list(CLUSTERERS))
    p.add_argument("--embeddings", nargs="+", default=list(EMBEDDINGS),
                   choices=list(EMBEDDINGS))
    p.add_argument("--k", nargs="+", type=int, default=[8, 27],
                   help="for methods that need one; meanshift discovers it")
    p.add_argument("--train", nargs="+", default=DEFAULT_TRAIN,
                   help="cohorts a clusterer may be FITTED on; every cohort "
                        "present is then labelled")
    p.add_argument("--meanshift-quantile", type=float, default=0.2,
                   help="bandwidth quantile; smaller finds more states")
    p.add_argument("--meanshift-fit-rows", type=int, default=50_000)
    p.add_argument("--hmm-iter", type=int, default=50)
    p.add_argument("--output-root")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
