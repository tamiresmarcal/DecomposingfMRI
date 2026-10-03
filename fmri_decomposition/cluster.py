#!/usr/bin/env python3
"""Stage 4b -- every brain-state definition, on latents that already exist.

    fmri-decomp cluster --atlas yeo7 --window-s 30 60 120 300
    fmri-decomp cluster --methods threshold meanshift hmm \\
                        --embeddings pca3 umap3 --k 8 27

THE ONLY PLACE A STATE IS DEFINED
---------------------------------
Stage 4 produces embeddings; this stage turns them into labels. ALL of them,
threshold included -- `--bins` used to do the thresholding inside `decompose`
and no longer exists, because one method sitting in a different stage from the
others is the wrong seam. It got re-fitted on every stage 4 re-run while the
others did not, and it wrote columns with no entry in the `clusterers`
provenance block, which is why `transitions.n_states_for` still carries a
fallback that reads K out of a column name.

A latents file holds `pca0/3..pca2/3` and `umap0/3..umap2/3`. A new way of
defining states needs neither PCA nor UMAP refitted -- it reads those columns
and appends label columns to the same file. Adding a sixth state definition must
not mean redoing the five that already work.

WHY THE LABELS GO IN THE LATENTS FILE AND NOT A TABLE OF THEIR OWN
------------------------------------------------------------------
A label is one small integer per row of a table that already exists, keyed by
exactly the columns that table is keyed by. A separate table would be the same
(cohort, task, sub, window_id) index repeated per state set, joined back on
every read, and joined wrongly the first time someone forgot one of the four
keys. The embeddings and their labels are one row of one table, which is also
how stage 4 has always written them.

It does mean this stage REWRITES each latents file (read, add columns, write,
rename) rather than appending in place, because parquet has no append. That is
why one job owns an (atlas, aperture) and the sbatch is not an array: two
writers would each rename over the other's columns.

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
                edges, fitted on the merged training rows. This is what
                `decompose --bins` used to write, moved here unchanged.
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

from .io import STATE_K_BAND, latents_root, meta_dir

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

    # `quantile` sets the bandwidth, and the right value is a property of the
    # DATA, not of the method: on the real grid, 0.2 -- the sklearn-ish default
    # this started with -- collapsed 7 of 8 harvardoxford state sets to K=1 or 2.
    # One state has one cell, `0->0 = 1.0` for every subject, so a K=1 state set
    # is not a weak arm in the selection grid, it is a column of a constant. And
    # the right quantile differs per (atlas, aperture, embedding), so there are
    # ~40 of them to pick and hand-tuning each is not a plan.
    #
    # So the quantile is SEARCHED rather than set: start wide, halve while K is
    # below the target band, widen while it is above, and stop at the first fit
    # that lands inside. The value that worked is recorded in params and
    # therefore in fit_hash, so two columns fitted at different quantiles are
    # never confused for the same model.
    def __init__(self, quantile=0.1, fit_rows=50_000, min_k=3, max_k=64,
                 max_tries=7):
        self.quantile, self.fit_rows = quantile, fit_rows
        self.min_k, self.max_k, self.max_tries = min_k, max_k, max_tries

    def fit(self, X, k=None, rng=None):
        from sklearn.cluster import MeanShift, estimate_bandwidth

        rng = rng or np.random.default_rng(0)
        idx = (rng.choice(len(X), self.fit_rows, replace=False)
               if self.fit_rows and len(X) > self.fit_rows else np.arange(len(X)))
        Xs = X[idx]

        q = float(self.quantile)
        self.search = []
        for _ in range(self.max_tries):
            bw = float(estimate_bandwidth(Xs, quantile=q,
                                          n_samples=min(10_000, len(Xs)),
                                          random_state=0))
            if not bw > 0:
                raise SystemExit(
                    f"estimate_bandwidth returned 0 at quantile {q:g} -- the "
                    f"embedding is degenerate (every row identical?), which no "
                    f"bandwidth can fix.")
            ms = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1).fit(Xs)
            k_found = int(len(ms.cluster_centers_))
            self.search.append({"quantile": round(q, 5),
                                "bandwidth": round(bw, 5), "k": k_found})
            self.ms, self.bandwidth, self.k_found, self.quantile_used = \
                ms, bw, k_found, q
            if self.min_k <= k_found <= self.max_k:
                break
            # Fewer modes than wanted means the bandwidth swallowed them, so
            # shrink it; more means it was too fine. Halving converges fast and
            # the 1.5x widening is deliberately gentler, so an overshoot does not
            # bounce straight back past the band.
            q = q / 2 if k_found < self.min_k else q * 1.5
        return self

    def labels(self, X):
        return self.ms.predict(X)

    def params(self):
        return {"quantile": round(self.quantile_used, 5),
                "quantile_start": self.quantile,
                "bandwidth": round(self.bandwidth, 5),
                "fit_rows": int(self.fit_rows),
                "target_k": [self.min_k, self.max_k],
                # The whole ladder, so a K at the edge of the band can be read
                # as "the nearest alternatives were 2 and 31" rather than taken
                # as a property of the data.
                "search": self.search}


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


def _training_block(frames: dict, train: list[str], cols: list[str], args):
    """The merged training matrix -> (X, per-(task,sub) lengths, row share).

    ONE FIT ON THE POOLED TRAINING COHORTS, projected onto everything else. That
    is what "discover overall states" means here, and it is what makes a state
    label comparable across cohorts at all: state 5 is only the same state in two
    cohorts if one fit defined it.

    Pooling is not neutral, though, and the log says so. The cohorts contribute
    very different numbers of rows -- different subject counts, different hours of
    film, and at the activation aperture different TRs, so ds002837 contributes
    ~2.5x the rows per minute of film that camcan would. A density method asked
    where the modes are will answer mostly about whichever cohort brought the most
    rows. `--balance-train` subsamples every cohort to the smallest one's row
    count so each has equal say; it is off by default because it throws data away,
    and which of the two you want is a judgement about the claim being made, not
    something this stage should decide.

    The sequence lengths follow the same subsampling, and they stay per
    (task, sub) either way -- an HMM given one stacked sequence would learn a
    transition from the last frame of one person to the first frame of the next.
    """
    counts = {c: len(frames[c]) for c in train}
    total = sum(counts.values()) or 1
    share = {c: counts[c] / total for c in train}

    if not args.balance_train:
        X = np.vstack([frames[c][cols].to_numpy(float) for c in train])
        lengths = [n for c in train for n in _seq_lengths(frames[c])]
        return X, lengths, share

    # Subsample by whole subjects, not by rows: dropping rows from the middle of
    # a subject's sequence would break the contiguity an HMM's `lengths` asserts.
    rng = np.random.default_rng(0)
    target = min(counts.values())
    blocks, lengths = [], []
    for c in train:
        f = frames[c]
        keys = list(dict.fromkeys(zip(f["task"], f["sub"])))
        rng.shuffle(keys)
        taken, kept = 0, []
        for key in keys:
            if taken >= target:
                break
            kept.append(key)
            taken += int(((f["task"] == key[0]) & (f["sub"] == key[1])).sum())
        keep = f.set_index(["task", "sub"]).index.isin(kept)
        sub = f.loc[keep]
        blocks.append(sub[cols].to_numpy(float))
        lengths += _seq_lengths(sub)
    return np.vstack(blocks), lengths, share


def has_columns(path: Path, cols: list[str]) -> bool:
    """Schema only -- no row group is touched. Used by the pre-flight, which asks
    about every embedding and would otherwise read each present one twice."""
    import pyarrow.parquet as pq

    return set(cols) <= set(pq.ParquetFile(path).schema_arrow.names)


def read_embedding(path: Path, cols: list[str]) -> pd.DataFrame | None:
    import pyarrow.parquet as pq

    from .io import read_file

    have = set(pq.ParquetFile(path).schema_arrow.names)
    need = cols + ["task", "sub", "window_id"]
    if not set(cols) <= have:
        return None
    # Via read_file rather than pd.read_parquet: the projection happens to
    # exclude `cohort` today, which is the only reason the dataset layer does not
    # raise here. Not a property worth depending on.
    return read_file(path, need).to_pandas()


def fit_hash(method: str, embedding: str, k, params: dict, train: list[str],
             n_rows: int, balanced: bool = False) -> str:
    payload = json.dumps({"method": method, "embedding": embedding, "k": k,
                          "params": params, "train": sorted(train),
                          "n_train_rows": n_rows, "balanced": balanced},
                         sort_keys=True, default=str)
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

    from .io import read_file

    # read_file, not pq.read_table: this path ends in `cohort=<c>/data.parquet`
    # and the file also has a `cohort` COLUMN, so read_table's hive inference
    # tries to merge the two and raises on pyarrow 18. See io.read_file.
    table = read_file(path)
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
def _require_embeddings(paths: dict, args, atlas: str, window_s) -> None:
    """Every requested embedding present in every cohort, or stop.

    A hard failure rather than a skip, and the same reasoning as stage 4's UMAP
    check: a run that writes 5 of the 10 state sets it was asked for and reports
    success is one whose gap surfaces much later, as `umap3 not in the latents`
    at the point someone is reading results.
    """
    gaps = {}
    for emb in args.embeddings:
        cols = EMBEDDINGS[emb]
        absent = [c for c, p in paths.items() if not has_columns(p, cols)]
        if absent:
            gaps[emb] = (cols, absent)
    if not gaps:
        return

    # The command that would produce them depends on which source wrote these
    # latents: window_s=-1 is only reachable with --source activation, and
    # suggesting a command that argparse rejects is worse than no suggestion.
    import json as _json

    import pyarrow.parquet as pq

    md = pq.ParquetFile(next(iter(paths.values()))).schema_arrow.metadata or {}
    raw = md.get(b"source")
    src = _json.loads(raw.decode()) if raw else "dfc"
    extra = " --source activation" if src == "activation" else ""
    have = [e for e in args.embeddings if e not in gaps]

    lines = [f"  {emb}: {', '.join(cols)} not in cohort(s) {absent}"
             for emb, (cols, absent) in gaps.items()]
    raise SystemExit(
        f"--embeddings asked for {', '.join(gaps)} at atlas={atlas} "
        f"window_s={window_s}, and those columns are not in the latents:\n"
        + "\n".join(lines) + "\n"
        f"Nothing is written. Refusing to add a partial set of state columns -- "
        f"a state set that is silently absent is one nothing downstream will "
        f"notice is missing.\n"
        f"Either give stage 4 that embedding:\n"
        f"    fmri-decomp decompose --atlas {atlas} --window-s {window_s}"
        f"{extra}       # without --no-umap\n"
        f"or ask only for the embeddings that exist:\n"
        f"    fmri-decomp cluster --embeddings {' '.join(have) or '<none left>'}")


def run_one(root: Path, atlas: str, window_s, args):
    """-> (state columns written, state sets refused for a degenerate K)."""
    paths = cohort_paths(root, atlas, window_s)
    if not paths:
        log(f"  atlas={atlas} window_s={window_s}: no latents, skipped")
        return [], []
    train = [c for c in args.train if c in paths]
    if not train:
        raise SystemExit(f"none of --train {args.train} has latents at "
                         f"atlas={atlas} window_s={window_s}; found "
                         f"{sorted(paths)}")

    # Every requested embedding is checked BEFORE anything is fitted. Checking
    # inside the loop would still have failed, but only after the earlier
    # embedding's columns had been written -- leaving the file with half the state
    # sets it was asked for, which is the state this stage exists to avoid.
    _require_embeddings(paths, args, atlas, window_s)

    entries, skipped = [], []
    for emb in args.embeddings:
        cols = EMBEDDINGS[emb]
        frames = {c: read_embedding(p, cols) for c, p in paths.items()}

        Xtr, len_tr, share = _training_block(frames, train, cols, args)
        how = "balanced" if args.balance_train else "pooled as-is"
        log(f"  {emb}: one fit on {len(Xtr):,} row(s) from {train} ({how}); "
            + "  ".join(f"{c} {share[c]:.0%}" for c in train)
            + " of the merged rows, then projected onto "
            + f"{sorted(set(paths) - set(train))}")

        for method in args.methods:
            ks = [None] if method in K_FREE else args.k
            for k in ks:
                cl = CLUSTERERS[method](**_opts(method, args))
                t0 = time.time()
                if method in SEQUENTIAL:
                    cl.fit(Xtr, k, lengths=len_tr)
                else:
                    cl.fit(Xtr, k)
                # Refused BEFORE anything is written. A K below --min-k cannot
                # carry a transition structure -- at K=1 every subject's table is
                # the single cell `0->0` = 1.0 -- so writing it would put a
                # column of a constant into the selection grid, where it costs
                # fits and can only dilute an FDR family. Skipped rather than
                # fatal, because the other columns in this cell are fine and a
                # 9-minute job should not be thrown away; the WARNING and the
                # non-zero exit at the end are what make it impossible to miss.
                if not args.min_k <= cl.k_found <= args.max_k:
                    extra = ""
                    if method in K_FREE:
                        ladder = " -> ".join(
                            f"q={t['quantile']:g}:K={t['k']}"
                            for t in getattr(cl, "search", []))
                        extra = (f"\n      searched {ladder}"
                                 f"\n      lower --meanshift-quantile below "
                                 f"{cl.params()['quantile']:g} to start finer")
                    log(f"  SKIPPED {column_name(method, emb, cl.k_found)}: "
                        f"K={cl.k_found} outside [{args.min_k}, {args.max_k}]"
                        + extra)
                    skipped.append({"atlas": atlas, "window_s": str(window_s),
                                    "method": method, "embedding": emb,
                                    "k": int(cl.k_found),
                                    "reason": f"K outside "
                                              f"[{args.min_k}, {args.max_k}]",
                                    "search": getattr(cl, "search", None)})
                    continue

                col = column_name(method, emb, cl.k_found)
                h = fit_hash(method, emb, cl.k_found, cl.params(), train,
                             len(Xtr), balanced=bool(args.balance_train))

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
    return entries, skipped


def _opts(method: str, args) -> dict:
    if method == "meanshift":
        return {"quantile": args.meanshift_quantile,
                "fit_rows": args.meanshift_fit_rows,
                "min_k": args.min_k, "max_k": args.max_k}
    if method == "hmm":
        return {"n_iter": args.hmm_iter}
    return {}


def check_grid(root: Path, args) -> pd.DataFrame:
    """One row per (atlas, aperture): what is there, before anything is fitted.

    The same idea as `transitions --check`, and here for the same reason: this
    stage's cost is the HMM, hours of it at K=27 across a full grid, and
    discovering a missing embedding or a missing cohort after three of those have
    run is the expensive way to find out.

    Nothing is written and nothing is fitted.
    """
    rows = []
    for atlas in args.atlas:
        for w in args.window_s:
            paths = cohort_paths(root, atlas, w)
            if not paths:
                rows.append({"atlas": atlas, "window_s": w, "ok": False,
                             "reason": "no latents (run stage 4)",
                             "cohorts": "-", "embeddings": "-", "existing": "-",
                             "rows": 0})
                continue
            train = [c for c in args.train if c in paths]
            have = [e for e in args.embeddings
                    if all(has_columns(p, EMBEDDINGS[e]) for p in paths.values())]
            missing = [e for e in args.embeddings if e not in have]
            existing = sorted({c for p in paths.values()
                               for c in _state_columns(p)})

            import pyarrow.parquet as pq
            n = sum(pq.ParquetFile(p).metadata.num_rows for p in paths.values())

            # The censor policy each cohort's latents were built under. Checked
            # HERE, one stage before `transitions --check` would catch it,
            # because by then the aperture has already cost an hour of stage 4.
            # This is the defect that let window_s=-1 be built uncensored while
            # every other aperture used `motion`.
            pol = {_meta(p, "censor_policy") for p in paths.values()}
            pol_s = ", ".join(sorted(str(x) for x in pol))

            if not train:
                reason = f"none of --train {args.train} is present"
            elif missing:
                reason = (f"embedding(s) {missing} absent -- rerun stage 4 "
                          f"without --no-umap, or drop them from --embeddings")
            elif len(pol) > 1:
                reason = (f"cohorts disagree on the censor policy ({pol_s}) -- "
                          f"they came from different stage 4 runs")
            else:
                reason = ""
            rows.append({"atlas": atlas, "window_s": w, "ok": not reason,
                         "reason": reason or "ready",
                         "cohorts": ",".join(sorted(paths)),
                         "embeddings": ",".join(have) or "-",
                         "censor": pol_s, "existing": len(existing), "rows": n})
    return pd.DataFrame(rows)


def _meta(path: Path, key: str):
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = md.get(key.encode())
    if raw is None:
        return None
    try:
        return json.loads(raw.decode())
    except Exception:                                            # noqa: BLE001
        return raw.decode()


def _state_columns(path: Path) -> list[str]:
    """State columns already in one latents file -- what a re-run would replace."""
    import re

    import pyarrow.parquet as pq

    return [n for n in pq.ParquetFile(path).schema_arrow.names
            if re.fullmatch(r"[A-Za-z]+_[a-z]+\d*_\d+", n)]


def report_check(df: pd.DataFrame, args) -> int:
    if df.empty:
        print("nothing in the grid -- check --atlas / --window-s")
        return 1
    print(f"\ngrid: {int(df['ok'].sum())}/{len(df)} (atlas, aperture) cell(s) "
          f"ready\n")
    print(df.to_string(index=False))
    planned = [column_name(m, e, k) for e in args.embeddings
               for m in args.methods for k in (args.k if m not in K_FREE else [0])]
    print(f"\nwould write {len(planned)} state column(s) per cell "
          f"(MeanShift's K is discovered, shown as 0 here):")
    print("  " + "  ".join(planned))
    print("\n`existing` counts state columns already in the file. A re-run "
          "REPLACES a column of the same name, which is what you want after "
          "changing a method's settings and not what you want otherwise.")
    # ACROSS cells, not just within one. Two apertures built under different
    # policies are each internally consistent and still not comparable, and
    # nothing downstream pools them -- so it has to be said here.
    pols = {p for p in df.loc[df["censor"] != "", "censor"] if p}
    if len(pols) > 1:
        print(f"\nWARNING: the grid spans more than one censor policy: "
              f"{sorted(pols)}.")
        print("         One aperture censored and another not is not a fair "
              "comparison of apertures.\n         Rebuild the odd one with "
              "`decompose --censor-policy <name>`.")
    if not df["ok"].all() or len(pols) > 1:
        bad = len(df) - int(df["ok"].sum())
        if bad:
            print(f"\n{bad} cell(s) are not ready.")
        return 1
    return 0


def run(args) -> int:
    root = Path(args.output_root) if args.output_root else _default_root()
    if not root.is_dir():
        raise SystemExit(f"output_root does not exist: {root}")
    log(f"output_root {root}")
    log(f"methods {args.methods}  embeddings {args.embeddings}  k {args.k}")
    log(f"fitted on {args.train}, applied to every cohort present")

    if args.check:
        return report_check(check_grid(root, args), args)

    entries, skipped = [], []
    for atlas in args.atlas:
        for w in args.window_s:
            log(f"atlas={atlas} window_s={w}")
            e, sk = run_one(root, atlas, w, args)
            entries += e
            skipped += sk

    if not entries:
        raise SystemExit("nothing was written -- check --atlas / --window-s "
                         "against what decompose produced")

    out = meta_dir(root) / "cluster" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"entries": entries, "skipped": skipped, "methods": args.methods,
         "embeddings": args.embeddings, "k": args.k, "train": args.train,
         "min_k": args.min_k, "max_k": args.max_k,
         "balance_train": bool(args.balance_train),
         "n_state_columns": len(entries), "n_skipped": len(skipped),
         "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, default=str))
    log(f"{len(entries)} state column(s) written -> {out.relative_to(root)}")

    if skipped:
        # Repeated at the end because the per-cell line scrolls past, and exiting
        # non-zero so a batch job that produced a short grid is not reported as a
        # clean success by sacct.
        print(f"\n{len(skipped)} state set(s) REFUSED for a degenerate K "
              f"(outside [{args.min_k}, {args.max_k}]):", flush=True)
        for sk in skipped:
            print(f"  {sk['atlas']:<14} {sk['window_s']:>4}s  {sk['method']}/"
                  f"{sk['embedding']}  K={sk['k']}", flush=True)
        print("Nothing was written for those. They are listed under `skipped` in "
              f"{out.relative_to(root)}.", flush=True)
        return 1
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
                   default=["30", "60", "120", "300", "-1"],
                   help="-1 is the activation aperture (one frame per row). A "
                        "cell with no latents is reported and skipped.")
    p.add_argument("--methods", nargs="+", default=list(CLUSTERERS),
                   choices=list(CLUSTERERS))
    p.add_argument("--embeddings", nargs="+", default=list(EMBEDDINGS),
                   choices=list(EMBEDDINGS))
    p.add_argument("--k", nargs="+", type=int, default=[8, 27],
                   help="for methods that need one; meanshift discovers it")
    p.add_argument("--train", nargs="+", default=DEFAULT_TRAIN,
                   help="cohorts a clusterer may be FITTED on, MERGED into one "
                        "fit; every cohort present is then labelled from it")
    p.add_argument("--balance-train", action="store_true",
                   help="subsample each training cohort to the smallest one's "
                        "row count, by whole subjects, so no cohort dominates "
                        "where the states are. Off by default: it discards data, "
                        "and the log prints each cohort's share either way.")
    p.add_argument("--meanshift-quantile", type=float, default=0.1,
                   help="STARTING bandwidth quantile for meanshift; smaller "
                        "finds more states. It is searched from here until K "
                        "lands between --min-k and --max-k, so this is a hint "
                        "rather than a setting.")
    p.add_argument("--min-k", type=int, default=STATE_K_BAND[0],
                   help="refuse to write a state set with fewer states than "
                        "this. K=1 has one cell, `0->0`=1.0 for every subject, "
                        "so it is a constant and not a weak predictor.")
    p.add_argument("--max-k", type=int, default=STATE_K_BAND[1],
                   help="upper end of the band meanshift searches for, and the "
                        "most states any method may write.")
    p.add_argument("--meanshift-fit-rows", type=int, default=50_000)
    p.add_argument("--hmm-iter", type=int, default=50)
    p.add_argument("--output-root")
    p.add_argument("--check", action="store_true",
                   help="report what each (atlas, aperture) has and what would "
                        "be written, fit nothing. The HMM is hours at K=27 "
                        "across a full grid; find the gaps first.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
