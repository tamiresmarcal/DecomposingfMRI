#!/usr/bin/env python3
"""Stage 4b -- every brain-state definition, on latents that already exist.

    fmri-decomp cluster --atlas yeo7 --window-s 30 60 120 300
    fmri-decomp cluster --methods threshold meanshift hmm1 hmm2 \\
                        --embeddings pca3 umap3 --k 8 10 27
    fmri-decomp cluster --atlas networks --window-s -1 \\
                        --methods hmm1 hmm2 --embeddings pca3 raw14

THE ONLY PLACE A STATE IS DEFINED
---------------------------------
Stage 4 produces embeddings; this stage turns them into labels. ALL of them,
threshold included -- `--bins` used to do the thresholding inside `decompose`
and no longer exists, because one method sitting in a different stage from the
others is the wrong seam. It got re-fitted on every stage 4 re-run while the
others did not, and it wrote columns with no entry in the `clusterers`
provenance block, which is why `transitions.n_states_for` still carries a
fallback that reads K out of a column name.

A latents file holds `pca0/3..pca2/3` and `umap0/3..umap2/3`, and -- where
`decompose --passthrough-features` ran -- the named input features themselves as
`raw/<name>`. A new way of defining states needs none of them refitted: it reads
those columns and appends label columns to the same file. Adding a sixth state
definition must not mean redoing the five that already work.

EMBEDDINGS, AND WHY ONE OF THEM IS RAGGED
-----------------------------------------
`pca3` and `umap3` exist in every cell. `raw<N>` is the N named, scaled features
and exists only where passthrough ran, which is the activation aperture on the
atlases small enough for a full-covariance fit -- `raw7` on yeo7, `raw14` on
networks. That is deliberate: asking for it across the grid is fine, and the
cells without it SKIP that embedding and say so rather than failing.

At full rank a PCA is an orthonormal rotation (round-trip error ~1e-14), so
`raw14` and `pca14` on a 14-feature atlas are the same space and a
full-covariance HMM on either is the same model. The raw columns earn their
place by being READABLE: a state mean over `raw/AM .. raw/WM` is a network
pattern, where the same mean over `pca0/14 ..` is a point nobody can interpret
until the rotation is undone. `fmri-decomp state-means` undoes it either way.

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
    hmm1        K is specified. Diagonal covariance, 50 EM iterations, ONE
                initialisation. The incumbent, deliberately frozen so that
                hmm2 has a fixed thing to be compared against.
    hmm2        The same model family, fitted the way van der Meer et al. 2020
                fit theirs: FULL covariance, 500 iterations, 15 restarts, and
                only a restart expressing all K states may win. Two orders of
                magnitude more expensive than hmm1 and the reason this stage
                needs its walltime measured before a full grid.

Both are fitted with per-(task, sub) sequence lengths, so neither models a
transition across a subject boundary.

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
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import (RAW_PREFIX, STATE_COLUMN_RE, STATE_K_BAND,
                 latents_root, meta_dir)

EMBEDDINGS = {
    "pca3": ["pca0/3", "pca1/3", "pca2/3"],
    "umap3": ["umap0/3", "umap1/3", "umap2/3"],
}

# FAMILY names, which is what a command should normally say.
#
# `pca` and `umap` resolve to exactly one member each: only pca3 and umap3 are
# clustering embeddings. `decompose --n-latents 3 14` also writes a
# 14-component PCA, but that exists so `raw14` is a rotation of a FULL-RANK one
# -- which is what lets `state-means` invert it -- and `pca14` is not a valid
# embedding name. So there is no ambiguity to resolve for those two.
#
# `raw` is the one that needs resolving, and the only reason this exists. Its
# width is a property of the atlas, not of the command -- raw7 on yeo7, raw14
# on networks, absent on harvardoxford -- so a command naming `raw14` is
# carrying a fact about the data that the cell can simply be asked. Forgetting
# to name it was a silent omission: the run succeeded and the named-network arm
# was just missing from the grid.
EMBEDDING_FAMILIES = {"pca": ["pca3"], "umap": ["umap3"], "raw": None}

DEFAULT_EMBEDDINGS = ["pca", "umap", "raw"]

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

    @staticmethod
    def accepts_k(k: int) -> bool:
        """K must be a perfect cube -- 8, 27, 64 -- because K = bins**3.

        Declared rather than raised. The default grid carries K=10 so hmm1 and
        hmm2 can be compared at the paper's choice, and 10 is not a cube; a
        SystemExit here would kill the whole job over a combination nobody asked
        for. The caller records it as not-applicable, like the other exclusions.
        """
        return k is not None and round(k ** (1 / 3)) ** 3 == k

    def fit(self, X, k, rng=None):
        from sklearn.preprocessing import KBinsDiscretizer

        # The only clusterer here that is not dimension-agnostic: K = bins**3
        # and `labels` ravels exactly three axes. On a 14-D embedding the ravel
        # raised a bare numpy shape error naming neither this class nor the
        # embedding, which is a poor way to learn that the method does not apply.
        if X.shape[1] != 3:
            raise SystemExit(
                f"{self.name} is defined on a 3-D embedding only; this one has "
                f"{X.shape[1]} dimension(s).\n"
                f"  K = bins**3 over three quantile axes is the definition, so "
                f"there is no {X.shape[1]}-D version of it.\n"
                f"  Drop `threshold` from --methods for this embedding, or use "
                f"--embeddings pca3 umap3.")
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


def _import_gaussian_hmm():
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise SystemExit(
            "hmmlearn is not installed in this environment.\n"
            "containers/stage45.def pins hmmlearn==0.3.3 -- use that image, "
            "or drop `hmm1`/`hmm2` from --methods.")
    return GaussianHMM


def _em_diagnostics(model, n_iter_cap: int) -> dict:
    """What EM actually did, as opposed to what it was asked to do.

    NOT `monitor_.converged`, which is a trap -- hmmlearn returns True when EM
    merely runs out of iterations:

        return (self.iter == self.n_iter or
                (len(self.history) >= 2 and
                 self.history[-1] - self.history[-2] < self.tol))

    So a fit that stopped at the cap with the likelihood still climbing reports
    `converged=True`. `stopped_early` is the honest question: did it finish
    before the cap, which is the only way the cap was not the binding
    constraint.
    """
    h = list(getattr(model.monitor_, "history", []))
    return {"n_iter_ran": int(model.monitor_.iter),
            "n_iter_cap": int(n_iter_cap),
            "stopped_early": bool(model.monitor_.iter < n_iter_cap),
            "final_delta": float(h[-1] - h[-2]) if len(h) >= 2 else None}


class HMM1Cluster:
    """Gaussian HMM, diagonal covariance, one fit. K is specified.

    THE INCUMBENT, DELIBERATELY FROZEN. Everything about the fit is as it was
    when this was called `HMM`: diagonal covariance, 50 EM iterations, a single
    initialisation at `random_state=0`. `params()` is unchanged too, which is
    the point -- it feeds `fit_hash`, so a column written now carries the same
    hash as the same column written before the rename, and the two are
    comparable across output trees.

    `HMM2Cluster` is what varies the estimator. Keeping this one fixed is what
    makes that comparison attributable.

    `lengths` is not optional. Without it hmmlearn treats the stacked rows as
    ONE sequence and learns a transition from the last window of each subject
    to the first window of the next -- a transition between two different
    people, which is not a thing.
    """

    name = "HMM1"

    def __init__(self, n_iter=50, covariance_type="diag"):
        self.n_iter, self.covariance_type = n_iter, covariance_type
        self.diagnostics = {}

    def fit(self, X, k, rng=None, lengths=None):
        GaussianHMM = _import_gaussian_hmm()
        self.hmm = GaussianHMM(n_components=k, covariance_type=self.covariance_type,
                               n_iter=self.n_iter, random_state=0)
        self.hmm.fit(X, lengths=lengths)
        self.k_found = k
        self.diagnostics = {**_em_diagnostics(self.hmm, self.n_iter),
                            "loglik": float(self.hmm.score(X, lengths)),
                            "states_expressed_in_train":
                                int(np.unique(self.hmm.predict(X, lengths)).size),
                            **_information_criteria(self.hmm, X, lengths)}
        return self

    def labels(self, X, lengths=None):
        return self.hmm.predict(X, lengths=lengths)

    def params(self):
        # SETTINGS ONLY, and never an outcome. This dict is hashed into
        # `fit_hash`; putting `n_iter_ran` in here would mean two identical
        # configurations hashed differently because one happened to converge
        # sooner, which is not what a configuration hash can mean. Outcomes go
        # to `fit_diagnostics`, beside the hash and outside it.
        return {"n_iter": self.n_iter, "covariance_type": self.covariance_type,
                "note": "the HMM fits its own transmat_ while defining the "
                        "states; stage 5a counts per-subject transitions from "
                        "the Viterbi path, which is a different quantity"}


def _fit_one_restart(X, lengths, k: int, covariance_type: str, n_iter: int,
                     seed: int) -> dict:
    """One HMM2 initialisation. Module level so a loky worker can pickle it.

    Returns the fitted model alongside its score rather than refitting the
    winner in the parent: a 500-iteration fit is the whole cost here, and the
    model itself is tiny (K x d x d covariances -- a few thousand floats at
    K=27, d=14), so shipping it back through the pickle is free by comparison.
    """
    GaussianHMM = _import_gaussian_hmm()
    m = GaussianHMM(n_components=k, covariance_type=covariance_type,
                    n_iter=n_iter, random_state=seed)
    try:
        m.fit(X, lengths=lengths)
        return {"seed": seed, "loglik": float(m.score(X, lengths)),
                "states_expressed": int(np.unique(m.predict(X, lengths)).size),
                "error": None, "model": m}
    except (ValueError, np.linalg.LinAlgError) as e:
        # A full-covariance state with too few rows assigned to it gives a
        # singular covariance. That is a legitimate outcome of asking for more
        # states than the data supports -- the same thing their K>=12 runs hit
        # -- so it costs this restart and not the run.
        return {"seed": seed, "loglik": None, "states_expressed": 0,
                "error": f"{type(e).__name__}: {e}", "model": None}


def _default_hmm2_jobs() -> int:
    """How many restarts to fit at once.

    SLURM_CPUS_PER_TASK, not os.cpu_count(): inside a job the latter reports
    the NODE's core count, so -1 would launch dozens of workers on an
    eight-core allocation and thrash.
    """
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        return max(1, int(n))
    except (TypeError, ValueError):
        return 1


class HMM2Cluster:
    """Gaussian HMM, FULL covariance, 500 iterations, N restarts, all-K-or-skip.

    The estimator van der Meer et al. 2020 (Nat Commun 11:5004) used, as their
    released code sets it up (brain-modelling-group/MovieBrainDynamics,
    `Step1_create_dirs_and_run_hmm.m:338-348`):

        options.order = 0;         % no autoregressive components
        options.zeromean = 0;      % model the mean
        options.covtype = 'full';  % full covariance matrix
        options.cyc = 500;
        HMMREPS = 15;              % re-run the HMM analysis 15 times

    `order = 0` with `zeromean = 0` is a plain Gaussian HMM despite the toolbox
    being called HMM-MAR, which is why hmmlearn's GaussianHMM is the right
    family and the remaining difference is the estimator rather than the model.

    Three things it does that HMM1 does not:

    FULL COVARIANCE. Diagonal asserts the axes are uncorrelated WITHIN each
    state. PCA decorrelates globally, not per state, so that is an assumption
    and not a free one -- measured at ~7,200 nats of log-likelihood on 14-D
    network-like data. On a `raw<N>` embedding it matters more again, since
    those columns are not even globally orthogonal.

    RESTARTS. EM is non-convex and lands wherever its initialisation leads. Two
    fits of the same model on the same data in rotated bases agreed on only 73%
    of the Viterbi path, which is the instability their 15 reps exist to
    absorb. Each restart gets its own seed; the best SURVIVING log-likelihood
    wins, and the spread across restarts is recorded -- that spread is the
    direct measure of how much HMM1's single initialisation was gambling.

    ALL K STATES EXPRESSED, OR NOTHING. Their `Step1b_Check_for_expressions.m`
    keeps only the runs where every state appears in the Viterbi path
    (`numel(unique(this_path)) == 10`) and discards the rest; it is why they
    capped K at 10. A dead state is not a harmless one: it is an all-zero
    occupancy column plus a zero row and column in every subject's K x K
    matrix, so it adds dimensionality and no signal, and dilutes the FDR
    family. The check here is on the TRAINING block, where the states are
    defined; `states_used` per cohort is already reported separately for the
    projected cohorts.

    NOT copied from them, deliberately: `smoothdata` (a moving average over each
    network, absent from the paper's methods, which shapes the dwell times the
    analysis then reports), and their per-condition `T` (their `scans_sum`
    accumulates across subjects, so their HMM does learn transitions across
    participant boundaries -- `lengths` here does not).
    """

    name = "HMM2"

    def __init__(self, n_iter=500, n_restarts=15, covariance_type="full",
                 min_k=STATE_K_BAND[0], max_k=STATE_K_BAND[1], n_jobs=None):
        self.n_iter, self.n_restarts = n_iter, n_restarts
        self.covariance_type = covariance_type
        self.min_k, self.max_k = min_k, max_k
        # Execution detail, NOT a model setting -- see params(). It must never
        # change a label, only how long they take to produce.
        self.n_jobs = _default_hmm2_jobs() if n_jobs is None else int(n_jobs)
        self.diagnostics = {}

    def _run_restarts(self, X, lengths, k) -> list[dict]:
        """Every initialisation, in parallel where there are cores for it.

        THE RESTARTS ARE THE PARALLEL AXIS, and for a while nothing used it. A
        single hmmlearn fit is sequential over time -- forward-backward cannot
        be split -- so giving one fit eight BLAS threads buys nothing; measured
        at 4.37 s on one thread against 4.70 s on eight, i.e. slightly worse
        from contention. The restarts, by contrast, are completely independent.
        On a measured cell (691,434 training rows, K=27) the sequential loop
        projects to ~29 h and eight-way restarts to ~4 h, for the same
        core-hours.

        `inner_max_num_threads=1` is the other half: without it each of eight
        workers would start its own eight-thread BLAS pool on eight cores.
        """
        work = [(X, lengths, k, self.covariance_type, self.n_iter, seed)
                for seed in range(self.n_restarts)]
        if self.n_jobs <= 1 or self.n_restarts == 1:
            return [_fit_one_restart(*w) for w in work]
        from joblib import Parallel, delayed, parallel_backend

        n_jobs = min(self.n_jobs, self.n_restarts)
        with parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
            return list(Parallel()(delayed(_fit_one_restart)(*w) for w in work))

    def fit(self, X, k, rng=None, lengths=None):
        attempts = self._run_restarts(X, lengths, k)
        valid = [a for a in attempts
                 if a["states_expressed"] == k and a["loglik"] is not None]
        # max over (loglik, -seed), not "first one that beats the record". With
        # the restarts running in parallel their completion order is not their
        # seed order, so a tie broken by arrival would make the winner -- and
        # therefore every label written -- depend on scheduling. Lowest seed on
        # an exact tie reproduces what the sequential loop did.
        best_attempt = (max(valid, key=lambda a: (a["loglik"], -a["seed"]))
                        if valid else None)
        best = best_attempt["model"] if best_attempt else None
        best_ll = best_attempt["loglik"] if best_attempt else -np.inf
        lls = [a["loglik"] for a in valid]
        # The models go no further: they are not JSON, and the one that won is
        # already held as `self.hmm`.
        self.attempts = [{x: y for x, y in a.items() if x != "model"}
                         for a in attempts]
        self.diagnostics = {
            "n_restarts": self.n_restarts,
            "n_valid_restarts": len(valid),
            "n_jobs": self.n_jobs,
            "states_expressed_per_restart": [a["states_expressed"] for a in attempts],
            "loglik_spread_across_restarts":
                float(max(lls) - min(lls)) if len(lls) > 1 else 0.0,
            "loglik_best": float(best_ll) if np.isfinite(best_ll) else None,
            "best_seed": best_attempt["seed"] if best_attempt else None,
            "restart_errors": [a["error"] for a in attempts if a["error"]][:3],
        }
        if best is None:
            # k_found is set to the best expressed count so the caller's existing
            # band check (`min_k <= k_found <= max_k`) rejects it and records the
            # reason, rather than this class inventing a second refusal path.
            self.hmm = None
            self.k_found = max((a["states_expressed"] for a in attempts), default=0)
            self.diagnostics["refused"] = (
                f"no restart expressed all {k} states in {self.n_restarts} "
                f"attempt(s); best was {self.k_found}")
            return self
        self.hmm = best
        self.k_found = k
        self.diagnostics.update(_em_diagnostics(best, self.n_iter))
        self.diagnostics["loglik"] = float(best_ll)
        self.diagnostics["states_expressed_in_train"] = k
        self.diagnostics.update(_information_criteria(best, X, lengths))
        return self

    def labels(self, X, lengths=None):
        if self.hmm is None:
            raise RuntimeError("HMM2 refused this fit; see diagnostics['refused']")
        return self.hmm.predict(X, lengths=lengths)

    def params(self):
        # Settings only -- see HMM1Cluster.params. The restart SEEDS are part of
        # the configuration (range(n_restarts), so reproducible from the count);
        # which one won is an outcome and lives in fit_diagnostics.
        #
        # `n_jobs` is DELIBERATELY ABSENT. It changes how long the fit takes and
        # nothing else: the same labels come out whether the restarts ran one at
        # a time or eight at a time, so a run on 8 cores and a run on 1 must
        # carry the SAME fit_hash or the two stop being comparable for no
        # reason. It is recorded in fit_diagnostics instead.
        return {"n_iter": self.n_iter, "n_restarts": self.n_restarts,
                "covariance_type": self.covariance_type,
                "restart_seeds": f"range({self.n_restarts})",
                "select_by": "highest log-likelihood among restarts expressing "
                             "all K states",
                "note": "all-K-expressed filter after "
                        "Step1b_Check_for_expressions.m in "
                        "brain-modelling-group/MovieBrainDynamics"}


def _information_criteria(model, X, lengths) -> dict:
    """AIC and BIC, RECORDED AND NOT ACTED ON.

    The paper selects K by AIC; this pipeline selects state sets by out-of-fold
    phenotype prediction in stage 5b, which is a different and deliberate
    choice. Recording both means the two can be compared after the fact -- "the
    predictively best K was also the AIC-best K", or that it was not -- without
    either criterion quietly becoming the selector.

    hmmlearn exposes these only for EM fits, where a maximised likelihood
    genuinely exists. Their HMM-MAR fit is variational Bayes, which yields a
    free energy rather than a maximised likelihood, so "AIC" there is already
    loose -- and consistent with that, no AIC appears anywhere in their code.
    """
    out = {}
    for name in ("aic", "bic"):
        try:
            out[name] = float(getattr(model, name)(X, lengths))
        except Exception:
            out[name] = None
    return out


CLUSTERERS = {
    "threshold": Threshold,
    "meanshift": MeanShiftCluster,
    "hmm1": HMM1Cluster,
    "hmm2": HMM2Cluster,
}
# Methods that discover K rather than being given one.
K_FREE = {"meanshift"}
SEQUENTIAL = {"hmm1", "hmm2"}      # need per-subject sequence lengths
THREE_D_ONLY = {"threshold"}       # K = bins**3 over exactly three axes

# Clusterers that may be fitted on a `raw<N>` embedding -- the named, scaled
# input features, which are correlated with one another and on a different
# number of axes than pca3/umap3.
#
# MeanShift is excluded, and not because it would crash. It works on Euclidean
# distance, so it is sensitive to correlation and to how variance is spread
# across the axes in a way the HMMs are not: the bandwidth quantile that finds
# 4 states in a 3-D embedding has no reason to find anything comparable in 14
# correlated dimensions, so the two would not be the same method measured on
# two inputs. Threshold is excluded by THREE_D_ONLY.
#
# HMM1 IS allowed here even though diagonal covariance on correlated features
# is mis-specified. That pairing is the cleanest isolation of the covariance
# question in the whole grid -- HMM1_raw14 against HMM2_raw14 differ by
# covariance type, iterations and restarts on identical input -- and an expected
# result is worth having on the record rather than assumed.
RAW_CAPABLE = {"hmm1", "hmm2"}


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


RAW_EMB_RE = re.compile(r"^raw(\d+)$")


def resolve_families(requested: list[str], paths: dict) -> list[str]:
    """Family names -> the concrete embeddings present in THIS cell.

    `raw` becomes whichever `raw<N>` the cohorts actually carry, read from the
    files rather than assumed, and becomes nothing where passthrough never ran.
    Exact names pass through untouched, so an existing command keeps working
    and a request for one specific width is still a request for that width.

    Order is preserved and duplicates dropped, so `--embeddings pca raw pca3`
    is `pca3, raw<N>` and not `pca3, raw<N>, pca3`.
    """
    out: list[str] = []
    for item in requested:
        if item not in EMBEDDING_FAMILIES:
            embedding_spec(item)                       # validate or raise
            out.append(item)
            continue
        members = EMBEDDING_FAMILIES[item]
        if members is not None:
            out += members
            continue
        # `raw`: ask the cell. Every cohort must agree -- a fit on one cohort's
        # matrix projected onto another's would silently relabel the axes if
        # the widths differed.
        widths = {len(raw_columns(pth)) for pth in paths.values()}
        widths.discard(0)
        if len(widths) > 1:
            raise SystemExit(
                f"the cohorts in this cell carry different numbers of raw "
                f"features ({sorted(widths)}), so `--embeddings raw` cannot "
                f"name one. Re-run `decompose --passthrough-features` for the "
                f"whole cell, or ask for an exact raw<N>.")
        out += [f"raw{w}" for w in widths]
    return list(dict.fromkeys(out))


def embedding_spec(emb: str) -> int | None:
    """`None` for a fixed embedding; the expected feature count for `raw<N>`.

    Validates the NAME without touching a file, so argparse can reject a typo
    before a job is submitted. `choices=` cannot do this: the raw embeddings are
    named by how many features the atlas has, so the valid set is not knowable
    until the latents exist.
    """
    if emb in EMBEDDINGS:
        return None
    m = RAW_EMB_RE.match(emb)
    if not m:
        raise SystemExit(
            f"unknown embedding {emb!r}. Choose a FAMILY -- "
            f"{sorted(EMBEDDING_FAMILIES)} -- which is resolved against each "
            f"cell, or an exact name: {sorted(EMBEDDINGS)}, or raw<N>, the N "
            f"named scaled features written by `decompose "
            f"--passthrough-features` (raw7 for yeo7, raw14 for networks).")
    n = int(m.group(1))
    if n < 2:
        raise SystemExit(f"{emb!r}: a clusterer needs at least 2 dimensions.")
    return n


def raw_columns(path: Path) -> list[str]:
    """The `raw/<name>` columns in this file, in the order decompose wrote them.

    Sorted, not schema order, so every cohort in a cell presents its features in
    the same order -- the fit is on one cohort's matrix and the projection on
    another's, and a column permutation between them would silently relabel the
    axes.
    """
    import pyarrow.parquet as pq

    return sorted(c for c in pq.ParquetFile(path).schema_arrow.names
                  if c.startswith(RAW_PREFIX))


def embedding_columns(emb: str, paths: dict) -> list[str] | None:
    """Resolve `emb` to a column list against the files in one cell.

    Fixed embeddings are a constant. `raw<N>` is discovered, and only accepted
    when EVERY cohort in the cell offers the same N columns under the same
    names: a fit on 14 features projected onto a cohort carrying 13 of them is
    not a projection, it is a different model.
    """
    want = embedding_spec(emb)
    if want is None:
        cols = EMBEDDINGS[emb]
        # Presence is checked for the fixed embeddings as well, not assumed.
        # Returning the constant unconditionally made `check_grid` report a cell
        # as ready when umap3 was absent from it -- the one thing that function
        # exists to catch.
        if not paths or not all(has_columns(p, cols) for p in paths.values()):
            return None
        return cols
    per_cohort = {c: raw_columns(path) for c, path in paths.items()}
    first = next(iter(per_cohort.values()), [])
    if len(first) != want or any(v != first for v in per_cohort.values()):
        return None
    return first


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


# Constructor options that change how long a fit takes and nothing about its
# result, so they must not enter the cache key -- the same reasoning that keeps
# `n_jobs` out of `fit_hash` (see HMM2Cluster.params).
_OPTS_NOT_IN_RECIPE = frozenset({"n_jobs"})


def recipe_hash(method: str, embedding: str, k, opts: dict, train: list[str],
                n_rows: int, balanced: bool = False) -> str:
    """What was ASKED FOR, as opposed to what came out. The cache key.

    `fit_hash` describes the FITTED model: it carries `cl.params()`, which for
    MeanShift reports the discovered bandwidth and the quantile the search
    settled on, and for Threshold reports bins_per_axis. Those do not exist
    until the fit has run, so `fit_hash` cannot be used to look a fit UP --
    only to describe one afterwards.
    This is the other half: the method, the embedding, the K asked for, the
    constructor options, and the training rows. All known in advance, for every
    method. Same recipe over the same rows -> the same fit, so this is what the
    saved clusterer is filed under.
    """
    payload = json.dumps(
        {"method": method, "embedding": embedding, "k": k,
         "opts": {k2: v for k2, v in sorted(opts.items())
                  if k2 not in _OPTS_NOT_IN_RECIPE},
         "train": sorted(train), "n_train_rows": n_rows,
         "balanced": balanced},
        sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cluster_libs() -> dict[str, str]:
    """What a pickled clusterer was produced by, for the staleness guard."""
    import numpy

    libs = {"numpy": numpy.__version__}
    for mod in ("sklearn", "hmmlearn"):
        try:
            libs[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return libs


def clusterer_cache_path(root: Path, atlas: str, window_s, h: str) -> Path:
    return (Path(root) / "meta" / "clusterers"
            / f"atlas-{atlas}_window-{window_s}_{h}.joblib")


def save_clusterer(path: Path, cl, meta: dict, log=print) -> None:
    """Persist a fitted clusterer so a later cohort can be labelled, not refit.

    Best effort: a cell that cannot write the cache has still done the work and
    must not fail for being unable to save a shortcut.
    """
    try:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        joblib.dump({"clusterer": cl, "libs": _cluster_libs(), **meta}, tmp)
        os.replace(tmp, path)
    except Exception as e:                                       # noqa: BLE001
        log(f"      (could not cache the fit: {type(e).__name__}: {e})")


def load_clusterer(path: Path, model_hash: str | None, log=print):
    """A saved clusterer, if it is safe to apply here. Else None.

    Two guards, both refusing rather than approximating:

    * `model_hash` of the latents it was fitted on. `fit_hash` does not carry
      it -- it describes the CLUSTERING, not the embedding underneath -- so
      without this check a saved fit could be applied to latents from a
      different `decompose` run that happened to have the same row count.
    * the libraries it was pickled under. A pickled hmmlearn or sklearn
      estimator is not guaranteed to behave across versions, and silently
      applying one that does not is worse than spending the fit again.
    """
    if not path.exists():
        return None
    try:
        import joblib

        blob = joblib.load(path)
    except Exception as e:                                       # noqa: BLE001
        log(f"      (ignoring unreadable cache {path.name}: {type(e).__name__})")
        return None
    if model_hash is not None and blob.get("model_hash") != model_hash:
        log(f"      (cached fit was made on model_hash "
            f"{blob.get('model_hash')}, these latents are {model_hash} -- "
            f"refitting)")
        return None
    want, have = _cluster_libs(), blob.get("libs") or {}
    drift = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
    if drift:
        log(f"      (cached fit was pickled under different libraries {drift} "
            f"-- refitting rather than trusting it)")
        return None
    return blob.get("clusterer")


def blob_fit_hash(path: Path) -> str | None:
    """The `fit_hash` recorded beside a cached clusterer, for the reload check."""
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path).get("fit_hash")
    except Exception:                                            # noqa: BLE001
        return None


def column_fit_hash(path: Path, col: str) -> str | None:
    """The `fit_hash` a latents file already records for one state column."""
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    block = json.loads((md.get(b"clusterers") or b"{}").decode())
    entry = block.get(col) or {}
    return entry.get("fit_hash")


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
def _require_embeddings(paths: dict, embeddings: list[str], atlas: str,
                        window_s) -> None:
    """Every requested embedding present in every cohort, or stop.

    A hard failure rather than a skip, and the same reasoning as stage 4's UMAP
    check: a run that writes 5 of the 10 state sets it was asked for and reports
    success is one whose gap surfaces much later, as `umap3 not in the latents`
    at the point someone is reading results.
    """
    gaps = {}
    for emb in embeddings:
        # `raw<N>` is exempt, and deliberately so: it exists only where
        # `decompose --passthrough-features` ran, which is the activation
        # aperture on the small atlases. The grid is RAGGED for it by design, so
        # a missing raw embedding is a skip recorded in the report, not a
        # failure -- see the main loop. pca3/umap3 stay mandatory, because those
        # are written for every cell and their absence means stage 4 is
        # incomplete.
        if embedding_spec(emb) is not None:
            continue
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
    have = [e for e in embeddings if e not in gaps]

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
    _require_embeddings(paths, resolve_families(args.embeddings, paths),
                        atlas, window_s)

    # Which `decompose` fit produced these embeddings. Carried into the fit
    # cache and checked on the way back out: `fit_hash` describes the
    # CLUSTERING, not the embedding underneath, so without this a cached fit
    # could be applied to latents from a different decompose run that happened
    # to have the same row count.
    from .decompose import latents_model_hash

    cell_model_hash = latents_model_hash(paths[train[0]])

    # WHICH COHORTS GET LABELLED, said rather than inferred. `train + project`,
    # exactly as `decompose` says it -- two stages with the same train/project
    # semantics should use the same vocabulary, and a stage that silently acts
    # on whatever is in the directory cannot be asked for less.
    # `None` (flag absent) and `[]` (flag given with no values) are different
    # answers and must not share a branch: absent means "every cohort here",
    # empty means "none beyond --train". Testing truthiness collapsed them, so
    # `--project` with no values labelled EVERYTHING -- the exact opposite of
    # what it reads as.
    project = getattr(args, "project", None)
    if project is not None:
        unknown = [c for c in project if c not in paths]
        if unknown:
            raise SystemExit(
                f"--project names cohort(s) with no latents at atlas={atlas} "
                f"window_s={window_s}: {unknown}\n"
                f"  this cell has {sorted(paths)}\n"
                f"  run `decompose --project {' '.join(unknown)}` first, or "
                f"fix the spelling.")
        label = [c for c in paths if c in set(train) | set(project)]
    else:
        label = list(paths)
    if len(label) < len(paths):
        log(f"  labelling {label}; not asked for: "
            f"{[c for c in paths if c not in label]}")

    # Families resolved against THIS cell, because `raw`'s width is a property
    # of the atlas and not of the command.
    embeddings = resolve_families(args.embeddings, paths)
    if embeddings != list(args.embeddings):
        log(f"  embeddings {list(args.embeddings)} -> {embeddings}")
    if not embeddings:
        log(f"  atlas={atlas} window_s={window_s}: none of "
            f"{list(args.embeddings)} resolves to anything here, skipped")
        return [], [{"atlas": atlas, "window_s": str(window_s), "kind": "n/a",
                     "method": "-", "embedding": ",".join(args.embeddings),
                     "k": 0, "reason": "no requested embedding exists in this "
                                       "cell", "search": None}]
    require_methods_are_reachable(args.methods, embeddings, atlas, window_s)

    entries, skipped = [], []
    for emb in embeddings:
        cols = embedding_columns(emb, paths)
        if cols is None:
            # Only reachable for a raw embedding: _require_embeddings above has
            # already raised for an absent pca3/umap3, which stays fatal because
            # those are written for every cell.
            want = embedding_spec(emb)
            have = {c: len(raw_columns(pth)) for c, pth in paths.items()}
            reason = (f"no {emb} columns ({want} {RAW_PREFIX}* wanted per "
                      f"cohort, found {have})")
            log(f"  SKIPPED {emb}: {reason}"
                f"\n      `decompose --passthrough-features` writes them; it is "
                f"meant for --source activation on a small atlas.")
            skipped.append({"atlas": atlas, "window_s": str(window_s),
                            "kind": "n/a", "method": "-", "embedding": emb,
                            "k": 0, "reason": reason, "search": None})
            continue
        frames = {c: read_embedding(p, cols) for c, p in paths.items()}

        Xtr, len_tr, share = _training_block(frames, train, cols, args)
        how = "balanced" if args.balance_train else "pooled as-is"
        log(f"  {emb}: {len(cols)}-D, one fit on {len(Xtr):,} row(s) from "
            f"{train} ({how}); "
            + "  ".join(f"{c} {share[c]:.0%}" for c in train)
            + " of the merged rows, then projected onto "
            + f"{sorted(set(paths) - set(train))}")

        methods = _methods_for(emb, args, atlas, window_s, skipped)
        for method in methods:
            ks = [None] if method in K_FREE else _ks_for(
                method, args, emb, atlas, window_s, skipped)
            for k in ks:
                cl = CLUSTERERS[method](**_opts(method, args))
                t0 = time.time()
                # Look the fit up BEFORE doing it. For the HMMs every component
                # of `fit_hash` is known in advance -- k is given and `params()`
                # is settings-only -- so a cell that has already been fitted on
                # these exact training rows can be labelled rather than refitted.
                # That is the difference between adding a cohort in minutes and
                # adding it in a day.
                recipe = recipe_hash(method, emb, k, _opts(method, args),
                                     train, len(Xtr),
                                     balanced=bool(args.balance_train))
                cache = clusterer_cache_path(root, atlas, window_s, recipe)
                reused = (None if args.refit
                          else load_clusterer(cache, cell_model_hash, log=log))
                if reused is not None:
                    cl = reused
                    log(f"  {column_name(method, emb, cl.k_found)}"
                        f"{'':<12} reusing the cached fit {recipe} "
                        f"-- labelling only, no refit")
                elif method in SEQUENTIAL:
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
                                    "kind": "refused", "method": method,
                                    "embedding": emb, "k": int(cl.k_found),
                                    "reason": getattr(cl, "diagnostics", {}).get(
                                        "refused")
                                    or f"K outside [{args.min_k}, {args.max_k}]",
                                    "search": getattr(cl, "search", None),
                                    "diagnostics": getattr(cl, "diagnostics", None)})
                    continue

                col = column_name(method, emb, cl.k_found)
                h = fit_hash(method, emb, cl.k_found, cl.params(), train,
                             len(Xtr), balanced=bool(args.balance_train))
                if reused is None:
                    # Filed under the RECIPE, so the next run can find it, and
                    # carrying the fit_hash so a reload can be checked against
                    # what the fit actually produced.
                    save_clusterer(
                        cache, cl,
                        {"model_hash": cell_model_hash, "fit_hash": h,
                         "recipe_hash": recipe, "method": method,
                         "embedding": emb, "k": int(cl.k_found),
                         "train": sorted(train), "n_train_rows": len(Xtr),
                         "column": col}, log=log)
                elif blob_fit_hash(cache) not in (None, h):
                    # The reloaded object produced a different fit_hash from
                    # the one saved beside it. That should be impossible --
                    # params() and k_found come out of the pickle -- so it means
                    # the pickle no longer behaves as it did, which is exactly
                    # what must not pass silently.
                    raise SystemExit(
                        f"{col}: the cached fit {recipe} reloaded to a "
                        f"DIFFERENT model.\n"
                        f"  saved fit_hash {blob_fit_hash(cache)}\n"
                        f"  reloaded as    {h}\n"
                        f"  Delete {cache} and re-run with --refit.")

                written, kept = {}, []
                for cohort, f in frames.items():
                    if cohort not in label:
                        continue
                    # A cohort already carrying THIS fit is left alone. Not an
                    # optimisation: append_columns rewrites the whole file, so
                    # relabelling a cohort that needs nothing would churn every
                    # file in the cell on every run.
                    if (not args.refit
                            and column_fit_hash(paths[cohort], col) == h):
                        kept.append(cohort)
                        continue
                    X = f[cols].to_numpy(float)
                    lab = (cl.labels(X, lengths=_seq_lengths(f))
                           if method in SEQUENTIAL else cl.labels(X))
                    append_columns(paths[cohort], {col: lab},
                                   {col: {"method": method, "embedding": emb,
                                          "embedding_columns": cols,
                                          "k": int(cl.k_found), "fit_hash": h,
                                          "params": cl.params(),
                                          # Beside the hash, NOT inside it: an
                                          # outcome, not a setting. See
                                          # HMM1Cluster.params.
                                          "fit_diagnostics":
                                              getattr(cl, "diagnostics", None) or None,
                                          "train_cohorts": train,
                                          "n_train_rows": int(len(Xtr)),
                                          "written_utc": time.strftime(
                                              "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}})
                    written[cohort] = int(len(np.unique(lab)))
                d = getattr(cl, "diagnostics", {}) or {}
                extra = ""
                if "n_valid_restarts" in d:
                    extra = (f"  valid {d['n_valid_restarts']}/"
                             f"{d['n_restarts']} restart(s)"
                             f"  loglik spread {d['loglik_spread_across_restarts']:.1f}")
                if d.get("stopped_early") is False:
                    extra += f"  HIT THE {d['n_iter_cap']}-ITER CAP"
                if kept:
                    extra += f"  kept {kept} (already at this fit_hash)"
                log(f"  {col:<28} K={cl.k_found:<4} {time.time()-t0:>5.1f}s  "
                    f"states used per cohort {written}{extra}")
                entries.append({"atlas": atlas, "window_s": str(window_s),
                                "column": col, "method": method,
                                "embedding": emb, "n_dim": len(cols),
                                "k": int(cl.k_found),
                                "fit_hash": h, "train_cohorts": train,
                                "states_used": written, "cohorts_kept": kept,
                                "fit_reused": bool(reused is not None),
                                "fit_diagnostics":
                                    getattr(cl, "diagnostics", None) or None})
    return entries, skipped


def _carries(method: str, emb: str) -> bool:
    """Can this embedding carry this method at all? Pure compatibility."""
    is_raw = embedding_spec(emb) is not None
    n_dim = None if is_raw else len(EMBEDDINGS[emb])
    if method in THREE_D_ONLY and (is_raw or n_dim != 3):
        return False
    return not (is_raw and method not in RAW_CAPABLE)


def require_methods_are_reachable(methods: list[str], embeddings: list[str],
                                  atlas: str, window_s) -> None:
    """Stop when a requested METHOD has no embedding here that can carry it.

    Per embedding this is already a recorded `n/a` skip, which is right: the
    grid is deliberately ragged. But a method that applies to NONE of the
    chosen embeddings is a different thing -- it produces nothing anywhere, the
    run reports success, and the state set you asked for is simply missing from
    the results. That is the failure worth being loud about, and it is knowable
    before any fit.
    """
    unreachable = {m: [e for e in embeddings if not _carries(m, e)]
                   for m in methods if not any(_carries(m, e)
                                               for e in embeddings)}
    if not unreachable:
        return
    lines = [f"atlas={atlas} window_s={window_s}: "
             f"{len(unreachable)} requested method(s) cannot run on ANY of "
             f"the chosen embeddings, so they would produce nothing:", ""]
    for m, _ in unreachable.items():
        why = ("defined on a 3-D embedding only" if m in THREE_D_ONLY
               else "not comparable on raw features (Euclidean distance over "
                    "correlated, unequally-scaled axes)")
        lines.append(f"    {m}: {why}")
    lines += [
        "",
        f"  chosen embeddings resolved to: {embeddings}",
        f"  3-D embeddings here: "
        f"{[e for e in embeddings if embedding_spec(e) is None] or '(none)'}",
        "",
        "  Add an embedding those methods can use -- `--embeddings pca umap "
        "raw` covers everything this cell has -- or drop them from --methods.",
    ]
    raise SystemExit("\n".join(lines))


def _methods_for(emb: str, args, atlas: str, window_s, skipped: list) -> list[str]:
    """Which of --methods apply to this embedding, with the rest recorded.

    Two exclusions, both about applicability rather than about failure, which is
    why they are reported here instead of raising inside a clusterer:

    THREE_D_ONLY on anything but a 3-D embedding. Threshold's K is bins**3 over
    exactly three quantile axes; there is no 14-D version of that definition.

    Everything outside RAW_CAPABLE on a `raw<N>` embedding. MeanShift is the one
    this excludes: Euclidean distance on correlated, unequally-scaled features
    is not the same method as MeanShift on pca3, so a bandwidth quantile tuned
    for one says nothing about the other.
    """
    is_raw = embedding_spec(emb) is not None
    n_dim = None if is_raw else len(EMBEDDINGS[emb])
    keep = []
    for m in args.methods:
        if m in THREE_D_ONLY and (is_raw or (n_dim is not None and n_dim != 3)):
            why = f"{m} is defined on a 3-D embedding only"
        elif is_raw and m not in RAW_CAPABLE:
            why = (f"{m} is not comparable on raw features (Euclidean distance "
                   f"over correlated, unequally-scaled axes)")
        else:
            keep.append(m)
            continue
        log(f"  SKIPPED {m} on {emb}: {why}")
        skipped.append({"atlas": atlas, "window_s": str(window_s),
                        "kind": "n/a", "method": m, "embedding": emb, "k": 0,
                        "reason": why, "search": None})
    return keep


def _accepts_k(method: str, k) -> bool:
    fn = getattr(CLUSTERERS[method], "accepts_k", None)
    return True if fn is None else bool(fn(k))


def _ks_for(method, args, emb, atlas, window_s, skipped) -> list:
    """The requested Ks this method can actually take, the rest recorded."""
    keep = []
    for k in args.k:
        if _accepts_k(method, k):
            keep.append(k)
            continue
        why = f"{method} cannot take K={k}"
        log(f"  SKIPPED {method}/{emb} K={k}: {why}")
        skipped.append({"atlas": atlas, "window_s": str(window_s),
                        "kind": "n/a", "method": method, "embedding": emb,
                        "k": int(k), "reason": why, "search": None})
    return keep


def _opts(method: str, args) -> dict:
    if method == "meanshift":
        return {"quantile": args.meanshift_quantile,
                "fit_rows": args.meanshift_fit_rows,
                "min_k": args.min_k, "max_k": args.max_k}
    if method == "hmm1":
        return {"n_iter": args.hmm_iter}
    if method == "hmm2":
        return {"n_iter": args.hmm2_iter, "n_restarts": args.hmm2_restarts,
                "n_jobs": args.hmm2_jobs,
                "min_k": args.min_k, "max_k": args.max_k}
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
            # Resolved per cell, exactly as run_one does -- otherwise --check
            # reports the FAMILY names as absent and the dry run disagrees with
            # the real one, which is the one thing a dry run must not do.
            wanted = resolve_families(args.embeddings, paths)
            have = [e for e in wanted
                    if embedding_columns(e, paths) is not None]
            # A missing raw<N> does NOT make the cell un-ready: it exists only
            # where `decompose --passthrough-features` ran, so the grid is
            # ragged for it by design and stage 4b skips it per cell. Only an
            # absent pca3/umap3 means stage 4 is incomplete here.
            missing = [e for e in wanted
                       if e not in have and embedding_spec(e) is None]
            missing_raw = [e for e in wanted
                           if e not in have and embedding_spec(e) is not None]
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
                         "raw_absent": ",".join(missing_raw) or "-",
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
            if STATE_COLUMN_RE.match(n)]


def report_check(df: pd.DataFrame, args) -> int:
    if df.empty:
        print("nothing in the grid -- check --atlas / --window-s")
        return 1
    print(f"\ngrid: {int(df['ok'].sum())}/{len(df)} (atlas, aperture) cell(s) "
          f"ready\n")
    print(df.to_string(index=False))
    # The union over cells of what each cell resolved to, so a family name
    # never reaches column_name() -- `MeanShift_raw_0` is not a column anyone
    # will see.
    resolved = sorted({e for v in df["embeddings"] if isinstance(v, str)
                       for e in v.split(",") if e and e != "-"})
    planned = [column_name(m, e, k)
               for e in resolved
               for m in args.methods
               if not (m in THREE_D_ONLY and embedding_spec(e) is not None)
               and not (embedding_spec(e) is not None and m not in RAW_CAPABLE)
               for k in (args.k if m not in K_FREE else [0])
               if m in K_FREE or _accepts_k(m, k)]
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
    # Names first, before a file is opened or a job's walltime is spent.
    # `--embeddings` lost its argparse `choices` when raw<N> arrived, since the
    # valid set depends on the atlas, so this is where a typo is caught.
    for emb in args.embeddings:
        if emb not in EMBEDDING_FAMILIES:
            embedding_spec(emb)
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

    if not entries and not skipped:
        # `and not skipped`: a run where every state set was legitimately
        # refused or did not apply DID do its job, and the report below says
        # which and why. Raising here instead would replace that with "check
        # --atlas / --window-s", which is the wrong thing to go and check.
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

    # Two kinds of skip, and only one of them is a problem.
    #
    #   n/a      the combination does not apply here -- threshold on a 14-D
    #            embedding, meanshift on raw features, a raw embedding in a cell
    #            where decompose was not asked for passthrough. The raw arm is
    #            RAGGED BY DESIGN (two cells out of fifteen), so these are the
    #            normal state of a full grid. Exiting non-zero on them would
    #            break every `--dependency=afterok` downstream for a run that
    #            did exactly what it was asked.
    #   refused  a state set that WAS asked for and could not be written: a
    #            degenerate K, or no restart expressing all K states. That is the
    #            short grid the non-zero exit exists to make impossible to miss.
    na = [sk for sk in skipped if sk.get("kind") == "n/a"]
    refused = [sk for sk in skipped if sk.get("kind") != "n/a"]
    if na:
        print(f"\n{len(na)} combination(s) not applicable (expected):", flush=True)
        for sk in na:
            print(f"  {sk['atlas']:<14} {sk['window_s']:>4}s  {sk['method']}/"
                  f"{sk['embedding']}  -- {sk['reason']}", flush=True)
    if refused:
        # Repeated at the end because the per-cell line scrolls past, and exiting
        # non-zero so a batch job that produced a short grid is not reported as a
        # clean success by sacct.
        print(f"\n{len(refused)} state set(s) REFUSED:", flush=True)
        for sk in refused:
            print(f"  {sk['atlas']:<14} {sk['window_s']:>4}s  {sk['method']}/"
                  f"{sk['embedding']}  K={sk['k']}  -- {sk['reason']}", flush=True)
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
    p.add_argument("--embeddings", nargs="+", default=list(DEFAULT_EMBEDDINGS),
                   metavar="EMB",
                   help=f"FAMILIES -- {' '.join(sorted(EMBEDDING_FAMILIES))} -- "
                        f"resolved against each cell, which is what a command "
                        f"should normally say. `raw` becomes whichever raw<N> "
                        f"the cohorts carry (raw7 on yeo7, raw14 on networks, "
                        f"nothing on harvardoxford), read from the files rather "
                        f"than named in the command, since the width is a fact "
                        f"about the atlas. An exact name still works "
                        f"({' '.join(sorted(EMBEDDINGS))}, raw<N>) when one "
                        f"specific width is the point. A family that resolves "
                        f"to nothing in a cell is skipped and reported, never "
                        f"fatal. "
                        f"Default: {' '.join(DEFAULT_EMBEDDINGS)}")
    p.add_argument("--k", nargs="+", type=int, default=[8, 10, 27],
                   help="for methods that need one; meanshift discovers it. "
                        "10 is in the default grid because it is the K van der "
                        "Meer et al. 2020 selected, so hmm1 and hmm2 can be "
                        "compared at it; 8 and 27 are perfect cubes, which "
                        "`threshold` requires.")
    p.add_argument("--project", nargs="*", default=None,
                   help="cohorts to LABEL, beside --train which is always "
                        "labelled. `--project` with NO values means none of "
                        "them: label only --train. Omitted entirely means "
                        "every cohort in the cell, which is "
                        "what you want for a full run. Name them to label a "
                        "subset -- adding one cohort months later is "
                        "`--project <it>`, and the rest keep the columns they "
                        "have. Same vocabulary as `decompose`, deliberately. A "
                        "name with no latents in the cell is refused rather "
                        "than ignored.")
    p.add_argument("--refit", action="store_true",
                   help="ignore the saved fits under outputs/meta/clusterers/ "
                        "and fit again, relabelling every cohort. The cache is "
                        "keyed by fit_hash AND by the latents' model_hash, and "
                        "refuses itself when the libraries it was pickled "
                        "under have moved, so this is for forcing the issue "
                        "rather than for correctness.")
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
    p.add_argument("--hmm-iter", type=int, default=50,
                   help="EM iterations for hmm1. 50, as it has always been -- "
                        "hmm1 is the frozen incumbent and changing this makes "
                        "it a different estimator rather than a baseline.")
    p.add_argument("--hmm2-iter", type=int, default=500,
                   help="EM iterations for hmm2 (their `options.cyc = 500`).")
    p.add_argument("--hmm2-jobs", type=int, default=None, metavar="N",
                   help="restarts to fit at once. Default: SLURM_CPUS_PER_TASK, "
                        "or 1 outside a job -- NOT os.cpu_count(), which "
                        "reports the whole node and would thrash an 8-core "
                        "allocation. The restarts are independent fits and a "
                        "single hmmlearn fit is sequential over time, so this "
                        "is where the cores go; it changes walltime only, never "
                        "a label, and is kept out of fit_hash for that reason.")
    p.add_argument("--hmm2-restarts", type=int, default=15,
                   help="initialisations for hmm2, seeds range(N) (their "
                        "`HMMREPS = 15`). The best restart EXPRESSING ALL K "
                        "STATES wins. Lower it to 5 for a first timing run: "
                        "each restart is a full 500-iteration fit, so this "
                        "multiplies the cost of every hmm2 state set directly.")
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
