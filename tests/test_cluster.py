"""Stage 4b: one fit on the merged training cohorts, projected onto the rest.

The property that matters is that a state label means the same thing in every
cohort, which is true only if ONE fit defined it. The tests below pin that, pin
the fact that the projected cohort is never fitted on, and pin the threshold
method reproducing what `decompose --bins` used to write before it moved here.
"""
import argparse
import json

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition import cluster as C


def args(**kw):
    d = dict(methods=["threshold"], embeddings=["pca3"], k=[8],
             train=["a", "b"], balance_train=False, refit=False,
             project=None,
             meanshift_quantile=0.2, meanshift_fit_rows=50_000, hmm_iter=5,
             hmm2_iter=5, hmm2_restarts=2,
             min_k=C.STATE_K_BAND[0], max_k=C.STATE_K_BAND[1])
    d.update(kw)
    return argparse.Namespace(**d)


def frame(n_subs, n_per_sub, seed=0, spread=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subs):
        rows.append(pd.DataFrame({
            "task": "movie", "sub": f"{s:02d}",
            "window_id": np.arange(n_per_sub),
            "pca0/3": rng.normal(scale=spread, size=n_per_sub),
            "pca1/3": rng.normal(scale=spread, size=n_per_sub),
            "pca2/3": rng.normal(scale=spread, size=n_per_sub)}))
    return pd.concat(rows, ignore_index=True)


COLS = C.EMBEDDINGS["pca3"]


class TestTrainingBlock:
    def test_the_cohorts_are_merged_into_one_matrix(self):
        frames = {"a": frame(3, 10), "b": frame(2, 10, seed=1)}
        X, lengths, share = C._training_block(frames, ["a", "b"], COLS, args())
        assert len(X) == 50
        assert sum(lengths) == 50

    def test_sequence_lengths_stay_per_subject(self):
        # One stacked sequence would let an HMM learn a transition from the last
        # frame of one person to the first frame of the next.
        frames = {"a": frame(3, 10), "b": frame(2, 7, seed=1)}
        _, lengths, _ = C._training_block(frames, ["a", "b"], COLS, args())
        assert sorted(lengths) == [7, 7, 10, 10, 10]

    def test_the_row_share_reports_the_imbalance(self):
        frames = {"a": frame(9, 10), "b": frame(1, 10, seed=1)}
        _, _, share = C._training_block(frames, ["a", "b"], COLS, args())
        assert share["a"] == pytest.approx(0.9)
        assert share["b"] == pytest.approx(0.1)

    def test_a_cohort_not_in_train_is_not_in_the_matrix(self):
        frames = {"a": frame(2, 10), "b": frame(2, 10, seed=1),
                  "held_out": frame(5, 10, seed=2)}
        X, _, share = C._training_block(frames, ["a", "b"], COLS, args())
        assert len(X) == 40 and "held_out" not in share


class TestBalanceTrain:
    def test_off_by_default_keeps_every_row(self):
        frames = {"a": frame(9, 10), "b": frame(1, 10, seed=1)}
        X, _, _ = C._training_block(frames, ["a", "b"], COLS, args())
        assert len(X) == 100

    def test_on_equalises_the_cohorts(self):
        frames = {"a": frame(9, 10), "b": frame(1, 10, seed=1)}
        X, _, _ = C._training_block(frames, ["a", "b"], COLS,
                                    args(balance_train=True))
        assert len(X) == 20                 # 10 from each, not 100

    def test_it_subsamples_whole_subjects_not_rows(self):
        # Dropping rows from the middle of a sequence would break the contiguity
        # that `lengths` asserts to the HMM.
        frames = {"a": frame(9, 10), "b": frame(1, 10, seed=1)}
        _, lengths, _ = C._training_block(frames, ["a", "b"], COLS,
                                          args(balance_train=True))
        assert set(lengths) == {10}

    def test_it_is_part_of_the_fit_hash(self):
        kw = dict(method="threshold", embedding="pca3", k=8, params={},
                  train=["a", "b"], n_rows=100)
        assert (C.fit_hash(**kw, balanced=False)
                != C.fit_hash(**kw, balanced=True))


class TestThresholdMovedHere:
    def test_k_must_be_a_perfect_cube(self):
        with pytest.raises(SystemExit, match="perfect cube"):
            C.Threshold().fit(np.zeros((30, 3)), 10)

    def test_it_is_the_definition_decompose_used_to_write(self):
        # Same estimator, same settings, same input: quantile KBinsDiscretizer on
        # the training rows' 3-D embedding, raveled. k=8 is 2 bins per axis.
        from sklearn.preprocessing import KBinsDiscretizer

        X = np.random.default_rng(0).normal(size=(200, 3))
        mine = C.Threshold().fit(X, 8).labels(X)
        kbd = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="quantile",
                               subsample=None).fit(X)
        theirs = np.ravel_multi_index(kbd.transform(X).astype(int).T, (2, 2, 2))
        assert (mine == theirs).all()

    def test_every_cell_of_the_grid_is_reachable(self):
        X = np.random.default_rng(0).normal(size=(2000, 3))
        lab = C.Threshold().fit(X, 27).labels(X)
        assert lab.min() >= 0 and lab.max() < 27


class TestColumnName:
    def test_is_the_convention_stage_5a_parses(self):
        from fmri_decomposition.transitions import STATE_SUFFIX_RE
        import re

        for method, k in (("threshold", 8), ("meanshift", 5),
                          ("hmm1", 27), ("hmm2", 10)):
            name = C.column_name(method, "umap3", k)
            assert re.match(STATE_SUFFIX_RE, name), name
            assert name.endswith(f"_{k}")


class TestCheckGrid:
    """--check must be able to say "not ready" for the reasons that actually
    occur, because its whole purpose is to be run instead of an HMM grid that
    would take hours to reach the same conclusion."""

    def latents(self, root, atlas, window_s, cohorts, cols, states=()):
        import pyarrow as pa
        import pyarrow.parquet as pq

        for c in cohorts:
            d = pd.DataFrame({"task": "m", "sub": "01", "window_id": range(5)})
            for col in cols:
                d[col] = 0.0
            for st in states:
                d[st] = 0
            p = (root / "latents" / f"atlas={atlas}" / f"window_s={window_s}"
                 / f"cohort={c}" / "data.parquet")
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(d, preserve_index=False), p)

    def test_a_missing_aperture_is_reported_not_raised(self, tmp_path):
        df = C.check_grid(tmp_path, args(atlas=["yeo7"], window_s=["30"]))
        assert not df["ok"].any()
        assert "no latents" in df["reason"].iloc[0]

    def test_a_missing_embedding_names_the_fix(self, tmp_path):
        self.latents(tmp_path, "yeo7", "30", ["a", "b"], COLS)
        df = C.check_grid(tmp_path, args(atlas=["yeo7"], window_s=["30"],
                                         embeddings=["pca3", "umap3"]))
        assert not df["ok"].iloc[0]
        assert "umap3" in df["reason"].iloc[0]

    def test_a_ready_cell_is_ready(self, tmp_path):
        self.latents(tmp_path, "yeo7", "30", ["a", "b"], COLS)
        df = C.check_grid(tmp_path, args(atlas=["yeo7"], window_s=["30"]))
        assert df["ok"].iloc[0] and df["reason"].iloc[0] == "ready"

    def test_a_missing_training_cohort_is_caught(self, tmp_path):
        self.latents(tmp_path, "yeo7", "30", ["held_out"], COLS)
        df = C.check_grid(tmp_path, args(atlas=["yeo7"], window_s=["30"]))
        assert not df["ok"].iloc[0]
        assert "--train" in df["reason"].iloc[0]

    def test_it_counts_state_columns_a_rerun_would_replace(self, tmp_path):
        self.latents(tmp_path, "yeo7", "30", ["a", "b"], COLS,
                     states=["HMM_pca3_8", "MeanShift_pca3_5"])
        df = C.check_grid(tmp_path, args(atlas=["yeo7"], window_s=["30"]))
        assert df["existing"].iloc[0] == 2

    def test_the_embeddings_are_not_counted_as_state_columns(self, tmp_path):
        self.latents(tmp_path, "yeo7", "30", ["a", "b"], COLS)
        df = C.check_grid(tmp_path, args(atlas=["yeo7"], window_s=["30"]))
        assert df["existing"].iloc[0] == 0

    def test_it_writes_nothing(self, tmp_path):
        self.latents(tmp_path, "yeo7", "30", ["a", "b"], COLS)
        before = sorted(p.name for p in tmp_path.rglob("*"))
        C.check_grid(tmp_path, args(atlas=["yeo7"], window_s=["30"]))
        assert sorted(p.name for p in tmp_path.rglob("*")) == before


def blobs(n_per=4000, k=4, spread=0.6, seed=0):
    """k separated gaussian blobs in 3-D -- a density a bandwidth can resolve."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=6.0, size=(k, 3))
    return np.vstack([c + rng.normal(scale=spread, size=(n_per, 3))
                      for c in centres])


class TestMeanShiftQuantileSearch:
    """The right bandwidth is a property of the data, and on the real grid the
    old fixed quantile of 0.2 collapsed 7 of 8 harvardoxford state sets to K=1
    or 2. There are ~40 of these to fit, so the quantile is searched."""

    def test_it_escapes_a_collapse_to_one_state(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8000, 3)) * 2.0
        X[:2000] += 3.0
        cl = C.MeanShiftCluster(quantile=0.2, fit_rows=4000).fit(X)
        assert cl.k_found >= 3
        assert cl.search[0]["k"] < 3        # the start really did collapse
        assert cl.params()["quantile"] < 0.2

    def test_it_stops_as_soon_as_k_is_in_band(self):
        cl = C.MeanShiftCluster(quantile=0.1, fit_rows=4000,
                               min_k=2, max_k=64).fit(blobs())
        assert len(cl.search) >= 1
        assert 2 <= cl.search[-1]["k"] <= 64
        # no wasted fit after a hit
        assert all(not (2 <= t["k"] <= 64) for t in cl.search[:-1])

    def test_the_ladder_is_recorded_for_reading_a_result(self):
        cl = C.MeanShiftCluster(quantile=0.2, fit_rows=4000).fit(blobs())
        s = cl.params()["search"]
        assert s and all({"quantile", "bandwidth", "k"} <= set(t) for t in s)

    def test_the_quantile_used_is_in_the_fit_hash(self):
        # Two columns fitted at different bandwidths are different models, and
        # the column name alone cannot tell them apart when K happens to match.
        a = C.fit_hash("meanshift", "pca3", 4, {"quantile": 0.1}, ["a"], 100)
        b = C.fit_hash("meanshift", "pca3", 4, {"quantile": 0.05}, ["a"], 100)
        assert a != b

    def test_it_gives_up_rather_than_looping_forever(self):
        cl = C.MeanShiftCluster(quantile=0.2, fit_rows=2000, min_k=500,
                               max_k=600, max_tries=3).fit(blobs())
        assert len(cl.search) == 3          # bounded
        assert cl.k_found < 500             # and honest about failing

    def test_a_degenerate_embedding_is_fatal_not_silent(self):
        # Every row identical: no bandwidth can find structure, and a K=1 column
        # would be a constant.
        X = np.ones((500, 3))
        with pytest.raises(SystemExit, match="degenerate"):
            C.MeanShiftCluster(fit_rows=None).fit(X)


class TestDegenerateKIsRefused:
    def frames(self, tmp_path, n_subs=4, n=50):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(0)
        for c in ("a", "b"):
            d = pd.concat([pd.DataFrame({
                "task": "m", "sub": f"{s:02d}", "window_id": np.arange(n),
                "pca0/3": rng.normal(size=n), "pca1/3": rng.normal(size=n),
                "pca2/3": rng.normal(size=n)}) for s in range(n_subs)],
                ignore_index=True)
            p = (tmp_path / "latents" / "atlas=toy" / "window_s=30"
                 / f"cohort={c}" / "data.parquet")
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(d, preserve_index=False), p)

    def test_a_k_below_min_k_is_not_written(self, tmp_path):
        self.frames(tmp_path)
        # min_k above anything threshold can produce at k=8 -> refused
        a = args(methods=["threshold"], k=[8], min_k=20, max_k=64)
        entries, skipped = C.run_one(tmp_path, "toy", "30", a)
        assert entries == []
        assert len(skipped) == 1 and skipped[0]["k"] == 8

    def test_nothing_is_added_to_the_file_when_refused(self, tmp_path):
        self.frames(tmp_path)
        a = args(methods=["threshold"], k=[8], min_k=20, max_k=64)
        C.run_one(tmp_path, "toy", "30", a)
        p = (tmp_path / "latents" / "atlas=toy" / "window_s=30"
             / "cohort=a" / "data.parquet")
        assert C._state_columns(p) == []

    def test_a_usable_k_alongside_a_refused_one_still_gets_written(self, tmp_path):
        self.frames(tmp_path)
        a = args(methods=["threshold"], k=[8, 27], min_k=10, max_k=64)
        entries, skipped = C.run_one(tmp_path, "toy", "30", a)
        assert [e["k"] for e in entries] == [27]
        assert [s["k"] for s in skipped] == [8]


# ---------------------------------------------------------------- HMM1/HMM2 ---
def seq(n_subs, n_per_sub, n_dim, seed=0, sep=3.0):
    """Two well-separated blobs with a per-subject sequence structure.

    Separated on purpose: a fit on noise expresses fewer states than it was
    asked for, which is what the all-K filter refuses -- so a test about
    anything else has to give the states something to find.
    """
    rng = np.random.default_rng(seed)
    X, lengths = [], []
    for _ in range(n_subs):
        half = n_per_sub // 2
        a = rng.normal(size=(half, n_dim))
        b = rng.normal(size=(n_per_sub - half, n_dim)) + sep
        X.append(np.vstack([a, b]))
        lengths.append(n_per_sub)
    return np.vstack(X), lengths


class TestHMM1IsFrozen:
    """HMM1 is the incumbent and its hash has to keep matching the old tree."""

    def test_the_name_changed_and_nothing_else_did(self):
        cl = C.HMM1Cluster()
        assert cl.name == "HMM1"
        assert cl.params() == {
            "n_iter": 50, "covariance_type": "diag",
            "note": cl.params()["note"]}

    def test_params_carries_no_outcome(self):
        """`params()` is hashed into fit_hash, so an outcome in it would make
        two identical configurations hash differently."""
        X, lengths = seq(6, 40, 3)
        cl = C.HMM1Cluster(n_iter=5).fit(X, 2, lengths=lengths)
        for outcome in ("n_iter_ran", "loglik", "stopped_early", "aic",
                        "final_delta"):
            assert outcome not in cl.params(), outcome
            assert outcome in cl.diagnostics, outcome

    def test_fit_hash_is_unchanged_by_running_the_fit(self):
        X, lengths = seq(6, 40, 3)
        a = C.HMM1Cluster(n_iter=5).fit(X, 2, lengths=lengths)
        b = C.HMM1Cluster(n_iter=50).fit(X, 2, lengths=lengths)
        h = C.fit_hash("hmm1", "pca3", 2, C.HMM1Cluster().params(), ["a"], 100)
        assert h != C.fit_hash("hmm1", "pca3", 2, a.params(), ["a"], 100)
        assert h == C.fit_hash("hmm1", "pca3", 2, b.params(), ["a"], 100)


class TestEMDiagnostics:
    def test_stopped_early_is_not_hmmlearns_converged(self):
        """hmmlearn reports converged=True when EM merely hits the cap:

            return (self.iter == self.n_iter or ...)

        so `stopped_early` is the only field that distinguishes the two.
        """
        X, lengths = seq(6, 40, 3)
        capped = C.HMM1Cluster(n_iter=2).fit(X, 2, lengths=lengths)
        assert capped.diagnostics["n_iter_ran"] == 2
        assert capped.diagnostics["n_iter_cap"] == 2
        assert capped.diagnostics["stopped_early"] is False
        # The thing we are NOT relying on, pinned so the reason stays visible.
        assert capped.hmm.monitor_.converged is True

    def test_a_fit_that_finishes_reports_stopped_early(self):
        X, lengths = seq(8, 60, 3, sep=8.0)
        cl = C.HMM1Cluster(n_iter=500).fit(X, 2, lengths=lengths)
        assert cl.diagnostics["stopped_early"] is True
        assert cl.diagnostics["n_iter_ran"] < 500

    def test_information_criteria_are_recorded(self):
        X, lengths = seq(6, 40, 3)
        cl = C.HMM1Cluster(n_iter=5).fit(X, 2, lengths=lengths)
        assert cl.diagnostics["aic"] is not None
        assert cl.diagnostics["bic"] is not None


class TestHMM2:
    def test_it_is_full_covariance_and_says_so(self):
        p = C.HMM2Cluster().params()
        assert p["covariance_type"] == "full"
        assert p["n_iter"] == 500 and p["n_restarts"] == 15

    def test_restarts_are_recorded_with_their_spread(self):
        X, lengths = seq(8, 50, 3)
        cl = C.HMM2Cluster(n_iter=20, n_restarts=3).fit(X, 2, lengths=lengths)
        d = cl.diagnostics
        assert d["n_restarts"] == 3
        assert len(d["states_expressed_per_restart"]) == 3
        assert d["loglik_spread_across_restarts"] >= 0.0

    def test_the_best_surviving_restart_wins(self):
        """Not the last one, and not restart 0 -- the highest log-likelihood
        among those expressing all K states."""
        X, lengths = seq(8, 50, 3)
        cl = C.HMM2Cluster(n_iter=20, n_restarts=4).fit(X, 2, lengths=lengths)
        valid = [a["loglik"] for a in cl.attempts
                 if a["states_expressed"] == 2 and a["loglik"] is not None]
        assert cl.diagnostics["loglik"] == pytest.approx(max(valid))

    def test_a_K_no_restart_can_express_is_refused_not_written(self):
        """Their Step1b_Check_for_expressions.m, which is why they capped at 10.

        Asking 40 states of two blobs cannot express 40, so k_found drops below
        the requested K and the caller's band check rejects it.
        """
        X, lengths = seq(4, 30, 3)
        cl = C.HMM2Cluster(n_iter=5, n_restarts=2).fit(X, 40, lengths=lengths)
        assert cl.hmm is None
        assert cl.k_found < 40
        assert "refused" in cl.diagnostics
        with pytest.raises(RuntimeError):
            cl.labels(X, lengths=lengths)

    def test_a_refusal_is_reported_as_a_K_outside_the_band(self):
        """k_found carries the best expressed count so the EXISTING band check
        rejects it, rather than this class inventing a second refusal path."""
        X, lengths = seq(4, 30, 3)
        cl = C.HMM2Cluster(n_iter=5, n_restarts=2).fit(X, 40, lengths=lengths)
        assert not (C.STATE_K_BAND[0] <= cl.k_found <= 40) or cl.k_found != 40

    def test_lengths_reach_the_fit(self):
        """Without them hmmlearn treats the stack as one sequence."""
        X, lengths = seq(6, 40, 3)
        cl = C.HMM2Cluster(n_iter=10, n_restarts=1).fit(X, 2, lengths=lengths)
        assert cl.hmm is not None
        assert cl.labels(X, lengths=lengths).shape == (len(X),)

    def test_full_covariance_fits_better_than_diagonal(self):
        """The 7,200-nat gap, in miniature. Correlated features within a state
        are exactly what `diag` cannot represent."""
        rng = np.random.default_rng(0)
        mix = rng.normal(size=(6, 6))
        X = np.vstack([rng.normal(size=(600, 6)) @ mix,
                       rng.normal(size=(600, 6)) @ mix + 4.0])
        lengths = [200] * 6
        full = C.HMM2Cluster(n_iter=50, n_restarts=1).fit(X, 2, lengths=lengths)
        diag = C.HMM1Cluster(n_iter=50).fit(X, 2, lengths=lengths)
        assert full.diagnostics["loglik"] > diag.diagnostics["loglik"]


# ------------------------------------------------------------- embeddings ---
class TestEmbeddingNames:
    """`--embeddings` lost its argparse `choices` when raw<N> arrived, because
    the valid set depends on how many features the atlas has."""

    def test_fixed_embeddings_resolve_to_none(self):
        assert C.embedding_spec("pca3") is None
        assert C.embedding_spec("umap3") is None

    def test_raw_carries_its_expected_width(self):
        assert C.embedding_spec("raw7") == 7
        assert C.embedding_spec("raw14") == 14
        assert C.embedding_spec("raw111") == 111

    @pytest.mark.parametrize("bad", ["raw", "pcaX", "raw0", "raw1", "RAW14",
                                     "pca", "14"])
    def test_a_name_that_is_neither_is_refused_by_name(self, bad):
        with pytest.raises(SystemExit) as e:
            C.embedding_spec(bad)
        assert bad in str(e.value)


class TestMethodApplicability:
    """Exclusions are recorded, not raised, and they are NOT refusals."""

    def test_threshold_is_excluded_from_a_raw_embedding(self):
        sk = []
        keep = C._methods_for("raw14",
                              args(methods=["threshold", "hmm1", "hmm2"]),
                              "networks", "-1", sk)
        assert keep == ["hmm1", "hmm2"]
        assert [r["method"] for r in sk] == ["threshold"]
        assert sk[0]["kind"] == "n/a"

    def test_meanshift_is_excluded_from_a_raw_embedding(self):
        sk = []
        keep = C._methods_for("raw7", args(methods=["meanshift", "hmm2"]),
                              "yeo7", "-1", sk)
        assert keep == ["hmm2"]
        assert "Euclidean" in sk[0]["reason"]

    def test_hmm1_is_allowed_on_raw_because_that_is_the_comparison(self):
        """diag on correlated features is mis-specified, and that is the
        finding -- HMM1_raw14 against HMM2_raw14 isolates covariance type."""
        sk = []
        assert "hmm1" in C._methods_for("raw14", args(methods=["hmm1"]),
                                        "networks", "-1", sk)
        assert sk == []

    def test_nothing_is_excluded_from_pca3(self):
        sk = []
        keep = C._methods_for(
            "pca3", args(methods=["threshold", "meanshift", "hmm1", "hmm2"]),
            "yeo7", "30", sk)
        assert keep == ["threshold", "meanshift", "hmm1", "hmm2"]
        assert sk == []


class TestThresholdNeedsACube:
    def test_k_10_is_declined_rather_than_fatal(self):
        """K=10 is in the default grid so hmm1 and hmm2 can be compared at the
        paper's choice. A SystemExit here would kill the whole job."""
        assert C.Threshold.accepts_k(8)
        assert C.Threshold.accepts_k(27)
        assert not C.Threshold.accepts_k(10)

    def test_the_hmms_take_any_k(self):
        for k in (8, 10, 27):
            assert C._accepts_k("hmm1", k) and C._accepts_k("hmm2", k)

    def test_the_declined_k_is_recorded_as_not_applicable(self):
        sk = []
        ks = C._ks_for("threshold", args(k=[8, 10, 27]), "pca3", "yeo7", "30", sk)
        assert ks == [8, 27]
        assert [r["k"] for r in sk] == [10]
        assert sk[0]["kind"] == "n/a"

    def test_a_non_cube_still_refuses_if_it_reaches_fit(self):
        X = np.random.default_rng(0).normal(size=(500, 3))
        with pytest.raises(SystemExit, match="perfect cube"):
            C.Threshold().fit(X, 10)

    def test_threshold_refuses_a_non_3d_embedding_by_name(self):
        X = np.random.default_rng(0).normal(size=(500, 14))
        with pytest.raises(SystemExit) as e:
            C.Threshold().fit(X, 8)
        assert "3-D" in str(e.value) and "14" in str(e.value)


# -------------------------------------------------- raw columns on real files ---
def write_latents(root, atlas, window_s, cohorts, cols=None, states=()):
    """`cohorts` is a list sharing `cols`, or a dict of cohort -> its own cols."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    per = (cohorts if isinstance(cohorts, dict)
           else {c: cols for c in cohorts})
    for c, these in per.items():
        d = pd.DataFrame({"task": "m", "sub": "01", "window_id": range(6)})
        for col in these:
            d[col] = np.linspace(0, 1, 6)
        for st in states:
            d[st] = 0
        p = (root / "latents" / f"atlas={atlas}" / f"window_s={window_s}"
             / f"cohort={c}" / "data.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(d, preserve_index=False), p)


RAW14 = [f"{C.RAW_PREFIX}net{i:02d}" for i in range(14)]
ALL_UMAP = C.EMBEDDINGS["umap3"]


class TestRawEmbeddingResolution:
    def test_it_resolves_when_every_cohort_has_the_same_features(self, tmp_path):
        write_latents(tmp_path, "networks", "-1", ["a", "b"], COLS + RAW14)
        paths = C.cohort_paths(tmp_path, "networks", "-1")
        cols = C.embedding_columns("raw14", paths)
        assert cols == sorted(RAW14)

    def test_the_order_is_sorted_not_schema_order(self, tmp_path):
        """A column permutation between the fit cohort and a projected one
        would silently relabel the axes."""
        write_latents(tmp_path, "networks", "-1",
                      {"a": COLS + RAW14, "b": COLS + list(reversed(RAW14))})
        paths = C.cohort_paths(tmp_path, "networks", "-1")
        assert C.embedding_columns("raw14", paths) == sorted(RAW14)

    def test_a_cohort_short_of_features_resolves_to_none(self, tmp_path):
        write_latents(tmp_path, "networks", "-1",
                      {"a": COLS + RAW14, "b": COLS + RAW14[:13]})
        paths = C.cohort_paths(tmp_path, "networks", "-1")
        assert C.embedding_columns("raw14", paths) is None

    def test_the_wrong_width_resolves_to_none(self, tmp_path):
        write_latents(tmp_path, "networks", "-1", ["a", "b"], COLS + RAW14)
        paths = C.cohort_paths(tmp_path, "networks", "-1")
        assert C.embedding_columns("raw7", paths) is None

    def test_no_passthrough_at_all_resolves_to_none(self, tmp_path):
        write_latents(tmp_path, "yeo7", "30", ["a", "b"], COLS)
        paths = C.cohort_paths(tmp_path, "yeo7", "30")
        assert C.embedding_columns("raw14", paths) is None

    def test_a_missing_fixed_embedding_also_resolves_to_none(self, tmp_path):
        """Not assumed present. Returning the constant unconditionally made
        check_grid call a cell ready when umap3 was absent."""
        write_latents(tmp_path, "yeo7", "30", ["a", "b"], COLS)
        paths = C.cohort_paths(tmp_path, "yeo7", "30")
        assert C.embedding_columns("pca3", paths) == COLS
        assert C.embedding_columns("umap3", paths) is None


class TestRawIsRaggedByDesign:
    """The raw arm exists in two cells out of fifteen. That must not make
    `--check` call the other thirteen un-ready, and must not make `cluster`
    exit non-zero -- every `--dependency=afterok` downstream depends on it."""

    def test_check_does_not_fail_a_cell_for_a_missing_raw_embedding(self, tmp_path):
        write_latents(tmp_path, "yeo7", "30", ["a", "b"], COLS + ALL_UMAP)
        df = C.check_grid(tmp_path, args(atlas=["yeo7"], window_s=["30"],
                                         embeddings=["pca3", "umap3", "raw14"]))
        assert df["ok"].iloc[0], df["reason"].iloc[0]
        assert df["raw_absent"].iloc[0] == "raw14"

    def test_check_still_fails_a_cell_for_a_missing_fixed_embedding(self, tmp_path):
        write_latents(tmp_path, "yeo7", "30", ["a", "b"], COLS)
        df = C.check_grid(tmp_path, args(atlas=["yeo7"], window_s=["30"],
                                         embeddings=["pca3", "umap3", "raw14"]))
        assert not df["ok"].iloc[0]
        assert "umap3" in df["reason"].iloc[0]
        assert "raw14" not in df["reason"].iloc[0]



class TestExitCodeSeparatesNotApplicableFromRefused:
    """Only a REFUSAL is non-zero.

    The raw arm is ragged by design and threshold declines K=10, so a correct
    full-grid run produces `n/a` entries every time. Exiting non-zero on those
    would break every `--dependency=afterok` downstream for a job that did
    exactly what it was asked.
    """

    def latents(self, tmp_path, cols):
        """Enough rows, enough subjects and real variation for a K=10 HMM.

        The shared `write_latents` helper writes six identical ramps, which is
        fine for resolving a column list and useless for fitting anything.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(0)
        for c in ("a", "b"):
            n_sub, n_per = 8, 120
            d = pd.DataFrame({
                "task": "m",
                "sub": np.repeat([f"{i:02d}" for i in range(n_sub)], n_per),
                "window_id": np.tile(np.arange(n_per), n_sub)})
            centres = rng.normal(scale=6.0, size=(12, len(cols)))
            pick = rng.integers(0, 12, size=n_sub * n_per)
            block = centres[pick] + rng.normal(size=(n_sub * n_per, len(cols)))
            for j, col in enumerate(cols):
                d[col] = block[:, j]
            p = (tmp_path / "latents" / "atlas=networks" / "window_s=-1"
                 / f"cohort={c}" / "data.parquet")
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(d, preserve_index=False), p)

    def run_cluster(self, tmp_path, **kw):
        d = dict(atlas=["networks"], window_s=["-1"], output_root=str(tmp_path),
                 check=False, train=["a"], project=["b"])
        d.update(kw)
        return C.run(args(**d))

    def test_a_grid_whose_only_skips_are_not_applicable_exits_zero(self, tmp_path):
        self.latents(tmp_path, COLS + RAW14)
        rc = self.run_cluster(tmp_path,
                              methods=["threshold", "hmm1"],
                              embeddings=["pca3", "raw14", "raw7"],
                              k=[8, 10])
        assert rc == 0

    def test_everything_that_applied_was_still_written(self, tmp_path):
        import pyarrow.parquet as pq

        self.latents(tmp_path, COLS + RAW14)
        self.run_cluster(tmp_path, methods=["threshold", "hmm1"],
                         embeddings=["pca3", "raw14"], k=[8, 10])
        cols = set(pq.ParquetFile(
            tmp_path / "latents" / "atlas=networks" / "window_s=-1"
            / "cohort=b" / "data.parquet").schema_arrow.names)
        # threshold: pca3 at K=8 only (10 is not a cube, raw14 is not 3-D)
        assert "ThresholdCluster_pca3_8" in cols
        assert "ThresholdCluster_pca3_10" not in cols
        assert not any(c.startswith("ThresholdCluster_raw") for c in cols)
        # hmm1: both K, both embeddings
        for name in ("HMM1_pca3_8", "HMM1_pca3_10",
                     "HMM1_raw14_8", "HMM1_raw14_10"):
            assert name in cols, name

    def test_the_manifest_records_why_each_skip_happened(self, tmp_path):
        import json as _json

        self.latents(tmp_path, COLS + RAW14)
        self.run_cluster(tmp_path, methods=["threshold", "hmm1"],
                         embeddings=["pca3", "raw14", "raw7"], k=[8, 10])
        man = _json.loads((tmp_path / "meta" / "cluster"
                           / "manifest.json").read_text())
        kinds = {s["kind"] for s in man["skipped"]}
        assert kinds == {"n/a"}
        reasons = " ".join(s["reason"] for s in man["skipped"])
        assert "3-D embedding only" in reasons      # threshold on raw14
        assert "cannot take K=10" in reasons        # threshold, non-cube
        assert "no raw7 columns" in reasons         # the ragged arm

    def test_a_refused_state_set_still_exits_non_zero(self, tmp_path):
        """A K outside the band is a REFUSAL -- a state set that was asked for
        and could not be written -- so the short grid is still loud."""
        self.latents(tmp_path, COLS + RAW14)
        rc = self.run_cluster(tmp_path, methods=["hmm1"], embeddings=["pca3"],
                              k=[8], max_k=5)
        assert rc == 1

    def test_a_refusal_and_a_not_applicable_in_one_run_exits_non_zero(
            self, tmp_path):
        """The n/a entries must not mask the refusal, nor the refusal hide
        them: both are reported, and the exit code follows the refusal."""
        self.latents(tmp_path, COLS + RAW14)
        rc = self.run_cluster(tmp_path, methods=["threshold", "hmm1"],
                              embeddings=["pca3", "raw7"], k=[8, 10], max_k=5)
        assert rc == 1


class TestRestartsInParallel:
    """The restarts are the parallel axis, and parallelising them must not
    change a single label.

    A single hmmlearn fit is sequential over time, so BLAS threads do nothing
    for it -- measured at 4.37 s on one thread against 4.70 s on eight. The
    restarts are independent, so they are where the cores belong. The risk this
    class exists to pin is that completion order becomes part of the answer.
    """

    def data(self):
        return seq(10, 60, 4, seed=1)

    def test_parallel_and_sequential_agree_exactly(self):
        X, lengths = self.data()
        one = C.HMM2Cluster(n_iter=20, n_restarts=4, n_jobs=1).fit(
            X, 2, lengths=lengths)
        many = C.HMM2Cluster(n_iter=20, n_restarts=4, n_jobs=4).fit(
            X, 2, lengths=lengths)
        assert np.array_equal(one.labels(X, lengths=lengths),
                              many.labels(X, lengths=lengths))
        assert one.diagnostics["loglik"] == pytest.approx(
            many.diagnostics["loglik"])
        assert one.diagnostics["best_seed"] == many.diagnostics["best_seed"]

    def test_every_restart_is_accounted_for_either_way(self):
        X, lengths = self.data()
        one = C.HMM2Cluster(n_iter=10, n_restarts=5, n_jobs=1).fit(
            X, 2, lengths=lengths)
        many = C.HMM2Cluster(n_iter=10, n_restarts=5, n_jobs=5).fit(
            X, 2, lengths=lengths)
        assert [a["seed"] for a in one.attempts] == list(range(5))
        assert [a["seed"] for a in many.attempts] == list(range(5))
        assert ([a["loglik"] for a in one.attempts]
                == pytest.approx([a["loglik"] for a in many.attempts]))

    def test_the_winner_is_the_lowest_seed_on_a_tie(self):
        """Not whichever finished first. With parallel restarts, completion
        order is not seed order, so a tie broken by arrival would make the
        labels depend on scheduling."""
        X, lengths = self.data()
        cl = C.HMM2Cluster(n_iter=10, n_restarts=3, n_jobs=1)
        cl.attempts = []
        # Same log-likelihood from two seeds; the selection must be total.
        tied = [{"seed": 2, "loglik": -5.0, "states_expressed": 2, "error": None},
                {"seed": 0, "loglik": -5.0, "states_expressed": 2, "error": None},
                {"seed": 1, "loglik": -9.0, "states_expressed": 2, "error": None}]
        best = max(tied, key=lambda a: (a["loglik"], -a["seed"]))
        assert best["seed"] == 0

    def test_models_are_not_kept_in_the_recorded_attempts(self):
        """`attempts` goes into the parquet provenance as JSON."""
        X, lengths = self.data()
        cl = C.HMM2Cluster(n_iter=10, n_restarts=2, n_jobs=2).fit(
            X, 2, lengths=lengths)
        assert all("model" not in a for a in cl.attempts)
        import json
        json.dumps(cl.diagnostics)          # must be serialisable

    def test_n_jobs_is_recorded_but_not_hashed(self):
        """A run on 8 cores and a run on 1 produce the same labels, so they
        must carry the same fit_hash or they stop being comparable."""
        a = C.HMM2Cluster(n_restarts=4, n_jobs=1)
        b = C.HMM2Cluster(n_restarts=4, n_jobs=8)
        assert a.params() == b.params()
        assert "n_jobs" not in a.params()
        h = C.fit_hash("hmm2", "pca3", 8, a.params(), ["x"], 100)
        assert h == C.fit_hash("hmm2", "pca3", 8, b.params(), ["x"], 100)
        X, lengths = self.data()
        assert a.fit(X, 2, lengths=lengths).diagnostics["n_jobs"] == 1

    def test_n_restarts_1_never_reaches_joblib(self):
        """One restart has nothing to parallelise, and importing joblib to
        discover that would be a needless dependency on a smoke run."""
        X, lengths = self.data()
        cl = C.HMM2Cluster(n_iter=5, n_restarts=1, n_jobs=8).fit(
            X, 2, lengths=lengths)
        assert cl.diagnostics["n_restarts"] == 1
        assert cl.hmm is not None

    def test_the_default_comes_from_slurm_not_the_node(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        assert C._default_hmm2_jobs() == 8
        assert C.HMM2Cluster().n_jobs == 8
        monkeypatch.delenv("SLURM_CPUS_PER_TASK")
        assert C._default_hmm2_jobs() == 1

    @pytest.mark.parametrize("bad", ["", "oops", "0"])
    def test_a_nonsense_cpu_count_falls_back_to_one(self, monkeypatch, bad):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", bad)
        assert C._default_hmm2_jobs() >= 1

    def test_a_restart_that_fails_does_not_lose_the_others(self):
        """A singular covariance costs that restart, not the run."""
        X, lengths = self.data()
        cl = C.HMM2Cluster(n_iter=10, n_restarts=3, n_jobs=3)
        cl.n_restarts = 3
        out = cl.fit(X, 2, lengths=lengths)
        assert out.diagnostics["n_valid_restarts"] >= 1
        assert out.hmm is not None


# ==========================================================================
# Embedding FAMILIES, method reachability, and --project.
# ==========================================================================
def _latents_cell(tmp_path, raw_widths, cohorts=("a", "b", "c")):
    """A cell with pca3/umap3 everywhere and `raw_widths[cohort]` raw columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fmri_decomposition.io import RAW_PREFIX

    paths = {}
    for ci, c in enumerate(cohorts):
        n = 60
        d = {"cohort": [c] * n, "task": ["m"] * n,
             "sub": [f"S{i // 20}" for i in range(n)],
             "window_id": list(range(n))}
        for pre in ("pca", "umap"):
            for j in range(3):
                d[f"{pre}{j}/3"] = list(np.linspace(0, 1, n) + j + ci)
        for j in range(raw_widths.get(c, 0)):
            d[f"{RAW_PREFIX}F{j}"] = list(np.linspace(0, 1, n) + j)
        path = (tmp_path / "latents" / "atlas=mini" / "window_s=-1"
                / f"cohort={c}" / "data.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        t = pa.Table.from_pandas(pd.DataFrame(d), preserve_index=False)
        pq.write_table(t.replace_schema_metadata(
            {b"model_hash": json.dumps("h0").encode(),
             b"source": json.dumps("activation").encode()}), path)
        paths[c] = path
    return paths


class TestEmbeddingFamilies:
    def test_pca_and_umap_resolve_to_their_single_member(self, tmp_path):
        paths = _latents_cell(tmp_path, {})
        assert C.resolve_families(["pca"], paths) == ["pca3"]
        assert C.resolve_families(["umap"], paths) == ["umap3"]

    def test_there_is_no_wider_pca_to_resolve_to(self):
        """`decompose --n-latents 3 14` writes a 14-component PCA, but it is
        there so raw14 is a rotation of a FULL-RANK one -- which is what lets
        state-means invert it -- and pca14 is not a clustering embedding. So
        `pca` is unambiguous."""
        assert C.EMBEDDING_FAMILIES["pca"] == ["pca3"]
        with pytest.raises(SystemExit):
            C.embedding_spec("pca14")

    def test_raw_is_read_from_the_cell_not_from_the_command(self, tmp_path):
        """The whole point: raw's width is a fact about the atlas -- raw7 on
        yeo7, raw14 on networks -- so the command should not have to carry it."""
        paths = _latents_cell(tmp_path, {"a": 7, "b": 7, "c": 7})
        assert C.resolve_families(["raw"], paths) == ["raw7"]
        paths = _latents_cell(tmp_path / "two", {"a": 14, "b": 14, "c": 14})
        assert C.resolve_families(["raw"], paths) == ["raw14"]

    def test_raw_resolves_to_nothing_where_passthrough_never_ran(self, tmp_path):
        """harvardoxford. Not an error -- the grid is ragged for raw by
        design."""
        paths = _latents_cell(tmp_path, {})
        assert C.resolve_families(["raw"], paths) == []
        assert C.resolve_families(["pca", "raw"], paths) == ["pca3"]

    def test_the_default_asks_for_everything_the_cell_has(self, tmp_path):
        """The gap this closes: --methods defaulted to all four while
        --embeddings defaulted to two of four, so forgetting raw14 gave a run
        that succeeded with the named-network arm silently missing."""
        assert C.DEFAULT_EMBEDDINGS == ["pca", "umap", "raw"]
        paths = _latents_cell(tmp_path, {"a": 14, "b": 14, "c": 14})
        assert C.resolve_families(C.DEFAULT_EMBEDDINGS, paths) == [
            "pca3", "umap3", "raw14"]

    def test_an_exact_name_still_works(self, tmp_path):
        paths = _latents_cell(tmp_path, {"a": 14, "b": 14, "c": 14})
        assert C.resolve_families(["raw14"], paths) == ["raw14"]
        assert C.resolve_families(["pca3", "raw14"], paths) == ["pca3", "raw14"]

    def test_order_is_kept_and_duplicates_dropped(self, tmp_path):
        paths = _latents_cell(tmp_path, {"a": 7, "b": 7, "c": 7})
        assert C.resolve_families(["raw", "pca", "pca3", "raw7"], paths) == [
            "raw7", "pca3"]

    def test_cohorts_disagreeing_on_width_is_refused(self, tmp_path):
        """A fit on one cohort's matrix projected onto another's would silently
        relabel the axes if the widths differed."""
        paths = _latents_cell(tmp_path, {"a": 7, "b": 14, "c": 7})
        with pytest.raises(SystemExit) as e:
            C.resolve_families(["raw"], paths)
        assert "different numbers of raw features" in str(e.value)
        assert "[7, 14]" in str(e.value)

    def test_a_typo_is_still_caught(self, tmp_path):
        paths = _latents_cell(tmp_path, {})
        with pytest.raises(SystemExit) as e:
            C.resolve_families(["pcca"], paths)
        assert "unknown embedding" in str(e.value)
        assert "pca" in str(e.value)        # the families are offered


class TestMethodsMustBeReachable:
    def test_threshold_with_only_raw_is_refused(self):
        """Per embedding this is an n/a skip, and rightly so. But a method that
        applies to NONE of the chosen embeddings produces nothing anywhere
        while the run reports success."""
        with pytest.raises(SystemExit) as e:
            C.require_methods_are_reachable(["threshold"], ["raw14"],
                                            "networks", -1)
        assert "threshold" in str(e.value)
        assert "3-D embedding only" in str(e.value)

    def test_meanshift_with_only_raw_is_refused(self):
        with pytest.raises(SystemExit) as e:
            C.require_methods_are_reachable(["meanshift"], ["raw7"], "yeo7", -1)
        assert "not comparable on raw features" in str(e.value)

    def test_it_names_what_to_do(self):
        with pytest.raises(SystemExit) as e:
            C.require_methods_are_reachable(["threshold"], ["raw14"], "a", -1)
        msg = str(e.value)
        assert "--embeddings pca umap raw" in msg
        assert "drop them from --methods" in msg

    def test_one_reachable_embedding_is_enough(self):
        """The grid stays deliberately ragged: threshold on pca3 plus a
        recorded n/a for threshold on raw14 is the correct outcome."""
        C.require_methods_are_reachable(["threshold", "hmm2"],
                                        ["pca3", "raw14"], "a", -1)

    def test_the_hmms_reach_everything(self):
        C.require_methods_are_reachable(["hmm1", "hmm2"], ["raw14"], "a", -1)

    def test_carries_matches_the_per_cell_skip_rules(self):
        assert C._carries("threshold", "pca3")
        assert not C._carries("threshold", "raw14")
        assert C._carries("meanshift", "umap3")
        assert not C._carries("meanshift", "raw7")
        assert C._carries("hmm2", "raw7") and C._carries("hmm2", "pca3")
