"""Stage 4b: one fit on the merged training cohorts, projected onto the rest.

The property that matters is that a state label means the same thing in every
cohort, which is true only if ONE fit defined it. The tests below pin that, pin
the fact that the projected cohort is never fitted on, and pin the threshold
method reproducing what `decompose --bins` used to write before it moved here.
"""
import argparse

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition import cluster as C


def args(**kw):
    d = dict(methods=["threshold"], embeddings=["pca3"], k=[8],
             train=["a", "b"], balance_train=False,
             meanshift_quantile=0.2, meanshift_fit_rows=50_000, hmm_iter=5)
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

        for method, k in (("threshold", 8), ("meanshift", 5), ("hmm", 27)):
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
