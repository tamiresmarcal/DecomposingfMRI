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
