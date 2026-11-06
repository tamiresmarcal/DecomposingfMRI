"""--group-col: folds that never split a family, and a seed that still matters.

The leak this closes: HCP is full of twins and siblings. With plain KFold a
sibling lands in the training fold and the held-out one becomes partly
predictable from family rather than from brain state, so the score comes out
optimistic and nothing in the output says so.

The second failure, which is the one worth tests: sklearn's GroupKFold took no
`shuffle` before 1.6, and the container pins scikit-learn 1.5.2. Used there,
every seed in --seeds would yield the SAME partition, the spread across seeds
would collapse to zero, and a reader would see a beautifully stable score that
is really one fit reported five times.
"""
import numpy as np
import pandas as pd
import pytest

from fmri_decomposition.bstm_selection import group_folds, read_phenotype, score_once


def pairs(n_fam=84):
    """The HCP shape: every family a pair, so sizes cannot break ties."""
    return np.repeat(np.arange(n_fam), 2)


class TestPartition:
    @pytest.mark.parametrize("seed", [0, 1, 7, 99])
    def test_no_group_is_split_across_a_fold(self, seed):
        g = pairs()
        for tr, te in group_folds(g, 5, seed):
            assert not (set(g[tr]) & set(g[te]))

    @pytest.mark.parametrize("seed", [0, 3])
    def test_the_folds_partition_every_row_exactly_once(self, seed):
        g = pairs()
        seen = np.concatenate([te for _, te in group_folds(g, 5, seed)])
        assert sorted(seen.tolist()) == list(range(len(g)))

    def test_train_and_test_are_disjoint_and_cover_everything(self):
        g = pairs()
        for tr, te in group_folds(g, 5, 0):
            assert not set(tr) & set(te)
            assert len(tr) + len(te) == len(g)

    def test_folds_are_balanced(self):
        sizes = [len(te) for _, te in group_folds(pairs(), 5, 0)]
        assert max(sizes) - min(sizes) <= 2

    def test_uneven_group_sizes_still_balance(self):
        # one big family and many singletons -- the big one must not make a
        # fold twice the size of the others
        g = np.array([0] * 9 + list(range(1, 41)))
        sizes = [len(te) for _, te in group_folds(g, 5, 0)]
        assert max(sizes) <= 2 * min(sizes)

    def test_a_single_huge_group_cannot_be_split(self):
        g = np.array([0] * 20 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5)
        for tr, te in group_folds(g, 5, 0):
            assert not (set(g[tr]) & set(g[te]))


class TestSeedMatters:
    def test_two_seeds_give_different_partitions(self):
        g = pairs()
        a = [frozenset(te.tolist()) for _, te in group_folds(g, 5, 0)]
        b = [frozenset(te.tolist()) for _, te in group_folds(g, 5, 1)]
        assert set(a) != set(b), (
            "every seed produced the same folds -- this is exactly the "
            "unshuffled-GroupKFold failure this function exists to avoid")

    def test_the_same_seed_is_reproducible(self):
        g = pairs()
        a = [te.tolist() for _, te in group_folds(g, 5, 4)]
        b = [te.tolist() for _, te in group_folds(g, 5, 4)]
        assert a == b

    def test_many_seeds_give_many_partitions(self):
        g = pairs()
        got = {tuple(frozenset(te.tolist()) for _, te in group_folds(g, 5, s))
               for s in range(10)}
        assert len(got) >= 8


class TestRefusal:
    def test_fewer_groups_than_folds_is_an_error(self):
        with pytest.raises(ValueError, match="cannot fill"):
            list(group_folds(np.array([0, 0, 1, 1]), 5, 0))


class TestScoreOnce:
    def _xy(self, n_fam=40, seed=0):
        rng = np.random.default_rng(seed)
        g = np.repeat(np.arange(n_fam), 2)
        fam_effect = rng.normal(size=n_fam)[g]        # shared within a family
        X = np.column_stack([fam_effect + rng.normal(0, .1, len(g)),
                             rng.normal(size=len(g))])
        y = fam_effect + rng.normal(0, .5, len(g))
        return X, y, g

    def test_grouped_scoring_runs_and_returns_a_number(self):
        X, y, g = self._xy()
        s = score_once(X, y, "ridge", 0, groups=g)
        assert np.isfinite(s)

    def test_grouping_lowers_a_score_driven_by_family(self):
        """The point of the flag, as a measurement rather than an assertion
        about one number: when the ONLY signal is shared within a family,
        ungrouped folds can see it through the sibling and grouped folds
        cannot, so the grouped score must not be the higher of the two."""
        X, y, g = self._xy(n_fam=40, seed=3)
        plain = np.mean([score_once(X, y, "ridge", s) for s in range(5)])
        grouped = np.mean([score_once(X, y, "ridge", s, groups=g)
                           for s in range(5)])
        assert grouped <= plain + 1e-9

    def test_no_groups_keeps_the_old_behaviour_exactly(self):
        X, y, _ = self._xy()
        assert score_once(X, y, "ridge", 0) == score_once(X, y, "ridge", 0,
                                                          groups=None)

    def test_a_single_class_training_split_returns_nan_not_a_crash(self):
        # a classifier target whose positives all sit in a few families
        rng = np.random.default_rng(0)
        g = np.repeat(np.arange(30), 2)
        y = np.zeros(60)
        y[:24] = 1                      # 12 whole families positive
        X = rng.normal(size=(60, 3))
        s = score_once(X, y, "logistic", 0, groups=g)
        assert np.isfinite(s) or np.isnan(s)      # never raises


class TestPhenotypeCarriesTheGroup:
    def _write(self, tmp_path, rows):
        p = tmp_path / "ph.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return f"{p}:,"

    def test_the_group_column_survives_to_the_frame(self, tmp_path):
        spec = self._write(tmp_path, {
            "ID": ["A", "B", "C"], "t": [0, 1, 2],
            "Age": [30, 40, 50], "Fam": ["F1", "F1", "F2"]})
        ph = read_phenotype([spec], "ID", "t", ["Age"], [], group_col="Fam")
        assert "Fam" in ph.columns
        assert ph.set_index("sub").loc["A", "Fam"] == "F1"

    def test_it_is_not_coerced_to_a_number(self, tmp_path):
        spec = self._write(tmp_path, {
            "ID": ["A", "B"], "t": [0, 1], "Age": [30, 40],
            "Fam": ["0012", "0013"]})
        ph = read_phenotype([spec], "ID", "t", ["Age"], [], group_col="Fam")
        # a leading zero survives, which it would not through to_numeric
        assert ph.set_index("sub").loc["A", "Fam"] == "0012"

    def test_a_subject_with_no_group_is_dropped(self, tmp_path):
        spec = self._write(tmp_path, {
            "ID": ["A", "B", "C"], "t": [0, 1, 2], "Age": [30, 40, 50],
            "Fam": ["F1", None, "F2"]})
        ph = read_phenotype([spec], "ID", "t", ["Age"], [], group_col="Fam")
        assert set(ph["sub"]) == {"A", "C"}, (
            "a subject with no group cannot be kept out of its relatives' "
            "fold, so it must not silently become its own group")

    def test_without_the_flag_the_column_is_absent(self, tmp_path):
        spec = self._write(tmp_path, {
            "ID": ["A", "B"], "t": [0, 1], "Age": [30, 40], "Fam": ["F1", "F2"]})
        ph = read_phenotype([spec], "ID", "t", ["Age"], [])
        assert "Fam" not in ph.columns

    def test_a_missing_group_column_is_reported(self, tmp_path):
        spec = self._write(tmp_path, {"ID": ["A"], "t": [0], "Age": [30]})
        with pytest.raises(SystemExit, match="Nope"):
            read_phenotype([spec], "ID", "t", ["Age"], [], group_col="Nope")
