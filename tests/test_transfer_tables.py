"""Train on one cohort, test on another: the parts that can silently lie.

Three hazards, each with its own class below.

1. COLUMN ALIGNMENT. `Study.matrix` one-hots the covariates with get_dummies,
   so the column NAMES depend on the values each cohort happens to hold:
   Cam-CAN's Sex of MALE/FEMALE and HCP's Gender of M/F give different dummy
   columns from the same variable. Lined up by position, the model would read
   sex as something else and every number downstream would still look
   reasonable.

2. THE ESTIMATOR. hads_tables._fit returns a cross-validated FLOAT, not a
   fitted model -- right inside one cohort, useless across two. A first draft
   of this module called .predict() on that float.

3. THE INTERVAL. There are no folds here, so the error bar can only come from
   resampling the test subjects. It has to bracket the observed score and it
   has to survive a resample where Spearman is undefined.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transfer_tables as tt  # noqa: E402


def data(n=200, p=5, beta0=1.0, noise=.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    b = np.zeros(p); b[0] = beta0
    return X, X @ b + rng.normal(0, noise, n)


class TestAlignment:
    def test_matching_names_pass_through_in_training_order(self):
        got = tt.align_columns(["Age", "Sex_M", "0->1"], ["0->1", "Sex_M", "Age"])
        assert got == ["Age", "Sex_M", "0->1"]

    def test_a_differently_spelled_covariate_is_refused(self):
        with pytest.raises(SystemExit) as e:
            tt.align_columns(["Age", "Sex_M"], ["Age", "Gender_M"])
        assert "Sex_M" in str(e.value)

    def test_the_refusal_names_the_unused_column_too(self):
        with pytest.raises(SystemExit) as e:
            tt.align_columns(["Sex_M"], ["Gender_M"])
        assert "Gender_M" in str(e.value)

    def test_the_refusal_points_at_covariate_aliases(self):
        with pytest.raises(SystemExit) as e:
            tt.align_columns(["Sex_M"], ["Gender_M"])
        assert "covariate_aliases" in str(e.value)

    def test_reorder_puts_the_test_columns_in_training_order(self):
        X = np.array([[1., 2., 3.], [4., 5., 6.]])
        got = tt.reorder(["c", "a", "b"], X, ["a", "b", "c"])
        assert got.tolist() == [[2., 3., 1.], [5., 6., 4.]]

    def test_reorder_never_relies_on_position(self):
        """The property, not the instance: whatever order the test frame is
        in, the result is the training order."""
        rng = np.random.default_rng(0)
        names = ["a", "b", "c", "d"]
        X = rng.normal(size=(5, 4))
        for perm in ([3, 1, 0, 2], [0, 1, 2, 3], [2, 3, 1, 0]):
            shuffled = [names[i] for i in perm]
            got = tt.reorder(shuffled, X[:, perm], names)
            assert np.allclose(got, X)


class TestFitPredict:
    def test_it_returns_one_prediction_per_test_subject(self):
        Xtr, ytr = data(200)
        Xte, _ = data(37, seed=1)
        assert tt.fit_predict(Xtr, ytr, Xte, "ridge", 0).shape == (37,)

    def test_a_planted_signal_transfers(self):
        Xtr, ytr = data(300, seed=0)
        Xte, yte = data(120, seed=1)
        pred = tt.fit_predict(Xtr, ytr, Xte, "ridge", 0)
        from scipy import stats
        assert stats.spearmanr(pred, yte).statistic > .5

    def test_pure_noise_does_not(self):
        Xtr, ytr = data(300, beta0=0.0, seed=0)
        Xte, yte = data(120, beta0=0.0, seed=1)
        pred = tt.fit_predict(Xtr, ytr, Xte, "ridge", 0)
        from scipy import stats
        assert abs(stats.spearmanr(pred, yte).statistic) < .4

    def test_hgb_also_works(self):
        Xtr, ytr = data(300, seed=0)
        Xte, _ = data(60, seed=1)
        assert np.isfinite(tt.fit_predict(Xtr, ytr, Xte, "hgb", 0)).all()

    def test_the_classifier_returns_a_probability(self):
        """logistic is the one classifier; score_once binarises at y > 0 and
        so must this, or the two would score different things."""
        Xtr, ytr = data(300, seed=0)
        Xte, _ = data(60, seed=1)
        p = tt.fit_predict(Xtr, ytr, Xte, "logistic", 0)
        assert ((0 <= p) & (p <= 1)).all()


class TestBootstrap:
    def test_the_interval_brackets_the_observed_score(self):
        Xtr, ytr = data(300, seed=0)
        Xte, yte = data(150, seed=1)
        pred = tt.fit_predict(Xtr, ytr, Xte, "ridge", 0)
        sc, lo, hi = tt.bootstrap_ci(yte, pred, 400, 0)
        assert lo <= sc <= hi

    def test_more_test_subjects_narrow_it(self):
        """The interval must be about the test cohort's size, since that is
        the only thing left varying once the model is fixed."""
        Xtr, ytr = data(400, seed=0)
        widths = []
        for n in (60, 400):
            Xte, yte = data(n, seed=2)
            pred = tt.fit_predict(Xtr, ytr, Xte, "ridge", 0)
            _, lo, hi = tt.bootstrap_ci(yte, pred, 400, 0)
            widths.append(hi - lo)
        assert widths[1] < widths[0]

    def test_it_is_reproducible(self):
        Xtr, ytr = data(200, seed=0)
        Xte, yte = data(80, seed=1)
        pred = tt.fit_predict(Xtr, ytr, Xte, "ridge", 0)
        assert tt.bootstrap_ci(yte, pred, 200, 7) == tt.bootstrap_ci(yte, pred, 200, 7)

    def test_a_constant_resample_does_not_crash_it(self):
        """With a target that is mostly one value -- which is what a floor-
        heavy questionnaire column is -- some resamples have no variance at
        all and Spearman is undefined there. Those are skipped, not fatal."""
        y = np.zeros(40); y[:3] = 1.0
        pred = np.random.default_rng(0).normal(size=40)
        sc, lo, hi = tt.bootstrap_ci(y, pred, 300, 0)
        assert np.isfinite([sc, lo, hi]).all()
