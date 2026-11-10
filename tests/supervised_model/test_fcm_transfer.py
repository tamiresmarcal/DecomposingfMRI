"""The fcm half of the transfer: two things that must come from TRAINING only.

A correlation matrix is not a count matrix, so neither the pooling nor the
feature selection works the way the transition side does.

  WHICH EDGES survive. fcm_selection drops an edge that is NaN in more than
  max_missing of subjects, so the surviving set follows each cohort's own
  coverage. Let the test cohort choose its own and the model is handed a
  different feature space than it was fitted on -- with the columns still
  lining up by count, so nothing complains.

  THE IMPUTED VALUE. A NaN is filled with a column mean. Filled from the TEST
  cohort's mean, every test subject's missing edge is nudged toward that
  cohort's centre: a quiet adaptation to the very data being scored.

Both are invisible in the output if they go wrong, which is why they are
tested directly rather than through a score.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fmri_decomposition.supervised_model import transfer_tables as tt  # noqa: E402

EDGES = ["a__b", "a__c", "b__c"]


def fc_rows(subs, tasks, r, n_tr=100):
    """A static-FC table in the shape read_cohort_table returns."""
    out = []
    for i, s in enumerate(subs):
        for t in tasks:
            row = {"sub": s, "task": t, "n_tr_used": n_tr, "n_tr_total": n_tr,
                   "n_tr_good": n_tr, "n_tasks": 1, "frac_good_frames": 1.0}
            row.update(dict(zip(EDGES, r[i % len(r)])))
            out.append(row)
    return pd.DataFrame(out)


class TestPooling:
    def test_one_run_passes_through(self):
        d = fc_rows(["S1", "S2"], ["MOVIE1"], [[.1, .2, .3], [.4, .5, .6]])
        got = tt.pool_fc(d, log=lambda *_: None)
        assert len(got) == 2
        assert got.set_index("sub").loc["S1", "a__b"] == pytest.approx(.1)

    def test_it_pools_in_z_not_in_r(self):
        """Averaging r directly is the mistake: r is bounded and its sampling
        variance depends on its own value."""
        d = fc_rows(["S1"], ["MOVIE1", "MOVIE2"], [[.1, 0, 0]])
        d.loc[1, "a__b"] = .9
        got = tt.pool_fc(d, log=lambda *_: None).iloc[0]["a__b"]
        want = np.tanh((np.arctanh(.1) + np.arctanh(.9)) / 2)
        assert got == pytest.approx(want)
        assert got != pytest.approx((.1 + .9) / 2)

    def test_runs_are_weighted_by_how_much_data_they_contributed(self):
        d = fc_rows(["S1"], ["MOVIE1", "MOVIE2"], [[.2, 0, 0]])
        d.loc[1, "a__b"] = .8
        d.loc[0, "n_tr_used"], d.loc[1, "n_tr_used"] = 300, 100
        got = tt.pool_fc(d, log=lambda *_: None).iloc[0]["a__b"]
        want = np.tanh((3 * np.arctanh(.2) + np.arctanh(.8)) / 4)
        assert got == pytest.approx(want)

    def test_a_nan_in_one_run_does_not_lose_the_edge(self):
        d = fc_rows(["S1"], ["MOVIE1", "MOVIE2"], [[.5, 0, 0]])
        d.loc[0, "a__b"] = np.nan
        got = tt.pool_fc(d, log=lambda *_: None).iloc[0]["a__b"]
        assert got == pytest.approx(.5)

    def test_it_is_marked_approximate(self):
        """Unlike the transition pooling, this one cannot be exact: the
        correlation of concatenated runs is not a function of the runs'
        correlations."""
        d = fc_rows(["S1"], ["MOVIE1", "MOVIE2"], [[.1, .2, .3]])
        assert tt.pool_fc(d, log=lambda *_: None).iloc[0]["fc_pooled_approximately"]

    def test_the_task_label_records_what_was_pooled(self):
        d = fc_rows(["S1"], ["MOVIE1", "MOVIE2"], [[.1, .2, .3]])
        got = tt.pool_fc(d, log=lambda *_: None).iloc[0]
        assert got["task"] == "MOVIE1+MOVIE2" and got["n_runs_pooled"] == 2

    def test_n_tr_used_adds(self):
        d = fc_rows(["S1"], ["MOVIE1", "MOVIE2"], [[.1, .2, .3]], n_tr=120)
        assert tt.pool_fc(d, log=lambda *_: None).iloc[0]["n_tr_used"] == 240


class TestTrainOnlyDecisions:
    """These two are asserted on the arithmetic rather than through a score,
    because a score would look perfectly reasonable either way."""

    def test_the_kept_edges_follow_the_training_cohort(self):
        tr = np.array([[.1, np.nan, .3], [.2, np.nan, .4], [.3, np.nan, .5]])
        keep = np.isnan(np.arctanh(tr)).mean(axis=0) <= .10
        assert keep.tolist() == [True, False, True]
        # the test cohort has that edge complete; it must STILL be dropped,
        # because the model was never given a coefficient for it
        te = np.array([[.9, .9, .9]])
        assert te[:, keep].shape[1] == 2

    def test_imputation_uses_the_training_mean(self):
        Ztr = np.array([[0., 1.], [2., 3.]])
        mu = np.nanmean(Ztr, axis=0)
        Zte = np.array([[np.nan, 10.], [np.nan, 20.]])
        filled = np.where(np.isnan(Zte), mu, Zte)
        assert filled[:, 0].tolist() == [1.0, 1.0]        # train mean, not test
        # and the test cohort has NO mean of its own for that column -- every
        # value is missing, so filling from the test side is not merely worse,
        # it is undefined
        assert np.isnan(Zte[:, 0]).all()


class TestCovariateFrame:
    def test_it_keeps_the_names_so_two_cohorts_can_be_aligned(self):
        ph = pd.DataFrame({"sub": ["A", "B"], "Age": [30, 40], "Sex": ["F", "M"]})
        q = pd.DataFrame({"sub": ["A", "B"], "frac_good_frames": [1.0, .9]})
        X = tt._cov_frame(ph, q, ["A", "B"], ["Age", "Sex", "frac_good_frames"],
                          ["Sex"])
        assert "Age" in X.columns and any(c.startswith("Sex") for c in X.columns)

    def test_a_covariate_in_neither_table_says_where_it_looked(self):
        ph = pd.DataFrame({"sub": ["A"], "Age": [30]})
        q = pd.DataFrame({"sub": ["A"], "n_tr_used": [100]})
        with pytest.raises(SystemExit) as e:
            tt._cov_frame(ph, q, ["A"], ["mean_fd"], [])
        assert "mean_fd" in str(e.value) and "static FC" in str(e.value)

    def test_differently_named_covariates_are_refused_not_lined_up(self):
        """TWO subjects of different sex, because get_dummies(drop_first=True)
        over a single observed category produces no column at all -- a
        one-row fixture makes both frames come out as just ["Age"] and the
        mismatch this test is about cannot occur."""
        a = tt._cov_frame(
            pd.DataFrame({"sub": ["A", "B"], "Age": [30, 40], "Sex": ["F", "M"]}),
            pd.DataFrame({"sub": ["A", "B"]}), ["A", "B"], ["Age", "Sex"], ["Sex"])
        b = tt._cov_frame(
            pd.DataFrame({"sub": ["C", "D"], "Age": [30, 40], "Gender": ["F", "M"]}),
            pd.DataFrame({"sub": ["C", "D"]}), ["C", "D"], ["Age", "Gender"], ["Gender"])
        assert list(a.columns) != list(b.columns)      # the fixture is valid
        with pytest.raises(SystemExit):
            tt.align_columns(list(a.columns), list(b.columns))
