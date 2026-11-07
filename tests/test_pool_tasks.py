"""Pooling four movie runs into one row: recomputed, not averaged.

HCP gives four movies and four rests per subject and the selections score one
row per subject. Averaging the columns would be wrong for most of them: an
entropy rate averaged over runs is not the entropy rate of anything, a switch
rate is a ratio whose denominators differ, and occupancy is a fraction of a
different number of windows each time.

The test that matters is the invariant: pooling the per-run rows of a subject
must give the same numbers as computing the row once from that subject's runs
concatenated. Everything below is a case of that.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transfer_tables as tt  # noqa: E402
from fmri_decomposition.transitions import _entropy_rate  # noqa: E402

K = 3
CELLS = [f"{i}->{j}" for i in range(K) for j in range(K)]


def row_from_counts(sub, task, M, n_windows, stride=1.0, dispersion=1.0):
    """One run's row, built the way transitions.py builds it."""
    M = np.asarray(M, float)
    n_tr = M.sum()
    n_change = n_tr - np.trace(M)
    occ = np.array([M[k].sum() + M[:, k].sum() for k in range(K)])
    occ = occ / occ.sum() if occ.sum() else occ
    r = dict(zip(CELLS, (M / n_tr).ravel()))
    r.update({
        "sub": sub, "task": task, "n_states": K,
        "n_windows": n_windows, "n_transitions": n_tr,
        "n_transitions_independent": n_windows // 5,
        "n_states_visited": int((occ > 0).sum()),
        "n_distinct_transitions": int((M > 0).sum()),
        "switch_rate": n_change / n_tr,
        "self_transition_rate": 1 - n_change / n_tr,
        "switches_per_min": n_change / (n_windows * stride / 60),
        "mean_dwell_s": n_windows / (n_change + 1) * stride,
        "entropy_rate_bits": _entropy_rate(M),
        "dispersion": dispersion,
    })
    r.update({f"occ_{k}": float(occ[k]) for k in range(K)})
    return r


def two_runs(stride_a=1.0, stride_b=1.0):
    A = np.array([[10., 2, 1], [1, 8, 3], [0, 2, 7]])
    B = np.array([[4., 1, 0], [2, 9, 1], [1, 0, 6]])
    d = pd.DataFrame([row_from_counts("S1", "MOVIE1", A, 40, stride_a, 1.0),
                      row_from_counts("S1", "MOVIE2", B, 30, stride_b, 2.0)])
    return d, A, B


class TestExactness:
    def test_the_cells_are_the_summed_counts(self):
        d, A, B = two_runs()
        got = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        want = (A + B) / (A + B).sum()
        assert np.allclose([got[c] for c in CELLS], want.ravel())

    def test_n_transitions_adds(self):
        d, A, B = two_runs()
        got = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        assert got["n_transitions"] == A.sum() + B.sum()

    def test_the_switch_rate_is_recomputed_not_averaged(self):
        d, A, B = two_runs()
        got = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        M = A + B
        want = (M.sum() - np.trace(M)) / M.sum()
        assert got["switch_rate"] == pytest.approx(want)
        # and it is NOT the mean of the two rates
        assert got["switch_rate"] != pytest.approx(d["switch_rate"].mean())

    def test_the_entropy_rate_is_recomputed_from_the_pooled_matrix(self):
        d, A, B = two_runs()
        got = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        assert got["entropy_rate_bits"] == pytest.approx(_entropy_rate(A + B))

    def test_occupancy_sums_to_one_and_weights_by_windows(self):
        d, _, _ = two_runs()
        got = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        occ = [got[f"occ_{k}"] for k in range(K)]
        assert sum(occ) == pytest.approx(1.0)
        w = d["n_windows"].to_numpy(float)
        want0 = (d["occ_0"].to_numpy() * w).sum() / w.sum()
        assert occ[0] == pytest.approx(want0)

    def test_mean_dwell_uses_episodes_not_a_mean_of_means(self):
        d, A, B = two_runs()
        got = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        sw = (A.sum() - np.trace(A)) + (B.sum() - np.trace(B))
        want = (40 + 30) / (sw + 2) * 1.0
        assert got["mean_dwell_s"] == pytest.approx(want)
        assert got["mean_dwell_s"] != pytest.approx(d["mean_dwell_s"].mean())

    def test_switches_per_min_pools_switches_over_minutes(self):
        d, A, B = two_runs()
        got = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        sw = (A.sum() - np.trace(A)) + (B.sum() - np.trace(B))
        assert got["switches_per_min"] == pytest.approx(sw / ((40 + 30) / 60))


class TestRefusals:
    def test_mixed_strides_are_refused(self):
        """A rate pooled over runs sampled at different strides is not a
        rate, and nothing in the table would show it had happened."""
        d, _, _ = two_runs(stride_a=1.0, stride_b=2.47)
        with pytest.raises(SystemExit, match="different"):
            tt.pool_tasks(d, K, log=lambda *_: None)

    def test_dispersion_is_flagged_as_approximate(self):
        """It needs the latents, not this table: the pooled value uses each
        run's own centroid rather than one over all of them."""
        d, _, _ = two_runs()
        got = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        assert got["dispersion_pooled_approximately"]
        assert got["dispersion"] == pytest.approx((1.0 * 40 + 2.0 * 30) / 70)


class TestShape:
    def test_one_row_per_subject(self):
        d, _, _ = two_runs()
        d2 = pd.concat([d, d.assign(sub="S2")], ignore_index=True)
        got = tt.pool_tasks(d2, K, log=lambda *_: None)
        assert sorted(got["sub"]) == ["S1", "S2"]

    def test_a_subject_with_one_run_passes_through(self):
        d, _, _ = two_runs()
        one = d.iloc[[0]].copy()
        got = tt.pool_tasks(one, K, log=lambda *_: None)
        assert len(got) == 1
        assert got.iloc[0]["switch_rate"] == pytest.approx(one.iloc[0]["switch_rate"])

    def test_the_task_label_records_what_was_pooled(self):
        d, _, _ = two_runs()
        got = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        assert got["task"] == "MOVIE1+MOVIE2"
        assert got["n_runs_pooled"] == 2
