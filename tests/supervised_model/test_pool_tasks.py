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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fmri_decomposition.supervised_model import transfer_tables as tt  # noqa: E402
from fmri_decomposition.unsupervised_model.transitions import _entropy_rate  # noqa: E402
from fmri_decomposition.io import tree_root

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


class TestTheStrideComesFromItsColumn:
    """`transitions.py` writes `stride_s` into every row. Pooling used to
    recover it by inverting switches_per_min instead:

        stride = switches / switches_per_min * 60 / n_windows

    which is 0/0 for a subject whose runs record NO switch. That gave a NaN
    stride, hence NaN mean_dwell_s and switches_per_min, hence two of the
    three columns the transfer then had to impute -- for a quantity sitting
    in the frame the whole time.
    """

    def frozen(self, stride=2.0, column=True):
        """A subject whose two runs never leave state 0: switches_per_min = 0,
        so the inversion has nothing to invert."""
        A = np.zeros((K, K)); A[0, 0] = 20
        B = np.zeros((K, K)); B[0, 0] = 15
        d = pd.DataFrame([row_from_counts("S1", "MOVIE1", A, 40, stride),
                          row_from_counts("S1", "MOVIE2", B, 30, stride)])
        assert (d["switches_per_min"] == 0).all()
        if column:
            d["stride_s"] = stride
        return d

    def test_a_subject_who_never_switched_still_gets_its_dwell(self):
        out = tt.pool_tasks(self.frozen(), K, log=lambda *_: None)
        assert len(out) == 1
        r = out.iloc[0]
        assert np.isfinite(r["mean_dwell_s"]), "mean_dwell_s is still NaN"
        assert np.isfinite(r["switches_per_min"])
        # 70 windows, one episode per run -> 70 / 2 episodes x 2 s
        assert r["mean_dwell_s"] == pytest.approx(70 / 2 * 2.0)
        assert r["switches_per_min"] == pytest.approx(0.0)

    def test_without_the_column_it_is_still_nan(self):
        """The old behaviour, kept only for a table that predates stride_s."""
        out = tt.pool_tasks(self.frozen(column=False), K, log=lambda *_: None)
        assert np.isnan(out.iloc[0]["mean_dwell_s"])

    def test_the_column_wins_over_the_inversion(self):
        """If the two disagreed, the recorded value is the one that is not a
        reconstruction."""
        d, A, B = two_runs()
        d["stride_s"] = 3.0                      # the rows were built at 1.0
        out = tt.pool_tasks(d, K, log=lambda *_: None)
        w = d["n_windows"].sum()
        sw = (d["switch_rate"] * d["n_transitions"]).sum()
        assert out.iloc[0]["mean_dwell_s"] == pytest.approx(w / (sw + 2) * 3.0)

    def test_mixed_strides_still_refuse(self):
        d, _, _ = two_runs()
        d["stride_s"] = [1.0, 2.0]
        with pytest.raises(SystemExit) as e:
            tt.pool_tasks(d, K, log=lambda *_: None)
        assert "different" in str(e.value) and "stride_s column" in str(e.value)

    def test_a_normal_subject_is_unchanged_by_the_switch(self):
        """The column and the inversion agree where the inversion works, so
        nothing about an ordinary subject moves."""
        d, _, _ = two_runs()
        without = tt.pool_tasks(d.copy(), K, log=lambda *_: None)
        with_col = tt.pool_tasks(d.assign(stride_s=1.0), K, log=lambda *_: None)
        for c in ("mean_dwell_s", "switches_per_min", "entropy_rate_bits"):
            assert without.iloc[0][c] == pytest.approx(with_col.iloc[0][c])


class TestTheDispersionComesFromTheLatents:
    """`dispersion` is the one dynamic the transition table cannot pool.

    Everything else in that table is counts, and counts add. Dispersion is the
    mean distance from the subject's centroid in latent space, and the table
    kept only the per-run answer -- so a mean of two runs' dispersions is the
    mean distance from EACH run's centroid, which is not the distance from the
    centroid of both runs together. For HCP, four runs per condition, that gap
    is the quantity itself, not a rounding error.

    So the coordinates are read back from the latents. These tests pin the
    invariant that matters -- the pooled value equals the value computed once
    over the subject's windows concatenated -- and the two things that make
    reading them affordable and honest: one read per cell however many state
    sets ask, and a per-row flag saying which of the two answers a row holds.
    """

    def _tree(self, tmp_path, coords_a, coords_b, boundary_b=None):
        """A latents tree with two MOVIE runs of one subject, as stage 3 writes it."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        for task, xyz, flag in (("MOVIE1", coords_a, None),
                                ("MOVIE2", coords_b, boundary_b)):
            xyz = np.asarray(xyz, float)
            d = {"cohort": ["hcp"] * len(xyz), "task": [task] * len(xyz),
                 "sub": ["S1"] * len(xyz),
                 "window_id": list(range(len(xyz))),
                 "crosses_run_boundary": (list(flag) if flag is not None
                                          else [False] * len(xyz))}
            for k, c in enumerate(("pca0/3", "pca1/3", "pca2/3")):
                d[c] = xyz[:, k]
            p = (tree_root(tmp_path, "latents") / "atlas=yeo7" / "window_s=-1"
                 / "cohort=hcp" / f"task={task}")
            p.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(d), p / "data.parquet")
        return tmp_path

    @staticmethod
    def _dispersion(xyz):
        xyz = np.asarray(xyz, float)
        return float(np.linalg.norm(xyz - xyz.mean(0), axis=1).mean())

    def test_it_is_the_value_over_the_runs_concatenated(self, tmp_path):
        rng = np.random.default_rng(0)
        A, B = rng.normal(size=(40, 3)), rng.normal(size=(30, 3)) + 5.0
        self._tree(tmp_path, A, B)
        got = tt.exact_dispersion(tmp_path, "yeo7", -1.0, [("hcp", "MOVIE1"),
                                                           ("hcp", "MOVIE2")])
        assert got["S1"] == pytest.approx(self._dispersion(np.vstack([A, B])))
        # And it is NOT the per-run answer, which is the whole point: the two
        # clouds sit 5 units apart, so each run's own dispersion is small and
        # the real one is dominated by the distance between them.
        assert got["S1"] > 2 * self._dispersion(A)

    def test_pool_tasks_uses_it_and_says_so(self, tmp_path):
        rng = np.random.default_rng(1)
        A, B = rng.normal(size=(40, 3)), rng.normal(size=(30, 3)) + 5.0
        self._tree(tmp_path, A, B)
        d, _, _ = two_runs()
        d["cohort"], d["atlas"], d["window_s"] = "hcp", "yeo7", -1.0
        disp = tt.exact_dispersion(tmp_path, "yeo7", -1.0,
                                   [("hcp", "MOVIE1"), ("hcp", "MOVIE2")])
        got = tt.pool_tasks(d, K, log=lambda *_: None,
                            dispersion=disp).iloc[0]
        assert got["dispersion"] == pytest.approx(self._dispersion(np.vstack([A, B])))
        assert not got["dispersion_pooled_approximately"]
        # Unchanged without it: the fallback stays, and still admits to itself.
        approx = tt.pool_tasks(d, K, log=lambda *_: None).iloc[0]
        assert approx["dispersion"] == pytest.approx((1.0 * 40 + 2.0 * 30) / 70)
        assert approx["dispersion_pooled_approximately"]

    def test_a_subject_the_latents_do_not_cover_falls_back(self, tmp_path):
        """A per-ROW flag, not a per-frame one.

        `dispersion` can legitimately miss a subject -- a latents file pruned,
        a run the transition table has and the latents do not. That subject
        must still get a row, with the approximation and with the flag that
        says it is one, rather than NaN or a refusal that loses everyone else.
        """
        rng = np.random.default_rng(2)
        self._tree(tmp_path, rng.normal(size=(40, 3)), rng.normal(size=(30, 3)))
        d, A, B = two_runs()
        other = pd.DataFrame([row_from_counts("S2", "MOVIE1", A, 40, 1.0, 3.0),
                              row_from_counts("S2", "MOVIE2", B, 30, 1.0, 4.0)])
        d = pd.concat([d, other], ignore_index=True)
        disp = tt.exact_dispersion(tmp_path, "yeo7", -1.0,
                                   [("hcp", "MOVIE1"), ("hcp", "MOVIE2")])
        got = tt.pool_tasks(d, K, log=lambda *_: None,
                            dispersion=disp).set_index("sub")
        assert not got.loc["S1", "dispersion_pooled_approximately"]
        assert got.loc["S2", "dispersion_pooled_approximately"]
        assert got.loc["S2", "dispersion"] == pytest.approx((3.0 * 40 + 4.0 * 30) / 70)

    def test_a_boundary_window_is_dropped_as_transitions_drops_it(self, tmp_path):
        """The same filter, or the two answers disagree for no reason but it.

        `subject_transitions` drops a window that straddles a run boundary
        outright -- its label was computed across a discontinuity. Keeping it
        here would make the pooled dispersion disagree with the table's own
        value for every single-run subject, which is the one case where both
        are computed from the same windows and must match.
        """
        rng = np.random.default_rng(3)
        A, B = rng.normal(size=(40, 3)), rng.normal(size=(30, 3))
        flag = np.zeros(len(B), dtype=bool)
        flag[:5] = True
        self._tree(tmp_path, A, B, boundary_b=flag)
        got = tt.exact_dispersion(tmp_path, "yeo7", -1.0,
                                  [("hcp", "MOVIE1"), ("hcp", "MOVIE2")])
        assert got["S1"] == pytest.approx(
            self._dispersion(np.vstack([A, B[~flag]])))

    def test_one_read_serves_every_state_set(self, tmp_path):
        """Memoised on the cell, which is what makes this affordable.

        The coordinates depend on (atlas, window_s, cohort, task) and NOT on
        the state set, so the 282 sets of one cell share one read. Without the
        cache this would be 282 reads of the same file per condition, and
        reading the latents for dispersion would cost more than the fits.
        """
        rng = np.random.default_rng(4)
        self._tree(tmp_path, rng.normal(size=(40, 3)), rng.normal(size=(30, 3)))
        pairs = [("hcp", "MOVIE1"), ("hcp", "MOVIE2")]
        tt.exact_dispersion(tmp_path, "yeo7", -1.0, pairs)
        before = tt._dispersion_cell.cache_info()
        for _ in range(5):
            tt.exact_dispersion(tmp_path, "yeo7", -1.0, pairs)
        after = tt._dispersion_cell.cache_info()
        assert after.misses == before.misses
        assert after.hits == before.hits + 5
        # And the order the pairs arrive in is not part of the key: `pool_fn`
        # builds them from a set, so two cells that read the same files must
        # not read them twice.
        tt.exact_dispersion(tmp_path, "yeo7", -1.0, list(reversed(pairs)))
        assert tt._dispersion_cell.cache_info().misses == before.misses

    def test_missing_latents_name_the_paths_and_the_alternative(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            tt.exact_dispersion(tmp_path, "yeo7", -1.0, [("hcp", "MOVIE1")])
        msg = str(e.value)
        assert "window_s=-1" in msg and "task=MOVIE1" in msg
        assert "dispersion=None" in msg
