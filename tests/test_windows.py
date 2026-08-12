import numpy as np
import pytest

from fmri_decomposition.windows import (Window, assign_window_ids,
                                        make_index_windows, make_stimulus_grid,
                                        n_windows, stride_seconds,
                                        window_tr_from_seconds)


class TestWindowTr:
    @pytest.mark.parametrize(
        "window_s,tr,expected",
        [
            (30, 1.0, 30), (60, 1.0, 60), (300, 1.0, 300),
            (30, 1.49, 20), (60, 1.49, 40), (120, 1.49, 81), (300, 1.49, 201),
            (30, 2.0, 15), (60, 2.0, 30),
            (30, 2.47, 12), (60, 2.47, 24), (120, 2.47, 49), (300, 2.47, 121),
        ],
    )
    def test_matches_the_documented_table(self, window_s, tr, expected):
        """The realized-n_tr table in the handoff is a contract, not a comment."""
        assert window_tr_from_seconds(window_s, tr) == expected

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            window_tr_from_seconds(30, 0)
        with pytest.raises(ValueError):
            window_tr_from_seconds(-1, 1.0)


class TestStimulusGrid:
    def test_starts_are_uniformly_spaced_and_unique(self):
        grid = make_stimulus_grid(300, 60, n_overlaps=3)
        starts = [w.start_s for w in grid]
        assert len(starts) == len(set(starts)), "duplicate window starts"
        diffs = np.diff(starts)
        assert np.allclose(diffs, 20.0), f"non-uniform stride: {set(diffs)}"

    def test_regression_legacy_overlap_bug(self):
        """Legacy rebound the loop variable, giving 0,5,15,15,20,30,30,...

        Offsets compounded so the +10 offset was never sampled (effective
        overlap 2, not 3) and every 15k start was duplicated.
        """
        grid = make_stimulus_grid(90, 15, n_overlaps=3, drop_incomplete=True)
        starts = [w.start_s for w in grid]
        assert starts == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
        assert 10.0 in starts, "the +10 offset must be sampled"
        assert len(starts) == len(set(starts))

    def test_window_ids_are_sequential_from_zero(self):
        grid = make_stimulus_grid(200, 30, n_overlaps=5)
        assert [w.window_id for w in grid] == list(range(len(grid)))

    def test_grid_is_identical_regardless_of_subject(self):
        """Two subjects, same stimulus -> byte-identical grids. The point of §4."""
        a = make_stimulus_grid(5470.0, 120, 5)
        b = make_stimulus_grid(5470.0, 120, 5)
        assert a == b

    def test_short_stimulus_yields_no_windows(self):
        assert make_stimulus_grid(200, 300, 5) == []
        assert n_windows(200, 300, 5) == 0

    def test_exactly_one_window_when_duration_equals_window(self):
        grid = make_stimulus_grid(300.0, 300.0, 5)
        assert len(grid) == 1 and grid[0].start_s == 0.0

    def test_float_stride_does_not_drift(self):
        """30/7 is not representable; the last start must still be exact."""
        grid = make_stimulus_grid(600, 30, n_overlaps=7)
        expected_last = (len(grid) - 1) * (30 / 7)
        assert grid[-1].start_s == pytest.approx(expected_last, abs=1e-9)
        assert grid[-1].end_s <= 600 + 1e-9

    def test_drop_incomplete_false_covers_the_tail(self):
        kept = make_stimulus_grid(100, 30, 5, drop_incomplete=True)
        allw = make_stimulus_grid(100, 30, 5, drop_incomplete=False)
        assert len(allw) > len(kept)
        assert allw[-1].start_s < 100

    def test_stride_seconds_rejects_zero_overlaps(self):
        with pytest.raises(ValueError):
            stride_seconds(30, 0)


class TestMembership:
    def test_interval_is_half_open(self):
        w = Window(0, 0.0, 30.0)
        t = np.array([-0.1, 0.0, 15.0, 29.9, 30.0, 30.1])
        assert w.contains(t).tolist() == [False, True, True, True, False, False]

    def test_no_frame_lands_in_two_adjacent_windows_twice(self):
        """With n_overlaps=k an interior frame belongs to exactly k windows."""
        grid = make_stimulus_grid(300, 30, n_overlaps=5)
        t = np.arange(0, 300, 1.0)
        counts = np.zeros(len(t), dtype=int)
        for w in grid:
            counts += w.contains(t)
        interior = counts[30:250]
        assert set(np.unique(interior)) == {5}

    def test_assign_window_ids_agrees_with_contains(self):
        window_s, n_overlaps = 30.0, 5
        grid = make_stimulus_grid(300, window_s, n_overlaps)
        t = np.array([0.0, 5.9, 6.0, 29.999, 30.0, 123.4])
        ids = assign_window_ids(t, window_s, n_overlaps)
        for ti, got in zip(t, ids):
            expected = [w.window_id for w in grid if w.contains(np.array([ti]))[0]]
            assert sorted(set(got) & set(w.window_id for w in grid)) == expected


class TestIndexWindows:
    def test_uniform_stride_no_duplicates(self):
        wins = make_index_windows(n_tr=100, window_tr=30, n_overlaps=5)
        starts = [s for s, _ in wins]
        assert starts == list(range(0, 71, 6))
        assert len(starts) == len(set(starts))

    def test_rejects_degenerate_window(self):
        with pytest.raises(ValueError):
            make_index_windows(100, 1)
