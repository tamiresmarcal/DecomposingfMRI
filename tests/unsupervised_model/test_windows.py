import numpy as np
import pytest

from fmri_decomposition.windows import (Window, assign_window_ids,
                                        is_rank_deficient, make_index_windows,
                                        make_stimulus_grid,
                                        min_window_s_for_nodes, n_windows,
                                        stride_seconds, window_tr_from_seconds)


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


class TestPerSizeOverrides:
    """`windows.by_size`: a window size that runs on only some atlases.

    15 s at TR=1 gives 15 samples. That is a usable estimate of yeo7's 21
    edges and a meaningless one of Harvard-Oxford's 6,105, so the size has to
    be able to reach one atlas and not the other.
    """

    def _cfg(self, **windows):
        from fmri_decomposition.config import config_from_dict

        base = {"sizes_s": [15, 30, 300], "n_overlaps": 5}
        base.update(windows)
        return config_from_dict({
            "cohort": "c", "tr": 1.0, "derivatives_root": "/d", "output_root": "/o",
            "atlases": ["harvardoxford", "yeo7", "networks"], "windows": base,
        })

    def test_unrestricted_sizes_run_on_every_atlas(self):
        cfg = self._cfg()
        assert cfg.windows.atlases_for(30, cfg.atlases) == cfg.atlases
        assert cfg.windows.overlaps_for(30) == 5

    def test_a_restricted_size_reaches_only_the_named_atlases(self):
        cfg = self._cfg(by_size={15: {"atlases": ["yeo7", "networks"]}})
        assert cfg.windows.atlases_for(15, cfg.atlases) == ["yeo7", "networks"]
        assert cfg.windows.atlases_for(300, cfg.atlases) == cfg.atlases

    def test_the_restriction_preserves_the_cohorts_atlas_order(self):
        cfg = self._cfg(by_size={15: {"atlases": ["networks", "yeo7"]}})
        assert cfg.windows.atlases_for(15, cfg.atlases) == ["yeo7", "networks"]

    def test_a_size_can_carry_its_own_stride(self):
        cfg = self._cfg(by_size={15: {"n_overlaps": 3}})
        assert cfg.windows.overlaps_for(15) == 3
        assert cfg.windows.overlaps_for(30) == 5

    def test_int_and_float_and_string_keys_all_resolve(self):
        for key in (15, 15.0, "15"):
            cfg = self._cfg(by_size={key: {"n_overlaps": 3}})
            assert cfg.windows.overlaps_for(15) == 3, key

    def test_an_override_for_a_size_that_never_runs_is_an_error(self):
        from fmri_decomposition.config import ConfigError

        with pytest.raises(ConfigError, match="not in"):
            self._cfg(by_size={16: {"atlases": ["yeo7"]}})

    def test_an_unknown_atlas_in_a_restriction_is_an_error(self):
        from fmri_decomposition.config import ConfigError

        with pytest.raises(ConfigError, match="yeo17"):
            self._cfg(by_size={15: {"atlases": ["yeo17"]}})

    def test_an_empty_atlas_list_is_an_error_not_a_silent_no_op(self):
        from fmri_decomposition.config import ConfigError

        with pytest.raises(ConfigError, match="empty"):
            self._cfg(by_size={15: {"atlases": []}})

    def test_a_typo_inside_the_override_is_an_error(self):
        from fmri_decomposition.config import ConfigError

        with pytest.raises(ConfigError, match="atlasses"):
            self._cfg(by_size={15: {"atlasses": ["yeo7"]}})


class TestFineApertureRowCount:
    """The row count is the thing to check before launching, not after.

    A 5,470 s film at TR=1: 300 s windows give 87 per subject, 15 s give
    1,819 -- 21x the rows for the same data.
    """

    @pytest.mark.parametrize("window_s,n_overlaps,expected", [
        (300, 5, 87), (120, 5, 223), (60, 5, 451), (30, 5, 907), (15, 5, 1819),
        (15, 3, 1092),      # a coarser stride at the fine aperture: ~40% fewer
    ])
    def test_windows_per_subject_on_a_full_length_film(self, window_s, n_overlaps, expected):
        assert n_windows(5470.0, window_s, n_overlaps) == expected

    def test_the_fine_aperture_is_about_21x_the_coarsest(self):
        fine = n_windows(5470.0, 15, 5)
        coarse = n_windows(5470.0, 300, 5)
        assert 20 < fine / coarse < 22


class TestRankDeficiency:
    """The arithmetic behind restricting 15 s to the coarse atlases."""

    @pytest.mark.parametrize("window_s,n_nodes,expected", [
        # ds002837, TR=1: window_s seconds == window_s samples.
        (15, 111, True),    # Harvard-Oxford: 14 samples, 111 nodes, 6,105 edges
        (30, 111, True),    # already true of the grid that runs today
        (60, 111, True),
        (120, 111, False),
        (300, 111, False),
        (15, 7, False),     # yeo7
        (15, 14, False),    # the coordinate networks, by exactly one sample
    ])
    def test_ds002837_grid(self, window_s, n_nodes, expected):
        assert is_rank_deficient(window_tr_from_seconds(window_s, 1.0), n_nodes) is expected

    def test_the_14_node_atlas_sits_exactly_on_the_boundary(self):
        """15 samples clear 14 nodes by one. 14 samples would not.

        Worth knowing before censoring is ever switched on for this cohort:
        one dropped frame inside a 15 s window flips these to rank_deficient.
        """
        assert not is_rank_deficient(15, 14)
        assert is_rank_deficient(14, 14)


class TestMinWindowForNodes:
    """The general form of "short windows are only for the coarse atlases".

    The instinct is right and a fixed number of seconds is the wrong shape for
    it: the threshold is a property of (atlas, TR), not a constant.
    """

    @pytest.mark.parametrize("n_nodes,tr,expected", [
        (7, 1.0, 7.5),        # yeo7 on ds002837
        (14, 1.0, 14.5),      # the coordinate networks
        (111, 1.0, 111.5),    # Harvard-Oxford -- so 30 s and 60 s are past it
        (254, 1.0, 254.5),    # networks_nodes
        (7, 1.49, 11.175),    # the same atlas on CNeuroMod
        (111, 1.49, 166.135),
    ])
    def test_the_floor_is_derived_from_the_atlas_and_the_tr(self, n_nodes, tr, expected):
        assert min_window_s_for_nodes(n_nodes, tr) == pytest.approx(expected)

    @pytest.mark.parametrize("n_nodes,tr", [(7, 1.0), (14, 1.0), (111, 1.0),
                                            (14, 1.49), (111, 2.47)])
    def test_it_is_exactly_the_inverse_of_the_rank_rule(self, n_nodes, tr):
        floor_s = min_window_s_for_nodes(n_nodes, tr)
        assert not is_rank_deficient(window_tr_from_seconds(floor_s, tr), n_nodes)
        assert is_rank_deficient(window_tr_from_seconds(floor_s - tr, tr), n_nodes)

    def test_a_fixed_30s_rule_would_have_been_wrong_in_both_directions(self):
        """30 s admits Harvard-Oxford, which it should not, and would exclude
        yeo7 at 15 s, which is fine there."""
        assert min_window_s_for_nodes(111, 1.0) > 30.0    # 30 s is already too short
        assert min_window_s_for_nodes(7, 1.0) < 15.0      # 15 s is comfortably enough


class TestRankPolicy:
    def _cfg(self, policy):
        from fmri_decomposition.config import config_from_dict

        return config_from_dict({
            "cohort": "c", "tr": 1.0, "derivatives_root": "/d", "output_root": "/o",
            "atlases": ["harvardoxford", "yeo7"],
            "windows": {"sizes_s": [15, 30, 300], "rank_policy": policy},
        })

    def test_warn_is_the_default_so_the_existing_grid_is_unchanged(self):
        from fmri_decomposition.config import config_from_dict

        cfg = config_from_dict({"cohort": "c", "tr": 1.0, "derivatives_root": "/d",
                                "output_root": "/o"})
        assert cfg.windows.rank_policy == "warn"

    def test_only_warn_and_skip_are_accepted(self):
        from fmri_decomposition.config import ConfigError

        self._cfg("warn"), self._cfg("skip")
        with pytest.raises(ConfigError, match="rank_policy"):
            self._cfg("ignore")
