import numpy as np
import pandas as pd
import pytest

from fmri_decomposition.config import config_from_dict
from fmri_decomposition.dfc import (build_dfc_table, dfc_for_run,
                                    full_matrix_from_upper, pearson_upper)
from fmri_decomposition.io import edge_storage_mode


@pytest.fixture
def cfg(tmp_path):
    return config_from_dict({
        "cohort": "t", "tr": 1.0,
        "derivatives_root": str(tmp_path), "output_root": str(tmp_path / "out"),
        "atlases": ["x"],
        "windows": {"sizes_s": [30.0], "n_overlaps": 5},
        "stimulus": {"durations_s": {"movie": 200.0}},
    })


def toy_atlas(n_nodes=5):
    from fmri_decomposition.atlases.registry import AtlasSpec

    labels = pd.DataFrame({
        "index": np.arange(1, n_nodes + 1),
        "name": [f"P{i}" for i in range(n_nodes)],
        "hemi": ["B"] * n_nodes, "x": 0.0, "y": 0.0, "z": 0.0, "network": "n",
    })
    return AtlasSpec(name="toy", kind="labels", labels=labels)


class TestPearsonUpper:
    def test_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 6))
        expected = np.corrcoef(X, rowvar=False)[np.triu_indices(6, k=1)]
        assert np.allclose(pearson_upper(X), expected, atol=1e-6)

    def test_perfect_correlation_is_exactly_one(self):
        x = np.arange(50, dtype=float)
        X = np.column_stack([x, 2 * x + 3])
        assert pearson_upper(X)[0] == pytest.approx(1.0, abs=1e-9)

    def test_anticorrelation(self):
        x = np.random.default_rng(1).normal(size=100)
        assert pearson_upper(np.column_stack([x, -x]))[0] == pytest.approx(-1.0, abs=1e-9)

    def test_all_nan_parcel_gives_nan_edges_not_an_exception(self):
        """An empty parcel must poison only its own edges."""
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 4))
        X[:, 1] = np.nan
        r = pearson_upper(X)
        full = full_matrix_from_upper(r, 4)
        assert np.isnan(full[1]).all() or np.isnan(full[1, [0, 2, 3]]).all()
        assert np.isfinite(full[0, 2]) and np.isfinite(full[2, 3])

    def test_constant_column_gives_nan_not_divide_error(self):
        X = np.column_stack([np.ones(50), np.random.default_rng(3).normal(size=50)])
        r = pearson_upper(X)
        assert np.isnan(r[0])

    def test_fewer_than_two_samples_is_undefined(self):
        assert np.isnan(pearson_upper(np.zeros((1, 4)))).all()
        assert np.isnan(pearson_upper(np.zeros((0, 4)))).all()

    def test_output_is_float32_and_clipped(self):
        rng = np.random.default_rng(4)
        r = pearson_upper(rng.normal(size=(80, 5)))
        assert r.dtype == np.float32
        assert np.nanmax(np.abs(r)) <= 1.0

    def test_invariant_to_per_column_affine_rescaling(self):
        """The claim that lets stage 3 read parquet instead of the NIfTI."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(150, 7))
        scale, shift = rng.uniform(0.5, 3, 7), rng.normal(size=7) * 10
        assert np.allclose(pearson_upper(X), pearson_upper(X * scale + shift), atol=1e-9)

    def test_upper_triangle_ordering_is_row_major(self):
        X = np.random.default_rng(6).normal(size=(60, 4))
        r = pearson_upper(X)
        full = np.corrcoef(X, rowvar=False)
        assert r[0] == pytest.approx(full[0, 1], abs=1e-6)
        assert r[1] == pytest.approx(full[0, 2], abs=1e-6)
        assert r[3] == pytest.approx(full[1, 2], abs=1e-6)


def _frame(n_tr, n_nodes, seed=0, censored=(), zero_fill=True, tr=1.0, baseline=100.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_tr, n_nodes)) + baseline
    good = np.ones(n_tr, dtype=bool)
    if len(censored):
        good[list(censored)] = False
        if zero_fill:
            X[list(censored), :] = 0.0    # AFNI writes zeros at censored TRs
    df = pd.DataFrame(X, columns=[f"P{i}" for i in range(n_nodes)])
    df["t"] = np.arange(n_tr)
    df["time_s"] = df["t"] * tr
    df["stimulus_time_s"] = df["time_s"]
    df["good_frame"] = good
    df["run_idx"] = 0
    return df


class TestCensoringPolicy:
    def test_zero_filled_frames_inflate_every_edge_in_the_same_direction(self, cfg):
        """The reason pairwise deletion is not optional (addendum §3).

        A censored TR is zero in every parcel at once. On scaled BOLD (mean
        ~100) that is a synchronous excursion shared by every parcel, so it
        drives all edges toward +1 together. Bias, not noise.
        """
        atlas = toy_atlas(5)
        censored = list(range(10, 40, 3))
        df = _frame(200, 5, seed=7, censored=censored, zero_fill=True)

        naive = df.copy()
        naive["good_frame"] = True         # pretend nothing was censored
        res_naive = {w.qc["window_id"]: w for w in dfc_for_run(naive, atlas, cfg, 30.0, 200.0)}
        res_ok = {w.qc["window_id"]: w for w in dfc_for_run(df, atlas, cfg, 30.0, 200.0)}

        affected = [k for k, w in res_ok.items()
                    if w.qc["n_tr_effective"] < w.qc["n_tr_available"]]
        assert affected, "the fixture must censor frames inside some windows"
        r_naive = np.concatenate([res_naive[k].r for k in affected])
        r_correct = np.concatenate([res_ok[k].r for k in affected])

        assert np.nanmean(r_naive) > 0.6, "zero-filling should drive edges toward +1"
        assert abs(np.nanmean(r_correct)) < 0.15, "pairwise deletion should recover ~0"

        # Untouched windows must be identical either way -- the fix is local.
        clean = [k for k in res_ok if k not in affected]
        for k in clean[:5]:
            assert np.allclose(res_ok[k].r, res_naive[k].r, equal_nan=True)

    def test_the_bias_is_small_on_mean_centred_input(self, cfg):
        """The scale of the bias depends on how far zero is from the parcel mean.

        On an AFNI residual dataset (already mean-centred) a zero-filled TR
        sits near the mean and inflates little; on scaled BOLD it is a large
        shared excursion. Worth knowing which one a cohort ships, because it
        determines how badly the legacy outputs are affected.
        """
        atlas = toy_atlas(5)
        censored = list(range(10, 40, 3))
        df = _frame(200, 5, seed=7, censored=censored, zero_fill=True, baseline=0.0)
        naive = df.copy()
        naive["good_frame"] = True
        r_naive = np.concatenate([w.r for w in dfc_for_run(naive, atlas, cfg, 30.0, 200.0)])
        assert abs(np.nanmean(r_naive)) < 0.15

    def test_n_tr_effective_counts_only_good_frames(self, cfg):
        atlas = toy_atlas(4)
        df = _frame(120, 4, seed=8, censored=[0, 1, 2, 3, 4])
        res = dfc_for_run(df, atlas, cfg, 30.0, 120.0)
        first = res[0]
        assert first.qc["n_tr_available"] == 30
        assert first.qc["n_tr_effective"] == 25
        assert first.qc["frac_good_frames"] == pytest.approx(25 / 30)

    def test_window_with_too_few_good_frames_is_dropped(self, cfg):
        atlas = toy_atlas(4)
        df = _frame(120, 4, seed=9, censored=list(range(0, 29)))
        res = dfc_for_run(df, atlas, cfg, 30.0, 120.0)
        assert all(w.qc["window_id"] > 0 for w in res)


class TestWindowGridAcrossSubjects:
    def test_same_window_id_means_same_stimulus_interval(self, cfg):
        """Subject A's window 47 is subject B's window 47, by construction."""
        atlas = toy_atlas(4)
        a = _frame(200, 4, seed=10)
        b = _frame(200, 4, seed=11, censored=list(range(50, 80)))
        ra = {w.qc["window_id"]: w.qc["stimulus_start_s"] for w in dfc_for_run(a, atlas, cfg, 30.0, 200.0)}
        rb = {w.qc["window_id"]: w.qc["stimulus_start_s"] for w in dfc_for_run(b, atlas, cfg, 30.0, 200.0)}
        shared = set(ra) & set(rb)
        assert shared
        for k in shared:
            assert ra[k] == rb[k]

    def test_a_shifted_subject_still_lands_on_the_shared_grid(self, cfg):
        """A 7 s offset shifts which frames fall in a window, never the window."""
        atlas = toy_atlas(4)
        aligned = _frame(200, 4, seed=12)
        shifted = aligned.copy()
        shifted["stimulus_time_s"] = shifted["time_s"] + 7.0   # scanner started early

        res_a = {w.qc["window_id"]: w.qc for w in dfc_for_run(aligned, atlas, cfg, 30.0, 200.0)}
        res_s = {w.qc["window_id"]: w.qc for w in dfc_for_run(shifted, atlas, cfg, 30.0, 200.0)}

        for k in set(res_a) & set(res_s):
            assert res_a[k]["stimulus_start_s"] == res_s[k]["stimulus_start_s"]
        # The shifted subject contributes fewer frames to the first window and
        # its anchor is a different file row -- which is what n_tr_available is for.
        assert res_s[0]["n_tr_available"] == 23
        assert res_a[0]["n_tr_available"] == 30

    def test_frames_selected_for_a_window_lie_inside_its_interval(self, cfg):
        atlas = toy_atlas(4)
        df = _frame(200, 4, seed=19)
        for w in dfc_for_run(df, atlas, cfg, 30.0, 200.0):
            lo, hi = w.qc["stimulus_start_s"], w.qc["stimulus_end_s"]
            inside = df[(df["stimulus_time_s"] >= lo) & (df["stimulus_time_s"] < hi)]
            assert w.qc["n_tr_available"] == len(inside)
            assert w.qc["start_tr"] == int(inside["t"].iloc[0])

    def test_run_boundary_is_flagged_not_dropped(self, cfg):
        atlas = toy_atlas(4)
        df = _frame(200, 4, seed=13)
        df.loc[df["t"] >= 100, "run_idx"] = 1
        res = dfc_for_run(df, atlas, cfg, 30.0, 200.0)
        flags = [w.qc["crosses_run_boundary"] for w in res]
        assert any(flags) and not all(flags)

    def test_rank_deficiency_is_flagged(self, cfg):
        atlas = toy_atlas(60)                     # 60 nodes, 30-TR windows
        df = _frame(120, 60, seed=14)
        res = dfc_for_run(df, atlas, cfg, 30.0, 120.0)
        assert all(w.qc["rank_deficient"] for w in res)


class TestTable:
    def test_column_mode_names_edges_from_the_label_table(self, cfg):
        atlas = toy_atlas(4)
        res = dfc_for_run(_frame(120, 4, seed=15), atlas, cfg, 30.0, 120.0)
        table = build_dfc_table(res, atlas, cfg, 30.0, {"cohort": "t", "sub": "1", "task": "m"})
        assert "P0__P1" in table.column_names
        assert len([c for c in table.column_names if "__" in c]) == atlas.n_edges
        assert table.schema.metadata[b"edge_storage"] == b"columns"

    def test_packed_mode_above_the_threshold(self, cfg):
        cfg.windows.edge_column_threshold = 5
        atlas = toy_atlas(6)                       # 15 edges > 5
        res = dfc_for_run(_frame(120, 6, seed=16), atlas, cfg, 30.0, 120.0)
        table = build_dfc_table(res, atlas, cfg, 30.0, {"cohort": "t", "sub": "1", "task": "m"})
        assert "edges" in table.column_names
        assert table.schema.field("edges").type.list_size == 15

    def test_edges_are_stored_as_raw_r_not_fisher_z(self, cfg):
        atlas = toy_atlas(3)
        df = _frame(120, 3, seed=17)
        df["P1"] = df["P0"]                        # r = 1 exactly
        res = dfc_for_run(df, atlas, cfg, 30.0, 120.0)
        table = build_dfc_table(res, atlas, cfg, 30.0, {"cohort": "t", "sub": "1", "task": "m"})
        assert table.column("P0__P1").to_numpy()[0] == pytest.approx(1.0, abs=1e-6)
        assert table.schema.metadata[b"fisher_z_applied"] == b"false"

    def test_partition_keys_are_not_written_as_columns(self, cfg):
        """A key duplicated as a column must match its inferred type exactly.

        pyarrow reads `sub=01` as int32, which collides with a string column and
        silently destroys Cam-CAN ids. The path is the single source of truth.
        """
        atlas = toy_atlas(4)
        res = dfc_for_run(_frame(120, 4, seed=20), atlas, cfg, 30.0, 120.0)
        table = build_dfc_table(res, atlas, cfg, 30.0,
                                {"cohort": "t", "sub": "01", "task": "m"})
        for key in ("cohort", "atlas", "task", "sub", "window_s"):
            assert key not in table.column_names
        assert table.schema.metadata[b"sub"] == b"01"

    def test_mismatched_atlas_raises_rather_than_mis_slicing(self, cfg):
        """Legacy bug 2: a hardcoded parcel count silently mis-sliced."""
        df = _frame(60, 4, seed=18)
        with pytest.raises(ValueError, match="missing"):
            dfc_for_run(df, toy_atlas(9), cfg, 30.0, 60.0)


def test_edge_storage_threshold():
    assert edge_storage_mode(6_105) == "columns"      # Harvard-Oxford 111
    assert edge_storage_mode(21) == "columns"         # Yeo-7
    assert edge_storage_mode(499_500) == "list"       # Schaefer-1000


class TestPerSizeOverrides:
    """`windows.by_size` reaches stage 3, and the metadata records what ran."""

    def _cfg(self, tmp_path, **windows):
        base = {"sizes_s": [15.0, 30.0], "n_overlaps": 5}
        base.update(windows)
        return config_from_dict({
            "cohort": "t", "tr": 1.0,
            "derivatives_root": str(tmp_path), "output_root": str(tmp_path / "out"),
            "atlases": ["x"], "windows": base,
            "stimulus": {"durations_s": {"movie": 200.0}},
        })

    def _frame(self, n=200):
        rng = np.random.default_rng(7)
        df = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"P{i}" for i in range(5)])
        df["stimulus_time_s"] = np.arange(n, dtype=float)
        df["time_s"] = df["stimulus_time_s"]
        df["t"] = np.arange(n)
        df["good_frame"] = True
        df["run_idx"] = np.int8(0)
        return df

    def test_a_per_size_stride_changes_the_window_count(self, tmp_path):
        cfg = self._cfg(tmp_path, by_size={15.0: {"n_overlaps": 3}})
        df, atlas = self._frame(), toy_atlas()
        fine = dfc_for_run(df, atlas, cfg, 15.0, 200.0)
        coarse_stride = dfc_for_run(df, atlas, cfg, 30.0, 200.0)
        # 15 s at stride 5 s vs the cohort default of 3 s
        assert len(fine) == 38
        assert len(coarse_stride) == 29
        assert fine[1].window.start_s == 5.0

    def test_an_explicit_argument_beats_the_config(self, tmp_path):
        cfg = self._cfg(tmp_path, by_size={15.0: {"n_overlaps": 3}})
        results = dfc_for_run(self._frame(), toy_atlas(), cfg, 15.0, 200.0, n_overlaps=5)
        assert results[1].window.start_s == 3.0

    def test_the_stride_that_ran_is_recorded_in_the_metadata(self, tmp_path):
        """The output path carries only window_s, so the stride must be here."""
        cfg = self._cfg(tmp_path, by_size={15.0: {"n_overlaps": 3}})
        results = dfc_for_run(self._frame(), toy_atlas(), cfg, 15.0, 200.0)
        table = build_dfc_table(results, toy_atlas(), cfg, 15.0, {"cohort": "t"})
        assert table.schema.metadata[b"n_overlaps"] == b"3"

    def test_rank_deficient_agrees_with_the_shared_predicate(self, tmp_path):
        """Stage 3's flag and the CLI's prediction are the same inequality."""
        from fmri_decomposition.windows import is_rank_deficient

        cfg = self._cfg(tmp_path)
        atlas = toy_atlas(n_nodes=5)
        df = self._frame()
        for window_s in (15.0, 30.0):
            for res in dfc_for_run(df, atlas, cfg, window_s, 200.0):
                assert res.qc["rank_deficient"] == is_rank_deficient(
                    res.qc["n_tr_effective"], atlas.n_nodes)
