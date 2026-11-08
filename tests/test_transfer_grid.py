"""The grid moved to the cluster: sharding, provenance, and the reader's refusals.

The fits are ~9,000 on every training subject, so they happen in
tools/transfer_grid.py as a SLURM array and the notebook reads parquet. That
buys three new ways to be quietly wrong, and each has a class here.

1. A HALF-FINISHED ARRAY reads as a perfectly valid frame -- just a smaller
   multiverse. "The best of 282 state sets" from the 120 that happened to
   finish is a different claim in the same words, so the reader refuses unless
   every shard a frame says exists is on disk.
2. SHARDS CUT FROM DIFFERENT GRIDS. A shard is `sets.iloc[i::n]` of a sorted
   list, so stage 5a finishing another cohort mid-array renumbers the sets and
   the shards silently overlap and leave gaps. Hence `grid_hash`.
3. NaN REACHING THE ESTIMATOR. Pooling leaves `mean_dwell_s` undefined for a
   subject whose runs record no switch, and RidgeCV reports that as
   `Input X contains NaN` from inside `.predict` -- seven frames from the
   column that caused it. Filled from the TRAINING mean, never the test
   cohort's.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import hads_tables as ht  # noqa: E402
import transfer_tables as tt  # noqa: E402
from test_transfer_wiring import (ATLAS, TE, TR, WIN, studies,  # noqa: E402
                                  write_summary, write_transitions)
from test_pool_tasks import K  # noqa: E402


def sets_frame(n=7):
    return pd.DataFrame({
        "atlas": ["yeo7"] * n,
        "window_s": [-1.0, 30.0, 60.0, 120.0, 300.0, -1.0, 30.0][:n],
        "states": [f"HMM1_pca3_{8 + i}" for i in range(n)],
    })


# -------------------------------------------------------------- imputation ---
class TestImputeFromTrain:
    def test_a_test_nan_is_filled_with_the_training_mean(self):
        Xtr = np.array([[1., 2.], [3., 4.]])
        Xte = np.array([[np.nan, 9.]])
        a, b, names, blocks, info = impute = tt.impute_from_train(
            Xtr, Xte, ["x", "y"], ["cells", "cells"], log=lambda *_: None)
        assert b[0, 0] == 2.0                      # mean of 1 and 3, not of 9
        assert info["n_imputed_test"] == 1 and info["n_imputed_train"] == 0
        assert names == ["x", "y"] and len(impute) == 5

    def test_the_test_cohorts_own_mean_is_never_used(self):
        """The hazard: filled from HCP's mean, every test subject's missing
        value is nudged toward the centre of the data being scored."""
        Xtr = np.array([[0.], [0.]])
        Xte = np.array([[100.], [np.nan]])
        _, b, *_ = tt.impute_from_train(Xtr, Xte, ["x"], ["cells"],
                                        log=lambda *_: None)
        assert b[1, 0] == 0.0                      # train's mean, not 100

    def test_a_training_nan_is_filled_too(self):
        Xtr = np.array([[1.], [np.nan], [3.]])
        a, _, _, _, info = tt.impute_from_train(Xtr, np.zeros((1, 1)), ["x"],
                                                ["cells"], log=lambda *_: None)
        assert a[1, 0] == 2.0 and info["n_imputed_train"] == 1

    def test_a_column_train_cannot_fill_is_dropped_from_both(self):
        Xtr = np.array([[1., np.nan], [2., np.nan]])
        Xte = np.array([[3., 4.]])
        a, b, names, blocks, info = tt.impute_from_train(
            Xtr, Xte, ["keep", "gone"], ["cells", "dynamics"],
            log=lambda *_: None)
        assert names == ["keep"] and blocks == ["cells"]
        assert a.shape == (2, 1) and b.shape == (1, 1)
        assert info["n_columns_dropped"] == 1

    def test_nothing_finite_survives_and_the_caller_can_see_it(self):
        a, b, names, _, _ = tt.impute_from_train(
            np.full((2, 2), np.nan), np.zeros((1, 2)), ["x", "y"],
            ["cells", "cells"], log=lambda *_: None)
        assert names == [] and a.shape == (2, 0)

    def test_the_transfer_survives_a_pooled_nan(self, tmp_path):
        """End to end: the real failure was RidgeCV raising `Input X contains
        NaN` from inside .predict after the baseline had fitted fine."""
        train, test = studies(tmp_path, TR, TE)
        space = dict(atlas=ATLAS, window_s=WIN, states=f"HMM1_pca3_{K}")
        # knock out a dynamics column for the whole test cohort the way an
        # unrecoverable stride does
        real = ht.Study.load_set

        def holed(self, sp, task):
            d = real(self, sp, task)
            if self.cfg is test.cfg:
                d = d.copy()
                d["mean_dwell_s"] = np.nan
            return d
        ht.Study.load_set = holed
        try:
            rows = tt.transfer_rows(train, test, models=["ridge"],
                                    arms=["dynamics"], n_boot=20,
                                    log=lambda *_: None)
        finally:
            ht.Study.load_set = real
        assert len(rows)
        assert rows["n_imputed_test"].sum() > 0


# ------------------------------------------------------------------ shards ---
class TestSharding:
    def test_every_set_lands_in_exactly_one_shard(self):
        s = sets_frame(7)
        seen = pd.concat([tt.shard_of(s, i, 3) for i in range(3)])
        assert len(seen) == 7
        assert set(map(tuple, seen[tt.GRID_KEYS].values)) == \
               set(map(tuple, s[tt.GRID_KEYS].values))

    def test_round_robin_spreads_the_expensive_sets(self):
        """K=27 is 729 transition cells against K=8's 64, so contiguous blocks
        would put every large K in one task."""
        s = pd.DataFrame({"atlas": ["a"] * 6, "window_s": [-1.0] * 6,
                          "states": [f"HMM1_pca3_{k}" for k in (8, 9, 10, 25, 26, 27)]})
        ks = [sorted(ht.n_states(x) for x in tt.shard_of(s, i, 2)["states"])
              for i in range(2)]
        assert all(any(k > 20 for k in shard) for shard in ks)

    def test_the_order_does_not_depend_on_how_the_grid_was_listed(self):
        s = sets_frame(7)
        a = tt.shard_of(s, 1, 3)[tt.GRID_KEYS].values.tolist()
        b = tt.shard_of(s.iloc[::-1], 1, 3)[tt.GRID_KEYS].values.tolist()
        assert a == b

    def test_a_shard_out_of_range_is_refused(self):
        with pytest.raises(SystemExit):
            tt.shard_of(sets_frame(), 3, 3)

    def test_the_grid_hash_ignores_row_order(self):
        s = sets_frame(5)
        assert tt.grid_hash(s) == tt.grid_hash(s.iloc[::-1])

    def test_the_grid_hash_changes_when_a_set_appears(self):
        assert tt.grid_hash(sets_frame(5)) != tt.grid_hash(sets_frame(6))


# ------------------------------------------------------------ write and read ---
def fake_rows(n=3, **over):
    d = pd.DataFrame({
        "atlas": ["yeo7"] * n, "window_s": [-1.0] * n,
        "states": [f"HMM1_pca3_{8 + i}" for i in range(n)],
        "K": [8 + i for i in range(n)], "task": ["movie"] * n,
        "phenotype": ["anxiety"] * n, "arm": ["cells"] * n,
        "model": ["ridge"] * n, "tree": ["bstm"] * n,
        "score": np.linspace(.1, .3, n), "lo": [0.] * n, "hi": [.4] * n,
        "base_score": [.05] * n, "base_lo": [-.1] * n, "base_hi": [.2] * n,
        "over_base": [True] * n, "n_train": [649] * n, "n_test": [168] * n,
        "n_covariates": [3] * n, "n_features": [67] * n,
        "n_imputed_train": [0] * n, "n_imputed_test": [0] * n,
        "n_columns_dropped": [0] * n})
    return d.assign(**over)


def put(root, shard, n_shards, gh="abc123", target="raw", tree="bstm", n=3):
    return tt.write_transferred(fake_rows(n), root, train="camcan",
                                test="hcp7t_fix", target_set=target, tree=tree,
                                shard=shard, n_shards=n_shards, grid_hash=gh,
                                n_sets_total=9, n_boot=2000, seed=0,
                                git_sha="deadbee", written_at="2026-01-01T00:00:00")


class TestWriteAndRead:
    def test_a_round_trip_keeps_the_rows_and_the_provenance(self, tmp_path):
        put(tmp_path, 0, 1)
        d = tt.read_transferred(tmp_path, "raw", trees=("bstm",),
                                log=lambda *_: None)
        assert len(d) == 3
        assert set(d["grid_hash"]) == {"abc123"} and d["n_boot"].iloc[0] == 2000
        assert d["target_set"].iloc[0] == "raw"

    def test_the_path_says_which_analysis_it_is(self, tmp_path):
        p = put(tmp_path, 7, 40, target="category")
        assert "targets=category" in str(p) and "shard=07" in str(p)
        assert "train=camcan" in str(p) and "test=hcp7t_fix" in str(p)

    def test_the_two_target_sets_never_mix(self, tmp_path):
        put(tmp_path, 0, 1, target="raw")
        put(tmp_path, 0, 1, target="category", n=5)
        assert len(tt.read_transferred(tmp_path, "raw", trees=("bstm",),
                                       log=lambda *_: None)) == 3
        assert len(tt.read_transferred(tmp_path, "category", trees=("bstm",),
                                       log=lambda *_: None)) == 5

    def test_a_half_finished_array_is_refused(self, tmp_path):
        for i in (0, 1, 3):
            put(tmp_path, i, 5)
        with pytest.raises(SystemExit) as e:
            tt.read_transferred(tmp_path, "raw", trees=("bstm",),
                                log=lambda *_: None)
        msg = str(e.value)
        assert "[2, 4]" in msg and "incomplete" in msg

    def test_a_run_in_progress_can_be_looked_at_on_purpose(self, tmp_path):
        for i in (0, 1):
            put(tmp_path, i, 5)
        seen = []
        d = tt.read_transferred(tmp_path, "raw", trees=("bstm",),
                                require_complete=False, log=seen.append)
        assert len(d) == 6
        assert any("MISSING" in m for m in seen)        # said, not hidden

    def test_shards_from_two_different_grids_are_refused(self, tmp_path):
        put(tmp_path, 0, 2, gh="aaa")
        put(tmp_path, 1, 2, gh="bbb")
        with pytest.raises(SystemExit) as e:
            tt.read_transferred(tmp_path, "raw", trees=("bstm",),
                                log=lambda *_: None)
        assert "different grids" in str(e.value)

    def test_a_missing_tree_names_the_directory(self, tmp_path):
        put(tmp_path, 0, 1, tree="bstm")
        with pytest.raises(SystemExit) as e:
            tt.read_transferred(tmp_path, "raw", trees=("bstm", "fcm"),
                                log=lambda *_: None)
        assert "tree=fcm" in str(e.value)

    def test_nothing_on_disk_at_all_says_to_run_the_script(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            tt.read_transferred(tmp_path, "raw", log=lambda *_: None)
        assert "transfer_grid.py" in str(e.value)

    def test_what_is_read_converts_for_the_figures(self, tmp_path):
        put(tmp_path, 0, 1)
        rows = tt.as_report_rows(tt.read_transferred(
            tmp_path, "raw", trees=("bstm",), log=lambda *_: None))
        assert np.allclose(rows["mean"], rows["score"])
        assert (rows["base_n"] == 168).all()


# --------------------------------------------------------------------- CLI ---
def camcan_tree(tmp_path, subs):
    """Cam-CAN's own phenotype pair, as transfer_config reads it by default."""
    d = tmp_path / "camcan_pheno"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "CCID": subs,
        "additional_HADS_anxiety": rng.integers(0, 22, len(subs)),
        # the sentinel target_range exists for: 9999 is "no answer" and HADS
        # runs 0-21
        "additional_HADS_depression": [9999] + list(rng.integers(0, 22, len(subs) - 1)),
        "additional_HADS_anx_category": ["Normal", "Mild", "Moderate", "Severe"]
                                        * (len(subs) // 4),
        "additional_HADS_dep_category": ["Normal", "Mild"] * (len(subs) // 2),
    }).to_csv(d / "approved_data.tsv", sep="\t", index=False)
    pd.DataFrame({"CCID": subs,
                  "Age": rng.integers(18, 89, len(subs)),
                  "Sex": ["MALE" if i % 2 else "FEMALE" for i in range(len(subs))]
                  }).to_csv(d / "standard_data.csv", index=False)
    return d


def hcp_file(tmp_path, subs):
    """HCP's own spelling: Subject, Gender, Age_in_Yrs, and an Age that is a band."""
    rng = np.random.default_rng(1)
    p = tmp_path / "hcp_pheno.csv"
    pd.DataFrame({"Subject": subs, "Age": ["26-30"] * len(subs),
                  "Age_in_Yrs": rng.integers(22, 36, len(subs)),
                  "Gender": ["M" if i % 2 else "F" for i in range(len(subs))],
                  "DSM_Anxi_Raw": rng.integers(0, 10, len(subs)),
                  "DSM_Depr_Raw": rng.integers(0, 10, len(subs)),
                  }).to_csv(p, index=False)
    return p


@pytest.fixture
def world(tmp_path):
    """One outputs tree both cohorts are in, plus both phenotype files."""
    root = tmp_path / "outputs"
    cc = [f"CC{i:04d}" for i in range(40)]
    hc = [f"{100000 + i}" for i in range(20)]
    for st in (f"HMM1_pca3_{K}", f"KMeans_pca3_{K}"):
        for w in (-1.0, 30.0):
            write_transitions(root, "camcan", "Movie", cc, w, st)
            write_transitions(root, "camcan_rest", "Rest", cc, w, st)
            for i in (1, 2, 3, 4):
                write_transitions(root, "hcp7t_fix", f"MOVIE{i}", hc, w, st)
                write_transitions(root, "hcp7t_fix", f"REST{i}", hc, w, st)
    for tree in ("bstm", "fcm"):
        write_summary(root, tree, "movie", "additional_HADS_anx_category")
    return dict(root=root, camcan=camcan_tree(tmp_path, cc),
                hcp=hcp_file(tmp_path, hc), out=tmp_path / "transferred_models")


def run(world, *args):
    import transfer_grid
    return transfer_grid.main([
        "--outputs", str(world["root"]), "--camcan-pheno", str(world["camcan"]),
        "--hcp-pheno", str(world["hcp"]), "--out", str(world["out"]), *args])


class TestTheScript:
    def test_a_dry_run_resolves_the_grid_and_fits_nothing(self, world, capsys):
        assert run(world, "--target", "raw", "--dry-run", "--n-shards", "4") == 0
        out = capsys.readouterr().out
        assert "4 set(s)" in out and "dry run" in out
        assert not world["out"].exists()

    def test_the_sentinel_is_dropped_for_the_raw_target(self, world, capsys):
        run(world, "--target", "raw", "--dry-run", "--n-shards", "4")
        assert "outside [0, 21]" in capsys.readouterr().out

    def test_one_shard_writes_one_parquet_the_reader_can_find(self, world):
        assert run(world, "--target", "raw", "--shard", "0", "--n-shards", "4",
                   "--arms", "cells", "--models", "ridge", "--n-boot", "20") == 0
        d = tt.read_transferred(world["out"], "raw", trees=("bstm",),
                                require_complete=False, log=lambda *_: None)
        assert len(d) and set(d["shard"]) == {0} and d["n_shards"].iloc[0] == 4
        assert d["n_test"].iloc[0] == 20          # the pooled HCP subjects

    def test_the_four_shards_together_are_the_whole_grid(self, world):
        for i in range(4):
            run(world, "--target", "raw", "--shard", str(i), "--n-shards", "4",
                "--arms", "cells", "--models", "ridge", "--n-boot", "20")
        d = tt.read_transferred(world["out"], "raw", trees=("bstm",),
                                log=lambda *_: None)
        # 4 state sets x 2 tasks x 2 targets x 1 arm x 1 model
        assert len(d) == 16
        assert set(d["shard"]) == {0, 1, 2, 3}
        assert len(set(d["grid_hash"])) == 1

    def test_a_finished_shard_is_skipped_not_refitted(self, world, capsys):
        a = ["--target", "raw", "--shard", "0", "--n-shards", "4",
             "--arms", "cells", "--models", "ridge", "--n-boot", "20"]
        run(world, *a)
        capsys.readouterr()
        run(world, *a)
        assert "exists, skipping" in capsys.readouterr().out

    def test_an_empty_shard_is_refused_by_name(self, world):
        with pytest.raises(SystemExit) as e:
            run(world, "--target", "raw", "--shard", "9", "--n-shards", "10",
                "--arms", "cells", "--models", "ridge", "--n-boot", "20")
        assert "EMPTY" in str(e.value) and "4 set(s)" in str(e.value)

    def test_a_filter_matching_nothing_points_at_stage_5a(self, world):
        with pytest.raises(SystemExit) as e:
            run(world, "--target", "raw", "--atlas", "nosuchatlas", "--dry-run")
        assert "STOP_AFTER=transitions" in str(e.value)

    def test_fcm_refuses_to_be_sharded(self, world):
        with pytest.raises(SystemExit) as e:
            run(world, "--target", "raw", "--tree", "fcm", "--n-shards", "4")
        assert "one job" in str(e.value)

    def test_both_target_sets_land_in_separate_partitions(self, world):
        for target in ("raw", "category"):
            run(world, "--target", target, "--shard", "0", "--n-shards", "1",
                "--arms", "cells", "--models", "ridge", "--n-boot", "20")
        for target in ("raw", "category"):
            d = tt.read_transferred(world["out"], target, trees=("bstm",),
                                    log=lambda *_: None)
            assert set(d["target_set"]) == {target}


# ----------------------------------------------- a model that ranked nobody ---
class TestConstantPrediction:
    """A model can predict ONE value for every test subject -- boosting on a
    weak signal and 168 people shrinks to the mean. Spearman is then undefined,
    and three things used to go wrong at once: scipy warned 2,000 times per
    row, the row carried a bare NaN that looked like a bug, and B1 died on an
    all-NaN group with `Encountered all NA values`."""

    def test_a_constant_prediction_scores_nan_without_warnings(self):
        import warnings
        y = np.arange(30.) + np.sin(np.arange(30.))
        with warnings.catch_warnings():
            warnings.simplefilter("error")           # any warning fails the test
            sc, lo, hi = tt.bootstrap_ci(y, np.full(30, 7.0), 50, 0)
        assert np.isnan(sc) and np.isnan(lo) and np.isnan(hi)

    def test_a_real_score_still_comes_out_bracketed(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=120)
        sc, lo, hi = tt.bootstrap_ci(y, y + rng.normal(0, .5, 120), 400, 0)
        assert lo < sc < hi and sc > .5

    def test_the_row_says_the_prediction_was_constant(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        rows = tt.transfer_rows(train, test, models=["ridge"], arms=["cells"],
                                n_boot=20, log=lambda *_: None)
        assert "pred_constant" in rows and rows["pred_constant"].dtype == bool

    def test_drop_degenerate_counts_what_it_removes(self):
        d = fake_rows(4)
        d.loc[[1, 3], "score"] = np.nan
        d.loc[[1, 3], "model"] = "hgb"
        seen = []
        out = tt.drop_degenerate(d, log=seen.append)
        assert len(out) == 2
        assert "2 of 4" in seen[0] and "hgb/cells x2" in seen[0]

    def test_a_nan_baseline_takes_the_row_too(self):
        """A score with no baseline to compare it to is not a comparison."""
        d = fake_rows(3)
        d.loc[[0], "base_score"] = np.nan
        assert len(tt.drop_degenerate(d, log=lambda *_: None)) == 2

    def test_a_clean_frame_is_untouched_and_silent(self):
        seen = []
        out = tt.drop_degenerate(fake_rows(3), log=seen.append)
        assert len(out) == 3 and seen == []

    def test_the_figures_survive_a_model_with_no_row_for_one_target(self):
        """The case this really happens in: hgb's Age + Sex baseline predicted a
        constant for depression, so every hgb depression row was dropped and
        the (depression, hgb) panel has nothing in it."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import hads_plots as hp

        d = pd.concat([
            fake_rows(3, phenotype="anxiety", model="ridge"),
            fake_rows(3, phenotype="anxiety", model="hgb"),
            fake_rows(3, phenotype="depression", model="ridge"),
        ], ignore_index=True)
        d["task"] = ["movie", "rest"] * 4 + ["movie"]
        rows = tt.as_report_rows(d)
        cfg = ht.Settings(targets={"anxiety": "a", "depression": "b"},
                          tasks={"movie": "Movie", "rest": "Rest"},
                          models=["ridge", "hgb"], cache_dir=None)
        st = object.__new__(ht.Study)
        st.cfg = cfg
        hp.fig_rankings(rows, cfg, st.above_both)
        hp.fig_factors(rows, cfg)
        hp.fig_k_curve(rows, cfg)
        plt.close("all")


# ------------------------------------------------------------------ threads ---
class TestThreads:
    """The crash that took two arrays, and the traceback that named it:

        scipy/linalg/_decomp.py          eigh
        sklearn/linear_model/_ridge.py   _eigen_decompose_gram
        transfer_tables.py               fit_predict
        Segmentation fault (core dumped)

    `_eigen_decompose_gram` is RidgeCV's n_features > n_samples branch, which
    is why the K=4 sets of a shard finished and its K=27 sets died. The cause
    is a BLAS thread count RAISED mid-process: OpenBLAS sizes its per-thread
    buffers on first use, and the container starts at
    OPENBLAS_NUM_THREADS=1 -- `apptainer exec --cleanenv` drops the job
    script's plain export, so a request for 4 threads was a raise from 1.
    """

    def setup_method(self):
        tt.usable_threads.cache_clear()
        tt._current_threads.cache_clear()

    teardown_method = setup_method

    def test_a_request_above_what_the_process_allows_is_clamped(self, monkeypatch):
        monkeypatch.setattr(tt, "_current_threads", lambda: 1)
        assert tt.usable_threads(4) == 1
        assert tt.usable_threads(99) == 1

    def test_a_request_below_it_is_honoured(self, monkeypatch):
        monkeypatch.setattr(tt, "_current_threads", lambda: 8)
        assert tt.usable_threads(4) == 4

    def test_never_zero_or_negative(self, monkeypatch):
        monkeypatch.setattr(tt, "_current_threads", lambda: 8)
        assert tt.usable_threads(0) == 1 and tt.usable_threads(-3) == 1

    def test_without_threadpoolctl_the_request_stands(self, monkeypatch):
        monkeypatch.setattr(tt, "_current_threads", lambda: None)
        assert tt.usable_threads(4) == 4

    def test_lowering_installs_a_limiter(self, monkeypatch):
        monkeypatch.setattr(tt, "_current_threads", lambda: 8)
        assert type(tt.thread_ctx(2)).__name__ == "threadpool_limits"

    def test_not_lowering_touches_nothing_at_all(self, monkeypatch):
        """The conservative half: with one thread requested in a process that
        already says one, nothing calls into OpenBLAS's thread machinery, so
        it cannot be the cause of a third crash."""
        import contextlib
        monkeypatch.setattr(tt, "_current_threads", lambda: 1)
        for want in (1, 4):
            assert isinstance(tt.thread_ctx(want), contextlib.nullcontext)

    def test_the_ridge_gram_branch_is_the_one_that_crashed(self):
        """Not a guess: with more features than subjects, RidgeCV takes the
        eigendecomposition path the traceback named. This is the shape the
        K=27 sets have (729 cells, ~600 subjects) and the K=4 sets do not."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 200))
        y = X[:, 0] + rng.normal(0, .5, 40)
        pred = tt.fit_predict(X, y, X[:5], "ridge", 0, threads=4)
        assert pred.shape == (5,) and np.isfinite(pred).all()

    def test_the_cli_asks_for_one_thread_unless_told_otherwise(self, world):
        """Its old default was SLURM_CPUS_PER_TASK, which is how a raise came
        to be requested at all."""
        import transfer_grid
        ap = transfer_grid.build_parser()
        assert ap.get_default("threads") == 1
        assert "CLAMPED" in ap.format_help()


class TestShardsThatDisagree:
    """The contamination the grid hash cannot see.

    A shard tried by hand -- `--shard 0 --n-shards 282 --arms cells --models
    ridge --n-boot 50`, to check the chain works on real data -- writes a real
    shard file into the real tree, and it reads as part of the array. The GRID
    was the same, so grid_hash matches; the ANALYSIS was not. It also makes the
    status count absurd, because `n_shards` is read from it.
    """

    def contaminate(self, root):
        for i in range(1, 4):                       # the array's own shards
            put(root, i, 4)
        # and the one tried by hand: a different n_shards, one arm, one model,
        # fewer draws
        tt.write_transferred(
            fake_rows(1, arm="cells", model="ridge"), root, train="camcan",
            test="hcp7t_fix", target_set="raw", tree="bstm", shard=0,
            n_shards=282, grid_hash="abc123", n_sets_total=9, n_boot=50,
            seed=0, git_sha="deadbee", written_at="2026-01-01T00:00:00")

    def test_the_reader_refuses_and_names_the_difference(self, tmp_path):
        self.contaminate(tmp_path)
        with pytest.raises(SystemExit) as e:
            tt.read_transferred(tmp_path, "raw", trees=("bstm",),
                                log=lambda *_: None)
        msg = str(e.value)
        assert "NOT computed the same way" in msg
        assert "n_boot=50" in msg and "n_boot=2000" in msg
        assert "n_shards=282" in msg

    def test_arms_and_models_alone_are_reported_not_refused(self, tmp_path):
        """A shard whose only state set lost an arm -- every column of it NaN
        across the training cohort -- is legitimately short of one. Refusing
        there would block a grid that is fine, so this half only warns."""
        put(tmp_path, 0, 2)
        d = fake_rows(3).assign(arm="occupancy")       # same n_boot/seed/n_shards
        tt.write_transferred(d, tmp_path, train="camcan", test="hcp7t_fix",
                             target_set="raw", tree="bstm", shard=1, n_shards=2,
                             grid_hash="abc123", n_sets_total=9, n_boot=2000,
                             seed=0, git_sha="x", written_at="t")
        seen = []
        out = tt.read_transferred(tmp_path, "raw", trees=("bstm",), log=seen.append)
        assert len(out) == 6
        assert any("do not all carry the same arms and models" in m for m in seen)

    def test_it_refuses_even_when_looking_at_a_run_in_progress(self, tmp_path):
        """Incomplete is a smaller analysis; contaminated is two analyses."""
        self.contaminate(tmp_path)
        with pytest.raises(SystemExit):
            tt.read_transferred(tmp_path, "raw", trees=("bstm",),
                                require_complete=False, log=lambda *_: None)

    def test_shards_computed_the_same_way_pass(self, tmp_path):
        for i in range(4):
            put(tmp_path, i, 4)
        d = tt.read_transferred(tmp_path, "raw", trees=("bstm",),
                                log=lambda *_: None)
        assert len(d) == 12

    def test_the_status_report_points_at_the_odd_shard(self, tmp_path, capsys):
        self.contaminate(tmp_path)
        import transfer_grid
        transfer_grid.main(["--status", "--target", "raw", "--out", str(tmp_path)])
        out = capsys.readouterr().out
        assert "DISAGREE" in out and "the odd one out is shard(s) [0]" in out
        assert "rm -r" in out and "shard=00" in out
        assert "why the count can look absurd" in out


class TestEmptyShardFiles:
    """`IndexError: single positional indexer is out-of-bounds`, from
    `.iloc[0]` on a shard file with no rows. The fcm job wrote one: every
    (atlas, task) was skipped, `fcm_transfer_rows` returned an empty frame and
    `write_transferred` saved it. A file with no rows reads as a finished
    shard."""

    def test_writing_an_empty_frame_is_refused(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            tt.write_transferred(fake_rows(0), tmp_path, train="camcan",
                                 test="hcp7t_fix", target_set="raw", tree="fcm",
                                 shard=0, n_shards=1, grid_hash="x")
        msg = str(e.value)
        assert "EMPTY" in msg and "skip lines" in msg

    def test_the_reader_names_an_empty_file_already_on_disk(self, tmp_path):
        """One is already there, written before the refusal existed."""
        import pyarrow as pa
        from fmri_decomposition.io import write_table_atomic
        d = fake_rows(0).assign(shard=0, n_shards=1, grid_hash="x", n_boot=2000,
                                seed=0)
        write_table_atomic(
            pa.Table.from_pandas(d, preserve_index=False),
            tt.transfer_dir(tmp_path, "camcan", "hcp7t_fix", "raw", "fcm")
            / "shard=00" / "rows.parquet")
        with pytest.raises(SystemExit) as e:
            tt.read_transferred(tmp_path, "raw", trees=("fcm",),
                                log=lambda *_: None)
        assert "hold NO rows" in str(e.value)

    def test_the_status_report_does_not_crash_on_one(self, tmp_path, capsys):
        import pyarrow as pa
        from fmri_decomposition.io import write_table_atomic
        d = fake_rows(0).assign(shard=0, n_shards=1, grid_hash="x", n_boot=2000,
                                seed=0)
        write_table_atomic(
            pa.Table.from_pandas(d, preserve_index=False),
            tt.transfer_dir(tmp_path, "camcan", "hcp7t_fix", "raw", "fcm")
            / "shard=00" / "rows.parquet")
        import transfer_grid
        assert transfer_grid.main(["--status", "--target", "raw",
                                   "--out", str(tmp_path)]) == 0
        assert "EMPTY FILE" in capsys.readouterr().out


class TestNoWrite:
    def test_it_scores_and_writes_nothing(self, world, capsys):
        assert run(world, "--target", "raw", "--shard", "0", "--n-shards", "4",
                   "--arms", "cells", "--models", "ridge", "--n-boot", "20",
                   "--no-write") == 0
        assert "NOT written" in capsys.readouterr().out
        assert not list(world["out"].rglob("rows.parquet"))


class TestStatus:
    """"Did they finish, or were they killed?" -- the question an array leaves
    behind, which squeue cannot answer once the tasks are gone."""

    def test_it_names_the_missing_shards(self, world, capsys):
        for i in (0, 2):
            put(world["out"], i, 4)
        import transfer_grid
        transfer_grid.main(["--status", "--target", "raw",
                            "--out", str(world["out"])])
        out = capsys.readouterr().out
        assert "2/4 shard(s)" in out and "MISSING [1, 3]" in out
        assert "transfer_grid.sh" in out          # and what to do about it

    def test_a_complete_tree_says_so(self, world, capsys):
        put(world["out"], 0, 1)
        put(world["out"], 0, 1, tree="fcm")
        import transfer_grid
        transfer_grid.main(["--status", "--target", "raw",
                            "--out", str(world["out"])])
        out = capsys.readouterr().out
        assert out.count("COMPLETE") == 2 and "MISSING" not in out

    def test_nothing_on_disk_is_reported_not_crashed(self, world, capsys):
        import transfer_grid
        assert transfer_grid.main(["--status", "--target", "raw",
                                   "--out", str(world["out"])]) == 0
        assert "NOTHING under" in capsys.readouterr().out

    def test_it_fits_nothing(self, world):
        import transfer_grid
        transfer_grid.main(["--status", "--target", "raw",
                            "--out", str(world["out"])])
        assert not list(world["out"].rglob("rows.parquet"))


class TestCoverageReport:
    """"40/40 shards" answers whether the TASKS finished. It does not answer
    whether every model scored in every atlas and every window -- a cell can be
    skipped, or scored and come out NaN because the model predicted one value
    for everyone. Both leave the shard count at 40/40."""

    def rows_over(self, atlases, windows, models, n_sets_total=4, nan_for=None):
        out = []
        for a in atlases:
            for w in windows:
                for m in models:
                    r = fake_rows(1, atlas=a, window_s=w, model=m,
                                  states=f"HMM1_pca3_8")
                    if m == nan_for:
                        r["score"] = np.nan
                    out.append(r)
        d = pd.concat(out, ignore_index=True)
        return d.assign(n_sets_total=n_sets_total)

    def write(self, root, d):
        return tt.write_transferred(
            d, root, train="camcan", test="hcp7t_fix", target_set="raw",
            tree="bstm", shard=0, n_shards=1, grid_hash="abc123",
            n_boot=2000, seed=0, git_sha="x", written_at="t")

    def report(self, root, capsys):
        import transfer_grid
        transfer_grid.main(["--status", "--target", "raw", "--out", str(root)])
        return capsys.readouterr().out

    def test_it_counts_the_state_sets_each_model_reached(self, tmp_path, capsys):
        self.write(tmp_path, self.rows_over(["yeo7", "networks"], [-1.0, 30.0],
                                            ["ridge", "hgb"]))
        out = self.report(tmp_path, capsys)
        assert "ridge: 4 set(s)" in out and "hgb: 4 set(s)" in out

    def test_a_model_short_of_the_others_is_flagged(self, tmp_path, capsys):
        d = pd.concat([
            self.rows_over(["yeo7", "networks"], [-1.0, 30.0], ["ridge"]),
            self.rows_over(["yeo7"], [-1.0], ["hgb"]),
        ], ignore_index=True)
        out = self.report(tmp_path, self.write(tmp_path, d) and capsys)
        assert "fewer sets than the other models" in out

    def test_rows_with_no_score_are_reported_per_model(self, tmp_path, capsys):
        self.write(tmp_path, self.rows_over(["yeo7"], [-1.0, 30.0],
                                            ["ridge", "hgb"], nan_for="hgb"))
        out = self.report(tmp_path, capsys)
        assert "with NO score (predicted a constant)" in out
        assert "ridge: 2 set(s), 2 row(s)\n" in out or "ridge: 2 set(s), 2 row(s)" in out

    def test_the_atlases_and_windows_are_broken_out(self, tmp_path, capsys):
        self.write(tmp_path, self.rows_over(["yeo7", "networks"],
                                            [-1.0, 30.0, 60.0], ["ridge"]))
        out = self.report(tmp_path, capsys)
        assert "atlas: networks=3, yeo7=3" in out
        assert "-1.0=2" in out and "60.0=2" in out

    def test_fewer_sets_than_the_grid_says_is_flagged(self, tmp_path, capsys):
        self.write(tmp_path, self.rows_over(["yeo7"], [-1.0], ["ridge"],
                                            n_sets_total=282))
        out = self.report(tmp_path, capsys)
        assert "1/282 state set(s)" in out and "SHORT" in out


class TestShortfallAccounting:
    """A shortfall has two causes and they need different actions.

    Her real report said `274/282 state set(s)  <-- SHORT: some cells were
    skipped` while one shard was missing. 282 sets into 40 shards puts
    ceil(282/40) = 8 in shard 0, and 274 + 8 = 282 -- so NOTHING was skipped.
    Telling her to grep for skips sent her after a second problem that was not
    there.
    """

    def put_sets(self, root, shard, n_shards, states, n_sets_total=282):
        d = pd.concat([fake_rows(1, states=st, atlas="yeo7", window_s=-1.0)
                       for st in states], ignore_index=True)
        return tt.write_transferred(
            d.assign(n_sets_total=n_sets_total), root, train="camcan",
            test="hcp7t_fix", target_set="raw", tree="bstm", shard=shard,
            n_shards=n_shards, grid_hash="g", n_boot=2000, seed=0, git_sha="x",
            written_at="t")

    def report(self, root, capsys):
        import transfer_grid
        transfer_grid.main(["--status", "--target", "raw", "--out", str(root)])
        return capsys.readouterr().out

    def test_a_shortfall_that_is_exactly_the_missing_shard_is_not_a_skip(
            self, tmp_path, capsys):
        # 10 sets, 2 shards -> shard 0 holds 5. Write only shard 1.
        self.put_sets(tmp_path, 1, 2, [f"HMM1_pca3_{i}" for i in range(5)],
                      n_sets_total=10)
        out = self.report(tmp_path, capsys)
        assert "5/10 state set(s)" in out
        assert "5 of them are the missing shard(s) [0]" in out
        assert "grep the logs" not in out

    def test_a_shortfall_beyond_the_missing_shard_is_a_skip(self, tmp_path, capsys):
        # shard 1 should hold 5 of 10 and only has 3
        self.put_sets(tmp_path, 1, 2, [f"HMM1_pca3_{i}" for i in range(3)],
                      n_sets_total=10)
        out = self.report(tmp_path, capsys)
        assert "SHORT by 2 beyond the missing shard(s)" in out
        assert "grep the logs for 'skipped'" in out

    def test_a_complete_grid_says_nothing_about_shortfalls(self, tmp_path, capsys):
        self.put_sets(tmp_path, 0, 1, [f"HMM1_pca3_{i}" for i in range(4)],
                      n_sets_total=4)
        out = self.report(tmp_path, capsys)
        assert "4/4 state set(s) with at least one row\n" in out
        assert "SHORT" not in out and "missing shard" not in out


class TestStaticFcPrecondition:
    """The fcm tree came out EMPTY, and the reason is structural: static FC is
    stage 3b, a separate job. `project_and_select.sh` does not run it, so a
    cohort projected onto an existing fit has transitions and no static FC --
    and every (atlas, task) cell then skips, leaving an empty frame that said
    nothing about why."""

    def static_fc(self, root, atlas, cohort, tasks, subs):
        import pyarrow as pa
        from fmri_decomposition.io import write_table_atomic
        for t in tasks:
            d = pd.DataFrame({
                "sub": subs, "task": t,
                "n_tr_used": 180, "n_tr_total": 200, "n_tr_good": 190,
                "frac_good_frames": .95,
                "r1__r2": np.linspace(.1, .4, len(subs)),
                "r1__r3": np.linspace(-.2, .2, len(subs)),
                "r2__r3": np.linspace(0, .3, len(subs))})
            write_table_atomic(
                pa.Table.from_pandas(d, preserve_index=False),
                root / "static_fc" / f"atlas={atlas}" / f"cohort={cohort}"
                / f"task={t}" / "subjects.parquet")

    def test_no_table_for_the_test_cohort_names_the_stage_and_the_command(
            self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        self.static_fc(train.cfg.root, "yeo7", "camcan", ["Movie"],
                       [f"CC{i:04d}" for i in range(60)])
        with pytest.raises(SystemExit) as e:
            tt.fcm_transfer_rows(train, test, log=lambda *_: None)
        msg = str(e.value)
        assert "hcp7t_fix" in msg and "NO static-FC table" in msg
        assert "STAGE 3b" in msg
        assert "sbatch slurm/static_fc.sbatch hcp7t_fix" in msg
        assert "--pool task" in msg and "pool movie WITH rest" in msg
        assert "camcan" in msg                      # and what IS there

    def test_an_empty_tree_says_to_run_stage_3b_for_both(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        with pytest.raises(SystemExit) as e:
            tt.fcm_transfer_rows(train, test, log=lambda *_: None)
        assert "no static-FC table" in str(e.value)
        assert "static_fc.sbatch" in str(e.value)

    def test_the_refusal_comes_before_any_fitting(self, tmp_path):
        """Not after an hour of cells that each skip."""
        train, test = studies(tmp_path, TR, TE)
        self.static_fc(train.cfg.root, "yeo7", "camcan", ["Movie"], ["CC0000"])
        with pytest.raises(SystemExit):
            tt.fcm_transfer_rows(train, test, models=["ridge"],
                                 log=lambda *_: None)

    def test_with_both_cohorts_present_it_proceeds(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        self.static_fc(train.cfg.root, "yeo7", "camcan", ["Movie"],
                       [f"CC{i:04d}" for i in range(60)])
        self.static_fc(train.cfg.root, "yeo7", "hcp7t_fix",
                       ["MOVIE1", "MOVIE2"], [f"H{i:04d}" for i in range(24)])
        rows = tt.fcm_transfer_rows(
            train, test, train_tasks={"movie": "Movie"},
            test_tasks={"movie": "MOVIE"}, models=["ridge"],
            covariates=["Age", "Sex", "frac_good_frames", "n_tr_used"],
            n_boot=20, log=lambda *_: None)
        assert len(rows) and set(rows["tree"]) == {"fcm"}
        assert rows["n_test"].iloc[0] == 24          # the four runs pooled


class TestTheNearMissCohort:
    """`find outputs/static_fc` came back with `hcp7t`, not `hcp7t_fix`.

    Those are deliberately different cohorts: 59 subjects at
    `_hp2000_clean_rclean_tclean` against 168 at `_hp2000_clean`. config/
    hcp7t_fix.yaml says why they are not one label -- two preprocessing depths
    under one name would put a preprocessing difference inside something the
    analysis treats as one thing. So the refusal has to say that the lookalike
    is not a substitute, or it reads as a typo to be worked around.
    """

    def static_fc(self, root, atlas, cohort, task="Movie", subs=("S1",)):
        import pyarrow as pa
        from fmri_decomposition.io import write_table_atomic
        d = pd.DataFrame({"sub": list(subs), "task": task, "n_tr_used": 180,
                          "frac_good_frames": .95, "r1__r2": .2, "r1__r3": .1,
                          "r2__r3": .3})
        write_table_atomic(
            pa.Table.from_pandas(d, preserve_index=False),
            root / "static_fc" / f"atlas={atlas}" / f"cohort={cohort}"
            / f"task={task}" / "subjects.parquet")

    def test_a_lookalike_cohort_is_called_out_as_not_a_substitute(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        self.static_fc(train.cfg.root, "yeo7", "camcan",
                       subs=[f"CC{i:04d}" for i in range(60)])
        self.static_fc(train.cfg.root, "yeo7", "hcp7t")      # NOT hcp7t_fix
        with pytest.raises(SystemExit) as e:
            tt.fcm_transfer_rows(train, test, log=lambda *_: None)
        msg = str(e.value)
        assert "NOT a substitute" in msg and "hcp7t" in msg
        assert "preprocessing difference" in msg

    def test_an_unrelated_cohort_is_not_called_a_lookalike(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        self.static_fc(train.cfg.root, "yeo7", "camcan",
                       subs=[f"CC{i:04d}" for i in range(60)])
        self.static_fc(train.cfg.root, "yeo7", "ds002837")
        with pytest.raises(SystemExit) as e:
            tt.fcm_transfer_rows(train, test, log=lambda *_: None)
        assert "NOT a substitute" not in str(e.value)

    def test_the_command_in_the_refusal_has_no_angle_brackets(self, tmp_path):
        """Pasted into bash, `<the one decompose used>` is a redirection and a
        syntax error -- which is exactly what it did."""
        train, test = studies(tmp_path, TR, TE)
        self.static_fc(train.cfg.root, "yeo7", "camcan",
                       subs=[f"CC{i:04d}" for i in range(60)])
        with pytest.raises(SystemExit) as e:
            tt.fcm_transfer_rows(train, test, log=lambda *_: None)
        line = next(l for l in str(e.value).splitlines() if "sbatch" in l)
        assert "<" not in line and ">" not in line

    def test_the_recorded_censor_policy_goes_into_the_command(self, tmp_path):
        import pyarrow as pa
        from fmri_decomposition.io import write_table_atomic
        train, test = studies(tmp_path, TR, TE)
        self.static_fc(train.cfg.root, "yeo7", "camcan",
                       subs=[f"CC{i:04d}" for i in range(60)])
        # stamp a policy onto one of the test cohort's own transition tables
        p = next(q for q in (train.cfg.root / "transitions").rglob("*.parquet")
                 if tt.hive(q, "cohort") == "hcp7t_fix")
        t = pa.parquet.read_table(p)
        t = t.replace_schema_metadata({**(t.schema.metadata or {}),
                                       b"censor_policy": b'"motion"'})
        write_table_atomic(t, p)
        assert tt.censor_policy_of(train.cfg.root, "hcp7t_fix") == "motion"
        with pytest.raises(SystemExit) as e:
            tt.fcm_transfer_rows(train, test, log=lambda *_: None)
        assert "--censor-policy motion" in str(e.value)


class TestFramesMustMatch:
    """slurm/static_fc.sbatch's header: "THIS MUST USE THE SAME FRAME SETTINGS
    AS THE STATE TREES ... or the two trees differ in preprocessing as well as
    in model". Nothing enforced it, and stage 3b for the test cohort was run by
    hand, days after the state trees, with the policy typed in again.

    E1 is two MODELS on one sample. If one tree's frames were censored and the
    other's were not, the panel compares preprocessing and calls it a model --
    and nothing in the figure could say so.
    """

    def static_fc(self, root, atlas, cohort, tasks, subs, policy="motion",
                  band="[0.01, 0.1]", zscore=True, pool="task"):
        import pyarrow as pa
        from fmri_decomposition.io import write_table_atomic
        for t in tasks:
            d = pd.DataFrame({
                "sub": list(subs), "task": t, "n_tr_used": 180,
                "frac_good_frames": .95,
                "r1__r2": np.linspace(.1, .4, len(subs)),
                "r1__r3": np.linspace(-.2, .2, len(subs)),
                "r2__r3": np.linspace(0, .3, len(subs)),
                "censor_policy": policy, "match_bandpass": band,
                "zscore_runs": zscore, "pool": pool})
            write_table_atomic(
                pa.Table.from_pandas(d, preserve_index=False),
                root / "static_fc" / f"atlas={atlas}" / f"cohort={cohort}"
                / f"task={t}" / "subjects.parquet")

    def stamp_state_policy(self, root, cohort, policy):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from fmri_decomposition.io import write_table_atomic
        for q in (root / "transitions").rglob("*.parquet"):
            if tt.hive(q, "cohort") != cohort:
                continue
            t = pq.read_table(q)
            write_table_atomic(t.replace_schema_metadata(
                {**(t.schema.metadata or {}),
                 b"censor_policy": f'"{policy}"'.encode()}), q)

    def world(self, tmp_path, **over):
        train, test = studies(tmp_path, TR, TE)
        self.static_fc(train.cfg.root, "yeo7", "camcan", ["Movie"],
                       [f"CC{i:04d}" for i in range(60)])
        self.static_fc(train.cfg.root, "yeo7", "hcp7t_fix",
                       ["MOVIE1", "MOVIE2"], [f"H{i:04d}" for i in range(24)],
                       **over)
        return train, test

    def run(self, train, test):
        return tt.fcm_transfer_rows(
            train, test, train_tasks={"movie": "Movie"},
            test_tasks={"movie": "MOVIE"}, models=["ridge"],
            covariates=["Age", "Sex", "frac_good_frames", "n_tr_used"],
            n_boot=20, log=lambda *_: None)

    def test_matching_frames_pass(self, tmp_path):
        train, test = self.world(tmp_path)
        assert len(self.run(train, test))

    def test_a_different_censor_policy_between_cohorts_is_refused(self, tmp_path):
        train, test = self.world(tmp_path, policy="none")
        with pytest.raises(SystemExit) as e:
            self.run(train, test)
        msg = str(e.value)
        assert "DIFFERENT frames" in msg
        assert "censor_policy: train 'motion'  test 'none'" in msg

    def test_a_different_bandpass_is_refused_too(self, tmp_path):
        train, test = self.world(tmp_path, band="[0.009, 0.08]")
        with pytest.raises(SystemExit) as e:
            self.run(train, test)
        assert "match_bandpass" in str(e.value)

    def test_static_fc_disagreeing_with_the_state_trees_is_refused(self, tmp_path):
        """Both sides agree with each other and neither agrees with the
        transitions -- so bstm and fcm are on different frames."""
        train, test = self.world(tmp_path)
        self.stamp_state_policy(train.cfg.root, "hcp7t_fix", "strict")
        with pytest.raises(SystemExit) as e:
            self.run(train, test)
        msg = str(e.value)
        assert "differ in PREPROCESSING as well as in model" in msg
        assert "--censor-policy strict" in msg

    def test_agreeing_with_the_state_trees_passes(self, tmp_path):
        train, test = self.world(tmp_path)
        self.stamp_state_policy(train.cfg.root, "hcp7t_fix", "motion")
        self.stamp_state_policy(train.cfg.root, "camcan", "motion")
        assert len(self.run(train, test))

    def test_a_side_whose_rows_disagree_is_refused(self):
        """Tested at the function, not through the pipeline: for camcan the two
        cohorts hold different TASKS, so a cell never loads both. The guard is
        cheap and covers a cohort pair that does overlap -- but a test that
        fabricated an unreachable path would be testing the fixture."""
        d = pd.DataFrame({"censor_policy": ["motion", "none"],
                          "match_bandpass": "[0.01, 0.1]",
                          "zscore_runs": True, "pool": "task"})
        with pytest.raises(SystemExit) as e:
            tt.frame_recipe(d, "train ['camcan', 'camcan_rest'] atlas=yeo7")
        msg = str(e.value)
        assert "disagree on censor_policy" in msg
        assert "different runs of stage" in msg and "camcan_rest" in msg

    def test_a_consistent_table_reads_back_its_recipe(self):
        d = pd.DataFrame({"censor_policy": ["motion"] * 3,
                          "match_bandpass": "[0.01, 0.1]",
                          "zscore_runs": True, "pool": "task"})
        assert tt.frame_recipe(d, "x") == {
            "censor_policy": "motion", "match_bandpass": "[0.01, 0.1]",
            "zscore_runs": True, "pool": "task"}

    def test_a_table_predating_the_provenance_columns_is_not_refused(self):
        """An older static_fc table has none of these columns. Absent is not
        disagreeing, and refusing there would block a tree that may be fine."""
        assert tt.frame_recipe(pd.DataFrame({"sub": ["a"]}), "x") == {}


class TestThePreflightChecksWhatIsSubmitted:
    """Asked for `TREES=fcm`, the submitter dry-ran **bstm** -- and then, since
    bstm's shard 0 was already written, took the "exists, skipping" path and
    returned. So the pre-flight reported on a grid nobody was about to run,
    and said nothing about the one that was. Twice silent."""

    def test_a_dry_run_is_not_short_circuited_by_an_existing_shard(
            self, world, capsys):
        run(world, "--target", "raw", "--shard", "0", "--n-shards", "4",
            "--arms", "cells", "--models", "ridge", "--n-boot", "20")
        capsys.readouterr()
        run(world, "--target", "raw", "--shard", "0", "--n-shards", "4",
            "--dry-run")
        out = capsys.readouterr().out
        assert "dry run:" in out
        assert "already there; the real run would skip it" in out

    def test_no_write_is_not_short_circuited_either(self, world, capsys):
        run(world, "--target", "raw", "--shard", "0", "--n-shards", "4",
            "--arms", "cells", "--models", "ridge", "--n-boot", "20")
        capsys.readouterr()
        run(world, "--target", "raw", "--shard", "0", "--n-shards", "4",
            "--arms", "cells", "--models", "ridge", "--n-boot", "20",
            "--no-write")
        assert "NOT written" in capsys.readouterr().out

    def test_the_fcm_dry_run_counts_atlases_not_state_sets(self, world, capsys):
        """`--tree fcm --dry-run` used to print the bstm arithmetic, which is
        wrong by a factor of the whole state-set grid."""
        run(world, "--target", "raw", "--tree", "fcm", "--n-shards", "1",
            "--dry-run")
        out = capsys.readouterr().out
        assert "static FC on 0 atlas(es)" in out
        assert "Stage 3b" in out and "static_fc.sbatch" in out

    def test_it_names_what_the_static_fc_tree_does_hold(self, world, capsys):
        import pyarrow as pa
        from fmri_decomposition.io import write_table_atomic
        d = pd.DataFrame({"sub": ["CC0000"], "task": "Movie", "r1__r2": .1})
        write_table_atomic(
            pa.Table.from_pandas(d, preserve_index=False),
            world["root"] / "static_fc" / "atlas=yeo7" / "cohort=camcan"
            / "task=Movie" / "subjects.parquet")
        run(world, "--target", "raw", "--tree", "fcm", "--n-shards", "1",
            "--dry-run")
        out = capsys.readouterr().out
        assert "atlas=yeo7: ['camcan']" in out
        assert "held by BOTH" in out

    def test_the_submitter_preflights_each_tree_it_submits(self):
        t = (Path(__file__).resolve().parent.parent / "slurm"
             / "transfer_grid.sh").read_text()
        i = t.index("pre-flight")
        block = t[i:i + 500]
        assert "for tree in $TREES" in block
        assert '--tree "$tree"' in block


class TestStatusReportsTheFcmInputs:
    """Both fcm jobs left the queue with nothing on disk, and `--status` could
    only say "NOTHING under ...". Whether the fcm half CAN run is knowable from
    the static_fc tree before anything is submitted: both cohorts need a table
    for one atlas, and the two tables need the same frame recipe."""

    def static_fc(self, root, atlas, cohort, task="Movie", policy="motion",
                  prov=True):
        import pyarrow as pa
        from fmri_decomposition.io import write_table_atomic
        d = pd.DataFrame({"sub": ["S1"], "task": task, "r1__r2": .1})
        if prov:
            d = d.assign(censor_policy=policy, match_bandpass="[0.01, 0.1]",
                         zscore_runs=True, pool="task")
        write_table_atomic(
            pa.Table.from_pandas(d, preserve_index=False),
            root / "static_fc" / f"atlas={atlas}" / f"cohort={cohort}"
            / f"task={task}" / "subjects.parquet")

    def report(self, world, capsys):
        import transfer_grid
        put(world["out"], 0, 1)
        transfer_grid.main(["--status", "--target", "raw",
                            "--out", str(world["out"]),
                            "--outputs", str(world["root"])])
        return capsys.readouterr().out

    def whole_train_side(self, root, atlas="yeo7", **kw):
        """transfer_config.TRAIN_COHORTS is camcan AND camcan_rest, and
        `_fc_cell` needs a table for both."""
        self.static_fc(root, atlas, "camcan", task="Movie", **kw)
        self.static_fc(root, atlas, "camcan_rest", task="Rest", **kw)

    def test_it_prints_each_tables_frame_recipe(self, world, capsys):
        self.whole_train_side(world["root"])
        self.static_fc(world["root"], "yeo7", "hcp7t_fix", task="MOVIE1")
        out = self.report(world, capsys)
        assert "static FC inputs" in out
        assert "cohort=camcan" in out and "cohort=hcp7t_fix" in out
        assert "censor_policy='motion'" in out

    def test_a_disagreement_is_named_before_anything_is_submitted(
            self, world, capsys):
        self.whole_train_side(world["root"], policy="motion")
        self.static_fc(world["root"], "yeo7", "hcp7t_fix", task="MOVIE1",
                       policy="none")
        out = self.report(world, capsys)
        assert "the two sides DISAGREE" in out
        assert "censor_policy: train 'motion' test 'none'" in out
        assert "the two sides disagree" in out        # and in the NOT finished line

    def test_an_atlas_only_one_side_has_is_called_out(self, world, capsys):
        self.whole_train_side(world["root"])
        out = self.report(world, capsys)
        assert "the test side is MISSING ['hcp7t_fix']" in out

    def test_a_side_missing_ONE_of_its_cohorts_is_called_out(self, world, capsys):
        """The real failure: camcan had a table, camcan_rest did not, and
        `_fc_cell` concatenates before filtering by task -- so all six
        (atlas, task) cells skipped, the movie ones included."""
        self.static_fc(world["root"], "yeo7", "camcan", task="Movie")
        self.static_fc(world["root"], "yeo7", "hcp7t_fix", task="MOVIE1")
        out = self.report(world, capsys)
        assert "the train side is MISSING ['camcan_rest']" in out
        assert "movie as well as rest" in out

    def test_an_empty_static_fc_tree_says_stage_3b_has_not_run(
            self, world, capsys):
        out = self.report(world, capsys)
        assert "stage 3b has not run" in out

    def test_a_table_predating_the_provenance_columns_is_labelled(
            self, world, capsys):
        self.static_fc(world["root"], "yeo7", "camcan", prov=False)
        self.static_fc(world["root"], "yeo7", "hcp7t_fix", task="MOVIE1")
        out = self.report(world, capsys)
        assert "no provenance columns (predates them)" in out
        assert "DISAGREE" not in out            # absent is not disagreeing


class TestAPartialSide:
    """One absent cohort skipped SIX cells, the movie ones included.

    `_fc_cell` reads one table per cohort of a side and concatenates them
    BEFORE filtering by task, so a missing `camcan_rest` takes the movie cells
    down with the rest cells -- even though camcan's own table was right there
    and the movie cells need nothing else. The job then scored nothing and the
    empty-frame refusal fired, which is the right failure for the wrong-looking
    reason: it reads as "static FC has not run" when static FC had run for two
    of the three cohorts.
    """

    def static_fc(self, root, atlas, cohort, task, policy="motion"):
        import pyarrow as pa
        from fmri_decomposition.io import write_table_atomic
        subs = ([f"CC{i:04d}" for i in range(60)] if cohort.startswith("camcan")
                else [f"H{i:04d}" for i in range(24)])
        d = pd.DataFrame({"sub": subs, "task": task, "n_tr_used": 180,
                          "frac_good_frames": .95,
                          "r1__r2": np.linspace(.1, .4, len(subs)),
                          "r1__r3": np.linspace(-.2, .2, len(subs)),
                          "r2__r3": np.linspace(0, .3, len(subs)),
                          "censor_policy": policy,
                          "match_bandpass": "[0.01, 0.1]",
                          "zscore_runs": True, "pool": "task"})
        write_table_atomic(
            pa.Table.from_pandas(d, preserve_index=False),
            root / "static_fc" / f"atlas={atlas}" / f"cohort={cohort}"
            / f"task={task}" / "subjects.parquet")

    def two_sided(self, tmp_path, rest_policy="motion", with_rest=True):
        train, test = studies(tmp_path, TR, TE)
        train.cfg.cohorts = ["camcan", "camcan_rest"]
        r = train.cfg.root
        self.static_fc(r, "yeo7", "camcan", "Movie")
        if with_rest:
            self.static_fc(r, "yeo7", "camcan_rest", "Rest", policy=rest_policy)
        for t in ("MOVIE1", "MOVIE2"):
            self.static_fc(r, "yeo7", "hcp7t_fix", t)
        # the test side's rest matches the train side's rest, so a disagreement
        # between CONDITIONS is not also a disagreement between SIDES -- which
        # would fire first and test the other check
        self.static_fc(r, "yeo7", "hcp7t_fix", "REST1", policy=rest_policy)
        return train, test

    def run(self, train, test, **kw):
        kw.setdefault("log", lambda *_: None)
        return tt.fcm_transfer_rows(
            train, test, train_tasks={"movie": "Movie", "rest": "Rest"},
            test_tasks={"movie": "MOVIE", "rest": "REST"}, models=["ridge"],
            covariates=["Age", "Sex", "frac_good_frames", "n_tr_used"],
            n_boot=20, **kw)

    def test_a_side_missing_one_cohort_is_refused_up_front(self, tmp_path):
        train, test = self.two_sided(tmp_path, with_rest=False)
        with pytest.raises(SystemExit) as e:
            self.run(train, test)
        msg = str(e.value)
        assert "INCOMPLETE" in msg and "camcan_rest" in msg
        assert "movie cells too" in msg
        assert "sbatch slurm/static_fc.sbatch camcan_rest" in msg

    def test_the_refusal_comes_before_the_cells_are_tried(self, tmp_path):
        """Not six skip lines and an empty frame."""
        train, test = self.two_sided(tmp_path, with_rest=False)
        seen = []
        with pytest.raises(SystemExit):
            self.run(train, test, log=seen.append)
        assert not [m for m in seen if "skip fcm" in m]

    def test_a_complete_side_runs(self, tmp_path):
        train, test = self.two_sided(tmp_path)
        rows = self.run(train, test)
        assert len(rows) and set(rows["task"]) == {"movie", "rest"}

    def test_the_two_conditions_of_one_side_must_share_a_recipe(self, tmp_path):
        """camcan's movie table and camcan_rest's rest table are two separate
        runs of stage 3b. If they disagree, rest-vs-movie is partly a
        preprocessing difference wearing a condition's name."""
        train, test = self.two_sided(tmp_path, rest_policy="none")
        with pytest.raises(SystemExit) as e:
            self.run(train, test)
        msg = str(e.value)
        assert "difference between the two CONDITIONS" in msg
        assert "censor_policy" in msg
        assert "different runs of stage 3b" in msg
