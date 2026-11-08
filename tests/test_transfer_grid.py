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
