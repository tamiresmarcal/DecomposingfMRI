"""Section 2 of 13_camcan_to_hcp.ipynb, end to end, on a synthetic tree.

WHY THIS FILE EXISTS. Every unit in the transfer chain had tests and the
notebook still broke four times in a row, each time at the JOIN: the thing
that had never been exercised was the whole path, from what the cluster writes
to what the figures read.

    cluster writes bstm shards + the fcm job
      -> read_transferred(trees=("bstm", "fcm"))
      -> as_report_rows
      -> drop_degenerate
      -> B1, B2, B2.K
      -> the section-3 pick and the E1 selection

The last break is the one this is named for: `as_report_rows` read K out of
the `states` label for every row, which was safe only while the fcm rows were
appended AFTER the conversion. Reading both trees from disk in one call sent
"(static FC)" into `n_states` and raised `IndexError: list index out of range`
from inside a `.map`, five frames from the cause. A test of either half would
have passed; only the join fails.
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
from test_pool_tasks import K  # noqa: E402
from test_transfer_grid import camcan_tree, hcp_file  # noqa: E402
from test_transfer_wiring import write_summary, write_transitions  # noqa: E402

ATLASES = ("yeo7", "networks")


def static_fc(root, atlas, cohort, task, subs, policy="motion"):
    import pyarrow as pa
    from fmri_decomposition.io import write_table_atomic
    rng = np.random.default_rng(abs(hash((atlas, cohort, task))) % 2**31)
    d = pd.DataFrame({
        "sub": list(subs), "task": task,
        "n_tr_total": 200, "n_tr_good": 190, "n_tr_used": 180,
        "frac_good_frames": .95, "n_tasks": 1,
        "r1__r2": rng.normal(size=len(subs)) * .1 + .2,
        "r1__r3": rng.normal(size=len(subs)) * .1,
        "r2__r3": rng.normal(size=len(subs)) * .1 + .3,
        "censor_policy": policy, "match_bandpass": "[0.01, 0.1]",
        "zscore_runs": True, "pool": "task"})
    write_table_atomic(
        pa.Table.from_pandas(d, preserve_index=False),
        root / "static_fc" / f"atlas={atlas}" / f"cohort={cohort}"
        / f"task={task}" / "subjects.parquet")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """One outputs tree shaped like the real one: Cam-CAN as ONE cohort with
    two tasks, HCP with four runs per condition, and static FC for both."""
    tmp = tmp_path_factory.mktemp("nb13")
    root = tmp / "outputs"
    cc = [f"CC{i:04d}" for i in range(60)]
    hc = [f"{100000 + i}" for i in range(30)]
    for atlas in ATLASES:
        for st in (f"HMM1_pca3_{K}", f"KMeans_pca3_{K}"):
            for w in (-1.0, 30.0):
                for task in ("Movie", "Rest"):
                    write_transitions(root, "camcan", task, cc, w, st,
                                      atlas=atlas)
                for i in (1, 2, 3, 4):
                    for task in (f"MOVIE{i}", f"REST{i}"):
                        write_transitions(root, "hcp7t_fix", task, hc, w, st,
                                          atlas=atlas)
        for task in ("Movie", "Rest"):
            static_fc(root, atlas, "camcan", task, cc)
        for i in (1, 2, 3, 4):
            for task in (f"MOVIE{i}", f"REST{i}"):
                static_fc(root, atlas, "hcp7t_fix", task, hc)
    for t in ("bstm", "fcm"):
        write_summary(root, t, "movie", "additional_HADS_anx_category")
    return dict(root=root, out=tmp / "transferred_models",
                camcan=camcan_tree(tmp, cc), hcp=hcp_file(tmp, hc))


@pytest.fixture(scope="module")
def rows(tree):
    """What the cluster writes, read back the way the notebook reads it."""
    import transfer_grid

    base = ["--outputs", str(tree["root"]), "--camcan-pheno", str(tree["camcan"]),
            "--hcp-pheno", str(tree["hcp"]), "--out", str(tree["out"]),
            "--target", "raw", "--n-boot", "40"]
    for shard in range(2):
        transfer_grid.main(base + ["--shard", str(shard), "--n-shards", "2",
                                   "--models", "ridge"])
    transfer_grid.main(base + ["--tree", "fcm", "--n-shards", "1",
                               "--models", "ridge"])
    raw = tt.read_transferred(tree["out"], "raw", trees=("bstm", "fcm"),
                              log=lambda *_: None)
    return raw, tt.drop_degenerate(tt.as_report_rows(raw), log=lambda *_: None)


class TestTheWholeChain:
    def test_both_trees_are_there(self, rows):
        raw, _ = rows
        assert set(raw["tree"]) == {"bstm", "fcm"}

    def test_the_conversion_survives_a_static_fc_label(self, rows):
        """The break this file is named for: "(static FC)" has no K."""
        _, d = rows
        fcm, bstm = d[d["tree"] == "fcm"], d[d["tree"] == "bstm"]
        assert len(fcm) and len(bstm)
        assert d.loc[d["tree"] == "fcm", "K"].isna().all()
        assert (bstm["K"] == K).all()

    def test_the_fcm_rows_get_no_method_or_embedding(self, rows):
        _, d = rows
        fcm = d[d["tree"] == "fcm"]
        for col in ("method", "embedding", "K_"):
            assert fcm[col].isna().all(), col

    def test_the_bstm_rows_keep_theirs(self, rows):
        _, d = rows
        bstm = d[d["tree"] == "bstm"]
        assert set(bstm["method"]) == {"HMM1", "KMeans"}
        assert set(bstm["embedding"]) == {"pca3"}

    def test_the_label_names_the_atlas_for_fcm_and_the_set_for_bstm(self, rows):
        _, d = rows
        assert set(d.loc[d["tree"] == "fcm", "label"]) <= set(ATLASES)
        assert d.loc[d["tree"] == "bstm", "label"].str.contains("·").all()

    def test_the_error_bar_converts_both_ways(self, rows):
        _, d = rows
        assert np.allclose(d["mean"], d["score"], equal_nan=True)
        assert (d["std"] >= 0).all()
        assert (d["base_n"] == d["n_test"]).all()

    def test_the_test_cohort_is_the_pooled_subject_count(self, rows):
        _, d = rows
        assert set(d["n_test"]) == {30}

    def test_the_frames_agree_so_nothing_was_conceded(self, rows):
        raw, _ = rows
        fcm = raw[raw["tree"] == "fcm"]
        assert (fcm["frames_mismatch"] == "").all()
        assert (fcm["train_censor_policy"] == "motion").all()


class TestTheFigures:
    """B1, B2 and B2.K over a frame holding BOTH trees. Each has died once on
    a shape the other tree introduced."""

    @pytest.fixture(autouse=True)
    def _agg(self):
        import matplotlib
        matplotlib.use("Agg")

    def cfg(self, tree):
        import transfer_config as tcfg
        _, _, train_cfg, _ = tcfg.studies(
            "raw", root=tree["root"], camcan=tree["camcan"],
            hcp_pheno=tree["hcp"], models=["ridge"])
        return train_cfg

    def test_b1_draws(self, rows, tree):
        import matplotlib.pyplot as plt
        import hads_plots as hp
        _, d = rows
        st = object.__new__(ht.Study)
        st.cfg = self.cfg(tree)
        hp.fig_rankings(d, st.cfg, st.above_both)
        plt.close("all")

    def test_b2_draws(self, rows, tree):
        import matplotlib.pyplot as plt
        import hads_plots as hp
        _, d = rows
        hp.fig_factors(d, self.cfg(tree))
        plt.close("all")

    def test_b2k_draws_and_ignores_the_fcm_rows(self, rows, tree):
        import matplotlib.pyplot as plt
        import hads_plots as hp
        _, d = rows
        hp.fig_k_curve(d, self.cfg(tree))
        plt.close("all")


class TestSectionThree:
    def test_the_pick_and_the_e1_selection_find_rows(self, rows):
        """Cell 17 and cell 19 of the notebook, which index into the frame by
        the chosen row's own values."""
        _, d = rows
        clears = d[d["lo"] > d["base_score"]].sort_values("score", ascending=False)
        best = (clears if len(clears) else
                d.sort_values("score", ascending=False)).iloc[0]
        sel = ((d["states"] == best["states"]) & (d["atlas"] == best["atlas"])
               & (d["arm"] == best["arm"]) & (d["model"] == best["model"]))
        fcm_sel = (d["tree"].eq("fcm") & d["atlas"].eq(best["atlas"])
                   & d["model"].eq(best["model"]))
        perf = d[sel | fcm_sel]
        assert len(perf)
        assert {"bstm", "fcm"} <= set(perf["tree"]) or best["tree"] == "fcm"
        assert isinstance(best["label"], str)


class TestE1:
    """The x axis showed `fcm movie` twice with different values, because the
    hand-rolled selection took every fcm row of the atlas and model -- which is
    two ARMS, `edges` and its control `global` -- and labelled both by tree and
    task alone."""

    @pytest.fixture(autouse=True)
    def _agg(self):
        import matplotlib
        matplotlib.use("Agg")

    def best_of(self, d):
        clears = d[d["lo"] > d["base_score"]].sort_values("score", ascending=False)
        return (clears if len(clears) else
                d.sort_values("score", ascending=False)).iloc[0]

    def test_one_row_per_cell(self, rows):
        _, d = rows
        best = self.best_of(d[d["tree"] == "bstm"])
        perf = tt.transfer_perf(d, best, fc_arm="edges")
        assert not perf.duplicated(["phenotype", "task", "tree"]).any()
        assert set(perf["tree"]) == {"bstm", "fcm"}
        assert set(perf.loc[perf["tree"] == "fcm", "arm"]) == {"edges"}

    def test_both_arms_in_one_frame_is_refused_not_drawn(self, rows):
        """What the old selection produced: two points sharing a label."""
        _, d = rows
        best = self.best_of(d[d["tree"] == "bstm"])
        both = pd.concat([
            d[(d["tree"] == "bstm") & (d["states"] == best["states"])
              & (d["atlas"] == best["atlas"])
              & (d["window_s"] == best["window_s"])
              & (d["arm"] == best["arm"])
              & (d["model"] == best["model"])],
            d[d["tree"].eq("fcm") & d["atlas"].eq(best["atlas"])
              & d["model"].eq(best["model"])]], ignore_index=True)
        assert both.duplicated(["phenotype", "task", "tree"]).any()

    def test_an_absent_arm_says_which_arms_exist(self, rows):
        _, d = rows
        best = self.best_of(d[d["tree"] == "bstm"])
        with pytest.raises(SystemExit) as e:
            tt.transfer_perf(d, best, fc_arm="nosucharm")
        assert "edges" in str(e.value) and "global" in str(e.value)

    def test_the_shares_are_per_block_and_sum_to_one(self, rows, tree):
        import transfer_config as tcfg
        _, d = rows
        train, test, _, _ = tcfg.studies("raw", root=tree["root"],
                                         camcan=tree["camcan"],
                                         hcp_pheno=tree["hcp"], models=["ridge"])
        best = self.best_of(d[d["tree"] == "bstm"])
        space = dict(atlas=best["atlas"], window_s=best["window_s"],
                     states=best["states"])
        sh = tt.transfer_shares(train, test, space, best["arm"], best["model"],
                                log=lambda *_: None)
        assert set(sh.columns) == {"task", "phenotype", "block", "share"}
        tot = sh.groupby(["task", "phenotype"])["share"].sum()
        assert np.allclose(tot, 1.0)
        assert "covariates" in set(sh["block"])

    def test_the_whole_e1_draws(self, rows, tree):
        import matplotlib.pyplot as plt
        import hads_plots as hp
        import transfer_config as tcfg
        _, d = rows
        train, test, train_cfg, _ = tcfg.studies(
            "raw", root=tree["root"], camcan=tree["camcan"],
            hcp_pheno=tree["hcp"], models=["ridge"])
        best = self.best_of(d[d["tree"] == "bstm"])
        space = dict(atlas=best["atlas"], window_s=best["window_s"],
                     states=best["states"])
        perf = tt.transfer_perf(d, best, fc_arm=train_cfg.fc_arm)
        sh = tt.transfer_shares(train, test, space, best["arm"], best["model"],
                                log=lambda *_: None)
        hp.fig_performance(perf, train_cfg, title="E1", model=best["model"],
                           legend={"bstm": "x", "fcm": "y"}, shares=sh,
                           share_label="on the TEST subjects")
        fig = plt.gcf()
        # `layout` title-cases every label, so compare case-insensitively
        assert any("test subjects" in a.get_xlabel().lower() for a in fig.axes)
        assert any(train_cfg.score_name.lower() in a.get_ylabel().lower()
                   for a in fig.axes if a.get_ylabel())
        plt.close("all")

    def test_the_same_states_label_at_two_apertures_is_not_one_row(self, rows):
        """A `states` label is not unique across apertures: HMM1_pca3_3 exists
        at 1 TR and at 30s. Selecting without window_s took both and the figure
        drew them on top of each other, saying nothing."""
        _, d = rows
        best = self.best_of(d[d["tree"] == "bstm"])
        same_label = d[(d["tree"] == "bstm") & (d["states"] == best["states"])
                       & (d["atlas"] == best["atlas"])
                       & (d["arm"] == best["arm"])
                       & (d["model"] == best["model"])]
        assert same_label["window_s"].nunique() > 1, "fixture has one aperture"
        perf = tt.transfer_perf(d, best, fc_arm="edges")
        assert (perf.loc[perf["tree"] == "bstm", "window_s"]
                == best["window_s"]).all()
