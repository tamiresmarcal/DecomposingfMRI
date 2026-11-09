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


class TestB1Fallback:
    """"Nothing cleared the baseline" is a different statement from "nothing
    ranks better than chance", and with HCP's anxiety baseline at rho ~ 0.2 the
    first can be true while the second is false. So an empty panel falls back
    to above-ZERO -- and says, in its own caption, that it has."""

    def frame(self, base, means):
        d = pd.DataFrame({
            "entity": [f"e{i}" for i in range(len(means))] * 2,
            "mean": list(means) + list(means),
            "std": [.01] * 2 * len(means),
            "task": ["movie"] * len(means) + ["rest"] * len(means),
            "base_mean": base, "base_sd": .05})
        return d

    def study(self):
        cfg = ht.Settings(tasks={"movie": "Movie", "rest": "Rest"},
                          targets={"anxiety": "a"}, cache_dir=None)
        st = object.__new__(ht.Study)
        st.cfg = cfg
        return st

    def test_above_both_is_empty_when_the_baseline_is_high(self):
        st = self.study()
        d = self.frame(.20, [.10, .15])
        assert st.above_both(d) == []

    def test_above_zero_still_finds_them(self):
        st = self.study()
        d = self.frame(.20, [.10, .15])
        assert st.above_zero(d) == ["e1", "e0"]      # best worst-task first

    def test_above_zero_needs_every_task(self):
        st = self.study()
        d = self.frame(.20, [.10, .15])
        d.loc[(d["entity"] == "e0") & (d["task"] == "rest"), "mean"] = -.05
        assert st.above_zero(d) == ["e1"]

    def test_the_caption_says_which_rule_is_showing(self, rows, tree):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import hads_plots as hp
        import transfer_config as tcfg
        _, d = rows
        train, _, train_cfg, _ = tcfg.studies(
            "raw", root=tree["root"], camcan=tree["camcan"],
            hcp_pheno=tree["hcp"], models=["ridge"])
        # a baseline nothing can clear
        d = d.assign(base_mean=9.0, base_sd=.01)
        hp.fig_rankings(d, train_cfg, train.above_both,
                        fallback=train.above_zero)
        titles = " ".join(a.get_title() or "" for a in plt.gcf().axes)
        legs = " ".join(t.get_text() for a in plt.gcf().axes
                        if a.get_legend() for t in [a.get_legend().get_title()])
        assert "zero" in (titles + legs).lower()
        assert "not better than age" in (titles + legs).lower()
        plt.close("all")

    def test_without_a_fallback_nothing_changes(self, rows, tree):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import hads_plots as hp
        import transfer_config as tcfg
        _, d = rows
        train, _, train_cfg, _ = tcfg.studies(
            "raw", root=tree["root"], camcan=tree["camcan"],
            hcp_pheno=tree["hcp"], models=["ridge"])
        hp.fig_rankings(d.assign(base_mean=9.0, base_sd=.01), train_cfg,
                        train.above_both)
        legs = " ".join(a.get_legend().get_title().get_text()
                        for a in plt.gcf().axes if a.get_legend())
        assert "zero" not in legs.lower()
        plt.close("all")


class TestTheShapGuard:
    """shap is not in the image we run. Ridge needs none of it; a tree does.
    The refusal has to say both, before anything is fitted."""

    def test_a_tree_model_without_shap_refuses_by_name(self, tree, monkeypatch):
        import importlib.util

        import transfer_config as tcfg
        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda m, *a, **k: None if m == "shap"
                            else real(m, *a, **k))
        train, test, _, _ = tcfg.studies("raw", root=tree["root"],
                                         camcan=tree["camcan"],
                                         hcp_pheno=tree["hcp"])
        with pytest.raises(SystemExit) as e:
            tt.transfer_shares(train, test,
                               dict(atlas="yeo7", window_s=-1.0,
                                    states=f"HMM1_pca3_{K}"),
                               "cells", "hgb", log=lambda *_: None)
        msg = str(e.value)
        assert "stage45_shap.def" in msg
        assert "ridge" in msg and "exact" in msg
        assert "without `shares`" in msg

    def test_ridge_needs_no_shap_at_all(self, tree, monkeypatch):
        import importlib.util

        import transfer_config as tcfg
        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda m, *a, **k: None if m == "shap"
                            else real(m, *a, **k))
        train, test, _, _ = tcfg.studies("raw", root=tree["root"],
                                         camcan=tree["camcan"],
                                         hcp_pheno=tree["hcp"],
                                         models=["ridge"])
        sh = tt.transfer_shares(train, test,
                                dict(atlas="yeo7", window_s=-1.0,
                                     states=f"HMM1_pca3_{K}"),
                                "cells", "ridge", log=lambda *_: None)
        assert len(sh) and np.allclose(
            sh.groupby(["task", "phenotype"])["share"].sum(), 1.0)


class TestInstallingIntoTheImage:
    """Two pip forms that look right and are not, and the one that is.

    `pip install --user` is refused here and NOT because of
    PYTHONNOUSERSITE: `/opt/venv` is a venv, and a venv has
    `site.ENABLE_USER_SITE = False` whatever that variable says. I had the
    cause wrong in containers/README.md until this was measured.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def v(tmp_path_factory):
        import subprocess
        import sys as _s
        d = tmp_path_factory.mktemp("venv")
        subprocess.run([_s.executable, "-m", "venv", str(d / "v")], check=True,
                       capture_output=True)
        return d / "v"

    def test_a_venv_disables_user_site_on_its_own(self, v):
        import subprocess
        code = "import site; print(site.ENABLE_USER_SITE)"
        for env in ({}, {"PYTHONNOUSERSITE": "1"}):
            import os
            r = subprocess.run([str(v / "bin" / "python"), "-c", code],
                               capture_output=True, text=True,
                               env={**os.environ, **env})
            assert r.stdout.strip() == "False", (env, r.stdout, r.stderr)

    def test_target_plus_append_is_the_form_that_works(self, v, tmp_path):
        """What the notebook cell does: install to a writable directory and
        append it, so nothing there can shadow a container package."""
        import subprocess
        libs = tmp_path / "pylibs"
        r = subprocess.run(
            [str(v / "bin" / "pip"), "install", "--no-cache-dir", "--no-deps",
             "--target", str(libs), "slicer==0.0.8"],
            capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"no network for pip: {r.stderr[:200]}")
        r = subprocess.run(
            [str(v / "bin" / "python"), "-c",
             f"import sys; sys.path.append({str(libs)!r}); import slicer; "
             f"print(slicer.__file__)"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert str(libs) in r.stdout

    def test_the_notebook_appends_rather_than_prepends(self):
        """`insert(0, ...)` would let an extra directory shadow the
        container's numpy, which is a segfault rather than an ImportError."""
        import json
        nb = json.load(open(Path(__file__).resolve().parent.parent
                            / "notebooks" / "13_camcan_to_hcp.ipynb"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"]
                        if c["cell_type"] == "code")
        assert "EXTRA_PYLIBS" in src
        assert "sys.path.append(EXTRA_PYLIBS)" in src
        assert "sys.path.insert(0, EXTRA_PYLIBS)" not in src

    def test_the_readme_does_not_blame_pythonnousersite(self):
        t = (Path(__file__).resolve().parent.parent / "containers"
             / "README.md").read_text()
        assert "is **not** `PYTHONNOUSERSITE`" in t
        assert "--target" in t and "--no-deps" in t


class TestE1bScatter:
    """One number per panel in E1; this is what it is made of. The hazard is
    the opposite of the usual one: the two axes are DIFFERENT questionnaires,
    so the familiar y = x line would draw an agreement nothing here claims."""

    @pytest.fixture(autouse=True)
    def _agg(self):
        import matplotlib
        matplotlib.use("Agg")

    @pytest.fixture(scope="class")
    @staticmethod
    def pred(tree, rows):
        import transfer_config as tcfg
        _, d = rows
        train, test, cfg, _ = tcfg.studies("raw", root=tree["root"],
                                           camcan=tree["camcan"],
                                           hcp_pheno=tree["hcp"],
                                           models=["ridge"])
        best = d[d["tree"] == "bstm"].sort_values("score", ascending=False).iloc[0]
        return tt.transfer_predictions(train, test, best, fc_arm=cfg.fc_arm,
                                       log=lambda *_: None), cfg, best

    def test_it_covers_both_trees_both_tasks_both_targets(self, pred):
        p, _, _ = pred
        assert set(p["tree"]) == {"bstm", "fcm"}
        assert set(p["task"]) == {"movie", "rest"}
        assert set(p["phenotype"]) == {"anxiety", "depression"}

    def test_one_row_per_test_subject_per_cell(self, pred):
        p, _, _ = pred
        n = p.groupby(["tree", "task", "phenotype"]).size()
        assert (n == 30).all(), n.to_dict()

    def test_only_the_named_fcm_arm(self, pred):
        p, _, cfg = pred
        assert set(p.loc[p["tree"] == "fcm", "arm"]) == {"edges"}

    def test_the_predictions_reproduce_the_stored_score(self, pred, rows):
        """The point of collecting them from the same fit: a scatter from a
        second fit is a scatter of a different model."""
        from scipy import stats
        p, _, best = pred
        _, d = rows
        d = d[(d["tree"] == "bstm") & (d["states"] == best["states"])
              & (d["atlas"] == best["atlas"])
              & (d["window_s"] == best["window_s"])
              & (d["arm"] == best["arm"]) & (d["model"] == best["model"])]
        for _, r in d.iterrows():
            h = p[(p["tree"] == "bstm") & (p["task"] == r["task"])
                  & (p["phenotype"] == r["phenotype"])]
            rho = stats.spearmanr(h["pred"], h["y"]).statistic
            assert abs(rho - r["score"]) < 1e-9, (r["task"], r["phenotype"])

    def test_it_draws_and_says_there_is_no_identity_line(self, pred):
        import matplotlib.pyplot as plt
        import hads_plots as hp
        p, cfg, _ = pred
        hp.fig_scatter(p, cfg, title="E1b", n_boot=40)
        fig = plt.gcf()
        notes = " ".join(t.get_text() for t in fig.texts).lower()
        assert "no identity line" in notes
        assert "jitter" in notes and "order" in notes
        assert "bootstrap band" in notes and "direction" in notes
        # 2 trees x (2 targets x 2 tasks)
        assert len([a for a in fig.axes if a.collections]) == 8
        plt.close("all")

    def test_the_jitter_never_touches_the_statistics(self, pred):
        """The rho printed in a panel is of the stored values, not the drawn
        ones -- otherwise the figure would disagree with the table beside it."""
        import matplotlib.pyplot as plt
        import hads_plots as hp
        p, cfg, _ = pred
        hp.fig_scatter(p, cfg, title="E1b", jitter=0, n_boot=20)
        a = [t.get_text() for ax in plt.gcf().axes for t in ax.texts
             if "ρ" in t.get_text()]
        plt.close("all")
        hp.fig_scatter(p, cfg, title="E1b", jitter=5.0, n_boot=20)
        b = [t.get_text() for ax in plt.gcf().axes for t in ax.texts
             if "ρ" in t.get_text()]
        plt.close("all")
        assert a == b and len(a) == 8


class TestB2KIsGone:
    def test_the_notebook_no_longer_draws_it(self):
        import json
        nb = json.load(open(Path(__file__).resolve().parent.parent
                            / "notebooks" / "13_camcan_to_hcp.ipynb"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"]
                        if c["cell_type"] == "code")
        assert "fig_k_curve" not in src

    def test_but_the_function_stays_for_notebook_10(self):
        import hads_plots as hp
        assert hasattr(hp, "fig_k_curve")


class TestE1HasNoConnectingLine:
    def test_no_line_joins_the_two_trees(self, rows, tree):
        """A slope between two MODELS on a nominal axis reads as a trend."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import hads_plots as hp
        import transfer_config as tcfg
        _, d = rows
        _, _, cfg, _ = tcfg.studies("raw", root=tree["root"],
                                    camcan=tree["camcan"],
                                    hcp_pheno=tree["hcp"], models=["ridge"])
        best = d[d["tree"] == "bstm"].sort_values("score", ascending=False).iloc[0]
        perf = tt.transfer_perf(d, best, fc_arm=cfg.fc_arm)
        hp.fig_performance(perf, cfg, title="E1", legend={"bstm": "x", "fcm": "y"},
                           model=best["model"])
        # A connector spans the two x positions AND changes y. The baseline
        # also spans them and does not change y -- it is a horizontal
        # reference, not a trend, and must stay.
        import numpy as _np
        spans = []
        for ax in plt.gcf().axes:
            for ln in ax.get_lines():
                xs, ys = _np.asarray(ln.get_xdata()), _np.asarray(ln.get_ydata())
                # errorbar keeps a Line2D for the markers even with
                # ls="none", and it spans both x. A line nobody can see is
                # not the thing being tested.
                visible = (str(ln.get_linestyle()).lower() not in ("none", "")
                           and ln.get_linewidth() > 0)
                if visible and len(xs) == 2 and abs(xs[1] - xs[0]) > .5:
                    spans.append((xs.tolist(), ys.tolist()))
        assert spans, "the baseline (a visible horizontal line) should still span"
        sloped = [s_ for s_ in spans if abs(s_[1][1] - s_[1][0]) > 1e-12]
        assert not sloped, f"a line joins the trees: {sloped}"
        plt.close("all")


class TestTheTrendLine:
    """A per-quantile median joined point to point was the first version: a
    staircase whose every step was an artefact of where a bin edge fell. One
    line and one interval instead, from the same bootstrap over test subjects
    as every other interval in this notebook."""

    def test_the_line_follows_the_data(self):
        import hads_plots as hp
        rng = np.random.default_rng(0)
        x = rng.normal(size=300)
        y = 2 * x + rng.normal(0, .3, 300)
        g, line, lo, hi = hp._trend(x, y, n_boot=200)
        assert np.all(np.diff(line) > 0)                  # rises with x
        assert abs(np.polyfit(g, line, 1)[0] - 2) < .2

    def test_the_band_brackets_the_line(self):
        import hads_plots as hp
        rng = np.random.default_rng(1)
        x = rng.normal(size=120)
        y = x + rng.normal(0, 1.5, 120)
        g, line, lo, hi = hp._trend(x, y, n_boot=300)
        assert np.all(lo <= line + 1e-9) and np.all(hi >= line - 1e-9)

    def test_the_band_is_wider_with_fewer_subjects(self):
        """The interval has to mean n, or it is decoration."""
        import hads_plots as hp
        rng = np.random.default_rng(2)
        wide = []
        for n in (30, 400):
            x = rng.normal(size=n)
            y = x + rng.normal(0, 2, n)
            _, _, lo, hi = hp._trend(x, y, n_boot=300)
            wide.append(float(np.median(hi - lo)))
        assert wide[0] > wide[1] * 1.5, wide

    def test_no_staircase_markers_are_drawn(self):
        """The old version used marker='_' per bin edge."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import hads_plots as hp
        import hads_tables as ht
        rng = np.random.default_rng(0)
        d = pd.DataFrame({"tree": "bstm", "task": "rest", "phenotype": "anxiety",
                          "arm": "cells", "model": "ridge",
                          "pred": rng.normal(size=80),
                          "y": rng.integers(0, 6, 80).astype(float)})
        cfg = ht.Settings(tasks={"rest": "Rest"}, targets={"anxiety": "a"},
                          cache_dir=None)
        hp.fig_scatter(d, cfg, title="x", n_boot=20)
        marks = {ln.get_marker() for ax in plt.gcf().axes for ln in ax.get_lines()}
        assert "_" not in marks
        plt.close("all")
