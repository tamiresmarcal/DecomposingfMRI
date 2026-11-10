"""The wiring between two cohorts' Settings, and the four ways it empties.

Every loop in `transfer_rows` can be emptied by a setting that looks right,
and an empty frame says nothing about which one it was. That is not a
hypothetical: a run of notebook 13 printed `0 model(s)`, skipped all 282 state
sets, and then failed seven frames down with `KeyError: 'score'` -- three
causes, one misleading symptom.

  1. THE MODELS came from the stored selection summaries, which exist only for
     the target columns the selections were RUN on. Asking for the raw HADS
     score rather than the banded category filtered `summ` to nothing, so
     there were no models to fit.
  2. THE TASK LABEL has to be the cohort's own hive value. Cam-CAN writes
     task=Movie; HCP writes task=MOVIE1..MOVIE4, four partitions of one
     condition that no equality can name.
  3. THE DUPLICATE ROWS that prefix matching then finds have to be pooled
     before anything is fitted, or `load_set` refuses.

So: a refusal that names the cause, in every one of the four cases, and a
default that does not depend on what the selections happened to rank.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hads_tables as ht  # noqa: E402
import transfer_tables as tt  # noqa: E402
from test_pool_tasks import CELLS, K, row_from_counts  # noqa: E402

ATLAS, WIN, STATES = "yeo7", -1.0, f"HMM1_pca3_{K}"
M = np.array([[10., 2, 1], [1, 8, 3], [0, 2, 7]])


def write_transitions(root, cohort, task, subs, win=WIN, states=STATES,
                      atlas=ATLAS):
    cell = (root / "transitions" / f"atlas={atlas}"
            / f"window_s={ht.fmt_window(win)}" / f"states={states}"
            / f"cohort={cohort}" / f"task={task}")
    cell.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(task)) % 2**31)
    rows = [row_from_counts(s, task, M + rng.integers(0, 4, (K, K)), 40)
            for s in subs]
    # The provenance columns `transitions.process` stamps into every row. They
    # are not decoration: `pool_fn` reads the cell off them to find the latents
    # for the exact dispersion, so a fixture without them exercises a path the
    # cluster never takes.
    pd.DataFrame(rows).assign(cohort=cohort, atlas=atlas,
                              window_s=float(win), states=states,
                              stride_s=1.0).to_parquet(cell / "subjects.parquet")


def write_summary(root, tree, task, target):
    d = root / f"{tree}_selection" / f"task={task}" / f"target={target}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "ridge", "arm": "all [cond]", "atlas": ATLAS,
                   "window_s": WIN, "states": STATES, "mean": .1, "std": .01,
                   "n": 100}]).to_csv(d / "summary.csv", index=False)


def write_pheno(path, subs, target_cols, seed=0):
    rng = np.random.default_rng(seed)
    d = {"sub": list(subs),
         "Age": rng.integers(20, 80, len(subs)),
         "Sex": ["MALE" if i % 2 else "FEMALE" for i in range(len(subs))]}
    for c in target_cols:
        d[c] = rng.integers(0, 15, len(subs))
    pd.DataFrame(d).to_csv(path, index=False)
    return path


def studies(tmp_path, train_targets, test_targets, summary_target=None,
            n_train=60, n_test=24, train_task="Movie", test_tasks=("MOVIE1", "MOVIE2"),
            task_match="prefix", pool=True):
    """Two Studies over one tmp outputs tree, as notebook 13 builds them."""
    root = tmp_path / "outputs"
    tr_subs = [f"CC{i:04d}" for i in range(n_train)]
    te_subs = [f"H{i:04d}" for i in range(n_test)]
    write_transitions(root, "camcan", train_task, tr_subs)
    for t in test_tasks:
        write_transitions(root, "hcp7t_fix", t, te_subs)
    for tree in ("bstm", "fcm"):
        write_summary(root, tree, "movie",
                      summary_target or list(train_targets.values())[0])
    tr_ph = write_pheno(tmp_path / "tr.csv", tr_subs, train_targets.values(), 0)
    te_ph = write_pheno(tmp_path / "te.csv", te_subs, test_targets.values(), 1)

    common = dict(root=root, covariates=["Age", "Sex", "n_transitions"],
                  baseline_covariates=["Age", "Sex"], pheno_id="sub",
                  cache_dir=None, models=None)
    train_cfg = ht.Settings(cohorts=["camcan"], tasks={"movie": train_task},
                            targets=train_targets,
                            pheno_files=[(tr_ph, ",")], **common)
    test_cfg = ht.Settings(
        cohorts=["hcp7t_fix"], tasks={"movie": "MOVIE"}, targets=test_targets,
        pheno_files=[(te_ph, ",")], task_match=task_match,
        pool_fn=(lambda d, k: tt.pool_tasks(d, k, log=lambda *_: None)) if pool else None,
        **common)
    return ht.Study(train_cfg), ht.Study(test_cfg)


TR = {"anxiety": "hads_anx_raw"}
TE = {"anxiety": "DSM_Anxi_Raw"}


# ------------------------------------------------------------ task matching ---
class TestTaskHit:
    def test_exact_is_case_insensitive(self):
        assert ht.task_hit("Movie", "movie")
        assert ht.task_hit("movie", "Movie")

    def test_exact_refuses_a_numbered_run(self):
        """The whole reason prefix matching had to be opt-in: MOVIE1 is not
        the movie condition under an equality, and silently treating it as one
        for every cohort would widen cohorts that never asked."""
        assert not ht.task_hit("MOVIE1", "MOVIE")

    def test_prefix_takes_every_numbered_run(self):
        assert all(ht.task_hit(f"MOVIE{i}", "MOVIE", "prefix") for i in range(1, 5))

    def test_prefix_does_not_cross_conditions(self):
        assert not ht.task_hit("REST1", "MOVIE", "prefix")
        assert not ht.task_hit("MOVIE1", "REST", "prefix")

    def test_prefix_is_still_case_insensitive(self):
        assert ht.task_hit("Movie", "MOVIE", "prefix")


class TestLoadSet:
    def test_prefix_and_pooling_give_one_row_per_subject(self, tmp_path):
        _, test = studies(tmp_path, TR, TE)
        d = test.load_set(dict(atlas=ATLAS, window_s=WIN, states=STATES), "movie")
        assert d["sub"].is_unique
        assert len(d) == 24
        assert set(d["task"]) == {"MOVIE1+MOVIE2"}

    def test_pooling_adds_the_transitions_of_both_runs(self, tmp_path):
        _, test = studies(tmp_path, TR, TE)
        space = dict(atlas=ATLAS, window_s=WIN, states=STATES)
        d = test.load_set(space, "movie")
        # the unpooled rows, read straight off the tree
        raw = pd.concat([pd.read_parquet(p) for p in
                         (test.cfg.root / "transitions").rglob("subjects.parquet")
                         if ht.hive(p, "cohort") == "hcp7t_fix"])
        want = raw.groupby("sub")["n_transitions"].sum()
        got = d.set_index("sub")["n_transitions"]
        assert np.allclose(got.loc[want.index], want)

    def test_without_pool_fn_the_duplicate_is_refused(self, tmp_path):
        _, test = studies(tmp_path, TR, TE, pool=False)
        with pytest.raises(SystemExit) as e:
            test.load_set(dict(atlas=ATLAS, window_s=WIN, states=STATES), "movie")
        assert "appears twice" in str(e.value)
        assert "pool_fn" in str(e.value)        # and says what to do about it

    def test_exact_matching_finds_no_numbered_run_and_says_what_it_wanted(self, tmp_path):
        _, test = studies(tmp_path, TR, TE, task_match="exact")
        with pytest.raises(SystemExit) as e:
            test.load_set(dict(atlas=ATLAS, window_s=WIN, states=STATES), "movie")
        msg = str(e.value)
        assert "no transition table" in msg
        assert "'MOVIE'" in msg and "==" in msg   # the rule it applied

    def test_the_training_side_is_untouched_by_the_test_settings(self, tmp_path):
        """Cam-CAN has one run per condition. Nothing here may pool it."""
        train, _ = studies(tmp_path, TR, TE)
        d = train.load_set(dict(atlas=ATLAS, window_s=WIN, states=STATES), "movie")
        assert len(d) == 60 and "n_runs_pooled" not in d


class TestMatrixReusesALoadedTable:
    def test_passing_t_gives_the_same_matrix(self, tmp_path):
        train, _ = studies(tmp_path, TR, TE)
        space = dict(atlas=ATLAS, window_s=WIN, states=STATES)
        a = train.matrix(space, "movie", "anxiety", "cells")
        t = train.load_set(space, "movie")
        b = train.matrix(space, "movie", "anxiety", "cells", t=t)
        assert np.allclose(a[0], b[0]) and a[2] == b[2]

    def test_the_passed_table_is_what_is_used(self, tmp_path):
        """Not a tautology: `t` must actually be read instead of the disk, or
        a pooled cohort re-pools once per arm."""
        train, _ = studies(tmp_path, TR, TE)
        space = dict(atlas=ATLAS, window_s=WIN, states=STATES)
        t = train.load_set(space, "movie").head(10)
        X, y, _, _ = train.matrix(space, "movie", "anxiety", "cells", t=t)
        assert len(y) == 10


# --------------------------------------------------------------- the models ---
class TestModelDefault:
    def test_a_target_the_selections_never_ran_on_still_has_models(self, tmp_path):
        """The bug exactly: summaries on disk for one target column, the
        notebook asking for another. `summ` empties, and the old default read
        the model names out of it."""
        train, test = studies(tmp_path, TR, TE, summary_target="something_else")
        assert train.models() == []                      # the old default
        rows = tt.transfer_rows(train, test, models=None, arms=["cells"],
                                n_boot=20, log=lambda *_: None)
        assert set(rows["model"]) == set(tt.DEFAULT_MODELS)

    def test_cfg_models_win_over_the_default(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        train.cfg.models = ["ridge"]
        rows = tt.transfer_rows(train, test, arms=["cells"], n_boot=20,
                                log=lambda *_: None)
        assert set(rows["model"]) == {"ridge"}

    def test_an_explicit_models_wins_over_both(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        train.cfg.models = ["hgb"]
        rows = tt.transfer_rows(train, test, models=["ridge"], arms=["cells"],
                                n_boot=20, log=lambda *_: None)
        assert set(rows["model"]) == {"ridge"}


# ------------------------------------------------------------- the refusals ---
class TestRefusals:
    def test_no_common_target_label_names_both_sides(self, tmp_path):
        train, test = studies(tmp_path, TR, {"depression": "DSM_Depr_Raw"})
        with pytest.raises(SystemExit) as e:
            tt.transfer_rows(train, test, log=lambda *_: None)
        assert "anxiety" in str(e.value) and "depression" in str(e.value)

    def test_no_common_task_key_says_the_keys_must_match(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        test.cfg.tasks = {"film": "MOVIE"}
        with pytest.raises(SystemExit) as e:
            tt.transfer_rows(train, test, log=lambda *_: None)
        assert "no task in common" in str(e.value) and "film" in str(e.value)

    def test_no_shared_state_set_points_at_stage_5a(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        test.cfg.cohorts = ["nobody"]
        with pytest.raises(SystemExit) as e:
            tt.transfer_rows(train, test, log=lambda *_: None)
        assert "STOP_AFTER=transitions" in str(e.value)

    def test_no_model_is_refused_rather_than_looped_zero_times(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        with pytest.raises(SystemExit) as e:
            tt.transfer_rows(train, test, models=[], log=lambda *_: None)
        assert "no model to fit" in str(e.value)

    def test_every_cell_skipped_is_refused_with_the_commonest_reason(self, tmp_path):
        """The real failure: the sets are shared, so the grid is not empty,
        but no cell can be loaded. The old code returned an empty frame."""
        train, test = studies(tmp_path, TR, TE)
        test.cfg.task_match = "exact"           # MOVIE1 no longer matches MOVIE
        with pytest.raises(SystemExit) as e:
            tt.transfer_rows(train, test, arms=["cells"], log=lambda *_: None)
        assert "no configuration could be scored" in str(e.value)
        assert "no transition table" in str(e.value)

    def test_the_skip_log_counts_causes_instead_of_printing_each(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        test.cfg.task_match = "exact"
        seen = []
        with pytest.raises(SystemExit):
            tt.transfer_rows(train, test, arms=["cells", "occupancy", "dynamics"],
                             log=seen.append)
        skips = [m for m in seen if "skipped" in m]
        assert len(skips) == 1 and "1x:" in skips[0]   # one cause, counted once


class TestReportRowsGuard:
    def test_an_empty_frame_is_refused_not_a_keyerror(self):
        with pytest.raises(SystemExit) as e:
            tt.as_report_rows(pd.DataFrame([]))
        assert "EMPTY" in str(e.value)

    def test_a_frame_missing_one_column_says_which(self):
        with pytest.raises(SystemExit) as e:
            tt.as_report_rows(pd.DataFrame({"score": [.1], "lo": [0.]}))
        assert "base_score" in str(e.value)

    def test_a_good_frame_still_converts(self, tmp_path):
        train, test = studies(tmp_path, TR, TE)
        rows = tt.transfer_rows(train, test, models=["ridge"], arms=["cells"],
                                n_boot=20, log=lambda *_: None)
        out = tt.as_report_rows(rows)
        assert np.allclose(out["mean"], out["score"])
        assert (out["std"] > 0).all()


class TestReasonCollapse:
    def test_two_cells_with_one_cause_collapse_to_one_key(self):
        a = tt._reason(SystemExit(
            "no transition table for outputs/transitions/atlas=yeo7/"
            "window_s=-1/states=HMM1_pca3_8 task=rest"))
        b = tt._reason(SystemExit(
            "no transition table for outputs/transitions/atlas=schaefer200/"
            "window_s=30/states=KMeans_pca5_12 task=rest"))
        assert a == b

    def test_a_different_cause_stays_different(self):
        a = tt._reason(SystemExit("no transition table for atlas=yeo7 task=rest"))
        b = tt._reason(SystemExit("covariate 'Age' is in neither table"))
        assert a != b


class TestTheFiguresReadTheTransferredRows:
    """B1/B2/B2.K are the report's figures, reused. Two things they read are
    not in transfer_rows' own output, and both failed silently-ish: `base_n`
    raised a KeyError deep inside B1, and the captions claimed a CV score and
    a spread over fold seeds, which a transfer has neither of."""

    def rows(self, tmp_path):
        train, test = studies(tmp_path, {"anxiety": "hads_anx"},
                              {"anxiety": "DSM_Anxi_Raw"})
        return train, tt.as_report_rows(tt.transfer_rows(
            train, test, models=["ridge"], arms=["cells", "dynamics"],
            n_boot=20, log=lambda *_: None))

    def test_n_and_base_n_are_the_test_cohort(self, tmp_path):
        _, rows = self.rows(tmp_path)
        assert (rows["n"] == rows["n_test"]).all()
        assert (rows["base_n"] == rows["n_test"]).all()

    def test_b1_draws_without_a_keyerror(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import hads_plots as hp
        train, rows = self.rows(tmp_path)
        hp.fig_rankings(rows, train.cfg, train.above_both)
        plt.close("all")

    def test_the_axis_label_is_the_settings_one(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import hads_plots as hp
        train, rows = self.rows(tmp_path)
        train.cfg.score_name = "transfer score, HCP (Spearman)"
        train.cfg.spread_name = "sd of the bootstrap over the test subjects"
        hp.fig_rankings(rows, train.cfg, train.above_both)
        fig = plt.gcf()
        # `layout` title-cases every label, so compare case-insensitively
        labels = [a.get_ylabel().lower() for a in fig.axes if a.get_ylabel()]
        assert "transfer score, hcp (spearman)" in labels
        assert "bootstrap" in fig._suptitle.get_text().lower()
        plt.close("all")

    def test_the_default_label_is_still_the_reports(self):
        """The knob must not change what notebooks 9-11 draw."""
        assert ht.Settings().score_name == "CV score (Spearman)"
        assert ht.Settings().spread_name == "sd over seeds"


# ------------------------------------------------- how a cohort spells things ---
def hcp_pheno(path, subs, target_cols, seed=1, band=True):
    """HCP's own spelling: Gender, Age_in_Yrs, and an `Age` that is a BAND."""
    rng = np.random.default_rng(seed)
    d = {"sub": list(subs),
         "Age_in_Yrs": rng.integers(22, 36, len(subs)),
         "Gender": ["M" if i % 2 else "F" for i in range(len(subs))]}
    if band:
        d["Age"] = ["26-30"] * len(subs)
    for c in target_cols:
        d[c] = rng.integers(0, 15, len(subs))
    pd.DataFrame(d).to_csv(path, index=False)
    return path


SEX = {"Sex": {"M": "M", "MALE": "M", "F": "F", "FEMALE": "F"}}
ALIASES = {"Sex": "Gender", "Age": "Age_in_Yrs"}


def hcp_study(tmp_path, aliases=ALIASES, values=SEX, band=True, n=24, **kw):
    root = tmp_path / "outputs"
    subs = [f"H{i:04d}" for i in range(n)]
    for t in ("MOVIE1", "MOVIE2"):
        write_transitions(root, "hcp7t_fix", t, subs)
    for tree in ("bstm", "fcm"):
        write_summary(root, tree, "movie", "DSM_Anxi_Raw")
    ph = hcp_pheno(tmp_path / "hcp.csv", subs, TE.values(), band=band)
    cfg = ht.Settings(
        root=root, cohorts=["hcp7t_fix"], tasks={"movie": "MOVIE"}, targets=TE,
        pheno_files=[(ph, ",")], pheno_id="sub", cache_dir=None,
        covariates=["Age", "Sex", "n_transitions"],
        baseline_covariates=["Age", "Sex"],
        covariate_aliases=aliases, covariate_values=values,
        task_match="prefix",
        pool_fn=lambda d, k: tt.pool_tasks(d, k, log=lambda *_: None), **kw)
    return cfg


class TestCovariateSpelling:
    """`KeyError: ['Sex']`, four frames below the setting that caused it.

    HCP's phenotype has Gender, not Sex. `_pheno` renamed case-insensitively,
    found nothing, and then DROPPED the covariate on its way out -- so the
    frame came back looking fine and `matrix` raised a bare KeyError on a
    column nobody had said was missing.
    """

    def test_the_alias_supplies_the_canonical_name(self, tmp_path):
        st = ht.Study(hcp_study(tmp_path))
        assert {"Age", "Sex"} <= set(st.pheno.columns)
        assert set(st.pheno["Sex"]) == {"M", "F"}

    def test_the_alias_beats_a_column_that_already_has_the_name(self, tmp_path, capsys):
        """HCP's file carries BOTH Age_in_Yrs and an `Age` holding the
        open-access band. Renaming onto the taken name would leave two columns
        called Age; taking the band would leave Age all-NaN."""
        st = ht.Study(hcp_study(tmp_path, band=True))
        assert st.pheno["Age"].between(22, 36).all()
        assert "Age_in_Yrs" in capsys.readouterr().out      # and it said so

    def test_no_alias_and_no_column_is_refused_by_name(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            ht.Study(hcp_study(tmp_path, aliases=None))
        msg = str(e.value)
        assert "'Sex'" in msg and "covariate_aliases" in msg
        assert "Gender" in msg                 # the column it does have

    def test_a_band_where_a_number_belongs_is_refused(self, tmp_path):
        """Without the alias, `Age` resolves to '26-30' and coerces to NaN for
        every subject -- which used to empty the frame silently."""
        with pytest.raises(SystemExit) as e:
            ht.Study(hcp_study(tmp_path, aliases={"Sex": "Gender"}))
        assert "not a number anywhere" in str(e.value)
        assert "26-30" in str(e.value)

    def test_an_unmapped_value_is_refused_rather_than_guessed(self, tmp_path):
        cfg = hcp_study(tmp_path, values={"Sex": {"MALE": "M", "FEMALE": "F"}})
        with pytest.raises(SystemExit) as e:
            ht.Study(cfg)
        assert "'M'" in str(e.value) and "covariate_values" in str(e.value)

    def test_the_two_cohorts_line_up_once_the_values_are_mapped(self, tmp_path):
        """End to end: MALE/FEMALE and M/F must give ONE dummy column name, or
        align_columns refuses and no row is ever produced."""
        train, _ = studies(tmp_path / "a", TR, TE)
        train.cfg.covariate_values = SEX
        train = ht.Study(train.cfg)
        test = ht.Study(hcp_study(tmp_path / "b"))
        # one tree per Study here, so point the test cohort at the train tree
        test.cfg.root = train.cfg.root
        test = ht.Study(test.cfg)
        rows = tt.transfer_rows(train, test, models=["ridge"], arms=["cells"],
                                n_boot=20, log=lambda *_: None)
        assert len(rows) and rows["n_test"].iloc[0] == 24

    def test_unmapped_values_do_reach_align_columns(self, tmp_path):
        """Not a hypothetical hazard: without the value map the dummy columns
        really are Sex_MALE against Sex_M, and the transfer refuses."""
        train, _ = studies(tmp_path / "a", TR, TE)
        test = ht.Study(hcp_study(tmp_path / "b", values=None))
        test.cfg.root = train.cfg.root
        test = ht.Study(test.cfg)
        with pytest.raises(SystemExit) as e:
            tt.transfer_rows(train, test, models=["ridge"], arms=["cells"],
                             n_boot=20, log=lambda *_: None)
        assert "Sex_" in str(e.value)
