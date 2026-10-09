"""Stage 5b's design bookkeeping.

The scores are only interpretable beside a correct statement of what varied and
what did not, and that statement is now derived rather than written down. These
tests pin the derivation, including the case that caused the rewrite: an axis
that became varied while the document still called it fixed.
"""
import argparse
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from fmri_decomposition import bstm_selection as B


def args(**kw):
    d = dict(covariates=["Age", "Sex", "n_transitions"], features=["cells"],
             p_norm=["joint"], models=["ridge"], seeds=[0])
    d.update(kw)
    return argparse.Namespace(**d)


def sets(rows):
    d = pd.DataFrame(rows)
    d["window_s_num"] = d["window_s"].astype(float)
    d["aperture"] = np.where(d["window_s_num"] < 0, "frame (1 TR)",
                             d["window_s"].astype(str) + "s")
    d["source"] = np.where(d["window_s_num"] < 0, "per-TR activation",
                           "windowed DFC edges")
    parts = d["states"].str.rsplit("_", n=2, expand=True)
    d["method"], d["embedding"] = parts[0], parts[1]
    d["K"] = parts[2].astype(int)
    return d


class TestApertureOrder:
    def test_the_frame_aperture_comes_first(self):
        d = sets([{"atlas": "yeo7", "window_s": "30", "states": "HMM_pca3_8"},
                  {"atlas": "yeo7", "window_s": "-1", "states": "HMM_pca3_8"},
                  {"atlas": "yeo7", "window_s": "300", "states": "HMM_pca3_8"}])
        assert B._aperture_order(d) == ["frame (1 TR)", "30s", "300s"]

    def test_accepts_the_score_table_column_name_too(self):
        d = sets([{"atlas": "a", "window_s": "60", "states": "HMM_pca3_8"},
                  {"atlas": "a", "window_s": "-1", "states": "HMM_pca3_8"}])
        d = d.drop(columns=["window_s_num"])
        d["window_s"] = d["window_s"].astype(float)
        assert B._aperture_order(d)[0] == "frame (1 TR)"

    def test_orders_numerically_not_lexically(self):
        d = sets([{"atlas": "a", "window_s": w, "states": "HMM_pca3_8"}
                  for w in ("15", "120", "30", "300")])
        assert B._aperture_order(d) == ["15s", "30s", "120s", "300s"]


class TestLabelAxes:
    def table(self):
        return pd.DataFrame({
            "window_s": [30.0, -1.0, 30.0],
            "states": ["HMM_pca3_8", "MeanShift_umap3_7", "(covariates only)"],
            "kind": ["set", "set", "base"]})

    def test_splits_method_and_embedding_out_of_the_name(self):
        out = B._label_axes(self.table())
        assert out["method"].to_list()[:2] == ["HMM", "MeanShift"]
        assert out["embedding"].to_list()[:2] == ["pca3", "umap3"]

    def test_a_baseline_row_gets_a_label_not_a_guess(self):
        out = B._label_axes(self.table())
        assert out["method"].iloc[2] == "(covariates only)"
        assert out["embedding"].iloc[2] == "(covariates only)"

    def test_a_baseline_row_still_gets_its_real_aperture(self):
        # The aperture is what the figure plots the baseline against, so it must
        # survive even though the baseline has no state set.
        out = B._label_axes(self.table())
        assert out["aperture"].iloc[2] == "30s"

    def test_the_frame_aperture_is_named_not_printed_as_minus_one(self):
        out = B._label_axes(self.table())
        assert out["aperture"].iloc[1] == "frame (1 TR)"
        assert out["source"].iloc[1] == "per-TR activation"
        assert out["source"].iloc[0] == "windowed DFC edges"


class TestHeldFixed:
    def test_an_axis_with_one_value_is_listed_as_fixed(self):
        d = sets([{"atlas": "yeo7", "window_s": "30", "states": "HMM_pca3_8"}])
        rows = "\n".join(B._held_fixed_rows(d, args()))
        assert "**clustering method**" in rows and "HMM" in rows

    def test_an_axis_with_several_values_is_NOT_listed_as_fixed(self):
        # The bug this replaces: `method` stayed in the held-fixed table while
        # stage 4b was adding methods, so the document asserted something false.
        d = sets([{"atlas": "yeo7", "window_s": "30", "states": "HMM_pca3_8"},
                  {"atlas": "yeo7", "window_s": "30",
                   "states": "MeanShift_pca3_7"}])
        rows = "\n".join(B._held_fixed_rows(d, args()))
        assert "**clustering method**" not in rows

    def test_the_input_feature_stops_being_fixed_once_both_apertures_run(self):
        one = sets([{"atlas": "a", "window_s": "30", "states": "HMM_pca3_8"}])
        both = sets([{"atlas": "a", "window_s": "30", "states": "HMM_pca3_8"},
                     {"atlas": "a", "window_s": "-1", "states": "HMM_pca3_8"}])
        assert "**input feature**" in "\n".join(B._held_fixed_rows(one, args()))
        assert "**input feature**" not in "\n".join(
            B._held_fixed_rows(both, args()))

    def test_the_rows_that_are_always_fixed_are_always_there(self):
        d = sets([{"atlas": "a", "window_s": "30", "states": "HMM_pca3_8"}])
        rows = "\n".join(B._held_fixed_rows(d, args()))
        for always in ("PCA components", "censor policy", "cross-validation",
                       "covariates"):
            assert always in rows


class TestPhenoSpecParsing:
    """`--pheno PATH:SEP`. The separator is after the LAST colon, so a path that
    contains one still works -- and a spec with no colon at all must not read the
    path itself as the separator, which is what rpartition does unguarded."""

    def write(self, tmp_path, name, sep):
        p = tmp_path / name
        rows = ["CCID,Age,HADS", "CC01,70,Mild", "CC02,62,Severe"]
        p.write_text("\n".join(r.replace(",", sep) for r in rows) + "\n")
        return p

    def test_explicit_comma(self, tmp_path):
        p = self.write(tmp_path, "t.csv", ",")
        d = B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [])
        assert len(d) == 2 and d["y"].to_list() == [1, 3]

    def test_explicit_tab(self, tmp_path):
        p = self.write(tmp_path, "t.tsv", "\t")
        d = B.read_phenotype([f"{p}:\t"], "CCID", "HADS", ["Age"], [])
        assert len(d) == 2

    def test_no_separator_defaults_to_comma(self, tmp_path):
        # Was read as "separator = the whole path", which pandas treats as a
        # regex; it failed later as "id column not found".
        p = self.write(tmp_path, "t.csv", ",")
        d = B.read_phenotype([str(p)], "CCID", "HADS", ["Age"], [])
        assert len(d) == 2

    def test_a_colon_in_the_path_is_not_the_separator(self, tmp_path):
        d = tmp_path / "od:d"
        d.mkdir()
        p = self.write(d, "t.csv", ",")
        assert len(B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [])) == 2

    def test_a_multi_character_separator_is_refused_by_name(self, tmp_path):
        p = self.write(tmp_path, "t.csv", ",")
        with pytest.raises(SystemExit, match="not a single character"):
            B.read_phenotype([f"{p}:;;"], "CCID", "HADS", ["Age"], [])

    def test_a_numeric_target_bypasses_ORDINAL_LEVELS(self, tmp_path):
        # A new cohort whose score is 0-21 rather than Normal/Mild/... still
        # works: to_numeric runs first and the ordinal map is only a fallback.
        p = tmp_path / "t.csv"
        p.write_text("CCID,Age,total\nCC01,70,14\nCC02,62,3\n")
        d = B.read_phenotype([f"{p}:,"], "CCID", "total", ["Age"], [])
        assert sorted(d["y"].to_list()) == [3, 14]


class TestOrdinalLevelCoding:
    """A label set that is not Cam-CAN's.

    ORDINAL_LEVELS is Cam-CAN's HADS wording. Any other cohort's -- low/mid/high,
    absent/borderline/case -- used to code to NaN for every row, which emptied
    the frame and surfaced as "the phenotype join matched nothing": the ids were
    blamed for a problem in the labels.
    """

    def write(self, tmp_path, labels):
        p = tmp_path / "t.csv"
        rows = [f"CC{i:02d},{60 + i},{v}" for i, v in enumerate(labels)]
        p.write_text("CCID,Age,HADS\n" + "\n".join(rows) + "\n")
        return p

    def test_an_unknown_label_set_fails_as_a_label_problem(self, tmp_path):
        p = self.write(tmp_path, ["low", "mid", "high"])
        with pytest.raises(SystemExit) as e:
            B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [])
        msg = str(e.value)
        assert "could be coded" in msg
        assert "join matched nothing" not in msg
        # Both halves of the fix: what it saw, and the flag that accepts it.
        assert "'high'" in msg and "'low'" in msg
        assert "--ordinal-levels" in msg

    def test_declaring_the_levels_codes_them_in_the_order_given(self, tmp_path):
        p = self.write(tmp_path, ["low", "mid", "high"])
        d = B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [],
                             levels=["low", "mid", "high"])
        assert d.sort_values("sub")["y"].to_list() == [0, 1, 2]

    def test_the_order_is_the_claim_not_the_alphabet(self, tmp_path):
        """Reversing the list reverses the code -- the target is fitted as a
        number, so nothing else can carry which end is worse."""
        p = self.write(tmp_path, ["low", "mid", "high"])
        d = B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [],
                             levels=["high", "mid", "low"])
        assert d.sort_values("sub")["y"].to_list() == [2, 1, 0]

    def test_matching_is_case_insensitive(self, tmp_path):
        p = self.write(tmp_path, ["NORMAL", "severe"])
        d = B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [])
        assert d.sort_values("sub")["y"].to_list() == [0, 3]

    def test_some_labels_coding_is_not_fatal(self, tmp_path):
        """One bad value among good ones is a dropped subject, not a dead run."""
        p = self.write(tmp_path, ["Mild", "???", "Severe"])
        d = B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [])
        assert sorted(d["y"].to_list()) == [1, 3]

    def test_default_levels_still_apply_when_none_is_passed(self, tmp_path):
        p = self.write(tmp_path, ["Normal", "Moderate"])
        d = B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [], levels=None)
        assert sorted(d["y"].to_list()) == [0, 2]

    def test_the_flag_default_matches_the_module_constant(self):
        """The help text quotes ORDINAL_LEVELS; the default must be it."""
        import argparse
        p = argparse.ArgumentParser()
        B.add_arguments(p)
        got = p.parse_args(["--target", "x", "--cohort", "camcan",
                            "--pheno", "p.csv:,", "--task", "movie"])
        assert got.ordinal_levels == B.ORDINAL_LEVELS

    def test_an_empty_covariate_is_named_rather_than_blamed_on_the_join(
            self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("CCID,Age,HADS\nCC01,,Mild\nCC02,,Severe\n")
        with pytest.raises(SystemExit) as e:
            B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [])
        msg = str(e.value)
        assert "entirely empty" in msg and "Age" in msg
        assert "join matched nothing" not in msg


class TestTheRunAndTargetColumns:
    """The path says which ranking a table is; a path is not in the table."""

    def test_the_label_pairs_the_tree_with_the_task(self):
        assert B.run_label("bstm_selection", "movie") == "bstm movie"
        assert B.run_label("bstm_selection", "rest") == "bstm rest"
        assert B.run_label("fcm_selection", "movie") == "fcm movie"
        assert B.run_label("fcm_selection", "rest") == "fcm rest"

    def test_an_unusual_tree_name_still_gets_a_label(self):
        """`--output-name` is a flag, so the label cannot assume the two
        names this project ships."""
        assert B.run_label("pilot", "movie") == "pilot movie"

    def test_run_and_target_go_first_and_the_rest_keeps_its_order(self):
        d = pd.DataFrame({"model": ["ridge"], "mean": [0.1]})
        got = B.label_the_run(d, "bstm_selection", "rest", "hads")
        assert list(got.columns) == ["run", "target", "model", "mean"]
        assert got["run"].iloc[0] == "bstm rest"
        assert got["target"].iloc[0] == "hads"

    def test_it_does_not_mutate_what_it_was_given(self):
        """`scores_df` is labelled on the way to csv and then handed to
        _figures and _save_models, which key on their own columns."""
        d = pd.DataFrame({"model": ["ridge"], "mean": [0.1]})
        B.label_the_run(d, "bstm_selection", "rest", "hads")
        assert list(d.columns) == ["model", "mean"]


class TestOneRowPerSubject:
    """The silent-leakage guard, shared with select-fcm.

    A subject scanned at two tasks has one row per (task, sub). This stage
    scores ONE row per subject and merges on `sub` alone, so without the guard
    the same person lands in the train and the test fold of the same split:
    leakage, `n` doubled, and a ranking of whichever task sorted first. Nothing
    downstream could see it. fcm has had this check; bstm had not.
    """

    def frame(self):
        return pd.DataFrame({"sub": ["A", "A", "B"],
                             "task": ["Movie", "Rest", "Movie"]})

    def test_two_tasks_for_one_subject_is_refused(self):
        with pytest.raises(SystemExit) as e:
            B.one_row_per_subject(self.frame(), "t")
        msg = str(e.value)
        assert "same subject" in msg
        assert "Movie" in msg and "Rest" in msg
        assert "--keep-tasks" in msg

    def test_one_task_per_subject_passes_through_untouched(self):
        d = pd.DataFrame({"sub": ["A", "B"], "task": ["Movie", "Movie"]})
        assert len(B.one_row_per_subject(d, "t")) == 2

    def test_a_task_filter_resolves_it(self):
        got = B.one_row_per_subject(self.frame(), "t", ["Movie"])
        assert got["sub"].tolist() == ["A", "B"]

    def test_a_filter_matching_nothing_is_refused_with_what_exists(self):
        with pytest.raises(SystemExit) as e:
            B.one_row_per_subject(self.frame(), "t", ["Nope"])
        assert "Nope" in str(e.value) and "Movie" in str(e.value)

    def test_the_caller_supplies_its_own_remedy(self):
        """One guard, two stages: the shared part of the message is here and
        the stage-specific advice comes from the caller, so neither has to
        mention the other's flags."""
        with pytest.raises(SystemExit) as e:
            B.one_row_per_subject(self.frame(), "t",
                                  remedy=("re-run `static-fc --pool subject`",))
        assert "--pool subject" in str(e.value)
        with pytest.raises(SystemExit) as e:
            B.one_row_per_subject(self.frame(), "t")
        assert "--pool subject" not in str(e.value)

    def test_both_stages_spell_the_flag_the_same_way(self):
        """The symmetry: --task names the output partition in both, and
        --keep-tasks filters rows in both, with the same dest."""
        import argparse

        from fmri_decomposition import fcm_selection as F

        dests = []
        for mod in (B, F):
            p = argparse.ArgumentParser()
            mod.add_arguments(p)
            dests.append({a.option_strings[0]: a.dest
                          for a in p._actions if a.option_strings})
        for got in dests:
            assert got["--task"] == "task"
            assert got["--keep-tasks"] == "tasks"

    def test_keep_tasks_defaults_to_no_filter(self):
        """So a single-condition cohort needs no new flag -- the guard only
        fires when there is genuinely something to disambiguate."""
        import argparse

        p = argparse.ArgumentParser()
        B.add_arguments(p)
        a = p.parse_args(["--target", "t", "--cohort", "c",
                          "--pheno", "p.csv:,", "--task", "movie"])
        assert a.tasks is None


class TestConstantPredictionWarning:
    """A model that predicted a constant scores every arm identically.

    The full table, every arm present, and one model's block byte-identical
    down its length -- which reads as "no arm beats any other" and is really
    "this model never split".
    """

    def test_identical_scores_across_arms_are_called_out(self):
        d = pd.DataFrame({"model": ["lgbm"] * 3,
                          "arm": ["a", "b", "c"],
                          "n": [40, 40, 40],
                          "mean": [-0.37, -0.37, -0.37]})
        w = B.warn_if_a_model_never_split(d)
        assert len(w) == 1
        assert "lgbm" in w[0] and "CONSTANT" in w[0] and "n=40" in w[0]

    def test_a_model_that_differentiates_is_silent(self):
        d = pd.DataFrame({"model": ["ridge"] * 3,
                          "arm": ["a", "b", "c"],
                          "n": [600, 600, 600],
                          "mean": [0.11, 0.09, 0.02]})
        assert B.warn_if_a_model_never_split(d) == []

    def test_one_arm_is_not_evidence_of_anything(self):
        """A single row is trivially constant, which is not this failure."""
        d = pd.DataFrame({"model": ["lgbm"], "arm": ["a"], "n": [600],
                          "mean": [0.1]})
        assert B.warn_if_a_model_never_split(d) == []

    def test_each_model_is_judged_on_its_own_block(self):
        d = pd.DataFrame({"model": ["lgbm"] * 2 + ["ridge"] * 2,
                          "arm": ["a", "b"] * 2,
                          "n": [600] * 4,
                          "mean": [-0.37, -0.37, 0.11, 0.02]})
        w = B.warn_if_a_model_never_split(d)
        assert len(w) == 1 and "lgbm" in w[0]


class TestRestrictSubjects:
    def test_one_id_per_line(self, tmp_path):
        p = tmp_path / "k.txt"
        p.write_text("CC110033\n cc110045 \n\n")
        assert B.read_subject_list(str(p)) == {"CC110033", "CC110045"}

    def test_a_csv_with_a_sub_column(self, tmp_path):
        p = tmp_path / "k.csv"
        p.write_text("sub,excluded\nCC110033,False\nCC110045,False\n")
        assert B.read_subject_list(str(p)) == {"CC110033", "CC110045"}

    def test_ids_are_upper_cased_and_stripped_like_every_other_join(self,
                                                                   tmp_path):
        """`build` upper-cases the transition tables' `sub`, so a list that was
        not normalised the same way would match nothing and read as an id
        mismatch in the data."""
        p = tmp_path / "k.txt"
        p.write_text("cc110033\n")
        assert B.read_subject_list(str(p)) == {"CC110033"}

    def test_a_missing_file_is_refused_by_name(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            B.read_subject_list(str(tmp_path / "nope.txt"))
        assert "no such file" in str(e.value)

    def test_an_empty_list_is_refused_rather_than_emptying_the_run(self,
                                                                  tmp_path):
        p = tmp_path / "k.txt"
        p.write_text("\n  \n")
        with pytest.raises(SystemExit) as e:
            B.read_subject_list(str(p))
        assert "no ids" in str(e.value)


class TestNoDatasetSpecificDefaults:
    """The source must not carry a fact about one dataset on one filesystem.

    `--pheno` used to default to two absolute paths into one person's scratch
    space, and `--cohort` to "camcan". That is how `select` appeared to need
    neither -- the same thing `decompose --project camcan` was doing.
    """

    def parser(self):
        import argparse

        p = argparse.ArgumentParser()
        B.add_arguments(p)
        return p

    def test_a_phenotype_path_must_be_given(self):
        with pytest.raises(SystemExit):
            self.parser().parse_args(["--target", "t", "--cohort", "c"])

    def test_a_cohort_must_be_given(self):
        with pytest.raises(SystemExit):
            self.parser().parse_args(["--target", "t", "--pheno", "p.csv:,"])

    def test_no_absolute_path_survives_in_the_source(self):
        src = Path(B.__file__).read_text()
        assert "/project/" not in src
        assert "DEFAULT_PHENO" not in src

    def test_the_fcm_tree_has_no_dataset_defaults_either(self):
        import argparse

        from fmri_decomposition import fcm_selection as F

        p = argparse.ArgumentParser()
        F.add_arguments(p)
        with pytest.raises(SystemExit):           # --cohorts and --pheno
            p.parse_args(["--target", "t"])
        a = p.parse_args(["--target", "t", "--cohorts", "camcan",
                          "--pheno", "p.csv:,", "--task", "movie"])
        assert a.cohorts == ["camcan"]
        assert "/project/" not in Path(F.__file__).read_text()

    def test_no_job_script_names_a_cohort_or_an_atlas(self):
        """A job script that names a cohort only works for one project."""
        for name in ("static_fc.sbatch", "fcm_selection.sbatch",
                     "model_selection.sbatch"):
            src = (Path(B.__file__).parent.parent / "slurm" / name).read_text()
            run = [ln for ln in src.splitlines()
                   if "fmri_decomposition.cli" in ln or ln.startswith("    --")]
            joined = " ".join(run)
            assert "--cohorts camcan" not in joined, name
            assert "--atlas harvardoxford" not in joined, name


class TestTaskIsAPartitionNotASecondTree:
    """One tree per model family, partitioned by condition.

    It used to be `--output-name resting_bstm_selection`: a second tree with a
    different NAME for the same ranking on different data. `task=` is a
    partition instead, so the four summary.csv files (bstm/fcm x movie/rest)
    share one shape and one layout.

    `task=` and `target=` are the ONLY directory keys in these trees. atlas,
    window_s, K, states and cohort are columns of the one csv, because the
    comparison is rows you can sort rather than a join across folders.
    """

    def test_the_path_is_tree_then_task_then_target(self, tmp_path):
        import argparse

        p = argparse.ArgumentParser()
        B.add_arguments(p)
        a = p.parse_args(["--target", "hads", "--cohort", "camcan",
                          "--pheno", "p.csv:,", "--task", "movie"])
        out = (tmp_path / a.output_name / f"task={a.task}"
               / f"target={a.target}")
        assert out.relative_to(tmp_path).as_posix() == (
            "bstm_selection/task=movie/target=hads")

    def test_a_task_must_be_given(self):
        import argparse

        p = argparse.ArgumentParser()
        B.add_arguments(p)
        with pytest.raises(SystemExit):
            p.parse_args(["--target", "t", "--cohort", "c",
                          "--pheno", "p.csv:,"])

    def test_wipe_owns_the_task_partition_not_the_tree(self, tmp_path):
        """The folder cleared before a run is one task's, so a rest run can
        never empty the movie ranking beside it."""
        movie = tmp_path / "bstm_selection" / "task=movie" / "target=hads"
        rest = tmp_path / "bstm_selection" / "task=rest" / "target=hads"
        for d in (movie, rest):
            d.mkdir(parents=True)
            (d / "summary.csv").write_text("x")
        B._wipe(rest, parent="task=rest")
        assert not rest.exists()
        assert (movie / "summary.csv").exists()

    def test_wipe_refuses_a_task_it_was_not_told_to_own(self, tmp_path):
        out = tmp_path / "bstm_selection" / "task=movie" / "target=hads"
        out.mkdir(parents=True)
        (out / "summary.csv").write_text("x")
        with pytest.raises(SystemExit):
            B._wipe(out, parent="task=rest")
        assert (out / "summary.csv").exists()

    def test_the_fcm_tree_takes_the_same_label(self):
        import argparse

        from fmri_decomposition import fcm_selection as F

        p = argparse.ArgumentParser()
        F.add_arguments(p)
        a = p.parse_args(["--target", "t", "--cohorts", "camcan_rest",
                          "--pheno", "p.csv:,", "--task", "rest"])
        assert a.task == "rest"
        assert a.output_name == "fcm_selection"

    def test_the_row_filter_is_a_different_flag_from_the_label(self):
        """`--task` names the output partition and is yours to choose;
        `--keep-tasks` filters rows and takes the BIDS labels as the data
        spells them. `--task` beside `--tasks` was a collision nobody should
        have to notice."""
        import argparse

        from fmri_decomposition import fcm_selection as F

        p = argparse.ArgumentParser()
        F.add_arguments(p)
        a = p.parse_args(["--target", "t", "--cohorts", "c",
                          "--pheno", "p.csv:,", "--task", "movie",
                          "--keep-tasks", "Movie"])
        assert a.task == "movie" and a.tasks == ["Movie"]
