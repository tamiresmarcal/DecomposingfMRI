"""Which conditions and which subjects the fit was fitted on.

`train_cohorts` names COHORTS, and the training cells take every task those
cohorts have. So before this, two things the comparisons now depend on were
invisible to `model_hash`:

  * a training cohort that GAINED a condition changed the training rows and
    nothing in the fit description moved -- "fit on movie" and "fit on movie
    and rest" were the same command before and after the rest shards landed,
    under one hash;
  * a fit restricted to some of a cohort's subjects was not recorded at all.

These pin the two flags that say it out loud, the digest that keeps the subject
LIST out of every output file, and above all the refusals -- because the
failure mode here is a run that looks like it worked and stamps its outputs
with a claim about a model that was never fitted.
"""
from __future__ import annotations

import argparse

import pytest

from fmri_decomposition import decompose as D
from fmri_decomposition.io import read_subject_list, subject_list_digest

EDGES = ["A__B", "A__C", "B__C"]


def parse(*argv):
    p = argparse.ArgumentParser()
    D.add_arguments(p)
    argv = list(argv)
    if "--censor-policy" not in argv:
        argv += ["--censor-policy", "none"]
    a = p.parse_args(argv)
    a.output_root = None
    return a


def h(a, window_s="30", features=EDGES):
    return D.model_hash(D.fit_meta(a, window_s, features), features)


BASE = ("--atlas", "yeo7", "--window-s", "30")


class TestAbsentMeansEveryTaskAndEverySubject:
    """Both keys follow the rule the payload already had: present only when the
    flag is given. An absent key is what every latents file on disk was written
    with, so adding these must move no hash -- the frozen DFC_HASH in
    test_decompose.py is the other half of this check."""

    def test_neither_key_is_in_the_default_payload(self):
        meta = D.fit_meta(parse(*BASE), "30", EDGES)
        assert "train_tasks" not in meta and "train_subjects" not in meta

    def test_the_flags_default_to_none(self):
        a = parse(*BASE)
        assert a.train_tasks is None and a.train_subjects is None

    def test_train_subjects_is_none_without_the_flag(self):
        assert D.train_subjects(parse(*BASE)) is None

    def test_train_subjects_tolerates_an_args_without_the_attribute(self):
        """`fit_meta` is reachable from code that built its own namespace."""
        assert D.train_subjects(argparse.Namespace()) is None


class TestTrainTasksIsHashed:
    def test_pinning_the_tasks_changes_the_hash(self):
        assert h(parse(*BASE, "--train-tasks", "Movie")) != h(parse(*BASE))

    def test_movie_only_and_movie_plus_rest_are_different_fits(self):
        one = h(parse(*BASE, "--train-tasks", "Movie"))
        two = h(parse(*BASE, "--train-tasks", "Movie", "Rest"))
        assert one != two

    def test_the_order_they_are_typed_in_does_not_matter(self):
        """Sorted into the payload: the same two conditions are the same fit
        however the command spelled them, or a re-run of the same analysis
        would read as a different model."""
        one = h(parse(*BASE, "--train-tasks", "Rest", "Movie"))
        two = h(parse(*BASE, "--train-tasks", "Movie", "Rest"))
        assert one == two

    def test_case_is_kept_because_the_labels_are_the_data_s_own(self):
        """Unlike subject ids, task labels are NOT normalised -- `--keep-tasks`
        matches the BIDS labels as the data spells them, and a config that
        disagreed with the filesystem about the name of a condition is the bug
        that rule exists to prevent."""
        assert (h(parse(*BASE, "--train-tasks", "Movie"))
                != h(parse(*BASE, "--train-tasks", "movie")))

    def test_the_payload_carries_the_list_itself(self):
        meta = D.fit_meta(parse(*BASE, "--train-tasks", "Rest", "Movie"),
                          "30", EDGES)
        assert meta["train_tasks"] == ["Movie", "Rest"]


class TestTrainSubjectsIsHashedAsADigest:
    def ids(self, tmp_path, *ids, name="keep.txt"):
        p = tmp_path / name
        p.write_text("\n".join(ids) + "\n")
        return str(p)

    def test_the_digest_is_in_the_payload_and_the_ids_are_not(self, tmp_path):
        """The list must not go in. It is hundreds of ids that would dwarf any
        diff, and -- the reason that matters -- a camcan split stratified on
        HADS encodes phenotype in its membership, which is not something to
        write into every output file's metadata."""
        f = self.ids(tmp_path, "CC110033", "CC110045")
        meta = D.fit_meta(parse(*BASE, "--train-subjects", f), "30", EDGES)
        assert meta["train_subjects"] == subject_list_digest(
            {"CC110033", "CC110045"})
        assert "CC110033" not in str(meta)

    def test_restricting_the_fit_changes_the_hash(self, tmp_path):
        f = self.ids(tmp_path, "CC110033")
        assert h(parse(*BASE, "--train-subjects", f)) != h(parse(*BASE))

    def test_a_different_set_is_a_different_fit(self, tmp_path):
        one = self.ids(tmp_path, "CC110033", "CC110045", name="a.txt")
        two = self.ids(tmp_path, "CC110033", "CC110099", name="b.txt")
        assert (h(parse(*BASE, "--train-subjects", one))
                != h(parse(*BASE, "--train-subjects", two)))

    def test_the_same_set_in_another_order_or_case_is_one_fit(self, tmp_path):
        """The digest is of a SET of normalised ids, so re-generating the split
        file -- which a stratified sampler does in whatever order it likes --
        does not invent a new model."""
        one = self.ids(tmp_path, "CC110033", "CC110045", name="a.txt")
        two = self.ids(tmp_path, " cc110045 ", "cc110033", name="b.txt")
        assert (h(parse(*BASE, "--train-subjects", one))
                == h(parse(*BASE, "--train-subjects", two)))

    def test_two_files_with_the_same_ids_are_one_fit(self, tmp_path):
        """The PATH is not hashed, only the ids: moving or renaming the split
        file is not a different analysis."""
        one = self.ids(tmp_path, "CC110033", name="here.txt")
        two = self.ids(tmp_path, "CC110033", name="moved.txt")
        assert (h(parse(*BASE, "--train-subjects", one))
                == h(parse(*BASE, "--train-subjects", two)))

    def test_the_list_is_read_once_per_invocation(self, tmp_path):
        """`fit_meta` runs per aperture. Re-reading would let a file edited
        while the command runs produce two different model_hash values inside
        one run, which is the one thing a hash is for."""
        f = self.ids(tmp_path, "CC110033")
        a = parse(*BASE, "--train-subjects", f)
        first = h(a, "30")
        (tmp_path / "keep.txt").write_text("CC999999\n")
        assert h(a, "30") == first

    def test_a_missing_file_names_this_flag_not_the_other_one(self, tmp_path):
        a = parse(*BASE, "--train-subjects", str(tmp_path / "nope.txt"))
        with pytest.raises(SystemExit, match="--train-subjects"):
            D.train_subjects(a)

    def test_an_empty_list_is_refused_rather_than_emptying_the_fit(self,
                                                                  tmp_path):
        p = tmp_path / "k.txt"
        p.write_text("\n  \n")
        with pytest.raises(SystemExit, match="no ids"):
            D.train_subjects(parse(*BASE, "--train-subjects", str(p)))


class TestTheSharedSubjectListReader:
    """One reader for `--restrict-subjects` and `--train-subjects`. The whole
    point of both is that the stage pinning who was FITTED and the stages
    pinning who was SCORED agree about what a subject list means, and two
    implementations of "read a list of ids" drift on whitespace or case."""

    def test_the_selection_trees_use_the_same_object(self):
        from fmri_decomposition import bstm_selection, fcm_selection

        assert (bstm_selection.read_subject_list
                is fcm_selection.read_subject_list is read_subject_list)

    def test_one_id_per_line(self, tmp_path):
        p = tmp_path / "k.txt"
        p.write_text("CC110033\n cc110045 \n\n")
        assert read_subject_list(str(p)) == {"CC110033", "CC110045"}

    def test_a_csv_with_a_sub_column(self, tmp_path):
        p = tmp_path / "k.csv"
        p.write_text("sub,excluded\nCC110033,False\nCC110045,False\n")
        assert read_subject_list(str(p)) == {"CC110033", "CC110045"}

    def test_a_participants_csv_can_be_handed_straight_in(self, tmp_path):
        """`sub` is not the first column there, and the rows repeat per task."""
        p = tmp_path / "participants.csv"
        p.write_text("participant_id,sub,cohort,task\n"
                     "sub-01,CC110033,camcan,Movie\n"
                     "sub-01,CC110033,camcan,Rest\n"
                     "sub-02,CC110045,camcan,Movie\n")
        assert read_subject_list(str(p)) == {"CC110033", "CC110045"}

    def test_the_flag_in_the_error_is_the_caller_s(self, tmp_path):
        with pytest.raises(SystemExit, match="--train-subjects"):
            read_subject_list(str(tmp_path / "nope.txt"),
                              flag="--train-subjects")

    def test_the_digest_ignores_order_case_and_whitespace(self):
        assert (subject_list_digest(["cc2", " CC1 "])
                == subject_list_digest(["CC1", "CC2"]))

    def test_the_digest_separates_sets_that_differ_by_one(self):
        assert (subject_list_digest(["CC1", "CC2"])
                != subject_list_digest(["CC1", "CC2", "CC3"]))


CELLS = [("ds002837", "movie1"), ("cneuromod", "s01e01a"),
         ("cneuromod", "rest")]


class TestKeepTrainTasks:
    def test_absent_leaves_every_cell(self):
        assert D.keep_train_tasks(CELLS, None, "-1", "yeo7") == CELLS

    def test_it_keeps_only_the_named_conditions(self):
        kept = D.keep_train_tasks(CELLS, ["s01e01a", "movie1"], "-1", "yeo7")
        assert kept == [("ds002837", "movie1"), ("cneuromod", "s01e01a")]

    def test_a_pinned_task_with_no_shards_is_refused(self):
        """NOT dropped quietly. `train_tasks` is in model_hash, so going ahead
        would stamp every output with a fit description naming a condition the
        fit never saw -- and the run would look like it worked."""
        with pytest.raises(SystemExit) as e:
            D.keep_train_tasks(CELLS, ["movie1", "rest2"], "-1", "yeo7")
        assert "rest2" in str(e.value) and "model_hash" in str(e.value)

    def test_the_refusal_lists_what_is_on_disk(self):
        with pytest.raises(SystemExit) as e:
            D.keep_train_tasks(CELLS, ["nope"], "-1", "yeo7")
        msg = str(e.value)
        assert "rest" in msg and "s01e01a" in msg and "movie1" in msg

    def test_matching_nothing_mentions_case_sensitivity(self):
        """The commonest version of this mistake: `--train-tasks movie` for a
        cohort whose shards say `Movie`."""
        with pytest.raises(SystemExit, match="case-sensitive"):
            D.keep_train_tasks([("camcan", "Movie")], ["movie"], "-1", "yeo7")

    def test_it_does_not_invent_cohorts(self):
        """Only the cells it was given, so a task present in one cohort and
        absent from another narrows rather than widens."""
        kept = D.keep_train_tasks(CELLS, ["rest"], "-1", "yeo7")
        assert kept == [("cneuromod", "rest")]


class TestCheckTrainComposition:
    NOW = ["cneuromod/rest", "cneuromod/s01e01a", "ds002837/movie1"]

    def test_an_unchanged_composition_passes(self):
        D.check_train_composition(list(self.NOW), list(self.NOW), "abc", "-1")

    def test_order_does_not_matter(self):
        D.check_train_composition(list(reversed(self.NOW)), list(self.NOW),
                                  "abc", "-1")

    def test_a_fit_saved_before_this_was_recorded_still_loads(self):
        """None means "nothing to compare", not "nothing matches" -- refusing
        every tree written before this key existed would be a migration, and
        this change is deliberately not one."""
        D.check_train_composition(None, list(self.NOW), "abc", "-1")

    def test_a_gained_condition_is_refused(self):
        """The silent case this exists for: rest shards land, the training rows
        change, and `train_cohorts` is still `[ds002837, cneuromod]`."""
        saved = ["cneuromod/s01e01a", "ds002837/movie1"]
        with pytest.raises(SystemExit) as e:
            D.check_train_composition(saved, list(self.NOW), "abc", "-1")
        assert "gained: cneuromod/rest" in str(e.value)

    def test_a_lost_condition_is_refused_too(self):
        with pytest.raises(SystemExit) as e:
            D.check_train_composition(list(self.NOW),
                                      ["ds002837/movie1"], "abc", "-1")
        assert "lost:" in str(e.value)

    def test_the_refusal_offers_both_pins_and_the_overwrite(self):
        """Three ways out, and they are not interchangeable: one keeps the fit
        on disk, one makes a new one beside it, one destroys the old."""
        saved = ["cneuromod/s01e01a", "ds002837/movie1"]
        with pytest.raises(SystemExit) as e:
            D.check_train_composition(saved, list(self.NOW), "abc", "-1")
        msg = str(e.value)
        assert msg.count("--train-tasks") == 2 and "--overwrite" in msg
        # Task labels, sorted, with the cohort stripped -- the pin is about
        # conditions, and the same condition in two cohorts is one name.
        assert "--train-tasks movie1 s01e01a\n" in msg       # what is saved
        assert "--train-tasks movie1 rest s01e01a\n" in msg  # what is here

    def test_it_names_the_hash_and_the_aperture(self):
        with pytest.raises(SystemExit) as e:
            D.check_train_composition(["a/b"], ["a/c"], "deadbeef", "300")
        assert "deadbeef" in str(e.value) and "window_s=300" in str(e.value)


class TestTheRoleIsPerCellAndMatchesItsFooter:
    """`role` used to be per COHORT, and once a cohort can be in --train for
    only some of its conditions that has no single answer -- a projected rest
    file would have been labelled `train`. Nothing branches on the value
    (`status` reports it), so the fix is to make it accurate, not to keep the
    old vocabulary."""

    def models(self, **over):
        m = {"fit_meta": {"atlas": "mini", "n_latents": [3]},
             "model_hash": "abc", "n_train_rows": 480, "umap": {},
             "umap_fitted": False, "role_of": {"camcan": "train"},
             "train_cells": ["camcan/Movie"], "n_train_subjects": 150}
        m.update(over)
        return m

    def footer(self, path):
        import json

        import pyarrow.parquet as pq

        md = pq.ParquetFile(path).schema_arrow.metadata or {}
        return {k.decode(): json.loads(v.decode()) for k, v in md.items()
                if k != b"pandas"}

    def frame(self, role):
        import pandas as pd

        return pd.DataFrame({"cohort": ["camcan"] * 2, "role": [role] * 2,
                             "task": ["Rest"] * 2, "sub": ["CC1", "CC2"],
                             "pca0/3": [0.1, 0.2]})

    def test_the_footer_takes_the_role_from_the_rows(self, tmp_path):
        """Read off the frame rather than from a per-cohort map, so the footer
        and the column it sits beside cannot disagree -- the same rule `tasks`
        already followed."""
        p = tmp_path / "data.parquet"
        D.write_latents(self.frame("projected"), p, self.models(), "camcan")
        assert self.footer(p)["role"] == "projected"

    def test_train_subset_survives_the_round_trip(self, tmp_path):
        """The honest label for a cell whose cohort fed the fit through
        --train-subjects: the file holds both the subjects that were fitted on
        and the ones deliberately held out for the supervised stage."""
        p = tmp_path / "data.parquet"
        D.write_latents(self.frame("train_subset"), p, self.models(), "camcan")
        assert self.footer(p)["role"] == "train_subset"

    def test_the_resolved_composition_is_in_the_footer(self, tmp_path):
        """So a latents file opened on its own says which conditions and how
        many subjects its fit saw -- the two things `train_cohorts` cannot."""
        p = tmp_path / "data.parquet"
        D.write_latents(self.frame("train"), p, self.models(), "camcan")
        f = self.footer(p)
        assert f["train_cells"] == ["camcan/Movie"]
        assert f["n_train_subjects"] == 150

    def test_neither_is_part_of_the_fit_description(self, tmp_path):
        """Both sit BESIDE fit_meta, so they are out of model_hash and out of
        any staleness diff -- otherwise every projected cohort would read as a
        different fit."""
        assert "train_cells" in D._NOT_FIT_KEYS
        assert "n_train_subjects" in D._NOT_FIT_KEYS
        assert D._fit_differences(
            {"train_cells": ["a/b"], "n_train_subjects": 9, "atlas": "mini"},
            {"atlas": "mini"}) == []

    def test_an_empty_frame_falls_back_to_the_cohort_map(self, tmp_path):
        p = tmp_path / "data.parquet"
        D.write_latents(self.frame("train").iloc[:0], p, self.models(),
                        "camcan")
        assert self.footer(p)["role"] == "train"
