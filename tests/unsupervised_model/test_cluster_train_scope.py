"""Stage 4b's half of "which conditions and which subjects define the states".

A state definition is `decompose` (the embedding) plus `cluster` (the
clusterer), and until now only the first could be asked for a subset. That left
an experiment that pins the tasks at stage 4 and not at 4b producing a
movie-only embedding under a clusterer that also saw rest -- a state set that is
a claim about neither.

The cache keys here already carried `n_train_rows`, so a changed training
composition already changed them and refitted. These add what the row count
cannot say: WHICH conditions, and which subjects.
"""
from __future__ import annotations

import argparse

import pandas as pd
import pytest

from fmri_decomposition.unsupervised_model import cluster as C

OPTS = {"n_jobs": 4, "seed": 0}
TRAIN = ["ds002837", "cneuromod"]


def parse(*argv):
    p = argparse.ArgumentParser()
    C.add_arguments(p)
    return p.parse_args(list(argv))


class TestTheFlagsExistAndDefaultToEverything:
    def test_both_default_to_none(self):
        a = parse()
        assert a.train_tasks is None and a.train_subjects is None

    def test_no_list_means_no_restriction(self):
        assert C.train_subjects_of(parse()) is None

    def test_an_args_without_the_attribute_is_tolerated(self):
        assert C.train_subjects_of(argparse.Namespace()) is None


class TestTheCacheKeysDoNotMoveWhenNothingIsPinned:
    """Same rule `decompose.fit_meta` follows: an absent key means "every task"
    / "every subject". A clusterer cache on disk was filed under a payload
    without these, so adding them must leave that payload alone -- otherwise
    every cached fit silently refits."""

    def test_the_scope_block_is_empty_when_nothing_is_given(self):
        assert C._train_scope(None, None) == {}

    def test_an_empty_task_list_is_also_no_restriction(self):
        assert C._train_scope([], None) == {}

    def test_recipe_hash_is_unchanged_without_the_flags(self):
        a = C.recipe_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000)
        b = C.recipe_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                          train_tasks=None, train_subjects=None)
        assert a == b

    def test_fit_hash_is_unchanged_without_the_flags(self):
        a = C.fit_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000)
        b = C.fit_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                       train_tasks=None, train_subjects=None)
        assert a == b


class TestPinningMovesBothKeys:
    def test_recipe_hash_separates_two_task_sets(self):
        one = C.recipe_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                            train_tasks=["Movie"])
        two = C.recipe_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                            train_tasks=["Movie", "Rest"])
        assert one != two

    def test_two_task_sets_with_the_same_row_count_are_still_separated(self):
        """`n_train_rows` was the only thing standing between them, and two
        different conditions can coincide on a row count."""
        assert (C.recipe_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                              train_tasks=["Movie"])
                != C.recipe_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                                 train_tasks=["Rest"]))

    def test_the_order_typed_does_not_matter(self):
        assert (C.recipe_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                              train_tasks=["Rest", "Movie"])
                == C.recipe_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                                 train_tasks=["Movie", "Rest"]))

    def test_the_subject_digest_separates_fits(self):
        assert (C.fit_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                           train_subjects="aaaa")
                != C.fit_hash("hmm1", "pca3", 8, OPTS, TRAIN, 1000,
                              train_subjects="bbbb"))

    def test_the_scope_carries_only_what_was_given(self):
        assert C._train_scope(["Movie"], None) == {"train_tasks": ["Movie"]}
        assert C._train_scope(None, "abc") == {"train_subjects": "abc"}


class TestTheTaskFilterIsTheSameOneDecomposeUses:
    """Imported rather than re-implemented: the two stages must say "these
    conditions" the same way, with the same refusals, or the pin means one
    thing at stage 4 and another at 4b."""

    CELLS = [("camcan", "Movie"), ("camcan", "Rest"), ("ds002837", "movie1")]

    def test_it_is_decompose_s_function(self):
        from fmri_decomposition.unsupervised_model.decompose import keep_train_tasks

        assert keep_train_tasks is not None
        assert keep_train_tasks(self.CELLS, ["Movie"], "-1", "yeo7") == [
            ("camcan", "Movie")]

    def test_a_pinned_condition_with_no_latents_is_refused(self):
        from fmri_decomposition.unsupervised_model.decompose import keep_train_tasks

        with pytest.raises(SystemExit, match="Rest2"):
            keep_train_tasks(self.CELLS, ["Movie", "Rest2"], "-1", "yeo7",
                             source="latents")


class TestTheSubjectFilterKeepsWholeSubjects:
    """By whole subjects, like --balance-train and for the same reason: an HMM
    is handed `lengths`, and dropping rows from the middle of someone's
    sequence would make that assertion false."""

    def frames(self):
        def f(task, subs):
            return pd.DataFrame({"task": [task] * len(subs), "sub": list(subs),
                                 "pca0/3": [0.1] * len(subs),
                                 "window_id": range(len(subs))})
        return {("camcan", "Movie"): f("Movie", ["CC1", "CC1", "CC2", "CC3"])}

    def args(self, path=None, balance=False):
        a = parse()
        a.train_subjects = path
        a.balance_train = balance
        return a

    def test_without_a_list_every_row_is_used(self):
        X, lengths, share = C._training_block(
            self.frames(), [("camcan", "Movie")], ["pca0/3"], self.args())
        assert len(X) == 4 and sum(lengths) == 4

    def test_only_the_named_subjects_reach_the_fit(self, tmp_path):
        p = tmp_path / "k.txt"
        p.write_text("CC1\nCC3\n")
        X, lengths, _ = C._training_block(
            self.frames(), [("camcan", "Movie")], ["pca0/3"],
            self.args(str(p)))
        assert len(X) == 3                 # CC1 twice + CC3
        assert sorted(lengths) == [1, 2]   # and the sequences stay whole

    def test_ids_are_matched_upper_cased_and_stripped(self, tmp_path):
        p = tmp_path / "k.txt"
        p.write_text(" cc2 \n")
        X, _, _ = C._training_block(
            self.frames(), [("camcan", "Movie")], ["pca0/3"],
            self.args(str(p)))
        assert len(X) == 1

    def test_matching_nobody_is_refused_rather_than_fitting_on_nothing(
            self, tmp_path):
        p = tmp_path / "k.txt"
        p.write_text("NOBODY\n")
        with pytest.raises(SystemExit, match="matched no subject"):
            C._training_block(self.frames(), [("camcan", "Movie")],
                              ["pca0/3"], self.args(str(p)))

    def test_a_cell_with_no_matching_subject_still_has_a_share(self,
                                                               tmp_path):
        """`run_one` logs `share[k]` for every cell it asked to train on, so a
        cell the split does not reach must stay in the mapping at 0 rather than
        take the log line down with a KeyError."""
        p = tmp_path / "k.txt"
        p.write_text("CC1\n")
        frames = {
            ("camcan", "Movie"): pd.DataFrame(
                {"task": ["Movie"] * 2, "sub": ["CC1", "CC2"],
                 "pca0/3": [0.1, 0.2], "window_id": [0, 1]}),
            ("camcan", "Rest"): pd.DataFrame(
                {"task": ["Rest"] * 2, "sub": ["CC9", "CC8"],
                 "pca0/3": [0.1, 0.2], "window_id": [0, 1]}),
        }
        cells = [("camcan", "Movie"), ("camcan", "Rest")]
        X, _, share = C._training_block(frames, cells, ["pca0/3"],
                                        self.args(str(p)))
        assert len(X) == 1
        assert set(share) == set(cells)
        assert share[("camcan", "Rest")] == 0

    def test_the_caller_s_frames_are_not_mutated(self, tmp_path):
        """`frames` is also what every cohort is LABELLED from -- filtering it
        in place would quietly restrict the labelling to the training half."""
        p = tmp_path / "k.txt"
        p.write_text("CC1\n")
        frames = self.frames()
        C._training_block(frames, [("camcan", "Movie")], ["pca0/3"],
                          self.args(str(p)))
        assert len(frames[("camcan", "Movie")]) == 4


class TestTheEmbeddingAndTheClustererMustAgree:
    PATH = "latents/.../data.parquet"

    def test_agreement_is_silent(self):
        assert C.check_against_embedding(
            ["camcan/Movie"], None, ["camcan/Movie"], None, self.PATH) == []

    def test_a_tree_written_before_this_was_recorded_is_silent(self):
        assert C.check_against_embedding(
            None, None, ["camcan/Movie"], None, self.PATH) == []

    def test_different_conditions_warn_and_name_both_sides(self):
        """A judgement call, not a mistake -- pooling the embedding widely and
        fitting the clusterer narrowly is coherent if meant."""
        w = C.check_against_embedding(
            ["camcan/Movie", "camcan/Rest"], None, ["camcan/Movie"], None,
            self.PATH)
        assert len(w) == 1
        assert "camcan/Rest" in w[0] and "--train-tasks" in w[0]

    def test_a_held_out_embedding_with_an_unrestricted_clusterer_is_refused(
            self):
        """THE leakage case. The embedding never saw the test half; a clusterer
        fitted on everyone puts them straight back into the definition, and the
        split is gone while every file still claims to have one."""
        with pytest.raises(SystemExit) as e:
            C.check_against_embedding(["camcan/Movie"], "abc123",
                                      ["camcan/Movie"], None, self.PATH)
        assert "held subjects out" in str(e.value)

    def test_the_reverse_is_refused_too(self):
        with pytest.raises(SystemExit) as e:
            C.check_against_embedding(["camcan/Movie"], None,
                                      ["camcan/Movie"], "abc123", self.PATH)
        assert "the embedding was not" in str(e.value)

    def test_two_different_lists_are_refused_and_said_to_be_different(self):
        with pytest.raises(SystemExit) as e:
            C.check_against_embedding(["camcan/Movie"], "aaa",
                                      ["camcan/Movie"], "bbb", self.PATH)
        assert "DIFFERENT subject lists" in str(e.value)

    def test_the_refusal_names_the_file_it_read(self):
        with pytest.raises(SystemExit, match="data.parquet"):
            C.check_against_embedding(["c/t"], "aaa", ["c/t"], None, self.PATH)

    def test_the_same_digest_passes_even_when_restricted(self):
        assert C.check_against_embedding(
            ["camcan/Movie"], "abc", ["camcan/Movie"], "abc", self.PATH) == []
