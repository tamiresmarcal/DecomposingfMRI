"""The transfer pair as a CONFIG FILE, not a Python module.

WHY THIS FILE EXISTS. The pair used to be module constants -- TRAIN_COHORTS,
CC_TARGETS, HCP_TASKS -- which worked perfectly for the one pair that existed
and encoded "Cam-CAN trains, HCP is tested" into code that notebook 13 and
every transfer job import. The second pair is the test of that design, and
writing it found two bugs immediately: `pool_runs` and `target_range` were
each wired on only the side v0 happened to need them. Both are pinned below.

What is pinned here is the RULE: anything that differs between two pairs is a
setting, and a setting that only works on one side is not a setting.
"""
import os
from pathlib import Path

import pytest

from fmri_decomposition.supervised_model import transfer_config as tc
from fmri_decomposition.supervised_model.transfer import pair


@pytest.fixture(autouse=True)
def _pheno(monkeypatch):
    """The shipped configs point at $HCP_PHENO; these tests never read it."""
    monkeypatch.setenv("HCP_PHENO", "/nonexistent/hcp_pheno.csv")


class TestFindingAConfig:
    def test_a_bare_name_resolves_under_config_transfer(self):
        assert tc.config_path("camcan_to_hcp").name == "camcan_to_hcp.yaml"

    def test_a_path_is_taken_as_given(self, tmp_path):
        f = tmp_path / "mine.yaml"
        f.write_text("train: {cohorts: [a]}\ntest: {cohorts: [b]}\n")
        assert tc.config_path(f) == f

    def test_an_unknown_name_lists_what_there_is(self):
        with pytest.raises(SystemExit) as e:
            tc.config_path("no_such_pair")
        msg = str(e.value)
        assert "no_such_pair" in msg and "camcan_to_hcp" in msg


class TestWhatItRefuses:
    """A key nothing reads looks like a setting that is in force."""

    def _write(self, tmp_path, body):
        f = tmp_path / "c.yaml"
        f.write_text(body)
        return f

    def test_a_missing_side_is_named(self, tmp_path):
        f = self._write(tmp_path, "train:\n  cohorts: [a]\n")
        with pytest.raises(SystemExit) as e:
            tc.load(f)
        assert "`test:`" in str(e.value)

    def test_an_unknown_top_level_key_is_refused(self, tmp_path):
        f = self._write(tmp_path, "train: {cohorts: [a]}\ntest: {cohorts: [b]}\n"
                                  "covarites: [Age]\n")
        with pytest.raises(SystemExit) as e:
            tc.load(f)
        assert "covarites" in str(e.value) and "covariates" in str(e.value)

    def test_an_unknown_per_side_key_is_refused(self, tmp_path):
        f = self._write(tmp_path, "train: {cohorts: [a], task_mach: exact}\n"
                                  "test: {cohorts: [b]}\n")
        with pytest.raises(SystemExit) as e:
            tc.load(f)
        assert "task_mach" in str(e.value) and "train" in str(e.value)


class TestAnUnsetVariable:
    """`${VAR}` is how a config carries no scratch path. An unset one is left
    in place and refused WHERE IT IS USED, which is neither of the two obvious
    choices and the only one that works."""

    def test_it_loads_so_a_caller_can_override_the_path(self, monkeypatch):
        monkeypatch.delenv("HCP_PHENO", raising=False)
        cfg = tc.load("camcan_to_hcp")
        assert "${HCP_PHENO}" in str(cfg["test"]["pheno_files"])

    def test_an_override_never_needs_the_variable(self, monkeypatch):
        monkeypatch.delenv("HCP_PHENO", raising=False)
        _, test = tc.build("raw", root="/o", hcp_pheno="/given/p.csv")
        assert test.pheno_files == [(Path("/given/p.csv"), ",")]

    def test_without_an_override_it_says_which_variable(self, monkeypatch):
        monkeypatch.delenv("HCP_PHENO", raising=False)
        with pytest.raises(SystemExit) as e:
            tc.build("raw", root="/o")
        msg = str(e.value)
        assert "HCP_PHENO" in msg and "--hcp-pheno" in msg

    def test_it_is_never_silently_emptied(self, monkeypatch):
        """`/participants_pheno.csv` fails much later, somewhere with no idea
        a variable was missing."""
        monkeypatch.delenv("HCP_PHENO", raising=False)
        cfg = tc.load("camcan_to_hcp")
        assert not str(cfg["test"]["pheno_files"]).startswith("[{'path': '/,")


class TestTheTargetSets:
    def test_both_forms_are_offered_and_pick_different_columns(self):
        cfg = tc.load("camcan_to_hcp")
        assert tc.target_names(cfg) == ["category", "raw"]
        raw = tc.targets_for(cfg, "raw")
        cat = tc.targets_for(cfg, "category")
        assert raw["anxiety"] == "additional_HADS_anxiety"
        assert cat["anxiety"] == "additional_HADS_anx_category"
        assert set(raw) == set(cat)          # same LABELS, different columns

    def test_a_config_with_one_set_does_not_ask(self):
        """Its `--target` is answered by the config, not by the caller."""
        cfg = tc.load("hcp_to_camcan")
        assert tc.target_names(cfg) == ["hcp"]

    def test_an_unknown_target_lists_the_sets(self):
        cfg = tc.load("camcan_to_hcp")
        with pytest.raises(SystemExit) as e:
            tc.targets_for(cfg, "rwa")
        assert "rwa" in str(e.value) and "raw" in str(e.value)


class TestTheTwoBugsTheSecondConfigFound:
    """Each of these worked on exactly the side v0 needed it, and nowhere
    else. A setting that works on one side is not a setting."""

    def test_pooling_is_per_side(self):
        """v0 pools the TEST side (HCP has four runs per condition). Swap the
        roles and HCP is the training cohort: without per-side pooling its
        subjects appear four times each and load_set refuses."""
        train, test = tc.build("raw", config="camcan_to_hcp", root="/o")
        assert train.pool_fn is None and test.pool_fn is not None
        train, test = tc.build("hcp", config="hcp_to_camcan", root="/o")
        assert train.pool_fn is not None and test.pool_fn is None

    def test_target_range_is_per_side(self):
        """v0 needs the 9999 sentinel guard on the TRAIN side, because that is
        where the raw HADS columns are. Swapped, it is on the TEST side -- and
        an unfiltered sentinel there is scored as a real value by Spearman."""
        train, test = tc.build("raw", config="camcan_to_hcp", root="/o")
        assert train.target_range == {"anxiety": (0, 21),
                                      "depression": (0, 21)}
        assert test.target_range is None
        train, test = tc.build("hcp", config="hcp_to_camcan", root="/o")
        assert train.target_range is None
        assert test.target_range == {"anxiety": (0, 21),
                                     "depression": (0, 21)}

    def test_the_range_follows_the_target_set(self):
        """`raw: [0, 21]` means the raw columns have a scale; the category
        ones are text and never coded, so they get none."""
        train, _ = tc.build("category", config="camcan_to_hcp", root="/o")
        assert train.target_range is None


class TestTheOutputPathNamesTheRealPair:
    """`train=` and `test=` are a CLAIM about which cohort was fitted. They
    were hardcoded to camcan / hcp7t_fix, which would have filed a role swap's
    rows under the old pair's keys, indistinguishable from it."""

    def test_v0(self):
        assert pair(tc.load("camcan_to_hcp")) == ("camcan", "hcp7t_fix")

    def test_the_swap_lands_somewhere_else(self):
        assert pair(tc.load("hcp_to_camcan")) == ("hcp7t_fix", "camcan")

    def test_several_cohorts_on_a_side_still_name_them_all(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text("train: {cohorts: [a, b]}\ntest: {cohorts: [c]}\n")
        assert pair(tc.load(f)) == ("a+b", "c")


class TestTheShippedConfigsAreCoherent:
    @pytest.mark.parametrize("name", ["camcan_to_hcp", "hcp_to_camcan"])
    def test_the_two_sides_agree_on_their_condition_labels(self, name):
        """The KEYS of `tasks` are what a transfer joins on; the values are
        each cohort's own spelling. A key only one side has produces no row for
        that condition, silently."""
        cfg = tc.load(name)
        assert set(cfg["train"]["tasks"]) == set(cfg["test"]["tasks"])

    @pytest.mark.parametrize("name", ["camcan_to_hcp", "hcp_to_camcan"])
    def test_the_two_sides_agree_on_their_target_labels(self, name):
        """The LABELS must match; the COLUMNS need not, because Spearman only
        needs the two to order people the same way -- which is why this
        transfer needs no equating of HADS against the ASR."""
        cfg = tc.load(name)
        for t in tc.target_names(cfg):
            assert set(tc.targets_for(cfg, t)) == set(cfg["test"]["targets"]), t

    @pytest.mark.parametrize("name", ["camcan_to_hcp", "hcp_to_camcan"])
    def test_a_prefix_side_also_pools(self, name):
        """Prefix matching says several hive partitions are ONE condition.
        Matching them without pooling gives several rows per subject, which
        the models will not average -- so the two settings travel together."""
        cfg = tc.load(name)
        for side in ("train", "test"):
            if cfg[side].get("task_match") == "prefix":
                assert cfg[side].get("pool_runs"), f"{name}/{side}"

    @pytest.mark.parametrize("name", ["camcan_to_hcp", "hcp_to_camcan"])
    def test_camcan_is_never_two_cohorts(self, name):
        """One cohort, two conditions. `camcan_rest` is not a cohort -- it
        matched nothing for a long time and then cost two cluster jobs."""
        cfg = tc.load(name)
        for side in ("train", "test"):
            assert "camcan_rest" not in cfg[side]["cohorts"], f"{name}/{side}"
