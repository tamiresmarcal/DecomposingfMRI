"""The shipped Cam-CAN config: one cohort, two conditions.

Movie and rest are THE SAME PEOPLE. Two cohort values for one group made the
output tree claim they were different samples, so an `n` that differed between
the two selection tables could be read as a difference in who was scanned
rather than in who completed which scan.

These tests read `config/camcan.yaml` itself rather than a fixture, because
what is being pinned is the project's own claim about its data -- a fixture
would pass while the real config said something else.
"""
from pathlib import Path

import pytest

from fmri_decomposition.config import ConfigError, load_config

CONFIG = Path(__file__).resolve().parent.parent / "config" / "camcan.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


class TestOneCohortTwoConditions:
    def test_both_conditions_are_one_cohort(self, cfg):
        assert cfg.cohort == "camcan"
        assert cfg.discovery.include_tasks == ["Movie", "Rest"]

    def test_the_retired_configs_are_gone(self):
        """Not merely unused -- removed. Two files for one cohort is the thing
        this change exists to end, and one left behind would be run by
        somebody."""
        for name in ("camcan_movie.yaml", "camcan_rest.yaml"):
            assert not (CONFIG.parent / name).exists(), name

    def test_one_derivatives_root_for_both(self, cfg):
        """fMRIPrep ran twice, into two trees. A cohort reads one root, so the
        configured one is a merged symlink view -- see RUNBOOK 4.1."""
        assert "camcan_all" in cfg.derivatives_root.name

    def test_the_globs_are_task_wildcarded(self, cfg):
        """Which is why the merged view needed no code change: one directory
        per subject holding both tasks' files satisfies them as written."""
        assert "task-*" in cfg.discovery.bold_glob
        assert "task-*" in cfg.confounds.confounds_glob


class TestTheTwoTRs:
    def test_each_condition_keeps_its_own_tr(self, cfg):
        assert cfg.tr_for("Movie") == 2.47
        assert cfg.tr_for("Rest") == 1.97

    def test_a_third_task_would_not_silently_take_the_default(self, cfg):
        """Cam-CAN also ships a sensorimotor task at 1.97s. It is excluded by
        include_tasks, which is why that list stays a correctness requirement
        and not a convenience."""
        assert "Sensorimotor" not in cfg.discovery.include_tasks
        assert set(cfg.tr_by_task) <= set(cfg.discovery.include_tasks)

    def test_the_window_sample_count_differs_as_documented(self, cfg):
        """30s is 12 samples of movie and 15 of rest, so at the same nominal
        aperture the REST edges are less noisy. The config says this; here it
        is computed."""
        assert cfg.window_tr(30, "Movie") == 12
        assert cfg.window_tr(30, "Rest") == 15

    def test_the_rank_floor_differs_as_documented(self, cfg):
        """The table in the config's `windows` comment, recomputed."""
        from fmri_decomposition.windows import min_window_s_for_nodes

        for nodes, movie_s, rest_s in ((7, 18.525, 14.775),
                                       (14, 35.815, 28.565),
                                       (111, 275.405, 219.655)):
            assert min_window_s_for_nodes(
                nodes, cfg.tr_for("Movie")) == pytest.approx(movie_s, abs=0.1)
            assert min_window_s_for_nodes(
                nodes, cfg.tr_for("Rest")) == pytest.approx(rest_s, abs=0.1)

    def test_fifteen_seconds_is_singular_for_movie_yeo7_but_not_rest(self, cfg):
        """The asymmetry the window plan has to report per task rather than
        average away: the same configured aperture is rank-deficient for one
        condition and estimable for the other."""
        from fmri_decomposition.windows import is_rank_deficient

        assert is_rank_deficient(cfg.window_tr(15, "Movie"), 7)
        assert not is_rank_deficient(cfg.window_tr(15, "Rest"), 7)


class TestTheGateAndTheStimulus:
    def test_movie_is_gated_and_rest_is_not(self, cfg):
        assert cfg.stimulus.isc_gate_for("Movie") == 1.0
        assert cfg.stimulus.isc_gate_for("Rest") is None

    def test_rest_has_no_configured_duration_and_movie_does(self, cfg):
        """Rest falls back to each subject's observed run length, which is
        honest until the volume count is confirmed. Movie is pinned to a
        193-volume acquisition verified across all 648 subjects."""
        assert cfg.stimulus.durations_s == {"Movie": 476.71}
        assert cfg.stimulus_duration_s("Movie") == 476.71
        with pytest.raises(ConfigError, match="no stimulus duration"):
            cfg.stimulus_duration_s("Rest")
        assert cfg.stimulus_duration_s("Rest", fallback=500.0) == 500.0


class TestTheSharedQCRule:
    def test_one_fd_threshold_now_covers_both_conditions(self, cfg):
        """The benefit beyond tidiness. It used to be duplicated in two files
        with a comment in each asking a human to keep them in sync, and a
        different threshold per condition removes a different amount of data
        from the same people -- so the contrast would partly measure the QC
        rule rather than the brain."""
        assert cfg.confounds.fd_threshold == 0.5
        assert cfg.confounds.dilate_tr == 1

    def test_one_band_now_covers_both_conditions(self, cfg):
        assert cfg.filtering.bandpass == [0.01, 0.1]
        assert cfg.filtering.already_applied is False


class TestTheOutputTree:
    def test_it_shares_the_one_output_root(self, cfg):
        """`cohort=` separates cohorts inside one tree and `task=` separates
        these two conditions inside this one, which is what makes "one atlas,
        one window size, every cohort" a single readable path."""
        from fmri_decomposition.io import default_output_root

        assert cfg.output_root == default_output_root()

    def test_no_library_module_builds_a_path_to_a_cohort_config(self):
        """The default output root used to be three lines repeated in eight
        modules, each building `config / "camcan_movie.yaml"`. Renaming the
        config broke every stage, and a user without Cam-CAN got a
        FileNotFoundError for a dataset they do not have. It is one helper now.

        Mentioning a cohort in a COMMENT is fine and common -- the modules
        explain cohort-specific facts all over. What must not come back is a
        module constructing a path into config/.
        """
        import re

        lib = Path(__file__).resolve().parent.parent / "fmri_decomposition"
        pattern = re.compile(r'"config"\s*/\s*f?"')
        offenders = sorted(f.name for f in lib.glob("*.py")
                           if pattern.search(f.read_text()))
        assert offenders == [], offenders
