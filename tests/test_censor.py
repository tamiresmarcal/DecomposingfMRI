"""Stage 2.5: a named keep/drop decision, written one file per (cohort, task).

Nothing covered this stage before. It is a WRITE/READ PAIR -- `censor.run`
writes the decision and `decompose.load_censor` consumes it -- so a change to
the layout touches both sides and a green suite that exercises neither proves
nothing. The task split was exactly such a change.

The decision itself is per (sub, task) and always was. Splitting the file by
task partitions it rather than altering what it says, which is what these
tests pin: same subjects kept, same reasons, read back identically.
"""
import argparse
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fmri_decomposition import censor as C
from fmri_decomposition.decompose import load_censor

POLICY = {"name": "motion",
          "subject": {"max_mean_fd": 0.5, "min_frac_good_frames": 0.5,
                      "max_frac_parcels_empty": None,
                      "max_abs_best_lag_tr": None,
                      "min_frac_stimulus_covered": None},
          "window": {"min_frac_good_frames": None, "min_n_tr_effective": None,
                     "drop_rank_deficient": False,
                     "drop_crosses_run_boundary": False,
                     "drop_crosses_clip_boundary": False}}

#: Two conditions of ONE cohort -- the Cam-CAN case. S2 moved during the movie
#: and S3 during rest, so each condition drops a different subject: the thing a
#: per-cohort file could only express by holding both tasks' rows at once.
QC = pd.DataFrame([
    {"sub": "S1", "task": "Movie", "mean_fd": 0.10, "frac_good_frames": 0.99},
    {"sub": "S2", "task": "Movie", "mean_fd": 0.90, "frac_good_frames": 0.99},
    {"sub": "S3", "task": "Movie", "mean_fd": 0.12, "frac_good_frames": 0.98},
    {"sub": "S1", "task": "Rest", "mean_fd": 0.11, "frac_good_frames": 0.97},
    {"sub": "S2", "task": "Rest", "mean_fd": 0.13, "frac_good_frames": 0.96},
    {"sub": "S3", "task": "Rest", "mean_fd": 0.14, "frac_good_frames": 0.20},
])


@pytest.fixture
def censored(tmp_path):
    """One `censor` run over a two-condition cohort."""
    qc_dir = tmp_path / "meta" / "cohorts" / "cohort=camcan"
    qc_dir.mkdir(parents=True)
    QC.to_csv(qc_dir / "participants_qc.csv", index=False)
    pol = tmp_path / "motion.yaml"
    pol.write_text(yaml.safe_dump(POLICY))
    rc = C.run(argparse.Namespace(
        policy=str(pol), cohorts=["camcan"], output_root=str(tmp_path),
        stage="subjects", atlas=None, window_s=None))
    assert rc == 0
    return tmp_path


class TestItWritesOneFilePerTask:
    def test_each_condition_gets_its_own_file(self, censored):
        base = censored / "censor" / "policy=motion" / "cohort=camcan"
        assert sorted(p.name for p in base.iterdir()) == ["task=Movie",
                                                          "task=Rest"]
        for t in ("Movie", "Rest"):
            assert (base / f"task={t}" / "subjects.parquet").exists()

    def test_each_file_holds_only_its_own_task(self, censored):
        base = censored / "censor" / "policy=motion" / "cohort=camcan"
        for t in ("Movie", "Rest"):
            d = pd.read_parquet(base / f"task={t}" / "subjects.parquet")
            assert set(d["task"]) == {t}
            assert sorted(d["sub"]) == ["S1", "S2", "S3"]

    def test_the_decision_is_unchanged_by_the_split(self, censored):
        """Partitioning the table must not alter what it says: S2 fails the
        motion gate under Movie, S3 fails the scrubbing gate under Rest, and
        nobody else is dropped."""
        base = censored / "censor" / "policy=motion" / "cohort=camcan"
        got = {}
        for t in ("Movie", "Rest"):
            d = pd.read_parquet(base / f"task={t}" / "subjects.parquet")
            got[t] = set(d.loc[~d["keep"], "sub"])
        assert got == {"Movie": {"S2"}, "Rest": {"S3"}}


class TestDecomposeReadsItBack:
    def test_the_union_across_tasks_is_what_the_fit_sees(self, censored):
        """THE PAIR. load_censor keys on (task, sub), so reading the parts and
        unioning them is the same answer one file gave -- and the gate is
        per condition, so S2 is out of Movie while still in Rest."""
        cen = load_censor(censored, "motion", "yeo7", -1, "camcan")
        assert cen["subjects"] == {("Movie", "S1"), ("Movie", "S3"),
                                   ("Rest", "S1"), ("Rest", "S2")}
        assert cen["n_subject_rows"] == 6
        assert cen["n_subject_kept"] == 4

    def test_no_window_gate_reads_as_subjects_only(self, censored):
        assert load_censor(censored, "motion", "yeo7", -1,
                           "camcan")["windows"] is None

    def test_no_policy_means_no_censoring(self, censored):
        assert load_censor(censored, None, "yeo7", -1, "camcan") is None

    def test_a_policy_that_was_asked_for_and_is_absent_REFUSES(self, censored):
        """Never silently no censoring: the run would look identical and be a
        different analysis."""
        with pytest.raises(SystemExit) as e:
            load_censor(censored, "strict", "yeo7", -1, "camcan")
        assert "strict" in str(e.value)
        assert "fmri-decomp censor" in str(e.value)

    def test_a_cohort_with_no_decision_REFUSES_too(self, censored):
        with pytest.raises(SystemExit, match="motion"):
            load_censor(censored, "motion", "yeo7", -1, "nope")


class TestOneConditionCanBeReGatedAlone:
    def test_re_running_one_task_leaves_the_other_file_alone(self, censored):
        """What the split buys beyond uniformity: the conditions are
        independent on disk, so re-gating one cannot touch the other."""
        base = censored / "censor" / "policy=motion" / "cohort=camcan"
        rest = base / "task=Rest" / "subjects.parquet"
        before = rest.stat().st_mtime_ns
        d = pd.read_parquet(base / "task=Movie" / "subjects.parquet")
        C._write(d, base / "task=Movie" / "subjects.parquet", POLICY, "h")
        assert rest.stat().st_mtime_ns == before
