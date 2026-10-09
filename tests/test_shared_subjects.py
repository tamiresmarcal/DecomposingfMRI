"""One id list for every selection run, so a gap between their tables cannot
be a difference in who was scored.

The task dimension is the part worth testing. Cam-CAN's movie and rest are ONE
cohort with two tasks, so a per-cohort intersection reads a table containing
both conditions and returns the subjects with EITHER -- a union dressed up as
an intersection, which is the opposite of the point.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import shared_subjects as S        # noqa: E402

#: A,B did the movie; B,C did rest. So the shared sample is {B} alone.
ROWS = [("A", "Movie"), ("B", "Movie"), ("B", "Rest"), ("C", "Rest")]


@pytest.fixture
def tree(tmp_path):
    for stage, extra in (
            ("transitions", "atlas=yeo7/window_s=-1/states=HMM2_pca3_8"),
            ("static_fc", "atlas=yeo7")):
        p = tmp_path / stage / extra / "cohort=camcan" / "subjects.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sub": [r[0] for r in ROWS],
                      "task": [r[1] for r in ROWS]}).to_parquet(p, index=False)
    return tmp_path


def run(tree, *extra):
    out = tree / "shared.txt"
    rc = S.main(["--cohorts", "camcan", "--output-root", str(tree),
                 "-o", str(out), *extra])
    return rc, sorted(out.read_text().split())


class TestTheTaskDimension:
    def test_intersecting_per_task_finds_the_shared_sample(self, tree):
        """THE POINT. B is the only subject with both conditions."""
        rc, got = run(tree, "--tasks", "Movie", "Rest")
        assert rc == 0
        assert got == ["B"]

    def test_without_tasks_it_is_a_union_not_an_intersection(self, tree):
        """Why --tasks had to exist. One cohort holding both conditions gives
        every subject with EITHER, which would be passed to every run as a
        'shared' list and silently widen all four samples."""
        rc, got = run(tree)
        assert got == ["A", "B", "C"]

    def test_one_task_is_just_that_task(self, tree):
        rc, got = run(tree, "--tasks", "Movie")
        assert got == ["A", "B"]

    def test_a_task_matching_nothing_is_refused_with_what_exists(self, tree):
        """Otherwise it intersects to the empty set and reads as 'no subject
        is in every tree', which points at ids rather than at the typo."""
        with pytest.raises(SystemExit) as e:
            run(tree, "--tasks", "Nope")
        msg = str(e.value)
        assert "Nope" in msg and "Movie" in msg and "Rest" in msg


class TestItRefusesRatherThanSkipping:
    def test_an_unbuilt_stage_is_refused(self, tmp_path):
        """A list that silently skips a source is worse than no list: every
        run would then still be scored on its own sample."""
        with pytest.raises(SystemExit, match="nothing matched"):
            S.main(["--cohorts", "camcan", "--output-root", str(tmp_path),
                    "-o", str(tmp_path / "s.txt")])

    def test_cohorts_is_required_with_no_default(self, tree):
        """No cohort name baked into the tool -- same rule as the selection
        stages, since a default only works for one project."""
        with pytest.raises(SystemExit):
            S.main(["--output-root", str(tree), "-o", str(tree / "s.txt")])

    def test_ids_are_upper_cased_and_stripped_like_every_other_join(self,
                                                                   tmp_path):
        p = (tmp_path / "transitions/atlas=yeo7/window_s=-1/states=s"
             / "cohort=c" / "subjects.parquet")
        p.parent.mkdir(parents=True)
        pd.DataFrame({"sub": [" cc1 ", "CC2"],
                      "task": ["m", "m"]}).to_parquet(p, index=False)
        q = tmp_path / "static_fc/atlas=yeo7/cohort=c/subjects.parquet"
        q.parent.mkdir(parents=True)
        pd.DataFrame({"sub": ["CC1", "cc2"],
                      "task": ["m", "m"]}).to_parquet(q, index=False)
        out = tmp_path / "s.txt"
        S.main(["--cohorts", "c", "--output-root", str(tmp_path),
                "-o", str(out)])
        assert sorted(out.read_text().split()) == ["CC1", "CC2"]
