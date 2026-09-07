"""Guards on the submit path.

slurm/submit_all.sh runs `fmri-decomp validate` and nothing else before it
burns core-hours, so anything that must not reach a compute node has to fail
here. tools/check_cohort.py is more thorough but nobody's submit script calls
it.
"""
from pathlib import Path

import pytest

from fmri_decomposition.activation import RunRef
from fmri_decomposition.cli import _path_collisions
from fmri_decomposition.config import config_from_dict


@pytest.fixture
def cfg(tmp_path):
    return config_from_dict({
        "cohort": "c", "tr": 1.0,
        "derivatives_root": str(tmp_path), "output_root": str(tmp_path / "out"),
        "atlases": ["yeo7"],
        "windows": {"sizes_s": [30.0], "n_overlaps": 5},
        "stimulus": {"durations_s": {"m": 200.0}},
    })


def ref(sub, task, ses=None, run=None, name=None):
    return RunRef(cohort="c", sub=sub, task=task,
                  bold=Path(name or f"sub-{sub}_task-{task}_bold.nii.gz"),
                  ses=ses, run=run)


class TestPathCollisions:
    def test_distinct_sub_task_pairs_do_not_collide(self, cfg):
        refs = [ref("01", "m"), ref("02", "m"), ref("01", "n")]
        assert _path_collisions(cfg, refs) == []

    def test_same_sub_task_in_two_sessions_collides(self, cfg):
        """The cost of leaf_filename being a constant `data.parquet`.

        These two used to land on ses-001.parquet and ses-002.parquet. Now
        they land on the same leaf and the second silently overwrites the
        first -- which is exactly what must not reach a compute node.
        """
        refs = [ref("01", "m", ses="001", name="a.nii.gz"),
                ref("01", "m", ses="002", name="b.nii.gz")]
        problems = _path_collisions(cfg, refs)
        assert len(problems) == 1
        assert "a.nii.gz" in problems[0] and "b.nii.gz" in problems[0]

    def test_same_sub_task_in_two_runs_collides(self, cfg):
        refs = [ref("01", "m", run="1", name="r1.nii.gz"),
                ref("01", "m", run="2", name="r2.nii.gz")]
        assert len(_path_collisions(cfg, refs)) == 1

    def test_message_names_the_leaf_and_the_sources(self, cfg):
        refs = [ref("01", "m", ses="001", name="a.nii.gz"),
                ref("01", "m", ses="002", name="b.nii.gz")]
        msg = _path_collisions(cfg, refs)[0]
        assert "task=m" in msg and "sub=01" in msg and "data.parquet" in msg
        assert "overwrite" in msg

    def test_many_colliding_sources_are_truncated_not_dumped(self, cfg):
        refs = [ref("01", "m", ses=f"{i:03d}", name=f"f{i}.nii.gz") for i in range(9)]
        msg = _path_collisions(cfg, refs)[0]
        assert msg.startswith("9 runs")
        assert "..." in msg

    def test_no_runs_is_not_a_collision(self, cfg):
        assert _path_collisions(cfg, []) == []
