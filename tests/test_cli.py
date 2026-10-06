"""Guards on the submit path.

slurm/activation_and_dfc.sh runs `fmri-decomp validate` and nothing else before it
burns core-hours, so anything that must not reach a compute node has to fail
here. tools/check_cohort.py is more thorough but nobody's submit script calls
it.
"""
import re
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


class TestDriverReferencesRealScripts:
    """Every script activation_and_dfc.sh submits has to exist.

    sbatch resolves the path at submit time, so a rename that misses one
    reference fails only after the jobs before it are already queued -- and
    `set -euo pipefail` then aborts the driver with half a chain submitted and
    the rest of the dependency graph never built. Cheap to catch here instead.
    """

    SLURM = Path(__file__).resolve().parent.parent / "slurm"

    @pytest.fixture
    def submitted(self):
        """The scripts the driver sbatches, in submission order.

        Only the part before the closing heredoc: that text is documentation
        printed to the user, and the commands quoted in it are deliberately
        NOT what this script runs. Line continuations are folded first, since
        every sbatch call here spans two lines.
        """
        body = (self.SLURM / "activation_and_dfc.sh").read_text().split("cat <<EOF")[0]
        body = body.replace("\\\n", " ")
        return re.findall(r'sbatch\b[^\n]*?"\$HERE/([A-Za-z0-9_.]+)"', body)

    def test_every_submitted_script_exists(self, submitted):
        assert submitted, "found no sbatch'd \"$HERE/<script>\" to check"
        missing = sorted({n for n in submitted if not (self.SLURM / n).is_file()})
        assert not missing, f"activation_and_dfc.sh submits missing script(s): {missing}"

    def test_the_chain_covers_every_per_cohort_stage(self, submitted):
        """The chain is the definition of "this cohort is prepared"."""
        for script in ("extract_activations.sbatch", "finalize.sbatch",
                       "extract_dfc.sbatch", "censor.sbatch"):
            assert script in submitted, f"{script} dropped from the chain"

    def test_censor_is_submitted_last(self, submitted):
        """Re-gating is only cheap while nothing in the chain follows censor.

        If a later job depended on the censor job, changing the policy would
        mean re-running that too, and `sbatch censor.sbatch` alone would no
        longer be the whole answer -- which is what README and RUNBOOK both
        promise it is.
        """
        assert submitted[-1] == "censor.sbatch", \
            f"submitted after censor: {submitted[submitted.index('censor.sbatch') + 1:]}"
