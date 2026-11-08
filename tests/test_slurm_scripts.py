"""Properties of the job scripts that only fail on the cluster, hours later.

Two of these have actually bitten:

* **A command above the #SBATCH block.** SLURM reads the directives only until
  the first command, so an `export` placed in the comment block above them
  silently throws away every `--cpus-per-task`, `--mem` and `--time` in the
  file. The job then runs with the partition defaults and nothing says so.
  Caught here while writing transfer.sbatch, not on the cluster.
* **Unset thread caps inside the container.** OpenBLAS and libgomp read the
  NODE's core count, not the cgroup's, so a 4-cpu task on a 192-core node
  starts 192 threads. The first real transfer array died
  `Segmentation fault (core dumped)`, forty times out of forty.
"""
import re
import subprocess
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parent.parent / "slurm"
SBATCH = sorted(SLURM.glob("*.sbatch"))
SHELL = sorted(SLURM.glob("*.sh")) + SBATCH
THREAD_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]


def lines(p):
    return p.read_text().splitlines()


def first_command(p):
    """(line number, text) of the first line that is neither blank nor a comment."""
    for i, t in enumerate(lines(p), 1):
        if t.strip() and not t.lstrip().startswith("#"):
            return i, t
    return None, None


@pytest.mark.parametrize("p", SBATCH, ids=lambda p: p.name)
def test_no_command_precedes_the_sbatch_directives(p):
    directives = [i for i, t in enumerate(lines(p), 1)
                  if t.startswith("#SBATCH")]
    assert directives, f"{p.name} has no #SBATCH directive at all"
    i, text = first_command(p)
    assert i > max(directives), (
        f"{p.name}: `{text.strip()}` at line {i} comes before the last #SBATCH "
        f"at line {max(directives)}. SLURM stops reading directives at the "
        f"first command, so every resource request after it is ignored.")


@pytest.mark.parametrize("p", SHELL, ids=lambda p: p.name)
def test_it_is_valid_bash(p):
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# The scripts that FIT MODELS, where an uncapped BLAS is the difference between
# a run and a core dump. The rest of slurm/ is measurement and I/O: censor,
# extract_dfc and finalize cap OMP_NUM_THREADS or nothing, have never shown the
# problem, and are slow enough that changing their threading without being able
# to time it on the cluster would be a worse bet than leaving them alone. They
# are exempt BY NAME, so the list is a decision rather than an oversight.
FITTING = ["decomposition.sbatch", "clustering.sbatch",
           "brain_states_transitions.sbatch", "model_selection.sbatch",
           "fcm_selection.sbatch", "static_fc.sbatch", "transfer.sbatch"]


@pytest.mark.parametrize("name", FITTING)
def test_the_thread_caps_are_set_before_python_runs(name):
    """Unset is not "use my cpus": inside the container the runtimes read the
    NODE's core count, so a 4-cpu task on a 192-core node starts 192 threads."""
    p = SLURM / name
    text = p.read_text()
    missing = [v for v in THREAD_VARS if f"export {v}" not in text]
    assert not missing, (
        f"{p.name} does not export {missing}. Inside the container the "
        f"runtimes then read the NODE's core count, not the cgroup's -- which "
        f"is how the first transfer array segfaulted forty times out of forty.")


def test_every_fitting_script_still_exists():
    """So a rename cannot quietly empty the list above."""
    for name in FITTING:
        assert (SLURM / name).is_file(), f"{name} is in FITTING but not in slurm/"


class TestTransferSbatch:
    p = SLURM / "transfer.sbatch"

    def test_the_cap_and_the_cpu_request_cannot_disagree(self):
        """THREADS above --cpus-per-task is the oversubscription that crashed;
        the script refuses rather than running it."""
        t = self.p.read_text()
        assert "THREADS" in t and 'SLURM_CPUS_PER_TASK:-1}" ]]' in t
        assert "More threads than cpus" in t

    def test_python_is_unbuffered(self):
        """A block-buffered stdout is lost when the process dies on a signal,
        which is how forty crashed tasks left a log holding only the script's
        own echo."""
        t = self.p.read_text()
        assert "export PYTHONUNBUFFERED=1" in t
        assert re.search(r'RUN\[@\]}" -u ', t)

    def test_the_default_is_one_thread_per_cpu_requested(self):
        t = self.p.read_text()
        assert ': "${THREADS:=1}"' in t
        assert "#SBATCH --cpus-per-task=1" in t
