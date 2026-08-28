#!/bin/bash
# Stage 2 -- activation. One SLURM array task per shard of the run list.
#
#   sbatch --array=0-19 slurm/01_extract.sbatch config/ds002837.yaml
#
# The array index and the array size are read from SLURM, so the same file
# works for 4 subjects or 650. Set the array range to the number of shards you
# want; each task processes runs [i::n] of the discovered list.
#
# Restartability: every shard is written with an atomic rename and
# skip-if-exists, so a requeued or timed-out task re-runs only what is missing.
# Re-submitting the identical array after a walltime kill is safe and cheap.

#SBATCH --job-name=fmridecomp_extract
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/extract_%A_%a.out
#SBATCH --error=slurm_logs/extract_%A_%a.err

set -euo pipefail

CONFIG="${1:?usage: sbatch 01_extract.sbatch <config.yaml> [extra cli args]}"
shift || true
# Anything after the config is forwarded to `fmri-decomp extract`. The one you
# will almost always need is --no-strict: a cohort with no fixed
# stimulus.durations_s (ds002837 and cneuromod both, since run lengths differ
# per subject) reports a validation problem, and strict mode makes that fatal.
# Without it the whole array dies on the first line of every task.
EXTRA=("$@")

# --cpus-per-task, not --ntasks. joblib/loky forks inside ONE task and can only
# see that task's cores; --ntasks=30 asks the scheduler for 30 independent tasks
# that may land on different nodes, where 29 of them would sit idle. This is the
# single most common way to wait days in the queue for a job that then runs
# single-threaded.
N_JOBS="${SLURM_CPUS_PER_TASK:-1}"

# BLAS must be capped or each of the N_JOBS workers spawns its own thread pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Keep the atlas cache off $HOME: it is small-file heavy and quota-limited.
# Compute nodes have no outbound network, so the atlases MUST already be
# cached or every task dies on a fetch timeout. Prefer an existing cache over
# a path that merely looks tidy.
if [[ -z "${NILEARN_DATA:-}" ]]; then
  for _c in "$HOME/nilearn_data" "${SCRATCH:-}/nilearn_data" "$PWD/nilearn_data"; do
    [[ -d "$_c" ]] && { NILEARN_DATA="$_c"; break; }
  done
fi
export NILEARN_DATA="${NILEARN_DATA:-${SCRATCH:-$PWD}/nilearn_data}"
if [[ ! -d "$NILEARN_DATA" ]]; then
  echo "WARNING: no atlas cache at $NILEARN_DATA -- built-in atlases are fetched" >&2
  echo "         over the network, which compute nodes usually cannot reach." >&2
  echo "         Pre-fetch on a login node first. See README." >&2
fi

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "host=$(hostname) shard=${SHARD_INDEX}/${SHARD_COUNT} cpus=${N_JOBS}"
echo "config=${CONFIG}"

# Two ways to get an interpreter. Prefer a venv on the cluster's own Python
# (Alliance wheels are prebuilt and faster than conda); the container path is
# kept for parity with the legacy pipeline.
if [[ -n "${FMRIDECOMP_SIF:-}" ]]; then
  # No version pin: clusters retire apptainer releases, and a pinned load that
  # no longer exists fails every array task at once.
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  # Binds are NOT optional. A container sees only what is bound into it, so
  # without these the compute node reports "no such file" for data and outputs
  # that plainly exist on the login node. FMRIDECOMP_BINDS can add more.
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  RUN=(apptainer exec --bind "${BINDS}" --pwd "${PWD}" "${FMRIDECOMP_SIF}" python)
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python)
fi

echo "extra args: ${EXTRA[*]:-(none)}"
"${RUN[@]}" -m fmri_decomposition.cli extract "${CONFIG}" \
    --shard "${SHARD_INDEX}/${SHARD_COUNT}" \
    --n-jobs "${N_JOBS}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "shard ${SHARD_INDEX} done"
