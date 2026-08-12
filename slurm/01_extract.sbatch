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

CONFIG="${1:?usage: sbatch 01_extract.sbatch <config.yaml>}"

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
export NILEARN_DATA="${NILEARN_DATA:-${SCRATCH:-$PWD}/nilearn_data}"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "host=$(hostname) shard=${SHARD_INDEX}/${SHARD_COUNT} cpus=${N_JOBS}"
echo "config=${CONFIG}"

# Two ways to get an interpreter. Prefer a venv on the cluster's own Python
# (Alliance wheels are prebuilt and faster than conda); the container path is
# kept for parity with the legacy pipeline.
if [[ -n "${FMRIDECOMP_SIF:-}" ]]; then
  module load apptainer/1.3.5
  RUN=(apptainer exec "${FMRIDECOMP_SIF}" python)
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python)
fi

"${RUN[@]}" -m fmri_decomposition.cli extract "${CONFIG}" \
    --shard "${SHARD_INDEX}/${SHARD_COUNT}" \
    --n-jobs "${N_JOBS}"

echo "shard ${SHARD_INDEX} done"
