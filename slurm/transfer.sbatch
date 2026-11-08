#!/bin/bash
# One shard of the Cam-CAN -> HCP transfer grid (tools/transfer_grid.py).
#
#   sbatch --array=0-39 slurm/transfer.sbatch raw 40
#   TREE=fcm sbatch slurm/transfer.sbatch raw 1
#
# Usually submitted by slurm/transfer_grid.sh, which does the pre-flight and
# submits both trees. This script is one array TASK: it fits the state sets of
# its own shard and writes one parquet.
#
#   $1  the Cam-CAN target set: raw | category
#   $2  how many shards the array has (must match --array)
#   TREE=bstm|fcm     which tree. fcm has no state set to shard by, so it is
#                     one job with n_shards=1.
#   anything after a bare `--` is forwarded to transfer_grid.py verbatim.
#
# AN ARRAY, UNLIKE slurm/model_selection.sbatch. That one keeps the grid in one
# process because every cell needs the same small tables in memory and joblib
# can own the parallelism. Here each fit is on EVERY training subject -- 649
# rows by up to 729 cells, no folds -- so the fits are big and few rather than
# small and many, re-reading one 100 KB transitions table per shard costs
# nothing, and a shard that hits the walltime is one shard to resubmit instead
# of the whole grid. Hence: array indices, and threads INSIDE the fit.
#
# THREADS ARE CAPPED, AND THE CAP IS EXPORTED BEFORE PYTHON STARTS. The first
# version of this script left OMP_NUM_THREADS alone, reasoning that with no
# joblib loop the estimator may as well have the task's whole allocation. Every
# task of the first real array then died with
#
#   Segmentation fault      (core dumped)   tools/transfer_grid.py
#
# and one hung. UNSET is not "use my 4 cpus": inside the container OpenBLAS and
# libgomp read the NODE's core count, not the cgroup's, so a 4-cpu task on a
# 192-core node starts 192 threads, and OpenBLAS allocating its per-thread
# buffers under a cgroup that small is a known way to crash. Which is exactly
# why every other stage 4-5 script here exports these to 1.
#
# So: THREADS, default 1, exported to all four runtimes AND passed to the
# script, so nothing can read the node instead. To give the fits real threads,
# raise both together -- they must agree:
#
#   THREADS=4 sbatch --cpus-per-task=4 --array=0-39 slurm/transfer.sbatch raw 40
#
# Block-buffered stdout is lost when a process is killed by a signal, which is
# how the segfaulting tasks came to leave a log holding only this script's own
# echo -- nothing about how far python had got. Hence PYTHONUNBUFFERED and
# `python3 -u` below.
#
# COST. ~9,000 fits + ~2,250 baselines over 282 state sets, two tasks, two
# targets, four arms and two models. Boosting dominates; ridge is milliseconds.
# At 40 shards that is ~230 fits each. --cpus-per-task is what buys walltime.
#
# MEMORY is not the constraint: the transition tables are ~100 KB each and the
# biggest design is 649 x 729 float64, about 4 MB.

#SBATCH --job-name=fmridecomp_transfer
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/transfer_%A_%a.out
#SBATCH --error=slurm_logs/transfer_%A_%a.err

set -euo pipefail

# The thread cap, and why it is exported here rather than left to the runtimes:
# see the block above. Below `set -euo pipefail`, because SLURM reads #SBATCH
# directives only until the first command -- an assignment above them would
# have silently thrown away every resource request in this file.
: "${THREADS:=1}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMBA_NUM_THREADS="${THREADS}"
export PYTHONUNBUFFERED=1

TARGET="${1:?usage: sbatch --array=0-N slurm/transfer.sbatch <raw|category> <n_shards> [-- extra]}"
N_SHARDS="${2:?how many shards the array has}"
shift 2
TREE="${TREE:-bstm}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

EXTRA=(); seen=0
for a in "$@"; do
  if [[ ${seen} -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen=1
  else echo "unexpected argument '$a' -- put extra args after a bare --" >&2
       exit 1; fi
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$HERE/env.sh" ]] && source "$HERE/env.sh"

echo "host=$(hostname) target=${TARGET} tree=${TREE} shard=${SHARD}/${N_SHARDS} cpus=${SLURM_CPUS_PER_TASK:-1} threads=${THREADS}"
if [[ "${THREADS}" -gt "${SLURM_CPUS_PER_TASK:-1}" ]]; then
  echo "ERROR: THREADS=${THREADS} but this task has ${SLURM_CPUS_PER_TASK:-1} cpu(s)." >&2
  echo "       More threads than cpus is what crashed the first array. Submit with" >&2
  echo "       --cpus-per-task=${THREADS}, or leave THREADS unset." >&2
  exit 1
fi

# Stages 4-5 need the stage45 image (containers/README.md); the reasoning and
# the full set of guards are in slurm/model_selection.sbatch.
SIF="${FMRIDECOMP_SIF45:-${FMRIDECOMP_SIF:-}}"
if [[ -n "${SIF}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  RUN=(apptainer exec --cleanenv --bind "${FMRIDECOMP_BINDS:-/project,/scratch,/home}" \
       --pwd "${PWD}" "${SIF}" python3)
  MISSING=$("${RUN[@]}" -c 'import importlib.util, sys
print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))' \
            sklearn pandas pyarrow scipy threadpoolctl) || {
    echo "ERROR: ${SIF} -- could not run python3 inside it to check imports" >&2
    exit 1; }
  if [[ -n "${MISSING// /}" ]]; then
    echo "ERROR: ${SIF} cannot import: ${MISSING}" >&2
    echo "       Either this is the stage 2/3 image -- export FMRIDECOMP_SIF45 --" >&2
    echo "       or the stage45 image needs rebuilding per containers/README.md." >&2
    exit 1
  fi
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python3)
fi

"${RUN[@]}" -u tools/transfer_grid.py \
    --target "${TARGET}" --tree "${TREE}" \
    --shard "${SHARD}" --n-shards "${N_SHARDS}" \
    --threads "${THREADS}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "target=${TARGET} tree=${TREE} shard=${SHARD} done"
