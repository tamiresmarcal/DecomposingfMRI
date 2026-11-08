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
# THREADS, NOT JOBLIB. There is no joblib loop in the script, so the estimator
# gets the task's whole allocation and OMP_NUM_THREADS is left alone -- the
# opposite of model_selection.sbatch, where a threaded estimator inside a
# joblib worker oversubscribes the node.
#
# COST. ~9,000 fits + ~2,250 baselines over 282 state sets, two tasks, two
# targets, four arms and two models. Boosting dominates; ridge is milliseconds.
# At 40 shards that is ~230 fits each. --cpus-per-task is what buys walltime.
#
# MEMORY is not the constraint: the transition tables are ~100 KB each and the
# biggest design is 649 x 729 float64, about 4 MB.

#SBATCH --job-name=fmridecomp_transfer
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/transfer_%A_%a.out
#SBATCH --error=slurm_logs/transfer_%A_%a.err

set -euo pipefail

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

echo "host=$(hostname) target=${TARGET} tree=${TREE} shard=${SHARD}/${N_SHARDS} cpus=${SLURM_CPUS_PER_TASK:-1}"

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

"${RUN[@]}" tools/transfer_grid.py \
    --target "${TARGET}" --tree "${TREE}" \
    --shard "${SHARD}" --n-shards "${N_SHARDS}" \
    --threads "${SLURM_CPUS_PER_TASK:-1}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "target=${TARGET} tree=${TREE} shard=${SHARD} done"
