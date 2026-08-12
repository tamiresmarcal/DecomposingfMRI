#!/bin/bash
# Stage 3 -- DFC. Cheap and re-run often: it reads parquet, never the NIfTI.
#
#   sbatch --array=0-3 slurm/02_dfc.sbatch config/ds002837.yaml
#   sbatch --array=0-3 slurm/02_dfc.sbatch config/ds002837.yaml 30 60
#
# Extra positional arguments restrict the window sizes, so re-running a single
# window size after changing your mind costs one short job rather than a full
# re-extraction. That separation is the whole reason stages 2 and 3 are split.
#
# Ask for far less than stage 2: no NIfTI is ever loaded here. One subject's
# parcel timeseries at 111 parcels x 5,470 TRs is about 2.4 MB.

#SBATCH --job-name=fmridecomp_dfc
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/dfc_%A_%a.out
#SBATCH --error=slurm_logs/dfc_%A_%a.err

set -euo pipefail

CONFIG="${1:?usage: sbatch 02_dfc.sbatch <config.yaml> [window_s ...]}"
shift || true
WINDOWS=("$@")

N_JOBS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NILEARN_DATA="${NILEARN_DATA:-${SCRATCH:-$PWD}/nilearn_data}"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "host=$(hostname) shard=${SHARD_INDEX}/${SHARD_COUNT} cpus=${N_JOBS}"

if [[ -n "${FMRIDECOMP_SIF:-}" ]]; then
  module load apptainer/1.3.5
  RUN=(apptainer exec "${FMRIDECOMP_SIF}" python)
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python)
fi

ARGS=(-m fmri_decomposition.cli dfc "${CONFIG}"
      --shard "${SHARD_INDEX}/${SHARD_COUNT}" --n-jobs "${N_JOBS}")
if [[ ${#WINDOWS[@]} -gt 0 ]]; then
  ARGS+=(--window-s "${WINDOWS[@]}")
fi

"${RUN[@]}" "${ARGS[@]}"
echo "shard ${SHARD_INDEX} done"
