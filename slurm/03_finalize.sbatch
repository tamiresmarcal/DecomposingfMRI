#!/bin/bash
# Finalize -- runs once, after an array completes.
#
# Everything in here writes a SHARED file, which is exactly why it is not in
# the array. Array tasks write only their own leaf; shared artifacts
# (manifest.json, the diagnostics, and parquet's _metadata if you ever want it)
# have exactly one writer, in a serial pass afterwards.
#
#   sbatch --dependency=afterok:<extract_jobid> slurm/03_finalize.sbatch config.yaml activation

#SBATCH --job-name=fmridecomp_finalize
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/finalize_%j.out
#SBATCH --error=slurm_logs/finalize_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch 03_finalize.sbatch <config.yaml> <activation|dfc>}"
STAGE="${2:?stage must be 'activation' or 'dfc'}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NILEARN_DATA="${NILEARN_DATA:-${SCRATCH:-$PWD}/nilearn_data}"

if [[ -n "${FMRIDECOMP_SIF:-}" ]]; then
  module load apptainer/1.3.5
  RUN=(apptainer exec "${FMRIDECOMP_SIF}" python)
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python)
fi

"${RUN[@]}" -m fmri_decomposition.cli merge-manifests "${CONFIG}" --stage "${STAGE}"

if [[ "${STAGE}" == "activation" ]]; then
  # Coverage, L-R correlation, and the ISC alignment gate. The last one exits
  # non-zero if median |best_lag| exceeds one TR -- with --dependency=afterok on
  # the DFC job, that stops stage 3 from running on misaligned data rather than
  # producing window_ids that silently mean different things per subject.
  "${RUN[@]}" -m fmri_decomposition.cli diagnose "${CONFIG}"
fi
