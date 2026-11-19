#!/bin/bash
# Notebook 08's latent-space diagnostics (tools/latent_diagnostics.py), one array task per part,
# all in parallel. Each task reads the dFC edges itself; the notebook only reads what they write:
#   <outputs>/modelling/version=v0/unsupervised_model/latent_diagnostics/atlas=/window_s=/
#
#   sbatch --array=0-5 slurm/latent_diagnostics.sbatch stats base all_rows no_scaler person_scaled nndb_only
#   sbatch --array=0-0 slurm/latent_diagnostics.sbatch all_rows -- --atlas harvardoxford --window-s 30
#
# Positional = the parts: `stats` (window statistics + state edge means, no fit) or a UMAP variant
# (base, all_rows, no_scaler, person_scaled, nndb_only, nndb_all, nndb_old). SLURM_ARRAY_TASK_ID picks one, so --array
# must be 0..(number of parts - 1). Options after a bare -- go to the script (default: harvardoxford,
# 30 s, HMM2_umap3_8). all_rows fits the UMAP on every training window (~124k x 6,105) and is the
# slow one; the others finish in about an hour.

#SBATCH --job-name=fmridecomp_latent_diag
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/latent_diag_%A_%a.out
#SBATCH --error=slurm_logs/latent_diag_%A_%a.err

set -euo pipefail

PARTS=(); EXTRA=(); seen=0
for a in "$@"; do
  if [[ $seen -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen=1
  else PARTS+=("$a"); fi
done
if [[ ${#PARTS[@]} -eq 0 ]]; then
  PARTS=(stats base all_rows no_scaler person_scaled nndb_only)
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#PARTS[@]} )); then
  echo "array index ${IDX} past the ${#PARTS[@]} part(s) given -- set --array=0-$(( ${#PARTS[@]} - 1 ))" >&2
  exit 1
fi
PART="${PARTS[$IDX]}"
if [[ "$PART" == "stats" ]]; then ARGS=(--part stats); else ARGS=(--part umap --variant "$PART"); fi

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "$REPO/slurm/env.sh" ]] && source "$REPO/slurm/env.sh"
SIF="${SIF:-${FMRIDECOMP_SIF45:-/project/6008063/tamires/singularity/fmri_decomp_stage45.sif}}"
export APPTAINERENV_PYTHONPATH="$REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS" APPTAINERENV_MKL_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" APPTAINERENV_NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1 APPTAINERENV_PYTHONUNBUFFERED=1
module load apptainer 2>/dev/null || module load apptainer/1.3.5

cd "$REPO"
mkdir -p slurm_logs
echo "host=$(hostname) part=${PART} task=${IDX}/${#PARTS[@]} extra=${EXTRA[*]:-}"
apptainer exec --bind "${FMRIDECOMP_BINDS:-/project,/scratch,/home}" --pwd "$REPO" \
    "$SIF" python3 tools/latent_diagnostics.py "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
