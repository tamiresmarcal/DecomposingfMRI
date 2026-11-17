#!/bin/bash
# Notebook 14's mixed models (VPC task / person) for every state set, once
# (tools/mixed_model_sweep.py). The notebook then reads them instead of refitting:
#   <outputs>/modelling/version=v0/unsupervised_model/mixed_model/atlas=/window_s=/states=/
#
#   sbatch slurm/mixed_model.sbatch                         # every atlas, window and state set
#   sbatch slurm/mixed_model.sbatch -- --atlas harvardoxford --window-s 30 --states 'HMM2_umap3_8'
#   sbatch slurm/mixed_model.sbatch -- --overwrite          # refit what is already there
#
# Options after a bare -- go to the script. Sets already written with the same settings are
# skipped, so a time-out can simply be resubmitted.

#SBATCH --job-name=fmridecomp_mixed_model
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/mixed_model_%j.out
#SBATCH --error=slurm_logs/mixed_model_%j.err

set -euo pipefail

EXTRA=()
seen=0
for a in "$@"; do
  if [[ $seen -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen=1
  else echo "put options after a bare --" >&2; exit 1; fi
done
JOBS="${SLURM_CPUS_PER_TASK:-1}"
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  ARGS=(--atlas all --window-s all --jobs "$JOBS" "${EXTRA[@]}")    # later options win
else
  ARGS=(--atlas all --window-s all --jobs "$JOBS")
fi

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "$REPO/slurm/env.sh" ]] && source "$REPO/slurm/env.sh"
SIF="${SIF:-${FMRIDECOMP_SIF_SHAP:-/project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif}}"
PYLIBS="${PYLIBS:-/project/6008063/tamires/singularity/pylibs}"
export APPTAINERENV_PYTHONPATH="$REPO:$PYLIBS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1      # one core per state set
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
module load apptainer 2>/dev/null || module load apptainer/1.3.5

cd "$REPO"
mkdir -p slurm_logs
echo "host=$(hostname) args=${ARGS[*]}"
apptainer exec --bind "$REPO" ${SCRATCH:+--bind "$SCRATCH"} ${FMRIDECOMP_OUTPUTS:+--bind "$FMRIDECOMP_OUTPUTS"} \
    "$SIF" python tools/mixed_model_sweep.py "${ARGS[@]}"
