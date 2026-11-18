#!/bin/bash
# The slow parts of notebooks 06 and 07, computed once into their caches
# (tools/predictive_cache.py): Cam-CAN SHAP refits and shares, the transfer's SHAP on HCP,
# its predictions and the SHAP shares of the best transferred configurations.
#
#   sbatch slurm/predictive_cache.sbatch
#   sbatch slurm/predictive_cache.sbatch -- --space harvardoxford,30,HMM2_umap3_8 --model hgb
#   sbatch slurm/predictive_cache.sbatch -- --skip shares contrib      # only what 07 needs
#
# Options after a bare -- go to the script. Whatever is already cached is read, not refitted,
# so a time-out can simply be resubmitted.

#SBATCH --job-name=fmridecomp_predictive_cache
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/predictive_cache_%j.out
#SBATCH --error=slurm_logs/predictive_cache_%j.err

set -euo pipefail

EXTRA=()
seen=0
for a in "$@"; do
  if [[ $seen -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen=1
  else echo "put options after a bare --" >&2; exit 1; fi
done

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "$REPO/slurm/env.sh" ]] && source "$REPO/slurm/env.sh"
SIF="${SIF:-${FMRIDECOMP_SIF_SHAP:-/project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif}}"
PYLIBS="${PYLIBS:-/project/6008063/tamires/singularity/pylibs}"
export APPTAINERENV_PYTHONPATH="$REPO:$PYLIBS"
export APPTAINERENV_HCP_PHENO="${HCP_PHENO:-/project/6008063/tamires/cohorts/hcp_pheno/participants_pheno.csv}"
module load apptainer 2>/dev/null || module load apptainer/1.3.5

cd "$REPO"
mkdir -p slurm_logs
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} args=${EXTRA[*]:-}"
apptainer exec --bind "$REPO" ${SCRATCH:+--bind "$SCRATCH"} ${FMRIDECOMP_OUTPUTS:+--bind "$FMRIDECOMP_OUTPUTS"} \
    --bind /project/6008063/tamires/cohorts \
    "$SIF" python tools/predictive_cache.py --n-jobs "${SLURM_CPUS_PER_TASK:-1}" "${EXTRA[@]}"
