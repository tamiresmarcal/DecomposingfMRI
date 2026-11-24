#!/bin/bash
# Figure 2 and Supplementary Fig. 1 (notebook 04) with the Burt null (variogram matching,
# Burt et al. 2020, brainsmash), drawn by notebook 04's own cells -- the same parameters, layout
# and fonts as the notebook; only the null differs (tools/notebook04_null.py).
#
#   sbatch slurm/neurobio_null.sbatch                          # Burt, the notebook's N_NULLS (10,000)
#   sbatch slurm/neurobio_null.sbatch -- --n-nulls 1000        # quicker check
#
# Output: notebooks/figures/04_burt/{4_0,4_1,4_2,4_2_2}.png, and the tables in a
# contextualization/<run>_burt/ directory. The Moran run and figures are not touched.
#
# COST: brainsmash fits a variogram per surrogate, ~5 min per reference map for 10,000
# surrogates at ~100 parcels. The tables test ~95 maps (molecular, Yeo 7, networks14, BrainMap
# domains and subcategories) and the matrices ~75 more: about 15 h on one core.
#
# NEEDS brainsmash in the pylibs folder the notebooks use, once:
#   apptainer exec $SIF python -m pip install --no-deps \
#       --target /project/6008063/tamires/singularity/pylibs brainsmash

#SBATCH --job-name=fmridecomp_neurobio_burt
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_logs/neurobio_burt_%j.out
#SBATCH --error=slurm_logs/neurobio_burt_%j.err

set -euo pipefail

EXTRA=(); seen=0
for a in "$@"; do
  if [[ $seen -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen=1
  else echo "put options after a bare --" >&2; exit 1; fi
done

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "$REPO/slurm/env.sh" ]] && source "$REPO/slurm/env.sh"
if [[ -z "${FMRIDECOMP_REFMAPS:-}" || ! -d "${FMRIDECOMP_REFMAPS}" ]]; then
  export FMRIDECOMP_REFMAPS="$REPO/reference_atlases"
fi
export APPTAINERENV_FMRIDECOMP_REFMAPS="$FMRIDECOMP_REFMAPS"
[[ -n "${FMRIDECOMP_OUTPUTS:-}" ]] && export APPTAINERENV_FMRIDECOMP_OUTPUTS="$FMRIDECOMP_OUTPUTS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONUNBUFFERED=1 APPTAINERENV_PYTHONUNBUFFERED=1

SIF="${SIF:-${FMRIDECOMP_SIF_SHAP:-/project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif}}"
PYLIBS="${PYLIBS:-/project/6008063/tamires/singularity/pylibs}"
export APPTAINERENV_PYTHONPATH="$REPO:$PYLIBS"
module load apptainer 2>/dev/null || module load apptainer/1.3.5

cd "$REPO"
mkdir -p slurm_logs
echo "host=$(hostname) cpus=$OMP_NUM_THREADS extra=${EXTRA[*]:-}"
apptainer exec --bind "${FMRIDECOMP_BINDS:-/project,/scratch,/home}" --pwd "$REPO" "$SIF" \
    python3 tools/notebook04_null.py --null burt ${EXTRA[@]+"${EXTRA[@]}"}
