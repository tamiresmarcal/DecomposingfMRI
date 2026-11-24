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
# COST: ~20 ms per variogram-matched surrogate; ~170 reference maps x 10,000 surrogates is
# ~10 h of CPU. The surrogates are independent, so they run in parallel over every core of the
# allocation (refmaps.burt_surrogates, one process per core): with 32 cores the nulls take
# ~20-25 min, plus the loading of the maps -- about 1 h in all. More cores, faster still
# (--cpus-per-task=48 or 64 where the nodes have them).
#
# NEEDS brainsmash in the pylibs folder the notebooks use, once:
#   apptainer exec $SIF python -m pip install --no-deps \
#       --target /project/6008063/tamires/singularity/pylibs brainsmash

#SBATCH --job-name=fmridecomp_neurobio_burt
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=03:00:00
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
# parallelism = processes (one per core) for the surrogates, so one BLAS thread each
export FMRIDECOMP_NULL_JOBS="${SLURM_CPUS_PER_TASK:-1}" APPTAINERENV_FMRIDECOMP_NULL_JOBS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 APPTAINERENV_PYTHONUNBUFFERED=1

SIF="${SIF:-${FMRIDECOMP_SIF_SHAP:-/project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif}}"
PYLIBS="${PYLIBS:-/project/6008063/tamires/singularity/pylibs}"
export APPTAINERENV_PYTHONPATH="$REPO:$PYLIBS"
module load apptainer 2>/dev/null || module load apptainer/1.3.5

cd "$REPO"
mkdir -p slurm_logs
echo "host=$(hostname) cpus=$OMP_NUM_THREADS extra=${EXTRA[*]:-}"
apptainer exec --bind "${FMRIDECOMP_BINDS:-/project,/scratch,/home}" --pwd "$REPO" "$SIF" \
    python3 tools/notebook04_null.py --null burt ${EXTRA[@]+"${EXTRA[@]}"}
