#!/bin/bash
# Stage 5b -- rank state sets by how well their transitions predict a phenotype.
#
#   sbatch slurm/05_select.sbatch additional_HADS_anx_category
#   sbatch slurm/05_select.sbatch additional_HADS_dep_category -- --models ridge hgb lgbm
#
# ONE JOB, NOT AN ARRAY. The grid (state sets x models x fold seeds) is
# parallelised inside the process with joblib, because every cell needs the same
# handful of small tables in memory and splitting across nodes would re-read and
# re-join them per task for no gain. Ask for cores, not array indices.
#
# COST. The fits are the whole cost and they scale as
# state_sets x models x seeds x 5 folds. At 16 state sets, 2 models and 5 seeds
# that is 800 fits; ridge is milliseconds each, boosting is seconds. Boosting
# dominates, so --cpus-per-task is what buys walltime.
#
# MEMORY is not the constraint: every transitions table is ~100 KB, so the whole
# grid is a few MB. 16G is already generous.

#SBATCH --job-name=fmridecomp_select
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/select_%j.out
#SBATCH --error=slurm_logs/select_%j.err

set -euo pipefail

TARGET="${1:?usage: sbatch 05_select.sbatch <phenotype_column> [-- extra args]}"
shift

# Anything after a bare `--` is forwarded verbatim, e.g.
#   sbatch slurm/05_select.sbatch <col> -- --models ridge hgb lgbm --seeds 0 1 2
EXTRA=(); seen_sep=0
for a in "$@"; do
  if [[ ${seen_sep} -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen_sep=1
  # `${a!r}` would be python; in bash that is a bad substitution and the error
  # handler would itself die with a confusing message instead of this one.
  else echo "unexpected argument '$a' -- put extra args after a bare --" >&2
       exit 1; fi
done

# The estimators are single-threaded by construction (see bstm_selection.make_model):
# joblib owns the parallelism, and a threaded estimator inside a joblib worker
# oversubscribes the node and runs slower than either alone.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

echo "host=$(hostname) target=${TARGET} cpus=${SLURM_CPUS_PER_TASK:-1}"

# Stages 4 and 5 need the OTHER image. containers/README.md: stages 2-3 run in
# fmri_decomp.sif, stages 4-5 in fmri_decomp_stage45.sif, which is where Stan,
# hmmlearn, umap-learn and lightgbm live. One FMRIDECOMP_SIF for both meant
# either editing slurm/env.sh between stages or watching this job die on
# `ModuleNotFoundError: hmmlearn` minutes in, so FMRIDECOMP_SIF45 wins here and
# FMRIDECOMP_SIF is the fallback for anyone who keeps one image.
SIF="${FMRIDECOMP_SIF45:-${FMRIDECOMP_SIF:-}}"
DEPS="sklearn, lightgbm"

if [[ -n "${SIF}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  # --cleanenv keeps the host's PYTHONPATH out of the container, where it can
  # shadow container packages with host builds for a different Python.
  RUN=(apptainer exec --cleanenv --bind "${BINDS}" --pwd "${PWD}" \
       "${SIF}" python3)
  # Fail in seconds rather than after the first fit. An image without these is
  # the stage 2/3 image, and the error it would otherwise give arrives late and
  # names only the first missing module.
  if ! "${RUN[@]}" -c "import ${DEPS}" 2>/dev/null; then
    echo "ERROR: ${SIF}" >&2
    echo "       cannot import: ${DEPS}" >&2
    echo "       That is the stage 2/3 image. Stages 4-5 need the stage45 one:" >&2
    echo "         export FMRIDECOMP_SIF45=/project/6008063/tamires/singularity/fmri_decomp_stage45.sif" >&2
    echo "       Build it per containers/README.md if it does not exist yet." >&2
    exit 1
  fi
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python3)
fi

"${RUN[@]}" -m fmri_decomposition.cli select \
    --target "${TARGET}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "target=${TARGET} done"
