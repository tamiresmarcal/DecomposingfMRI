#!/bin/bash
# Stage 3b -- one connectivity matrix per subject. The benchmark's control arm.
#
#   sbatch slurm/static_fc.sbatch camcan camcan_rest
#   sbatch slurm/static_fc.sbatch camcan -- --atlas yeo7 --pool subject
#
# Cohorts are positional; anything after a bare `--` is forwarded verbatim.
# --atlas defaults to all three below, which is what the benchmark wants: it
# compares atlases, and an atlas missing at rest silently drops that arm.
#
# WHAT IT COSTS. One pass over the stage-2 parquet per (atlas, cohort), then a
# correlation per subject. The correlation is nothing -- 160 x 111 is
# microseconds. The cost is reading the frames, which is the same read
# `decompose --window-s -1` does, so budget what that took and halve it (no
# PCA, no UMAP, no writing a row per TR).
#
# MEMORY is the whole cohort's frames at once: n_subjects x n_tr x n_parcels x
# 4 bytes, about 650 x 193 x 111 x 4 = 56 MB for camcan at harvardoxford. 32G
# is already generous; it is sized for the read buffers, not the matrix.
#
# ONE TASK, NOT AN ARRAY. Each (atlas, cohort) writes a single
# subjects.parquet, so an array would have several tasks racing to write one
# file -- the same one-writer-per-cell constraint as stage 4b, and here the
# cell is the whole cohort.
#
# THIS MUST USE THE SAME FRAME SETTINGS AS THE STATE ARM. --match-bandpass and
# --zscore-runs default to exactly what `decompose` defaults to, which is why
# the two arms are comparable at all. If you passed either flag to `decompose`,
# pass the same here, or the benchmark compares preprocessing.

#SBATCH --job-name=fmridecomp_staticfc
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/static_fc_%j.out
#SBATCH --error=slurm_logs/static_fc_%j.err

set -euo pipefail

COHORTS=(); EXTRA=(); seen_sep=0
for a in "$@"; do
  if [[ ${seen_sep} -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen_sep=1
  elif [[ "$a" == -* ]]; then
    echo "unexpected option '$a' -- put extra args after a bare --" >&2; exit 1
  else COHORTS+=("$a"); fi
done
if [[ ${#COHORTS[@]} -eq 0 ]]; then
  echo "usage: sbatch slurm/static_fc.sbatch <cohort> [<cohort> ...] [-- extra args]" >&2
  echo "  e.g. sbatch slurm/static_fc.sbatch camcan camcan_rest" >&2
  exit 1
fi

_ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

# The work is one correlation per subject in a plain loop -- no joblib here --
# so BLAS may use the cores. This is the opposite of clustering.sbatch, where
# joblib owns the parallelism and a threaded BLAS inside a worker
# oversubscribes the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

# Stage 3b reads stage 2 and needs only numpy/pandas/pyarrow/scipy, so the
# STAGE 2-3 image is the right one -- unlike stages 4-5. See
# slurm/env.sh.example for the two-image boundary.
SIF="${FMRIDECOMP_SIF:-${FMRIDECOMP_SIF45:-}}"
DEPS="pandas pyarrow scipy"

echo "host=$(hostname) cpus=${OMP_NUM_THREADS} cohorts=${COHORTS[*]}"

if [[ -n "${SIF}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  RUN=(apptainer exec --cleanenv --bind "${BINDS}" --pwd "${PWD}" \
       "${SIF}" python3)
  # Module names go in as ARGV, not through the environment: --cleanenv strips
  # exported variables, so a guard reading os.environ cannot run at all. The
  # long version of this story is in slurm/brain_states_transitions.sbatch.
  MISSING=$("${RUN[@]}" -c 'import importlib.util, sys
print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))' \
            ${DEPS}) || {
    echo "ERROR: ${SIF} -- could not run python3 inside it to check imports" >&2
    exit 1; }
  if [[ -n "${MISSING// /}" ]]; then
    echo "ERROR: ${SIF} cannot import: ${MISSING}" >&2
    exit 1
  fi
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python3)
fi

# --atlas goes in FIRST so a caller's own --atlas after `--` overrides it:
# argparse keeps the last value for a repeated option. Same ordering trick as
# clustering.sbatch's --hmm2-jobs.
"${RUN[@]}" -m fmri_decomposition.cli static-fc \
    --atlas harvardoxford yeo7 networks \
    --cohorts "${COHORTS[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "static-fc done: ${COHORTS[*]}"
