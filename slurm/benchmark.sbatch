#!/bin/bash
# Stage 5c -- transitions vs static FC, movie vs rest, on one subject set.
#
#   sbatch slurm/benchmark.sbatch additional_HADS_anx_category
#   sbatch slurm/benchmark.sbatch additional_HADS_dep_category \
#       -- --conditions movie=camcan rest=camcan_rest --models ridge hgb
#
# --conditions defaults to `movie=camcan rest=camcan_rest` below, which is the
# design this stage exists for. A movie-only run (just the bstm-vs-fc half) is
# `-- --conditions movie=camcan`.
#
# WHAT HAS TO EXIST FIRST
#   stage 5a  transitions  for every cohort named in --conditions
#   stage 3b  static-fc    for every cohort named in --conditions
# The FC table is not optional even for a bstm-only reading: it is where
# `frac_good_frames` and `n_tr_used` come from, and those are the covariates
# every arm shares. `benchmark` says so by name if one is missing.
#
# COST. Fits = cells x models x seeds x 5 folds, where a cell is one
# (condition, arm, atlas, state set). The FC arm is the expensive one: 6,105
# edges at harvardoxford against ~600 subjects. RidgeCV over 25 alphas on that
# is seconds per fold; HistGradientBoosting on 6,105 features is MINUTES per
# fold, which is why --models defaults to ridge alone in the Python layer. Add
# hgb deliberately and raise --time with it.
#
# MEMORY is the one real constraint here, and it is the FC arm again: every
# cell is held in memory at once so the grid is fitted without re-reading.
# 600 subjects x 6,105 edges x 8 bytes is 29 MB per cell, x 3 atlases x 2
# conditions is under 200 MB. 32G is generous; it covers the parquet read
# buffers and joblib's per-worker copies, which is where it actually goes --
# loky pickles each matrix to each worker, so peak is roughly n_jobs x the
# largest cell.
#
# ONE JOB, NOT AN ARRAY: every cell needs the same phenotype join and the same
# shared subject set, and that set is defined by the intersection across ALL
# cells. An array would compute a different intersection per task, which is
# precisely the bug the design exists to prevent.

#SBATCH --job-name=fmridecomp_benchmark
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/benchmark_%j.out
#SBATCH --error=slurm_logs/benchmark_%j.err

set -euo pipefail

TARGET="${1:?usage: sbatch slurm/benchmark.sbatch <phenotype_column> [-- extra args]}"
shift

EXTRA=(); seen_sep=0
for a in "$@"; do
  if [[ ${seen_sep} -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen_sep=1
  else echo "unexpected argument '$a' -- put extra args after a bare --" >&2
       exit 1; fi
done

_ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

# Single-threaded estimators by construction (bstm_selection.make_model):
# joblib owns the parallelism over grid cells, and a threaded estimator inside
# a joblib worker oversubscribes the node and runs slower than either alone.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Stage 5 needs the stage45 image -- sklearn, and lightgbm for --models lgbm.
# See slurm/env.sh.example for the two-image boundary.
SIF="${FMRIDECOMP_SIF45:-${FMRIDECOMP_SIF:-}}"
# matplotlib is a HARD requirement here, unlike in select: this stage always
# writes both figures and there is no flag to skip them.
DEPS="sklearn matplotlib"

echo "host=$(hostname) target=${TARGET} cpus=${SLURM_CPUS_PER_TASK:-1}"

if [[ -n "${SIF}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  RUN=(apptainer exec --cleanenv --bind "${BINDS}" --pwd "${PWD}" \
       "${SIF}" python3)
  # Module names go in as ARGV, not through the environment: --cleanenv strips
  # exported variables, so a guard reading os.environ cannot run at all. The
  # long version of this story is in slurm/model_selection.sbatch.
  MISSING=$("${RUN[@]}" -c 'import importlib.util, sys
print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))' \
            ${DEPS}) || {
    echo "ERROR: ${SIF} -- could not run python3 inside it to check imports" >&2
    exit 1; }
  _DEF="${SLURM_SUBMIT_DIR:-$PWD}/containers/stage45.def"
  if [[ -f "${_DEF}" ]]; then
    _WANT=$(sha256sum "${_DEF}" | cut -d" " -f1)
    # `|| true` because this is a WARNING and must never kill a job: the script
    # runs under `set -e`.
    _HAVE=$("${RUN[@]}" -c 'import pathlib,sys
p = pathlib.Path("/opt/def_sha256")
sys.stdout.write(p.read_text().strip() if p.exists() else "")' 2>/dev/null || true)
    if [[ -n "${_HAVE}" && "${_HAVE}" != "${_WANT}" ]]; then
      echo "NOTE: ${SIF} was built from a DIFFERENT containers/stage45.def" >&2
      echo "      image: ${_HAVE}" >&2
      echo "      repo : ${_WANT}" >&2
    fi
  fi
  if [[ -n "${MISSING// /}" ]]; then
    echo "ERROR: ${SIF}" >&2
    echo "       cannot import: ${MISSING}" >&2
    echo "       Either this is the stage 2/3 image --" >&2
    echo "         export FMRIDECOMP_SIF45=<the stage45 .sif>" >&2
    echo "       -- or it needs rebuilding per containers/README.md." >&2
    exit 1
  fi
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python3)
fi

# --conditions goes in FIRST so a caller's own --conditions after `--`
# overrides it: argparse keeps the last value for a repeated option.
"${RUN[@]}" -m fmri_decomposition.cli benchmark \
    --conditions movie=camcan rest=camcan_rest \
    --target "${TARGET}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "target=${TARGET} done"
