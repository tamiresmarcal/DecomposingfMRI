#!/bin/bash
# Stage 5b -- rank state sets by how well their transitions predict a phenotype.
#
#   sbatch slurm/05_select.sbatch additional_HADS_anx_category
#   sbatch slurm/05_select.sbatch additional_HADS_dep_category -- --models ridge hgb lgbm
#
# --models defaults to `ridge hgb`, both of which are sklearn. `lgbm` needs
# lightgbm, which containers/stage45.def pins but an older build of the image may
# not carry; bstm_selection.make_model says so by name and offers hgb, the same
# algorithm family. So lightgbm is NOT checked up front -- doing that blocked a
# default run that never wanted it.
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
export DEPS="sklearn"

if [[ -n "${SIF}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  # --cleanenv keeps the host's PYTHONPATH out of the container, where it can
  # shadow container packages with host builds for a different Python.
  RUN=(apptainer exec --cleanenv --bind "${BINDS}" --pwd "${PWD}" \
       "${SIF}" python3)
  # Fail in seconds rather than after the first fit, and name the module that is
  # ACTUALLY missing. Printing the whole list instead sent a real diagnosis
  # astray: the image had sklearn and lacked lightgbm, and the message said
  # "cannot import: sklearn, lightgbm", which reads as the wrong image entirely.
  #
  # Only HARD requirements belong here. An optional dependency -- lightgbm for
  # --models lgbm, hmmlearn for --methods hmm, umap for --embeddings umap3 -- is
  # already handled where it is used, with a message that knows which flag asked
  # for it. Demanding it here is stricter than the code it guards, and blocked a
  # default `select` run that needed neither.
  # `import importlib.util` explicitly: a bare `import importlib` does NOT make
  # `.util` an attribute, so the block raised AttributeError, 2>/dev/null ate it,
  # MISSING came back empty and the guard passed everything. A guard that fails
  # open is worse than no guard, so the exit status is checked too.
  MISSING=$("${RUN[@]}" - <<'PY'
import importlib.util, os, sys
print(" ".join(m for m in os.environ["DEPS"].replace(",", " ").split()
               if importlib.util.find_spec(m) is None))
PY
) || { echo "ERROR: ${SIF} -- could not even run python3 to check imports" >&2
       exit 1; }
  # IS THIS IMAGE BUILT FROM THE CURRENT DEFINITION? A warning, not a failure:
  # most changes to the definition do not affect a given run, and refusing to
  # start would be worse than saying so. But it is the check that was missing --
  # the image on disk was built from a definition three commits old, and the
  # lightgbm added in the newest of them went unnoticed until a job died for it.
  _DEF="${SLURM_SUBMIT_DIR:-$PWD}/containers/stage45.def"
  if [[ -f "${_DEF}" ]]; then
    _WANT=$(sha256sum "${_DEF}" | cut -d" " -f1)
    # Through RUN as it stands (it ends in python3) rather than trying to swap
    # the interpreter out of the array: `${RUN[@]%python3}` strips the suffix from
    # EVERY element and leaves an empty string where python3 was, so the command
    # it builds is unrunnable.
    _HAVE=$("${RUN[@]}" -c 'import pathlib,sys
p = pathlib.Path("/opt/def_sha256")
sys.stdout.write(p.read_text().strip() if p.exists() else "")' 2>/dev/null)
    if [[ -z "${_HAVE}" ]]; then
      echo "NOTE: ${SIF} records no def_sha256, so it predates provenance" >&2
      echo "      being stamped into the image. It may be missing packages the" >&2
      echo "      current containers/stage45.def installs." >&2
    elif [[ "${_HAVE}" != "${_WANT}" ]]; then
      echo "NOTE: ${SIF} was built from a DIFFERENT containers/stage45.def" >&2
      echo "      image: ${_HAVE}" >&2
      echo "      repo : ${_WANT}" >&2
      echo "      Rebuild per containers/README.md if a run needs a package the" >&2
      echo "      definition has gained since." >&2
    fi
  fi

  if [[ -n "${MISSING// /}" ]]; then
    echo "ERROR: ${SIF}" >&2
    echo "       cannot import: ${MISSING}" >&2
    echo "       Those are required here. Either this is the stage 2/3 image --" >&2
    echo "         export FMRIDECOMP_SIF45=<the stage45 .sif>" >&2
    echo "       -- or the stage45 image needs rebuilding per containers/README.md." >&2
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
