#!/bin/bash
# Stage 5b' -- rank STATIC connectivity models. The third selection tree.
#
#   sbatch slurm/fcm_selection.sbatch additional_HADS_anx_category
#   sbatch slurm/fcm_selection.sbatch additional_HADS_anx_category \
#       -- --cohorts camcan camcan_rest --models ridge hgb
#
# THREE TREES, ONE SHAPE, read side by side:
#
#   outputs/bstm_selection/          sbatch slurm/model_selection.sbatch $Y
#   outputs/resting_bstm_selection/  sbatch slurm/model_selection.sbatch $Y \
#                                      -- --cohort camcan_rest \
#                                         --output-name resting_bstm_selection
#   outputs/fcm_selection/           this
#
# They are separate RUNS, so each uses whoever it has. If a gap between their
# summary.csv files is going to be read as a result, pass the SAME
# --restrict-subjects file to all three and check `n` matches.
#
# WHAT HAS TO EXIST FIRST: stage 3b, for every cohort named.
#   sbatch slurm/static_fc.sbatch camcan camcan_rest
# The static-FC table is also where frac_good_frames and n_tr_used come from,
# which are covariates here and have no equivalent in the state trees.
#
# --models defaults to `ridge` ALONE here, unlike model_selection's `ridge hgb`:
# a tree ensemble on 6,105 features against ~600 subjects is both slow and badly
# matched. Add hgb or lgbm deliberately. `lgbm` needs lightgbm, which
# containers/stage45.def pins but an older build of the image may not carry;
# bstm_selection.make_model says so by name and offers hgb, the same algorithm
# family. So lightgbm is NOT checked up front -- doing that blocked a default
# run that never wanted it.
#
# ONE JOB, NOT AN ARRAY. The grid (cells x arms x models x fold seeds) is
# parallelised inside the process with joblib, because every cell needs the same
# phenotype join in memory and splitting across nodes would re-read and re-join
# it per task for no gain. Ask for cores, not array indices.
#
# COST. Fits scale as cells x arms x models x seeds x 5 folds, where a cell is
# one (cohort, atlas). The width is what differs from model_selection:
# harvardoxford is 6,105 edges against ~600 subjects. RidgeCV over 25 alphas on
# that is seconds per fold; HistGradientBoosting is MINUTES per fold, which is
# why --models defaults to ridge alone. Add hgb deliberately and raise --time.
#
# MEMORY is bigger than model_selection's for the same reason -- a transitions
# table is ~100 KB and a harvardoxford FC table is ~15 MB -- but still small.
# It goes mostly on loky pickling each matrix to each worker, so peak is
# roughly n_jobs x the largest cell. 32G is generous.

#SBATCH --job-name=fmridecomp_fcmselect
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/fcm_select_%j.out
#SBATCH --error=slurm_logs/fcm_select_%j.err

set -euo pipefail

TARGET="${1:?usage: sbatch fcm_selection.sbatch <phenotype_column> [-- extra args]}"
shift

# Anything after a bare `--` is forwarded verbatim, e.g.
#   sbatch slurm/fcm_selection.sbatch <col> -- --cohorts camcan camcan_rest
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
# matplotlib is a HARD requirement here: this stage always writes its two
# figures and has no flag to skip them.
DEPS="sklearn matplotlib"

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
  # The module names go in as ARGV, not through the environment. RUN carries
  # --cleanenv -- deliberately, to keep the host's PYTHONPATH out of the
  # container -- and --cleanenv strips exported variables too, so os.environ
  # ["DEPS"] raised KeyError inside the container, python exited non-zero, and
  # this guard killed every job in 10 seconds. A check that cannot run is not a
  # check; argv crosses the boundary that the environment does not.
  MISSING=$("${RUN[@]}" -c 'import importlib.util, sys
print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))' \
            ${DEPS}) || {
    echo "ERROR: ${SIF} -- could not run python3 inside it to check imports" >&2
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
    # `|| true` because this is a WARNING and must never be able to kill a job.
    # The script runs under `set -e`, so a non-zero exit from a purely
    # informational check would abort the run -- which is precisely how the
    # import guard above took out three jobs in ten seconds.
    _HAVE=$("${RUN[@]}" -c 'import pathlib,sys
p = pathlib.Path("/opt/def_sha256")
sys.stdout.write(p.read_text().strip() if p.exists() else "")' 2>/dev/null || true)
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

# --cohorts goes in FIRST so a caller's own --cohorts after `--` overrides it:
# argparse keeps the last value for a repeated option.
"${RUN[@]}" -m fmri_decomposition.cli fcm-select \
    --cohorts camcan camcan_rest \
    --target "${TARGET}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "target=${TARGET} done"
