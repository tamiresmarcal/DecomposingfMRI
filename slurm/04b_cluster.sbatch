#!/bin/bash
# Stage 4b -- add brain-state definitions to latents that already exist.
#
#   sbatch slurm/04b_cluster.sbatch harvardoxford 30 60 120 300
#   sbatch slurm/04b_cluster.sbatch yeo7 -1
#   sbatch slurm/04b_cluster.sbatch yeo7 30 -- --methods hmm --k 8
#
# LOOK AT THE GRID FIRST, on a login node -- it fits nothing and takes seconds:
#
#   fmri-decomp cluster --check --atlas harvardoxford yeo7 networks \
#                       --window-s 15 30 60 120 300 -1
#
# It prints, per (atlas, aperture), which cohorts and embeddings are there, how
# many state columns a re-run would REPLACE, and what would be written. The HMM
# is the cost here and it is hours at K=27 across a full grid; finding a missing
# embedding after three of those have run is the expensive way to find out.
#
# NOT AN ARRAY, and deliberately so. Every clusterer for a given
# (atlas, aperture) is fitted on the SAME embedding block and the labels are
# appended to the SAME parquet file. Two array tasks writing different state
# columns into one file would race: each reads the file, adds its column, and
# renames over the other's work, so the one that finishes second wins and the
# first column is gone. One task per atlas, every aperture inside it.
#
# COST is the clusterers, and they are not alike. Threshold is a quantile
# transform (seconds). MeanShift is O(n_fit^2) in the bandwidth search, capped
# by --meanshift-fit-rows. HMM is Baum-Welch over the full sequence and is by
# far the slowest at K=27 -- it dominates the walltime, so --hmm-iter is the
# knob if this times out.
#
# MEMORY is small: only three embedding columns per row are ever loaded, never
# the edges. Even Harvard-Oxford at 15 s is a few hundred MB.
#
# WHAT GETS WRITTEN. Label columns are APPENDED to
# latents/atlas=<a>/window_s=<w>/cohort=<c>/data.parquet, with per-column
# provenance in the schema metadata. The embeddings are not refitted, so this is
# safe to re-run when a new clusterer is added -- it does not redo the ones that
# already work.

#SBATCH --job-name=fmridecomp_cluster
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/cluster_%j.out
#SBATCH --error=slurm_logs/cluster_%j.err

set -euo pipefail

ATLAS="${1:?usage: sbatch 04b_cluster.sbatch <atlas> <window_s> [...] [-- extra args]}"
shift

# Window sizes up to a bare `--`; anything after it is forwarded verbatim.
WINDOWS=(); EXTRA=(); seen_sep=0
for a in "$@"; do
  if [[ ${seen_sep} -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen_sep=1
  else WINDOWS+=("$a"); fi
done
if [[ ${#WINDOWS[@]} -eq 0 ]]; then
  echo "give at least one window size (-1 is the activation aperture)" >&2
  exit 1
fi

_ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

# hmmlearn and MeanShift both go through BLAS; one pool sized to the task beats
# several fighting over the same cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMBA_NUM_THREADS="${OMP_NUM_THREADS}"

echo "host=$(hostname) atlas=${ATLAS} windows=${WINDOWS[*]} cpus=${OMP_NUM_THREADS}"

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

"${RUN[@]}" -m fmri_decomposition.cli cluster \
    --atlas "${ATLAS}" --window-s "${WINDOWS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}

echo "atlas=${ATLAS} done"
