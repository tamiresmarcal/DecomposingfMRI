#!/bin/bash
# Stage 4 -- latent decomposition. Fit on some cohorts, project onto others.
#
#   sbatch --array=0-3 slurm/04_decompose.sbatch harvardoxford 30 60 120 300 \
#            -- --censor-policy motion
#
# --censor-policy IS REQUIRED and has no default. Pass `none` to fit on every
# window, which is a different analysis and is recorded as one. It used to
# default to none with a warning, and window_s=-1 got built that way while every
# other aperture used `motion` -- a warning in a 30-line log is not a guard.
#
# THE ACTIVATION APERTURE. Per-TR frames instead of windowed edges, written to
# window_s=-1. One array task, because a frame has no window to vary:
#
#   sbatch --array=0-0 slurm/04_decompose.sbatch yeo7 -1 \
#            -- --source activation --censor-policy motion
#
# -1 is not a duration: it is one TR, which differs per cohort. Two transforms
# are applied to the frames that the dfc source does not need -- a common
# band-pass and a per-run z-score -- both recorded in model_hash. See
# fmri_decomposition/frames.py for why a correlation needs neither.
#
# ONE ARRAY TASK PER WINDOW SIZE. The window sizes are positional arguments
# after the atlas, and SLURM_ARRAY_TASK_ID indexes them -- so the array range
# must match the number of sizes you list (four sizes -> --array=0-3). Each
# fit is independent, so nothing is lost by splitting them and a lot of
# walltime is saved.
#
# CHECK THE MEMORY FIRST. Cost here is the training matrix: rows x edges x 4
# bytes, and Harvard-Oxford is 6,105 edges. On a login node:
#
#   fmri-decomp decompose --atlas harvardoxford --window-s 30 --dry-run
#
# which prints rows, edges and an estimated peak per window size. Set
# --mem below to at least double what it says; the default here is sized for
# Harvard-Oxford at 30 s (~124k training rows, ~3 GB matrix).
#
# UMAP is the slow part, not PCA: it fits a neighbour graph on
# --umap-fit-rows (30,000 by default) rows of full edge dimensionality, once
# per entry in --n-latents. Pass --no-umap for a PCA-only run, which finishes
# in minutes.

#SBATCH --job-name=fmridecomp_decompose
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/decompose_%A_%a.out
#SBATCH --error=slurm_logs/decompose_%A_%a.err

set -euo pipefail

ATLAS="${1:?usage: sbatch 04_decompose.sbatch <atlas> <window_s> [...] [-- extra args]}"
shift

# Window sizes up to a bare `--`; anything after it is forwarded to
# tools/decompose.py verbatim, e.g.
#   sbatch --array=0-1 slurm/04_decompose.sbatch yeo7 15 30 -- --no-umap
WINDOWS=(); EXTRA=(); seen_sep=0
for a in "$@"; do
  if [[ ${seen_sep} -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen_sep=1
  else WINDOWS+=("$a"); fi
done
if [[ ${#WINDOWS[@]} -eq 0 ]]; then
  echo "give at least one window size" >&2
  exit 1
fi

# Same env.sh contract as the other stages: interpreter, binds, account.
_ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

# BLAS threads are capped for the same reason as stage 2: randomized PCA and
# numba both spawn their own pools, and oversubscribing a 4-core task makes it
# slower, not faster.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMBA_NUM_THREADS="${OMP_NUM_THREADS}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#WINDOWS[@]} )); then
  echo "array index ${IDX} past the ${#WINDOWS[@]} window size(s) given -- " \
       "set --array=0-$(( ${#WINDOWS[@]} - 1 ))" >&2
  exit 1
fi
WINDOW="${WINDOWS[$IDX]}"

echo "host=$(hostname) atlas=${ATLAS} window_s=${WINDOW} " \
     "task=${IDX}/${#WINDOWS[@]} cpus=${OMP_NUM_THREADS}"

# Stages 4 and 5 need the OTHER image. containers/README.md: stages 2-3 run in
# fmri_decomp.sif, stages 4-5 in fmri_decomp_stage45.sif, which is where Stan,
# hmmlearn, umap-learn and lightgbm live. One FMRIDECOMP_SIF for both meant
# either editing slurm/env.sh between stages or watching this job die on
# `ModuleNotFoundError: hmmlearn` minutes in, so FMRIDECOMP_SIF45 wins here and
# FMRIDECOMP_SIF is the fallback for anyone who keeps one image.
SIF="${FMRIDECOMP_SIF45:-${FMRIDECOMP_SIF:-}}"
DEPS="sklearn, umap"

if [[ -n "${SIF}" ]]; then
  # No version pin: clusters retire apptainer releases, and a pinned load that
  # no longer exists fails every array task at once.
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  # --cleanenv keeps the host's PYTHONPATH out of the container, where it can
  # shadow container packages with host builds for a different Python.
  RUN=(apptainer exec --cleanenv --bind "${BINDS}" --pwd "${PWD}" \
       "${SIF}" python3)   # -m fmri_decomposition.cli below
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

# ${EXTRA[@]+...} so an empty array expands to nothing under `set -u`
# rather than to one empty string argument.
"${RUN[@]}" -m fmri_decomposition.cli decompose \
    --atlas "${ATLAS}" --window-s "${WINDOW}" ${EXTRA[@]+"${EXTRA[@]}"}
echo "atlas=${ATLAS} window_s=${WINDOW} done"
