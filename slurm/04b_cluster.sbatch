#!/bin/bash
# Stage 4b -- add brain-state definitions to latents that already exist.
#
#   sbatch slurm/04b_cluster.sbatch harvardoxford 30 60 120 300
#   sbatch slurm/04b_cluster.sbatch yeo7 -1
#   sbatch slurm/04b_cluster.sbatch yeo7 30 -- --methods hmm --k 8
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

if [[ -n "${FMRIDECOMP_SIF:-}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  RUN=(apptainer exec --cleanenv --bind "${BINDS}" --pwd "${PWD}" \
       "${FMRIDECOMP_SIF}" python3)
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python3)
fi

"${RUN[@]}" -m fmri_decomposition.cli cluster \
    --atlas "${ATLAS}" --window-s "${WINDOWS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}

echo "atlas=${ATLAS} done"
