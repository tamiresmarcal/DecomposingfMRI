#!/bin/bash
# Finalize -- runs once, after an array completes.
#
# Everything in here writes a SHARED file, which is exactly why it is not in
# the array. Array tasks write only their own leaf; shared artifacts
# (manifest.json, the diagnostics, and parquet's _metadata if you ever want it)
# have exactly one writer, in a serial pass afterwards.
#
#   sbatch --dependency=afterok:<extract_jobid> slurm/03_finalize.sbatch config.yaml activation

#SBATCH --job-name=fmridecomp_finalize
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/finalize_%j.out
#SBATCH --error=slurm_logs/finalize_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch 03_finalize.sbatch <config.yaml> <activation|dfc>}"
STAGE="${2:?stage must be 'activation' or 'dfc'}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Compute nodes have no outbound network, so the atlases MUST already be
# cached or every task dies on a fetch timeout. Prefer an existing cache over
# a path that merely looks tidy.
if [[ -z "${NILEARN_DATA:-}" ]]; then
  for _c in "$HOME/nilearn_data" "${SCRATCH:-}/nilearn_data" "$PWD/nilearn_data"; do
    [[ -d "$_c" ]] && { NILEARN_DATA="$_c"; break; }
  done
fi
export NILEARN_DATA="${NILEARN_DATA:-${SCRATCH:-$PWD}/nilearn_data}"
if [[ ! -d "$NILEARN_DATA" ]]; then
  echo "WARNING: no atlas cache at $NILEARN_DATA -- built-in atlases are fetched" >&2
  echo "         over the network, which compute nodes usually cannot reach." >&2
  echo "         Pre-fetch on a login node first. See README." >&2
fi

if [[ -n "${FMRIDECOMP_SIF:-}" ]]; then
  # No version pin: clusters retire apptainer releases, and a pinned load that
  # no longer exists fails every array task at once.
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  # Binds are NOT optional. A container sees only what is bound into it, so
  # without these the compute node reports "no such file" for data and outputs
  # that plainly exist on the login node. FMRIDECOMP_BINDS can add more.
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  RUN=(apptainer exec --bind "${BINDS}" --pwd "${PWD}" "${FMRIDECOMP_SIF}" python)
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python)
fi

"${RUN[@]}" -m fmri_decomposition.cli merge-manifests "${CONFIG}" --stage "${STAGE}"

if [[ "${STAGE}" == "activation" ]]; then
  # Coverage, L-R correlation, and the ISC alignment gate. The last one exits
  # non-zero if median |best_lag| exceeds one TR -- with --dependency=afterok on
  # the DFC job, that stops stage 3 from running on misaligned data rather than
  # producing window_ids that silently mean different things per subject.
  "${RUN[@]}" -m fmri_decomposition.cli diagnose "${CONFIG}"
fi
