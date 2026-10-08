#!/bin/bash
# Censor -- turn the QC measurements into a named keep/drop decision.
#
#   sbatch slurm/censor.sbatch config/ds002837.yaml
#   sbatch slurm/censor.sbatch config/ds002837.yaml config/censor/strict.yaml
#
# THIS IS THE ONE PLACE A MEASUREMENT BECOMES A DECISION. Everything before it
# measures; `finalize` says so in its own header and means it. The threshold
# lives here, in a policy file that is named and hashed, and the policy name is
# carried into every `model_hash` downstream -- so two runs under different
# policies can never be mistaken for one another.
#
# CHEAP, AND SEPARATE ON PURPOSE. It reads one CSV and writes one small parquet,
# in seconds. That is what makes a change of mind cheap: to re-gate after
# everything has run, copy the policy, edit the copy, and run THIS script again.
# Nothing upstream is touched, and the stages that consume it are selected by
# `--censor-policy <name>`, so the old decision stays on disk beside the new one.
#
#   cp config/censor/motion.yaml config/censor/strict.yaml
#   $EDITOR config/censor/strict.yaml          # change `name:` too
#   sbatch slurm/censor.sbatch config/<cohort>.yaml config/censor/strict.yaml
#
# NEVER EDIT A POLICY IN PLACE. Outputs already on disk record the policy by
# NAME; editing the file it came from leaves them claiming a gate that no longer
# means what it says.
#
# Per cohort, because `activation_and_dfc.sh` submits it per cohort -- but the
# command itself defaults to every cohort that has a participants_qc.csv, so
# running it by hand with no --cohorts does all of them at once.

#SBATCH --job-name=fmridecomp_censor
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/censor_%j.out
#SBATCH --error=slurm_logs/censor_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch censor.sbatch <cohort.yaml> [policy.yaml]}"
POLICY="${2:-config/censor/motion.yaml}"

_ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

if [[ -n "${FMRIDECOMP_SIF:-}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  RUN=(apptainer exec --bind "${BINDS}" --pwd "${PWD}" "${FMRIDECOMP_SIF}" python)
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python)
fi

# The cohort name comes from the config, so the caller passes the same YAML it
# passes every other stage rather than spelling the cohort a second time. Read
# with ${RUN[@]}, not a bare `python3`: a compute node's default interpreter has
# no pyyaml, and on some it does not exist at all.
COHORT=$("${RUN[@]}" -c "
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['cohort'])" "$CONFIG")
COHORT="${COHORT//[[:space:]]/}"
[[ -n "$COHORT" ]] || { echo "ERROR: no 'cohort:' key in $CONFIG" >&2; exit 1; }

echo "host=$(hostname) cohort=${COHORT} policy=${POLICY}"

"${RUN[@]}" -m fmri_decomposition.cli censor \
    --policy "${POLICY}" --cohorts "${COHORT}"

echo "cohort=${COHORT} censored under ${POLICY}"
