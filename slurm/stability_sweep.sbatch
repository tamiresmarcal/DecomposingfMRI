#!/bin/bash
# Cross-cohort stability for every state set (tools/stability_sweep.py), as one job.
#
#   sbatch slurm/stability_sweep.sbatch                       # every atlas, window and state set
#   sbatch slurm/stability_sweep.sbatch -- --atlas harvardoxford --window-s 30 --states '.*_8'
#   sbatch slurm/stability_sweep.sbatch -- --atlas all --window-s all --balance time --hours 10 --n-boot 50 \
#          --out stability_sweep_balanced.csv     # replication balanced by scan time (notebook 14)
#
# Options after a bare -- go to the script. Output: stability_sweep_all.csv in the submit
# directory (or --out), rewritten every 10 sets, so a time-out still leaves the sets done.
# Cost is reading every cohort's transition tables once per set: about a minute per set.

#SBATCH --job-name=fmridecomp_stability_sweep
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/stability_sweep_%j.out
#SBATCH --error=slurm_logs/stability_sweep_%j.err

set -euo pipefail

EXTRA=()
seen=0
for a in "$@"; do
  if [[ $seen -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen=1
  else echo "put options after a bare --" >&2; exit 1; fi
done
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  ARGS=(--out stability_sweep_all.csv "${EXTRA[@]}")     # a later --out wins
else
  ARGS=(--atlas all --window-s all --out stability_sweep_all.csv)
fi

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "$REPO/slurm/env.sh" ]] && source "$REPO/slurm/env.sh"
SIF="${SIF:-${FMRIDECOMP_SIF_SHAP:-/project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif}}"
PYLIBS="${PYLIBS:-/project/6008063/tamires/singularity/pylibs}"
export APPTAINERENV_PYTHONPATH="$REPO:$PYLIBS"
module load apptainer 2>/dev/null || module load apptainer/1.3.5

cd "$REPO"
echo "host=$(hostname) args=${ARGS[*]}"
apptainer exec --bind "$REPO" ${SCRATCH:+--bind "$SCRATCH"} ${FMRIDECOMP_OUTPUTS:+--bind "$FMRIDECOMP_OUTPUTS"} \
    "$SIF" python tools/stability_sweep.py "${ARGS[@]}"
