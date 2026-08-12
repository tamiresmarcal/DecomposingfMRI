#!/usr/bin/env bash
# Submit the whole chain with dependencies:
#
#   extract array -> finalize+ISC gate -> dfc array -> merge dfc manifests
#
# Usage:
#   ./slurm/submit_all.sh config/ds002837.yaml [n_extract_shards] [n_dfc_shards]
#
# Nothing runs if validation fails, and stage 3 does not start unless the ISC
# alignment gate passes -- afterok, not afterany, on purpose.

set -euo pipefail

CONFIG="${1:?usage: submit_all.sh <config.yaml> [n_extract_shards] [n_dfc_shards]}"
N_EXTRACT="${2:-20}"
N_DFC="${3:-8}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p slurm_logs

echo "== validating config and cohort before burning any core-hours"
python -m fmri_decomposition.cli validate "$CONFIG"

echo "== stage 2: ${N_EXTRACT} array task(s)"
EXTRACT_ID=$(sbatch --parsable --array=0-$((N_EXTRACT - 1)) \
  "$HERE/01_extract.sbatch" "$CONFIG")
echo "   jobid ${EXTRACT_ID}"

echo "== finalize stage 2 (manifest merge + diagnostics + ISC gate)"
FINAL2_ID=$(sbatch --parsable --dependency=afterok:"${EXTRACT_ID}" \
  "$HERE/03_finalize.sbatch" "$CONFIG" activation)
echo "   jobid ${FINAL2_ID}"

echo "== stage 3: ${N_DFC} array task(s), gated on the diagnostics passing"
DFC_ID=$(sbatch --parsable --dependency=afterok:"${FINAL2_ID}" \
  --array=0-$((N_DFC - 1)) "$HERE/02_dfc.sbatch" "$CONFIG")
echo "   jobid ${DFC_ID}"

echo "== finalize stage 3"
FINAL3_ID=$(sbatch --parsable --dependency=afterok:"${DFC_ID}" \
  "$HERE/03_finalize.sbatch" "$CONFIG" dfc)
echo "   jobid ${FINAL3_ID}"

cat <<EOF

submitted. watch with:
  squeue -u \$USER -o '%.10i %.20j %.8T %.10M %R'
  tail -f slurm_logs/extract_${EXTRACT_ID}_0.out

a timed-out or preempted task is safe to resubmit as-is; skip-if-exists means
it redoes only the shards that are missing:
  sbatch --array=0-$((N_EXTRACT - 1)) $HERE/01_extract.sbatch $CONFIG
EOF
