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
N_EXTRACT="${2:-8}"   # see the sizing check below
N_DFC="${3:-8}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p slurm_logs

echo "== validating config and cohort before burning any core-hours"
EXTRA_EXTRACT=()
if VALIDATE_OUT="$(python -m fmri_decomposition.cli validate "$CONFIG" 2>&1)"; then
  echo "$VALIDATE_OUT"
else
  # Exit 1 from `validate` means it found problems, not that it crashed. The
  # commonest is "no stimulus duration for task X", which is expected on every
  # cohort whose runs differ in length -- dfc.py falls back to each file's own
  # observed duration. That must not abort the chain; it must switch extract to
  # --no-strict, which is exactly what a human would do by hand.
  echo "$VALIDATE_OUT"
  echo
  echo "   validate reported problem(s) above."
  echo "   Adding --no-strict to stage 2 so they warn instead of aborting."
  echo "   Read them first: --no-strict is right for a missing stimulus"
  echo "   duration, and WRONG for a participants/disk mismatch you did not"
  echo "   intend."
  read -r -p "   continue with --no-strict? [y/N] " reply
  [[ "$reply" == [yY]* ]] || { echo "   aborted."; exit 1; }
  EXTRA_EXTRACT=(--no-strict)
fi

# ---------------------------------------------------------------- sizing ---
# The unit of stage-2 work is one (bold file x atlas) pair, so the job count is
# runs x atlases -- NOT the subject count. Asking for more worker slots than
# there are jobs bills you for idle cores and queues longer for no gain.
N_RUNS="$(sed -n 's/.*runs_discovered=\([0-9]*\).*/\1/p' <<<"$VALIDATE_OUT")"
N_ATLAS="$(sed -n "s/^atlases=\[\(.*\)\] config_hash.*/\1/p" <<<"$VALIDATE_OUT" \
           | tr ',' '\n' | grep -c .)"
CPUS="$(sed -n 's/^#SBATCH --cpus-per-task=\([0-9]*\).*/\1/p' "$HERE/01_extract.sbatch" | head -1)"
CPUS="${CPUS:-1}"

if [[ -n "$N_RUNS" && -n "$N_ATLAS" && "$N_ATLAS" -gt 0 ]]; then
  N_JOBS_TOTAL=$(( N_RUNS * N_ATLAS ))
  SLOTS=$(( N_EXTRACT * CPUS ))
  echo "   stage 2: ${N_RUNS} run(s) x ${N_ATLAS} atlas(es) = ${N_JOBS_TOTAL} job(s)"
  echo "   requesting ${N_EXTRACT} task(s) x ${CPUS} cpu(s) = ${SLOTS} worker slot(s)"
  if (( SLOTS > N_JOBS_TOTAL )); then
    SUGGEST=$(( (N_JOBS_TOTAL + CPUS - 1) / CPUS ))
    (( SUGGEST < 1 )) && SUGGEST=1
    echo
    echo "   WARNING: ${SLOTS} slots for ${N_JOBS_TOTAL} jobs -- $(( SLOTS - N_JOBS_TOTAL )) will idle."
    echo "            You are billed for the whole allocation and it queues slower."
    echo "            At --cpus-per-task=${CPUS}, ${SUGGEST} array task(s) is enough:"
    echo "              ./slurm/submit_all.sh $CONFIG ${SUGGEST} ${N_DFC}"
    echo
    read -r -p "   submit anyway? [y/N] " reply
    [[ "$reply" == [yY]* ]] || { echo "   aborted."; exit 1; }
  fi
fi

echo "== stage 2: ${N_EXTRACT} array task(s)"
EXTRACT_ID=$(sbatch --parsable --array=0-$((N_EXTRACT - 1)) \
  "$HERE/01_extract.sbatch" "$CONFIG" ${EXTRA_EXTRACT[@]+"${EXTRA_EXTRACT[@]}"})
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
