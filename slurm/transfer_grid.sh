#!/usr/bin/env bash
# Submit the whole Cam-CAN -> HCP transfer grid: one array for the transitions,
# one job for the static FC, per target set.
#
#   ./slurm/transfer_grid.sh                    # both target sets, 40 shards
#   TARGETS=raw N_SHARDS=20 ./slurm/transfer_grid.sh
#   ./slurm/transfer_grid.sh -- --atlas yeo7    # a slice, to try the chain
#
# The notebook then READS what this writes:
#
#   rows = tt.read_transferred(ROOT, "raw")     # 13_camcan_to_hcp.ipynb, §2
#
# WHAT THE PRE-FLIGHT IS FOR. It resolves the grid the way the array tasks
# will, on the login node, and refuses before submitting if there is nothing
# to do: no state set on disk for BOTH cohorts (stage 5a has not run for HCP),
# or more shards than sets (empty tasks). It costs seconds and is the
# difference between a refusal now and 40 tasks that each die in ten.
#
# IT DOES NOT CLEAN UP. A shard whose rows.parquet exists is SKIPPED, so a
# resubmit fills the gaps and leaves the finished shards alone. To redo the
# grid, delete the `targets=` partition -- the reader refuses to mix shards cut
# from different grids, but it cannot see a stale shard that still matches.

set -euo pipefail

: "${TARGETS:=raw category}"
: "${N_SHARDS:=40}"
: "${TREES:=bstm fcm}"

EXTRA=(); seen=0
for a in "$@"; do
  if [[ $seen -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen=1; fi
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p slurm_logs
[[ -f "$HERE/env.sh" ]] && source "$HERE/env.sh"
ACCOUNT_ARG=()
[[ -n "${FMRIDECOMP_ACCOUNT:-}" ]] && ACCOUNT_ARG=(--account="$FMRIDECOMP_ACCOUNT")

RUN=(python3)
if [[ -n "${FMRIDECOMP_SIF45:-${FMRIDECOMP_SIF:-}}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5 2>/dev/null || true
  RUN=(apptainer exec --cleanenv \
       --bind "${FMRIDECOMP_BINDS:-/project,/scratch,/home}" --pwd "$PWD" \
       "${FMRIDECOMP_SIF45:-$FMRIDECOMP_SIF}" python3)
elif [[ -x "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/python" ]]; then
  RUN=("${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/python")
fi

for target in $TARGETS; do
  echo "== pre-flight: ${target}"
  "${RUN[@]}" tools/transfer_grid.py --target "$target" --dry-run \
      --n-shards "$N_SHARDS" ${EXTRA[@]+"${EXTRA[@]}"}
done

for target in $TARGETS; do
  for tree in $TREES; do
    if [[ "$tree" == "fcm" ]]; then
      J=$(TREE=fcm sbatch ${ACCOUNT_ARG[@]+"${ACCOUNT_ARG[@]}"} --parsable \
          "$HERE/transfer.sbatch" "$target" 1 \
          ${EXTRA[@]+-- "${EXTRA[@]}"})
      echo "   ${target}/fcm   jobid ${J}  (one job: static FC has no state set)"
    else
      J=$(TREE=bstm sbatch ${ACCOUNT_ARG[@]+"${ACCOUNT_ARG[@]}"} --parsable \
          --array=0-$((N_SHARDS - 1)) "$HERE/transfer.sbatch" "$target" "$N_SHARDS" \
          ${EXTRA[@]+-- "${EXTRA[@]}"})
      echo "   ${target}/bstm  jobid ${J}  (${N_SHARDS} shards)"
    fi
  done
done

cat <<MSG

submitted. watch with:
  squeue -u \$USER -o '%.14i %.22j %.8T %.10M %R'

a shard that failed, by name:
  grep -l Error slurm_logs/transfer_*_*.out

then read it in notebook 13 §2. The reader REFUSES a half-finished grid --
"the best of 282 state sets" from the shards that happened to finish is a
different claim in the same words -- so resubmit the missing shard(s) first:
the same command skips the ones already written.
MSG
