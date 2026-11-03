#!/usr/bin/env bash
# Project an EXISTING fit onto a new cohort, then rank its models. One chain:
#
#   decompose array -> cluster -> transitions -> select-bstm + select-fcm
#
# Usage:
#   TARGETS="DSM_Depr_Raw DSM_Anxi_Raw" CONDITIONS="movie=MOVIE1 rest=REST1" \
#     ./slurm/project_and_select.sh hcp7t harvardoxford -- \
#         --pheno config/phenotype/hcp7t_phenotype.csv:, --id-col sub \
#         --restrict-subjects config/phenotype/hcp7t_one_per_family.txt \
#         --covariates age sex --categorical sex
#
# WHAT THIS IS FOR, AND WHAT IT IS NOT
# ------------------------------------
# This stage 4 RE-USES a fit; it does not make one. That is the whole point:
# the embedding and the clusterers were fitted on --train and never saw this
# cohort, so a score here is a TRANSFER claim. If any flag makes `model_hash`
# differ from what is on disk, `decompose` refits instead -- the state
# definitions change, and the claim quietly becomes something weaker. So the
# pre-flight below reads the stored fit description and REFUSES rather than
# submitting a chain that would refit.
#
# CONDITIONS maps an output partition to the task(s) it scores:
# `movie=MOVIE1` writes task=movie and filters rows to MOVIE1. A cohort whose
# subjects have several runs of one condition needs one here, because the
# selections score ONE row per subject and will not average them.
#
# Anything after a bare `--` is forwarded verbatim to BOTH selections, so they
# are scored on the same sample. DECOMPOSE_EXTRA and CLUSTER_EXTRA carry the
# per-stage flags that cannot be shared.

set -euo pipefail

COHORT="${1:?usage: project_and_select.sh <cohort> <atlas> [-- extra selection args]}"
ATLAS="${2:?atlas}"
shift 2
SEL_EXTRA=(); seen=0
for a in "$@"; do
  if [[ $seen -eq 1 ]]; then SEL_EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen=1; fi
done

: "${APERTURES:=30 60 120 300}"   # the dfc fits. -1 is its own source and is
                                  # assumed already done -- see the check below.
: "${ALL_APERTURES:=-1 $APERTURES}"
: "${TRAIN:=ds002837 cneuromod}"
: "${TARGETS:?set TARGETS, e.g. TARGETS=\"DSM_Depr_Raw DSM_Anxi_Raw\"}"
: "${CONDITIONS:?set CONDITIONS, e.g. CONDITIONS=\"movie=MOVIE1 rest=REST1\"}"
: "${MODELS:=ridge hgb}"
: "${SEEDS:=0 1 2 3 4}"
: "${OUT_SUFFIX:=_${COHORT}}"     # the selections WIPE their target folder, so
                                  # never write into the default name.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p slurm_logs
[[ -f "$HERE/env.sh" ]] && source "$HERE/env.sh"
ACCOUNT_ARG=()
[[ -n "${FMRIDECOMP_ACCOUNT:-}" ]] && ACCOUNT_ARG=(--account="$FMRIDECOMP_ACCOUNT")

# The login node cannot import the package without the container or venv, and
# the pre-flight below is the only thing standing between a typo'd flag and a
# chain that refits every aperture. Pick the interpreter the way the sbatch
# scripts do, and refuse if it cannot import.
RUN=(python)
if [[ -n "${FMRIDECOMP_SIF:-}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5 2>/dev/null || true
  RUN=(apptainer exec --bind "${FMRIDECOMP_BINDS:-/project,/scratch,/home}" \
       --pwd "$PWD" "$FMRIDECOMP_SIF" python)
elif [[ -x "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/python" ]]; then
  RUN=("${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/python")
fi
if ! "${RUN[@]}" -c 'import fmri_decomposition.decompose' >/dev/null 2>&1; then
  echo "ERROR: cannot import fmri_decomposition with: ${RUN[*]}" >&2
  echo "       cp slurm/env.sh.example slurm/env.sh and set FMRIDECOMP_SIF" >&2
  exit 1
fi

# ------------------------------------------------------------- pre-flight ---
# Everything here is a footer read: no row group is touched and nothing is
# written. It costs seconds and it is the difference between a transfer test
# and an accidental refit.
echo "== pre-flight: $COHORT onto the fit trained on ${TRAIN}"
"${RUN[@]}" - "$ATLAS" "$COHORT" "$TRAIN" "$ALL_APERTURES" \
    "${DECOMPOSE_EXTRA:-}" <<'PY'
import sys
from pathlib import Path
from fmri_decomposition.decompose import latents_fit_description

atlas, cohort = sys.argv[1], sys.argv[2]
train, apertures = sys.argv[3].split(), sys.argv[4].split()
# The bridge, if it was already passed. A fit carrying a key no flag writes is
# only a problem while nothing restores it; saying so again once the bridge is
# there would make the check impossible to satisfy.
bridged = "--legacy-hash-from" in (sys.argv[5] if len(sys.argv) > 5 else "")
root = Path("outputs/latents")
problems, notes = [], []
for w in apertures:
    ref = root / f"atlas={atlas}" / f"window_s={w}" / f"cohort={train[0]}"
    hits = sorted(ref.rglob("data.parquet")) if ref.is_dir() else []
    if not hits:
        problems.append(f"window_s={w}: no latents for the training cohort "
                        f"{train[0]} -- there is no fit to project onto")
        continue
    meta, cols = latents_fit_description(hits[0])
    # The keys that no current flag writes. A fit carrying one of these cannot
    # be reproduced by any command line, only by --legacy-hash-from.
    legacy = [k for k in ("bins", "project_cohorts") if k in meta]
    notes.append(f"  window_s={w:>4}  model_hash {meta.get('model_hash')}  "
                 f"n_latents={meta.get('n_latents')}  "
                 f"{len(cols)} state set(s)"
                 + (f"  LEGACY KEYS {legacy}"
                    f"{' (bridged)' if bridged else ''}" if legacy else ""))
    if legacy and not bridged:
        problems.append(
            f"window_s={w}: the stored fit carries {legacy}, which the current "
            f"code no longer writes. Pass "
            f"--legacy-hash-from {hits[0]} in DECOMPOSE_EXTRA, or this run "
            f"refits and the state definitions change.")
print("\n".join(notes))
if problems:
    print("\nREFUSING:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
PY
echo "   pre-flight ok"

# ------------------------------------------------------------------ chain ---
N_AP=$(wc -w <<<"$APERTURES")
echo "== stage 4: projecting onto ${N_AP} dfc aperture(s)"
D_ID=$(sbatch ${ACCOUNT_ARG[@]+"${ACCOUNT_ARG[@]}"} --parsable \
  --array=0-$((N_AP - 1)) "$HERE/decomposition.sbatch" "$ATLAS" $APERTURES \
  -- --train $TRAIN --project $COHORT ${DECOMPOSE_EXTRA:-})
echo "   jobid ${D_ID}"

# afterok on an ARRAY needs every task to have succeeded, which is what we
# want: one aperture that refused is one aperture whose states are undefined,
# and clustering it would write columns nobody can compare.
# --kill-on-invalid-dep so a refusal CANCELS the rest instead of leaving five
# jobs pending forever with (DependencyNeverSatisfied).
echo "== stage 4b: labelling with the cached clusterers"
C_ID=$(sbatch ${ACCOUNT_ARG[@]+"${ACCOUNT_ARG[@]}"} --parsable \
  --dependency=afterok:"${D_ID}" --kill-on-invalid-dep=yes \
  "$HERE/clustering.sbatch" "$ATLAS" $APERTURES \
  -- --train $TRAIN --project $COHORT ${CLUSTER_EXTRA:-})
echo "   jobid ${C_ID}"

echo "== stage 5a: transitions, every aperture including -1"
T_ID=$(sbatch ${ACCOUNT_ARG[@]+"${ACCOUNT_ARG[@]}"} --parsable \
  --dependency=afterok:"${C_ID}" --kill-on-invalid-dep=yes \
  "$HERE/brain_states_transitions.sbatch" \
  -- --atlas "$ATLAS" --window-s $ALL_APERTURES)
echo "   jobid ${T_ID}"

echo "== stage 5b: the selections"
for cond in $CONDITIONS; do
  label="${cond%%=*}"; tasks="${cond#*=}"; tasks="${tasks//,/ }"
  for y in $TARGETS; do
    B=$(sbatch ${ACCOUNT_ARG[@]+"${ACCOUNT_ARG[@]}"} --parsable \
      --dependency=afterok:"${T_ID}" --kill-on-invalid-dep=yes \
      "$HERE/model_selection.sbatch" "$y" -- \
      --cohort "$COHORT" --task "$label" --keep-tasks $tasks \
      --atlas "$ATLAS" --window-s $ALL_APERTURES \
      --models $MODELS --seeds $SEEDS \
      --output-name "bstm_selection${OUT_SUFFIX}" \
      ${SEL_EXTRA[@]+"${SEL_EXTRA[@]}"})
    # fcm has NO aperture: one matrix per subject over every usable frame, the
    # static twin of window_s=-1. Gated on the transitions anyway, so that the
    # four tables of one condition appear together or not at all.
    F=$(sbatch ${ACCOUNT_ARG[@]+"${ACCOUNT_ARG[@]}"} --parsable \
      --dependency=afterok:"${T_ID}" --kill-on-invalid-dep=yes \
      "$HERE/fcm_selection.sbatch" "$y" -- \
      --cohorts "$COHORT" --task "$label" --keep-tasks $tasks \
      --atlas "$ATLAS" --models $MODELS --seeds $SEEDS \
      --output-name "fcm_selection${OUT_SUFFIX}" \
      ${SEL_EXTRA[@]+"${SEL_EXTRA[@]}"})
    echo "   ${label}/${y}: bstm ${B}  fcm ${F}"
  done
done

cat <<EOF

submitted. watch with:
  squeue -u \$USER -o '%.12i %.22j %.8T %.10M %R'

THE LINE THAT MATTERS, in each stage-4 log: it must say
  reusing the saved fit for model_hash ...
If it says 'loading training cohorts' instead, the hash diverged and it is
REFITTING -- cancel the chain, because every cohort downstream gets relabelled:
  grep -hE 'reusing the saved fit|loading training cohorts' slurm_logs/decompose_${D_ID}_*.out

and in the stage-4b log, one 'reusing the cached fit' per state set, with the
other cohorts listed as '(already at this fit_hash)':
  grep -cE 'reusing the cached fit' slurm_logs/cluster_${C_ID}.out

the four tables per condition, when it is done:
  outputs/bstm_selection${OUT_SUFFIX}/task=<cond>/target=<y>/summary.csv
  outputs/fcm_selection${OUT_SUFFIX}/ task=<cond>/target=<y>/summary.csv
EOF
