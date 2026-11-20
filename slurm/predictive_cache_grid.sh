#!/bin/bash
# Every combination select-bstm ran -- atlas x window x state set x model x arm (graph_arms) -- the
# SHAP caches of notebooks 06 / 07 (tools/predictive_cache.py --grid), ONE ARRAY TASK PER
# few combinations, all in parallel (at most MAXPAR at a time). NTASKS caps the array (default 300:
# clusters limit how many jobs one user may submit, and every array task counts as one), so each
# task fits ~N/NTASKS combinations in turn.
#
#   ./slurm/predictive_cache_grid.sh              # every state set (ALL)
#   ./slurm/predictive_cache_grid.sh HMM2_umap3_8 # one state set
#   MAXPAR=50 ./slurm/predictive_cache_grid.sh    # fewer at once
#   NTASKS=150 ./slurm/predictive_cache_grid.sh   # a smaller array, more combinations per task
#   ARMS="all [joint]" ./slurm/predictive_cache_grid.sh   # one arm only (the transfer's "all" is [joint])
#
# Lists the combinations first (in the container, a few seconds), then submits the array. A
# combination already cached is read, not refitted, so resubmitting the whole grid is safe and cheap.
# EXCLUDE: "atlas,window_s,model,arm" or "atlas,window_s,states,model,arm", ';'-separated.
set -euo pipefail
STATES="${1:-ALL}"
EXCLUDE="${EXCLUDE-}"
MAXPAR="${MAXPAR:-100}"
NTASKS="${NTASKS:-300}"
ARMS="${ARMS-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "$REPO/slurm/env.sh" ]] && source "$REPO/slurm/env.sh"
SIF="${SIF:-${FMRIDECOMP_SIF_SHAP:-/project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif}}"
PYLIBS="${PYLIBS:-/project/6008063/tamires/singularity/pylibs}"
module load apptainer 2>/dev/null || module load apptainer/1.3.5
IFS=';' read -r -a EX <<< "$EXCLUDE"
EXARGS=(); [[ ${#EX[@]} -gt 0 && -n "${EX[0]}" ]] && EXARGS=(--exclude "${EX[@]}")
IFS=';' read -r -a AR <<< "$ARMS"
[[ ${#AR[@]} -gt 0 && -n "${AR[0]}" ]] && EXARGS+=(--grid-arms "${AR[@]}")
cd "$REPO"
LIST=$(APPTAINERENV_PYTHONPATH="$REPO:$PYLIBS" apptainer exec --bind "$REPO" ${FMRIDECOMP_OUTPUTS:+--bind "$FMRIDECOMP_OUTPUTS"} \
       "$SIF" python tools/predictive_cache.py --grid "$STATES" --list ${EXARGS[@]+"${EXARGS[@]}"})
echo "$LIST" | tail -25
N=$(echo "$LIST" | sed -n 's/^N=//p')
[[ "$N" -gt 0 ]] || { echo "nothing to run"; exit 0; }
mkdir -p slurm_logs
# Every task recomputes the same sorted list and takes rows i, i + T, i + 2T, ... (T = tasks).
T=$(( N < NTASKS ? N : NTASKS ))
echo "${N} combinations over ${T} array tasks (~$(( (N + T - 1) / T )) each), ${MAXPAR} at a time"
sbatch --array="0-$((T - 1))%${MAXPAR}" --job-name="predcache_grid" --time=24:00:00 --cpus-per-task=16 --mem=64G \
       --output="slurm_logs/predcache_grid_%A_%a.out" --error="slurm_logs/predcache_grid_%A_%a.err" \
       slurm/predictive_cache.sbatch -- --grid "$STATES" --task-id-from-array --n-tasks "$T" --fc-arm edges \
       --skip shares contrib ${EXARGS[@]+"${EXARGS[@]}"}
