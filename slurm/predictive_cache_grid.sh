#!/bin/bash
# Every model, window and atlas for one state set, as a job ARRAY (one combination per task, in
# parallel): the SHAP caches of notebooks 06 / 07 (tools/predictive_cache.py --grid).
#
#   ./slurm/predictive_cache_grid.sh                     # HMM2_umap3_8
#   ./slurm/predictive_cache_grid.sh HMM2_pca3_8
#
# Lists the combinations first (in the container, a few seconds), then submits the array.
# Left out: the four already submitted by hand (hgb · all [cond] · harvardoxford/yeo7 · 30/-1);
# set EXCLUDE="" to include them. Cached combinations are read, not refitted, so resubmitting is safe.
set -euo pipefail
STATES="${1:-HMM2_umap3_8}"
EXCLUDE="${EXCLUDE-harvardoxford,30,hgb,all [cond];harvardoxford,-1,hgb,all [cond];yeo7,30,hgb,all [cond];yeo7,-1,hgb,all [cond]}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "$REPO/slurm/env.sh" ]] && source "$REPO/slurm/env.sh"
SIF="${SIF:-${FMRIDECOMP_SIF_SHAP:-/project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif}}"
PYLIBS="${PYLIBS:-/project/6008063/tamires/singularity/pylibs}"
module load apptainer 2>/dev/null || module load apptainer/1.3.5
IFS=';' read -r -a EX <<< "$EXCLUDE"
EXARGS=(); [[ ${#EX[@]} -gt 0 && -n "${EX[0]}" ]] && EXARGS=(--exclude "${EX[@]}")
cd "$REPO"
LIST=$(APPTAINERENV_PYTHONPATH="$REPO:$PYLIBS" apptainer exec --bind "$REPO" ${FMRIDECOMP_OUTPUTS:+--bind "$FMRIDECOMP_OUTPUTS"} \
       "$SIF" python tools/predictive_cache.py --grid "$STATES" --list "${EXARGS[@]}")
echo "$LIST"
N=$(echo "$LIST" | sed -n 's/^N=//p')
[[ "$N" -gt 0 ]] || { echo "nothing to run"; exit 0; }
mkdir -p slurm_logs
sbatch --array=0-$((N - 1)) --job-name=fmridecomp_predcache_grid --time=12:00:00 --cpus-per-task=16 --mem=64G \
       --output=slurm_logs/predcache_grid_%A_%a.out --error=slurm_logs/predcache_grid_%A_%a.err \
       slurm/predictive_cache.sbatch -- --grid "$STATES" --task-id-from-array "${EXARGS[@]}"
