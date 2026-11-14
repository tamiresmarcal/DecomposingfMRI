#!/bin/bash
# Yeo 7 x BrainMap matrix (notebook 08, section 4) with the Burt null.
#
#   sbatch slurm/reference_matrix.sbatch
#   sbatch slurm/reference_matrix.sbatch -- --behavioural neurosynth_zreverse --n-nulls 10000
#
# The matrix compares reference maps with each other, so it does not depend on
# which state set is reported: it can be finalised now. Same call as the
# notebook (cortical parcels only whenever Yeo 7 is a side), but with
# null="burt" (variogram matching, Burt et al. 2020, brainsmash) instead of
# Moran. Only the stars change: r is the same.
#
# COST: brainsmash refits a variogram per surrogate, ~5 min per map for 10,000
# surrogates at ~100 parcels; ~45 behavioural maps -> about 4 h. Hence a job.
#
# NEEDS brainsmash in the pylibs folder the notebooks use, once:
#   apptainer exec $SIF python -m pip install --no-deps \
#       --target /project/6008063/tamires/singularity/pylibs brainsmash
#
# Output: reference_matrix_burt/reference_matrix_<atlas>_yeo7_x_behavioural_<...>_cortical.{png,csv}
# in the submit directory (a separate folder, so the Moran version is kept).

#SBATCH --job-name=fmridecomp_refmatrix_burt
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/refmatrix_burt_%j.out
#SBATCH --error=slurm_logs/refmatrix_burt_%j.err

set -euo pipefail

ATLAS=harvardoxford
BEHAVIOURAL=neurosynth_zreverse
TAXONOMY=brainmap
N_NULLS=10000
OUT=reference_matrix_burt
seen_sep=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  a="${args[$i]}"
  if [[ $seen_sep -eq 0 ]]; then
    [[ "$a" == "--" ]] && { seen_sep=1; continue; }
    echo "unexpected argument '$a' -- put options after a bare --" >&2; exit 1
  fi
  case "$a" in
    --atlas) ATLAS="${args[++i]}" ;;
    --behavioural) BEHAVIOURAL="${args[++i]}" ;;
    --taxonomy) TAXONOMY="${args[++i]}" ;;
    --n-nulls) N_NULLS="${args[++i]}" ;;
    --out) OUT="${args[++i]}" ;;
    *) echo "unknown option '$a'" >&2; exit 1 ;;
  esac
done

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
_ENV_SH="$REPO/slurm/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

# The image the notebooks use, plus the pylibs folder (nimare, nilearn,
# brainsmash) and the repo on the path.
SIF="${SIF:-${FMRIDECOMP_SIF_SHAP:-/project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif}}"
PYLIBS="${PYLIBS:-/project/6008063/tamires/singularity/pylibs}"
export APPTAINERENV_PYTHONPATH="$REPO:$PYLIBS"
module load apptainer 2>/dev/null || module load apptainer/1.3.5

echo "host=$(hostname) cpus=$OMP_NUM_THREADS atlas=$ATLAS behavioural=$BEHAVIOURAL" \
     "taxonomy=$TAXONOMY n_nulls=$N_NULLS null=burt out=$OUT"
cd "$REPO"
apptainer exec --bind "$REPO" ${SCRATCH:+--bind "$SCRATCH"} "$SIF" python - <<EOF
import brainsmash  # fail now, not after loading the maps
from pathlib import Path
from fmri_decomposition.figures import states as sf
png = sf.reference_matrix("$ATLAS", "yeo7", "behavioural", out=Path("$OUT"),
                          taxonomy="$TAXONOMY", behavioural_csv="$BEHAVIOURAL",
                          n_nulls=$N_NULLS, null="burt")
print("wrote", png)
EOF
