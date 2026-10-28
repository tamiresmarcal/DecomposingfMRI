#!/bin/bash
# Stage 4b -- add brain-state definitions to latents that already exist.
#
#   sbatch slurm/clustering.sbatch harvardoxford 30 60 120 300
#   sbatch slurm/clustering.sbatch yeo7 -1
#   sbatch slurm/clustering.sbatch yeo7 30 -- --methods hmm1 --k 8
#   sbatch slurm/clustering.sbatch networks -1 -- --project camcan_rest
#
# LOOK AT THE GRID FIRST, on a login node -- it fits nothing and takes seconds:
#
#   fmri-decomp cluster --check --atlas harvardoxford yeo7 networks \
#                       --window-s 15 30 60 120 300 -1
#
# It prints, per (atlas, aperture), which cohorts and embeddings are there, how
# many state columns a re-run would REPLACE, and what would be written. The HMM
# is the cost here and it is hours at K=27 across a full grid; finding a missing
# embedding after three of those have run is the expensive way to find out.
#
# ONE WRITER PER CELL -- and a cell is an (atlas, APERTURE), not an atlas.
# Every clusterer for one cell is fitted on the same embedding block and the
# labels are appended to the SAME parquet file, which parquet cannot do in
# place: the file is read, rewritten and renamed. Two jobs on one cell would
# race -- each reads, each adds its column, and the one that renames second
# wins, so the other's columns are gone.
#
# Different APERTURES are different files, so splitting by aperture is safe and
# is usually what you want -- fifteen one-cell jobs schedule faster than three
# long ones and finish in about a fifth of the wall-clock:
#
#   for A in harvardoxford yeo7 networks; do
#     for W in 30 60 120 300 -1; do
#       sbatch slurm/clustering.sbatch $A $W -- <args>
#     done
#   done
#
# What you must NOT do is split one cell by METHOD -- an hmm2-only job beside
# an hmm1 job on the same (atlas, aperture) is exactly the race above.
#
# COST is the clusterers, and they are not alike. Threshold is a quantile
# transform (seconds). MeanShift is O(n_fit^2) in the bandwidth search, capped
# by --meanshift-fit-rows. The HMMs are Baum-Welch over the full sequence and
# dominate the walltime.
#
# HMM2 IS THE EXPENSIVE ONE, BY TWO ORDERS OF MAGNITUDE. It is full covariance
# (parameters grow with the square of the dimension, not linearly), 500
# iterations against hmm1's 50, and --hmm2-restarts FULL FITS per state set
# rather than one. 15 restarts x 10 iterations is a >150x multiplier on the
# per-fit cost before covariance is counted.
#
# So MEASURE BEFORE COMMITTING TO THE GRID. One cheap cell tells you what the
# walltime has to be:
#
#   sbatch slurm/clustering.sbatch yeo7 30 -- --methods hmm2 --k 10 \
#          --hmm2-restarts 5
#
# then scale: cost is roughly linear in --hmm2-restarts and in --hmm2-iter, and
# steeply worse in K and in the embedding's width. If it times out, lower
# --hmm2-restarts first (5 still beats 1, which is what hmm1 does), and raise
# #SBATCH --time rather than cutting --hmm2-iter -- 500 is the paper's number
# and the log says `HIT THE 500-ITER CAP` when it was not enough.
#
# MEMORY is small: only the embedding columns are ever loaded, never the edges
# -- three of them for pca3/umap3, or the atlas's feature count for a raw<N>
# embedding (7, 14). Even Harvard-Oxford at 15 s is a few hundred MB.
#
# THE RAW ARM IS RAGGED ON PURPOSE. `raw<N>` exists only where
# `decompose --passthrough-features` ran, which is the activation aperture on
# the small atlases, so asking for it everywhere is fine: a cell without those
# columns SKIPS that embedding and says so, and the job still exits 0. Only a
# state set that was asked for and could not be written -- a degenerate K, or no
# restart expressing all K states -- is a refusal, and that exits non-zero.
#
# WHAT GETS WRITTEN. Label columns are APPENDED to
# latents/atlas=<a>/window_s=<w>/cohort=<c>/data.parquet, with per-column
# provenance in the schema metadata. The embeddings are not refitted, so this is
# safe to re-run when a new clusterer is added -- it does not redo the ones that
# already work.

#SBATCH --job-name=fmridecomp_cluster
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/cluster_%j.out
#SBATCH --error=slurm_logs/cluster_%j.err

set -euo pipefail

ATLAS="${1:?usage: sbatch clustering.sbatch <atlas> <window_s> [...] [-- extra args]}"
shift

# Window sizes up to a bare `--`; anything after it is forwarded verbatim.
WINDOWS=(); EXTRA=(); seen_sep=0
for a in "$@"; do
  if [[ ${seen_sep} -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen_sep=1
  else WINDOWS+=("$a"); fi
done
# A WINDOW NEVER STARTS WITH `--`. Forgetting the bare `--` separator does not
# fail loudly: the forwarded flags land in WINDOWS, argparse's nargs="+" stops
# at the first one anyway, and the run proceeds with a WINDOWS array that is
# wrong -- which makes the array-range check and the task log wrong with it.
# `-1` is a real window and does not start with `--`, so this is exact.
for w in ${WINDOWS[@]+"${WINDOWS[@]}"}; do
  if [[ "$w" == --* ]]; then
    echo "ERROR: '$w' is not a window size." >&2
    echo "       Extra arguments go after a bare --, e.g." >&2
    echo "         sbatch $(basename "$0") <atlas> <window>... -- $w ..." >&2
    exit 1
  fi
done

if [[ ${#WINDOWS[@]} -eq 0 ]]; then
  echo "give at least one window size (-1 is the activation aperture)" >&2
  exit 1
fi

_ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

# BLAS GETS ONE THREAD, ON PURPOSE. This used to hand it every core, on the
# reasoning that hmmlearn and MeanShift both go through BLAS so one big pool
# beats several small ones. For hmmlearn that reasoning was simply wrong:
# forward-backward is sequential over time and cannot be split, so one fit on
# eight threads is no faster than on one -- measured at 4.37 s against 4.70 s,
# i.e. slightly WORSE from contention.
#
# So the cores go to hmm2's restarts instead, which are independent fits, via
# --hmm2-jobs below. Two settings have to agree for that to work: BLAS at one
# thread here, and `inner_max_num_threads=1` on the joblib backend. Without
# both, eight workers each start an eight-thread pool on eight cores.
#
# MeanShift is unaffected: its `n_jobs=-1` is joblib over seeds, not BLAS.
#
# (fmri_decomposition/cli.py already setdefault()s these to 1. An `export`
# here BEATS a setdefault there, which is how a run ended up with eight idle
# BLAS threads AND sequential restarts -- the worst of both.)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "host=$(hostname) atlas=${ATLAS} windows=${WINDOWS[*]} cpus=${SLURM_CPUS_PER_TASK:-1} (BLAS threads 1; the cores go to --hmm2-jobs)"

# Stages 4 and 5 need the OTHER image. containers/README.md: stages 2-3 run in
# fmri_decomp.sif, stages 4-5 in fmri_decomp_stage45.sif, which is where Stan,
# hmmlearn, umap-learn and lightgbm live. One FMRIDECOMP_SIF for both meant
# either editing slurm/env.sh between stages or watching this job die on
# `ModuleNotFoundError: hmmlearn` minutes in, so FMRIDECOMP_SIF45 wins here and
# FMRIDECOMP_SIF is the fallback for anyone who keeps one image.
SIF="${FMRIDECOMP_SIF45:-${FMRIDECOMP_SIF:-}}"
DEPS="sklearn"

if [[ -n "${SIF}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  RUN=(apptainer exec --cleanenv --bind "${BINDS}" --pwd "${PWD}" \
       "${SIF}" python3)
  # Fail in seconds rather than after the first fit, and name the module that is
  # ACTUALLY missing. Printing the whole list instead sent a real diagnosis
  # astray: the image had sklearn and lacked lightgbm, and the message said
  # "cannot import: sklearn, lightgbm", which reads as the wrong image entirely.
  #
  # Only HARD requirements belong here. An optional dependency -- lightgbm for
  # --models lgbm, hmmlearn for --methods hmm, umap for --embeddings umap3 -- is
  # already handled where it is used, with a message that knows which flag asked
  # for it. Demanding it here is stricter than the code it guards, and blocked a
  # default `select` run that needed neither.
  # `import importlib.util` explicitly: a bare `import importlib` does NOT make
  # `.util` an attribute, so the block raised AttributeError, 2>/dev/null ate it,
  # MISSING came back empty and the guard passed everything. A guard that fails
  # open is worse than no guard, so the exit status is checked too.
  # The module names go in as ARGV, not through the environment. RUN carries
  # --cleanenv -- deliberately, to keep the host's PYTHONPATH out of the
  # container -- and --cleanenv strips exported variables too, so os.environ
  # ["DEPS"] raised KeyError inside the container, python exited non-zero, and
  # this guard killed every job in 10 seconds. A check that cannot run is not a
  # check; argv crosses the boundary that the environment does not.
  MISSING=$("${RUN[@]}" -c 'import importlib.util, sys
print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))' \
            ${DEPS}) || {
    echo "ERROR: ${SIF} -- could not run python3 inside it to check imports" >&2
    exit 1; }
  # IS THIS IMAGE BUILT FROM THE CURRENT DEFINITION? A warning, not a failure:
  # most changes to the definition do not affect a given run, and refusing to
  # start would be worse than saying so. But it is the check that was missing --
  # the image on disk was built from a definition three commits old, and the
  # lightgbm added in the newest of them went unnoticed until a job died for it.
  _DEF="${SLURM_SUBMIT_DIR:-$PWD}/containers/stage45.def"
  if [[ -f "${_DEF}" ]]; then
    _WANT=$(sha256sum "${_DEF}" | cut -d" " -f1)
    # Through RUN as it stands (it ends in python3) rather than trying to swap
    # the interpreter out of the array: `${RUN[@]%python3}` strips the suffix from
    # EVERY element and leaves an empty string where python3 was, so the command
    # it builds is unrunnable.
    # `|| true` because this is a WARNING and must never be able to kill a job.
    # The script runs under `set -e`, so a non-zero exit from a purely
    # informational check would abort the run -- which is precisely how the
    # import guard above took out three jobs in ten seconds.
    _HAVE=$("${RUN[@]}" -c 'import pathlib,sys
p = pathlib.Path("/opt/def_sha256")
sys.stdout.write(p.read_text().strip() if p.exists() else "")' 2>/dev/null || true)
    if [[ -z "${_HAVE}" ]]; then
      echo "NOTE: ${SIF} records no def_sha256, so it predates provenance" >&2
      echo "      being stamped into the image. It may be missing packages the" >&2
      echo "      current containers/stage45.def installs." >&2
    elif [[ "${_HAVE}" != "${_WANT}" ]]; then
      echo "NOTE: ${SIF} was built from a DIFFERENT containers/stage45.def" >&2
      echo "      image: ${_HAVE}" >&2
      echo "      repo : ${_WANT}" >&2
      echo "      Rebuild per containers/README.md if a run needs a package the" >&2
      echo "      definition has gained since." >&2
    fi
  fi

  if [[ -n "${MISSING// /}" ]]; then
    echo "ERROR: ${SIF}" >&2
    echo "       cannot import: ${MISSING}" >&2
    echo "       Those are required here. Either this is the stage 2/3 image --" >&2
    echo "         export FMRIDECOMP_SIF45=<the stage45 .sif>" >&2
    echo "       -- or the stage45 image needs rebuilding per containers/README.md." >&2
    exit 1
  fi
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python3)
fi

# --hmm2-jobs goes FIRST so a caller passing their own after `--` overrides it:
# argparse takes the last occurrence. The default inside the CLI is
# SLURM_CPUS_PER_TASK anyway, so this is belt and braces for a hand-run that
# exports the variable differently.
"${RUN[@]}" -m fmri_decomposition.cli cluster \
    --hmm2-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --atlas "${ATLAS}" --window-s "${WINDOWS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}

echo "atlas=${ATLAS} done"
