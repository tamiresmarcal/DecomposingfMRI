#!/bin/bash
# Stage 5a -- per-subject brain-state transition tables, one per state set.
#
#   sbatch slurm/brain_states_transitions.sbatch
#   sbatch slurm/brain_states_transitions.sbatch -- --atlas yeo7 --window-s -1
#
# NO ARGUMENTS NEEDED. Every default is the full grid, and the state columns are
# DISCOVERED from the latents schema rather than named -- a state set added by
# stage 4b is picked up without editing anything here. Anything after a bare
# `--` is forwarded verbatim.
#
# LOOK AT THE GRID FIRST, on a login node -- it writes nothing:
#
#   fmri-decomp transitions --check
#
# It reports every (atlas, aperture, state set, cohort) cell, and in particular
# whether `model_hash` is identical across cohorts in a cell. That check is the
# one that would otherwise fail silently: state 5 is the same state in two
# cohorts only if one fit defined it, and if camcan's latents came from a
# different `decompose` run than ds002837's then every transition compared
# across them is meaningless and nothing downstream would notice.
#
# COST is IO, not arithmetic: it reads a few columns per latents file and writes
# a ~100 KB table per cell. Minutes for a full grid, and the memory is trivial
# because the edges are never read. One task; there is nothing here worth an
# array.

#SBATCH --job-name=fmridecomp_transitions
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/transitions_%j.out
#SBATCH --error=slurm_logs/transitions_%j.err

set -euo pipefail

EXTRA=(); seen_sep=0
for a in "$@"; do
  if [[ ${seen_sep} -eq 1 ]]; then EXTRA+=("$a")
  elif [[ "$a" == "--" ]]; then seen_sep=1
  else echo "unexpected argument '$a' -- put extra args after a bare --" >&2
       exit 1; fi
done

_ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

# Stages 4 and 5 need the OTHER image -- see slurm/env.sh.example.
SIF="${FMRIDECOMP_SIF45:-${FMRIDECOMP_SIF:-}}"
DEPS="pandas pyarrow"

echo "host=$(hostname) cpus=${OMP_NUM_THREADS}"

if [[ -n "${SIF}" ]]; then
  module load apptainer 2>/dev/null || module load apptainer/1.3.5
  BINDS="${FMRIDECOMP_BINDS:-/project,/scratch,/home}"
  RUN=(apptainer exec --cleanenv --bind "${BINDS}" --pwd "${PWD}" \
       "${SIF}" python3)
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
    echo "ERROR: ${SIF} cannot import: ${MISSING}" >&2
    exit 1
  fi
else
  module load python/3.11 scipy-stack
  source "${FMRIDECOMP_VENV:-$HOME/venvs/fmridecomp}/bin/activate"
  RUN=(python3)
fi

"${RUN[@]}" -m fmri_decomposition.cli transitions ${EXTRA[@]+"${EXTRA[@]}"}

echo "transitions done"
