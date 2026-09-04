#!/usr/bin/env bash
# Fetch season 1 of CNeuroMod `friends` fMRIPrep derivatives -- LOGIN NODE.
#
#   ./preprocessing/cneuromod/fetch_friends_s01.sh --dry-run   # size, no download
#   ./preprocessing/cneuromod/fetch_friends_s01.sh             # ~345 GiB
#
# Compute nodes have no outbound network and the container has no datalad, so
# this runs on a login node, outside any job. It is resumable: git-annex skips
# anything already present, so re-running after a dropped connection costs a
# listing pass and nothing else.
#
# WHAT IT FETCHES, AND WHY SO LITTLE
# ----------------------------------
# Exactly the three files per run that config/cneuromod_friends.yaml reads:
#
#   *_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz   the image
#   *_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz     discovery.mask_glob
#   *_desc-confounds_timeseries.tsv                        confounds.confounds_glob
#
# `datalad get sub-01` would also pull space-T1w BOLD, the fsLR CIFTI, aseg and
# aparcaseg segmentations, boldrefs and every report figure -- several times the
# volume, none of it read by this pipeline.
#
# THE SIZE IS NOT A TYPO. ~345 GiB for 288 runs, ~1.5 GB per run. These
# derivatives were written with `--output-spaces MNI152NLin2009cAsym` and no
# res- specifier, so fMRIPrep 20.2.6 emitted them on the template's own 1 mm
# grid rather than a 2 mm one. Check your project quota BEFORE starting:
#   diskusage_report
#
# ACCESS. Content lives in an S3 bucket at s3.unf-montreal.ca that autoenables
# on `datalad get`, and reading it needs the credentials CNeuroMod issues after
# the data transfer agreement (https://docs.cneuromod.ca/en/latest/ACCESS.html).
# Export them before running, or put them in ~/.aws/credentials:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
# A wrong or missing key shows up as every file failing with the same error,
# not as a partial fetch.
#
# ONE UPSTREAM CORRECTION TO BE AWARE OF. sub-01's season-1 episodes were once
# mislabelled: what is now s01e01 under ses-003 was published as s01e06, and
# vice versa (friends.fmriprep commit "rename run for stimuli mistake"). A
# clone made before that fix has the two episodes swapped for sub-01 and
# nothing downstream can detect it. `datalad update` below is what stops you
# analysing the old labels; do not skip it on an existing clone.

set -euo pipefail

# --- where the clone lives ------------------------------------------------
# Kept identical to derivatives_root in config/cneuromod_friends.yaml. Override
# for another site with FRIENDS_FMRIPREP=/somewhere/else.
DERIV="${FRIENDS_FMRIPREP:-/project/6008063/tamires/cohorts/cneuromod/friends.fmriprep}"
SUBJECTS=(01 02 03 04 05 06)
SEASON="${FRIENDS_SEASON:-s01}"
JOBS="${FRIENDS_FETCH_JOBS:-6}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

command -v datalad >/dev/null || {
  echo "datalad not on PATH. On Alliance:" >&2
  echo "  module load python/3.11 git-annex && pip install --user datalad" >&2
  exit 1
}

# --- install or update ----------------------------------------------------
if [[ ! -d "$DERIV/.git" ]]; then
  echo "== installing friends.fmriprep into $DERIV"
  mkdir -p "$(dirname "$DERIV")"
  # No -r: sub-01..06 live in THIS dataset. The only subdatasets are
  # sourcedata/ and containers/, which are provenance, not data we read.
  datalad install -s https://github.com/courtois-neuromod/friends.fmriprep "$DERIV"
else
  echo "== updating existing clone at $DERIV"
  datalad update --how ff-only -d "$DERIV"
fi

# --- build the file list --------------------------------------------------
# Globbed from the clone rather than hardcoded: every file exists as an annex
# symlink whether or not its content has been fetched, and which SESSION an
# episode was watched in differs per subject, so a hardcoded ses- path would be
# wrong for someone.
mapfile -t WANTED < <(
  for sub in "${SUBJECTS[@]}"; do
    find "$DERIV/sub-$sub" -path "*/func/*" \( \
         -name "*_task-${SEASON}e*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" -o \
         -name "*_task-${SEASON}e*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" -o \
         -name "*_task-${SEASON}e*_desc-confounds_timeseries.tsv" \)
  done | sort
)

if (( ${#WANTED[@]} == 0 )); then
  echo "no ${SEASON} files found under $DERIV -- is the clone empty, or the season wrong?" >&2
  exit 1
fi

# git-annex encodes the byte size in the object key, so the total is exact and
# costs no network. An already-fetched file is a real file, not a symlink.
python3 - "$DRY_RUN" "${WANTED[@]}" <<'PY'
import os, re, sys
dry = sys.argv[1] == "1"
paths = sys.argv[2:]
key = re.compile(r'-s(\d+)--')
todo = have = 0
n_todo = n_have = 0
for p in paths:
    if os.path.islink(p):
        m = key.search(os.path.basename(os.readlink(p)))
        size = int(m.group(1)) if m else 0
        if os.path.exists(p):            # symlink resolves -> content present
            have += size; n_have += 1
        else:
            todo += size; n_todo += 1
    else:
        have += os.path.getsize(p); n_have += 1
G = 1024 ** 3
print(f"   {len(paths)} file(s) wanted")
print(f"   already present: {n_have:5d}  {have / G:8.1f} GiB")
print(f"   to fetch:        {n_todo:5d}  {todo / G:8.1f} GiB")
if dry:
    print("\n   --dry-run: nothing downloaded.")
PY

(( DRY_RUN )) && exit 0

# --- fetch ----------------------------------------------------------------
# In chunks so a failure names a bounded set of files, and so the progress line
# moves. -J parallelises within a chunk; more than ~6 mostly queues on the S3
# endpoint rather than going faster.
echo "== fetching with -J ${JOBS} (resumable: rerun after any interruption)"
CHUNK=24
for (( i = 0; i < ${#WANTED[@]}; i += CHUNK )); do
  echo "-- files $((i + 1))..$(( i + CHUNK < ${#WANTED[@]} ? i + CHUNK : ${#WANTED[@]} )) of ${#WANTED[@]}"
  datalad get -d "$DERIV" -J "$JOBS" "${WANTED[@]:i:CHUNK}"
done

# --- verify ---------------------------------------------------------------
# A dangling symlink is the failure mode this whole script exists to avoid: it
# is not an error until stage 2 opens it, an hour into a job array.
MISSING=0
for f in "${WANTED[@]}"; do [[ -e "$f" ]] || { echo "MISSING: $f"; MISSING=$((MISSING + 1)); }; done
if (( MISSING )); then
  echo
  echo "${MISSING} file(s) still unfetched. Rerun this script -- it resumes." >&2
  exit 1
fi

echo
echo "all ${#WANTED[@]} file(s) present. Next:"
echo "  python tools/check_cohort.py config/cneuromod_friends.yaml"
