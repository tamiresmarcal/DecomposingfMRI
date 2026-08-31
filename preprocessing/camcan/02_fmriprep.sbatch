#!/bin/bash
# Stage 1 for Cam-CAN -- fMRIPrep, one SLURM array task per subject.
#
#   sbatch --array=0-9 preprocessing/camcan/02_fmriprep.sbatch
#
# The array index selects a line of $BIDS_ROOT/subjects.txt, which
# 01_build_bids.py wrote sorted. Line N is the same subject on every
# submission, so a re-submitted array after a walltime kill maps to the same
# subjects as the first attempt.
#
# NOT part of the pipeline. See preprocessing/README.md.

#SBATCH --job-name=camcan_fmriprep
#SBATCH --account=rpp-aevans-ab
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/camcan_fmriprep_%A_%a.out
#SBATCH --error=slurm_logs/camcan_fmriprep_%A_%a.err

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/project/6008063/tamires/cohorts/camcan_bids}"
OUT_ROOT="${OUT_ROOT:-/project/6008063/tamires/cohorts/camcan_fmriprep}"
WORK_ROOT="${WORK_ROOT:-/project/6008063/tamires/work/camcan_fmriprep}"

# ---------------------------------------------------------------- prereqs ---
# All of these are checked BEFORE the module loads, because every one of them
# fails late and confusingly otherwise. A wrong licence path costs you a queue
# wait; a missing TemplateFlow cache costs you the whole walltime.

if [[ ! -f "$BIDS_ROOT/subjects.txt" ]]; then
  echo "ERROR: no $BIDS_ROOT/subjects.txt" >&2
  echo "       Run 01_build_bids.py on a login node first." >&2
  exit 1
fi

# FreeSurfer LICENCE, not FreeSurfer. fMRIPrep carries its own copy of the
# software; what it needs is the four-line text file. It checks at startup and
# refuses to run without one -- even with --fs-no-reconall, because skipping
# recon-all skips surface reconstruction, not the FreeSurfer binaries used
# elsewhere in the anatomical workflow.
FS_LICENSE="${FS_LICENSE:-/project/6008063/tamires/freesurfer_license.txt}"
if [[ ! -f "$FS_LICENSE" ]]; then
  echo "ERROR: no FreeSurfer licence at $FS_LICENSE" >&2
  echo "       Copy your lab's \$FREESURFER_HOME/license.txt there, or" >&2
  echo "       register free at https://surfer.nmr.mgh.harvard.edu/registration.html" >&2
  echo "       Then: export FS_LICENSE=/path/to/license.txt" >&2
  exit 1
fi
export FS_LICENSE

# Compute nodes have NO outbound network. fMRIPrep downloads its normalisation
# templates through TemplateFlow at run time; with no cache and no network it
# does not fail fast, it HANGS on an HTTP timeout and dies hours in. Same trap
# as NILEARN_DATA in slurm/01_extract.sbatch.
export TEMPLATEFLOW_HOME="${TEMPLATEFLOW_HOME:-/project/6008063/tamires/templateflow}"
# Check for a real IMAGE, not just the directory. TemplateFlow lays down the
# whole tpl-* skeleton on first import and downloads the actual files lazily,
# so `-d tpl-MNI152NLin2009cAsym` passes on a cache that contains no data at
# all -- and then the job hangs on the download this guard exists to prevent.
if ! find "$TEMPLATEFLOW_HOME/tpl-MNI152NLin2009cAsym" \
        -name '*_T1w.nii.gz' -size +1M -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: no downloaded template images under $TEMPLATEFLOW_HOME" >&2
  echo "       (the tpl-* directories may exist -- TemplateFlow creates the" >&2
  echo "        skeleton on import and fetches files lazily. What is missing" >&2
  echo "        is the data.)" >&2
  echo "       Run 00_prefetch_templateflow.sh on a LOGIN node, and check it" >&2
  echo "       reports a size of several GB at the end." >&2
  exit 1
fi

# --------------------------------------------------------------- subject ---
IDX="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"
N_SUBS=$(wc -l < "$BIDS_ROOT/subjects.txt")
if (( IDX >= N_SUBS )); then
  echo "array index $IDX is beyond the $N_SUBS subject(s) in subjects.txt" >&2
  exit 1
fi
SUB=$(sed -n "$((IDX + 1))p" "$BIDS_ROOT/subjects.txt")
echo "array task $IDX/$((N_SUBS - 1))  ->  sub-$SUB"

# One work dir per subject. Cleaned on success, KEPT on failure so the crash
# is debuggable. This cleanup is not optional housekeeping: a single subject's
# nipype work dir holds tens of thousands of files, so 649 uncleaned subjects
# would be ~13-39M files against a 30M inode quota that is already 17M used.
WORK="$WORK_ROOT/sub-$SUB"
mkdir -p "$WORK" "$OUT_ROOT" slurm_logs

# Keep matplotlib's cache off $HOME -- it is small-file heavy and quota-limited.
export MPLCONFIGDIR="$WORK/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

module load StdEnv/2023 fmriprep/25.1.1

# fMRIPrep wants megabytes, and wants slightly less than the allocation: if it
# sizes its own scheduler to the full request it gets OOM-killed by SLURM at
# the peak instead of backing off. 90% leaves room for the interpreter itself.
if [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then
  MEM_MB="$SLURM_MEM_PER_NODE"
else
  MEM_MB=$(( ${SLURM_MEM_PER_CPU:-8192} * ${SLURM_CPUS_PER_TASK:-8} ))
fi
MEM_MB=$(( MEM_MB * 9 / 10 ))
NCPU="${SLURM_CPUS_PER_TASK:-8}"

echo "cpus=$NCPU  mem=${MEM_MB}MB  work=$WORK"
echo "started $(date -Is)"

set +e
fmriprep "$BIDS_ROOT" "$OUT_ROOT" participant \
  --participant-label "$SUB" \
  --task-id Movie \
  -w "$WORK" \
  --fs-license-file "$FS_LICENSE" \
  --nprocs "$NCPU" --omp-nthreads 4 --mem-mb "$MEM_MB" \
  --output-spaces MNI152NLin2009cAsym:res-2 \
  --fs-no-reconall \
  --use-syn-sdc \
  --notrack \
  --stop-on-first-crash
STATUS=$?
set -e

echo "finished $(date -Is) with status $STATUS"

# WHY EACH FLAG, since every one of them is a decision someone will want to
# revisit rather than a magic incantation:
#
#   --task-id Movie        Cam-CAN's TR DIFFERS BY TASK -- movie 2.47 s,
#                          sensorimotor 1.97 s. The pipeline config carries one
#                          `tr:`, so mixing tasks would silently rescale every
#                          time column. Rest is also present in the archive but
#                          is not naturalistic viewing, so it is not built here.
#   --fs-no-reconall       We need volumetric parcels in MNI, never surfaces.
#                          This is roughly 3 h/subject instead of 12.
#   --output-spaces        MNI152NLin2009cAsym:res-2 -- the SAME space
#     MNI152NLin2009cAsym  ds002837 and CNeuroMod already use, so one atlas
#     :res-2               serves all three cohorts and `space:` in the config
#                          is literally true rather than approximately true.
#   --use-syn-sdc          Cam-CAN shipped NO fieldmaps (verified: no `fmap`
#                          directory anywhere in the archive), so distortion
#                          correction has to be estimated from the T1w with the
#                          warp constrained to the phase-encode direction. The
#                          sidecar has PhaseEncodingDirection j- and
#                          TotalReadoutTime 0.0105 s, so this will run.
#                          It is better than nothing and worse than a real
#                          fieldmap. CNeuroMod HAS fieldmaps, so Cam-CAN is the
#                          one cohort without measured distortion correction --
#                          that belongs in the config comments, not just here.
#   --notrack              fMRIPrep phones home for usage stats by default.
#                          No outbound network here; without this it waits.
#   --stop-on-first-crash  Pilot setting. A partial subject that "succeeded"
#                          is worse than a failed one, because it reaches the
#                          pipeline looking complete.

if (( STATUS == 0 )); then
  echo "cleaning work dir $WORK"
  rm -rf "$WORK"
else
  echo "FAILED -- work dir kept for debugging: $WORK" >&2
  echo "  crash reports: $OUT_ROOT/sub-$SUB/log/" >&2
fi
exit "$STATUS"
