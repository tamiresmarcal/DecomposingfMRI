#!/bin/bash
# RUN THIS ON A LOGIN NODE. It needs the network.
#
#   bash preprocessing/camcan/00_prefetch_templateflow.sh
#
# WHY THIS EXISTS
# ---------------
# fMRIPrep does not ship the templates it normalises to. It downloads them at
# run time through TemplateFlow, into $TEMPLATEFLOW_HOME. Compute nodes on this
# cluster have NO outbound network, so a job that reaches the normalisation
# step without a populated cache does not fail fast with a clear message -- it
# hangs on an HTTP timeout, burns its walltime, and dies hours in with a
# traceback that looks like a bug in fMRIPrep.
#
# This is the same trap as NILEARN_DATA in slurm/01_extract.sbatch, one layer
# down. Same fix: fetch on a login node, export the path in the job.
#
# Cost: a few GB, once. Re-running is cheap -- TemplateFlow skips what it has.

set -euo pipefail

export TEMPLATEFLOW_HOME="${TEMPLATEFLOW_HOME:-/project/6008063/tamires/templateflow}"

echo "TEMPLATEFLOW_HOME = $TEMPLATEFLOW_HOME"
mkdir -p "$TEMPLATEFLOW_HOME"

# The site may already provide a shared, populated cache. If the fmriprep
# module sets one, prefer it over building a second copy -- check before
# running this script:
#
#   module load StdEnv/2023 fmriprep/25.1.1 && echo "${TEMPLATEFLOW_HOME:-unset}"
#
# If that prints a path that already contains tpl-* directories, you can skip
# this script entirely and just export that path in 02_fmriprep.sbatch.

python3 - <<'PY'
import os, sys

try:
    from templateflow import api as tf
except ImportError:
    sys.exit(
        "templateflow is not importable.\n"
        "\n"
        "On Alliance clusters the bare interpreter has no pip -- `python3 -m pip`\n"
        "reports 'No module named pip'. Python comes from a module instead:\n"
        "\n"
        "    module load StdEnv/2023 python/3.11\n"
        "    virtualenv --no-download ~/venvs/tflow\n"
        "    source ~/venvs/tflow/bin/activate\n"
        "    pip install --no-index --upgrade pip\n"
        "    pip install templateflow\n"
        "\n"
        "then re-run this script with that venv active.\n"
        "\n"
        "This is a LOGIN NODE, one-off step. templateflow is only a downloader:\n"
        "once the templates are on disk the compute nodes never import it, so it\n"
        "does not belong in a container image.\n"
        "\n"
        "Alternative, if the site's fmriprep module is an apptainer wrapper (check\n"
        "with `head -20 $(which fmriprep)`): templateflow already lives inside\n"
        "that image, so run this script's fetch through it rather than installing\n"
        "anything --\n"
        "\n"
        "    apptainer exec --bind /project \\\n"
        f"        --env TEMPLATEFLOW_HOME={os.environ.get('TEMPLATEFLOW_HOME', '')} \\\n"
        "        <image.sif> \\\n"
        "        python -c \"from templateflow import api; "
        "[api.get(t) for t in ('MNI152NLin2009cAsym','OASIS30ANTs',"
        "'MNI152NLin6Asym')]\"\n"
    )

# Whole templates rather than a hand-picked file list. Picking individual
# files is how you discover, four hours into a compute job, that fMRIPrep also
# wanted one more resolution of one more mask. A few extra GB on a 997 TiB
# allocation is not worth that risk.
#
#   MNI152NLin2009cAsym -- our output space, and the space ds002837 and
#                          CNeuroMod already use. Same grid, same atlases.
#   OASIS30ANTs         -- antsBrainExtraction's template. Still needed with
#                          --fs-no-reconall: skipping recon-all skips surfaces,
#                          not skull-stripping.
#   MNI152NLin6Asym     -- used internally by several fMRIPrep steps even when
#                          you never ask for it as an output space.
failed = []
for template in ("MNI152NLin2009cAsym", "OASIS30ANTs", "MNI152NLin6Asym"):
    print(f"fetching {template} ...", flush=True)
    got = tf.get(template)
    # tf.get returns [] rather than raising when it matches nothing, so an
    # empty result here is a silent no-op that would look like success.
    if not got:
        failed.append(template)

if failed:
    sys.exit("\nFAILED to fetch: " + ", ".join(failed))

print("\nfetched, now verifying real files landed on disk ...")
PY

# Verify DATA, not directory structure. TemplateFlow writes the whole tpl-*
# skeleton on import and downloads lazily, so `ls` showing tpl-MNI152... proves
# nothing. 02_fmriprep.sbatch applies the same test before it will submit.
if ! find "$TEMPLATEFLOW_HOME/tpl-MNI152NLin2009cAsym" \
        -name '*_T1w.nii.gz' -size +1M -print -quit 2>/dev/null | grep -q .; then
  echo >&2
  echo "ERROR: the tpl-* directories exist but hold no image data." >&2
  echo "       This is the lazy-download skeleton, not a usable cache." >&2
  echo "       Check the fetch output above for network or proxy errors." >&2
  exit 1
fi

echo
echo "Contents:"
ls -1 "$TEMPLATEFLOW_HOME" | head -20
echo
echo "Largest files (proof the data is really here):"
find "$TEMPLATEFLOW_HOME" -name '*.nii.gz' -size +1M -printf '%10s  %p\n' \
  2>/dev/null | sort -rn | head -5
echo
du -sh "$TEMPLATEFLOW_HOME"

cat <<MSG

Next: export this in your shell and in 02_fmriprep.sbatch --

    export TEMPLATEFLOW_HOME=$TEMPLATEFLOW_HOME

MSG
