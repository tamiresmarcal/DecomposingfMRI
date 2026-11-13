#!/bin/bash
# BrainMap behavioural profiles of the states' parcels, from the BrainMap
# database itself (no Neurosynth, no synonyms).
#
# Runs the official Mango "Behavioral Analysis" plugin (Lancaster et al. 2012,
# plugin v4.0 = BrainMap data of 2023) headless on every parcel of the atlas:
#   1. download Mango (unix) and the plugin, once, into $WORK
#   2. tools/brainmap/make_rois.py   one Talairach ROI per parcel
#   3. Mango CLI + ba_profile.py     counts and z per BrainMap category, per parcel
#   4. tools/brainmap/collect.py     <data>/brainmap/<atlas>_mango_{normals,all}.csv
# Then notebook 08 with BEHAVIOURAL = "brainmap" correlates them with the states.
#
# Usage (from the repo root, on a login or compute node with internet for step 1):
#   tools/brainmap/run_brainmap_profiles.sh [--atlas harvardoxford] [--work DIR]
# Environment:
#   PY    python with the repo importable (default: python). In the container:
#         PY="apptainer exec --bind $PWD $SIF python"
#   JAVA  java >= 7 (default: the JRE bundled with Mango)
#   FMRI_DECOMP_REFERENCE_MAPS   reference maps dir, as for the notebooks
set -euo pipefail

ATLAS=harvardoxford
WORK=${WORK:-${SCRATCH:-$PWD}/brainmap_mango}
while [ $# -gt 0 ]; do
    case "$1" in
        --atlas) ATLAS=$2; shift 2 ;;
        --work)  WORK=$2;  shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done
PY=${PY:-python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$WORK"/{dl,home/Mango/Plugins,out,log}

# 1. Mango + plugin -----------------------------------------------------------
if [ ! -f "$WORK/Mango/Mango.jar" ]; then
    curl -fL -o "$WORK/dl/mango_unix.zip" https://mangoviewer.com/downloads/mango_unix.zip
    unzip -q -o "$WORK/dl/mango_unix.zip" -d "$WORK"
fi
PLUGIN_JAR="$WORK/home/Mango/Plugins/BehavioralAnalysis.jar"
if [ ! -f "$PLUGIN_JAR" ]; then
    curl -fL -o "$PLUGIN_JAR" https://mangoviewer.com/downloads/BehavioralAnalysis.jar
fi
JAVA=${JAVA:-$WORK/Mango/jre7/bin/java}
chmod +x "$WORK/Mango/jre7/bin/"* 2>/dev/null || true
"$JAVA" -version 2>&1 | head -1

# 2. ROIs ---------------------------------------------------------------------
if ! $PY -c "import fmri_decomposition, nibabel, scipy" >/dev/null; then
    echo "PY ($PY) cannot import the repo: in the container, load apptainer, bind the"
    echo "repo and \$SCRATCH, and put the repo on APPTAINERENV_PYTHONPATH."
    exit 1
fi
if [ ! -f "$WORK/rois/rois.tsv" ]; then
    $PY "$HERE/make_rois.py" --atlas "$ATLAS" --out "$WORK/rois"
fi

# 3. plugin on every parcel (finished parcels are skipped: safe to rerun) -----
n=0; fail=0
for roi in "$WORK"/rois/parcel_*.nii.gz; do
    p=$(basename "$roi" .nii.gz)
    out="$WORK/out/$p.csv"
    [ -s "$out" ] && continue
    # user.home -> $WORK/home: Mango looks for plugins in ~/Mango/Plugins.
    if BA_OUT="$out" timeout 300 "$JAVA" -Xmx2g -Djava.awt.headless=true \
            -Duser.home="$WORK/home" \
            -cp "$WORK/Mango/Mango.jar" edu.uthscsa.ric.cli.script.MangoScript \
            -f "$HERE/ba_profile.py" "$roi" > "$WORK/log/$p.log" 2>&1 && [ -s "$out" ]; then
        n=$((n + 1))
    else
        fail=$((fail + 1)); echo "FAILED $p (log: $WORK/log/$p.log)"; tail -5 "$WORK/log/$p.log"
        # Stop early if the very first parcel fails: it is the setup, not the parcel.
        [ $n -eq 0 ] && [ $fail -eq 1 ] && exit 1
    fi
done
echo "parcels run now: $n, failed: $fail"
[ $fail -eq 0 ] || exit 1

# 4. tables + checks + comparison with Neurosynth -----------------------------
$PY "$HERE/collect.py" --atlas "$ATLAS" --work "$WORK"
