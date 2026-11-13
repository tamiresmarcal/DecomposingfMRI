# Mango script (Jython 2), run headless by run_brainmap_profiles.sh:
#   java ... edu.uthscsa.ric.cli.script.MangoScript -f ba_profile.py parcel_XXX.nii.gz
#
# The loaded image is one parcel (1 inside, 0 outside) in Talairach space. It
# becomes the ROI, and the Behavioral Analysis plugin (Lancaster et al. 2012)
# is asked, for that ROI, for the raw foci counts (data type 0) and the
# z-scores (data type 3), on healthy-participant experiments ("Normals") and on
# all experiments ("All"). Category names come from the plugin itself, so the
# column order can never drift from its data. Output: long CSV at $BA_OUT.
import os
from jarray import array
from java.lang import String
from edu.uthscsa.ric.cli import CLIUtils

PLUGIN = "Behavioral Analysis"
out = os.environ["BA_OUT"]

mgr = m_VolMan_manager
mgr.roiColor = 0
mgr.runThresholdToROI(0.5, False, True, False, False)
mask = mgr.roiData.usedMask
if mask == 0:
    raise SystemExit("empty ROI")

plugin = CLIUtils.findPlugin(PLUGIN)
n = plugin.getNumCategories()
names = [".".join(plugin.getCategoryNames(i)) for i in range(n)]

f = open(out + ".tmp", "w")
f.write("group,measure,category,value\n")
for group in ("Normals", "All"):
    for dtype, measure in ((0, "count"), (3, "z")):
        vals = mgr.runPlugin(PLUGIN, array([group, str(dtype), str(mask)], String))
        if vals is None or len(vals) != n:
            raise SystemExit("plugin returned %r" % (vals,))
        for name, v in zip(names, vals):
            f.write('%s,%s,"%s",%r\n' % (group.lower(), measure, name, float(v)))
f.close()
os.rename(out + ".tmp", out)
print("wrote " + out)
