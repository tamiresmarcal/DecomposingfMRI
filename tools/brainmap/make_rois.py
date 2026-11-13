"""One Talairach ROI per parcel of the states' atlas, for the Mango plugin.

The Behavioral Analysis plugin (Lancaster et al. 2012) counts BrainMap foci
inside a ROI drawn on an image in Talairach space. This takes the pipeline's
own label volume (the parcels the states were fitted on, on the reference grid
in MNI152NLin6Asym 2 mm), carries it to Talairach with icbm_spm2tal
(Lancaster et al. 2007; the transform the Neurosynth path already uses for the
reverse direction), and writes one binary NIfTI per parcel plus rois.tsv.

    python tools/brainmap/make_rois.py --atlas harvardoxford --out WORK/rois
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmri_decomposition.io import require_reference_maps
from fmri_decomposition.unsupervised_model.contextualization import refmaps as rm

# Talairach grid: 2 mm, RAS, covering the Talairach brain with margin.
TAL_AFFINE = np.array([[2.0, 0, 0, -90], [0, 2.0, 0, -126], [0, 0, 2.0, -72], [0, 0, 0, 1]])
TAL_SHAPE = (91, 109, 91)


def tal_labels(lab_mni: np.ndarray, ref_affine: np.ndarray) -> np.ndarray:
    """MNI label volume -> Talairach grid, nearest neighbour (pulled back)."""
    ijk = np.indices(TAL_SHAPE).reshape(3, -1)
    tal = nib.affines.apply_affine(TAL_AFFINE, ijk.T)
    mni = nib.affines.apply_affine(np.linalg.inv(rm._ICBM_SPM2TAL), tal)
    src = np.rint(nib.affines.apply_affine(np.linalg.inv(ref_affine), mni)).astype(int)
    ok = np.all((src >= 0) & (src < np.array(lab_mni.shape)), axis=1)
    out = np.zeros(len(src), dtype=np.int32)
    out[ok] = lab_mni[src[ok, 0], src[ok, 1], src[ok, 2]]
    return out.reshape(TAL_SHAPE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas", default="harvardoxford")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None,
                    help="reference maps dir (default: FMRI_DECOMP_REFERENCE_MAPS / repo)")
    a = ap.parse_args(argv)

    data = a.data or require_reference_maps()
    ref = rm.reference_grid(data)
    spec, src = rm.state_atlas(a.atlas, data)
    parc = rm.Parcellator.build(spec, ref)
    lab_mni = parc.label_volume()
    lab_tal = tal_labels(lab_mni, ref.affine)

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p, name in enumerate(parc.names, start=1):
        m = (lab_tal == p).astype(np.uint8)
        img = nib.Nifti1Image(m, TAL_AFFINE)
        img.set_qform(TAL_AFFINE, code=3)      # 3 = Talairach
        img.set_sform(TAL_AFFINE, code=3)
        f = a.out / f"parcel_{p:03d}.nii.gz"
        nib.save(img, str(f))
        xyz = nib.affines.apply_affine(TAL_AFFINE, np.argwhere(m))
        rows.append({"parcel": p, "name": name, "file": f.name,
                     "n_vox_mni": int((lab_mni == p).sum()), "n_vox_tal": int(m.sum()),
                     "x_tal": xyz[:, 0].mean() if len(xyz) else np.nan,
                     "y_tal": xyz[:, 1].mean() if len(xyz) else np.nan,
                     "z_tal": xyz[:, 2].mean() if len(xyz) else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "rois.tsv", sep="\t", index=False)
    empty = t[t.n_vox_tal == 0]
    print(f"{len(t)} parcels of {a.atlas} ({src}) -> {a.out}")
    if len(empty):
        print("WARNING: parcels with no Talairach voxel:", ", ".join(empty.name))


if __name__ == "__main__":
    main()
