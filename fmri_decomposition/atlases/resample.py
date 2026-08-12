"""Label-image resampling, isolated so nilearn stays a lazy import.

Nilearn resamples affinely and does not warn about template mismatch. Harvard-
Oxford ships in FSL's MNI152NLin6Asym; NNDb v2 and CNeuroMod are
MNI152NLin2009cAsym. Those differ nonlinearly by a few mm, most noticeably in
subcortex -- exactly where the 15 subcortical parcels live. The decision taken
was to accept the offset and record it in the manifest (addendum §1); the
header cannot tell you which variant you have (addendum §2), so this module
does not pretend to check.
"""

from __future__ import annotations

import numpy as np


def same_grid(a, b, tol: float = 1e-4) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=tol)


def resample_labels_to(labels_img, reference_img):
    """Nearest-neighbour resample of a label image onto the data grid."""
    if same_grid(labels_img, reference_img):
        return labels_img
    from nilearn.image import resample_to_img

    ref3d = _first_volume(reference_img)
    return resample_to_img(
        labels_img, ref3d, interpolation="nearest",
        force_resample=True, copy_header=True,
    )


def _first_volume(img):
    if len(img.shape) < 4:
        return img
    import nibabel as nib

    return nib.Nifti1Image(
        np.asarray(img.dataobj[..., 0]), img.affine, img.header
    )
