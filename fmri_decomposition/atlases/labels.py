"""Label-image atlases: Harvard-Oxford (canonical 111) and Yeo-7.

The label-table arithmetic is separated from the nilearn fetch so it can be
unit-tested without a network round trip: `plan_harvard_oxford_merge` is a
pure function of two label name lists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import AtlasSpec

# Nuisance compartments in the subcortical atlas. Left in, "Left Cerebral
# Cortex" is the mean of the entire left cortex: it correlates near 1 with
# most cortical nodes and dominates any PCA on the flattened edges.
SUB_DROP = frozenset(
    {
        "Background",
        "Left Cerebral White Matter", "Right Cerebral White Matter",
        "Left Cerebral Cortex", "Right Cerebral Cortex",
        "Left Lateral Ventricle", "Right Lateral Ventricle",
    }
)


@dataclass
class MergePlan:
    """How subcortical labels are renumbered on top of the cortical image."""

    names: list[str]                  # final ordered parcel names
    cort_ids: list[int]               # cortical ids kept, in the cortical image
    sub_remap: dict[int, int]         # old subcortical id -> new merged id
    n_cort: int
    n_sub: int

    @property
    def n_parcels(self) -> int:
        return len(self.names)

    @property
    def final_ids(self) -> list[int]:
        return list(range(1, self.n_parcels + 1))


def plan_harvard_oxford_merge(cort_labels, sub_labels, drop=SUB_DROP) -> MergePlan:
    """Pure: two nilearn label lists -> the merged parcel definition.

    Cortical labels keep their ids 1..n_cort. Surviving subcortical labels are
    appended with fresh ids and, at paint time, overwrite cortex where they
    overlap -- subcortical wins, which is the intended behaviour since the
    cortical maxprob image extends into subcortical territory.
    """
    cort = [str(name) for name in cort_labels]
    sub = [str(name) for name in sub_labels]
    if cort and cort[0].lower() in ("background", ""):
        cort_names = cort[1:]
    else:
        cort_names = cort
    n_cort = len(cort_names)

    sub_remap: dict[int, int] = {}
    sub_names: list[str] = []
    next_id = n_cort + 1
    for old_id, name in enumerate(sub):
        if old_id == 0 or name in drop:
            continue
        sub_remap[old_id] = next_id
        sub_names.append(name)
        next_id += 1

    return MergePlan(
        names=list(cort_names) + sub_names,
        cort_ids=list(range(1, n_cort + 1)),
        sub_remap=sub_remap,
        n_cort=n_cort,
        n_sub=len(sub_names),
    )


def harvard_oxford(threshold: int = 25, resolution_mm: int = 2,
                   symmetric_split: bool = True, drop=SUB_DROP,
                   data_dir: str | None = None) -> AtlasSpec:
    """Canonical Harvard-Oxford: thr25, symmetric_split, merged -> 111 parcels.

    On lateralization: the argument for symmetric_split is not that laterality
    certainly matters, it is that the decision is asymmetric. Extracting 96 and
    collapsing later is a groupby; extracting 48 and recovering laterality is a
    re-run of the expensive stage. The real cost is that nilearn splits at the
    x-midline including genuinely midline structures, so run the L-R
    correlation diagnostic and merge pairs above ~0.95 empirically.
    """
    import nibabel as nib
    from nilearn.datasets import fetch_atlas_harvard_oxford

    from .resample import resample_labels_to

    cort = fetch_atlas_harvard_oxford(
        f"cort-maxprob-thr{threshold}-{resolution_mm}mm",
        symmetric_split=symmetric_split, data_dir=data_dir,
    )
    sub = fetch_atlas_harvard_oxford(
        f"sub-maxprob-thr{threshold}-{resolution_mm}mm", data_dir=data_dir,
    )
    cort_img = _as_nifti(cort.maps)
    sub_img = resample_labels_to(_as_nifti(sub.maps), cort_img)

    plan = plan_harvard_oxford_merge(cort.labels, sub.labels, drop=drop)
    merged = np.asarray(cort_img.dataobj).astype(np.int16)
    sub_data = np.asarray(sub_img.dataobj).astype(np.int16)
    for old_id, new_id in plan.sub_remap.items():
        merged[sub_data == old_id] = new_id   # subcortical wins on overlap

    atlas_img = nib.Nifti1Image(merged, cort_img.affine, cort_img.header)
    network = ["cortical"] * plan.n_cort + ["subcortical"] * plan.n_sub
    spec = AtlasSpec.from_label_image(
        name="harvardoxford", labels_img=atlas_img, names=plan.names,
        indices=plan.final_ids, network=network,
        provenance={
            "atlas": "Harvard-Oxford",
            "threshold": threshold,
            "resolution_mm": resolution_mm,
            "symmetric_split": symmetric_split,
            "dropped": sorted(drop),
            "template": "MNI152NLin6Asym",
            "template_mismatch_accepted": True,
            "n_cortical": plan.n_cort,
            "n_subcortical": plan.n_sub,
        },
    )
    return spec


def harvard_oxford_legacy69(resolution_mm: int = 2, data_dir: str | None = None) -> AtlasSpec:
    """The old 69-parcel definition: thr50, no split, nuisance labels kept.

    Available for reproducing old results. Not the default, and not
    recommended: six of its 69 "regions" are nuisance compartments and it has
    no laterality.
    """
    spec = harvard_oxford(threshold=50, resolution_mm=resolution_mm,
                          symmetric_split=False, drop=frozenset({"Background"}),
                          data_dir=data_dir)
    spec.name = "harvardoxford69"
    spec.provenance["legacy"] = True
    spec.provenance["warning"] = "includes white matter / whole-hemisphere / ventricle nodes"
    return spec


def yeo7(thick: bool = True, data_dir: str | None = None) -> AtlasSpec:
    """Yeo 2011 7-network parcellation. 7 parcels, 21 edges."""
    import nibabel as nib
    from nilearn.datasets import fetch_atlas_yeo_2011

    yeo = fetch_atlas_yeo_2011(data_dir=data_dir)
    maps = getattr(yeo, "thick_7" if thick else "thin_7", None) or yeo["thick_7"]
    img = _as_nifti(maps)
    data = np.asarray(img.dataobj).astype(np.int16)
    if data.ndim == 4:                       # yeo ships as a 4D singleton
        data = data[..., 0]
        img = nib.Nifti1Image(data, img.affine, img.header)

    names = [
        "Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
        "Limbic", "Frontoparietal", "Default",
    ]
    return AtlasSpec.from_label_image(
        name="yeo7", labels_img=img, names=names, indices=list(range(1, 8)),
        network=names,
        provenance={"atlas": "Yeo2011", "variant": "thick_7" if thick else "thin_7"},
    )


def _as_nifti(maps):
    """nilearn returns a path or an image depending on version and atlas."""
    import nibabel as nib

    if isinstance(maps, (str, bytes)) or hasattr(maps, "__fspath__"):
        return nib.load(str(maps))
    if hasattr(maps, "get_fdata"):
        return maps
    fn = maps.get_filename() if hasattr(maps, "get_filename") else None
    if fn:
        return nib.load(fn)
    raise TypeError(f"cannot resolve atlas image from {type(maps)}")
