"""Reference maps for interpreting brain states, on the states' own parcels.

A brain state, as `fmri-decomp state-means` writes it, is one number per parcel
of the atlas it was fitted on. Interpreting it means correlating that vector
with maps that carry meaning, parcellated into THE SAME parcels:

    molecular    19 PET receptor/transporter volumes      (neuromaps / Hansen 2022)
    behavioural  BrainMap domain profiles per Brainnetome  (Fan et al. 2016)
                 subregion, at every level of the BrainMap taxonomy
    canonical    Yeo 2011 7 networks, and this repo's own 14 coordinate networks

Everything is parcellated in the pipeline's own space (MNI152NLin6Asym, 2 mm)
with the pipeline's own operator, `AtlasSpec.membership` -- the matrix that
turned voxels into the parcel time series the states were fitted on. So a
reference map's value for parcel k is computed exactly the way the state's
value for parcel k was, subcortex included. The surface appears only when a
state is drawn.

WHY NOT A PARAMETRIC P
----------------------
Two smooth brain maps correlate far more often than two random vectors of the
same length: neighbouring parcels are not independent, so a p-value over ~100
parcels counts ~100 observations where there are far fewer. The null is a set
of surrogate maps that keep the reference map's spatial autocorrelation over
the distances between parcel centroids: Moran spectral randomisation by
default (fast), brainsmash's variogram matching (Burt et al. 2020) on request.
Unlike the surface spin test both are defined for subcortical parcels and for
sphere atlases, i.e. for every atlas this pipeline has.

An atlas with a handful of parcels (yeo7, networks: 7 and 14) gets no p-value
at all: a correlation over 7 points is a description, and no null makes it
more than that.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from fmri_decomposition.atlases.registry import AtlasSpec, get_atlas  # noqa: E402

HEMIS = ("L", "R")
BRAINNETOME_CSV = Path(__file__).resolve().parent / "data" / "brainnetome_subregions.csv"
MIN_PARCELS_FOR_NULL = 30

# Harvard-Oxford cortical labels in atlas order (TemplateFlow's HOCPAL table
# lists only 52 of the 96). The image interleaves them -- 2i+1 = Left <i>,
# 2i+2 = Right <i> -- which is nilearn's symmetric_split order, hence the order
# of the pipeline's `harvardoxford` columns.
HO_CORTICAL = [
    "Frontal Pole", "Insular Cortex", "Superior Frontal Gyrus",
    "Middle Frontal Gyrus", "Inferior Frontal Gyrus, pars triangularis",
    "Inferior Frontal Gyrus, pars opercularis", "Precentral Gyrus",
    "Temporal Pole", "Superior Temporal Gyrus, anterior division",
    "Superior Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, anterior division",
    "Middle Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, temporooccipital part",
    "Inferior Temporal Gyrus, anterior division",
    "Inferior Temporal Gyrus, posterior division",
    "Inferior Temporal Gyrus, temporooccipital part", "Postcentral Gyrus",
    "Superior Parietal Lobule", "Supramarginal Gyrus, anterior division",
    "Supramarginal Gyrus, posterior division", "Angular Gyrus",
    "Lateral Occipital Cortex, superior division",
    "Lateral Occipital Cortex, inferior division", "Intracalcarine Cortex",
    "Frontal Medial Cortex",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)",
    "Subcallosal Cortex", "Paracingulate Gyrus",
    "Cingulate Gyrus, anterior division", "Cingulate Gyrus, posterior division",
    "Precuneous Cortex", "Cuneal Cortex", "Frontal Orbital Cortex",
    "Parahippocampal Gyrus, anterior division",
    "Parahippocampal Gyrus, posterior division", "Lingual Gyrus",
    "Temporal Fusiform Cortex, anterior division",
    "Temporal Fusiform Cortex, posterior division",
    "Temporal Occipital Fusiform Cortex", "Occipital Fusiform Gyrus",
    "Frontal Operculum Cortex", "Central Opercular Cortex",
    "Parietal Operculum Cortex", "Planum Polare",
    "Heschl's Gyrus (includes H1 and H2)", "Planum Temporale",
    "Supracalcarine Cortex", "Occipital Pole",
]

YEO7 = ["Visual", "Somatomotor", "Dorsal Attention", "Ventral Attention",
        "Limbic", "Frontoparietal", "Default"]
# Yeo et al. 2011's own colours -- the field reads these as the networks' names.
YEO7_COLORS = ["#781286", "#4682b4", "#00760e", "#c43afa", "#dcf8a4",
               "#e69422", "#cd3e4e"]

RECEPTOR_LABELS = {
    "5HT1a": "5-HT1A", "5HT1b": "5-HT1B", "5HT2a": "5-HT2A", "5HT4": "5-HT4",
    "5HT6": "5-HT6", "5HTT": "5-HTT", "A4B2": "α4β2", "CB1": "CB1", "D1": "D1",
    "D2": "D2", "DAT": "DAT", "GABAa-bz": "GABA-A", "H3": "H3", "M1": "M1",
    "mGluR5": "mGluR5", "MU": "MOR", "NAT": "NET", "NMDA": "NMDA",
    "VAChT": "VAChT",
}
# Grouped by transmitter system: 5-HT, DA, NE, ACh, GABA/Glu, other.
RECEPTOR_ORDER = ["5-HT1A", "5-HT1B", "5-HT2A", "5-HT4", "5-HT6", "5-HTT",
                  "D1", "D2", "DAT", "NET", "α4β2", "M1", "VAChT", "GABA-A",
                  "NMDA", "mGluR5", "CB1", "MOR", "H3"]

# The BrainMap behavioural-domain taxonomy (Fox et al. 2005; brainmap.org/
# taxonomy), as the leaves Brainnetome's profiles are reported at: domain,
# then subdomain, then -- where BrainMap splits further -- sub-subdomain.
# Written out by hand here because brainmap.org was unreachable; a real
# behavioural table brings its own columns and this list is then unused.
BRAINMAP_TAXONOMY = [
    "Action.Execution.Speech", "Action.Execution.Other", "Action.Imagination",
    "Action.Inhibition", "Action.Motor Learning", "Action.Observation",
    "Action.Preparation", "Action.Rest",
    "Cognition.Attention", "Cognition.Language.Orthography",
    "Cognition.Language.Phonology", "Cognition.Language.Semantics",
    "Cognition.Language.Speech", "Cognition.Language.Syntax",
    "Cognition.Memory.Explicit", "Cognition.Memory.Working",
    "Cognition.Music", "Cognition.Reasoning", "Cognition.Social Cognition",
    "Cognition.Soma", "Cognition.Space", "Cognition.Time",
    "Emotion.Anger", "Emotion.Anxiety", "Emotion.Disgust", "Emotion.Fear",
    "Emotion.Happiness", "Emotion.Sadness",
    "Interoception.Air Hunger", "Interoception.Baroregulation",
    "Interoception.Bladder", "Interoception.Hunger", "Interoception.Sexuality",
    "Interoception.Sleep", "Interoception.Thermoregulation", "Interoception.Thirst",
    "Perception.Audition", "Perception.Gustation", "Perception.Olfaction",
    "Perception.Somesthesis.Pain", "Perception.Somesthesis.Other",
    "Perception.Vision.Color", "Perception.Vision.Motion",
    "Perception.Vision.Shape", "Perception.Vision.Other",
]

# Brainnetome's own grouping: 24 gyri per hemisphere, 20 cortical + 4 deep.
BN_SUBCORTICAL = ("Amyg", "Hipp", "BG", "Tha")


# ------------------------------------------------------------------ grid ---
def _tf(data: Path, name: str) -> Path:
    return Path(data) / "templateflow" / name


def reference_grid(data: Path):
    """MNI152NLin6Asym 2 mm, the grid every map is parcellated on."""
    if not (Path(data) / "templateflow").is_dir():
        raise SystemExit(
            f"no reference maps in {data}.\n"
            f"  They are not in git (~285 MB). Fetch them once, on a node with internet:\n"
            f"    {Path(__file__).parent / 'fetch_reference_maps.sh'} {data}")
    return nib.load(str(_tf(data, "tpl-MNI152NLin6Asym_res-02_atlas-HOCPAL_desc-th25_dseg.nii.gz")))


def resample(img, ref, interpolation: str):
    from nilearn.image import resample_to_img

    try:
        return resample_to_img(img, ref, interpolation=interpolation,
                               force_resample=True, copy_header=True)
    except TypeError:                                   # older nilearn
        return resample_to_img(img, ref, interpolation=interpolation)


# ----------------------------------------------------------- state atlas ---
def state_atlas(name: str, data: Path) -> tuple[AtlasSpec, str]:
    """The pipeline's atlas `name`, and where it came from.

    `get_atlas` first: on a machine where nilearn's cache or its download
    hosts are there, this IS the atlas the states were fitted on. Where they
    are not, `harvardoxford` and `yeo7` are rebuilt from the same source
    images (TemplateFlow / neuroparc) by the same code path the repo uses --
    `plan_harvard_oxford_merge`, `AtlasSpec.from_label_image` -- so the
    parcel names, order and count are the pipeline's.
    """
    try:
        return get_atlas(name), "fmri_decomposition.get_atlas"
    except Exception as e:                              # noqa: BLE001
        why = f"{type(e).__name__}"
    key = name.lower().replace("_", "")
    if key == "harvardoxford":
        return _harvard_oxford_offline(data), f"rebuilt from TemplateFlow ({why} on fetch)"
    if key == "yeo7":
        return _yeo7_offline(data), f"rebuilt from neuroparc ({why} on fetch)"
    raise


def _harvard_oxford_offline(data: Path) -> AtlasSpec:
    from fmri_decomposition.atlases.labels import SUB_DROP, plan_harvard_oxford_merge
    from fmri_decomposition.atlases.resample import resample_labels_to

    cort = nib.load(str(_tf(data, "tpl-MNI152NLin6Asym_res-02_atlas-HOCPAL_desc-th25_dseg.nii.gz")))
    sub = nib.load(str(_tf(data, "tpl-MNI152NLin6Asym_res-02_atlas-HOSPA_desc-th25_dseg.nii.gz")))
    cort_labels = ["Background"] + [f"{s} {n}" for n in HO_CORTICAL for s in ("Left", "Right")]
    sub_tsv = pd.read_csv(_tf(data, "tpl-MNI152NLin6Asym_atlas-HOSPA_dseg.tsv"), sep="\t")
    # FSL's XML spells it "Ventrical"; nilearn's list (which SUB_DROP is
    # written against) does not. Normalise so the same six are dropped.
    sub_labels = [str(n).strip().replace("Ventrical", "Ventricle")
                  for n in sub_tsv.sort_values("index")["name"]]
    plan = plan_harvard_oxford_merge(cort_labels, sub_labels, drop=SUB_DROP)
    merged = np.asarray(cort.dataobj).astype(np.int16)
    sub_data = np.asarray(resample_labels_to(sub, cort).dataobj).astype(np.int16)
    for old_id, new_id in plan.sub_remap.items():
        merged[sub_data == old_id] = new_id
    img = nib.Nifti1Image(merged, cort.affine, cort.header)
    network = ["cortical"] * plan.n_cort + ["subcortical"] * plan.n_sub
    return AtlasSpec.from_label_image(
        name="harvardoxford", labels_img=img, names=plan.names,
        indices=plan.final_ids, network=network,
        provenance={"atlas": "Harvard-Oxford", "source": "TemplateFlow HOCPAL+HOSPA th25 2mm"})


def _yeo7_offline(data: Path) -> AtlasSpec:
    img = nib.load(str(Path(data) / "yeo" / "Yeo-7-liberal_space-MNI152NLin6_res-1x1x1.nii.gz"))
    names = ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
             "Limbic", "Frontoparietal", "Default"]
    return AtlasSpec.from_label_image(
        name="yeo7", labels_img=img, names=names, indices=list(range(1, 8)),
        network=names, provenance={"atlas": "Yeo2011", "variant": "liberal (neuroparc)"})


@dataclass
class Parcellator:
    """The pipeline's voxel -> parcel operator, on the reference grid."""

    spec: AtlasSpec
    ref: object
    support: object = field(repr=False)                 # sparse (P, V) 0/1

    @classmethod
    def build(cls, spec: AtlasSpec, ref) -> "Parcellator":
        M, _ = spec.membership(ref)
        return cls(spec, ref, (M > 0).astype(np.float64).tocsr())

    @property
    def names(self) -> list:
        return list(self.spec.labels["name"])

    @property
    def columns(self) -> list:
        return self.spec.columns

    @property
    def n(self) -> int:
        return self.spec.n_nodes

    @property
    def centroids(self) -> np.ndarray:
        return self.spec.labels[["x", "y", "z"]].to_numpy(float)

    def parcellate(self, vols: np.ndarray) -> np.ndarray:
        """(..., V) voxel values -> (..., P) parcel means, NaN voxels skipped."""
        v = np.asarray(vols, float)
        lead = v.shape[:-1]
        v = v.reshape(-1, v.shape[-1])
        ok = np.isfinite(v)
        sums = self.support @ np.where(ok, v, 0.0).T
        counts = self.support @ ok.T.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (sums / counts).T
        return out.reshape(lead + (self.n,))

    def label_volume(self) -> np.ndarray:
        """Voxel -> parcel (1..P, 0 = none); the first parcel wins an overlap."""
        S = self.support.tocoo()
        order = np.argsort(-S.row, kind="stable")     # lowest parcel written last
        lab = np.zeros(S.shape[1], dtype=np.int32)
        lab[S.col[order]] = S.row[order] + 1
        return lab.reshape(self.ref.shape[:3])

    def paint(self, values: np.ndarray):
        """(P,) -> NIfTI on the reference grid, 0 off-parcel (for plotting)."""
        vals = np.concatenate([[0.0], np.asarray(values, float)])
        return nib.Nifti1Image(vals[self.label_volume()].astype(np.float32), self.ref.affine)


# --------------------------------------------------------- reference maps ---
@dataclass
class ReferenceMaps:
    family: str
    names: list
    voxels: np.ndarray                  # (n_maps, V) on the reference grid, NaN = no data
    simulated: bool = False
    source: str = ""


def _flat(img) -> np.ndarray:
    return np.asarray(img.dataobj, dtype=np.float64).reshape(-1)


def receptor_maps(data: Path, ref) -> ReferenceMaps:
    """The 19 PET maps, resampled to the reference grid.

    Several volumes are zero outside a brain mask. Zero there is "not
    measured", not "no receptor": those voxels are NaN, so they drop out of
    the parcel mean instead of pulling the edge parcels toward zero.
    """
    files = {}
    for f in sorted((Path(data) / "pet").glob("*.nii*")):
        key = f.name.split("_")[0]
        files[RECEPTOR_LABELS.get(key, key)] = f
    names = [n for n in RECEPTOR_ORDER if n in files]
    vox = []
    for n in names:
        img = nib.load(str(files[n]))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        img = nib.Nifti1Image(arr, img.affine)
        v = _flat(resample(img, ref, "continuous"))
        if (arr != 0).mean() < 0.5:
            m = nib.Nifti1Image((arr != 0).astype(np.float32), img.affine)
            with warnings.catch_warnings():   # linear on a 0/1 mask is the point: >.5 = mostly measured
                warnings.simplefilter("ignore", UserWarning)
                v[_flat(resample(m, ref, "linear")) < 0.5] = np.nan
        vox.append(v)
    return ReferenceMaps("molecular", names, np.stack(vox),
                         source="Hansen et al. 2022 PET atlas (neuromaps MNI152 annotations)")


def yeo7_maps(data: Path, ref) -> ReferenceMaps:
    """One indicator per network; parcellated = each parcel's share of it."""
    img = resample(nib.load(str(Path(data) / "yeo" / "Yeo-7-liberal_space-MNI152NLin6_res-1x1x1.nii.gz")),
                   ref, "nearest")
    lab = np.rint(_flat(img)).astype(int)
    return ReferenceMaps("yeo7", list(YEO7),
                         np.stack([(lab == k).astype(float) for k in range(1, 8)]),
                         source="Yeo et al. 2011, 7 networks, liberal mask (neuroparc, MNI152NLin6)")


def network14_maps(ref) -> ReferenceMaps:
    """This repo's 14 coordinate networks as indicator maps: a voxel is in a
    network if it is in any of that network's 5 mm spheres."""
    spec = get_atlas("networks")
    M, _ = spec.membership(ref)
    ind = (M > 0).astype(float).toarray()
    return ReferenceMaps("networks14", list(spec.labels["name"]), ind,
                         source="fmri_decomposition `networks` (14 networks, 254 spheres, 5 mm)")


# ------------------------------------------------------------ Brainnetome ---
def brainnetome_table(data: Path) -> pd.DataFrame:
    """The 246 BNA subregions: bn_index, hemi, gyrus (abbrev.), short name."""
    # Shipped as CSV (derived from BNA_subregions.xlsx by `_brainnetome_from_xlsx`)
    # so reading it needs no Excel engine on the cluster. It lives in the
    # package's data/ (committed, 7 KB, derived here) and NOT in
    # reference_atlases/ (gitignored, downloaded): a fresh checkout must have it.
    return pd.read_csv(BRAINNETOME_CSV)


def _brainnetome_from_xlsx(data: Path) -> pd.DataFrame:
    """How brainnetome_subregions.csv was made, from the atlas's own table."""
    x = pd.read_excel(Path(data) / "brainnetome" / "BrainnetomeAtlas_BNA_subregions.xlsx")
    x.columns = ["lobe", "gyrus", "region", "id_l", "id_r", "short", "descr", "mni_l", "mni_r"]
    x["gyrus"] = x["gyrus"].ffill().str.split(",").str[0].str.strip()
    x["lobe"] = x["lobe"].ffill()
    rows = []
    for _, r in x.iterrows():
        short = str(r["short"]).split(",")[0].strip()
        for h, col in (("L", "id_l"), ("R", "id_r")):
            rows.append({"bn_index": int(r[col]), "hemi": h, "gyrus": r["gyrus"],
                         "lobe": r["lobe"], "name": short})
    return pd.DataFrame(rows).sort_values("bn_index").reset_index(drop=True)


def brainnetome_labels(data: Path, ref) -> np.ndarray:
    img = nib.load(str(Path(data) / "brainnetome" / "BrainnetomeAtlas_BNA_MPM_thr25_1.25mm.nii.gz"))
    return np.rint(_flat(resample(img, ref, "nearest"))).astype(int)


def simulate_behavioural_table(bn: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A STAND-IN for the BrainMap profiles of Fan et al. 2016. Not data.

    The real per-subregion profiles are served by atlas.brainnetome.org, which
    could not be reached. This fills a table of the same shape -- 246
    subregions x the BrainMap leaves -- from a textbook prior: what each gyrus
    does, sharpened for the handful of subregions whose name says more (MT+,
    A41/42, A44/45, tongue/larynx motor cortex ...), plus noise. It exists so
    the panel and its statistics can be built; every figure that uses it says
    SIMULATED, and `load_behavioural_table` replaces it when the real table is
    on disk.
    """
    gyrus = {
        "SFG": {"Cognition.Reasoning": .6, "Action.Preparation": .5, "Cognition.Social Cognition": .3,
                "Cognition.Memory.Working": .3},
        "MFG": {"Cognition.Reasoning": .8, "Cognition.Attention": .6, "Cognition.Memory.Working": .9,
                "Action.Inhibition": .3},
        "IFG": {"Cognition.Language.Semantics": .7, "Cognition.Language.Syntax": .7,
                "Cognition.Language.Phonology": .6, "Action.Inhibition": .5,
                "Action.Observation": .3},
        "OrG": {"Emotion.Happiness": .5, "Emotion.Disgust": .3, "Perception.Gustation": .6,
                "Perception.Olfaction": .6, "Interoception.Hunger": .4, "Cognition.Reasoning": .3},
        "PrG": {"Action.Execution.Other": 1.0, "Action.Imagination": .5, "Action.Motor Learning": .4},
        "PCL": {"Action.Execution.Other": .8, "Perception.Somesthesis.Other": .6,
                "Interoception.Bladder": .5},
        "STG": {"Perception.Audition": 1.0, "Cognition.Language.Speech": .6, "Cognition.Music": .6},
        "MTG": {"Cognition.Language.Semantics": .8, "Cognition.Social Cognition": .5,
                "Perception.Vision.Motion": .3},
        "ITG": {"Perception.Vision.Shape": .6, "Cognition.Language.Semantics": .4,
                "Cognition.Memory.Explicit": .3},
        "FuG": {"Perception.Vision.Shape": .9, "Perception.Vision.Color": .5,
                "Cognition.Language.Orthography": .6},
        "PhG": {"Cognition.Memory.Explicit": 1.0, "Cognition.Space": .6, "Emotion.Fear": .3},
        "pSTS": {"Cognition.Social Cognition": 1.0, "Cognition.Language.Speech": .5,
                 "Action.Observation": .6},
        "SPL": {"Cognition.Attention": .8, "Cognition.Space": .8, "Action.Observation": .4},
        "IPL": {"Cognition.Attention": .5, "Cognition.Social Cognition": .4,
                "Cognition.Language.Phonology": .4, "Perception.Somesthesis.Other": .3},
        "Pcun": {"Cognition.Memory.Explicit": .7, "Cognition.Social Cognition": .5,
                 "Action.Rest": .6, "Cognition.Space": .4},
        "PoG": {"Perception.Somesthesis.Other": 1.0, "Perception.Somesthesis.Pain": .5,
                "Action.Execution.Other": .4},
        "INS": {"Interoception.Thirst": .4, "Interoception.Air Hunger": .4, "Emotion.Disgust": .8,
                "Perception.Somesthesis.Pain": .7, "Perception.Gustation": .7,
                "Interoception.Baroregulation": .4},
        "CG": {"Emotion.Sadness": .5, "Perception.Somesthesis.Pain": .7, "Action.Inhibition": .5,
               "Cognition.Attention": .4, "Emotion.Anxiety": .4},
        "MVOcC": {"Perception.Vision.Other": 1.0, "Perception.Vision.Color": .6},
        "LOcC": {"Perception.Vision.Shape": .8, "Perception.Vision.Motion": .6,
                 "Action.Observation": .4},
        "Amyg": {"Emotion.Fear": 1.0, "Emotion.Anxiety": .7, "Emotion.Sadness": .5,
                 "Emotion.Anger": .5, "Interoception.Sexuality": .4},
        "Hipp": {"Cognition.Memory.Explicit": 1.0, "Cognition.Space": .6, "Emotion.Anxiety": .3},
        "BG": {"Action.Motor Learning": .8, "Action.Execution.Other": .6, "Emotion.Happiness": .5,
               "Interoception.Hunger": .3, "Cognition.Time": .5},
        "Tha": {"Perception.Somesthesis.Pain": .6, "Cognition.Attention": .5,
                "Interoception.Sleep": .6, "Interoception.Thermoregulation": .3},
    }
    by_name = {
        "A41/42": {"Perception.Audition": 1.4}, "TE1.0 and TE1.2": {"Perception.Audition": 1.4},
        "V5/MT+": {"Perception.Vision.Motion": 1.5}, "A44d": {"Cognition.Language.Phonology": 1.0},
        "A44op": {"Cognition.Language.Speech": .9}, "A45c": {"Cognition.Language.Semantics": 1.0},
        "A45r": {"Cognition.Language.Semantics": 1.0}, "A4tl": {"Action.Execution.Speech": 1.4},
        "A1/2/3tonIa": {"Action.Execution.Speech": .8}, "A6vl": {"Action.Execution.Speech": .7},
        "A22c": {"Cognition.Language.Speech": .9}, "A37mv": {"Perception.Vision.Color": .9},
        "A39rd": {"Cognition.Social Cognition": .8}, "cCunG": {"Perception.Vision.Other": .5},
        "vIa": {"Interoception.Thirst": .6}, "G": {"Perception.Gustation": 1.0},
    }
    rng = np.random.default_rng(seed)
    out = bn[["bn_index", "hemi", "gyrus", "name"]].copy()
    for d in BRAINMAP_TAXONOMY:
        base = np.array([gyrus.get(g, {}).get(d, 0.0) + by_name.get(n, {}).get(d, 0.0)
                         for g, n in zip(out["gyrus"], out["name"])])
        out[d] = np.clip(base + rng.normal(0, 0.12, len(out)), 0, None)
    return out


# BrainMap category -> the Neurosynth (v7, abstract tf-idf) terms that stand
# for it. A study belongs to a category if any of its terms passes Neurosynth's
# usual 0.001 threshold. Categories with no term in the vocabulary (air hunger,
# bladder, thermoregulation, thirst) are left out rather than forced.
NEUROSYNTH_TERMS = {
    "Action.Execution.Speech": ["speech production", "articulatory"],
    "Action.Execution.Other": ["motor", "movement", "movements", "finger tapping"],
    "Action.Imagination": ["motor imagery", "imagined"],
    "Action.Inhibition": ["response inhibition", "inhibitory control", "stop signal", "nogo"],
    "Action.Motor Learning": ["skill"],
    "Action.Observation": ["action observation"],
    "Action.Preparation": ["preparation", "preparatory"],
    "Action.Rest": ["resting state", "rest"],
    "Cognition.Attention": ["attention", "attentional", "selective attention"],
    "Cognition.Language.Orthography": ["orthographic", "letters", "reading"],
    "Cognition.Language.Phonology": ["phonological"],
    "Cognition.Language.Semantics": ["semantic", "semantics"],
    "Cognition.Language.Speech": ["speech perception"],
    "Cognition.Language.Syntax": ["syntactic", "sentence comprehension"],
    "Cognition.Memory.Explicit": ["episodic memory", "memory retrieval", "memory encoding",
                                  "recognition memory"],
    "Cognition.Memory.Working": ["working memory"],
    "Cognition.Music": ["music", "musical"],
    "Cognition.Reasoning": ["reasoning", "inference"],
    "Cognition.Social Cognition": ["social cognition", "mentalizing", "theory mind"],
    "Cognition.Soma": ["body"],
    "Cognition.Space": ["spatial", "navigation"],
    "Cognition.Time": ["timing", "duration"],
    "Emotion.Anger": ["anger"],
    "Emotion.Anxiety": ["anxiety"],
    "Emotion.Disgust": ["disgust"],
    "Emotion.Fear": ["fear", "fearful"],
    "Emotion.Happiness": ["happy"],
    "Emotion.Sadness": ["sad"],
    "Interoception.Baroregulation": ["heart rate"],
    "Interoception.Hunger": ["food", "eating"],
    "Interoception.Sexuality": ["sexual"],
    "Interoception.Sleep": ["sleep"],
    "Perception.Audition": ["auditory", "sounds"],
    "Perception.Gustation": ["taste"],
    "Perception.Olfaction": ["olfactory"],
    "Perception.Somesthesis.Pain": ["pain", "painful", "nociceptive"],
    "Perception.Somesthesis.Other": ["somatosensory", "tactile", "touch"],
    "Perception.Vision.Color": ["color"],
    "Perception.Vision.Motion": ["visual motion", "motion"],
    "Perception.Vision.Shape": ["shape", "shapes", "object recognition"],
    "Perception.Vision.Other": ["visual", "primary visual"],
}

# Beam, Potts, Poldrack & Etkin 2021 (Nat Neurosci 24:1733), data-driven
# ontology, 6 domains: each domain's term list as published with the paper
# (ehbeam/neuro-knowledge-engine, ontology/lists/lists_data-driven_lr.csv).
# Each term is its own Neurosynth category, so the taxonomy has two depths
# like BrainMap's -- domain, and the terms under it. Terms absent from the
# Neurosynth vocabulary (mostly multi-word phrases) drop out at run time.
BEAM2021_TERMS = {
    "Memory": ["memory", "episodic memory", "declarative memory", "recognition memory",
               "remembering", "emotion", "fear", "recognition", "emotional memory", "encoding",
               "retrieval", "valence", "facial expression", "picture", "recall",
               "autobiographical memory", "face", "memory process", "source memory",
               "encoding task", "arousal", "episodic simulation"],
    "Reward": ["reward", "reward processing", "reward anticipation", "anticipation",
               "monetary incentive delay task", "reward learning"],
    "Cognition": ["monitoring", "cognitive control", "arousal", "cognitive process",
                  "reaction time", "retrieval", "decision making", "strategy", "error detection",
                  "context", "valence", "cognitive", "manipulation", "working memory",
                  "response selection", "goal", "word", "judgment", "anticipation", "memory",
                  "remembering", "decision", "autobiographical memory", "salience", "emotion"],
    "Vision": ["recognition", "face perception", "familiarity", "vision", "object recognition",
               "retrieval", "visual perception", "memory", "remembering", "word",
               "visual attention", "picture", "visual object", "visual angle",
               "recognition memory", "reading", "semantic processing", "encoding",
               "visual imagery", "naming"],
    "Manipulation": ["execution", "hand", "movement", "working memory", "manipulation",
                     "rehearsal", "attention", "spatial attention", "action", "covert",
                     "motor control", "strategy", "reaction time", "grasping"],
    "Language": ["speech", "hearing", "listening", "speech perception", "speech processing",
                 "tone", "language", "music", "passive listening", "syllable", "melody",
                 "speech production", "auditory perception", "language processing", "prosody",
                 "intonation", "audition", "pitch discrimination", "language comprehension",
                 "articulation", "phonetics", "word", "semantic processing", "syntax",
                 "voice perception"],
}
# The same categories with NO synonyms: each is the Neurosynth term that is its
# own name, and nothing else. The rule: the leaf of the BrainMap path, lower
# case ("Emotion.Fear" -> "fear"); where the leaf is only a modifier of its
# parent, the two together ("Memory.Working" -> "working memory"). A category
# whose own name is not in the Neurosynth v7 vocabulary is left out, not
# replaced: imagination, motor learning, orthography, phonology, syntax,
# explicit memory, soma, happiness, sadness, audition, gustation, olfaction,
# every interoceptive category but sleep, and the "... Other" bins.
# Action.Execution.Speech and Cognition.Language.Speech both name "speech",
# so their two maps are the same map.
NEUROSYNTH_TERMS_PURE = {
    "Action.Execution.Speech": ["speech"],
    "Action.Inhibition": ["inhibition"],
    "Action.Observation": ["observation"],
    "Action.Preparation": ["preparation"],
    "Action.Rest": ["rest"],
    "Cognition.Attention": ["attention"],
    "Cognition.Language.Semantics": ["semantics"],
    "Cognition.Language.Speech": ["speech"],
    "Cognition.Memory.Working": ["working memory"],
    "Cognition.Music": ["music"],
    "Cognition.Reasoning": ["reasoning"],
    "Cognition.Social Cognition": ["social cognition"],
    "Cognition.Space": ["space"],
    "Cognition.Time": ["time"],
    "Emotion.Anger": ["anger"],
    "Emotion.Anxiety": ["anxiety"],
    "Emotion.Disgust": ["disgust"],
    "Emotion.Fear": ["fear"],
    "Interoception.Sleep": ["sleep"],
    "Perception.Somesthesis.Pain": ["pain"],
    "Perception.Vision.Color": ["color"],
    "Perception.Vision.Motion": ["motion"],
    "Perception.Vision.Shape": ["shape"],
}

TAXONOMIES = {
    "brainmap": ("BrainMap (Fox et al. 2005)", NEUROSYNTH_TERMS),
    "brainmap_pure": ("BrainMap (Fox et al. 2005), own names only, no synonyms",
                      NEUROSYNTH_TERMS_PURE),
    "beam2021": ("Beam et al. 2021",
                 {f"{d}.{t}": [t] for d, terms in BEAM2021_TERMS.items() for t in terms}),
}


# Lancaster et al. 2007, icbm_spm2tal; its inverse takes Talairach peaks to MNI.
_ICBM_SPM2TAL = np.array([[0.9254, 0.0024, -0.0118, -1.0207],
                          [-0.0048, 0.9316, -0.0871, -1.7667],
                          [0.0152, 0.0883, 0.8924, 4.0926],
                          [0.0, 0.0, 0.0, 1.0]])


LANCASTER_STATISTICS = ("lancaster", "lancaster_studies", "lancaster_weighted",
                        "lancaster_reverse")


def neurosynth_behavioural_table(data: Path, ref, taxonomy: str = "brainmap",
                                 radius_mm: float = 6.0,
                                 threshold: float = 0.001,
                                 labels: np.ndarray | None = None,
                                 key: str = "brainnetome",
                                 statistic: str = "zforward") -> pd.DataFrame:
    """Parcels x BrainMap categories from Neurosynth, by NiMARE's BrainMap decoder.

    The regional behavioural analysis of BrainMap (Lancaster et al. 2012), as
    implemented by the Neurosynth team in NiMARE (Salo et al. 2023;
    `nimare.decode.discrete.brainmap_decode`), run on Neurosynth v7. For each
    parcel, the studies with a peak in it (or, outside every parcel, whose
    nearest parcel is within `radius_mm`) are the selection, and the decoder
    returns, per category, the one-sided binomial forward test of enrichment
    (`zForward`, >= 0). That z is the parcel's value.

    A category is a set of Neurosynth terms; a study carries it when any of
    them has tf-idf >= `threshold` (NiMARE's frequency threshold).

    `labels` is the parcellation to count in: an int volume on `ref`, 0 = none,
    1..P = parcel. Pass the states' own atlas (`Parcellator.label_volume`).
    Omitted, it falls back to the 246 Brainnetome subregions. The table's
    first column is `parcel` (1..P), plus `bn_index` for Brainnetome.
    Cached as neurosynth/<key>_nimare_<taxonomy>.csv; delete to recompute.

    `statistic="lancaster"` replaces the decoder by the statistic of the
    Mango Behavioral Analysis plugin (`lancaster_z`), the one the BrainMap
    database profiles use: the share of a category's foci that fall inside
    the parcel against the parcel's share of the atlas volume. Foci are
    counted strictly inside parcels (no `radius_mm`), as the plugin does, so
    the two databases then differ in their data only. Cached as
    neurosynth/<key>_lancaster_<taxonomy>.csv.
    """
    import scipy.sparse as sp
    from scipy.ndimage import distance_transform_edt

    nsd = Path(data) / "neurosynth"
    if taxonomy not in TAXONOMIES:
        raise SystemExit(f"taxonomy must be one of {sorted(TAXONOMIES)}, got {taxonomy!r}")
    if statistic not in LANCASTER_STATISTICS + ("zforward",):
        raise SystemExit(f"statistic must be one of {('zforward',) + LANCASTER_STATISTICS}, "
                         f"got {statistic!r}")
    label, categories = TAXONOMIES[taxonomy]
    if statistic.startswith("lancaster"):
        unit = {"lancaster": "foci", "lancaster_studies": "studies",
                "lancaster_weighted": "foci weighted 1/n per study",
                "lancaster_reverse": "reverse inference, foci weighted 1/n per study"}[statistic]
        source = (f"Neurosynth v7, Lancaster 2012 z (Mango plugin statistic) on {unit}, "
                  f"{label} taxonomy, counted in {key} parcels")
        cache = nsd / f"{key}_{statistic.replace('_', '')}_{taxonomy}.csv"
        radius_mm = 0.0
    else:
        source = (f"Neurosynth v7, NiMARE BrainMap decoder (zForward), {label} taxonomy, "
                  f"counted in {key} parcels")
        cache = nsd / f"{key}_nimare_{taxonomy}.csv"
    if cache.exists():
        out = pd.read_csv(cache)
        out.attrs["source"] = source
        return out
    pre = nsd / "data-neurosynth_version-7_"
    if not (nsd / "data-neurosynth_version-7_coordinates.tsv.gz").exists():
        raise SystemExit(f"no Neurosynth data in {nsd}: run fetch_reference_maps.sh")
    coords = pd.read_table(f"{pre}coordinates.tsv.gz")
    meta = pd.read_table(f"{pre}metadata.tsv.gz")
    feats = sp.load_npz(f"{pre}vocab-terms_source-abstract_type-tfidf_features.npz").tocsc()
    vocab = open(f"{pre}vocab-terms_vocabulary.txt").read().strip().split("\n")

    # Peaks to MNI, then to the nearest label within radius_mm.
    space = coords["id"].map(meta.set_index("id")["space"])
    xyz = coords[["x", "y", "z"]].to_numpy(float)
    tal = (space == "TAL").to_numpy()
    xyz[tal] = nib.affines.apply_affine(np.linalg.inv(_ICBM_SPM2TAL), xyz[tal])
    if labels is None:
        labels = brainnetome_labels(data, ref)
    lab = np.asarray(labels).astype(int).reshape(ref.shape[:3])
    n_parcels = int(lab.max())
    dist, idx = distance_transform_edt(lab == 0, return_indices=True,
                                       sampling=np.abs(np.diag(ref.affine)[:3]))
    near = np.where(dist <= radius_mm, lab[tuple(idx)], 0)
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(ref.affine), xyz)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(ref.shape[:3])), axis=1)
    peak_lab = np.zeros(len(xyz), dtype=int)
    peak_lab[inside] = near[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]

    # Annotations: one column per category, the max tf-idf over its terms, so
    # ">= threshold" is "any of its terms passes" in the decoder's own binarizing.
    ann = pd.DataFrame({"id": meta["id"].to_numpy()})
    n_studies = {}
    for cat, terms in categories.items():
        cols = [vocab.index(t) for t in terms if t in vocab]
        if not cols:
            continue
        ann[cat] = np.asarray(feats[:, cols].max(axis=1).todense()).ravel()
        n_studies[cat] = int((ann[cat] >= threshold).sum())
    cats = list(n_studies)
    foci = coords[["id"]].copy()

    out = pd.DataFrame({"parcel": np.arange(1, n_parcels + 1)})
    if key == "brainnetome":
        out["bn_index"] = out["parcel"]
    if statistic.startswith("lancaster"):
        # "lancaster": every focus counts, as in the plugin. "lancaster_studies":
        # a study counts once per parcel it has any focus in (as in MKDA,
        # Wager et al. 2007), so a study reporting many peaks weighs no more
        # than one reporting a single peak; the total is then the category's
        # studies with any focus inside the atlas. "lancaster_weighted": every
        # focus counts with weight 1/n, n = its study's foci inside the atlas,
        # so each study is one vote shared out among its foci: no study
        # dominates (as with studies) and a parcel's expected share still grows
        # with its volume (as with foci), which counting studies loses.
        n_vox = np.bincount(lab.ravel(), minlength=n_parcels + 1)[1:]
        has = ann.set_index("id")
        ids_all = coords["id"].to_numpy()
        # Every study's foci inside the atlas, weighted 1/n: the reverse
        # inference's denominator (what is reported in a parcel at all).
        inside = peak_lab > 0
        w_all = np.zeros(len(peak_lab))
        sid_in = pd.Series(ids_all[inside])
        w_all[inside] = 1.0 / sid_in.map(sid_in.value_counts()).to_numpy(float)
        T_par = np.bincount(peak_lab[inside], weights=w_all[inside],
                            minlength=n_parcels + 1)[1:]
        z = np.zeros((n_parcels, len(cats)))
        for j, cat in enumerate(cats):
            ids = set(has.index[has[cat].to_numpy() >= threshold])
            in_cat = coords["id"].isin(ids).to_numpy() & (peak_lab > 0)
            if statistic == "lancaster_reverse":
                # Of what is reported in this parcel (weighted foci of every
                # study), the share from this category's studies, against
                # that share over the whole atlas. Relative to the parcel's
                # own total, so a parcel the literature samples little
                # (signal dropout: orbitofrontal, temporal pole) is judged by
                # its profile, not its density.
                C_par = np.bincount(peak_lab[in_cat], weights=w_all[in_cat],
                                    minlength=n_parcels + 1)[1:]
                z[:, j] = reverse_z(C_par, T_par)
                continue
            if statistic == "lancaster_weighted":
                sid = ids_all[in_cat]
                n_per = pd.Series(sid).map(pd.Series(sid).value_counts()).to_numpy(float)
                count = np.bincount(peak_lab[in_cat], weights=1.0 / n_per,
                                    minlength=n_parcels + 1)[1:]
                total = len(set(sid))
            elif statistic == "lancaster_studies":
                pairs = pd.DataFrame({"id": ids_all[in_cat], "p": peak_lab[in_cat]})
                pairs = pairs.drop_duplicates()
                count = np.bincount(pairs["p"], minlength=n_parcels + 1)[1:]
                total = pairs["id"].nunique()
            else:
                count = np.bincount(peak_lab[in_cat], minlength=n_parcels + 1)[1:]
                total = in_cat.sum()
            z[:, j] = lancaster_z(count, total, n_vox, n_vox.sum())
        out[cats] = z
        out.to_csv(cache, index=False)
        pd.Series(n_studies, name="n_studies").to_csv(
            str(cache).replace(".csv", "_n_studies.csv"))
        out.attrs["source"] = source
        return out

    from nimare.decode.discrete import brainmap_decode
    z = np.zeros((n_parcels, len(cats)))
    for r in range(1, n_parcels + 1):
        ids = sorted(set(coords["id"].to_numpy()[peak_lab == r]))
        if not ids:
            continue
        res = brainmap_decode(foci, ann, ids, features=cats,
                              frequency_threshold=threshold, correction=None)
        z[r - 1] = res.reindex(cats)["zForward"].to_numpy(float)
    out[cats] = np.nan_to_num(z)
    out.to_csv(cache, index=False)
    pd.Series(n_studies, name="n_studies").to_csv(
        nsd / f"{key}_nimare_{taxonomy}_n_studies.csv")
    out.attrs["source"] = source
    return out


# The BrainMap database itself, through the official Behavioral Analysis
# plugin of Mango (Lancaster et al. 2012), run headless on each parcel by
# tools/brainmap/run_brainmap_profiles.sh. The plugin names some leaves
# differently from BRAINMAP_TAXONOMY; these are the renames. Plugin categories
# with no counterpart here (Emotion.Guilt, Cognition.Memory.Implicit, ...) stay
# in the *_all.csv the tool writes, not in the table the panels use.
BRAINMAP_PLUGIN_RENAME = {
    "Cognition.Somatic": "Cognition.Soma",
    "Cognition.Spatial": "Cognition.Space",
    "Cognition.Temporal": "Cognition.Time",
    "Interoception.Gastrointestinal/Genitourinary": "Interoception.Bladder",
    "Interoception.Respiration Regulation": "Interoception.Air Hunger",
}
# The plugin's general categories are kept under their own names, NOT folded
# into the "... .Other" bins the figures hide: in the BrainMap database they
# are the main motor, somatosensory and visual categories, not leftovers.
# Each sits just before its subcategories; the Neurosynth bin it is compared to
# is the value.
BRAINMAP_DB_GENERAL = {
    "Action.Execution": "Action.Execution.Other",
    "Perception.Somesthesis": "Perception.Somesthesis.Other",
    "Perception.Vision": "Perception.Vision.Other",
}


def brainmap_db_columns(available) -> list:
    """BRAINMAP_TAXONOMY order, the general categories before their children."""
    cols = []
    for c in BRAINMAP_TAXONOMY:
        for g in BRAINMAP_DB_GENERAL:
            if c.startswith(g + ".") and g in available and g not in cols:
                cols.append(g)
        if c in available:
            cols.append(c)
    return cols


def brainmap_db_table(data: Path, key: str, group: str = "normals",
                      reverse: bool = False) -> pd.DataFrame:
    """Parcels x BrainMap categories from the BrainMap database itself.

    Lancaster et al. (2012)'s regional behavioural analysis, run by the Mango
    Behavioral Analysis plugin (v4.0, BrainMap data of 2023) on each parcel of
    `key`, converted to Talairach with icbm_spm2tal. The value is the plugin's
    z-score: the proportion of a category's foci that fall in the parcel
    against the parcel's share of brain volume. `group` "normals" uses
    experiments on healthy participants only, "all" every experiment.
    Written by tools/brainmap/run_brainmap_profiles.sh as
    <data>/brainmap/<key>_mango_<group>.csv.
    """
    if reverse:
        return _brainmap_db_reverse(data, key, group)
    path = Path(data) / "brainmap" / f"{key}_mango_{group}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run tools/brainmap/run_brainmap_profiles.sh "
                         f"--atlas {key}")
    out = pd.read_csv(path)
    out.attrs["source"] = (f"BrainMap database (Mango Behavioral Analysis v4.0, {group}), "
                           f"Lancaster 2012 z, counted in {key} parcels")
    return out


def _brainmap_db_reverse(data: Path, key: str, group: str) -> pd.DataFrame:
    """Reverse inference from the plugin's raw foci counts (`reverse_z`).

    The plugin gives no count of all foci in a parcel, so the parcel's total
    is the sum of its counts over the leaf categories (a focus coded with two
    categories counts twice there and in the atlas-wide share alike).
    """
    path = Path(data) / "brainmap" / f"{key}_mango_long.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run tools/brainmap/run_brainmap_profiles.sh")
    long = pd.read_csv(path)
    cnt = (long.query("group == @group and measure == 'count'")
           .pivot(index="parcel", columns="category", values="value"))
    # Leaves only: a general category (Action.Execution) is a separate code
    # in the plugin, not the sum of its children, so it stays a leaf too.
    cats = brainmap_db_columns(cnt.columns)
    total = cnt[cats].sum(axis=1).to_numpy(float)
    out = pd.DataFrame({"parcel": cnt.index.to_numpy()})
    for c in cats:
        out[c] = reverse_z(cnt[c].to_numpy(float), total)
    out.attrs["source"] = (f"BrainMap database (Mango Behavioral Analysis v4.0, {group}), "
                           f"reverse inference on foci counts, counted in {key} parcels")
    return out


def lancaster_z(count, n_total, n_vox, n_vox_total):
    """The z of the Mango Behavioral Analysis plugin (Lancaster et al. 2012).

    p  = count / n_total          share of the category's foci inside the ROI
    pe = n_vox / n_vox_total      share of the volume the ROI takes
    z  = (p - pe) / sqrt((p(1-p) + pe(1-pe)) / n_total)

    Read off the plugin's convertToZscore (v4.0), so the Neurosynth profiles
    can be computed with exactly the statistic the BrainMap ones use. NaN for
    a category with no foci.
    """
    count = np.asarray(count, float)
    if n_total <= 0:
        return np.full(count.shape, np.nan)
    p = count / n_total
    pe = np.asarray(n_vox, float) / float(n_vox_total)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (p - pe) / np.sqrt((p * (1 - p) + pe * (1 - pe)) / n_total)


def reverse_z(count, total):
    """Reverse inference per parcel: p = count/total (the category's share of
    what is reported in the parcel) against p0, the same share over the whole
    atlas; z = (p - p0) / sqrt((p(1-p) + p0(1-p0)) / total), the plugin's
    form with the parcel's own total as n. NaN for a parcel with nothing."""
    count, total = np.asarray(count, float), np.asarray(total, float)
    p0 = count.sum() / total.sum() if total.sum() > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        p = count / total
        z = (p - p0) / np.sqrt((p * (1 - p) + p0 * (1 - p0)) / total)
    return np.where(total > 0, z, np.nan)


def load_behavioural_table(path: Path) -> pd.DataFrame:
    """The real profiles: CSV keyed by `bn_index` (1..246, BNA numbering),
    one column per BrainMap category named by its dotted path
    (`Perception.Vision.Motion`), any monotone score (z, LR, 0/1)."""
    t = pd.read_csv(path)
    if "bn_index" not in t:
        raise SystemExit(f"{path}: needs a `bn_index` column (1..246, BNA numbering)")
    return t


def behavioural_maps(data: Path, ref, table: pd.DataFrame | None = None,
                     seed: int = 0, labels: np.ndarray | None = None) -> ReferenceMaps:
    """The table painted onto voxels by the labels it was counted in.

    With `labels` (the states' own atlas) the table must have a `parcel`
    column, and parcellating the painted maps with the same atlas gives back
    each parcel's value exactly. Without, the labels are Brainnetome's and the
    table is keyed by `bn_index` (Fan et al.'s table, or the simulation).
    """
    simulated = table is None
    if simulated:
        table = simulate_behavioural_table(brainnetome_table(data), seed=seed)
    cats = [c for c in table.columns if "." in c or c in ("Action", "Cognition", "Emotion",
                                                           "Interoception", "Perception")]
    # The "... .Other" bins are the general categories (motor execution,
    # somesthesis, vision): shown under their general name, so the figures,
    # which hide leftovers, keep them. Same names as the BrainMap database.
    table = table.rename(columns={c: c[:-len(".Other")] for c in cats if c.endswith(".Other")})
    cats = [c[:-len(".Other")] if c.endswith(".Other") else c for c in cats]
    if labels is not None:
        lab = np.asarray(labels).astype(int).reshape(-1)
        lut = np.full((int(lab.max()) + 1, len(cats)), np.nan)
        lut[table["parcel"].to_numpy(int)] = table[cats].to_numpy(float)
        vox = lut[lab].T
        return ReferenceMaps("behavioural", cats, vox, simulated=False,
                             source=table.attrs.get("source", "Neurosynth v7 forward inference"))
    lut = np.full((247, len(cats)), np.nan)
    lut[table["bn_index"].to_numpy(int)] = table[cats].to_numpy(float)
    lab = brainnetome_labels(data, ref)
    vox = lut[lab].T                                    # (n_cats, V); label 0 -> NaN
    src = ("SIMULATED stand-in on the real Brainnetome atlas (246 incl. subcortex)"
           if simulated else table.attrs.get(
               "source", "BrainMap behavioural domains on Brainnetome (Fan et al. 2016)"))
    return ReferenceMaps("behavioural", cats, vox, simulated=simulated, source=src)


def behavioural_domains(leaves: ReferenceMaps) -> ReferenceMaps:
    """The same profiles one level up the taxonomy: each BrainMap domain
    (Action, Cognition, ...) is the mean of its subcategories, voxel by voxel.
    One table, two depths -- the panels differ in grain, never in data."""
    domains = list(dict.fromkeys(n.split(".")[0] for n in leaves.names))
    with warnings.catch_warnings():     # voxels outside Brainnetome: all-NaN, stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        vox = np.stack([np.nanmean(leaves.voxels[[n.split(".")[0] == d for n in leaves.names]], 0)
                        for d in domains])
    return ReferenceMaps("behavioural_domain", domains, vox, simulated=leaves.simulated,
                         source=leaves.source + ", domain level")


def subcortical_structures(data: Path, ref) -> dict:
    """Brainnetome's four deep gyri as indicator volumes -- for the
    simulation's ground truth, so simulated states have subcortex too."""
    bn = brainnetome_table(data)
    lab = brainnetome_labels(data, ref)
    return {g: np.isin(lab, bn.loc[bn["gyrus"] == g, "bn_index"]).astype(float)
            for g in BN_SUBCORTICAL}


# ------------------------------------------------------------- statistics ---
def correlate(states: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """(K, P) x (..., M, P) -> (..., K, M) Pearson r over parcels, NaN-safe
    per pair: a parcel missing from either side drops out of that pair."""
    s = np.asarray(states, float)[:, None, :]
    m = np.asarray(maps, float)[..., None, :, :]
    ok = np.isfinite(s) & np.isfinite(m)
    n = ok.sum(-1)
    ms = np.where(ok, s, 0.0).sum(-1) / n
    mm = np.where(ok, m, 0.0).sum(-1) / n
    ds = np.where(ok, s - ms[..., None], 0.0)
    dm = np.where(ok, m - mm[..., None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ds * dm).sum(-1) / np.sqrt((ds ** 2).sum(-1) * (dm ** 2).sum(-1))


def fdr_bh(p: np.ndarray) -> np.ndarray:
    flat = np.asarray(p, float).ravel()
    out = np.full_like(flat, np.nan)
    ok = np.isfinite(flat)
    f = flat[ok]
    order = np.argsort(f)
    ranked = f[order] * len(f) / np.arange(1, len(f) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty_like(f)
    tmp[order] = np.minimum(q, 1.0)
    out[ok] = tmp
    return out.reshape(np.shape(p))


NULLS = ("moran", "burt")


def _moran_surrogates(x: np.ndarray, D: np.ndarray, n: int, rng) -> np.ndarray:
    """Moran spectral randomisation, singleton procedure (Wagner & Dray 2015,
    as in BrainSpace). The map is written in the eigenbasis of the doubly-
    centred inverse-distance matrix (Moran eigenvector maps); flipping the
    sign of each coefficient at random keeps its projection on every spatial
    scale, hence its Moran's I, exactly -- and costs one matrix product."""
    W = np.zeros_like(D)
    off = D > 0
    W[off] = 1.0 / D[off]
    C = np.eye(len(x)) - 1.0 / len(x)
    lam, V = np.linalg.eigh(C @ W @ C)
    V = V[:, np.abs(lam) > 1e-10]
    a = V.T @ (x - x.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(a)))
    return x.mean() + (signs * a) @ V.T


def surrogates(parcel_maps: np.ndarray, centroids: np.ndarray, n: int,
               seed: int = 0, method: str = "moran") -> np.ndarray:
    """(M, P) -> (n, M, P) surrogates that keep each map's spatial autocorrelation.

    `moran` (default): Moran spectral randomisation, milliseconds per map.
    `burt`: variogram matching (Burt et al. 2020, brainsmash), the usual choice
    for a final analysis -- ~45 s per map at 111 parcels, ~2.5 min at 254, so
    a full run is hours: a cluster job, not a pilot.

    Parcels a map has no value for stay NaN in every surrogate; a constant map
    (a network with no voxel in this atlas) gets NaN surrogates and no p.
    """
    from scipy.spatial.distance import cdist

    if method not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {method!r}")
    if method == "burt":
        from brainsmash.mapgen.base import Base

        if not hasattr(np, "row_stack"):   # brainsmash 0.11 still calls it; numpy >= 2.4 dropped it
            np.row_stack = np.vstack
    rng = np.random.default_rng(seed)
    M, P = parcel_maps.shape
    out = np.full((n, M, P), np.nan)
    for m in range(M):
        x = parcel_maps[m]
        ok = np.isfinite(x)
        if ok.sum() < MIN_PARCELS_FOR_NULL or np.nanstd(x) == 0:
            continue
        D = cdist(centroids[ok], centroids[ok])
        if method == "moran":
            out[:, m, ok] = _moran_surrogates(x[ok], D, n, rng)
        else:
            out[:, m, ok] = Base(x=x[ok], D=D, seed=seed + m, resample=True)(n=n)
    return out


def null_test(states: np.ndarray, parcel_maps: np.ndarray, centroids: np.ndarray,
              n: int, seed: int = 0, method: str = "moran", chunk: int = 500) -> dict:
    """r, two-sided p against surrogates, BH q within the family.

    The surrogates are drawn and scored `chunk` at a time: 10,000 of them for
    45 maps over 254 parcels would otherwise be gigabytes held at once just
    to count exceedances.
    """
    r = correlate(states, parcel_maps)
    if n <= 0 or parcel_maps.shape[1] < MIN_PARCELS_FOR_NULL:
        nan = np.full_like(r, np.nan)
        return {"r": r, "p": nan, "q": nan}
    exceed = np.zeros(r.shape)
    valid = np.ones(r.shape, dtype=bool)
    for k, start in enumerate(range(0, n, chunk)):
        m = min(chunk, n - start)
        null = correlate(states, surrogates(parcel_maps, centroids, m, seed + 7919 * k, method))
        valid &= np.isfinite(null).all(0)
        exceed += (np.abs(null) >= np.abs(r)[None]).sum(0)
    p = (1 + exceed) / (1 + n)
    p = np.where(valid & np.isfinite(r), p, np.nan)
    return {"r": r, "p": p, "q": fdr_bh(p)}


# ---------------------------------------------------------------- surface ---
@dataclass
class Surface:
    pial: dict
    white: dict
    sphere: dict
    cortex: dict


def load_surface(data: Path) -> Surface:
    def mesh(p):
        g = nib.load(str(p))
        return g.darrays[0].data.astype(float), g.darrays[1].data.astype(int)

    pial, white, sphere, cortex = {}, {}, {}, {}
    for h in HEMIS:
        pial[h] = mesh(_tf(data, f"tpl-fsaverage_hemi-{h}_den-10k_pial.surf.gii"))
        white[h] = mesh(_tf(data, f"tpl-fsaverage_hemi-{h}_den-10k_white.surf.gii"))
        sphere[h] = mesh(_tf(data, f"tpl-fsaverage_hemi-{h}_den-10k_sphere.surf.gii"))
        cortex[h] = np.asarray(nib.load(str(_tf(
            data, f"tpl-fsaverage_hemi-{h}_den-10k_atlas-Yeo2011_seg-7n_dseg.label.gii"
        ))).darrays[0].data) > 0
    return Surface(pial, white, sphere, cortex)


def vertex_labels(parc: Parcellator, surf: Surface) -> dict:
    """Parcel (1..P, 0 = none) per fsaverage5 vertex, for drawing a state.

    Sampled between white and pial with the most frequent label. For a label
    atlas two repairs follow, both near the midline or label edges: a vertex
    holding the other hemisphere's parcel is cleared, and every cleared or
    unlabelled cortical vertex takes its nearest labelled neighbour on the
    sphere. Sphere atlases are left as they are -- 5 mm balls are meant to
    look like balls, not to tile the cortex.
    """
    from nilearn.surface import vol_to_surf
    from scipy.spatial import cKDTree

    lab_img = nib.Nifti1Image(parc.label_volume().astype(np.float32), parc.ref.affine)
    hemi_of = np.concatenate([["-"], parc.spec.labels["hemi"].astype(str).to_numpy()])
    out = {}
    for h in HEMIS:
        # Meshes as (coords, faces) pairs: every nilearn version takes them.
        # "nearest_most_frequent" is >= 0.11; plain "nearest" before that.
        try:
            v = vol_to_surf(lab_img, list(surf.pial[h]), inner_mesh=list(surf.white[h]),
                            interpolation="nearest_most_frequent")
        except ValueError:
            # Old nilearn would AVERAGE nearest labels over depths -- labels that
            # do not exist. One sample at mid-thickness keeps them integers.
            v = vol_to_surf(lab_img, list(surf.pial[h]), inner_mesh=list(surf.white[h]),
                            interpolation="nearest", kind="depth", depth=[0.5])
        lab = np.rint(np.nan_to_num(v)).astype(int)
        if parc.spec.kind == "labels":
            wrong = np.isin(hemi_of[lab], ["L", "R"]) & (hemi_of[lab] != h)
            lab[wrong] = 0
            have = surf.cortex[h] & (lab > 0)
            need = surf.cortex[h] & (lab == 0)
            if need.any() and have.any():
                xyz = surf.sphere[h][0]
                _, nn = cKDTree(xyz[have]).query(xyz[need])
                lab[need] = lab[have][nn]
        lab[~surf.cortex[h]] = 0
        out[h] = lab
    return out


def short(name: str) -> str:
    """`Perception.Vision.Motion` -> `Vision · Motion`; one level -> itself."""
    parts = name.split(".")
    return " · ".join(parts[1:]) if len(parts) > 1 else name
