"""Reference maps for interpreting brain states, on the states' own parcels.

A brain state, as `fmri-decomp state-means` writes it, is one number per parcel
of the atlas it was fitted on. Interpreting it means correlating that vector
with maps that carry meaning -- receptor densities, behavioural domains,
canonical networks. The maps come in three different shapes:

    molecular    PET volumes, MNI152, mixed grids        (neuromaps / Hansen 2022)
    behavioural  a table over Brainnetome subregions     (BrainMap domains)
    canonical    a label map, fsaverage                  (Yeo 2011, 7 networks)

and the only place all three meet is a cortical surface. So everything goes to
fsaverage5 (10,242 vertices per hemisphere), and from there into the parcels of
the state atlas. The correlation is computed across those parcels: one value
per parcel on both sides, never voxels against parcels.

WHY A SPIN TEST
---------------
Two smooth brain maps correlate far more often than two random vectors of the
same length, because neighbouring parcels are not independent. A parametric
p-value over ~96 parcels treats them as 96 independent observations and is
badly anticonservative. The null here is the spin test (Alexander-Bloch et al.
2018): rotate the reference map on the sphere, which keeps its spatial
autocorrelation and destroys its alignment with the state, re-parcellate, and
correlate again. The right hemisphere gets the mirror image of the left one's
rotation so the two stay homologous. Vertices that rotate in from the medial
wall carry no value and drop out of the parcel means.

Cortex only. Subcortical parcels of the state atlas have no vertices and are
ignored -- they are reported, not silently dropped.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from fmri_decomposition.atlases.registry import clean_name  # noqa: E402

HEMIS = ("L", "R")

# Harvard-Oxford cortical labels in atlas order. TemplateFlow's HOCPAL table
# lists only 52 of the 96 lateralised labels, so the names come from here: the
# image interleaves them, value 2i+1 = Left <name i>, 2i+2 = Right <name i>,
# which is the order nilearn's symmetric_split produces and therefore the order
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
YEO7_SHORT = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
# Yeo et al. 2011's own colours -- the field reads these as the networks' names.
YEO7_COLORS = ["#781286", "#4682b4", "#00760e", "#c43afa", "#dcf8a4",
               "#e69422", "#cd3e4e"]

# Display names for the PET files, keyed by the target before the first "_".
RECEPTOR_LABELS = {
    "5HT1a": "5-HT1A", "5HT1b": "5-HT1B", "5HT2a": "5-HT2A", "5HT4": "5-HT4",
    "5HT6": "5-HT6", "5HTT": "5-HTT", "A4B2": "α4β2", "CB1": "CB1", "D1": "D1",
    "D2": "D2", "DAT": "DAT", "GABAa-bz": "GABA-A", "H3": "H3", "M1": "M1",
    "mGluR5": "mGluR5", "MU": "MOR", "NAT": "NET", "NMDA": "NMDA",
    "VAChT": "VAChT",
}
# Grouped by transmitter system, which is how the panel reads best.
RECEPTOR_ORDER = ["5-HT1A", "5-HT1B", "5-HT2A", "5-HT4", "5-HT6", "5-HTT",
                  "D1", "D2", "DAT", "NET", "α4β2", "M1", "VAChT", "GABA-A",
                  "NMDA", "mGluR5", "CB1", "MOR", "H3"]

# BrainMap behavioural (sub)domains shown in the behavioural panel. BrainMap
# has 5 domains and ~50 subdomains; these 12 are the ones a naturalistic-
# viewing state is likely to load on. Whatever the real table carries, the
# panel shows the columns it is given.
BRAINMAP_DOMAINS = [
    "Action.Execution", "Action.Observation", "Cognition.Attention",
    "Cognition.Language", "Cognition.Memory", "Cognition.Reasoning",
    "Cognition.Social", "Emotion", "Interoception", "Perception.Audition",
    "Perception.Somesthesis", "Perception.Vision",
]


# --------------------------------------------------------------- surface ---
@dataclass
class Surface:
    """fsaverage5, both hemispheres, plus the cortex mask (Yeo's medial wall)."""

    pial: dict
    white: dict
    sphere: dict
    cortex: dict                    # hemi -> bool (10242,)

    @property
    def n_vertices(self) -> int:
        return len(self.cortex["L"])


def _tf(data: Path, name: str) -> Path:
    return Path(data) / "templateflow" / name


def _gifti_mesh(path: Path):
    g = nib.load(str(path))
    return (g.darrays[0].data.astype(float), g.darrays[1].data.astype(int))


def _gifti_labels(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).darrays[0].data).astype(int)


def load_surface(data: Path) -> Surface:
    pial, white, sphere, cortex = {}, {}, {}, {}
    for h in HEMIS:
        pial[h] = _gifti_mesh(_tf(data, f"tpl-fsaverage_hemi-{h}_den-10k_pial.surf.gii"))
        white[h] = _gifti_mesh(_tf(data, f"tpl-fsaverage_hemi-{h}_den-10k_white.surf.gii"))
        sphere[h] = _gifti_mesh(_tf(data, f"tpl-fsaverage_hemi-{h}_den-10k_sphere.surf.gii"))
        cortex[h] = _gifti_labels(_tf(
            data, f"tpl-fsaverage_hemi-{h}_den-10k_atlas-Yeo2011_seg-7n_dseg.label.gii")) > 0
    return Surface(pial, white, sphere, cortex)


def vol_to_fsaverage(img, surf: Surface, interpolation: str = "linear",
                     mask_img=None) -> dict:
    """Volume -> {hemi: (10242,)}, sampled between the white and pial surfaces.

    This is nilearn's registration-free projection: fsaverage coordinates are
    taken to be MNI coordinates. It is the standard shortcut and good enough
    for parcels of a few cm; neuromaps' `mni152_to_fsaverage` (registration-
    fusion) is the better tool where it can be fetched.
    """
    from nilearn.surface import InMemoryMesh, vol_to_surf

    out = {}
    for h in HEMIS:
        pial = InMemoryMesh(*surf.pial[h])
        white = InMemoryMesh(*surf.white[h])
        v = vol_to_surf(img, pial, inner_mesh=white, interpolation=interpolation,
                        mask_img=mask_img)
        out[h] = np.asarray(v, dtype=float)
    return out


# ------------------------------------------------------------ state atlas ---
@dataclass
class SurfaceParcellation:
    """A label per cortical vertex, and the parcel names those labels mean."""

    names: list                     # parcel names, in label order (1..n)
    labels: dict                    # hemi -> int (10242,), 0 = not in any parcel

    @property
    def columns(self) -> list:
        return [clean_name(n) for n in self.names]

    def parcellate(self, maps: dict) -> np.ndarray:
        """{hemi: (..., 10242)} -> (..., n_parcels). NaN vertices drop out."""
        n = len(self.names)
        lead = np.shape(maps["L"])[:-1]
        sums = np.zeros(lead + (n + 1,))
        counts = np.zeros(lead + (n + 1,))
        for h in HEMIS:
            v = np.asarray(maps[h], dtype=float)
            ok = np.isfinite(v)
            lab = self.labels[h]
            for arr, w in ((sums, np.where(ok, v, 0.0)), (counts, ok.astype(float))):
                flat = w.reshape(-1, w.shape[-1])
                acc = np.stack([np.bincount(lab, weights=row, minlength=n + 1)
                                for row in flat])
                arr += acc.reshape(lead + (n + 1,))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (sums / counts)[..., 1:]

    def paint(self, values: np.ndarray) -> dict:
        """(n_parcels,) -> {hemi: (10242,)}, NaN off-parcel, for plotting."""
        vals = np.concatenate([[np.nan], np.asarray(values, dtype=float)])
        return {h: vals[self.labels[h]] for h in HEMIS}


def harvard_oxford_on_surface(data: Path, surf: Surface) -> SurfaceParcellation:
    """The cortical 96 of the pipeline's `harvardoxford`, on fsaverage5.

    Two repairs after projection, both of which only matter near the midline
    and the label edges. A vertex that picked up the other hemisphere's label
    (the medial wall is ~1 voxel from the midline) gets its own hemisphere's
    homologue -- the parity bit of the label. A cortical vertex that landed in
    no label at all (thr25 leaves gaps) takes its nearest labelled neighbour
    on the sphere, so every cortical vertex belongs to exactly one parcel.
    """
    from scipy.spatial import cKDTree

    img = nib.load(str(_tf(data, "tpl-MNI152NLin6Asym_res-02_atlas-HOCPAL_desc-th25_dseg.nii.gz")))
    proj = vol_to_fsaverage(img, surf, interpolation="nearest_most_frequent")
    names = [f"{side} {n}" for n in HO_CORTICAL for side in ("Left", "Right")]
    labels = {}
    for h in HEMIS:
        lab = np.rint(np.nan_to_num(proj[h])).astype(int)
        wrong_side = (lab > 0) & ((lab % 2 == 1) != (h == "L"))
        lab[wrong_side] += -1 if h == "L" else 1   # Right 2i+2 <-> Left 2i+1
        cortex = surf.cortex[h]
        have = cortex & (lab > 0)
        need = cortex & (lab == 0)
        if need.any():
            xyz = surf.sphere[h][0]
            _, nn = cKDTree(xyz[have]).query(xyz[need])
            lab[need] = lab[have][nn]
        lab[~cortex] = 0
        labels[h] = lab
    return SurfaceParcellation(names, labels)


# --------------------------------------------------------- reference maps ---
@dataclass
class ReferenceMaps:
    """Vertex-level maps of one family, plus their names and provenance."""

    family: str
    names: list
    vertex: dict                    # hemi -> (n_maps, 10242), NaN off-cortex
    simulated: bool = False
    source: str = ""


def receptor_maps(data: Path, surf: Surface) -> ReferenceMaps:
    """The 19 PET maps, projected, z-scored over cortex per map.

    Several of the volumes are zero outside a brain mask. A zero there is "not
    measured", not "no receptor", so those zeros are masked out of the
    sampling rather than averaged into the cortical ribbon's edge.
    """
    files = {}
    for f in sorted((Path(data) / "pet").glob("*.nii*")):
        key = f.name.split("_")[0]
        files[RECEPTOR_LABELS.get(key, key)] = f
    names = [n for n in RECEPTOR_ORDER if n in files]
    vertex = {h: [] for h in HEMIS}
    for n in names:
        img = nib.load(str(files[n]))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
            img = nib.Nifti1Image(arr, img.affine)
        mask = None
        if (arr != 0).mean() < 0.5:
            mask = nib.Nifti1Image((arr != 0).astype(np.int8), img.affine)
        proj = vol_to_fsaverage(img, surf, mask_img=mask)
        for h in HEMIS:
            proj[h][~surf.cortex[h]] = np.nan
        both = np.concatenate([proj["L"], proj["R"]])
        mu, sd = np.nanmean(both), np.nanstd(both)
        for h in HEMIS:
            vertex[h].append((proj[h] - mu) / sd)
    return ReferenceMaps("molecular", names,
                         {h: np.stack(vertex[h]) for h in HEMIS},
                         source="Hansen et al. 2022 PET atlas (neuromaps MNI152 annotations)")


def yeo7_maps(data: Path, surf: Surface) -> ReferenceMaps:
    """One indicator map per network. Parcellated, it is each parcel's share
    of that network -- the correlation with a state then asks how much the
    state's pattern follows the network's footprint."""
    vertex = {}
    for h in HEMIS:
        lab = _gifti_labels(_tf(
            data, f"tpl-fsaverage_hemi-{h}_den-10k_atlas-Yeo2011_seg-7n_dseg.label.gii"))
        ind = np.stack([(lab == k).astype(float) for k in range(1, 8)])
        ind[:, ~surf.cortex[h]] = np.nan
        vertex[h] = ind
    return ReferenceMaps("canonical", list(YEO7), vertex,
                         source="Yeo et al. 2011, 7 networks (TemplateFlow fsaverage)")


# ------------------------------------------------------------ Brainnetome ---
# Brainnetome cortical subregions per hemisphere, in BNA order, grouped by the
# gyrus they subdivide (Fan et al. 2016, Table 1). Needed to give each
# subregion an identity beyond its abbreviation.
BN_GYRI = [("SFG", 7), ("MFG", 7), ("IFG", 6), ("OrG", 6), ("PrG", 6),
           ("PCL", 2), ("STG", 6), ("MTG", 4), ("ITG", 7), ("FuG", 3),
           ("PhG", 6), ("pSTS", 2), ("SPL", 5), ("IPL", 6), ("Pcun", 4),
           ("PoG", 4), ("INS", 6), ("CG", 7), ("MVOcC", 5), ("LOcC", 6)]


def brainnetome_on_surface(data: Path) -> tuple[pd.DataFrame, dict]:
    """Brainnetome's 210 cortical subregions -> (table, {hemi: labels}).

    TemplateFlow has the atlas on fsaverage (164k) only. fsaverage5 is the
    first 10,242 vertices of fsaverage -- the icosahedra are nested -- so the
    downsampling is a slice, not a resampling. `bn_index` is the 1..246
    numbering of the BNA tables (odd = left, even = right), which is what a
    behavioural table downloaded from the Brainnetome site is keyed by.
    """
    tsv = pd.read_csv(_tf(data, "tpl-fsaverage_atlas-brainnetome_dseg.tsv"), sep="\t")
    gyri = [g for g, n in BN_GYRI for _ in range(n)]
    rows, labels = [], {}
    for h in HEMIS:
        names = tsv.loc[(tsv["hemi"] == h) & ~tsv["name"].str.contains("unknown"), "name"]
        names = list(names)
        if len(names) != 105:
            raise ValueError(f"expected 105 Brainnetome cortical regions in {h}, got {len(names)}")
        for k, n in enumerate(names):
            rows.append({"bn_index": 2 * k + (1 if h == "L" else 2), "hemi": h,
                         "name": n, "gyrus": gyri[k], "local": k + 1})
        g = nib.load(str(_tf(data, f"tpl-fsaverage_hemi-{h}_den-164k_atlas-brainnetome_dseg.label.gii")))
        labels[h] = np.asarray(g.darrays[0].data).astype(int)[:10242]
    return pd.DataFrame(rows).sort_values("bn_index").reset_index(drop=True), labels


def simulate_behavioural_table(bn: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A STAND-IN for the BrainMap behavioural-domain profiles. Not data.

    The real profiles (Fan et al. 2016, per-subregion forward inference over
    BrainMap) are served by atlas.brainnetome.org, which this pilot could not
    reach. This builds a table of the same shape -- one row per subregion, one
    column per domain -- from a textbook prior on what each gyrus does, plus
    noise. It exists so the panel and its statistics can be built and checked;
    every figure that uses it says SIMULATED, and `load_behavioural_table`
    replaces it the moment the real table is on disk.
    """
    prior = {
        "SFG": {"Cognition.Reasoning": .6, "Action.Execution": .4, "Cognition.Social": .3},
        "MFG": {"Cognition.Reasoning": .9, "Cognition.Attention": .7, "Cognition.Memory": .6},
        "IFG": {"Cognition.Language": 1.0, "Cognition.Reasoning": .4, "Action.Observation": .3},
        "OrG": {"Emotion": .8, "Interoception": .5, "Cognition.Reasoning": .3},
        "PrG": {"Action.Execution": 1.0, "Action.Observation": .3},
        "PCL": {"Action.Execution": .8, "Perception.Somesthesis": .6},
        "STG": {"Perception.Audition": 1.0, "Cognition.Language": .6},
        "MTG": {"Cognition.Language": .7, "Cognition.Social": .5, "Cognition.Memory": .3},
        "ITG": {"Perception.Vision": .5, "Cognition.Memory": .4, "Cognition.Language": .3},
        "FuG": {"Perception.Vision": .9, "Cognition.Social": .3},
        "PhG": {"Cognition.Memory": 1.0, "Emotion": .5},
        "pSTS": {"Cognition.Social": 1.0, "Cognition.Language": .5, "Action.Observation": .5},
        "SPL": {"Cognition.Attention": .9, "Action.Observation": .5, "Perception.Somesthesis": .3},
        "IPL": {"Cognition.Attention": .6, "Cognition.Social": .4, "Cognition.Language": .3},
        "Pcun": {"Cognition.Memory": .7, "Cognition.Social": .5},
        "PoG": {"Perception.Somesthesis": 1.0, "Action.Execution": .4},
        "INS": {"Interoception": 1.0, "Emotion": .7, "Perception.Somesthesis": .3},
        "CG": {"Emotion": .7, "Cognition.Attention": .5, "Interoception": .4},
        "MVOcC": {"Perception.Vision": 1.0},
        "LOcC": {"Perception.Vision": .9, "Action.Observation": .4},
    }
    rng = np.random.default_rng(seed)
    out = bn[["bn_index", "hemi", "name", "gyrus"]].copy()
    for d in BRAINMAP_DOMAINS:
        base = out["gyrus"].map(lambda g: prior[g].get(d, 0.0)).to_numpy()
        out[d] = np.clip(base + rng.normal(0, 0.15, len(out)), 0, None)
    out.attrs["simulated"] = True
    return out


def load_behavioural_table(path: Path) -> pd.DataFrame:
    """The real profiles: a CSV keyed by `bn_index` (1..246), one column per
    domain, any monotone score (z, likelihood ratio, 0/1). Subcortical rows
    (211..246) are allowed and ignored -- they have no surface."""
    t = pd.read_csv(path)
    if "bn_index" not in t:
        raise SystemExit(f"{path}: needs a `bn_index` column (1..246, BNA numbering)")
    t = t[t["bn_index"] <= 210].copy()
    t.attrs["simulated"] = False
    return t


BN_LEVELS = ("subregion", "gyrus")


def behavioural_maps(data: Path, surf: Surface, table: pd.DataFrame | None = None,
                     seed: int = 0, level: str = "subregion") -> ReferenceMaps:
    """BrainMap domain profiles painted on Brainnetome, at one of its levels.

    `subregion` is the 210 cortical subregions the profiles are published at.
    `gyrus` is the atlas' own coarser level: the 20 cortical gyri per
    hemisphere (SFG, MFG, ... LOcC) that the subregions subdivide. Each gyrus
    gets the mean profile of its subregions, so both levels come from ONE
    table and differ only in grain -- what changes between the two panels is
    the resolution, never the data.
    """
    if level not in BN_LEVELS:
        raise ValueError(f"level must be one of {BN_LEVELS}, got {level!r}")
    bn, labels = brainnetome_on_surface(data)
    simulated = table is None
    if simulated:
        table = simulate_behavioural_table(bn, seed=seed)
    domains = [c for c in table.columns
               if c not in ("bn_index", "hemi", "name", "gyrus", "local")]
    t = bn[["bn_index", "hemi", "local", "gyrus"]].merge(
        table[["bn_index"] + domains], on="bn_index", how="left")
    if level == "gyrus":
        t[domains] = t.groupby(["hemi", "gyrus"])[domains].transform("mean")
    vertex = {}
    for h in HEMIS:
        th = t[t["hemi"] == h].sort_values("local")
        lut = np.full((106, len(domains)), np.nan)
        lut[th["local"].to_numpy()] = th[domains].to_numpy(float)
        v = lut[labels[h]].T                       # (n_domains, 10242)
        v[:, ~surf.cortex[h]] = np.nan
        vertex[h] = v
    src = ("SIMULATED stand-in on the real Brainnetome parcellation" if simulated
           else "BrainMap behavioural domains on Brainnetome (Fan et al. 2016)")
    src += f", {level} level ({'210 subregions' if level == 'subregion' else '40 gyri'})"
    family = "behavioural" if level == "subregion" else f"behavioural_{level}"
    return ReferenceMaps(family, domains, vertex, simulated=simulated, source=src)


# -------------------------------------------------------------- spin test ---
def spin_indices(surf: Surface, n_spins: int, seed: int = 0) -> dict:
    """{hemi: (n_spins, 10242) int} -- rotated vertex i takes the value of
    original vertex idx[s, i]. Left gets R, right gets R mirrored in x."""
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    rots = Rotation.random(n_spins, random_state=seed).as_matrix()
    flip = np.diag([-1.0, 1.0, 1.0])
    out = {}
    for h in HEMIS:
        xyz = surf.sphere[h][0]
        xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        tree = cKDTree(xyz)
        idx = np.empty((n_spins, len(xyz)), dtype=np.int32)
        for s, R in enumerate(rots):
            Rh = R if h == "L" else flip @ R @ flip
            idx[s] = tree.query(xyz @ Rh.T)[1]
        out[h] = idx
    return out


def null_parcel_maps(ref: ReferenceMaps, parc: SurfaceParcellation,
                     spins: dict) -> np.ndarray:
    """(n_spins, n_maps, n_parcels): each reference map, rotated and
    re-parcellated. Medial-wall vertices are NaN and stay NaN when rotated,
    so a parcel that rotates onto the wall averages over fewer vertices."""
    n_spins = spins["L"].shape[0]
    out = np.empty((n_spins, len(ref.names), len(parc.names)))
    for s in range(n_spins):
        rotated = {}
        for h in HEMIS:
            v = ref.vertex[h][:, spins[h][s]]
            v[:, ~parc.labels[h].astype(bool)] = np.nan
            rotated[h] = v
        out[s] = parc.parcellate(rotated)
    return out


def correlate(states: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """(K, P) x (..., M, P) -> (..., K, M) Pearson r over parcels, NaN-safe
    per pair: a parcel missing from either side drops out of that pair."""
    s = np.asarray(states, float)[:, None, :]          # K,1,P
    m = np.asarray(maps, float)[..., None, :, :]       # ...,1,M,P
    ok = np.isfinite(s) & np.isfinite(m)
    n = ok.sum(-1)
    sx = np.where(ok, s, 0.0)
    mx = np.where(ok, m, 0.0)
    ms = sx.sum(-1) / n
    mm = mx.sum(-1) / n
    ds = np.where(ok, s - ms[..., None], 0.0)
    dm = np.where(ok, m - mm[..., None], 0.0)
    num = (ds * dm).sum(-1)
    den = np.sqrt((ds ** 2).sum(-1) * (dm ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def fdr_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values over every entry of `p`."""
    flat = np.asarray(p, float).ravel()
    order = np.argsort(flat)
    ranked = flat[order] * len(flat) / np.arange(1, len(flat) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(flat)
    out[order] = np.minimum(q, 1.0)
    return out.reshape(np.shape(p))


def spin_test(states: np.ndarray, ref: ReferenceMaps, parc: SurfaceParcellation,
              spins: dict) -> dict:
    """r, two-sided spin p, and BH q within the family (all states x maps)."""
    obs = parc.parcellate(ref.vertex)                    # M, P
    r = correlate(states, obs)                           # K, M
    null = correlate(states, null_parcel_maps(ref, parc, spins))  # S, K, M
    p = (1 + (np.abs(null) >= np.abs(r)[None]).sum(0)) / (1 + null.shape[0])
    return {"r": r, "p": p, "q": fdr_bh(p), "parcel_maps": obs}


def short(name: str) -> str:
    """Compact axis label for long names."""
    return re.sub(r"^(Perception|Cognition|Action)\.", lambda m: m.group(1)[0] + ".", name)
