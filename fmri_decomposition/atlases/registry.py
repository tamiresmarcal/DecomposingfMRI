"""AtlasSpec: the one contract every atlas satisfies.

Stage 3 never knows which atlas it is reading. Stage 2 never knows how an
atlas was built. Both talk to `AtlasSpec`, which exposes:

  * `labels`  -- a table with index, name, hemi, x, y, z, network
  * `columns` -- the canonical, ordered parcel column names
  * `membership(reference_img, mask)` -- a sparse (n_nodes, n_voxels) matrix

Reducing both label atlases and sphere atlases to a membership matrix is what
makes extraction a single code path. Parcel averaging is a fixed linear
operation either way; overlapping spheres are just a matrix with overlapping
support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

LABEL_COLUMNS = ["index", "name", "hemi", "x", "y", "z", "network"]


def clean_name(name: str) -> str:
    """Parcel name -> column name. Must round-trip to a valid identifier-ish."""
    s = str(name).strip()
    for a, b in ((" ", "_"), (",", ""), ("'", ""), ("(", ""), (")", ""), ("/", "-")):
        s = s.replace(a, b)
    return s


@dataclass
class AtlasSpec:
    name: str
    kind: str                                  # "labels" | "spheres"
    labels: pd.DataFrame
    labels_img: Any = None                     # nibabel image, kind == "labels"
    seeds: np.ndarray | None = None            # (n_seeds, 3) MNI mm, kind == "spheres"
    seed_group: np.ndarray | None = None       # node index per seed (aggregation)
    radius_mm: float = 5.0
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [c for c in LABEL_COLUMNS if c not in self.labels.columns]
        if missing:
            raise ValueError(f"atlas {self.name!r} label table missing columns: {missing}")
        if self.labels["index"].duplicated().any():
            raise ValueError(f"atlas {self.name!r} has duplicate label indices")
        cols = self.columns
        if len(set(cols)) != len(cols):
            dupes = sorted({c for c in cols if cols.count(c) > 1})
            raise ValueError(f"atlas {self.name!r} has duplicate column names: {dupes}")

    # ---------------------------------------------------------- shape ---
    @property
    def n_nodes(self) -> int:
        return len(self.labels)

    @property
    def n_edges(self) -> int:
        n = self.n_nodes
        return n * (n - 1) // 2

    @property
    def columns(self) -> list[str]:
        return [clean_name(n) for n in self.labels["name"]]

    def edge_names(self) -> list[str]:
        """Upper-triangle edge names in canonical (row-major, k=1) order.

        The ordering is defined by the label table and nothing else -- this is
        what makes the packed `edges` list column interpretable, and it is the
        reason the label table ships alongside every dataset.
        """
        cols = self.columns
        iu, ju = np.triu_indices(len(cols), k=1)
        return [f"{cols[i]}__{cols[j]}" for i, j in zip(iu, ju)]

    def write_labels(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = self.labels.copy()
        out["column"] = self.columns
        out.to_csv(path, index=False)
        return path

    # ----------------------------------------------------- membership ---
    def membership(self, reference_img, mask: np.ndarray | None = None):
        """Sparse (n_nodes, n_voxels_in_mask) row-normalised averaging matrix.

        `mask` is a boolean array on the reference grid; voxels outside it are
        excluded from every parcel. Rows that end up empty are left all-zero
        and become NaN at extraction time -- never dropped. Silently returning
        fewer columns for one subject is how you get 110 columns among 84 files
        with 111 and an unhelpful error at read time.
        """
        from scipy import sparse

        shape = reference_img.shape[:3]
        if mask is None:
            mask = np.ones(shape, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"mask shape {mask.shape} != reference {shape}")
        flat_mask = mask.reshape(-1)
        vox_index = np.full(flat_mask.size, -1, dtype=np.int64)
        vox_index[flat_mask] = np.arange(flat_mask.sum())
        n_vox = int(flat_mask.sum())

        if self.kind == "labels":
            rows, cols = self._label_support(reference_img, vox_index)
        elif self.kind == "spheres":
            rows, cols = self._sphere_support(reference_img, vox_index)
        else:
            raise ValueError(f"unknown atlas kind {self.kind!r}")

        data = np.ones(len(rows), dtype=np.float64)
        M = sparse.csr_matrix((data, (rows, cols)), shape=(self.n_nodes, n_vox))
        # Row-normalise to a mean. Empty rows stay zero.
        counts = np.asarray(M.sum(axis=1)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(counts > 0, 1.0 / counts, 0.0)
        return sparse.diags(inv) @ M, counts

    def _label_support(self, reference_img, vox_index):
        from .resample import resample_labels_to

        lab = resample_labels_to(self.labels_img, reference_img)
        lab_flat = np.asarray(lab.dataobj).astype(np.int32).reshape(-1)
        rows, cols = [], []
        for node_i, label_id in enumerate(self.labels["index"].to_numpy()):
            hits = np.flatnonzero(lab_flat == int(label_id))
            hits = hits[vox_index[hits] >= 0]
            if hits.size:
                rows.append(np.full(hits.size, node_i, dtype=np.int64))
                cols.append(vox_index[hits])
        if not rows:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(rows), np.concatenate(cols)

    def _sphere_support(self, reference_img, vox_index):
        shape = reference_img.shape[:3]
        affine = reference_img.affine
        inv = np.linalg.inv(affine)
        # Voxel centres in world (mm) coordinates.
        gi, gj, gk = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        ijk1 = np.stack([gi.ravel(), gj.ravel(), gk.ravel(), np.ones(gi.size)], axis=0)
        world = (affine @ ijk1)[:3].T                       # (n_vox_all, 3)

        rows, cols = [], []
        r2 = float(self.radius_mm) ** 2
        # Bounding box per seed keeps this cheap even for 254 seeds.
        vox_size = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        pad = np.ceil(self.radius_mm / vox_size).astype(int) + 1
        for seed_i, seed in enumerate(np.asarray(self.seeds, dtype=float)):
            centre_ijk = (inv @ np.append(seed, 1.0))[:3]
            lo = np.maximum(np.floor(centre_ijk - pad).astype(int), 0)
            hi = np.minimum(np.ceil(centre_ijk + pad).astype(int) + 1, shape)
            if np.any(lo >= hi):
                continue
            sub = np.ravel_multi_index(
                np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"),
                shape,
            ).ravel()
            d2 = ((world[sub] - seed) ** 2).sum(axis=1)
            hits = sub[d2 <= r2]
            hits = hits[vox_index[hits] >= 0]
            if not hits.size:
                continue
            node_i = int(self.seed_group[seed_i]) if self.seed_group is not None else seed_i
            rows.append(np.full(hits.size, node_i, dtype=np.int64))
            cols.append(vox_index[hits])
        if not rows:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(rows), np.concatenate(cols)

    # ---------------------------------------------------- constructors ---
    @classmethod
    def from_label_image(cls, name: str, labels_img, names: Sequence[str],
                         indices: Sequence[int] | None = None,
                         network: str | Sequence[str] = "unknown",
                         provenance: dict | None = None) -> "AtlasSpec":
        indices = list(range(1, len(names) + 1)) if indices is None else list(indices)
        centroids = label_centroids(labels_img, indices)
        net = [network] * len(names) if isinstance(network, str) else list(network)
        table = pd.DataFrame(
            {
                "index": indices,
                "name": list(names),
                "hemi": [infer_hemi(n) for n in names],
                "x": centroids[:, 0],
                "y": centroids[:, 1],
                "z": centroids[:, 2],
                "network": net,
            }
        )
        return cls(name=name, kind="labels", labels=table, labels_img=labels_img,
                   provenance=provenance or {})


def infer_hemi(name: str) -> str:
    n = str(name).lower()
    if n.startswith("left") or n.endswith(" l") or "_left" in n:
        return "L"
    if n.startswith("right") or n.endswith(" r") or "_right" in n:
        return "R"
    return "B"


def label_centroids(labels_img, indices: Sequence[int]) -> np.ndarray:
    """Centre of mass of each label, in world (MNI mm) coordinates."""
    data = np.asarray(labels_img.dataobj).astype(np.int32)
    affine = labels_img.affine
    out = np.full((len(indices), 3), np.nan)
    for i, idx in enumerate(indices):
        ijk = np.argwhere(data == int(idx))
        if ijk.size == 0:
            continue
        mean_ijk = ijk.mean(axis=0)
        out[i] = (affine @ np.append(mean_ijk, 1.0))[:3]
    return out


# ------------------------------------------------------------ registry ---
def get_atlas(name: str, **params) -> AtlasSpec:
    """name -> AtlasSpec. The only place atlas names are resolved.

    A user-supplied parcellation is a first-class citizen, not a fork: pass
    `labels_img` (+ optional `labels_csv`) for a label image, or `csv_path`
    for coordinates, and the atlas satisfies the same contract as the built-in
    ones.
    """
    from . import labels as _labels
    from . import spheres as _spheres

    if "labels_img" in params:
        return custom_label_atlas(name=name, **params)
    if "csv_path" in params and name.lower().replace("_", "") not in (
        "networks", "networks14", "coordnetworks", "networksnodes", "networks254", "nodes"
    ):
        spec = _spheres.coordinate_networks(**params)
        spec.name = name           # caller's name wins; it keys the output path
        return spec

    key = name.lower().replace("-", "").replace("_", "")
    if key in ("harvardoxford", "ho", "harvardoxford111"):
        return _labels.harvard_oxford(**params)
    if key in ("harvardoxford69", "ho69", "legacy69"):
        return _labels.harvard_oxford_legacy69(**params)
    if key in ("yeo7", "yeo2011", "yeo"):
        return _labels.yeo7(**params)
    if key in ("networks", "networks14", "coordnetworks"):
        params.setdefault("aggregate", "network")
        return _spheres.coordinate_networks(**params)
    if key in ("networksnodes", "networks254", "nodes"):
        params["aggregate"] = "node"
        return _spheres.coordinate_networks(**params)
    raise KeyError(
        f"unknown atlas {name!r}. Known: harvardoxford, harvardoxford69, yeo7, "
        "networks, networks_nodes"
    )


def custom_label_atlas(name: str, labels_img, labels_csv: str | Path | None = None,
                       names: Sequence[str] | None = None,
                       indices: Sequence[int] | None = None,
                       network: str = "unknown") -> AtlasSpec:
    """Build an AtlasSpec from a label NIfTI plus a name list or CSV."""
    import nibabel as nib

    img = labels_img if hasattr(labels_img, "get_fdata") else nib.load(str(labels_img))
    if labels_csv is not None:
        tbl = pd.read_csv(labels_csv)
        names = tbl["name"].tolist()
        indices = tbl["index"].tolist() if "index" in tbl else None
        network = tbl["network"].tolist() if "network" in tbl else network
    if names is None:
        found = sorted(int(v) for v in np.unique(np.asarray(img.dataobj)) if v != 0)
        indices, names = found, [f"parcel_{v:03d}" for v in found]
    return AtlasSpec.from_label_image(
        name=name, labels_img=img, names=names, indices=indices, network=network,
        provenance={"source": "custom", "labels_csv": str(labels_csv)},
    )


AVAILABLE = ["harvardoxford", "harvardoxford69", "yeo7", "networks", "networks_nodes"]
