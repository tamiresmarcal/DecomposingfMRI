"""Coordinate-defined networks: 254 MNI peaks across 14 networks.

`aggregate` is a flag, not a hardcoded choice. The legacy notebook averaged
nodes within a network unconditionally, which is a defensible default (14
signals, 91 edges, ~2% of Harvard-Oxford's storage) but hides the node-level
option behind an edit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .registry import AtlasSpec

REQUIRED_CSV_COLUMNS = {"network", "x", "y", "z"}

#: Filename of the coordinate table shipped with the package.
BUNDLED_CSV = "mni_space_of_networks.csv"


def bundled_csv_path() -> Path:
    """Path to the packaged coordinate table.

    Unlike Harvard-Oxford and Yeo, this atlas has no fetcher behind it -- the
    CSV *is* the parcellation. Shipping it as package data is what lets
    `get_atlas("networks")` work from the name alone, like the other two.
    """
    from importlib.resources import files

    return Path(str(files("fmri_decomposition.atlases") / "data" / BUNDLED_CSV))


def load_coordinate_table(csv_path: str | Path | None = None) -> pd.DataFrame:
    csv_path = bundled_csv_path() if csv_path is None else Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    missing = REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"coordinate CSV missing columns: {sorted(missing)}")
    for c in ("x", "y", "z"):
        df[c] = df[c].astype(float)
    if "node number" in df.columns:
        df = df.rename(columns={"node number": "node_number"})
    return df


def coordinate_networks(csv_path: str | Path | None = None,
                        aggregate: str = "network",
                        radius_mm: float = 5.0) -> AtlasSpec:
    """Build a sphere atlas from a coordinate table.

    aggregate="network" -> one signal per network (14 nodes, 91 edges)
    aggregate="node"    -> one signal per coordinate (254 nodes, 32,131 edges,
                           which crosses the column-per-edge threshold and is
                           stored as a packed list column)
    """
    if aggregate not in ("network", "node"):
        raise ValueError(f"aggregate must be 'network' or 'node', got {aggregate!r}")
    resolved = bundled_csv_path() if csv_path is None else Path(csv_path)
    df = load_coordinate_table(resolved)
    seeds = df[["x", "y", "z"]].to_numpy(dtype=float)

    if aggregate == "network":
        networks = list(dict.fromkeys(df["network"].tolist()))   # first-appearance order
        net_to_idx = {n: i for i, n in enumerate(networks)}
        seed_group = df["network"].map(net_to_idx).to_numpy(dtype=np.int64)
        centroids = np.stack(
            [seeds[seed_group == i].mean(axis=0) for i in range(len(networks))]
        )
        full = {
            n: (df.loc[df["network"] == n, "name"].iloc[0]
                if "name" in df.columns else n)
            for n in networks
        }
        # Provenance travels with the atlas. The description column carries the
        # citation each network was defined from; dropping it here would mean a
        # reader of meta/atlas-networks_labels.csv has no way back to the source.
        desc = {
            n: (str(df.loc[df["network"] == n, "description"].iloc[0])
                if "description" in df.columns else "")
            for n in networks
        }
        table = pd.DataFrame(
            {
                "index": np.arange(1, len(networks) + 1),
                "name": networks,
                "hemi": "B",                       # a network spans both sides
                "x": centroids[:, 0],
                "y": centroids[:, 1],
                "z": centroids[:, 2],
                "network": networks,
                "n_seeds": [int((seed_group == i).sum()) for i in range(len(networks))],
                "full_name": [full[n] for n in networks],
                "description": [desc[n] for n in networks],
            }
        )
        name = "networks"
    else:
        node_no = (df["node_number"] if "node_number" in df.columns
                   else pd.Series(np.arange(1, len(df) + 1)))
        seed_group = np.arange(len(df), dtype=np.int64)
        table = pd.DataFrame(
            {
                "index": np.arange(1, len(df) + 1),
                "name": [f"{net}_{int(no):03d}" for net, no in zip(df["network"], node_no)],
                "hemi": np.where(seeds[:, 0] < 0, "L", np.where(seeds[:, 0] > 0, "R", "B")),
                "x": seeds[:, 0],
                "y": seeds[:, 1],
                "z": seeds[:, 2],
                "network": df["network"].to_numpy(),
                "full_name": (df["name"].to_numpy() if "name" in df.columns
                              else df["network"].to_numpy()),
                "description": (df["description"].astype(str).to_numpy()
                                if "description" in df.columns else ""),
            }
        )
        name = "networks_nodes"

    return AtlasSpec(
        name=name,
        kind="spheres",
        labels=table,
        seeds=seeds,
        seed_group=seed_group,
        radius_mm=radius_mm,
        provenance={
            "source_csv": str(resolved),
            "bundled": csv_path is None,
            "aggregate": aggregate,
            "radius_mm": radius_mm,
            "n_seeds": int(len(df)),
            "n_networks": int(df["network"].nunique()),
            # The coordinates are MNI peaks from the source publications. The
            # template caveat bites hardest here: a 5 mm sphere is small enough
            # that the few-mm NLin6/NLin2009c offset matters, and many of these
            # seeds sit in subcortex where the offset is largest.
            "coordinate_space": "MNI",
            "template_mismatch_accepted": True,
        },
    )
