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


def load_coordinate_table(csv_path: str | Path) -> pd.DataFrame:
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


def coordinate_networks(csv_path: str | Path, aggregate: str = "network",
                        radius_mm: float = 5.0) -> AtlasSpec:
    """Build a sphere atlas from a coordinate table.

    aggregate="network" -> one signal per network (14 nodes, 91 edges)
    aggregate="node"    -> one signal per coordinate (254 nodes, 32,131 edges,
                           which crosses the column-per-edge threshold and is
                           stored as a packed list column)
    """
    if aggregate not in ("network", "node"):
        raise ValueError(f"aggregate must be 'network' or 'node', got {aggregate!r}")
    df = load_coordinate_table(csv_path)
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
            "source_csv": str(csv_path),
            "aggregate": aggregate,
            "radius_mm": radius_mm,
            "n_seeds": int(len(df)),
        },
    )
