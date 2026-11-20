"""The brain states' names: set once (notebook 04), read by every figure (notebooks 05, 07, 08).

    sn.save({0: "Social Cognition", ...}, root, "yeo7", -1, "ThresholdCluster_umap3_8")
    sn.use_for(root, "yeo7", -1, "ThresholdCluster_umap3_8", mode="name")   # then draw

Stored per state set -- a name belongs to one set's states -- as
<outputs>/modelling/version=<v>/unsupervised_model/state_names/atlas=/window_s=/states=<set>.json,
keys = the state index (0 .. K-1, the S<k> of the figures).

`mode`, for every figure drawn after use()/use_for():
    "number"  S3                         (as before; also when no names are set)
    "name"    Somesthesis & Action Execution
    "both"    S3 · Somesthesis & Action Execution
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MODES = ("number", "name", "both")
_NAMES: dict[int, str] = {}
_MODE = "number"


def path(root, atlas, window_s, states, version=None) -> Path:
    from fmri_decomposition.io import _fmt_window, unsupervised_root
    return (unsupervised_root(root, version) / "state_names" / f"atlas={atlas}"
            / f"window_s={_fmt_window(float(window_s))}" / f"states={states}.json")


def save(names: dict, root, atlas, window_s, states, version=None) -> Path:
    """Write {state index: name} for one state set; returns the file."""
    p = path(root, atlas, window_s, states, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({str(int(k)): str(v) for k, v in sorted(names.items(), key=lambda x: int(x[0]))},
                            indent=1, ensure_ascii=False))
    return p


def load(root, atlas, window_s, states, version=None) -> dict | None:
    p = path(root, atlas, window_s, states, version)
    if not p.exists():
        return None
    return {int(k): v for k, v in json.loads(p.read_text()).items()}


def use(names: dict | None = None, mode: str = "number") -> None:
    """Make every figure drawn from now on label the states by `mode` with `names`."""
    global _NAMES, _MODE
    if mode not in MODES:
        raise SystemExit(f"state label mode must be one of {MODES}, got {mode!r}")
    _NAMES = {int(k): v for k, v in (names or {}).items()}
    _MODE = mode if _NAMES else "number"


def use_for(root, atlas, window_s, states, version=None, mode: str = "name") -> dict | None:
    """load() the set's names (notebook 04 writes them) and use() them; with none saved, numbers."""
    names = load(root, atlas, window_s, states, version)
    if names is None and mode != "number":
        print(f"no state names saved for {atlas} / {window_s} / {states} (notebook 04 writes them): numbers")
    use(names, mode)
    if names:
        print(f"state labels: {mode} ·", ", ".join(f"S{k} {v}" for k, v in sorted(names.items())))
    return names


def label(k, mode: str | None = None) -> str:
    """S<k>, its name, or both, as set by use()."""
    k = int(k)
    mode = mode or _MODE
    name = _NAMES.get(k)
    if mode == "number" or name is None:
        return f"S{k}"
    return name if mode == "name" else f"S{k} · {name}"


def active() -> bool:
    """True when figures should show names (mode name / both, with names set)."""
    return _MODE != "number" and bool(_NAMES)


_CELL = re.compile(r"^(\d+)->(\d+)$")


def var_label(v: str) -> str:
    """A model variable as the figures write it: occ_3 -> "Occupancy S3" (or its name), 3->7 ->
    "S3 → S7"; anything else unchanged."""
    v = str(v)
    if v.startswith("occ_") and v[4:].isdigit():
        return f"Occupancy {label(v[4:])}"
    m = _CELL.match(v)
    if m:
        return f"{label(m[1])} → {label(m[2])}"
    return v
