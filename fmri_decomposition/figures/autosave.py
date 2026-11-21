"""Every figure a notebook shows, written to disk under its plot number, overwritten on every run.

    from fmri_decomposition.figures import autosave
    autosave.enable(REPO / "notebooks" / "figures" / "04")

From then on, a cell whose first line is `# Plot 4.1` writes what it displays as 4_1.png (a second
image of the same cell 4_1_2.png, ...); `# Figure 5a` writes figure_5a.png, `# Figure 4, left half`
figure_4_left_half.png. A second cell with the same title gets _b (4_1_b.png). Interactive plotly
figures are written as .html under the same name. Cells without such a line are left alone.
A line `SAVE_AS = "4_1_version2"` in the cell names its file instead (None = the default name), so a
second version can be kept beside the first.

It listens to what the notebook DISPLAYS (IPython's display publisher), so it catches matplotlib
figures drawn inline, images shown with display(Image(...)) and plotly's .show() alike, without any
figure function knowing about it.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

_TAG = re.compile(r"^\s*#\s*(Plot|Figure)\s+([0-9]+(?:\.[0-9]+)?[a-z]?)\b(,[^(—]*)?", re.I | re.M)
_SAVE_AS = re.compile(r"""^\s*SAVE_AS\s*=\s*(?:["']([^"']+)["']|None)""", re.M)
_state = {"dir": None, "stem": None, "n": 0, "installed": False, "cells": {}}


def _stem(source: str):
    first = next((ln for ln in source.splitlines() if ln.strip()), "")
    m = _TAG.match(first)
    if not m:
        return None
    kind, num = m[1].lower(), m[2].replace(".", "_")
    stem = num if kind == "plot" else f"figure_{num}"
    if m[3]:                                         # "# Figure 4, left half" -> figure_4_left_half
        stem += "_" + "_".join(re.findall(r"[a-z0-9]+", m[3].lower()))
    return stem


def _next_path(ext: str) -> Path:
    _state["n"] += 1
    n = _state["n"]
    return _state["dir"] / (f"{_state['stem']}{'' if n == 1 else f'_{n}'}.{ext}")


def _pre_run_cell(info):
    src = getattr(info, "raw_cell", "") or ""
    stem = _stem(src)
    own = _SAVE_AS.search(src) if stem else None
    if own and own[1]:                               # SAVE_AS = "4_1_version2": the cell names its file
        _state["stem"], _state["n"] = Path(own[1]).stem, 0
        return
    if stem:                                         # two different cells with one title: _b, _c, ...
        cell = getattr(info, "cell_id", None) or (getattr(info, "raw_cell", "") or "")
        seen = _state["cells"].setdefault(stem, [])
        if cell not in seen:
            seen.append(cell)
        k = seen.index(cell)
        stem += "" if k == 0 else "_" + "abcdefghijklmnopqrstuvwxyz"[k]
    _state["stem"] = stem
    _state["n"] = 0


def enable(out_dir) -> Path:
    """Start writing the displayed figures of numbered cells to out_dir (created; files overwritten)."""
    from IPython import get_ipython
    ip = get_ipython()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _state["dir"] = out
    if ip is None:
        print("autosave: not in IPython; nothing is saved")
        return out
    if not _state["installed"]:
        ip.events.register("pre_run_cell", _pre_run_cell)
        pub = ip.display_pub
        original = pub.publish

        def publish(data, metadata=None, *args, **kwargs):
            try:
                if _state["stem"] and _state["dir"] is not None and isinstance(data, dict):
                    if "image/png" in data:
                        png = data["image/png"]
                        _next_path("png").write_bytes(base64.b64decode(png) if isinstance(png, str) else png)
                    elif "application/vnd.plotly.v1+json" in data:
                        import plotly.io as pio
                        fig = pio.from_json(__import__("json").dumps(data["application/vnd.plotly.v1+json"]))
                        fig.write_html(_next_path("html"), include_plotlyjs="cdn")
            except Exception as e:                  # saving must never break the notebook
                print(f"autosave: {e}")
            return original(data, metadata, *args, **kwargs)

        pub.publish = publish
        _state["installed"] = True
    print(f"autosave: figures of '# Plot N.k' / '# Figure N' cells -> {out}")
    return out
