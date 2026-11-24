#!/usr/bin/env python3
"""Figure 2 and Supplementary Fig. 1 of notebook 04, re-run with another spatial null (Burt).

    python tools/notebook04_null.py --null burt                # what slurm/neurobio_null.sbatch runs
    python tools/notebook04_null.py --null burt --n-nulls 1000 # a quicker check

The figures must be IDENTICAL to the notebook's except for the null, so nothing here re-states
the notebook's parameters: the script executes notebook 04's own cells (setup, Parameters,
state names, the correlation tables, Plot 4.0, the first Plot 4.1 and Plot 4.2) in order, and
overrides only NULL (and N_NULLS, if given) right after the Parameters cell. Change a parameter in
the notebook and this picks it up.

Nothing of the Moran run is overwritten: the correlation tables go to a `_<null>` run directory
(correlations.compute adds the suffix) and the matrices get a `_<null>` stem. The figures are
copied, under the notebook's plot numbers, to notebooks/figures/04_<null>/:
    4_0.png    Figure 2 (Plot 4.0)
    4_1.png    states x BrainMap        (Supplementary Fig. 1a)
    4_2.png    states x Yeo 7           (Supplementary Fig. 1b, left)
    4_2_2.png  states x molecular       (Supplementary Fig. 1b, right)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "04_BrainStates_Neurobiological.ipynb"


def cells(nb) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def pick(code: list[str], start: str, nth: int = 0) -> str:
    hits = [c for c in code if c.lstrip().startswith(start)]
    if len(hits) <= nth:
        raise SystemExit(f"notebook 04 has no code cell starting with {start!r} (#{nth + 1})")
    return hits[nth]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--null", default="burt", choices=["burt", "moran"])
    ap.add_argument("--n-nulls", type=int, default=None, help="default: the notebook's N_NULLS")
    ap.add_argument("--out", type=Path, default=None, help="default: notebooks/figures/04_<null>")
    a = ap.parse_args()
    out = a.out or REPO / "notebooks" / "figures" / f"04_{a.null}"
    out.mkdir(parents=True, exist_ok=True)

    if a.null == "burt":
        import brainsmash  # noqa: F401  -- fail now, not after hours of tables
    try:                                         # the cells display images; outside Jupyter, a no-op
        import IPython.display  # noqa: F401
    except ImportError:
        mod = types.ModuleType("IPython.display")
        mod.Image = lambda filename=None, **k: filename
        mod.display = lambda *x, **k: None
        sys.modules.setdefault("IPython", types.ModuleType("IPython"))
        sys.modules["IPython.display"] = mod

    code = cells(json.loads(NOTEBOOK.read_text()))
    ns: dict = {"__name__": "notebook04"}
    import os
    os.chdir(REPO / "notebooks")                 # the notebook's working directory (REPO = cwd.parent)

    def run(src: str, what: str):
        print(f"--- {what}", flush=True)
        exec(compile(src, f"<notebook 04: {what}>", "exec"), ns)

    run(pick(code, "import sys"), "setup")
    ns["display"] = lambda *x, **k: None
    run(pick(code, "STATES"), "Parameters")
    ns["NULL"] = a.null
    if a.n_nulls:
        ns["N_NULLS"] = a.n_nulls
    print(f"NULL={ns['NULL']}  N_NULLS={ns['N_NULLS']}", flush=True)
    run(pick(code, "STATE_NAMES"), "state names")
    run(pick(code, "dirs = sc.compute"), "correlation tables")

    run(pick(code, "# Plot 4.0"), "Plot 4.0 (Figure 2)")
    shutil.copy(ns["paths"][0], out / "4_0.png")
    run(pick(code, "# Plot 4.1"), "Plot 4.1 (states x BrainMap)")
    shutil.copy(ns["png"], out / "4_1.png")

    shown = []                                   # Plot 4.2 draws one matrix per family
    ns["display"] = lambda img, *x, **k: shown.append(img)
    ns["Image"] = lambda filename=None, **k: filename
    run(pick(code, "# Plot 4.2"), "Plot 4.2 (states x Yeo 7, molecular)")
    for k, f in enumerate(shown):
        shutil.copy(f, out / ("4_2.png" if k == 0 else f"4_2_{k + 1}.png"))
    print("figures ->", out)
    for f in sorted(out.glob("*.png")):
        print("  ", f.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
