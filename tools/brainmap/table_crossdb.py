"""Supplementary Table: agreement of the reverse-inference profiles of
Neurosynth and of the BrainMap database.

Neurosynth: NiMARE's BrainMap decoder, zReverse (signed two-way chi-square on
studies). BrainMap database: the same test on the foci counts of the Mango
Behavioral Analysis plugin (it gives foci, not studies). Per category, the
Spearman correlation across the atlas's parcels and each database's top
parcel.

    python tools/brainmap/table_crossdb.py --atlas harvardoxford
Writes <data>/brainmap/<atlas>_table_crossdb.{csv,tex}; the .tex is the whole
table environment, to paste over Supplementary Table tab:crossdb.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fmri_decomposition.io import require_reference_maps
from fmri_decomposition.unsupervised_model.contextualization import refmaps as rm

ABBREV = {
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "SMA",
    "Juxtapositional Lobule Cortex": "SMA",
    "Inferior Frontal Gyrus, pars opercularis": "IFG pars opercularis",
    "Inferior Frontal Gyrus, pars triangularis": "IFG pars triangularis",
    "Superior Temporal Gyrus, posterior division": "posterior STG",
    "Superior Temporal Gyrus, anterior division": "anterior STG",
    "Middle Temporal Gyrus, posterior division": "posterior MTG",
    "Middle Temporal Gyrus, anterior division": "anterior MTG",
    "Inferior Temporal Gyrus, anterior division": "anterior ITG",
    "Inferior Temporal Gyrus, posterior division": "posterior ITG",
    "Inferior Temporal Gyrus, temporooccipital part": "temporo-occipital ITG",
    "Lateral Occipital Cortex, superior division": "superior LOC",
    "Lateral Occipital Cortex, inferior division": "inferior LOC",
    "Supramarginal Gyrus, posterior division": "posterior SMG",
    "Supramarginal Gyrus, anterior division": "anterior SMG",
    "Cingulate Gyrus, anterior division": "anterior cingulate",
    "Cingulate Gyrus, posterior division": "posterior cingulate",
    "Parahippocampal Gyrus, posterior division": "posterior parahippocampal",
    "Parahippocampal Gyrus, anterior division": "anterior parahippocampal",
    "Superior Parietal Lobule": "SPL", "Middle Frontal Gyrus": "MFG",
    "Superior Frontal Gyrus": "SFG", "Frontal Orbital Cortex": "orbitofrontal",
    "Central Opercular Cortex": "central operculum", "Insular Cortex": "insula",
    "Heschl's Gyrus (includes H1 and H2)": "Heschl's gyrus",
}
GENERAL = {"Action.Execution": "Execution (general)",
           "Perception.Somesthesis": "Somesthesis (general)",
           "Perception.Vision": "Vision (general)"}
DOMAINS = ["Action", "Cognition", "Emotion", "Interoception", "Perception"]


def short(name: str) -> str:
    h = "L" if name.startswith("Left") else "R" if name.startswith("Right") else ""
    n = re.sub(r"^(Left|Right) ", "", name)
    return f"{h} {ABBREV.get(n, n[0].lower() + n[1:])}".strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas", default="harvardoxford")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--group", default="normals", choices=["normals", "all"])
    a = ap.parse_args(argv)

    data = a.data or require_reference_maps()
    ref = rm.reference_grid(data)
    spec, _ = rm.state_atlas(a.atlas, data)
    parc = rm.Parcellator.build(spec, ref)
    names = pd.Series(parc.names, index=np.arange(1, parc.n + 1))
    ns = rm.neurosynth_behavioural_table(data, ref, "brainmap", labels=parc.label_volume(),
                                         key=a.atlas, statistic="zreverse").set_index("parcel")
    ns = ns.rename(columns={c: c[:-len(".Other")] for c in ns.columns if c.endswith(".Other")})
    bm = rm.brainmap_db_table(data, a.atlas, group=a.group, reverse=True).set_index("parcel")

    rows = []
    for c in [c for c in bm.columns if c in ns.columns]:
        x, y = bm[c].astype(float), ns.loc[bm.index, c].astype(float)
        r = spearmanr(x, y)[0] if x.std() > 0 and y.std() > 0 else np.nan
        dom, *rest = c.split(".")
        rows.append({"category": c, "domain": dom,
                     "subdomain": GENERAL.get(c, ": ".join(rest)),
                     "spearman_r": r, "brainmap_top": names[bm[c].idxmax()],
                     "neurosynth_top": names[ns[c].idxmax()]})
    t = pd.DataFrame(rows)
    t["order"] = t["domain"].map(DOMAINS.index)
    t = t.sort_values(["order", "category"]).drop(columns="order")
    out = Path(data) / "brainmap"
    t.to_csv(out / f"{a.atlas}_table_crossdb.csv", index=False)

    body, last = [], None
    for _, r in t.iterrows():
        if last is not None and r.domain != last:
            body.append(r"\addlinespace")
        body.append(f"{r.domain if r.domain != last else ''} & {r.subdomain} & "
                    f"{r.spearman_r:.2f} & {short(r.brainmap_top)} & {short(r.neurosynth_top)} \\\\")
        last = r.domain
    med = t["spearman_r"].median()
    tex = rf"""% Generated by tools/brainmap/table_crossdb.py ({a.atlas}, BrainMap {a.group}).
% Rerun the script and replace the table rather than editing numbers by hand.
\begin{{table}}[h]
\centering\footnotesize
\caption{{Agreement between the reverse-inference behavioural profiles from Neurosynth and
from the BrainMap database, in the {parc.n} parcels of the states' atlas. Both use the
reverse-inference test of the BrainMap decoder in NiMARE~\cite{{Salo_2023}} (signed two-way
$\chi^2$): on studies for Neurosynth, on foci for the BrainMap database, whose Mango
Behavioral Analysis plugin~\cite{{Lancaster_2012}} (version~4.0, experiments in healthy
participants) provides foci counts only. $r$: Spearman correlation across parcels between
the two profiles of a category (median {med:.2f}). Top parcel: the parcel with the highest
$z$ in each database. L, left; R, right; IFG, inferior frontal gyrus; ITG, inferior
temporal gyrus; LOC, lateral occipital cortex; MFG, middle frontal gyrus; MTG, middle
temporal gyrus; SFG, superior frontal gyrus; SMA, supplementary motor area
(juxtapositional lobule); SMG, supramarginal gyrus; SPL, superior parietal lobule; STG,
superior temporal gyrus.}}
\label{{tab:crossdb}}
\begin{{tabular}}{{llrll}}
\toprule
 & & & \multicolumn{{2}}{{c}}{{Top parcel}} \\
\cmidrule(l){{4-5}}
Domain & Subdomain & $r$ & BrainMap & Neurosynth \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (out / f"{a.atlas}_table_crossdb.tex").write_text(tex)
    print(t[["category", "spearman_r", "brainmap_top", "neurosynth_top"]]
          .to_string(index=False, float_format="%.2f"))
    print(f"\nmedian r = {med:.2f}\nwrote {out / f'{a.atlas}_table_crossdb.tex'}")


if __name__ == "__main__":
    main()
