"""Supplementary Table: agreement of the reverse-inference profiles of
Neurosynth and of the BrainMap database.

Like for like: both databases with the same test (NiMARE's signed two-way
chi-square, `chi2_reverse_z`) on the same unit, foci counted strictly inside
parcels -- the BrainMap plugin gives foci counts only, so Neurosynth is
counted the same way here (the main analysis keeps NiMARE's zReverse on
studies). Per category: the Spearman correlation across the atlas's parcels;
whether the two top parcels coincide (hemisphere ignored) and whether each
database's top parcel is among the other's five highest; and the number of
foci each database has for the category, so thin categories show.

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


def region(name: str) -> str:
    """The parcel's region without its hemisphere."""
    return re.sub(r"^(Left|Right) ", "", name)


def short(name: str) -> str:
    n = region(name)
    return ABBREV.get(n, n.lower())


def top_regions(col: pd.Series, names: pd.Series, k: int) -> list:
    return [region(names[p]) for p in col.astype(float).nlargest(k).index]


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
                                         key=a.atlas, statistic="chi2_foci").set_index("parcel")
    general = {c: c[:-len(".Other")] for c in ns.columns if c.endswith(".Other")}
    ns = ns.rename(columns=general)
    ns_n = pd.read_csv(Path(data) / "neurosynth" / f"{a.atlas}_chi2foci_brainmap_n.csv",
                       index_col=0).rename(index=general)["n_foci"]
    bm = rm.brainmap_db_table(data, a.atlas, group=a.group, reverse=True).set_index("parcel")
    long = pd.read_csv(Path(data) / "brainmap" / f"{a.atlas}_mango_long.csv")
    bm_n = (long.query("group == @a.group and measure == 'count'")
            .groupby("category")["value"].sum())

    rows = []
    for c in [c for c in bm.columns if c in ns.columns]:
        x, y = bm[c].astype(float), ns.loc[bm.index, c].astype(float)
        r = spearmanr(x, y)[0] if x.std() > 0 and y.std() > 0 else np.nan
        dom, *rest = c.split(".")
        b5, n5 = top_regions(bm[c], names, 5), top_regions(ns[c], names, 5)
        rows.append({"category": c, "domain": dom,
                     "subdomain": GENERAL.get(c, ": ".join(rest)),
                     "spearman_r": r,
                     "brainmap_top": region(names[bm[c].astype(float).idxmax()]),
                     "neurosynth_top": region(names[ns[c].astype(float).idxmax()]),
                     "same_top": b5[0] == n5[0],
                     "top_in_other_top5": b5[0] in n5 or n5[0] in b5,
                     "brainmap_foci": int(bm_n.get(c, 0)),
                     "neurosynth_foci": int(ns_n.get(c, 0))})
    t = pd.DataFrame(rows)
    t["order"] = t["domain"].map(DOMAINS.index)
    t = t.sort_values(["order", "category"]).drop(columns="order")
    out = Path(data) / "brainmap"
    t.to_csv(out / f"{a.atlas}_table_crossdb.csv", index=False)

    body, last = [], None
    for _, r in t.iterrows():
        if last is not None and r.domain != last:
            body.append(r"\addlinespace")
        mark = "=" if r.same_top else ("$\\approx$" if r.top_in_other_top5 else "")
        body.append(f"{r.domain if r.domain != last else ''} & {r.subdomain} & "
                    f"{r.spearman_r:.2f} & {short(r.brainmap_top)} & {short(r.neurosynth_top)} & "
                    f"{mark} & {r.brainmap_foci:,} & {r.neurosynth_foci:,} \\\\".replace(",", "{,}"))
        last = r.domain
    med = t["spearman_r"].median()
    n_same, n_top5, n_cat = int(t.same_top.sum()), int(t.top_in_other_top5.sum()), len(t)
    tex = rf"""% Generated by tools/brainmap/table_crossdb.py ({a.atlas}, BrainMap {a.group}).
% Rerun the script and replace the table rather than editing numbers by hand.
\begin{{table}}[h]
\centering\footnotesize
\caption{{Agreement between the reverse-inference behavioural profiles from Neurosynth and
from the BrainMap database, in the {parc.n} parcels of the states' atlas. To compare the
databases alone, both were computed with the same test, the reverse-inference test of the
BrainMap decoder in NiMARE~\cite{{Salo_2023}} (signed two-way $\chi^2$), on the same unit,
activation foci inside each parcel, because the Mango Behavioral Analysis
plugin~\cite{{Lancaster_2012}} (version~4.0, experiments in healthy participants) provides
foci counts only. $r$: Spearman correlation across parcels between the two profiles of a
category (median {med:.2f}). Top region: the region of the parcel with the highest $z$ in
each database, hemisphere ignored; =, the same region in both ({n_same} of {n_cat}
categories); $\approx$, each database's top region among the other's five highest
({n_top5 - n_same} more). Foci: the category's foci inside the atlas in each database.
IFG, inferior frontal gyrus; ITG, inferior
temporal gyrus; LOC, lateral occipital cortex; MFG, middle frontal gyrus; MTG, middle
temporal gyrus; SFG, superior frontal gyrus; SMA, supplementary motor area
(juxtapositional lobule); SMG, supramarginal gyrus; SPL, superior parietal lobule; STG,
superior temporal gyrus.}}
\label{{tab:crossdb}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrlllrr}}
\toprule
 & & & \multicolumn{{3}}{{c}}{{Top region}} & \multicolumn{{2}}{{c}}{{Foci}} \\
\cmidrule(lr){{4-6}}\cmidrule(l){{7-8}}
Domain & Subdomain & $r$ & BrainMap & Neurosynth & & BrainMap & Neurosynth \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    (out / f"{a.atlas}_table_crossdb.tex").write_text(tex)
    print(t[["category", "spearman_r", "same_top", "top_in_other_top5", "brainmap_foci",
             "neurosynth_foci", "brainmap_top", "neurosynth_top"]]
          .to_string(index=False, float_format="%.2f"))
    print(f"\nmedian r = {med:.2f}; same top region {n_same}/{n_cat}; "
          f"top region in the other's top 5 {n_top5}/{n_cat}\nwrote {out / f'{a.atlas}_table_crossdb.tex'}")


if __name__ == "__main__":
    main()
