"""Gather the per-parcel plugin outputs into the tables the pipeline reads.

Writes, in <data>/brainmap/:
    <atlas>_mango_long.csv      every parcel x group x measure x plugin category
    <atlas>_mango_normals.csv   parcel x BRAINMAP_TAXONOMY, z, healthy participants
    <atlas>_mango_all.csv       the same on all experiments
    <atlas>_mango_vs_neurosynth-*.csv   per category, Spearman r across parcels
                                with the Neurosynth tables: the decoder's zForward
                                (if notebook 08 cached it) and the same Lancaster
                                statistic, computed here
and prints sanity checks: the top categories of landmark parcels, and the
left/right asymmetry of language in the inferior frontal gyrus, which a
mirrored ROI would flip.

    python tools/brainmap/collect.py --atlas harvardoxford --work WORK
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fmri_decomposition.io import require_reference_maps
from fmri_decomposition.unsupervised_model.contextualization import refmaps as rm

LANDMARKS = ["Heschl", "Occipital Pole", "Precentral", "Postcentral", "Amygdala",
             "Hippocampus", "Inferior Frontal Gyrus, pars opercularis"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas", default="harvardoxford")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--no-neurosynth", action="store_true",
                    help="skip computing the Lancaster-statistic Neurosynth table")
    a = ap.parse_args(argv)

    data = a.data or require_reference_maps()
    rois = pd.read_csv(a.work / "rois" / "rois.tsv", sep="\t")
    parts, missing = [], []
    for _, r in rois.iterrows():
        f = a.work / "out" / f"parcel_{r.parcel:03d}.csv"
        if not f.exists():
            missing.append(r["name"])
            continue
        d = pd.read_csv(f)
        d.insert(0, "parcel", r.parcel)
        parts.append(d)
    if missing:
        raise SystemExit(f"{len(missing)} parcels without output, e.g. {missing[:5]}: "
                         "rerun run_brainmap_profiles.sh (it skips finished parcels)")
    long = pd.concat(parts, ignore_index=True)
    long["category"] = long["category"].replace(rm.BRAINMAP_PLUGIN_RENAME)

    outdir = Path(data) / "brainmap"
    outdir.mkdir(parents=True, exist_ok=True)
    long.to_csv(outdir / f"{a.atlas}_mango_long.csv", index=False)

    tables = {}
    for group in ("normals", "all"):
        z = (long.query("group == @group and measure == 'z'")
             .pivot(index="parcel", columns="category", values="value"))
        cats = rm.brainmap_db_columns(z.columns)
        t = z[cats].reset_index()
        t.to_csv(outdir / f"{a.atlas}_mango_{group}.csv", index=False)
        tables[group] = t
        absent = [c for c in rm.BRAINMAP_TAXONOMY
                  if c not in z.columns and c not in rm.BRAINMAP_DB_GENERAL.values()]
        print(f"[{group}] {len(t)} parcels x {len(cats)} categories"
              + (f"; not in the plugin: {', '.join(absent)}" if absent else ""))

    # Foci per category over the whole atlas, so thin categories are visible.
    cnt = (long.query("group == 'normals' and measure == 'count'")
           .groupby("category")["value"].sum().sort_values())
    print("\nfoci inside the atlas, fewest first (normals):")
    print(cnt.head(15).to_string())

    # Sanity: landmark parcels' top categories (normals, z).
    t = tables["normals"].merge(rois[["parcel", "name"]], on="parcel")
    cats = [c for c in t.columns if "." in c]
    print("\nlandmarks, top 4 categories by z (normals):")
    for lm in LANDMARKS:
        for _, r in t[t["name"].str.contains(lm, regex=False)].iterrows():
            top = r[cats].astype(float).sort_values(ascending=False).head(4)
            print(f"  {r['name'][:45]:45s} " +
                  ", ".join(f"{k} {v:.1f}" for k, v in top.items()))
    lang = [c for c in cats if c.startswith("Cognition.Language")]
    ifg = t[t["name"].str.contains("pars opercularis", regex=False)]
    if len(ifg) == 2:
        print("\nIFG pars opercularis, mean language z:",
              ", ".join(f"{n}: {r[lang].astype(float).mean():.2f}"
                        for n, (_, r) in zip(ifg["name"], ifg.iterrows())),
              "(left should exceed right)")

    # The Neurosynth profiles: the decoder's zForward (cached by notebook 08)
    # and the same Lancaster statistic as the BrainMap ones (computed here),
    # so database and statistic can be told apart.
    size = rois.set_index("parcel")["n_vox_mni"].astype(float)
    pname = rois.set_index("parcel")["name"]
    bm = tables["normals"].set_index("parcel")
    tabs = {"BrainMap": bm}
    ns_dir = Path(data) / "neurosynth"
    if (ns_dir / f"{a.atlas}_nimare_brainmap.csv").exists():
        tabs["Neurosynth-zForward"] = pd.read_csv(
            ns_dir / f"{a.atlas}_nimare_brainmap.csv").set_index("parcel")
    if (ns_dir / f"{a.atlas}_nimare_brainmap_pure.csv").exists():
        tabs["Neurosynth-zForward-pure"] = pd.read_csv(
            ns_dir / f"{a.atlas}_nimare_brainmap_pure.csv").set_index("parcel")
    if not a.no_neurosynth and (ns_dir / "data-neurosynth_version-7_coordinates.tsv.gz").exists():
        ref = rm.reference_grid(data)
        spec, _ = rm.state_atlas(a.atlas, data)
        parc = rm.Parcellator.build(spec, ref)
        for tax, stat, k in (("brainmap", "lancaster", "Neurosynth-Lancaster"),
                             ("brainmap", "lancaster_studies", "Neurosynth-Lancaster-studies"),
                             ("brainmap", "lancaster_weighted", "Neurosynth-Lancaster-weighted"),
                             ("brainmap_pure", "lancaster", "Neurosynth-Lancaster-pure")):
            tabs[k] = rm.neurosynth_behavioural_table(
                data, ref, tax, labels=parc.label_volume(), key=a.atlas,
                statistic=stat).set_index("parcel")

    # Parcel-size check: a profile that tracks parcel volume measures size,
    # not function. Spearman across parcels, per category, then the median.
    print("\nparcel size vs profile (Spearman across parcels, per category):")
    for k, t in tabs.items():
        cs = [c for c in t.columns if "." in c]
        rs = pd.Series({c: spearmanr(t[c].astype(float), size.loc[t.index])[0] for c in cs
                        if t[c].astype(float).std() > 0})
        print(f"  {k:30s} median r = {rs.median():+.2f}, "
              f"{(rs < -.3).sum()} categories r < -.3, {(rs > .3).sum()} r > .3")
        rs.rename("r_size").to_csv(outdir / f"{a.atlas}_size_check_{k.lower()}.csv")

    for k in [k for k in tabs if k != "BrainMap"]:
        ns = tabs[k]
        pairs = [(c, c) for c in bm.columns if c in ns.columns] + \
                [(g, o) for g, o in rm.BRAINMAP_DB_GENERAL.items()
                 if g in bm.columns and o in ns.columns]
        rows = []
        for c, c_ns in pairs:
            x, y = bm[c].astype(float), ns.loc[bm.index, c_ns].astype(float)
            r, p = spearmanr(x, y) if x.std() > 0 and y.std() > 0 else (np.nan, np.nan)
            rows.append({"category": c, "spearman_r": r, "p": p,
                         "brainmap_top": pname[bm[c].idxmax()],
                         "neurosynth_top": pname[ns[c_ns].idxmax()]})
        cmp_ = pd.DataFrame(rows).sort_values("spearman_r")
        cmp_.to_csv(outdir / f"{a.atlas}_mango_vs_{k.lower()}.csv", index=False)
        print(f"\nBrainMap vs {k}, Spearman across {len(bm)} parcels "
              f"(median r = {cmp_.spearman_r.median():.2f}):")
        print(cmp_[["category", "spearman_r", "brainmap_top", "neurosynth_top"]]
              .to_string(index=False, float_format="%.2f"))
    if len(tabs) == 1:
        print("\n(no Neurosynth data or cache: nothing to compare with)")
    print(f"\ntables in {outdir}")


if __name__ == "__main__":
    main()
