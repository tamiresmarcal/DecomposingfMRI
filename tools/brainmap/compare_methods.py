"""Which behavioural-profile method describes the Yeo 7 networks best?

For every method available on disk -- the BrainMap database (Mango plugin)
and the Neurosynth variants (NiMARE zForward; Lancaster on foci, on studies,
on foci weighted 1/n per study, reverse inference; with and without synonyms) -- this correlates
each BrainMap category with each Yeo network across the parcels of the states'
atlas (Pearson, as the matrix in notebook 08) and scores the matrix:

  mean|r|, max|r|, n(|r|>=.4)   how strong the associations are at all
  r_expected, r_other           mean r in the cells the literature expects to be
                                positive (EXPECTED below) and in all the others
  specificity                   r_expected - r_other: strong where it should be,
                                weak elsewhere -- the score to rank by
  hits                          networks (of 7) whose top category is expected

Strength alone rewards a method that correlates with everything; specificity
rewards the one that tells the networks apart in the expected way. EXPECTED
is a judgement from the literature (e.g. Yeo et al. 2011's own BrainMap
decoding): read it, and edit it, before trusting the ranking.

Each scope is run twice: all parcels, and cortical parcels only. Yeo 2011 is a
cortical parcellation: in subcortical parcels every network is 0, so an
amygdala full of emotion foci counts AGAINST the limbic network.

    python tools/brainmap/compare_methods.py --atlas harvardoxford
Writes <data>/brainmap/<atlas>_method_comparison.csv and one matrix per method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fmri_decomposition.io import require_reference_maps
from fmri_decomposition.unsupervised_model.contextualization import refmaps as rm

# Categories each Yeo network is expected to carry (positive r). Categories are
# the general names: "... .Other" bins are shown as their general category.
EXPECTED = {
    "Visual": ["Perception.Vision", "Perception.Vision.Color", "Perception.Vision.Motion",
               "Perception.Vision.Shape", "Cognition.Language.Orthography"],
    "Somatomotor": ["Action.Execution", "Action.Execution.Speech", "Action.Imagination",
                    "Action.Motor Learning", "Perception.Somesthesis", "Perception.Audition",
                    "Cognition.Music", "Cognition.Language.Speech"],
    "Dorsal Attention": ["Cognition.Attention", "Cognition.Space", "Action.Observation",
                         "Perception.Vision.Motion", "Action.Preparation"],
    "Ventral Attention": ["Perception.Somesthesis.Pain", "Interoception.Baroregulation",
                          "Interoception.Hunger", "Interoception.Air Hunger",
                          "Interoception.Bladder", "Interoception.Thermoregulation",
                          "Interoception.Thirst", "Action.Inhibition", "Cognition.Attention",
                          "Perception.Gustation", "Emotion.Disgust"],
    "Limbic": ["Emotion.Anger", "Emotion.Anxiety", "Emotion.Disgust", "Emotion.Fear",
               "Emotion.Happiness", "Emotion.Sadness", "Perception.Olfaction",
               "Perception.Gustation", "Interoception.Hunger", "Interoception.Sexuality"],
    "Frontoparietal": ["Cognition.Memory.Working", "Cognition.Reasoning", "Action.Inhibition",
                       "Cognition.Attention"],
    "Default": ["Cognition.Social Cognition", "Cognition.Language.Semantics",
                "Cognition.Memory.Explicit", "Action.Rest"],
}


def _general(t: pd.DataFrame) -> pd.DataFrame:
    return t.rename(columns={c: c[:-len(".Other")] for c in t.columns if c.endswith(".Other")})


def load_methods(data: Path, atlas: str, parc, ref) -> dict:
    """{method name: parcels x categories}, for every method on disk or cheap
    to compute (the Lancaster variants take seconds; zForward only if cached)."""
    out = {}
    bm = Path(data) / "brainmap"
    for g in ("normals", "all"):
        f = bm / f"{atlas}_mango_{g}.csv"
        if f.exists():
            out[f"BrainMap ({g})"] = pd.read_csv(f).set_index("parcel")
        if (bm / f"{atlas}_mango_long.csv").exists():
            out[f"BrainMap reverse ({g})"] = rm.brainmap_db_table(
                data, atlas, group=g, reverse=True).set_index("parcel")
    ns = Path(data) / "neurosynth"
    for tax, suffix in (("brainmap", ""), ("brainmap_pure", ", no synonyms")):
        f = ns / f"{atlas}_nimare_{tax}.csv"
        if f.exists():
            out[f"Neurosynth zForward{suffix}"] = pd.read_csv(f).set_index("parcel")
        if (ns / "data-neurosynth_version-7_coordinates.tsv.gz").exists():
            # NiMARE's BrainMap decoder, reverse inference: no added weighting
            # or statistic (computed on first use, cached with zForward).
            out[f"Neurosynth NiMARE zReverse{suffix}"] = rm.neurosynth_behavioural_table(
                data, ref, tax, labels=parc.label_volume(), key=atlas,
                statistic="zreverse").set_index("parcel")
        if not (ns / "data-neurosynth_version-7_coordinates.tsv.gz").exists():
            continue
        for stat, name in (("lancaster", "Lancaster foci"), ("lancaster_studies", "Lancaster studies"),
                           ("lancaster_weighted", "Lancaster weighted"),
                           ("lancaster_reverse", "Lancaster reverse")):
            out[f"Neurosynth {name}{suffix}"] = rm.neurosynth_behavioural_table(
                data, ref, tax, labels=parc.label_volume(), key=atlas,
                statistic=stat).set_index("parcel")
    return {k: _general(v[[c for c in v.columns if "." in c]]) for k, v in out.items()}


def score(r: pd.DataFrame) -> dict:
    exp = np.zeros(r.shape, bool)
    for i, net in enumerate(r.index):
        exp[i] = [c in EXPECTED.get(net, []) for c in r.columns]
    v = r.to_numpy(float)
    ok = np.isfinite(v)
    top = r.idxmax(axis=1)
    return {
        "n_categories": r.shape[1],
        "mean_abs_r": np.nanmean(np.abs(v)),
        "max_abs_r": np.nanmax(np.abs(v)),
        "n_abs_r_ge_.4": int((np.abs(v[ok]) >= .4).sum()),
        "r_expected": np.nanmean(v[exp & ok]) if (exp & ok).any() else np.nan,
        "r_other": np.nanmean(v[~exp & ok]),
        "specificity": (np.nanmean(v[exp & ok]) - np.nanmean(v[~exp & ok]))
        if (exp & ok).any() else np.nan,
        "hits": int(sum(t in EXPECTED.get(n, []) for n, t in top.items())),
        "limbic_best": f"{top.get('Limbic', '')} ({r.loc['Limbic'].max():+.2f})"
        if "Limbic" in r.index else "",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas", default="harvardoxford")
    ap.add_argument("--data", type=Path, default=None)
    a = ap.parse_args(argv)

    data = a.data or require_reference_maps()
    ref = rm.reference_grid(data)
    spec, _ = rm.state_atlas(a.atlas, data)
    parc = rm.Parcellator.build(spec, ref)
    yeo = pd.DataFrame(parc.parcellate(rm.yeo7_maps(data, ref).voxels),
                       index=rm.YEO7, columns=np.arange(1, parc.n + 1))
    kind = (spec.labels["network"].to_numpy() if "network" in spec.labels
            else np.array(["cortical"] * parc.n))
    cortical = np.arange(1, parc.n + 1)[kind == "cortical"]
    methods = load_methods(data, a.atlas, parc, ref)
    if not methods:
        raise SystemExit("no profile tables found: run run_brainmap_profiles.sh / notebook 08")

    outdir = Path(data) / "brainmap" / f"{a.atlas}_method_comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scope, parcels in (("all parcels", yeo.columns.to_numpy()), ("cortical only", cortical)):
        for name, t in methods.items():
            t = t.reindex(parcels)
            r = pd.DataFrame(rm.correlate(yeo[parcels].to_numpy(float),
                                          t.to_numpy(float).T),
                             index=rm.YEO7, columns=t.columns)
            r.to_csv(outdir / f"{name.replace(' ', '_').replace(',', '')}_{scope.split()[0]}.csv")
            rows.append({"scope": scope, "method": name, **score(r)})
    res = pd.DataFrame(rows).sort_values(["scope", "specificity"], ascending=[True, False])
    res.to_csv(Path(data) / "brainmap" / f"{a.atlas}_method_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    for scope, g in res.groupby("scope", sort=False):
        print(f"\n=== {scope} ({len(cortical) if scope.startswith('cortical') else parc.n} parcels), "
              "ranked by specificity ===")
        print(g.drop(columns="scope").to_string(index=False, float_format="%.2f"))
    print(f"\nmatrices in {outdir}")


if __name__ == "__main__":
    main()
