#!/usr/bin/env python3
"""Reference values of the brain-state dynamics: one CSV, for other studies to place their data.

    python3 tools/reference_values.py            # the final model: yeo7, window -1, ThresholdCluster_umap3_8

Writes <outputs>/modelling/version=<v>/unsupervised_model/
       reference_values_atlas-<a>_window-<w>_states-<s>.csv

One row per feature x cohort x condition (rest | movie): n, mean, s.d. and the 5th, 25th, 50th,
75th and 95th percentiles over PARTICIPANTS. Each participant is one value per condition, built
exactly as in notebook 05 (Figure 3): transition counts summed over that person's scans and runs,
p(j | i) from the pooled counts, entropy rate Miller--Madow corrected, distinct transitions
rarefied to the shortest scan. Features: the dynamic metrics, the occupancy of every state and
every transition probability p(j | i). "All cohorts" pools every participant of every cohort
(each once per condition) beside the cohort rows. Read-only on everything else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from fmri_decomposition.io import unsupervised_root
from fmri_decomposition.unsupervised_model import cohort_tables as ct
from fmri_decomposition.figures import state_names as sn

COHORTS = {"ds002837": dict(dirs=["ds002837"], role="train"),
           "cneuromod": dict(dirs=["cneuromod"], role="train"),
           "camcan": dict(dirs=["camcan", "camcan_rest"], role="projected"),
           "hcp7t_fix": dict(dirs=["hcp7t_fix"], role="projected")}
COHORT_NAME = {"all": "All cohorts", "ds002837": "NNDb", "cneuromod": "CNeuroMod", "camcan": "Cam-CAN", "hcp7t_fix": "HCP 7T"}
PCTS = (5, 25, 50, 75, 95)


def feature_name(v: str) -> str:
    if v.startswith("occ_") and v[4:].isdigit():
        return f"occupancy · {sn.label(v[4:], 'long')}"
    if "->" in v:
        a, b = v.split("->")
        return f"p({sn.label(b, 'long')} | {sn.label(a, 'long')})"
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", default="yeo7")
    ap.add_argument("--window-s", type=float, default=-1)
    ap.add_argument("--states", default="ThresholdCluster_umap3_8")
    ap.add_argument("--out", type=Path, default=None, help="default: <outputs>/.../unsupervised_model/")
    a = ap.parse_args()

    cfg = ct.Settings(space=dict(atlas=a.atlas, window_s=a.window_s, states=a.states),
                      entropy_estimator="miller_madow", distinct_minutes="min",
                      cohorts=COHORTS, rest_pattern=r"rest\d*")
    sn.use_for(cfg.root, a.atlas, a.window_s, a.states, cfg.modelling_version, mode="name")
    co = ct.Cohorts(cfg)
    feats = [*co.dyn, *co.occ, *co.cells]
    rows = []
    groups = dict(co.data)
    for cond in ("rest", "movie"):              # every cohort together: each participant once
        parts = [d for (c, k), d in co.data.items() if k == cond]
        if parts:
            groups[("all", cond)] = pd.concat(parts)
    for (cohort, cond), d in groups.items():
        for v in feats:
            if v not in d:
                continue
            x = pd.to_numeric(d[v], errors="coerce").dropna().to_numpy(float)
            if not len(x):
                continue
            q = np.percentile(x, PCTS)
            rows.append(dict(feature=feature_name(v), variable=v,
                             family=("dynamics" if v in co.dyn else "occupancy" if v in co.occ
                                     else "transition p(j|i)"),
                             cohort=COHORT_NAME.get(cohort, cohort),
                             role=COHORTS[cohort]["role"] if cohort in COHORTS else "all",
                             condition=cond, n=len(x), mean=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                             **{f"p{p}": v_ for p, v_ in zip(PCTS, q)}))
    t = pd.DataFrame(rows)
    order = {"dynamics": 0, "occupancy": 1, "transition p(j|i)": 2}
    t = t.sort_values(["family", "variable", "cohort", "condition"],
                      key=lambda s: s.map(order) if s.name == "family" else s).reset_index(drop=True)
    out = a.out or unsupervised_root(cfg.root, cfg.modelling_version)
    out.mkdir(parents=True, exist_ok=True)
    w = f"{a.window_s:g}"
    f = out / f"reference_values_atlas-{a.atlas}_window-{w}_states-{a.states}.csv"
    t.to_csv(f, index=False, float_format="%.4g")
    print(t[t["family"] != "transition p(j|i)"].head(30).to_string(index=False))
    print(f"\n{len(t)} rows ({t['feature'].nunique()} features x cohort x condition) -> {f}")
    print(f"distinct transitions rarefied to {co.distinct_minutes} min; entropy rate Miller-Madow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
