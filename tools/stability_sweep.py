"""Cross-cohort stability for every state set: one row per set, to help choose it.

For each state set with every cohort (atlas x window fixed), the two questions of the manuscript's
cross-cohort section, answered with notebook 14's methods (cohort_tables.Cohorts):

  replication   does the transition STRUCTURE replicate across cohorts?  r between cohorts'
                mean p(j | i) on movie, divided by each cohort's split-half noise ceiling
                (`replication`): ~1 = cohorts agree as well as a cohort agrees with itself.
                Also r between rest and movie in the same people (Cam-CAN, HCP).
  context       which properties depend on context?  |d_av| movie vs rest in the same people for
                the transition fit and every dynamic metric, and their tier (< .2 stable,
                < .5 moderate, else context-dependent; `stability`).
  reliability   ICC(3,1) rest <-> movie per cohort, median over the dynamic metrics.
  states        rare states (mean occupancy < 1%): a set that wastes states on transits.

No bootstrap (n_boot=0): this ranks sets; the CIs come from notebook 14 on the chosen one.

    python tools/stability_sweep.py --atlas harvardoxford --window-s 30 --out stability_sweep.csv
    python tools/stability_sweep.py --states 'HMM2_.*_8'          # a regex over set names
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from fmri_decomposition.unsupervised_model import cohort_tables as ct

COHORTS = {"ds002837": dict(dirs=["ds002837"], role="train"),
           "cneuromod": dict(dirs=["cneuromod"], role="train"),
           "camcan": dict(dirs=["camcan", "camcan_rest"], role="projected"),
           "hcp7t_fix": dict(dirs=["hcp7t_fix"], role="projected")}


def summarize(co: ct.Cohorts) -> dict:
    out = {"K": co.cfg.K}
    occ = pd.concat([v[co.occ].mean() for v in co.data.values()], axis=1).mean(axis=1)
    out["n_rare_states"] = int((occ < .01).sum())
    out["min_occupancy"] = float(occ.min())

    pairs, ceil = co.replication("movie")
    out["repl_r_median"] = pairs["r"].median()
    out["repl_rnorm_median"] = pairs["r_norm"].median()
    out["repl_rnorm_min"] = pairs["r_norm"].min()
    out["repl_worst_pair"] = f"{pairs.loc[pairs['r_norm'].idxmin(), 'a']}-{pairs.loc[pairs['r_norm'].idxmin(), 'b']}"
    for c, v in ceil.items():
        out[f"split_half_{c}"] = v

    S = co.similarity()
    for c in co.cfg.cohorts:
        a, b = f"{c} rest", f"{c} movie"
        if a in S.index and b in S.index:
            out[f"rest_movie_r_{c}"] = S.loc[a, b]

    vs = ["transition_fit", *co.dyn]
    st = co.stability(vs, n_boot=0).set_index("variable")
    for v in vs:
        out[f"d_{v}"] = abs(st.loc[v, "task_d"])
    out["n_stable"] = int((st["tier"] == "stable").sum())
    out["n_context"] = int((st["tier"] == "context-dependent").sum())
    out["icc_median"] = st["icc"].median()
    for col in [c for c in st.columns if c.startswith("icc_") and c not in ("icc_lo", "icc_hi")]:
        out[f"{col}_median"] = st[col].median()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas", default="harvardoxford")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--states", default=None, help="regex over state-set names (default: all)")
    ap.add_argument("--entropy-estimator", default="miller_madow")
    ap.add_argument("--out", type=Path, default=Path("stability_sweep.csv"))
    a = ap.parse_args(argv)

    base = ct.Settings(cohorts=COHORTS)
    sets = ct.available_sets(base.root, COHORTS, base.modelling_version)
    sets = sets[(sets["atlas"] == a.atlas) & np.isclose(sets["window_s"].astype(float), a.window_s)]
    if a.states:
        sets = sets[sets["states"].str.fullmatch(a.states)]
    print(f"{len(sets)} state sets ({a.atlas}, {a.window_s:g} s)", flush=True)

    rows = []
    for i, s in enumerate(sets["states"], 1):
        cfg = ct.Settings(space=dict(atlas=a.atlas, window_s=a.window_s, states=s), cohorts=COHORTS,
                          entropy_estimator=a.entropy_estimator)
        try:
            rows.append({"states": s, **summarize(ct.Cohorts(cfg))})
            print(f"[{i}/{len(sets)}] {s}", flush=True)
        except (Exception, SystemExit) as e:                 # one bad set must not stop the sweep
            print(f"[{i}/{len(sets)}] {s} FAILED: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(limit=1)
    if not rows:
        raise SystemExit("no state set could be summarised")
    t = pd.DataFrame(rows).sort_values(["repl_rnorm_median", "n_rare_states"], ascending=[False, True])
    t.to_csv(a.out, index=False)
    show = ["states", "K", "n_rare_states", "repl_rnorm_median", "repl_rnorm_min", "repl_worst_pair",
            *[c for c in t.columns if c.startswith("rest_movie_r_")], "n_stable", "n_context",
            *[c for c in t.columns if c.startswith("icc_") and c.endswith("_median")]]
    pd.set_option("display.width", 250)
    print(t[show].to_string(index=False, float_format="%.2f"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
