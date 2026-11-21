"""Fit notebook 14's mixed models once, for every state set, and write them where the notebook
reads them (cohort_tables.mixed_dir):

    <outputs>/modelling/version=v0/unsupervised_model/mixed_model/atlas=/window_s=/states=/
        vpc_properties.csv    transition structure, occupancy profile, dynamic metrics, occupancies
        vpc_transitions.csv   transition structure and every transition p(j | i)
        meta.json             the settings they were made with (a mismatch makes the notebook refit)

Same settings as notebook 14 (cohorts, Miller-Madow entropy, rarefied distinct transitions, rest
pattern). A set already written with the same settings is skipped unless --overwrite.

    python tools/mixed_model_sweep.py --atlas harvardoxford --window-s 30 --states 'HMM2_umap3_8'
    python tools/mixed_model_sweep.py --atlas all --window-s all --jobs 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from fmri_decomposition.unsupervised_model import cohort_tables as ct

COHORTS = {"ds002837": dict(dirs=["ds002837"], role="train"),
           "cneuromod": dict(dirs=["cneuromod"], role="train"),
           "camcan": dict(dirs=["camcan", "camcan_rest"], role="projected"),
           "hcp7t_fix": dict(dirs=["hcp7t_fix"], role="projected")}


def settings(atlas, w, s):
    return ct.Settings(space=dict(atlas=atlas, window_s=float(w), states=s), cohorts=COHORTS,
                       entropy_estimator="miller_madow", distinct_minutes="min", rest_pattern=r"rest\d*")


def one(atlas, w, s, overwrite, cohort_term=True, only_cohort=None):
    t0 = time.time()
    cfg = settings(atlas, w, s)
    d = ct.mixed_dir(cfg, cohort_term, only_cohort)
    co = ct.Cohorts(cfg)
    if not overwrite and (d / "meta.json").exists() and json.loads((d / "meta.json").read_text()) == ct._mixed_meta(co):
        return f"{atlas} {w:g} {s}: already there, skipped"
    ct.mixed_tables(co, recompute=True, cohort_term=cohort_term, only_cohort=only_cohort)
    return f"{atlas} {w:g} {s}: written to {d} ({time.time() - t0:.0f} s)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas", default="harvardoxford", help='an atlas, or "all"')
    ap.add_argument("--window-s", default="30", help='a window in s, or "all"')
    ap.add_argument("--states", default=None, help="regex over state-set names (default: all)")
    ap.add_argument("--jobs", type=int, default=1, help="state sets fitted in parallel")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-cohort-term", dest="cohort_term", action="store_false",
                    help="fit value ~ 1 + (1|task) + (1|person): no cohort effect, the cohorts' differences "
                         "carried by the task component (written to mixed_model_nocohort/)")
    ap.add_argument("--cohort", default=None, nargs="+",
                    help="fit on these cohorts one at a time, each alone (mixed_model_cohort-<c>/), e.g. camcan hcp7t_fix")
    a = ap.parse_args(argv)

    base = ct.Settings(cohorts=COHORTS)
    sets = ct.available_sets(base.root, COHORTS, base.modelling_version)
    if a.atlas != "all":
        sets = sets[sets["atlas"] == a.atlas]
    if a.window_s != "all":
        sets = sets[np.isclose(sets["window_s"].astype(float), float(a.window_s))]
    if a.states:
        sets = sets[sets["states"].str.fullmatch(a.states)]
    todo = [(r.atlas, float(r.window_s), r.states) for r in sets.itertuples()]
    print(f"{len(todo)} state sets, {a.jobs} at a time", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        jobs = [(*t, c) for t in todo for c in (a.cohort or [None])]
        futs = {ex.submit(one, *j[:3], a.overwrite, a.cohort_term, j[3]): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(todo)}] {f.result()}", flush=True)
            except Exception:                                  # one bad set must not stop the rest
                failed += 1
                print(f"[{i}/{len(todo)}] {' '.join(map(str, futs[f]))} FAILED", file=sys.stderr, flush=True)
                traceback.print_exc()
    if failed:
        raise SystemExit(f"{failed} state set(s) failed")


if __name__ == "__main__":
    main()
