"""Compute, once, the slow parts of notebooks 06 and 07 and leave them in their caches
(fmri_decomposition.supervised_model.report): the notebooks then read instead of refitting.

  Cam-CAN    baselines, the C shares (Study.shares), the refits with out-of-fold SHAP (Study.deep)
             and the fits on every subject (Study.full) of the chosen model on the chosen state set
  transfer   SHAP on the HCP subjects and the predictions of the chosen configuration, and the
             SHAP shares of the best transferred configurations (notebook 06's analogue of C)

The choices are the notebooks' defaults (best state set and model; best transferred row) unless
given here; give the same ones in the notebook, or its figure is computed afresh.

    python tools/predictive_cache.py
    python tools/predictive_cache.py --space harvardoxford,30,HMM2_umap3_8 --model hgb --arm "all [cond]"
    # every combination of several choices, one job:
    python tools/predictive_cache.py --space harvardoxford,30,HMM2_umap3_8 harvardoxford,-1,HMM2_umap3_8 \
        yeo7,30,HMM2_umap3_8 yeo7,-1,HMM2_umap3_8 --model hgb ridge --arm "all [cond]" "all [joint]"
"""

from __future__ import annotations

import argparse
import json

from fmri_decomposition.supervised_model import hads as ht
from fmri_decomposition.supervised_model import report as rp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--space", nargs="+", help="atlas,window_s,states, one or more (default: the notebook's best)")
    ap.add_argument("--model", nargs="+", help="with --space: one or more models (default: the best of graph_arms)")
    ap.add_argument("--arm", nargs="+", default=["all [cond]"])
    ap.add_argument("--fc-arm", default="edges")
    ap.add_argument("--target", default="category")
    ap.add_argument("--pick", default="{}", help='JSON, e.g. \'{"model": "hgb"}\' (notebook PICK)')
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], choices=["cam", "shares", "transfer", "contrib"])
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args(argv)

    over = {"n_jobs": a.n_jobs} if a.n_jobs else {}
    spaces = a.space or [None]
    models = a.model or [None]
    combos = [(sp, m, arm) for sp in spaces for m in models for arm in a.arm]
    print(f"{len(combos)} combination(s): {combos}", flush=True)
    st = None
    if "cam" not in a.skip or "shares" not in a.skip:
        st = ht.Study(rp.cam_settings(**over))
        if "shares" not in a.skip:
            st.shares()
        if "cam" not in a.skip:
            for sp, m, arm in combos:
                space = model = None
                if sp:
                    atlas, w, states = sp.split(",")
                    space = dict(atlas=atlas, window_s=float(w), states=states)
                    model = dict(model=m, arm=arm) if m else None
                try:
                    _, _, space, model = rp.cam_choices(st, space, model)
                    print("Cam-CAN:", ht.set_label(space), "|", model, flush=True)
                    st.deep(space, model)          # Figure 4 left, 5a: out-of-fold SHAP (cache "deep")
                    st.full(space, model)          # Figure 5b: fitted on all of Cam-CAN (cache "full")
                except Exception as e:             # one missing combination must not stop the rest
                    print(f"  FAILED {sp} {m} {arm}: {e}", flush=True)
    if "transfer" not in a.skip or "contrib" not in a.skip:
        train, test, train_cfg, _, root = rp.transfer_setup(a.target)
        _, rows = rp.transferred_rows(root, a.target)
        if "transfer" not in a.skip:
            for sp, m, arm in combos:
                pick = json.loads(a.pick)
                if sp and not pick:                    # the notebook's PICK: its parameters
                    atlas, w, states = sp.split(",")
                    pick = dict(atlas=atlas, window_s=float(w), states=states, **({"model": m} if m else {}))
                    if arm in set(rows["arm"]):
                        pick["arm"] = arm
                try:
                    best, _, _ = rp.choose_best(rows, pick, a.rank)
                    print("transfer:", best["label"], best["arm"], best["model"], flush=True)
                    rp.transfer_shares_cached(train, test, rp._space(best), best["arm"], best["model"])
                    rp.transfer_predictions_cached(train, test, best, fc_arm=a.fc_arm)
                except (Exception, SystemExit) as e:
                    print(f"  FAILED transfer {sp} {m} {arm}: {e}", flush=True)
        if "contrib" not in a.skip:
            rp.transfer_contrib(train, test, rows, arms=rp.transfer_share_arms(train_cfg, rows))

if __name__ == "__main__":
    main()
