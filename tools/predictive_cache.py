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


def grid(st, states, exclude=(), arms=None):
    """(atlas, window_s, states, model, arm) select-bstm ran, every model, the graph_arms; for one
    state set, or every state set with states="ALL". `exclude`: "atlas,window_s,model,arm" (any
    state set) or "atlas,window_s,states,model,arm"."""
    s = st.summ
    s = s[(s["tree"] == "bstm") & s["arm"].isin(st.cfg.graph_arms) & s["model"].isin(st.models())]
    if states != "ALL":
        s = s[s["states"] == states]
    if arms:
        s = s[s["arm"].isin(arms)]
    out = sorted({(r.atlas, float(r.window_s), r.states, r.model, r.arm) for r in s.itertuples()})
    skip = [x.split(",") for x in exclude]

    def hit(c):
        for x in skip:
            w = f"{c[1]:g}"
            if len(x) == 4 and (x[0], x[1], x[2], x[3]) in {(c[0], w, c[3], c[4]), (c[0], str(c[1]), c[3], c[4])}:
                return True
            if len(x) == 5 and (x[0], x[1], x[2], x[3], x[4]) in {(c[0], w, *c[2:]), (c[0], str(c[1]), *c[2:])}:
                return True
        return False
    return [c for c in out if not hit(c)]


def transfer_arm(arm, have):
    """The transfer's name for a Cam-CAN arm, or None. The transfer runs its arms WITHOUT a suffix,
    so its cells stay as stored, p(i, j): its "all" is Cam-CAN's "all [joint]". "all [cond]"
    (p(j | i)) has no transfer unless one was run under that name."""
    if arm in have:
        return arm
    base, _, sfx = arm.partition(" [")
    return base if sfx == "joint]" and base in have else None


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
    ap.add_argument("--grid", metavar="STATES",
                    help="every atlas x window x model x arm (graph_arms) select-bstm ran for these states "
                         "(ALL = every state set too); "
                         "with --task-id, only that one combination (a job array)")
    ap.add_argument("--grid-arms", nargs="+", default=None, metavar="ARM",
                    help='with --grid: only these arms, e.g. "all [joint]" (the arm the transfer ran)')
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="with --grid and --task-id: the array has this many tasks, task i takes "
                         "combinations i, i + n, i + 2n, ... (default: one combination per task)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help='combinations to leave out, "atlas,window_s,model,arm" (e.g. ones already running)')
    ap.add_argument("--list", action="store_true", help="with --grid: print the combinations and their count, fit nothing")
    a = ap.parse_args(argv)

    over = {"n_jobs": a.n_jobs} if a.n_jobs else {}
    if a.grid:
        combos = grid(ht.Study(rp.cam_settings(**over)), a.grid, a.exclude, a.grid_arms)
        if a.list:
            for k, c in enumerate(combos):
                print(k, *c, sep="\t")
            print(f"N={len(combos)}")
            return
        if a.task_id is not None:                  # this task's share: every n_tasks-th combination
            combos = combos[a.task_id::a.n_tasks or len(combos)]
            print(f"task {a.task_id} of {a.n_tasks or len(combos)}: {len(combos)} combination(s)", flush=True)
            if not combos:
                return
        a.space = sorted({f"{c[0]},{c[1]},{c[2]}" for c in combos})
        a.model = sorted({c[3] for c in combos})
        a.arm = sorted({c[4] for c in combos})
        if a.task_id is not None:
            a.skip = sorted(set(a.skip) | {"shares", "contrib"})   # shared tables: not in every task
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
                    have = set(rows["arm"])
                    a_t = transfer_arm(arm, have)
                    if a_t is None:
                        print(f"  SKIPPED transfer {sp} {m} {arm}: no such arm in the transfer rows "
                              f"({sorted(have)})", flush=True)
                        continue
                    pick["arm"] = a_t
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
