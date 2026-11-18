"""The predictive-model notebooks' shared setup and their slow parts, cached on disk.

`06_PredictiveModel_Exploration` and `07_PredictiveModel_Final` both start here, so the notebooks
and the cluster job (tools/predictive_cache.py, slurm/predictive_cache.sbatch) build the SAME
objects with the SAME settings, and a figure in the notebook reads what the job computed instead
of refitting it:

  * Cam-CAN (cross-validated): `cam_settings()` -> `ht.Study`, whose own disk cache holds the
    baselines, the SHAP shares (C) and the refits with out-of-fold SHAP (`deep`).
  * Cam-CAN -> HCP (transfer): `transfer_setup()`, `transferred_rows()`, `choose_best()`, and the
    fits the transfer figures need -- SHAP on the test subjects (`transfer_shares_cached`), the
    predictions (`transfer_predictions_cached`) and the SHAP shares of the best transferred
    configurations (`transfer_contrib`) -- cached as parquet beside the Cam-CAN cache.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pandas as pd

from fmri_decomposition.supervised_model import hads as ht

# notebook 10's settings, the figures' and the cache's
CAM = dict(
    case_from="Mild", alpha=0.10, fc_line=0.25, shrink=1.0, min_present=0.10,
    covariates=["Age", "Sex", "n_transitions"], entropy_estimator="miller_madow",
    multiverse_arm="all [cond]", xlim=2, label_max=25,
    models=None, arms=None, baseline_covariates=["Age", "Sex"], seeds=[0, 1, 2, 3, 4], margin_sd=0,
    highlight_max=10,
    factors={"window": "window size", "atlas": "atlas", "model": "model", "method": "state method",
             "arm": "arm", "K_": "number of states (K)"},
    factor_trees=["bstm"], share_arms=["all [joint]", "all [cond]"], share_top=10, shap_seed=0,
    graph_arms=["all [joint]", "all [cond]"], edge_top=12, label_pp=None, top_vars=15,
    seeds_stab=[0, 1, 2, 3, 4], highlight_top=8, stable_min=0.8, min_d=0.2,
    font_scale=2.0, fig_scale=1.5, dyn_gap=2.0, dyn_away=4.0, dyn_icon=1.0, save_dir=None)

HCP_PHENO = os.environ.get("HCP_PHENO", "/project/6008063/tamires/cohorts/hcp_pheno/participants_pheno.csv")


def cam_settings(**over) -> ht.Settings:
    return ht.Settings(**{**CAM, **over})


def cam_choices(st, space=None, model=None):
    """(base, space, model) as the notebook's defaults pick them: the best state set by CV
    (st.space_options, without the multiverse) and the best model of graph_arms on it."""
    base = st.baselines()
    options = st.space_options(base)
    space = st.choose_space(space, options)
    model = st.choose_model(model, st.model_options(space))
    return base, options, space, model


# -- transfer -----------------------------------------------------------------------------------
def transfer_setup(target="category", config=None, save_dir=None, hcp_pheno=HCP_PHENO, pool_record=None,
                   transferred=None):
    """(train, test, train_cfg, test_cfg, ROOT): notebook 13's setup."""
    from fmri_decomposition.supervised_model import transfer_config as tcfg
    from fmri_decomposition.supervised_model import transfer_tables as tt
    train, test, train_cfg, test_cfg = tcfg.studies(target, config=config, save_dir=save_dir,
                                                    pool_record=pool_record, hcp_pheno=hcp_pheno)
    root = (Path(transferred) if transferred else
            tt.tree_root(train_cfg.root, "transferred_models", train_cfg.modelling_version))
    return train, test, train_cfg, test_cfg, root


def transferred_rows(root, target="category"):
    from fmri_decomposition.supervised_model import transfer_tables as tt
    raw = tt.read_transferred(root, target, trees=("bstm", "fcm"))
    return raw, tt.drop_degenerate(tt.as_report_rows(raw))


def choose_best(rows, pick=None, rank=0):
    """Notebook 13's choice: bstm rows narrowed by `pick` (column -> value), those whose interval
    clears their baseline first, the rank-th best by score. Returns (best, ranked, n_clear)."""
    pool = rows[rows["tree"] == "bstm"]
    for col, want in (pick or {}).items():
        if col not in pool.columns:
            raise SystemExit(f"{col!r} is not a column (model, arm, atlas, window_s, states, task, phenotype)")
        left = pool[pool[col] == want]
        if not len(left):
            raise SystemExit(f"nothing left after {col} == {want!r}; present: {sorted(pool[col].unique())[:12]}")
        pool = left
    clears = pool[pool["lo"] > pool["base_score"]].sort_values("score", ascending=False)
    ranked = clears if len(clears) else pool.sort_values("score", ascending=False)
    if rank >= len(ranked):
        raise SystemExit(f"rank={rank} but only {len(ranked)} row(s)")
    return ranked.iloc[rank], ranked, len(clears)


def _cache_dir(train) -> Path | None:
    root = train.cache_root
    return None if root is None else Path(root) / "transfer"


def cached(train, name, key, compute):
    """A parquet beside the Cam-CAN cache, keyed by the training Study's stamp and `key`."""
    d = _cache_dir(train)
    if d is None:
        return compute()
    p = d / f"{name}_{hashlib.md5(repr((train.stamp, key)).encode()).hexdigest()[:12]}.parquet"
    if p.exists():
        print(f"{name}: read {p}")
        return pd.read_parquet(p)
    if getattr(train, "cache_only", False):
        raise SystemExit(f"{name} is not cached for {key} and the Study is cache_only: run "
                         f"slurm/predictive_cache.sbatch with the notebook's parameters, then rerun this cell.")
    t0 = time.time()
    out = compute()
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p)
    print(f"{name}: built in {time.time() - t0:.0f}s, cached in {p}")
    return out


def _space(r):
    return {"atlas": r["atlas"], "window_s": float(r["window_s"]), "states": r["states"]}


def transfer_shares_cached(train, test, space, arm, model):
    from fmri_decomposition.supervised_model import transfer_tables as tt
    key = ("shares", space["atlas"], float(space["window_s"]), space["states"], arm, model)
    return cached(train, "transfer_shares", key, lambda: tt.transfer_shares(train, test, space, arm, model))


def transfer_predictions_cached(train, test, best, fc_arm="edges"):
    from fmri_decomposition.supervised_model import transfer_tables as tt
    key = ("pred", best["atlas"], float(best["window_s"]), best["states"], best["arm"], best["model"], fc_arm)
    return cached(train, "transfer_predictions", key, lambda: tt.transfer_predictions(train, test, best, fc_arm=fc_arm))


def transfer_share_arms(train_cfg, rows):
    """The share arms the transfer has (its arm names may lack the [joint]/[cond] suffix)."""
    have = set(rows.loc[rows["tree"] == "bstm", "arm"])
    want = [a for a in train_cfg.share_arms if a in have]
    return want or sorted(a for a in have if a.startswith("all")) or None


def transfer_contrib(train, test, rows, arms=None, top=None):
    """The transfer's analogue of notebook 10's C (`Study.shares`): for the `top` best transferred
    bstm configurations per (task, phenotype, model, arm), the share of the model's mean |SHAP|
    on the HCP subjects that each variable type takes (one fit on all of Cam-CAN). The shape
    `hp.fig_shares` reads; `cv` is the transferred score."""
    c = train.cfg
    top = c.share_top if top is None else top
    b = rows[rows["tree"] == "bstm"]
    if arms:
        b = b[b["arm"].isin(arms)]
    b = (b.sort_values("score", ascending=False)
         .groupby(["task", "phenotype", "model", "arm"], sort=False).head(top))
    configs = b[["atlas", "window_s", "states", "arm", "model"]].drop_duplicates()
    print(f"transfer SHAP shares: {len(configs)} configuration(s), 4 fits each")
    parts = []
    for r in configs.itertuples(index=False):
        sh = transfer_shares_cached(train, test, _space(r._asdict()), r.arm, r.model)
        parts.append(sh.assign(atlas=r.atlas, window_s=float(r.window_s), states=r.states, arm=r.arm, model=r.model))
    sh = pd.concat(parts, ignore_index=True)
    keys = ["atlas", "window_s", "states", "arm", "model", "task", "phenotype"]
    b = b.assign(window_s=b["window_s"].astype(float))
    out = sh.merge(b[keys + ["score"]], on=keys, how="inner").rename(columns={"score": "cv"})
    out["label"] = (out["model"] + " · " + out["arm"] + " · "
                    + out.apply(lambda r: ht.set_label(_space(r)), axis=1))
    return out
