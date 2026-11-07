"""Train on one cohort, test on another: the tables behind 13_camcan_to_hcp.

    import hads_tables as ht, transfer_tables as tt
    rows = tt.transfer_rows(train=ht.Study(train_cfg), test=ht.Study(test_cfg))

WHY THIS IS NOT MORE CROSS-VALIDATION
-------------------------------------
`select-bstm` trains and tests inside one cohort: five folds, each model fitted
on four fifths. The limit there is n, because every fold must estimate as many
coefficients as the whole model has.

A transfer test splits those two jobs. TRAINING uses all 649 Cam-CAN subjects
at once -- the best model that cohort can produce. TESTING is one correlation
over the 168 HCP subjects, and a correlation needs far fewer people than a fit
does. Simulated with this pipeline's own estimator, the same 168 subjects give
about 11% power as a training cohort at a true R^2 of 0.20 and about 83% as a
test set at rho = 0.20.

AND IT NEEDS NO EQUATING OF THE TWO QUESTIONNAIRES. Spearman uses ranks. The
prediction comes out in Cam-CAN's units and the truth is in the ASR's, and the
two are never compared as values -- only as order. "Does the model rank these
people the way their own instrument does" is a far weaker assumption than
making HADS and the ASR commensurate, which no linking design here could
support: the instruments share no items and no person took both.

WHERE THE ERROR BAR COMES FROM
------------------------------
Not from folds -- there are none. From resampling the TEST cohort with
replacement and recomputing the score, which is the only randomness left once
the model is fixed. A percentile interval over those is what the figures draw.

WHAT THIS CANNOT TELL YOU
-------------------------
A null here is ambiguous, and the notebook says so where it matters. Three
things must all hold for a transfer score to appear: the state space has to
carry over from 3T to 7T, the model has to carry over from one age range to
another, and the two questionnaires have to measure the same thing in the same
direction. A zero does not say which one failed. So the within-Cam-CAN result
stays the primary finding and this is confirmation -- if Cam-CAN's own CV is
null, a null here adds nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from hads_tables import ARM_BLOCKS, Study, fmt_window, hive, n_states


# ------------------------------------------------------------------ design ---
def align_columns(names_tr: list[str], names_te: list[str]) -> list[str]:
    """The predictor names both cohorts have, in the TRAINING order, or refuse.

    The hazard this exists for is quiet. `Study.matrix` one-hots the covariates
    with `pd.get_dummies`, so the column names depend on the values each cohort
    happens to contain: Cam-CAN's `Sex` of MALE/FEMALE and HCP's of M/F produce
    DIFFERENT dummy columns from the same variable. Line them up by position
    and the model would read sex as something else entirely, with every number
    downstream looking perfectly reasonable.

    So: match by NAME, and refuse rather than guess. The state-set columns
    (the cells) cannot differ -- the same `states=` key means the same K and
    the same cell names -- so a mismatch here is always about the covariates,
    and the message says which.
    """
    tr, te = list(names_tr), set(names_te)
    missing = [n for n in tr if n not in te]
    if missing:
        extra = [n for n in names_te if n not in set(tr)]
        raise SystemExit(
            "the test cohort has no column for: " + ", ".join(missing[:8])
            + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
            + ("\n  it does have, unused: " + ", ".join(extra[:8]) if extra else "")
            + "\n  These are covariate dummies, not state cells: the same "
              "`states=` key gives both cohorts the same cells. Set "
              "`covariate_aliases` on the test Settings so the two spell the "
              "same variable the same way, e.g. {'Sex': 'Gender', "
              "'Age': 'Age_in_Yrs'}.")
    return tr


def reorder(names_te: list[str], X_te: np.ndarray, want: list[str]) -> np.ndarray:
    """Test columns in the training order. Position is never assumed."""
    idx = {n: i for i, n in enumerate(names_te)}
    return X_te[:, [idx[n] for n in want]]


# ------------------------------------------------------------------ scoring ---
def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n_boot: int, seed: int,
                 alpha: float = .05) -> tuple[float, float, float]:
    """(score, lo, hi) by resampling the TEST subjects with replacement.

    The model is fixed, so the only randomness left is which people were
    tested. A fold-based interval would be meaningless here -- there are no
    folds -- and quoting a bare correlation with no interval would hide that
    168 subjects is 168 subjects.
    """
    rng = np.random.default_rng(seed)
    obs = stats.spearmanr(pred, y).statistic
    n = len(y)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        # a resample can be constant in y (or in pred); Spearman is undefined
        # there and returns NaN, which nanpercentile then steps over
        draws[b] = (np.nan if len(np.unique(y[i])) < 2
                    else stats.spearmanr(pred[i], y[i]).statistic)
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(obs), float(lo), float(hi)


def fit_predict(X_tr, y_tr, X_te, model: str, seed: int) -> np.ndarray:
    """One model, fitted on EVERY training subject, applied to the test set.

    `make_model`, not `hads_tables._fit`: that one calls `score_once` and hands
    back a cross-validated FLOAT, which is the right thing inside one cohort
    and the wrong thing here -- there are no folds to cross-validate over, and
    a number cannot predict anything. This builds the same estimator
    select-bstm would, fits it on every training row, and predicts.
    """
    from threadpoolctl import threadpool_limits
    from sklearn.base import clone
    from fmri_decomposition.bstm_selection import make_model

    est, kind, _ = make_model(model, seed)
    with threadpool_limits(1):     # many tiny fits; let the outer loop have the cores
        if kind == "clf":
            # the only classifier here is `logistic`, binarised at y > 0 the
            # way score_once does, and its prediction is a probability so that
            # Spearman still means something
            fitted = clone(est).fit(X_tr, (y_tr > 0).astype(int))
            return fitted.predict_proba(X_te)[:, 1]
        return clone(est).fit(X_tr, y_tr).predict(X_te)


# -------------------------------------------------------------------- rows ---
def common_sets(train: Study, test: Study) -> pd.DataFrame:
    """The (atlas, window_s, states) both cohorts have a transition table for.

    A set present in only one is not an error and not a result: it is a stage
    that has not run yet for the other cohort. Reported by the caller, skipped
    here.
    """
    a = train.sets().set_index(["atlas", "window_s", "states"])
    b = test.sets().set_index(["atlas", "window_s", "states"])
    return a.loc[a.index.intersection(b.index)].reset_index()


def transfer_rows(train: Study, test: Study, arms=None, models=None,
                  seed: int = 0, n_boot: int = 2000, log=print) -> pd.DataFrame:
    """One row per (atlas, window, states, task, target, arm, model).

    `score` is Spearman on the test cohort; `lo`/`hi` its bootstrap interval;
    `base_score` the same for a covariates-only model trained the same way, so
    the comparison is like for like. Both targets are each cohort's own: the
    labels must match between the two Settings, the columns need not.
    """
    arms = arms or [a for a in ARM_BLOCKS if a != "summary"]
    models = models or train.models()
    labels = [lab for lab in train.cfg.targets if lab in test.cfg.targets]
    if not labels:
        raise SystemExit(
            f"no target label in common: train has {list(train.cfg.targets)}, "
            f"test has {list(test.cfg.targets)}. The LABELS must match (both "
            f"'anxiety'); the columns behind them need not.")
    tasks = [t for t in train.cfg.tasks if t in test.cfg.tasks]
    sets = common_sets(train, test)
    log(f"{len(sets)} state set(s) in both cohorts; "
        f"{len(tasks)} task(s), {len(labels)} target(s), "
        f"{len(arms)} arm(s), {len(models)} model(s)")

    rows = []
    for s in sets.itertuples():
        space = {"atlas": s.atlas, "window_s": s.window_s, "states": s.states}
        for task in tasks:
            for lab in labels:
                # The baseline is computed ONCE per (space, task, target), not
                # once per arm: `matrix` drops subjects on the target and the
                # covariates, never on the arm's own columns, so every arm of
                # one cell has the same people and the same covariate block.
                # Fitting it per arm would be the same fit under six names.
                base = {}
                for arm in arms:
                    try:
                        Xtr, ytr, ntr, btr = train.matrix(space, task, lab, arm)
                        Xte, yte, nte, _ = test.matrix(space, task, lab, arm)
                    except SystemExit as e:        # a set one cohort lacks
                        log(f"  skip {s.states} {task} {arm}: {e}")
                        continue
                    want = align_columns(ntr, nte)
                    Xte = reorder(nte, Xte, want)
                    cov = [i for i, b in enumerate(btr) if b == "covariates"]

                    for model in models:
                        if model not in base:
                            bp = fit_predict(Xtr[:, cov], ytr, Xte[:, cov],
                                             model, seed)
                            base[model] = bootstrap_ci(yte, bp, n_boot, seed)
                        bsc, blo, bhi = base[model]
                        pred = fit_predict(Xtr, ytr, Xte, model, seed)
                        sc, lo, hi = bootstrap_ci(yte, pred, n_boot, seed)
                        rows.append({
                            "atlas": s.atlas, "window_s": s.window_s,
                            "states": s.states, "K": n_states(s.states),
                            "task": task, "phenotype": lab, "arm": arm,
                            "model": model, "tree": "bstm",
                            "score": sc, "lo": lo, "hi": hi,
                            "base_score": bsc, "base_lo": blo, "base_hi": bhi,
                            "over_base": sc > bsc,
                            "n_train": len(ytr), "n_test": len(yte),
                            "n_covariates": len(cov),
                            "n_features": Xtr.shape[1]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pooling ---
def pool_tasks(d: "pd.DataFrame", n_states: int, log=print) -> "pd.DataFrame":
    """Several runs of one condition -> ONE row per subject, recomputed not averaged.

    HCP gives four movies and four rests per subject; the selections score one
    row per subject and will not average them. Averaging the columns would be
    wrong for most of them, so each is pooled the way its own definition
    allows. Almost all of it is exact:

      cells        p_joint * n_transitions is the COUNT matrix, exactly (the
                   table's own docstring says so). Counts add. The pooled
                   cells are the summed counts over the summed transitions.
      occupancy    occ_k * n_windows is a window count. Counts add.
      n_windows,
      n_transitions, n_transitions_independent      sums.
      n_states_visited, n_distinct_transitions,
      switch_rate, self_transition_rate,
      entropy_rate_bits                             RECOMPUTED from the pooled
                   count matrix, which is what they are functions of. Averaging
                   an entropy rate over runs is not an entropy rate of
                   anything.
      mean_dwell_s the mean dwell is total windows over number of episodes,
                   and the episode count is recoverable: switches + 1, with
                   switches = switch_rate * n_transitions. So the pooled mean
                   dwell is sum(windows) / sum(episodes), times the stride.
      switches_per_min   switches over minutes; both add.

    The two that are NOT exact, and are reported rather than hidden:

      stride_s     not a column. It is recovered per run from
                   switches_per_min, and the runs of one subject must agree:
                   a pooled rate over runs sampled at different strides is not
                   a rate. Mixed strides REFUSE.
      dispersion   the mean distance from the subject's centroid in latent
                   space, which needs the latents, not this table. Pooled as a
                   mean weighted by n_windows, which is the centroid of each
                   run rather than of all of them together. Flagged in the
                   returned frame as `dispersion_pooled_approximately`.
    """
    from fmri_decomposition.transitions import _entropy_rate
    from hads_tables import cell_cols

    cells = cell_cols(d)
    occ = [c for c in d.columns if c.startswith("occ_")]
    out = []
    for sub, g in d.groupby("sub", sort=True):
        if len(g) == 1:
            out.append(g.iloc[0].to_dict())
            continue
        w = g["n_windows"].to_numpy(float)
        ntr = g["n_transitions"].to_numpy(float)
        counts = (g[cells].to_numpy(float) * ntr[:, None]).sum(0)
        tot_tr = ntr.sum()
        M = counts.reshape(n_states, n_states)
        n_change = tot_tr - np.trace(M)

        # the stride each run was sampled at, recovered and required to agree
        sw = (g["switch_rate"].to_numpy(float) * ntr)       # switches per run
        spm = g["switches_per_min"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            stride = np.where(spm > 0, sw / spm * 60 / np.where(w > 0, w, np.nan),
                              np.nan)
        ok = np.isfinite(stride)
        if ok.sum() and not np.allclose(stride[ok], stride[ok][0], rtol=1e-3):
            raise SystemExit(
                f"sub={sub}: the runs being pooled were sampled at different "
                f"strides ({np.unique(np.round(stride[ok], 3))}). A rate "
                f"pooled across them is not a rate. Pool only runs of one "
                f"condition at one aperture.")
        s = float(stride[ok][0]) if ok.sum() else np.nan

        row = dict(zip(cells, counts / tot_tr if tot_tr else np.nan))
        row.update({f: (g[f].to_numpy(float) * w).sum() / w.sum()
                    for f in occ})                       # window counts, added
        episodes = sw.sum() + len(g)                     # switches + 1 per run
        row.update({
            "sub": sub,
            "task": "+".join(sorted(g["task"].astype(str))),
            "n_windows": w.sum(),
            "n_transitions": tot_tr,
            "n_transitions_independent": g["n_transitions_independent"].sum(),
            "n_states": n_states,
            "n_states_visited": int(sum(row[f] > 0 for f in occ)),
            "n_distinct_transitions": int((M > 0).sum()),
            "switch_rate": n_change / tot_tr if tot_tr else np.nan,
            "self_transition_rate": 1 - n_change / tot_tr if tot_tr else np.nan,
            "entropy_rate_bits": _entropy_rate(M),
            "switches_per_min": (sw.sum() / (w.sum() * s / 60)
                                 if np.isfinite(s) and w.sum() else np.nan),
            "mean_dwell_s": (w.sum() / episodes * s
                             if np.isfinite(s) and episodes else np.nan),
            "dispersion": ((g["dispersion"].to_numpy(float) * w).sum() / w.sum()
                           if "dispersion" in g else np.nan),
            "dispersion_pooled_approximately": True,
            "n_runs_pooled": len(g),
        })
        for c in g.columns:                        # anything else: first value
            row.setdefault(c, g.iloc[0][c])
        out.append(row)
    pooled = pd.DataFrame(out)
    log(f"  pooled {len(d)} run-rows into {len(pooled)} subject-row(s); "
        f"runs per subject {pooled['n_runs_pooled'].min() if 'n_runs_pooled' in pooled else 1}"
        f"-{pooled['n_runs_pooled'].max() if 'n_runs_pooled' in pooled else 1}")
    return pooled


# ------------------------------------------------------- the report's shape ---
# 1.96 sd each side of a percentile interval, so an interval width -> one sd.
_CI_TO_SD = 2 * 1.959963985


def as_report_rows(rows: "pd.DataFrame") -> "pd.DataFrame":
    """Transfer rows in the shape hads_plots' B1 and B2 already read.

    A NAMED conversion rather than columns quietly emitted by transfer_rows,
    because one of them changes meaning and the figures cannot say so
    themselves.

    In the Cam-CAN report `std` is the spread of one model's score over the
    FOLD SEEDS: how much the answer moves when the folds are drawn again.
    There are no folds here. `std` becomes the bootstrap sd over the TEST
    SUBJECTS: how much the answer moves when a different 168 people are
    tested. Both are an error bar and both belong on the same axis, but they
    are answers to different questions, and a figure captioned "+/- sd over
    seeds" would be wrong about this one. The notebook relabels it; this
    docstring is why.

    The conversion itself: a 95% percentile interval spans 1.96 sd either
    side, so sd = (hi - lo) / 3.92.
    """
    from hads_tables import n_states, set_label, window_name

    d = rows.copy()
    d["mean"] = d["score"]
    d["std"] = (d["hi"] - d["lo"]) / _CI_TO_SD
    d["base_mean"] = d["base_score"]
    d["base_sd"] = (d["base_hi"] - d["base_lo"]) / _CI_TO_SD
    d["window"] = d["window_s"].map(window_name)
    d[["method", "embedding", "K_"]] = (
        d["states"].str.rsplit("_", n=2, expand=True).reindex(columns=range(3)))
    d["label"] = d.apply(set_label, axis=1)
    d["entity"] = d["tree"] + " · " + d["label"] + " · " + d["arm"] + " · " + d["model"]
    d["K"] = d["states"].map(n_states)
    return d


# --------------------------------------------------------------- static FC ---
def pool_fc(d: "pd.DataFrame", log=print) -> "pd.DataFrame":
    """Several runs of one condition -> one row per subject, for a static-FC table.

    NOT exact, unlike `pool_tasks`, and the difference is worth stating. A
    transition table stores counts in disguise, so pooling them is addition. A
    correlation is not: the correlation of four runs concatenated is not any
    function of the four correlations, because each was computed against its
    own run's mean.

    What is done instead is the standard meta-analytic pooling -- a mean of
    the Fisher-z values weighted by how much data each run contributed
    (`n_tr_used`). It is the right approximation and it is still an
    approximation.

    THE EXACT ALTERNATIVE EXISTS: `static-fc --pool subject` concatenates a
    subject's timeseries and correlates once. It pools every task though, rest
    together with movie, which is not what a per-condition score wants. A
    `--pool` that took a task filter would make this function unnecessary.
    """
    from fmri_decomposition import static_fc as _sfc

    cols = _sfc.fc_columns(d)
    if d["sub"].is_unique:
        return d
    out = []
    for sub, g in d.groupby("sub", sort=True):
        if len(g) == 1:
            out.append(g.iloc[0].to_dict())
            continue
        w = g["n_tr_used"].to_numpy(float) if "n_tr_used" in g else np.ones(len(g))
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if not w.sum():
            w = np.ones(len(g))
        z = np.arctanh(np.clip(g[cols].to_numpy(float), -0.999999, 0.999999))
        # a NaN edge in one run must not take the subject's whole edge with it
        m = np.isfinite(z)
        num = np.nansum(np.where(m, z, 0) * w[:, None], axis=0)
        den = (m * w[:, None]).sum(axis=0)
        pooled = np.divide(num, den, out=np.full(len(cols), np.nan), where=den > 0)
        row = dict(zip(cols, np.tanh(pooled)))           # back to r, as stored
        row.update({
            "sub": sub,
            "task": "+".join(sorted(g["task"].astype(str))),
            "n_runs_pooled": len(g),
            "fc_pooled_approximately": True,
        })
        for c in ("n_tr_total", "n_tr_good", "n_tr_used", "n_tasks"):
            if c in g:
                row[c] = g[c].sum()
        if {"frac_good_frames", "n_tr_used"} <= set(g.columns):
            row["frac_good_frames"] = float(
                (g["frac_good_frames"].to_numpy(float) * w).sum() / w.sum())
        for c in g.columns:
            row.setdefault(c, g.iloc[0][c])
        out.append(row)
    pooled = pd.DataFrame(out)
    log(f"  pooled {len(d)} run-rows into {len(pooled)} subject-row(s)")
    return pooled


def _fc_cell(root, atlas, cohorts, task_value, pheno, covariates, categorical,
             pool: bool, log):
    """One cohort group's static-FC table and phenotype, as (edges_r, cov, y, subs)."""
    from fmri_decomposition import static_fc as _sfc

    frames = [_sfc.read_cohort_table(root, atlas, c) for c in cohorts]
    d = pd.concat(frames, ignore_index=True)
    d["sub"] = d["sub"].astype(str).str.strip().str.upper()
    if task_value is not None:
        d = d[d["task"].astype(str).str.upper().str.startswith(task_value.upper())]
    if d.empty:
        raise SystemExit(f"no static-FC rows for {cohorts} atlas={atlas} "
                         f"task~{task_value}")
    if pool:
        d = pool_fc(d, log=lambda *_: None)
    if not d["sub"].is_unique:
        raise SystemExit(
            f"{cohorts} atlas={atlas} task~{task_value}: a subject appears "
            f"more than once and pooling is off. Either narrow the task or "
            f"pass pool=True.")
    d = d[d["sub"].isin(set(pheno["sub"]))].sort_values("sub").reset_index(drop=True)
    subs = d["sub"].tolist()
    cols = _sfc.fc_columns(d)
    return d, cols, subs


def fcm_transfer_rows(train: Study, test: Study, atlases=None,
                      train_tasks=None, test_tasks=None,
                      models=None, covariates=None, categorical=("Sex",),
                      max_missing: float = .10, seed: int = 0,
                      n_boot: int = 2000, pool_test: bool = True,
                      log=print) -> pd.DataFrame:
    """The fcm half of the transfer, so the four tables can be compared.

    Mirrors fcm_selection, split into a fit and an apply. Two things have to
    come from the TRAINING cohort alone, and both would be invisible if they
    did not:

      WHICH EDGES. fcm_selection drops an edge that is NaN in more than
      `max_missing` of subjects, so the surviving set depends on the cohort's
      own coverage. Let the test cohort choose its own and the model is handed
      a different feature space than it was fitted on, with the columns still
      lining up by count.

      THE IMPUTED VALUE. A NaN is filled with the column mean. Filled from the
      TEST cohort's mean, every test subject's missing edge is shifted toward
      that cohort's centre -- a quiet adaptation to the data being scored.

    Everything else follows transfer_rows: Spearman on the test cohort, a
    bootstrap interval over the test subjects, and a covariates-only baseline
    trained and applied the same way.
    """
    from fmri_decomposition import static_fc as _sfc

    root = train.cfg.root
    covariates = list(covariates or ["Age", "Sex", "frac_good_frames", "n_tr_used"])
    models = models or train.models()
    labels = [lab for lab in train.cfg.targets if lab in test.cfg.targets]
    atlases = atlases or sorted({hive(p, "atlas") for p in
                                 (root / "static_fc").rglob("subjects.parquet")})
    tr_tasks = train_tasks or {k: v for k, v in train.cfg.tasks.items()}
    te_tasks = test_tasks or {k: v for k, v in test.cfg.tasks.items()}
    rows = []
    for atlas in atlases:
        for task in [t for t in tr_tasks if t in te_tasks]:
            try:
                dtr, ctr, str_ = _fc_cell(root, atlas, train.cfg.cohorts,
                                          tr_tasks[task], train.pheno,
                                          covariates, categorical, False, log)
                dte, cte, ste = _fc_cell(root, atlas, test.cfg.cohorts,
                                         te_tasks[task], test.pheno,
                                         covariates, categorical, pool_test, log)
            except SystemExit as e:
                log(f"  skip fcm {atlas} {task}: {e}")
                continue
            edges = [c for c in ctr if c in set(cte)]
            if not edges:
                raise SystemExit(
                    f"fcm {atlas} {task}: the two cohorts share no edge "
                    f"column. Same atlas must give the same parcels -- check "
                    f"that both were run with `static-fc --atlas {atlas}`.")

            Ztr = np.arctanh(np.clip(dtr[edges].to_numpy(float), -.999999, .999999))
            Zte = np.arctanh(np.clip(dte[edges].to_numpy(float), -.999999, .999999))
            keep = np.isnan(Ztr).mean(axis=0) <= max_missing       # TRAIN only
            Ztr, Zte = Ztr[:, keep], Zte[:, keep]
            mu = np.nanmean(Ztr, axis=0)                           # TRAIN only
            Ztr = np.where(np.isnan(Ztr), mu, Ztr)
            Zte = np.where(np.isnan(Zte), mu, Zte)

            arms = {"edges": (Ztr, Zte),
                    "global": (np.column_stack([Ztr.mean(1), Ztr.std(1)]),
                               np.column_stack([Zte.mean(1), Zte.std(1)]))}
            for lab in labels:
                ytr = train.pheno.set_index("sub").reindex(str_)[lab].to_numpy(float)
                yte = test.pheno.set_index("sub").reindex(ste)[lab].to_numpy(float)
                ok_tr, ok_te = np.isfinite(ytr), np.isfinite(yte)
                Ctr = _cov_frame(train.pheno, dtr, str_, covariates, categorical)
                Cte = _cov_frame(test.pheno, dte, ste, covariates, categorical)
                want = align_columns(list(Ctr.columns), list(Cte.columns))
                Ctr_, Cte_ = Ctr[want].to_numpy(float), Cte[want].to_numpy(float)
                base = {}
                for arm, (Atr, Ate) in arms.items():
                    Xtr = np.column_stack([Ctr_, Atr])[ok_tr]
                    Xte = np.column_stack([Cte_, Ate])[ok_te]
                    for model in models:
                        if model not in base:
                            bp = fit_predict(Ctr_[ok_tr], ytr[ok_tr],
                                             Cte_[ok_te], model, seed)
                            base[model] = bootstrap_ci(yte[ok_te], bp, n_boot, seed)
                        bsc, blo, bhi = base[model]
                        pred = fit_predict(Xtr, ytr[ok_tr], Xte, model, seed)
                        sc, lo, hi = bootstrap_ci(yte[ok_te], pred, n_boot, seed)
                        rows.append({
                            "atlas": atlas, "window_s": np.nan,
                            "states": "(static FC)", "K": np.nan,
                            "task": task, "phenotype": lab, "arm": arm,
                            "model": model, "tree": "fcm",
                            "score": sc, "lo": lo, "hi": hi,
                            "base_score": bsc, "base_lo": blo, "base_hi": bhi,
                            "over_base": sc > bsc,
                            "n_train": int(ok_tr.sum()), "n_test": int(ok_te.sum()),
                            "n_covariates": len(want),
                            "n_features": Xtr.shape[1],
                            "n_edges_used": int(keep.sum()),
                            "n_edges_dropped": int((~keep).sum())})
    return pd.DataFrame(rows)


def _cov_frame(pheno, quality, subs, covariates, categorical):
    """covariate_block's design, kept as a NAMED frame so it can be aligned.

    fcm_selection returns a bare array, which is right there -- one cohort,
    one call. Across two cohorts the names are the only safe way to line the
    columns up, since get_dummies names them after the values each cohort
    holds.
    """
    ph = pheno.set_index("sub").reindex(subs)
    q = quality.set_index("sub").reindex(subs)
    cols = []
    for c in covariates:
        src = q if (c in q.columns and c not in ph.columns) else ph
        if c not in src.columns:
            raise SystemExit(
                f"covariate {c!r} is in neither the phenotype nor the "
                f"static-FC table.\n  phenotype: "
                f"{[x for x in ph.columns if x != 'y'][:10]}\n  static FC: "
                f"{[x for x in q.columns if '__' not in x][:10]}")
        cols.append(src[c].rename(c))
    X = pd.concat(cols, axis=1)
    X = pd.get_dummies(X, columns=[c for c in categorical if c in X.columns],
                       drop_first=True, dummy_na=False).astype(float)
    return X.fillna(X.mean())
