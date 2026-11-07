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
