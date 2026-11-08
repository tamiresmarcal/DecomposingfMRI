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

import functools
import re
import time
from collections import Counter

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


def impute_from_train(Xtr, Xte, names, blocks, log=print):
    """NaN filled with the TRAINING column mean; a column train cannot fill dropped.

    WHY THIS IS NEEDED AT ALL. `Study.matrix` fills the COVARIATES
    (`cov.fillna(cov.mean())`) and drops any subject missing a covariate or
    the target -- but it does nothing about a NaN in the arm's own columns,
    because inside one cohort of single runs there are none to find. POOLING
    makes some: `mean_dwell_s` and `switches_per_min` need the stride,
    recovered per run from `switches_per_min`, and a run in which nobody
    switched gives 0/0; `dispersion` needs the latents rather than this table.
    RidgeCV then raises `Input X contains NaN` from inside `.predict`, seven
    frames from the column that caused it.

    WHY THE TRAINING MEAN, as in `fcm_transfer_rows`: filled from the test
    cohort's own mean, every test subject's missing value is nudged toward the
    centre of the data being scored, which is adaptation to the test set.

    A column that is NaN for EVERY training subject has no mean to fill from,
    and is dropped from BOTH sides -- the model could not have learned
    anything from it either. Returns the count of each, to be written into the
    row rather than left for the reader to wonder about.
    """
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    # column by column, because np.nanmean over an all-NaN column warns
    # ("Mean of empty slice") and a warning in a 9,000-fit loop is noise that
    # buries the messages that matter
    keep = np.isfinite(Xtr).any(axis=0) if len(Xtr) else np.zeros(Xtr.shape[1], bool)
    mu = np.full(Xtr.shape[1], np.nan)
    if keep.any():
        mu[keep] = np.nanmean(Xtr[:, keep], axis=0)
    dropped = [n for n, k in zip(names, keep) if not k]
    if dropped:
        log(f"  dropping {len(dropped)} column(s) the training cohort cannot "
            f"fill (all-NaN there): {', '.join(dropped[:6])}"
            + (f" +{len(dropped) - 6} more" if len(dropped) > 6 else ""))
    Xtr, Xte, mu = Xtr[:, keep], Xte[:, keep], mu[keep]
    names = [n for n, k in zip(names, keep) if k]
    blocks = [b for b, k in zip(blocks, keep) if k]
    n_tr, n_te = int(np.isnan(Xtr).sum()), int(np.isnan(Xte).sum())
    Xtr = np.where(np.isnan(Xtr), mu, Xtr)
    Xte = np.where(np.isnan(Xte), mu, Xte)
    return Xtr, Xte, names, blocks, {"n_imputed_train": n_tr,
                                     "n_imputed_test": n_te,
                                     "n_columns_dropped": len(dropped)}


# ------------------------------------------------------------------ scoring ---
def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n_boot: int, seed: int,
                 alpha: float = .05) -> tuple[float, float, float]:
    """(score, lo, hi) by resampling the TEST subjects with replacement.

    The model is fixed, so the only randomness left is which people were
    tested. A fold-based interval would be meaningless here -- there are no
    folds -- and quoting a bare correlation with no interval would hide that
    168 subjects is 168 subjects.
    """
    def rho(a, b):
        # Spearman is undefined on a constant input and scipy says so with a
        # ConstantInputWarning. That is a real outcome -- a model that predicts
        # one value for everyone has ranked nobody -- but 2,000 draws x 9,000
        # rows of the same warning buries every message that matters, so the
        # case is decided here and the NaN returned quietly.
        if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            return np.nan
        return stats.spearmanr(a, b).statistic

    rng = np.random.default_rng(seed)
    obs = rho(pred, y)
    n = len(y)
    draws = np.array([rho(pred[i], y[i])
                      for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    if not np.isfinite(draws).any():
        return float(obs), float("nan"), float("nan")
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(obs), float(lo), float(hi)


@functools.lru_cache(maxsize=None)
def usable_threads(threads: int) -> int:
    """`threads`, but NEVER more than the process already allows.

    threadpoolctl can lower a thread limit safely and cannot safely RAISE one.
    OpenBLAS sizes its per-thread buffers the first time it runs, so asking for
    4 threads in a process that started at OPENBLAS_NUM_THREADS=1 can walk off
    the end of them.

    That is what killed the cluster array, twice, and the traceback said so
    exactly:

        scipy/linalg/_decomp.py  eigh
        sklearn/linear_model/_ridge.py  _eigen_decompose_gram
        ...
        transfer_tables.py  fit_predict
        Segmentation fault

    `_eigen_decompose_gram` is RidgeCV's n_features > n_samples branch -- which
    is why the K=4 sets finished (16 cells, fewer features than subjects, a
    different solver) and the K=27 sets died (729 cells). The first array had
    the limits unset, the second had them at 1 inside the container; both then
    had `threadpool_limits(4)` raise the count mid-process.

    A thread count has to be raised BEFORE any BLAS call, which means in the
    ENVIRONMENT, not here. Inside `apptainer --cleanenv` that takes
    APPTAINERENV_*, because --cleanenv drops a plain export -- see
    slurm/transfer.sbatch.
    """
    cur = _current_threads()
    return max(1, threads if cur is None else min(threads, cur))


@functools.lru_cache(maxsize=None)
def _current_threads():
    """The tightest thread limit the loaded BLAS/OpenMP runtimes already have.

    Cached: nothing here ever RAISES a limit, so the answer cannot go up, and
    asking per fit would cost a library scan 9,000 times.
    """
    try:
        from threadpoolctl import threadpool_info
        have = [i["num_threads"] for i in threadpool_info() if i.get("num_threads")]
    except Exception:                                   # pragma: no cover
        return None
    return min(have) if have else None


def thread_ctx(threads: int):
    """A limiter only when it LOWERS the count; otherwise nothing at all.

    Setting a limit to the value it already has ought to be harmless, and
    probably is. But both crashes were inside OpenBLAS's LAPACK right after a
    limit had been set, and a no-op that touches nothing cannot be the cause of
    a third. With the default of one thread in a container that already says
    one, this path never calls into OpenBLAS's thread machinery at all.
    """
    import contextlib

    use, cur = usable_threads(threads), _current_threads()
    if cur is None or use >= cur:
        return contextlib.nullcontext()
    from threadpoolctl import threadpool_limits
    return threadpool_limits(use)


def fit_predict(X_tr, y_tr, X_te, model: str, seed: int,
                threads: int = 1) -> np.ndarray:
    """One model, fitted on EVERY training subject, applied to the test set.

    `make_model`, not `hads_tables._fit`: that one calls `score_once` and hands
    back a cross-validated FLOAT, which is the right thing inside one cohort
    and the wrong thing here -- there are no folds to cross-validate over, and
    a number cannot predict anything. This builds the same estimator
    select-bstm would, fits it on every training row, and predicts.
    """
    from sklearn.base import clone
    from fmri_decomposition.bstm_selection import make_model

    est, kind, _ = make_model(model, seed)
    # One thread by default: inside a joblib loop a threaded estimator
    # oversubscribes the node and runs slower than either alone. The sharded
    # script has no such loop -- one process, one cell at a time -- so there
    # `threads` is the task's whole cpu allocation, IF the process started with
    # room for them. `usable_threads` is why it cannot be taken on trust.
    with thread_ctx(threads):
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


# The models a transfer fits when nothing says otherwise. NOT `train.models()`:
# that reads the model names out of the stored SELECTION SUMMARIES, which only
# exist for the target columns the selections were run on. Ask for a target
# they were not run on -- the raw HADS score rather than the banded category --
# and `summ` filters to nothing, `models()` returns [], and the loop below runs
# zero times and hands back an empty frame that then fails seven frames later
# on a missing column. A transfer notebook fits its OWN models; what the
# selections happened to rank is not what it can fit.
DEFAULT_MODELS = ("ridge", "hgb")


def _reason(e) -> str:
    """One skip message with the per-cell names removed, so causes collapse.

    282 state sets x 2 tasks x 2 targets x 6 arms is 13,536 chances to print
    the same sentence. Counting them by cause says the one thing the flood
    did not: THAT it was always the same cause.
    """
    m = " ".join(str(e).split())
    m = re.sub(r"\S*atlas=\S*", "<cell>", m)
    m = re.sub(r"\b(HMM|KMeans|GMM|hmm|kmeans|gmm)\w*", "<states>", m)
    return m


def transfer_rows(train: Study, test: Study, arms=None, models=None,
                  sets=None, seed: int = 0, n_boot: int = 2000,
                  threads: int = 1, log=print) -> pd.DataFrame:
    """One row per (atlas, window, states, task, target, arm, model).

    `score` is Spearman on the test cohort; `lo`/`hi` its bootstrap interval;
    `base_score` the same for a covariates-only model trained the same way, so
    the comparison is like for like. Both targets are each cohort's own: the
    labels must match between the two Settings, the columns need not.

    REFUSES rather than returning an empty frame. Every one of the four loops
    can be emptied by a setting that looks right -- a target the selections
    never ran on, a task label that is not the hive value, a cohort whose
    transitions are not on disk yet -- and an empty frame reports none of
    that. It reports `KeyError` in whatever reads it next.
    """
    arms = arms or [a for a in ARM_BLOCKS if a != "summary"]
    # None means "decide for me"; an empty list is a request for nothing and
    # is refused below rather than quietly turned into the default.
    models = list(DEFAULT_MODELS if models is None and train.cfg.models is None
                  else train.cfg.models if models is None else models)
    labels = [lab for lab in train.cfg.targets if lab in test.cfg.targets]
    if not labels:
        raise SystemExit(
            f"no target label in common: train has {list(train.cfg.targets)}, "
            f"test has {list(test.cfg.targets)}. The LABELS must match (both "
            f"'anxiety'); the columns behind them need not.")
    tasks = [t for t in train.cfg.tasks if t in test.cfg.tasks]
    if not tasks:
        raise SystemExit(
            f"no task in common: train has {list(train.cfg.tasks)}, test has "
            f"{list(test.cfg.tasks)}. The KEYS must match ('movie'); the hive "
            f"values they map to are each cohort's own ('Movie' vs 'MOVIE').")
    # `sets` restricts the grid: the full multiverse is one hgb fit per
    # (set, task, target, arm, model) and there are hundreds of sets, which is
    # hours. A subset of `common_sets(train, test)` is the way to try the
    # chain end to end before paying for all of it.
    sets = common_sets(train, test) if sets is None else sets
    if not len(sets):
        raise SystemExit(
            f"no state set under transitions/ for both cohorts: train has "
            f"{len(train.sets())} for {list(train.cfg.cohorts)}, test has "
            f"{len(test.sets())} for {list(test.cfg.cohorts)}. Stage 5a has to "
            f"have run for the test cohort with the SAME fit -- see "
            f"slurm/project_and_select.sh with STOP_AFTER=transitions.")
    if not models:
        raise SystemExit("no model to fit: `models` is empty and so is "
                         "`train.cfg.models`.")
    n_cell = len(sets) * len(tasks) * len(labels)
    log(f"{len(sets)} state set(s); {len(tasks)} task(s), {len(labels)} "
        f"target(s), {len(arms)} arm(s), {len(models)} model(s): "
        f"{', '.join(models)}\n"
        f"  up to {n_cell * len(arms) * len(models):,} model fit(s) + "
        f"{n_cell * len(models):,} baseline(s), each on every training "
        f"subject. Pass `sets=` a subset of tt.common_sets(train, test) to "
        f"try the chain before paying for all of it.")

    rows, skipped = [], Counter()
    t0 = time.time()
    for i, s in enumerate(sets.itertuples(), 1):
        space = {"atlas": s.atlas, "window_s": s.window_s, "states": s.states}
        for task in tasks:
            # ONE probe per (space, task). A missing transition table is a
            # property of the cell, not of the arm or the target, and loading
            # it once also means a pooled cohort pools once per cell instead
            # of once per arm per target.
            try:
                t_tr = train.load_set(space, task)
                t_te = test.load_set(space, task)
            except SystemExit as e:
                skipped[_reason(e)] += 1
                continue
            for lab in labels:
                # The baseline is computed ONCE per (space, task, target), not
                # once per arm: `matrix` drops subjects on the target and the
                # covariates, never on the arm's own columns, so every arm of
                # one cell has the same people and the same covariate block.
                # Fitting it per arm would be the same fit under six names.
                base = {}
                for arm in arms:
                    try:
                        Xtr, ytr, ntr, btr = train.matrix(space, task, lab, arm, t=t_tr)
                        Xte, yte, nte, _ = test.matrix(space, task, lab, arm, t=t_te)
                    except SystemExit as e:        # an arm one cohort lacks
                        skipped[_reason(e)] += 1
                        continue
                    if len(yte) < 3 or len(np.unique(yte)) < 2:
                        skipped[f"test cohort has {len(yte)} usable subject(s) "
                                f"with {len(np.unique(yte))} distinct score(s) "
                                f"-- Spearman is undefined"] += 1
                        continue
                    want = align_columns(ntr, nte)
                    Xte = reorder(nte, Xte, want)
                    # before any fit: a NaN that only pooling creates would
                    # otherwise surface as `Input X contains NaN` from inside
                    # RidgeCV.predict, with nothing naming the column
                    Xtr, Xte, want, btr, imp = impute_from_train(
                        Xtr, Xte, want, btr, log=lambda *_: None)
                    if not len(want):
                        skipped[f"{arm}: every column was NaN across the "
                                f"training cohort"] += 1
                        continue
                    cov = [i for i, b in enumerate(btr) if b == "covariates"]

                    for model in models:
                        if model not in base:
                            bp = fit_predict(Xtr[:, cov], ytr, Xte[:, cov],
                                             model, seed, threads)
                            base[model] = bootstrap_ci(yte, bp, n_boot, seed)
                        bsc, blo, bhi = base[model]
                        pred = fit_predict(Xtr, ytr, Xte, model, seed, threads)
                        sc, lo, hi = bootstrap_ci(yte, pred, n_boot, seed)
                        # a model that predicts ONE value for everyone has
                        # ranked nobody: Spearman is undefined and the score is
                        # NaN. Kept as a row, because it is an outcome, and
                        # flagged, because otherwise the NaN looks like a bug
                        const = len(np.unique(pred)) < 2
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
                            "n_features": Xtr.shape[1],
                            "pred_constant": bool(const), **imp})
        # One line per state set, so a log says where a run got to. The first
        # cluster array left nothing but the job script's own echo: stdout was
        # block-buffered and the task was killed by a signal, so there was no
        # way to tell an import crash from a crash 200 fits in.
        # n_train/n_test of this cell, because the FITTED n is the number that
        # limits every claim and the phenotype file's row count is not it:
        # HCP's table carries all 1,206 S1200 subjects and only the 7T ones
        # have transitions.
        last = rows[-1] if rows else {}
        log(f"  [{i}/{len(sets)}] {s.states} {s.atlas} "
            f"{fmt_window(s.window_s)}s  K={n_states(s.states)}  "
            f"n={last.get('n_train', 0)}->{last.get('n_test', 0)}  "
            f"{len(rows)} row(s), {time.time() - t0:.0f}s")
    for why, k in skipped.most_common():
        log(f"  skipped {k}x: {why}")
    if not rows:
        raise SystemExit(
            "no configuration could be scored. "
            + (f"Every one of {sum(skipped.values())} attempt(s) was skipped; "
               f"the commonest reason was: {skipped.most_common(1)[0][0]}"
               if skipped else
               f"{len(sets)} set(s) x {len(tasks)} task(s) x {len(labels)} "
               f"target(s) x {len(arms)} arm(s) and not one was reached, which "
               f"should be impossible -- check `arms`."))
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

        # np.full, not a bare np.nan: zip() over a float raises TypeError, so a
        # subject whose pooled runs hold no transition at all would have
        # crashed the pooling instead of coming out as a missing row.
        row = dict(zip(cells, counts / tot_tr if tot_tr
                       else np.full(len(cells), np.nan)))
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

    need = ["score", "lo", "hi", "base_score", "base_lo", "base_hi",
            "states", "window_s", "tree", "arm", "model", "n_test"]
    missing = [c for c in need if c not in rows.columns]
    if missing:
        raise SystemExit(
            f"as_report_rows got a frame with no {', '.join(missing)} column"
            + (" -- it is EMPTY (no rows, no columns), so nothing upstream "
               "produced a score. Read the refusal or the skip counts from "
               "the function that built it; this one cannot know why."
               if not len(rows.columns) else
               f". It has: {', '.join(map(str, rows.columns[:12]))}"))

    d = rows.copy()
    d["mean"] = d["score"]
    d["std"] = (d["hi"] - d["lo"]) / _CI_TO_SD
    d["base_mean"] = d["base_score"]
    d["base_sd"] = (d["base_hi"] - d["base_lo"]) / _CI_TO_SD
    # B1 prints the baseline's n. In the report that is the number of subjects
    # the CV ran over; here it is the number of TEST subjects the score was
    # computed on, which is the n that limits this claim.
    d["n"] = d["n_test"]
    d["base_n"] = d["n_test"]
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
        raise SystemExit(
            f"no static-FC rows for {list(cohorts)} atlas={atlas} "
            f"task~{task_value}"
            + (f"; the tasks present are {sorted(set(pd.concat(frames)['task'].astype(str)))[:8]}"
               if len(frames) and len(pd.concat(frames)) else " (the table is empty)"))
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


def censor_policies(root, cohorts) -> dict:
    """{cohort: the censor policy its STATE tables were built with}, in one pass.

    One pass because `transitions/` holds a table per (atlas, window, states,
    cohort, task) -- thousands of files over a 282-set grid -- and scanning it
    once per cohort would read the same footers five times.

    Used for two things: printing a command that RUNS (rather than one with a
    placeholder in angle brackets, which bash reads as a redirection), and
    checking that the static-FC tables were built from the same frames.
    """
    from pathlib import Path

    from fmri_decomposition.status import _meta

    want, out = set(cohorts), {}
    for stem in ("transitions", "latents"):
        for q in (Path(root) / stem).rglob("*.parquet"):
            c = hive(q, "cohort")
            if c not in want or c in out:
                continue
            v = _meta(q, "censor_policy")
            if v is not None:
                out[c] = str(v)
            if want <= set(out):
                return out
    return out


def censor_policy_of(root, cohort: str) -> str | None:
    """One cohort's state-table censor policy. See `censor_policies`."""
    return censor_policies(root, [cohort]).get(cohort)


# What static-fc records about the FRAMES it correlated. The two cohorts must
# agree on all of it, and it must agree with what the state trees were built
# from -- otherwise the bstm-vs-fcm comparison is a preprocessing comparison
# wearing a model's name. slurm/static_fc.sbatch's header says exactly this,
# and until now nothing enforced it.
FRAME_RECIPE = ("censor_policy", "match_bandpass", "zscore_runs", "pool")


def frame_recipe(d: "pd.DataFrame", where: str) -> dict:
    """What one static-FC table says it was built from, or a refusal.

    `static_fc.write_cohort` stores its provenance as COLUMNS, so this is a
    read of the table already in memory -- no extra I/O.
    """
    out = {}
    for c in FRAME_RECIPE:
        if c not in d:
            continue
        vals = sorted({"" if pd.isna(v) else v for v in d[c]}, key=str)
        if len(vals) > 1:
            raise SystemExit(
                f"{where}: the static-FC rows disagree on {c} ({vals}). Two "
                f"cohorts of one side were built by different runs of stage "
                f"3b; rebuild them together, or the side is not one sample.")
        out[c] = vals[0] if vals else None
    return out


def fcm_transfer_rows(train: Study, test: Study, atlases=None,
                      train_tasks=None, test_tasks=None,
                      models=None, covariates=None, categorical=("Sex",),
                      max_missing: float = .10, seed: int = 0,
                      n_boot: int = 2000, pool_test: bool = True,
                      threads: int = 1, log=print) -> pd.DataFrame:
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
    # same reason as transfer_rows: not train.models(), which is empty for a
    # target the selections never ran on
    # None means "decide for me"; an empty list is a request for nothing and
    # is refused below rather than quietly turned into the default.
    models = list(DEFAULT_MODELS if models is None and train.cfg.models is None
                  else train.cfg.models if models is None else models)
    labels = [lab for lab in train.cfg.targets if lab in test.cfg.targets]
    if not labels or not models:
        raise SystemExit(
            f"nothing to fit: target labels in common = {labels}, models = "
            f"{models}. The LABELS must match between the two Settings.")
    # WHICH COHORTS ARE IN THE static_fc TREE AT ALL. Checked up front, because
    # the failure without it is silent in the worst way: every (atlas, task)
    # cell skips, `fcm_transfer_rows` returns an empty frame, and whatever
    # reads it next dies on a missing column with nothing naming the cause.
    # Static FC is STAGE 3b, a separate job -- `project_and_select.sh` does not
    # run it, so a cohort projected onto an existing fit has transitions and no
    # static FC until stage 3b is run for it too.
    present = {}
    for q in (root / "static_fc").rglob("subjects.parquet"):
        present.setdefault(hive(q, "atlas"), set()).add(hive(q, "cohort"))
    atlases = atlases or sorted(present)
    if not present:
        raise SystemExit(
            f"no static-FC table under {root / 'static_fc'} at all. It is stage "
            f"3b and nothing upstream runs it:\n"
            f"  sbatch slurm/static_fc.sbatch {' '.join(train.cfg.cohorts)} "
            f"{' '.join(test.cfg.cohorts)} -- --atlas <atlas ...>")
    # EVERY cohort of a side, not any of them. `_fc_cell` reads one table per
    # cohort and concatenates before filtering by task, so one missing cohort
    # kills every cell of that side -- which is how a job skipped all six
    # (atlas, task) cells over a single absent `camcan_rest`, including the
    # movie cells that camcan alone could have served.
    for side, cfg in (("train", train.cfg), ("test", test.cfg)):
        gaps = {a: [c for c in cfg.cohorts if c not in present.get(a, set())]
                for a in atlases}
        if all(gaps.values()):                    # no atlas has the full side
            short = sorted({c for v in gaps.values() for c in v})
            pol = next((censor_policy_of(root, c) for c in cfg.cohorts
                        if censor_policy_of(root, c)), None)
            if len(short) < len(cfg.cohorts):
                # SOME of the side is there: a partial side, which reads as
                # "static FC has run" everywhere except where it matters
                raise SystemExit(
                    f"the {side} side is INCOMPLETE: {short} ha"
                    + ("s" if len(short) == 1 else "ve")
                    + f" no static-FC table, while {[c for c in cfg.cohorts if c not in short]} "
                      f"do(es).\n  `_fc_cell` reads one table per cohort and "
                      f"concatenates them before filtering by task, so a "
                      f"missing cohort skips EVERY cell of this side -- the "
                      f"movie cells too, not just the rest ones.\n"
                      f"  Run stage 3b for it with the SAME settings as the "
                      f"rest of this side (see `--status` for what they are):\n"
                    + f"    sbatch slurm/static_fc.sbatch {' '.join(short)} -- "
                      f"--atlas {' '.join(sorted(present)) or 'ATLAS'}"
                    + (f" --censor-policy {pol}" if pol else "")
                    + "\n")
        covered = [a for a in atlases if not gaps[a]]
        if not covered:
            have = sorted({c for v in present.values() for c in v})
            # the policy its OWN tables carry, so the command below runs as
            # printed rather than carrying a placeholder
            pol = next((censor_policy_of(root, c) for c in cfg.cohorts
                        if censor_policy_of(root, c)), None)
            # a cohort whose name is a prefix of, or starts with, one of ours:
            # hcp7t beside hcp7t_fix is the case this exists for
            near = [c for c in have if c not in cfg.cohorts and any(
                c.startswith(x) or x.startswith(c) for x in cfg.cohorts)]
            raise SystemExit(
                f"the {side} cohort{'s' if len(cfg.cohorts) > 1 else ''} "
                f"{list(cfg.cohorts)} have NO static-FC table for any atlas.\n"
                f"  the tree has: "
                + "; ".join(f"atlas={a}: {sorted(v)}" for a, v in sorted(present.items()))
                + f"\n  Static FC is STAGE 3b and is a separate job -- "
                  f"project_and_select.sh does not run it, so a cohort "
                  f"projected onto an existing fit has transitions and no "
                  f"static FC. Run it with the SAME frame settings as the state "
                  f"trees (the header of slurm/static_fc.sbatch says why), and "
                  f"the same --censor-policy `decompose` was given:\n"
                  f"    sbatch slurm/static_fc.sbatch {' '.join(cfg.cohorts)} -- "
                  f"--atlas {' '.join(sorted(present)) or 'ATLAS'} "
                  f"--censor-policy "
                  + (pol or "POLICY  # not recorded; the one decompose was given")
                  + "\n"
                  f"  Keep the default --pool task: it writes one row per "
                  f"(task, sub), which is what pools MOVIE1-4 into one here. "
                  f"`--pool subject` would pool movie WITH rest."
                  f"\n  Only cohorts {have} have one now."
                + (f"\n  NOTE: {near} is NOT a substitute for "
                   f"{list(cfg.cohorts)}. A differently named cohort is a "
                   f"different cohort here, and that is the whole reason the "
                   f"label exists: two preprocessing depths under one name "
                   f"would put a preprocessing difference inside something "
                   f"the analysis treats as one thing. Its subjects and its "
                   f"cleaning are not the ones the transitions were built "
                   f"from, so an E1 panel mixing them would not be two "
                   f"models on one sample." if near else ""))
    # read once: transitions/ holds thousands of files over a 282-set grid
    state_pol = censor_policies(
        root, list(train.cfg.cohorts) + list(test.cfg.cohorts))
    tr_tasks = train_tasks or {k: v for k, v in train.cfg.tasks.items()}
    te_tasks = test_tasks or {k: v for k, v in test.cfg.tasks.items()}
    # an atlas is usable only where BOTH sides are complete
    usable = [a for a in atlases
              if all(c in present.get(a, set())
                     for c in list(train.cfg.cohorts) + list(test.cfg.cohorts))]
    if usable != list(atlases):
        log(f"  using {len(usable)} of {len(atlases)} atlas(es): {usable}"
            f" (the rest are missing a cohort on one side)")
    rows, side_recipe = [], {}
    for atlas in usable:
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
            # THE FRAMES, before the edges. The whole point of E1 is two
            # MODELS on one sample; if the two tables were correlated over
            # different frames -- a different censor policy, a different
            # band -- the panel compares preprocessing and calls it a model.
            rtr = frame_recipe(dtr, f"train {list(train.cfg.cohorts)} atlas={atlas}")
            rte = frame_recipe(dte, f"test {list(test.cfg.cohorts)} atlas={atlas}")
            differ = {k: (rtr.get(k), rte.get(k)) for k in FRAME_RECIPE
                      if k in rtr and k in rte and rtr[k] != rte[k]}
            if differ:
                raise SystemExit(
                    f"fcm {atlas} {task}: the two cohorts' static-FC tables "
                    f"were built from DIFFERENT frames, so a difference "
                    f"between them is not a difference between cohorts:\n"
                    + "\n".join(f"    {k}: train {a!r}  test {b!r}"
                                 for k, (a, b) in differ.items())
                    + "\n  Re-run stage 3b for the side that is wrong with "
                      "the other's settings. slurm/static_fc.sbatch's header "
                      "says why they have to match.")
            # ... and against the STATE trees, which is what makes bstm and
            # fcm comparable at all.
            for side, cfg, rec in (("train", train.cfg, rtr),
                                   ("test", test.cfg, rte)):
                want = {state_pol.get(c) for c in cfg.cohorts
                        if state_pol.get(c) is not None}
                got = rec.get("censor_policy") or None
                if want and got not in want:
                    raise SystemExit(
                        f"fcm {atlas} {task}: the {side} cohort's static FC "
                        f"was built with censor_policy={got!r} while its "
                        f"transition tables carry {sorted(want)}. The two "
                        f"trees would then differ in PREPROCESSING as well as "
                        f"in model, and E1 could not tell the two apart.\n"
                        f"  Re-run stage 3b with --censor-policy "
                        f"{sorted(want)[0]}, or say in the write-up that the "
                        f"fcm arm is on different frames.")
            # ONE recipe per side across tasks, too. The train side's movie
            # cells come from camcan and its rest cells from camcan_rest, and
            # those are two separate runs of stage 3b -- so rest-vs-movie
            # could be a preprocessing difference wearing a condition's name.
            for side, rec in (("train", rtr), ("test", rte)):
                prev = side_recipe.setdefault((side, atlas), (task, rec))
                if prev[1] != rec:
                    raise SystemExit(
                        f"fcm {atlas}: the {side} side's {prev[0]} and {task} "
                        f"tables were built from different frames, so a "
                        f"difference between the two CONDITIONS would partly "
                        f"be a difference in preprocessing:\n"
                        + "\n".join(f"    {k}: {prev[0]} {prev[1].get(k)!r}  "
                                     f"{task} {rec.get(k)!r}"
                                     for k in rec if prev[1].get(k) != rec.get(k))
                        + f"\n  Those cells come from different cohorts "
                          f"({list(train.cfg.cohorts) if side == 'train' else list(test.cfg.cohorts)}) "
                          f"and therefore from different runs of stage 3b. "
                          f"Re-run the odd one with the other's settings.")

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
                                             Cte_[ok_te], model, seed, threads)
                            base[model] = bootstrap_ci(yte[ok_te], bp, n_boot, seed)
                        bsc, blo, bhi = base[model]
                        pred = fit_predict(Xtr, ytr[ok_tr], Xte, model, seed, threads)
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


def drop_degenerate(rows: "pd.DataFrame", log=print) -> "pd.DataFrame":
    """The rows a figure can draw, with the ones it cannot COUNTED, not hidden.

    A score is NaN when the model predicted one value for every test subject:
    Spearman has nothing to rank. That is an outcome and it stays on disk, but
    B1 takes `idxmax` per group and dies on an all-NaN group ("Encountered all
    NA values"), so it is filtered here -- out loud, with the models it
    happened to, because "hgb predicted a constant for every configuration" is
    a finding about hgb, not a plotting detail.
    """
    bad = rows["score"].isna() | rows["base_score"].isna()
    if bad.any():
        by = rows[bad].groupby(["model", "arm"], dropna=False).size()
        log(f"  {int(bad.sum())} of {len(rows)} row(s) have NO score: the model "
            f"predicted one value for every test subject, so Spearman has "
            f"nothing to rank. By (model, arm): "
            + "; ".join(f"{m}/{a} x{n}" for (m, a), n in by.items()))
    return rows[~bad].reset_index(drop=True)


# ------------------------------------------------------------------- on disk ---
# The grid is ~9,000 fits on every training subject, which is hours in a
# notebook kernel and minutes as a SLURM array. So the fits happen in
# tools/transfer_grid.py, one shard per array task, and the notebook READS the
# rows and draws. The layout:
#
#   <root>/train=camcan/test=hcp7t_fix/targets=raw/tree=bstm/shard=07/rows.parquet
#
# `targets` is which Cam-CAN column set was used (raw | category), because
# those are two different analyses and must never be concatenated by accident.
GRID_KEYS = ["atlas", "window_s", "states"]


def grid_hash(sets: "pd.DataFrame") -> str:
    """A fingerprint of the WHOLE grid each shard was cut from.

    Shards are `sets.iloc[i::n]` of a sorted list, so they only fit together
    if every task saw the same list. Stage 5a finishing another cohort between
    two array tasks would silently renumber the sets, and the shards would
    then cover an overlapping, gap-ridden subset of two different grids. The
    reader compares these and refuses.
    """
    import hashlib
    d = sets[GRID_KEYS].astype(str).sort_values(GRID_KEYS)
    return hashlib.md5(d.to_csv(index=False).encode()).hexdigest()[:16]


def shard_of(sets: "pd.DataFrame", shard: int, n_shards: int) -> "pd.DataFrame":
    """Shard `shard` of `n_shards`, from the grid sorted deterministically."""
    if not (0 <= shard < n_shards):
        raise SystemExit(f"shard must be in [0, {n_shards}), not {shard}")
    d = sets.sort_values(GRID_KEYS).reset_index(drop=True)
    return d.iloc[shard::n_shards].reset_index(drop=True)


def transfer_dir(root, train: str, test: str, target_set: str, tree: str):
    from pathlib import Path
    return (Path(root) / f"train={train}" / f"test={test}"
            / f"targets={target_set}" / f"tree={tree}")


def write_transferred(rows: "pd.DataFrame", root, *, train: str, test: str,
                      target_set: str, tree: str, shard: int, n_shards: int,
                      **provenance) -> "Path":
    """One shard's rows, with what produced them written INTO the frame.

    Provenance as columns rather than a sidecar file: a frame that travels
    without it is a frame somebody will plot next year not knowing which
    target set, which seed or how many bootstrap draws it came from.
    """
    import pyarrow as pa
    from fmri_decomposition.io import write_table_atomic

    if not len(rows):
        raise SystemExit(
            f"refusing to write an EMPTY {tree} shard. A file with no rows "
            f"reads as a finished shard downstream -- the status report counts "
            f"it as present and `.iloc[0]` on its provenance raises an "
            f"IndexError seven frames away, which is exactly what it did.\n"
            f"  Nothing was scored, so read why: the skip lines above name "
            f"every (atlas, task) that was dropped and the reason. For fcm "
            f"that is usually no static_fc table for one of the two cohorts.")

    d = rows.copy()
    for k, v in {"train_cohorts": train, "test_cohorts": test,
                 "target_set": target_set, "tree": tree, "shard": int(shard),
                 "n_shards": int(n_shards), **provenance}.items():
        d[k] = v
    path = (transfer_dir(root, train, test, target_set, tree)
            / f"shard={int(shard):02d}" / "rows.parquet")
    return write_table_atomic(pa.Table.from_pandas(d, preserve_index=False), path)


# What a shard was computed WITH. Two shards of one analysis must agree on all
# of it; if they do not, concatenating them is concatenating two analyses.
# `grid_hash` is NOT here: it has its own check, because it means something
# different. A changed grid renumbers the sets, so the shards overlap and leave
# gaps and the whole tree has to go; shards that merely disagree on n_boot or
# arms are one odd shard to delete.
RECIPE = ["n_shards", "n_boot", "seed"]


def recipes(d: "pd.DataFrame") -> dict:
    """{shard: (n_shards, n_boot, seed, arms, models)} per shard.

    Only the first three are grounds to REFUSE. Shards differing in their arms
    or models is suspicious and reported, but it is not proof of contamination:
    a shard whose only state set lost an arm -- every column of it NaN across
    the training cohort -- is legitimately short of one, and refusing there
    would block a grid that is fine.
    """
    out = {}
    for sh, h in d.groupby("shard", sort=True):
        out[int(sh)] = tuple(
            [h[c].iloc[0] for c in RECIPE if c in h]
            + [tuple(sorted(h["arm"].unique())), tuple(sorted(h["model"].unique()))])
    return out


def disagreements(rec: dict) -> tuple[dict, dict]:
    """(hard, soft): shards grouped by the part that refuses, and by arms/models."""
    hard, soft = {}, {}
    for sh, r in rec.items():
        hard.setdefault(r[:len(RECIPE)], []).append(sh)
        soft.setdefault(r[len(RECIPE):], []).append(sh)
    return hard, soft


def inventory(root, target_set: str, train: str, test: str, tree: str) -> dict:
    """What is on disk for one tree: files, shards, gaps, recipes, rows.

    Shared by `read_transferred` and the status report so the two cannot
    disagree about whether a grid is finished.
    """
    base = transfer_dir(root, train, test, target_set, tree)
    files = sorted(base.glob("shard=*/rows.parquet"))
    out = dict(base=base, files=files, rows=None, n_shards=None, have=[],
               gaps=[], recipes={}, empty=[])
    if not files:
        return out
    frames, empty = [], []
    for f in files:
        h = pd.read_parquet(f)
        (empty if not len(h) else frames).append(f if not len(h) else h)
    out["empty"] = empty
    if not frames:
        return out
    d = pd.concat(frames, ignore_index=True)
    out["rows"] = d
    out["recipes"] = recipes(d) if "shard" in d else {}
    out["have"] = sorted(out["recipes"]) or list(range(len(frames)))
    out["n_shards"] = (max(int(v[0]) for v in out["recipes"].values())
                       if out["recipes"] and "n_shards" in d else len(frames))
    out["gaps"] = [i for i in range(out["n_shards"]) if i not in set(out["have"])]
    return out


def read_transferred(root, target_set: str, train: str = "camcan",
                     test: str = "hcp7t_fix", trees=("bstm", "fcm"),
                     require_complete: bool = True, log=print) -> "pd.DataFrame":
    """Every shard of one analysis, concatenated, or a refusal saying what is missing.

    THE POINT OF THE CHECKS. A half-finished array reads as a perfectly valid
    frame -- just a smaller multiverse -- and "the best of 282 state sets" from
    the 120 that happened to have finished is a different claim with the same
    words. So: every shard a frame says exists must be present, and every
    shard must have been cut from the same grid.

    `require_complete=False` is for watching a run in progress, and says so in
    the log rather than in the figure.
    """
    from pathlib import Path

    frames, missing = [], []
    for tree in trees:
        inv = inventory(root, target_set, train, test, tree)
        if inv["empty"]:
            raise SystemExit(
                f"tree={tree}: {len(inv['empty'])} shard file(s) hold NO rows: "
                + ", ".join(str(p) for p in inv["empty"][:4])
                + f"\n  Nothing was scored there, and an empty file reads as a "
                  f"finished shard. Read that job's log for the skip reasons, "
                  f"delete the file(s) and resubmit.")
        if inv["rows"] is None:
            missing.append(f"tree={tree}: nothing under {inv['base']}")
            continue
        d, rec = inv["rows"], inv["recipes"]
        hashes = sorted(set(d["grid_hash"])) if "grid_hash" in d else []
        if len(hashes) > 1:
            raise SystemExit(
                f"tree={tree}: the shards were cut from {len(hashes)} different "
                f"grids ({hashes}). A shard is sets[i::n] of a sorted list, so "
                f"the state sets on disk changing mid-array renumbers them: "
                f"these shards overlap AND leave gaps, and no subset of them is "
                f"a grid. Delete {inv['base']} and resubmit the whole tree.")
        # DO THE SHARDS AGREE ON WHAT THEY ARE? A hand-run shard -- one tried
        # interactively with --arms cells --models ridge --n-boot 50, say --
        # lands in the same tree as the array's and reads as part of it. The
        # grid hash cannot see it: the GRID was the same, the ANALYSIS was not.
        hard, soft = disagreements(rec)
        if len(hard) > 1:
            lines = [f"    shard(s) {v}: n_shards={k[0]}, n_boot={k[1]}, "
                     f"seed={k[2]}" for k, v in hard.items()]
            raise SystemExit(
                f"tree={tree}: the shards on disk were NOT computed the same "
                f"way, so concatenating them would concatenate two analyses:\n"
                + "\n".join(lines)
                + f"\n  Delete the odd one out of {inv['base']} and resubmit "
                  f"that shard. A shard tried by hand -- with --n-shards or "
                  f"--n-boot of its own, to see the chain work -- is the usual "
                  f"cause; `--no-write` is how to do that without writing one.")
        if len(soft) > 1:
            # Reported, not refused: see `recipes`. Either a hand-run shard
            # with --arms/--models, or cells that were skipped.
            log(f"  tree={tree}: NOTE -- the shards do not all carry the same "
                f"arms and models. "
                + "; ".join(f"shard(s) {v}: arms={list(k[0])}, models={list(k[1])}"
                            for k, v in soft.items())
                + ". Either a shard was run by hand with fewer of them, or "
                  "those cells were skipped -- the per-model counts in "
                  "`transfer_grid.py --status` say which.")
        n, have, gaps = inv["n_shards"], inv["have"], inv["gaps"]
        log(f"  tree={tree}: {len(d):,} row(s) from {len(have)}/{n} shard(s)"
            + (f"  MISSING {gaps}" if gaps else ""))
        if gaps:
            missing.append(f"tree={tree}: shard(s) {gaps} of {n} are not on disk")
        frames.append(d)
    if not frames:
        raise SystemExit(
            f"no rows under {Path(root)} for targets={target_set}, tree(s) "
            f"{list(trees)}: has tools/transfer_grid.py run?\n  "
            + "\n  ".join(missing))
    if missing and require_complete:
        raise SystemExit(
            "the transfer grid is incomplete:\n  " + "\n  ".join(missing)
            + "\n  A smaller multiverse reads as a valid one, and 'the best of "
              "282 state sets' from the shards that happened to finish is a "
              "different claim in the same words. Check the array's logs, "
              "resubmit the missing shard(s), or pass "
              "require_complete=False to look at a run in progress.")
    return pd.concat(frames, ignore_index=True)
