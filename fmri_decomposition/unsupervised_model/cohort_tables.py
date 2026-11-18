"""Tables behind 11_normal_brain_cohorts.ipynb: the normal brain's state dynamics across cohorts.

The question is what the cohorts SHARE, not how they differ. `Cohorts(cfg)` reads one state set's
`transitions/` tables for every cohort, splits each cohort's scans into rest and movie, and averages
a subject's runs or films into one row. Every method returns one table:

    graphs()          per (cohort, rest|movie) and their mean: p(j | i) and time in state
    task_change()     movie vs rest in the same people, per cohort: what passes, and the dynamics
    agreement(vars)   each cohort vs all the others, TOST: is it equivalent within +-SESOI?
    cohort_means()    every variable's mean per cohort, for train vs projected

Equivalence, not "no difference": a cohort is equivalent to the others on a variable when the 90%
CI of the difference lies inside +-SESOI (two one-sided tests at 5%, Lakens 2017). SESOI is
`sesoi_d` x the pooled SD of that variable: 0.2 = Cohen's "small effect", a convention to justify.
A small cohort has a wide CI and so often cannot be shown equivalent: that is not a difference.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fmri_decomposition.io import latents_root, transitions_root, tree_root, unsupervised_root, _fmt_window, _key
from fmri_decomposition.supervised_model import hads as ht

DYNAMICS = ["switch_rate", "mean_dwell_s", "entropy_rate_bits", "dispersion", "n_distinct_transitions"]


@dataclass
class Settings:
    root: Path = Path(os.environ.get("FMRIDECOMP_OUTPUTS", "/project/6008063/tamires/DecomposingfMRI/outputs"))
    # Which modelling version's latents and transitions to read. None = the
    # resolved default ($FMRIDECOMP_MODELLING_VERSION, else v0).
    modelling_version: str | None = None
    # one state set, as in notebook 10's SPACE
    space: dict = field(default_factory=lambda: dict(atlas="yeo7", window_s=30.0, states="HMM1_pca3_8"))
    # label -> the cohort= directories it reads, and its role in the state fit (train / projected)
    cohorts: dict = field(default_factory=lambda: {
        "ds002837": dict(dirs=["ds002837"], role="train"),
        "cneuromod": dict(dirs=["cneuromod"], role="train"),
        "camcan": dict(dirs=["camcan"], role="projected")})
    rest_pattern: str = r"rest\d*"                    # a task= value that IS this (whole name, any case) is rest;
                                                      # every other task is movie ("theprestige" is a film)
    dynamics: list = field(default_factory=lambda: list(DYNAMICS))
    # figure 1: movie vs rest
    alpha: float = 0.10                               # BH-FDR
    fc_task: float = 1.0                              # |log2FC| for the graph (1 = twofold)
    fc_dyn: float = 0.25                              # |log2FC| for the dynamics
    shrink: float = .015                              # log2FC pseudocount for probabilities
    edge_top: int = 12                                # the mean's strongest arrows, per task
    # figures 2 and 4: equivalence
    sesoi_d: float = 0.2                              # TOST bounds, in pooled SDs
    top_transitions: int = 8                          # figure 2: the strongest transitions shown
    # fixed x ranges (figures.cohorts.fixed_range): probabilities 0-1; entropy 0-log2(K) bits and
    # distinct 0-K*K are set from K; dispersion from the data (its units depend on the latent space)
    ranges: dict = field(default_factory=lambda: {"switch_rate": (0, 1), "mean_dwell_s": (0, 60)})
    save_dir: Path | None = None
    # None = entropy_rate_bits as stored; "plugin" / "miller_madow" =
    # recomputed from each table's counts (see hads.Settings.entropy_estimator).
    entropy_estimator: str | None = None
    # film labels merged in the per-film figures: (regex, replacement). Default: the two halves of
    # a Friends episode (s01e01a, s01e01b) are one film, s01e01.
    film_merge: tuple = (r"^(s\d+e\d+)[a-z]$", r"\1")
    # n_distinct_transitions grows with scan length. None = as stored; minutes (float) or "min" =
    # rarefied: the number of distinct transitions EXPECTED in that many minutes of the scan
    #, from its transition counts; "min" = the shortest scan read, so none is lost.
    # Scans shorter than the target are NaN.
    distinct_minutes: float | str | None = None
    # a person's p(j | i) comes from their counts summed over scans; row i is NaN (not 0) when they
    # leave state i fewer times than this in ALL their scans, rest and movie together (and NaN in a
    # condition where they never leave it): never visiting i says nothing about where i leads.
    min_out_transitions: int = 5

    @property
    def K(self):
        return ht.n_states(self.space["states"])


def _combine(est, var, how="random"):
    """Cohorts' estimates (cohorts x variables) and their variances -> (combined, SE).
    "equal": plain mean; "fixed": inverse-variance; "random": DerSimonian-Laird random effects."""
    ok = np.isfinite(est) & np.isfinite(var)
    k = ok.sum(0)
    if how == "equal":
        m = np.nanmean(np.where(ok, est, np.nan), 0)
        return m, np.sqrt(np.nansum(np.where(ok, var, 0), 0)) / np.maximum(k, 1)
    v = np.where(ok, np.maximum(var, 1e-10), np.inf)
    w = np.where(ok, 1 / v, 0)
    e = np.where(ok, est, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fixed = (w * e).sum(0) / w.sum(0)
        if how == "fixed":
            return np.where(k > 0, fixed, np.nan), np.where(k > 0, np.sqrt(1 / w.sum(0)), np.nan)
        Q = (w * (e - fixed) ** 2).sum(0)
        tau2 = np.maximum(0, (Q - (k - 1)) / (w.sum(0) - (w ** 2).sum(0) / w.sum(0)))
        tau2 = np.where(k > 1, tau2, 0)
        ws = np.where(ok, 1 / (v + tau2), 0)
        m = (ws * e).sum(0) / ws.sum(0)
        return np.where(k > 0, m, np.nan), np.where(k > 0, np.sqrt(1 / ws.sum(0)), np.nan)


TIER_CUTS = (0.2, 0.5)                                   # |d|: stable < .2 <= moderate < .5 <= context-dependent


def _tier_index(a):
    """0 stable, 1 moderate, 2 context-dependent (Cohen's small / medium cuts on |d|)."""
    return np.digitize(a, TIER_CUTS)


def _scan_minutes(d: pd.DataFrame) -> pd.Series:
    if not {"n_windows", "stride_s"} <= set(d):
        raise SystemExit("distinct_minutes needs n_windows and stride_s in the transition tables")
    return (d["n_windows"] * d["stride_s"]).astype(float) / 60


def expected_distinct(counts, n: int) -> float:
    """Rarefaction: the expected number of distinct categories in n draws without
    replacement from `counts`, sum_k 1 - C(N - c_k, n) / C(N, n). NaN when n > N."""
    from scipy.special import gammaln
    c = np.asarray(counts, float).ravel()
    c = c[c > 0]
    N = c.sum()
    if n > N or n <= 0:
        return np.nan
    rest = N - c
    ok = rest >= n                                       # else the category is certainly drawn
    p_absent = np.zeros_like(c)
    p_absent[ok] = np.exp(gammaln(rest[ok] + 1) - gammaln(rest[ok] - n + 1)
                          - gammaln(N + 1) + gammaln(N - n + 1))
    return float((1 - p_absent).sum())


def rarefied_distinct(d: pd.DataFrame, n_states: int, minutes: float) -> pd.Series:
    """n_distinct_transitions per row, rarefied to the transitions of `minutes` of that scan
    (its own transitions per minute, so the window stride is respected). Cells must be p_joint."""
    cells = ht.cell_cols(d)
    C = np.rint(d[cells].to_numpy(float) * d["n_transitions"].to_numpy(float)[:, None])
    rate = d["n_transitions"].to_numpy(float) / _scan_minutes(d).to_numpy()
    n = np.rint(rate * minutes)
    out = [expected_distinct(C[i], int(n[i])) if np.isfinite(n[i]) else np.nan for i in range(len(d))]
    return pd.Series(out, index=d.index, name="n_distinct_transitions")


# -- stability: what is stable, what depends on context (figure 3 of the manuscript) ---------
def _icc_c1(x, y):
    """ICC(3,1), two-way mixed, consistency: do the people keep their rank between x and y,
    whatever the shift of the mean (Shrout & Fleiss 1979; McGraw & Wong 1996)."""
    Y = np.c_[x, y]
    Y = Y[np.isfinite(Y).all(1)]
    n, k = Y.shape
    if n < 3:
        return np.nan
    gm = Y.mean()
    ss_r = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_c = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_e = ((Y - gm) ** 2).sum() - ss_r - ss_c
    ms_r, ms_e = ss_r / (n - 1), ss_e / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e) if (ms_r + (k - 1) * ms_e) > 0 else np.nan


def _paired(r, m):
    """Movie - rest in the same people: d_av (mean difference over the mean of the two SDs,
    Lakens 2013), so it is on the same scale as the between-cohort d."""
    diff = m - r
    sd = (r.std(ddof=1) + m.std(ddof=1)) / 2
    return diff.mean() / sd if sd > 0 else 0.0


class StabilityMixin:
    """Methods of `Cohorts` behind the stability panels; split out only for reading."""

    def _ref_tables(self):
        """The reference: every cohort once, each person over all their scans (any task)."""
        return list(getattr(self, "pooled", {}).values()) or list(self.data.values())

    def transition_fit(self, min_occ=0.01):
        """Per subject and scan kind: Pearson r between the subject's p(j | i) and the reference
        p(j | i) (`_ref_tables`: every cohort once, all its scans), on the rows of the
        states the reference occupies >= min_occ. One number for 'does this person move between
        states the way people do' -- the transition structure as a per-subject variable."""
        K = self.cfg.K
        ref = pd.concat([v[self.cells].mean() for v in self._ref_tables()], axis=1).mean(axis=1)
        occ = pd.concat([v[self.occ].mean() for v in self._ref_tables()], axis=1).mean(axis=1).to_numpy()
        rows = np.repeat(occ >= min_occ, K)
        R = ref.to_numpy(float)[rows]
        for key, d in [*self.data.items(), *getattr(self, "by_task", {}).items(), *getattr(self, "pooled", {}).items()]:
            X = d[self.cells].to_numpy(float)[:, rows]
            ok = np.isfinite(X) & np.isfinite(R)                  # each person's own observed rows
            fit = np.full(len(X), np.nan)
            for n_, (x, m) in enumerate(zip(X, ok)):
                if m.sum() >= 3:
                    xc, rc = x[m] - x[m].mean(), R[m] - R[m].mean()
                    den = np.sqrt((xc ** 2).sum() * (rc ** 2).sum())
                    fit[n_] = (xc @ rc) / den if den > 0 else np.nan
            d["transition_fit"] = fit
        return "transition_fit"

    def row_coverage(self):
        """Per (cohort, rest|movie) and state i: people, how many have p(j | i) (left i at least
        min_out_transitions times over their scans), and the mean occupancy of i. Why a transition
        row can be missing from the movie-vs-rest comparison."""
        rows = []
        for (c, k), d in self.data.items():
            for i in range(self.cfg.K):
                rows.append(dict(cohort=c, kind=k, state=i, people=len(d),
                                 with_row=int(d[f"{i}->{i}"].notna().sum()), occupancy=d[f"occ_{i}"].mean()))
        return pd.DataFrame(rows)

    def occupancy_fit(self):
        """Per subject and scan kind: Pearson r between the subject's occupancy (share of time in
        each state) and the reference occupancy (`_ref_tables`: every cohort once, all its
        scans). The occupancy profile as one per-subject variable, like `transition_fit`."""
        R = pd.concat([v[self.occ].mean() for v in self._ref_tables()], axis=1).mean(axis=1).to_numpy(float)
        Rc = R - R.mean()
        for key, d in [*self.data.items(), *getattr(self, "by_task", {}).items(), *getattr(self, "pooled", {}).items()]:
            X = d[self.occ].to_numpy(float)
            Xc = X - X.mean(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                d["occupancy_fit"] = (Xc @ Rc) / np.sqrt((Xc ** 2).sum(1) * (Rc ** 2).sum())
        return "occupancy_fit"

    def dynamics_fit(self):
        """Per subject and scan kind: Pearson r between the subject's dynamic metrics and the
        reference's (`_ref_tables`: every cohort once, all its scans), each metric
        divided by its SD over all subjects so units do not decide the profile. With only a few
        metrics this r is coarse: a check, not a measure to report on its own."""
        allv = pd.concat(list(self.data.values()))
        sd = allv[self.dyn].std().to_numpy(float)
        R = pd.concat([v[self.dyn].mean() for v in self._ref_tables()], axis=1).mean(axis=1).to_numpy(float) / sd
        Rc = R - R.mean()
        for key, d in [*self.data.items(), *getattr(self, "by_task", {}).items(), *getattr(self, "pooled", {}).items()]:
            if not set(self.dyn) <= set(d):
                continue
            X = d[self.dyn].to_numpy(float) / sd
            Xc = X - X.mean(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                d["dynamics_fit"] = (Xc @ Rc) / np.sqrt((Xc ** 2).sum(1) * (Rc ** 2).sum())
        return "dynamics_fit"

    def stability_variables(self):
        fit = self.transition_fit()
        return [fit, self.occupancy_fit(), *self.dyn, *self.occ, *self.strongest()]

    def paired_rest_movie(self):
        """{variable-ready}: rest and movie of the people with both, one row each, indexed
        cohort:sub, with the cohort in a column (for cohort-demeaning)."""
        r, m = [], []
        for c in self.cfg.cohorts:
            if (c, "rest") in self.data and (c, "movie") in self.data:
                both = self.data[c, "rest"].index.intersection(self.data[c, "movie"].index)
                r.append(self.data[c, "rest"].loc[both].rename(index=lambda s, c=c: f"{c}:{s}").assign(cohort=c))
                m.append(self.data[c, "movie"].loc[both].rename(index=lambda s, c=c: f"{c}:{s}").assign(cohort=c))
        if not r:
            raise SystemExit("no cohort has the same people in rest and movie")
        return pd.concat(r), pd.concat(m)

    def stability(self, variables=None, n_boot=2000, seed=0):
        """One row per variable, everything panels A and B need.

        between cohorts (movie): the largest |d| of any cohort vs the others (`cohen`), and whether
            every cohort is equivalent to the others (TOST, `agreement`).
        movie vs rest (same people, cohorts with both): d_av with a bootstrap 95% CI over people,
            and a paired TOST: the 90% CI of the mean difference inside +- sesoi_d x pooled SD.
        individual consistency: ICC(3,1) between rest and movie on cohort-demeaned values (a
            cohort's offset is not 'keeping your rank'), bootstrap 95% CI; and per cohort.

        `tier` -- the classification the figures use -- rests on movie vs rest in the SAME people
        only, the one contrast that isolates context: |d_av| < 0.2 "stable", 0.2-0.5 "moderate",
        > 0.5 "context-dependent" (Cohen's small / medium). The cohorts differ in age, scanner,
        TR and run length, so their levels are not expected to be equal; whether the cohorts
        share the STRUCTURE is `replication()`'s question, not a per-variable equivalence.
        `stable` (equivalent in both comparisons, TOST) is kept for reference only.
        n_boot=0 skips the bootstrap CIs (fast, for sweeps over state sets).
        """
        vs = variables or self.stability_variables()
        if "transition_fit" in vs:
            self.transition_fit()                  # computed on demand, not stored in the tables
        if "occupancy_fit" in vs:
            self.occupancy_fit()
        if "dynamics_fit" in vs:
            self.dynamics_fit()
        rng = np.random.default_rng(seed)
        coh, agr = self.cohen(vs), self.agreement(vs)
        r, m = self.paired_rest_movie()
        demean = lambda t, v: t[v] - t.groupby("cohort")[v].transform("mean")
        n = len(r)
        boot = rng.integers(0, n, size=(n_boot, n)) if n_boot else np.empty((0, n), int)
        rows = []
        for v in vs:
            cv = coh[coh["variable"] == v]
            if cv["d"].notna().any():
                i = cv["d"].abs().idxmax()
            else:                                         # no cohort comparable: keep the row, NaN
                cv = cv.assign(d=np.nan, hi=np.nan).iloc[:1]
                i = cv.index[0]
            rv, mv_ = r[v].to_numpy(float), m[v].to_numpy(float)
            ok = np.isfinite(rv) & np.isfinite(mv_)
            if ok.sum() < 3:                              # too few people with both: nothing to estimate
                rows.append(dict(variable=v, between_d=abs(cv.loc[i, "d"]), between_cohort=cv.loc[i, "cohort"],
                                 between_equivalent=False, task_d=np.nan, task_lo=np.nan, task_hi=np.nan,
                                 task_p_tier=np.nan, task_equivalent=False, task_p=np.nan, task_p_tost=np.nan,
                                 n_paired=int(ok.sum()), icc=np.nan, icc_lo=np.nan, icc_hi=np.nan))
                continue
            d_av = _paired(pd.Series(rv[ok]), pd.Series(mv_[ok]))
            bd = [_paired(pd.Series(rv[b][ok[b]]), pd.Series(mv_[b][ok[b]])) for b in boot]
            diff = (mv_ - rv)[ok]
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            h = stats.t.ppf(.95, len(diff) - 1) * se
            bound = self.cfg.sesoi_d * np.nanstd(np.r_[rv, mv_], ddof=1)
            rd, md = demean(r, v).to_numpy(float), demean(m, v).to_numpy(float)
            icc = _icc_c1(rd, md)
            bi = [_icc_c1(rd[b], md[b]) for b in boot]
            row = dict(variable=v,
                       between_d=abs(cv.loc[i, "d"]), between_lo=abs(cv.loc[i, "d"]) - (cv.loc[i, "hi"] - cv.loc[i, "d"]),
                       between_hi=abs(cv.loc[i, "d"]) + (cv.loc[i, "hi"] - cv.loc[i, "d"]),
                       between_cohort=cv.loc[i, "cohort"],
                       between_equivalent=bool(agr.loc[agr["variable"] == v, "equivalent"].all()),
                       task_mean_rest=float(np.mean(rv[ok])), task_mean_movie=float(np.mean(mv_[ok])),
                       task_d=d_av, task_lo=np.nanpercentile(bd, 2.5) if n_boot else np.nan,
                       task_hi=np.nanpercentile(bd, 97.5) if n_boot else np.nan,
                       # confidence in the tier: share of bootstrap |d| in the observed |d|'s tier
                       task_p_tier=(float(np.mean(_tier_index(np.abs(np.asarray(bd, float)))
                                                  == _tier_index(abs(d_av)))) if n_boot else np.nan),
                       task_equivalent=bool((diff.mean() - h > -bound) and (diff.mean() + h < bound)),
                       task_p=stats.ttest_rel(mv_[ok], rv[ok]).pvalue, n_paired=int(ok.sum()),
                       # TOST: both one-sided tests against +-bound must reject; p = the larger
                       task_p_tost=(max(stats.t.sf((diff.mean() + bound) / se, len(diff) - 1),
                                        stats.t.cdf((diff.mean() - bound) / se, len(diff) - 1))
                                    if se > 0 and len(diff) > 1 else np.nan),
                       icc=icc, icc_lo=np.nanpercentile(bi, 2.5) if n_boot else np.nan,
                       icc_hi=np.nanpercentile(bi, 97.5) if n_boot else np.nan)
            for c in r["cohort"].unique():
                s = (r["cohort"] == c).to_numpy()
                row[f"icc_{c}"] = _icc_c1(rv[s], mv_[s])
            rows.append(row)
        out = pd.DataFrame(rows)
        out["task_p_fdr"] = stats.false_discovery_control(out["task_p"].fillna(1))
        out["task_p_tost_fdr"] = stats.false_discovery_control(out["task_p_tost"].fillna(1))
        out["stable"] = out["between_equivalent"] & out["task_equivalent"]
        a = out["task_d"].abs()
        out["tier"] = np.where(a.isna(), "n/a",
                               np.select([a < .2, a < .5], ["stable", "moderate"], "context-dependent"))
        return out

    def task_variation(self, variables=None, n_boot=2000, seed=0, ci="jackknife", combine="random",
                       min_people=3):
        """How much each property depends on the TASK: every task of every cohort, rest included as
        one task (`films(rest=True)`, so the halves of an episode are one task). One row per variable.

        omega^2 (Hays) of task, computed in each cohort on its own and averaged over cohorts, each
        cohort once (a large cohort does not outweigh a small one; scanner and population, shown
        per cohort in Figure 3 top, are not counted as a task effect). omega^2 rather than
        eta^2: eta^2 is inflated when tasks have few people (NNDb films, CNeuroMod episodes).
            omega^2 = (SS_task - df_task MS_within) / (SS_task + SS_within + MS_within),
        per cohort, df_task = tasks_c - 1. Values are treated as independent, which
        is conservative for cohorts whose people did several tasks (HCP, CNeuroMod).
        Tiers (Cohen 1988, the variance-explained form of d = .2 / .5): < .01 stable,
        .01-.06 moderate, >= .06 task-dependent.
        combine: how the cohorts' omega^2 become one. "random" (default): random-effects
        meta-analysis (DerSimonian & Laird 1986) with each cohort's jackknife variance -- weights
        1 / (v_c + tau^2), nearly equal when the cohorts disagree, by precision when they agree;
        "fixed": inverse-variance weights (tau^2 = 0); "equal": the plain mean, each cohort once;
        "mixed": no per-cohort omega^2 -- one linear mixed model per variable on all the data,
        value ~ cohort + (1 | task in cohort) + (1 | person), and the share of the variance that is
        the task's (variance partition coefficient, Nakagawa & Schielzeth 2010; see `_task_vpc`).
        min_people: a task group (cohort x task) needs this many people to enter (fewer gives
        omega^2 near 1 by chance, e.g. a transition row only 1-2 people have in a film).
        ci="jackknife" (default): 95% CI and P(tier) from the delete-one-person jackknife SE within
        each cohort (a person leaves with all their tasks), normal approximation; n_boot only turns
        it on (> 0) or off (0). ci="bootstrap": n_boot resamples of people -- biased upward for
        omega^2 with few people per cohort (repeated people shrink the within-task variance).
        Columns named as `stability` (task_d = omega^2, task_lo, task_hi, tier, task_p_tier), plus
        task_mean_lo / task_mean_hi: the lowest and highest task mean. st.attrs["scale"] = "omega2"."""
        if variables is None:
            variables = ["transition_fit", "occupancy_fit", *self.dyn, *self.occ]
        for f in ("transition_fit", "occupancy_fit", "dynamics_fit"):
            if f in variables:
                getattr(self, f)()
        F = self.films(rest=True)
        L = pd.concat([d.reindex(columns=variables).assign(_c=c, _t=t, _s=[f"{c}:{x}" for x in d.index])
                       for (c, t), d in F.items()], ignore_index=True)
        X = L[variables].to_numpy(float)
        valid = np.isfinite(X)
        X0 = np.where(valid, X, 0.0)
        g_id, g_lab = pd.factorize(L["_c"] + "|" + L["_t"])                 # task groups
        c_of_g = pd.factorize(pd.Series(g_lab).str.split("|").str[0])[0]     # their cohort
        n_c = c_of_g.max() + 1
        s_id, s_lab = pd.factorize(L["_s"])                                  # people
        s_coh = pd.factorize(pd.Series(s_lab).str.split(":").str[0])[0]

        def omega(w):                                       # w: weight per row (bootstrap multiplicity)
            W = valid * w[:, None]
            G = len(g_lab)
            n_g, S_g, Q_g = (np.zeros((G, X.shape[1])) for _ in range(3))
            np.add.at(n_g, g_id, W)
            np.add.at(S_g, g_id, W * X0)
            np.add.at(Q_g, g_id, W * X0 ** 2)
            small = n_g < min_people                        # too few people in this task for this variable
            n_g[small], S_g[small], Q_g[small] = 0, 0, 0
            # per cohort: its own one-way omega^2 of task; then the mean over cohorts, each once
            Z = lambda: np.zeros((n_c, X.shape[1]))
            n_cc, S_c, T_c, Sb, Sw = Z(), Z(), Z(), Z(), Z()
            np.add.at(n_cc, c_of_g, n_g)
            np.add.at(S_c, c_of_g, S_g)
            np.add.at(T_c, c_of_g, (n_g > 0).astype(float))
            with np.errstate(invalid="ignore", divide="ignore"):
                np.add.at(Sw, c_of_g, np.where(n_g > 0, Q_g - S_g ** 2 / np.where(n_g > 0, n_g, 1), 0))
                np.add.at(Sb, c_of_g, np.where(n_g > 0, S_g ** 2 / np.where(n_g > 0, n_g, 1), 0))
                ss_b = Sb - np.where(n_cc > 0, S_c ** 2 / np.where(n_cc > 0, n_cc, 1), 0)
                df_b, df_w = T_c - 1, n_cc - T_c
                ms_w = Sw / df_w
                om_c = (ss_b - df_b * ms_w) / (ss_b + Sw + ms_w)
                om_c[(df_b < 1) | (df_w < 1)] = np.nan
                means = np.where(n_g >= 2, S_g / np.where(n_g > 0, n_g, 1), np.nan)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                om = np.nanmean(om_c, 0)
            return om, means, om_c

        if combine == "mixed":
            return self._task_vpc(L, variables, min_people, ci=bool(n_boot))
        om, means, om_c = omega(np.ones(len(L)))
        c_names = list(pd.unique(pd.Series(g_lab).str.split("|").str[0]))      # c_of_g's order
        if ci == "jackknife" and (n_boot or combine != "equal"):
            # delete-one-person jackknife within each cohort (a person leaves with all their tasks).
            # The bootstrap is biased upward here: a resample repeats people, identical copies shrink
            # the within-task variance, and with 5 people (CNeuroMod) omega^2 jumps towards 1.
            from scipy.stats import norm
            var_c = np.zeros_like(om_c)
            sub_c = pd.Series(s_lab).str.split(":").str[0].to_numpy()
            for k, c in enumerate(c_names):
                people = np.flatnonzero(sub_c == c)
                n = len(people)
                if n < 3:
                    var_c[k] = np.nan
                    continue
                J = []
                for sp in people:
                    w = np.ones(len(L))
                    w[s_id == sp] = 0
                    J.append(omega(w)[2][k])
                J = np.array(J)
                var_c[k] = (n - 1) / n * np.nansum((J - np.nanmean(J, 0)) ** 2, 0)
            om, se = _combine(om_c, var_c, combine)
            cuts_ = (.01, .06)
            lo_, hi_ = om - 1.96 * se, om + 1.96 * se
            band = np.digitize(om, cuts_)
            edges = np.array([-np.inf, *cuts_, np.inf])
            with np.errstate(invalid="ignore", divide="ignore"):
                p_tier = norm.cdf((edges[band + 1] - om) / se) - norm.cdf((edges[band] - om) / se)
            p_tier[~np.isfinite(om)] = np.nan
            B = None
        rng = np.random.default_rng(seed)
        boots = []
        subs_by_c = [np.flatnonzero(s_coh == k) for k in range(s_coh.max() + 1)] if ci == "bootstrap" else []
        for _ in range(n_boot if ci == "bootstrap" else 0):
            m = np.zeros(len(s_lab))
            for idx in subs_by_c:
                np.add.at(m, rng.choice(idx, len(idx)), 1)
            boots.append(omega(m[s_id])[0])
        if ci == "bootstrap":
            B = np.array(boots) if n_boot else np.empty((0, len(variables)))
        cuts = (.01, .06)
        tier = lambda a: np.digitize(a, cuts)
        out = pd.DataFrame({
            "variable": variables, "task_d": om,
            "task_lo": (lo_ if ci == "jackknife" else np.nanpercentile(B, 2.5, axis=0)) if n_boot else np.nan,
            "task_hi": (hi_ if ci == "jackknife" else np.nanpercentile(B, 97.5, axis=0)) if n_boot else np.nan,
            "task_p_tier": ((p_tier if ci == "jackknife" else
                             [float(np.mean(tier(B[:, k]) == tier(om[k]))) if np.isfinite(om[k]) else np.nan
                              for k in range(len(variables))]) if n_boot else np.nan),
            "task_mean_lo": np.nanmin(means, 0), "task_mean_hi": np.nanmax(means, 0),
            "n_tasks": (np.isfinite(means)).sum(0), "n_paired": valid.sum(0)})
        out["tier"] = np.where(np.isnan(om), "n/a",
                               np.select([om < cuts[0], om < cuts[1]], ["stable", "moderate"], "context-dependent"))
        out.attrs["scale"] = "omega2"
        return out

    def _task_vpc(self, L, variables, min_people=3, ci=True):
        """task_variation(combine="mixed"). Per variable, REML fit of
            value ~ C(cohort) + (1 | task) + (1 | person),   task and person nested in cohort,
        (statsmodels MixedLM, cohort as the block, task and person as variance components), and
            VPC_task = s2_task / (s2_task + s2_person + s2_residual).
        Repeated measures are modelled (HCP, CNeuroMod: the same people in many tasks), cohorts are
        neither averaged nor weighted, and cohort differences are the fixed effect, not the task's.
        95% CI and P(tier): delta method on the logit of the VPC from the REML covariance of the
        variance parameters (asymptotic; at a variance on its boundary, 0, the CI collapses there).
        Same columns as task_variation; st.attrs["scale"] = "omega2" (same tiers)."""
        import warnings
        import statsmodels.formula.api as smf
        from scipy.stats import norm
        rows = []
        cuts = (.01, .06)
        for v in variables:
            df = pd.DataFrame({"y": L[v].to_numpy(float), "cohort": L["_c"].to_numpy(),
                               "task": (L["_c"] + "|" + L["_t"]).to_numpy(), "person": L["_s"].to_numpy()})
            df = df[np.isfinite(df["y"])]
            df = df[df.groupby("task")["y"].transform("size") >= min_people]
            means = df.groupby("task")["y"].mean()
            row = dict(variable=v, task_d=np.nan, task_lo=np.nan, task_hi=np.nan, task_p_tier=np.nan,
                       task_mean_lo=means.min() if len(means) else np.nan,
                       task_mean_hi=means.max() if len(means) else np.nan,
                       n_tasks=int(df["task"].nunique()), n_paired=len(df))
            if df.groupby("cohort")["task"].nunique().max() < 2 or df["y"].std() == 0:
                rows.append(row)
                continue
            df["y"] = (df["y"] - df["y"].mean()) / df["y"].std()          # scale-free; the VPC is unchanged
            try:
                model = smf.mixedlm("y ~ C(cohort)", df, groups="cohort", re_formula="0",
                                    vc_formula={"task": "0 + C(task)", "person": "0 + C(person)"})
                r, errs = None, []
                for method in ("lbfgs", "bfgs", "powell", "nm"):     # the default optimiser can fail
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            r = model.fit(reml=True, method=method)
                        if np.isfinite(r.params.get("task Var", np.nan)):
                            break
                    except Exception as e:
                        errs.append(f"{method}: {type(e).__name__}: {e}"[:120])
                        r = None
                if r is None:
                    raise RuntimeError(" | ".join(errs))
                tau, pi = (float(r.params[f"{n} Var"]) for n in ("task", "person"))   # relative to s2_residual
                D = tau + pi + 1
                vpc = tau / D
                row["task_d"] = vpc
                # the whole partition, cohort included: the cohort's share is the variance of the
                # fixed-effect predictions (Nakagawa & Schielzeth 2013's marginal R^2); task, person
                # and residual from the same fit. Shares of the TOTAL variance; task_d stays the VPC
                # within cohorts (cohort out of the denominator).
                s2 = float(r.scale)
                comp = {"cohort": float(np.var(r.model.exog @ r.fe_params.to_numpy())),
                        "task": tau * s2, "person": pi * s2, "residual": s2}
                tot = sum(comp.values())
                for k_, v_ in comp.items():
                    row[f"var_{k_}"] = v_
                    row[f"share_{k_}"] = v_ / tot if tot > 0 else np.nan
                row["r2_marginal"] = row["share_cohort"]
                row["r2_conditional"] = 1 - row["share_residual"]
                if ci:
                    # not named C: patsy evaluates the formula's C(...) in THIS function's namespace
                    cov = r.cov_params().loc[["task Var", "person Var"], ["task Var", "person Var"]].to_numpy(float)
                    g = np.array([(pi + 1) / D ** 2, -tau / D ** 2])               # d VPC / d (tau, pi)
                    se = float(np.sqrt(max(g @ cov @ g, 0)))
                    if 0 < vpc < 1 and se > 0:
                        lg, se_l = np.log(vpc / (1 - vpc)), se / (vpc * (1 - vpc))   # logit scale
                        expit = lambda z: 1 / (1 + np.exp(-z))
                        row["task_lo"], row["task_hi"] = expit(lg - 1.96 * se_l), expit(lg + 1.96 * se_l)
                        band = int(np.digitize(vpc, cuts))
                        e = [0.0, *cuts, 1.0]
                        lgt = lambda x: np.log(x / (1 - x)) if 0 < x < 1 else (-np.inf if x <= 0 else np.inf)
                        row["task_p_tier"] = float(norm.cdf((lgt(e[band + 1]) - lg) / se_l)
                                                   - norm.cdf((lgt(e[band]) - lg) / se_l))
                    else:
                        row["task_lo"], row["task_hi"] = vpc, vpc
            except Exception as err:                           # a fit that fails stays NaN, with a note
                row["fit_error"] = f"{type(err).__name__}: {err}"[:300]
            rows.append(row)
        out = pd.DataFrame(rows)
        if "fit_error" in out and out["fit_error"].notna().any():
            bad = out[out["fit_error"].notna()]
            print(f"mixed model failed for {len(bad)} of {len(out)} variables; first: "
                  f"{bad.iloc[0]['variable']}: {bad.iloc[0]['fit_error']}")
        out["tier"] = np.where(out["task_d"].isna(), "n/a",
                               np.select([out["task_d"] < cuts[0], out["task_d"] < cuts[1]],
                                         ["stable", "moderate"], "context-dependent"))
        out.attrs["scale"] = "omega2"
        out.attrs["measure"] = "vpc"
        return out

    def replication(self, kind="movie", n_split=20, min_occ=0.01, seed=0, balance=None, hours=10.0,
                    n_boot=200, versus="pairs"):
        """Does the transition STRUCTURE replicate across cohorts, whatever their levels?

        r between cohorts' mean p(j | i) (rows of states occupied >= min_occ), divided by the
        noise ceiling: the r between two random halves of the same cohort (mean of n_split
        splits), geometric mean of the two cohorts' ceilings. ~1 = two cohorts agree as well as a
        cohort agrees with itself. Returns (pairs, ceilings).

        balance="time" -- people AND stimulus duration at once: every cohort contributes the SAME
        amount of data. Each draw takes scans (runs, films, episodes) at random until `hours` of
        scan time, pools their transition COUNTS (a long scan weighs by its length, as it should)
        and turns them into p(j | i). A cohort of many short scans (Cam-CAN) then contributes many
        people, one of few long ones (CNeuroMod, NNDb) few people with much data each; neither is
        left out and neither wins by sheer size. The ceiling is two such draws from DISJOINT halves
        of the cohort's people (so a person is never in both). `n_boot` draws give the 95% CI;
        rest vs movie in the same people is added for the cohorts that have both.

        versus="mean" (balance="time" only): each cohort against the mean of the OTHER cohorts
        (leave-one-out, so a cohort is never compared with itself), one row per cohort instead of
        one per pair. The ceiling of that mean is the r between the means of the others' halves.
        """
        if versus not in ("pairs", "mean"):
            raise ValueError(f'versus must be "pairs" or "mean", got {versus!r}')
        if balance == "time":
            return self._replication_time(min_occ, seed, hours, n_boot, versus)
        if versus == "mean":
            raise ValueError('versus="mean" needs balance="time"')
        if balance is not None:
            raise ValueError(f'balance must be None or "time", got {balance!r}')
        K, rng = self.cfg.K, np.random.default_rng(seed)
        occ = pd.concat([v[self.occ].mean() for v in self.data.values()], axis=1).mean(axis=1).to_numpy()
        rows = np.repeat(occ >= min_occ, K)
        vec = lambda d: d[self.cells].mean().to_numpy(float)[rows]
        groups = {c: d for (c, k), d in self.data.items() if k == kind}
        ceil = {}
        for c, d in groups.items():
            if len(d) < 4:
                ceil[c] = np.nan
                continue
            rs = []
            for _ in range(n_split):
                idx = rng.permutation(len(d))
                rs.append(np.corrcoef(vec(d.iloc[idx[: len(d) // 2]]), vec(d.iloc[idx[len(d) // 2:]]))[0, 1])
            ceil[c] = float(np.mean(rs))
        pairs = []
        labels = list(groups)
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                r = float(np.corrcoef(vec(groups[a]), vec(groups[b]))[0, 1])
                pairs.append(dict(a=a, b=b, r=r, r_norm=r / np.sqrt(ceil[a] * ceil[b])))
        return pd.DataFrame(pairs), pd.Series(ceil, name="split_half_r")

    # -- variation between films ----------------------------------------------------------
    def films(self, rest=False):
        """{(cohort, film): one row per subject} for every movie task (film, episode, run). Tasks
        whose labels `film_merge` maps to the same film (the halves of an episode) are one film: a
        person's values are averaged over its parts. rest=True adds each cohort's rest as one more
        task, "rest" (its runs averaged per person), first in the cohort."""
        pat, sub = self.cfg.film_merge or (None, None)
        groups = {}
        items = sorted(self.by_task.items(), key=lambda kv: (kv[0][0], kv[0][1] != "rest", kv[0][2]))
        for (c, k, t), d in items:
            if k == "movie" or rest:
                name = "rest" if k == "rest" else (re.sub(pat, sub, t) if pat else t)
                groups.setdefault((c, name), []).append(d)
        return {key: (ds[0] if len(ds) == 1 else pd.concat(ds).groupby(level=0).mean())
                for key, ds in groups.items()}

    def duration_check(self, variables=None):
        """Does a metric follow the amount of scan? One row per (cohort, rest|movie, variable): the
        Spearman correlation, over every (person, task) value, between the metric and the scan
        minutes behind it. Within a cohort, so scanner and population are fixed; across tasks the
        content changes too (NNDb: each film is its own length), so a correlation is a flag to look
        at, not a proof. Cohorts whose scans all last the same have nothing to show (rho NaN)."""
        from scipy.stats import spearmanr
        variables = variables or [*self.dyn, *self.occ]
        rows = []
        for (c, k) in sorted({(c, k) for c, k, _ in self.by_task}):
            t = pd.concat([d for (c_, k_, _), d in self.by_task.items() if (c_, k_) == (c, k)])
            m = t["scan_s"] / 60 if "scan_s" in t else pd.Series(np.nan, index=t.index)
            for v in [v for v in variables if v in t]:
                ok = m.notna() & t[v].notna()
                varies = ok.sum() >= 10 and m[ok].nunique() > 1 and m[ok].std() > .05 * m[ok].mean()
                rho, p = spearmanr(m[ok], t.loc[ok, v]) if varies else (np.nan, np.nan)
                rows.append(dict(cohort=c, kind=k, variable=v, n=int(ok.sum()),
                                 minutes_min=m[ok].min(), minutes_max=m[ok].max(), rho=rho, p=p))
        out = pd.DataFrame(rows, columns=["cohort", "kind", "variable", "n", "minutes_min",
                                          "minutes_max", "rho", "p"])
        q = out["p"].dropna()
        out["p_fdr"] = np.nan
        if len(q):
            out.loc[q.index, "p_fdr"] = stats.false_discovery_control(q)
        return out

    def film_summary(self, variables=None, rest=False):
        """One row per (film, variable): mean and 95% CI over the film's viewers, and whether the
        film is equivalent to the OTHER films of its cohort (TOST at 5%, bounds +- sesoi_d x the
        variable's SD over every film).

        Within person when the cohort's films share viewers (>= 3 people saw this film and another):
        each person's value in this film against their own mean over their other films, paired.
        Between people otherwise (each film seen by different people): Welch against the cohort's
        other films pooled. A cohort with one film has nothing to compare: equivalent = NaN.
        `differs`: the same comparison as a two-sided test, BH-FDR over every film and variable.
        With few viewers a film is rarely shown equivalent; `differs` is what the figures mark.
        """
        if variables is None:
            variables = [self.transition_fit(), *self.dyn, *self.occ, *self.cells]
        else:
            self.transition_fit()
        F = self.films(rest=rest)
        allv = pd.concat(F.values())
        sd = allv[variables].std()
        rows = []
        for (c, t), d in F.items():
            others = {k: v for k, v in F.items() if k[0] == c and k != (c, t)}
            pooled = pd.concat(others.values()) if others else None
            shared = d.index.intersection(pooled.index) if pooled is not None else pd.Index([])
            within = len(shared) >= 3
            other_mean = pooled.groupby(level=0).mean() if within else None
            for v in variables:
                x = d[v].dropna()
                n = len(x)
                h = stats.t.ppf(.975, max(n - 1, 1)) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                eq, diff, pv = np.nan, np.nan, np.nan
                bound = self.cfg.sesoi_d * sd[v]
                if within:
                    dd = (d.loc[shared, v] - other_mean.loc[shared, v]).dropna()
                    if len(dd) >= 3:
                        hh = stats.t.ppf(.95, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
                        diff = dd.mean()
                        eq = bool((diff - hh > -bound) and (diff + hh < bound))
                        pv = stats.ttest_1samp(dd, 0).pvalue
                elif pooled is not None and len(x) >= 2 and pooled[v].notna().sum() >= 2:
                    y = pooled[v].dropna()
                    va, vb = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
                    se = np.sqrt(va + vb)
                    df = (va + vb) ** 2 / (va ** 2 / max(len(x) - 1, 1) + vb ** 2 / max(len(y) - 1, 1))
                    diff = x.mean() - y.mean()
                    hh = stats.t.ppf(.95, df) * se
                    eq = bool((diff - hh > -bound) and (diff + hh < bound))
                    pv = stats.ttest_ind(x, y, equal_var=False).pvalue
                rows.append(dict(cohort=c, film=t, variable=v, n=n, mean=x.mean() if n else np.nan, lo=x.mean() - h,
                                 hi=x.mean() + h, within=within, diff=diff, d=diff / sd[v] if sd[v] > 0 else np.nan,
                                 equivalent=eq, p=pv))
        out = pd.DataFrame(rows)
        ok = out["p"].notna()
        out["p_fdr"] = np.nan
        out.loc[ok, "p_fdr"] = stats.false_discovery_control(out.loc[ok, "p"])
        out["differs"] = out["p_fdr"] < self.cfg.alpha
        return out

    def context_summary(self, st, fs):
        """One row per variable, the three comparisons of the text on one scale (|d|, Cohen's tiers):
        rest vs movie in the same people (`stability`); film vs film in the same people (median |d|
        of each film against the person's other films, HCP runs and CNeuroMod episodes); and movie
        across cohorts, different people (median |d| of each cohort against the others, `cohen`) --
        film, people, scanner and age change together there, so it can show that a property is
        alike DESPITE all that, not why it differs."""
        tier = lambda a: np.select([a < .2, a < .5], ["stable", "moderate"], "context-dependent")
        coh = self.cohen(list(st["variable"]))
        out = st[["variable"]].copy()
        out["rest_vs_movie"] = st["task_d"].abs().to_numpy()
        fs = fs[fs["film"] != "rest"]                     # film vs film only
        w = fs[fs["within"]].groupby("variable")["d"].apply(lambda x: x.abs().median())
        out["film_vs_film_same_people"] = out["variable"].map(w)
        out["movie_across_cohorts"] = out["variable"].map(coh.groupby("variable")["d"].apply(lambda x: x.abs().median()))
        for c in ("rest_vs_movie", "film_vs_film_same_people", "movie_across_cohorts"):
            out[f"tier_{c}"] = np.where(out[c].notna(), tier(out[c].fillna(0)), "")
        return out

    def rest_means(self, variables):
        """Every cohort's rest pooled (each cohort once): the reference line of the film figures."""
        r = [self.data[c, "rest"][variables].mean() for c in self.cfg.cohorts if (c, "rest") in self.data]
        return pd.concat(r, axis=1).mean(axis=1) if r else pd.Series(np.nan, variables)

    # -- reference structure across cohorts ------------------------------------------------
    def similarity(self, min_occ=0.01):
        """Pearson r between the mean p(j | i) of every (cohort, rest|movie), on the rows of the
        states the reference occupies >= min_occ (rare states' rows are noise). The question is
        how alike the transition structure is across cohorts and conditions; its null (the same
        with time circularly shifted) needs the state sequences and is not computed here."""
        K = self.cfg.K
        occ = pd.concat([v[self.occ].mean() for v in self.data.values()], axis=1).mean(axis=1).to_numpy()
        rows = np.repeat(occ >= min_occ, K)
        names = [f"{c} {k}" for (c, k) in self.data]
        M = np.array([v[self.cells].mean().to_numpy(float)[rows] for v in self.data.values()])
        return pd.DataFrame(np.corrcoef(M), index=names, columns=names)

    def _replication_time(self, min_occ, seed, hours, n_boot, versus="pairs"):
        K, rng = self.cfg.K, np.random.default_rng(seed)
        occ = pd.concat([v[self.occ].mean() for v in self.data.values()], axis=1).mean(axis=1).to_numpy()
        rows = np.repeat(occ >= min_occ, K)
        U = self.units.dropna(subset=["minutes"])
        groups = {(c, k): g for (c, k), g in U.groupby(["cohort", "kind"])}
        target = hours * 60

        def draw(g):
            idx = rng.permutation(len(g))
            m = g["minutes"].to_numpy()[idx]
            take = idx[: int(np.searchsorted(np.cumsum(m), target)) + 1]
            C = g[self.cells].to_numpy(float)[take].sum(0).reshape(K, K)
            P = C / np.where(C.sum(1, keepdims=True) > 0, C.sum(1, keepdims=True), 1)
            return P.ravel()[rows]

        def half(g):
            subs = g["sub"].unique()
            pick = set(rng.permutation(subs)[: len(subs) // 2])
            return g[g["sub"].isin(pick)], g[~g["sub"].isin(pick)]

        r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
        ceil = {key: [] for key in groups}
        pairs = {}
        keys = list(groups)
        movie = [k for k in keys if k[1] == "movie"]
        compare = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
                   if (versus == "pairs" and a[1] == b[1] == "movie") or (a[0] == b[0] and a[1] != b[1])]
        others = {a: [b for b in movie if b != a] for a in movie} if versus == "mean" else {}
        for a in others:
            ceil[("others", a)] = []
        for _ in range(n_boot):
            halves = {}
            for key, g in groups.items():
                if g["sub"].nunique() >= 2:
                    h1, h2 = half(g)
                    halves[key] = draw(h1), draw(h2)
                    ceil[key].append(r(*halves[key]))
            v = {key: draw(g) for key, g in groups.items()}
            for a, b in compare:
                pairs.setdefault((a, b), []).append(r(v[a], v[b]))
            for a, bs in others.items():
                pairs.setdefault((a, ("others", a)), []).append(r(v[a], np.mean([v[b] for b in bs], 0)))
                if all(b in halves for b in bs):
                    ceil[("others", a)].append(r(np.mean([halves[b][0] for b in bs], 0),
                                                 np.mean([halves[b][1] for b in bs], 0)))
        cm = {k: float(np.nanmedian(x)) if x else np.nan for k, x in ceil.items()}
        rows_out = []
        for (a, b), rs in pairs.items():
            rs = np.array(rs)
            norm = rs / np.sqrt(cm[a] * cm[b])
            rows_out.append(dict(a=f"{a[0]} {a[1]}",
                                 b="mean of the others" if b[0] == "others" else f"{b[0]} {b[1]}",
                                 kind=("vs mean" if b[0] == "others" else
                                       "rest vs movie" if a[0] == b[0] else "between cohorts"),
                                 r=np.median(rs), r_norm=np.median(norm),
                                 lo=np.percentile(norm, 2.5), hi=np.percentile(norm, 97.5)))
        ceil_s = pd.Series({(f"others than {k[1][0]}" if k[0] == "others" else f"{k[0]} {k[1]}"): v
                            for k, v in cm.items()}, name="split_half_r")
        return pd.DataFrame(rows_out), ceil_s


class Cohorts(StabilityMixin):
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or Settings()
        self.base = transitions_root(
            self.cfg.root, self.cfg.space["atlas"], self.cfg.space["window_s"],
            self.cfg.space["states"], version=self.cfg.modelling_version)
        self.data = self._load()

    # -- reading ---------------------------------------------------------------------------
    def _files(self, label):
        out = {"rest": [], "movie": []}
        for d in self.cfg.cohorts[label]["dirs"]:
            for f in sorted(self.base.glob(f"cohort={d}/**/subjects.parquet")):
                task = ht.hive(f, "task") or ""
                out["rest" if re.fullmatch(self.cfg.rest_pattern, task, re.I) else "movie"].append(f)
        return out

    def _load(self):
        """{(cohort, rest|movie): one row per subject}, cells as p(j | i), runs and films averaged."""
        K, data = self.cfg.K, {}
        self.by_task = {}                                                     # (cohort, rest|movie, task) -> subjects
        units = []                                                            # one row per scan: counts + duration
        frames = {}                                                           # cohort -> its scans, every kind
        raw = {(label, kind): pd.concat(map(pd.read_parquet, files), ignore_index=True)
               for label in self.cfg.cohorts for kind, files in self._files(label).items() if files}
        target = self.cfg.distinct_minutes
        if target == "min":
            target = min(float(_scan_minutes(d).min()) for d in raw.values())
        self.distinct_minutes = target
        for (label, kind), d in raw.items():
            if target is not None and "n_distinct_transitions" in d:
                d["n_distinct_transitions"] = rarefied_distinct(d, K, target)
            d["sub"] = d["sub"].astype(str).str.strip().str.upper()
            if self.cfg.entropy_estimator is not None and "entropy_rate_bits" in d:
                # before as_p_cond: the recompute needs the p_joint cells
                from fmri_decomposition.unsupervised_model.transitions import entropy_from_table
                d["entropy_rate_bits"] = entropy_from_table(d, K, self.cfg.entropy_estimator)
            self.cells = ht.cell_cols(d)
            # the scan as a UNIT of data: its transition counts (p_joint x n) and its duration,
            # kept before p(j | i) throws the counts away -- for data-balanced resampling
            u = d[["sub", "task"]].copy()
            u[self.cells] = d[self.cells].to_numpy(float) * d["n_transitions"].to_numpy(float)[:, None]
            u["minutes"] = ((d["n_windows"] * d["stride_s"]).astype(float) / 60
                            if {"n_windows", "stride_s"} <= set(d) else np.nan)
            units.append(u.assign(cohort=label, kind=kind))
            d["scan_s"] = (d["n_windows"] * d["stride_s"]).astype(float) if {"n_windows", "stride_s"} <= set(d) else np.nan
            frames.setdefault(label, []).append(d.assign(_kind=kind))
        if not frames:
            raise SystemExit(f"no transition tables under {self.base}")

        # p(j | i) per PERSON, from the transition counts summed over their scans -- not a mean of
        # per-scan p(j | i): a scan that never leaves state i has no row i, and averaging it in as 0
        # mixed occupancy into the transitions. Whether a person has a row i at all is decided on ALL
        # their scans, every task and condition together: fewer than min_out_transitions exits from
        # i in total -> NaN everywhere; a task in which an eligible person never left i -> NaN there.
        m_out = max(self.cfg.min_out_transitions, 1)
        self.pooled = {}                                                      # cohort -> person, all scans
        for label, ds in frames.items():
            d = pd.concat(ds, ignore_index=True)
            counts = d[self.cells].to_numpy(float) * d["n_transitions"].to_numpy(float)[:, None]
            other = [*self.occ, *[v for v in self.cfg.dynamics if v in d]]

            def per_sub(t, eligible=None):
                g = t.groupby("sub")
                out = g[other].mean()
                Cs = (pd.DataFrame(counts[t.index], index=t.index).groupby(t["sub"]).sum()
                      .loc[out.index].to_numpy(float).reshape(len(out), K, K))
                n_out = Cs.sum(2, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    P = np.where(n_out > 0, Cs / np.where(n_out > 0, n_out, 1), np.nan)
                if eligible is not None:
                    P[~eligible.reindex(out.index).fillna(False).to_numpy(bool)] = np.nan
                with np.errstate(invalid="ignore", divide="ignore"):
                    H = -np.where(P > 0, P * np.log2(P), 0).sum(2)
                H[np.isnan(P).any(2)] = np.nan                               # H(p(. | i)) per state i
                out[self.cells] = P.reshape(len(out), -1)
                out[self.row_h] = H
                res = out[[*self.cells, *self.occ, *self.row_h, *[v for v in other if v not in self.occ]]]
                return res.assign(scan_s=g["scan_s"].sum(min_count=1)), pd.DataFrame(n_out[..., 0], index=out.index)

            _, total = per_sub(d)
            eligible = total >= m_out                                          # people x states
            self.pooled[label], _ = per_sub(d, eligible)
            for kind, t in d.groupby("_kind"):
                data[label, kind], _ = per_sub(t, eligible)
                for task, tt in t.groupby(t["task"].astype(str)):              # and one table per task
                    self.by_task[label, kind, task], _ = per_sub(tt, eligible)
        self.units = pd.concat(units, ignore_index=True)
        return data

    @property
    def occ(self):
        return [f"occ_{k}" for k in range(self.cfg.K)]

    @property
    def row_h(self):
        """H(p(. | i)) per state i, in bits: the entropy rate is their occupancy-weighted mean."""
        return [f"H_{k}" for k in range(self.cfg.K)]

    @property
    def dyn(self):
        return [v for v in self.cfg.dynamics if all(v in d for d in self.data.values())]

    @property
    def variables(self):
        return [*self.cells, *self.occ, *self.dyn]

    def role(self, label):
        return self.cfg.cohorts[label]["role"]

    def tasks(self):
        """What is read, from where, and as which kind: one row per (cohort, directory, task)."""
        rows = []
        for label in self.cfg.cohorts:
            for kind, files in self._files(label).items():
                for f in files:
                    rows.append(dict(cohort=label, dir=ht.hive(f, "cohort"), task=ht.hive(f, "task"), kind=kind,
                                     subjects=pd.read_parquet(f, columns=["sub"])["sub"].nunique()))
        t = pd.DataFrame(rows)
        return t.groupby(["cohort", "dir", "kind", "task"])["subjects"].sum().to_frame()

    def counts(self):
        """Subjects per cohort and task."""
        return pd.DataFrame({k: len(v) for k, v in self.data.items()}, index=["n"]).T.unstack().droplevel(0, axis=1)

    def movie(self):
        """Every cohort's movie subjects, one table, with a cohort column."""
        return pd.concat([self.data[c, "movie"].assign(cohort=c) for c in self.cfg.cohorts if (c, "movie") in self.data])

    # -- figure 1 -------------------------------------------------------------------------------
    def _graph(self, means):
        K = self.cfg.K
        T = np.nan_to_num(means[self.cells].to_numpy(float).reshape(K, K))   # a row nobody left: 0
        T /= np.where(T.sum(1, keepdims=True) > 0, T.sum(1, keepdims=True), 1)
        return T, means[self.occ].to_numpy(float)

    def graphs(self):
        """{(cohort | "mean", rest|movie): (p(j | i), occupancy)}; the mean weighs every cohort once."""
        out = {k: self._graph(v.mean()) for k, v in self.data.items()}
        for kind in ("rest", "movie"):
            m = [v.mean() for (c, k), v in self.data.items() if k == kind]
            if m:
                out["mean", kind] = self._graph(pd.concat(m, axis=1).mean(axis=1))
        return out

    def task_change(self):
        """Movie vs rest per cohort (and all cohorts together, "mean"): paired in the people with both
        scans (Welch when fewer than 3 have both), BH-FDR over every variable. Returns (marks, dynamics):
        marks = one row per variable with its test; dynamics = the values beside each diagram."""
        c = self.cfg
        marks, dyn = [], []
        labels = [x for x in c.cohorts if (x, "rest") in self.data and (x, "movie") in self.data]
        for label in [*labels, "mean"]:
            group = labels if label == "mean" else [label]
            r = pd.concat([self.data[x, "rest"].rename(index=lambda s, x=x: f"{x}:{s}") for x in group])
            m = pd.concat([self.data[x, "movie"].rename(index=lambda s, x=x: f"{x}:{s}") for x in group])
            v = self.variables
            both = r.index.intersection(m.index)
            if len(both) >= 3:
                p, test = stats.ttest_rel(m.loc[both, v], r.loc[both, v]).pvalue, "paired"
            else:
                p, test = stats.ttest_ind(m[v], r[v], equal_var=False).pvalue, "Welch"
            q = stats.false_discovery_control(np.nan_to_num(p, nan=1))
            fc = np.log2((m[v].mean() + c.shrink) / (r[v].mean() + c.shrink))
            line = pd.Series(c.fc_task, v)
            line[self.dyn] = c.fc_dyn
            t = pd.DataFrame(dict(cohort=label, variable=v, x=fc.to_numpy(), p=p, p_fdr=q, test=test,
                                  n=len(both) if test == "paired" else min(len(m), len(r)),
                                  rest=r[v].mean().to_numpy(), movie=m[v].mean().to_numpy()))
            t["hit"] = (t["p_fdr"] < c.alpha) & (t["x"].abs() >= line.to_numpy())
            marks.append(t)
            for kind, d in (("rest", r), ("movie", m)):
                dd = t[t["variable"].isin(self.dyn)]
                dyn.append(pd.DataFrame(dict(cohort=label, task=kind, variable=dd["variable"],
                                             value=dd[kind].to_numpy(),
                                             change=(dd["movie"] / dd["rest"] - 1).to_numpy() if kind == "movie" else np.nan,
                                             hit=dd["hit"].to_numpy() & (kind == "movie"))))
        return pd.concat(marks, ignore_index=True), pd.concat(dyn, ignore_index=True)

    # -- figures 2 and 4: equivalence ----------------------------------------------------------
    def groups(self, by="cohort", tasks="kind", cohort=None):
        """{label: one row per subject} -- what figures 2 and 4 compare.
        by="cohort": each cohort's movie (the default); tasks="all": each person over all their
        scans, every task (rest included) pooled -- transitions from the summed counts.
        by="task", tasks="kind": rest vs movie; tasks="film": every movie task (film, episode) apart.
        cohort = one cohort label to stay within; None pools every cohort (a film is then "cohort:film").
        Rest vs movie are often the same people: the tests treat the groups as independent, which is
        conservative for the difference and makes equivalence harder to show, not easier."""
        if by == "cohort":
            if tasks == "all" and getattr(self, "pooled", None):   # each person over ALL their scans
                return dict(self.pooled)
            if tasks == "all":                            # (toy objects without `pooled`)
                out = {}
                for c in self.cfg.cohorts:
                    parts = [self.data[c, k] for k in ("rest", "movie") if (c, k) in self.data]
                    if parts:
                        out[c] = pd.concat(parts).groupby(level=0).mean(numeric_only=True)
                return out
            return {c: self.data[c, "movie"] for c in self.cfg.cohorts if (c, "movie") in self.data}
        if by != "task":
            raise SystemExit(f'by must be "cohort" or "task", not {by!r}')
        labels = [cohort] if cohort else list(self.cfg.cohorts)
        tag = lambda c, s: s if cohort else f"{c}:{s}"
        if tasks == "kind":
            out = {}
            for kind in ("rest", "movie"):
                parts = [self.data[c, kind].rename(index=lambda x, c=c: f"{c}:{x}") for c in labels if (c, kind) in self.data]
                if parts:
                    out[kind] = pd.concat(parts)
            return out
        if tasks == "film":
            return {tag(c, t): d for (c, k, t), d in sorted(self.by_task.items()) if k == "movie" and c in labels}
        raise SystemExit(f'tasks must be "kind" or "film", not {tasks!r}')

    def _long(self, groups):
        """The groups as one table with a `cohort` column holding the group label."""
        groups = groups if groups is not None else self.groups()
        return pd.concat([d.assign(cohort=g) for g, d in groups.items()])

    def agreement(self, variables=None, groups=None):
        """Each group vs all the others pooled, per variable: Welch difference, its 90% CI, and TOST
        at 5% with bounds +-sesoi_d x the pooled SD. groups: from groups(); None = the cohorts' movie."""
        mv, rows = self._long(groups), []
        for v in variables or self.variables:
            bound = self.cfg.sesoi_d * mv[v].std()
            for label in mv["cohort"].unique():
                a, b = mv.loc[mv["cohort"] == label, v].dropna(), mv.loc[mv["cohort"] != label, v].dropna()
                if len(a) < 2 or len(b) < 2:
                    rows.append(dict(variable=v, cohort=label, n=len(a), diff=np.nan, lo=np.nan, hi=np.nan,
                                     bound=bound, equivalent=False, mean=a.mean() if len(a) else np.nan,
                                     mean_lo=np.nan, mean_hi=np.nan, others=b.mean() if len(b) else np.nan,
                                     p=np.nan))
                    continue
                va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
                se = np.sqrt(va + vb)
                df = (va + vb) ** 2 / (va ** 2 / max(len(a) - 1, 1) + vb ** 2 / max(len(b) - 1, 1))
                d, h = a.mean() - b.mean(), stats.t.ppf(.95, df) * se
                ha = stats.t.ppf(.95, max(len(a) - 1, 1)) * a.std(ddof=1) / np.sqrt(len(a))
                rows.append(dict(variable=v, cohort=label, n=len(a), diff=d, lo=d - h, hi=d + h, bound=bound,
                                 equivalent=bool((d - h > -bound) & (d + h < bound)),
                                 mean=a.mean(), mean_lo=a.mean() - ha, mean_hi=a.mean() + ha, others=b.mean(),
                                 p=stats.ttest_ind(a, b, equal_var=False).pvalue))
        out = pd.DataFrame(rows)
        out["p_fdr"] = stats.false_discovery_control(out["p"].fillna(1))     # the difference, BH over everything
        out["differs"] = out["p_fdr"] < self.cfg.alpha
        return out

    def cohen(self, variables=None, groups=None):
        """Each group vs all the others pooled, per variable: Cohen's d (pooled SD) and its 95% CI."""
        mv, rows = self._long(groups), []
        for v in variables or self.variables:
            for label in mv["cohort"].unique():
                a, b = mv.loc[mv["cohort"] == label, v].dropna(), mv.loc[mv["cohort"] != label, v].dropna()
                na, nb = len(a), len(b)
                if na < 2 or nb < 2:                  # e.g. p(j | i) of a state no one here leaves
                    rows.append(dict(variable=v, cohort=label, n=na, d=np.nan, lo=np.nan, hi=np.nan))
                    continue
                sd = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / max(na + nb - 2, 1))
                d = (a.mean() - b.mean()) / sd if sd > 0 else 0.0
                se = np.sqrt(1 / na + 1 / nb + d ** 2 / (2 * (na + nb)))
                rows.append(dict(variable=v, cohort=label, n=na, d=d, lo=d - 1.96 * se, hi=d + 1.96 * se))
        return pd.DataFrame(rows)

    def strongest(self, n=None):
        """The movie mean's `n` strongest transitions (figure 2)."""
        m = pd.concat([self.data[c, "movie"].mean() for c in self.cfg.cohorts if (c, "movie") in self.data], axis=1).mean(axis=1)
        return sorted(self.cells, key=lambda c: -m[c])[:n or self.cfg.top_transitions]

    # -- figure 3 -----------------------------------------------------------------------------
    def cohort_means(self):
        """Every variable's movie mean, one row per cohort, with its role."""
        mv = self.movie()
        out = mv.groupby("cohort")[self.variables].mean()
        return out.assign(role=[self.role(c) for c in out.index])


def latent_options(co, atlas=None, window_s=None):
    """What stage 4 wrote: every (atlas, window_s) with latents, and the embeddings and state sets
    (cluster columns) in them. Narrow with atlas / window_s."""
    import pyarrow.parquet as pq
    rows = {}
    for f in tree_root(co.cfg.root, "latents",
                       co.cfg.modelling_version).rglob("*.parquet"):
        a, w = ht.hive(f, "atlas"), ht.hive(f, "window_s")
        if (atlas and a != atlas) or (window_s is not None and float(w) != float(window_s)):
            continue
        names = pq.read_schema(f).names
        emb = sorted({n.split("/")[0][:-1] + n.split("/")[1] for n in names if "/" in n and n[0].isalpha()})
        states = sorted(n for n in names if re.fullmatch(r"[A-Za-z0-9]+_[a-z]+\d+_\d+", n))
        r = rows.setdefault((a, float(w)), dict(atlas=a, window_s=float(w), embeddings=set(), states=set()))
        r["embeddings"] |= set(emb); r["states"] |= set(states)
    t = pd.DataFrame(rows.values())
    if len(t):
        t["embeddings"] = t["embeddings"].map(sorted); t["states"] = t["states"].map(sorted)
    return t


def load_latents(co, embedding="umap3", max_windows=150, seed=0, atlas=None, window_s=None, states=None):
    """Every cohort's windows in the latent space (stage 4, latents/atlas=/window_s=/cohort=/data.parquet):
    sub, task, the embedding's 3 coordinates and the state column `states` when the latents carry it.
    atlas / window_s / states: any that stage 4 wrote (latent_options lists them); None = cfg.space's.
    At most `max_windows` windows per (cohort, subject), drawn at random, so a cohort with long scans
    (cneuromod) does not bury the others."""
    import pyarrow.parquet as pq
    c = co.cfg
    atlas = atlas or c.space["atlas"]
    window_s = c.space["window_s"] if window_s is None else window_s
    states = states or c.space["states"]
    xyz = [f"{embedding[:-1]}{k}/{embedding[-1]}" for k in range(3)]
    base = latents_root(c.root, atlas, window_s,
                        version=c.modelling_version)
    frames = []
    for label, spec in c.cohorts.items():
        for d in spec["dirs"]:
            for f in sorted((base / f"cohort={d}").rglob("*.parquet")):
                have = set(pq.read_schema(f).names)
                if not set(xyz) <= have:
                    raise SystemExit(f"{f} has no {embedding} columns ({xyz}); was it fitted?")
                cols = ["sub", "task", *xyz, *([states] if states in have else [])]
                frames.append(pd.read_parquet(f, columns=cols).assign(cohort=label))
    if not frames:
        raise SystemExit(f"no latents under {base}; latent_options(co) lists what exists")
    lat = pd.concat(frames, ignore_index=True)
    lat["sub"] = lat["sub"].astype(str).str.strip().str.upper()
    lat["kind"] = np.where(lat["task"].astype(str).str.fullmatch(c.rest_pattern, case=False), "rest", "movie")
    lat = lat.sample(frac=1, random_state=seed)                   # shuffled, then the first max_windows each
    lat = lat[lat.groupby(["cohort", "sub"]).cumcount() < max_windows].sort_index()
    lat = lat.rename(columns=dict(zip(xyz, ["x", "y", "z"]))).reset_index(drop=True)
    lat.attrs.update(atlas=atlas, window_s=float(window_s), states=states, embedding=embedding)
    return lat


def available_sets(root, cohorts, modelling_version=None):
    """Every state set with a table for every cohort label (any of its dirs)."""
    found = {}
    for p in tree_root(root, "transitions",
                       modelling_version).rglob("subjects.parquet"):
        key = (ht.hive(p, "atlas"), float(ht.hive(p, "window_s")), ht.hive(p, "states"))
        found.setdefault(key, set()).add(ht.hive(p, "cohort"))
    rows = [dict(atlas=a, window_s=w, states=s) for (a, w, s), have in sorted(found.items())
            if all(set(c["dirs"]) & have for c in cohorts.values())]
    return pd.DataFrame(rows)



# -- the mixed models, computed once (tools/mixed_model_sweep.py, slurm/mixed_model.sbatch) -----
MIXED_SETTINGS = ("entropy_estimator", "distinct_minutes", "min_out_transitions", "rest_pattern", "film_merge")


def mixed_dir(cfg: Settings) -> Path:
    """<outputs>/modelling/version=<v>/unsupervised_model/mixed_model/atlas=/window_s=/states=."""
    sp = cfg.space
    return (unsupervised_root(cfg.root, cfg.modelling_version) / "mixed_model" / f"atlas={_key(sp['atlas'])}"
            / f"window_s={_key(_fmt_window(sp['window_s']))}" / f"states={_key(sp['states'])}")


def _mixed_meta(co) -> dict:
    return {"cohorts": sorted(co.cfg.cohorts), **{k: str(getattr(co.cfg, k)) for k in MIXED_SETTINGS}}


def mixed_tables(co, n_boot=1, recompute=False, save=True, min_occ=0.01, fit_missing=True):
    """(properties, transitions): task_variation(combine="mixed") for the transition structure,
    occupancy profile, dynamic metrics and occupancies, and for the transition structure and every
    transition p(j | i). Read from `mixed_dir` when there (written by tools/mixed_model_sweep.py /
    slurm/mixed_model.sbatch); a cache made with other settings is still read, with a warning naming
    what differs (recompute=True refits). When absent: fitted here and written there, or, with
    fit_missing=False, an error saying to run the sbatch."""
    import json
    import warnings
    d = mixed_dir(co.cfg)
    files = d / "vpc_properties.csv", d / "vpc_transitions.csv"
    meta = _mixed_meta(co)
    if not recompute and all(f.exists() for f in files):
        stored = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        diff = {k: (stored.get(k), v) for k, v in meta.items() if stored.get(k) != v}
        if diff:
            warnings.warn(f"{d}: made with other settings ({diff}, stored vs now); read anyway -- "
                          f"recompute=True to refit", stacklevel=2)
        out = []
        for f in files:
            t = pd.read_csv(f)
            t.attrs.update(scale="omega2", measure="vpc")
            out.append(t)
        print(f"mixed models read from {d}")
        return tuple(out)
    if not recompute and not fit_missing:
        raise SystemExit(f"no mixed models in {d}: run `sbatch slurm/mixed_model.sbatch -- --atlas "
                         f"{co.cfg.space['atlas']} --window-s {co.cfg.space['window_s']:g} --states "
                         f"'{co.cfg.space['states']}'`, or call mixed_tables(co, fit_missing=True) to fit here")
    props = co.task_variation(["transition_fit", "occupancy_fit", "dynamics_fit", *co.dyn, *co.occ], n_boot=n_boot,
                              combine="mixed")
    cells = co.task_variation(["transition_fit", *co.cells], n_boot=n_boot, combine="mixed")
    if save:
        d.mkdir(parents=True, exist_ok=True)
        props.to_csv(files[0], index=False)
        cells.to_csv(files[1], index=False)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return props, cells
