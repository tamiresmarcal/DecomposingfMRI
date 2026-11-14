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

from fmri_decomposition.io import latents_root, transitions_root, tree_root
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
    # figure 2a's x ranges: probabilities 0-1; entropy 0-log2(K) bits and distinct 0-K*K are set from K
    ranges: dict = field(default_factory=lambda: {"switch_rate": (0, 1), "mean_dwell_s": (0, 60), "dispersion": (0, 10)})
    save_dir: Path | None = None
    # None = entropy_rate_bits as stored; "plugin" / "miller_madow" =
    # recomputed from each table's counts (see hads.Settings.entropy_estimator).
    entropy_estimator: str | None = None

    @property
    def K(self):
        return ht.n_states(self.space["states"])


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

    def transition_fit(self, min_occ=0.01):
        """Per subject and scan kind: Pearson r between the subject's p(j | i) and the reference
        p(j | i) (the mean over every cohort and both kinds, each cohort once), on the rows of the
        states the reference occupies >= min_occ. One number for 'does this person move between
        states the way people do' -- the transition structure as a per-subject variable."""
        K = self.cfg.K
        ref = pd.concat([v[self.cells].mean() for v in self.data.values()], axis=1).mean(axis=1)
        occ = pd.concat([v[self.occ].mean() for v in self.data.values()], axis=1).mean(axis=1).to_numpy()
        rows = np.repeat(occ >= min_occ, K)
        R = ref.to_numpy(float)[rows]
        for key, d in [*self.data.items(), *getattr(self, "by_task", {}).items()]:
            X = d[self.cells].to_numpy(float)[:, rows]
            Xc, Rc = X - X.mean(1, keepdims=True), R - R.mean()
            with np.errstate(invalid="ignore", divide="ignore"):
                d["transition_fit"] = (Xc @ Rc) / np.sqrt((Xc ** 2).sum(1) * (Rc ** 2).sum())
        return "transition_fit"

    def stability_variables(self):
        fit = self.transition_fit()
        return [fit, *self.dyn, *self.occ, *self.strongest()]

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
        rng = np.random.default_rng(seed)
        coh, agr = self.cohen(vs), self.agreement(vs)
        r, m = self.paired_rest_movie()
        demean = lambda t, v: t[v] - t.groupby("cohort")[v].transform("mean")
        n = len(r)
        boot = rng.integers(0, n, size=(n_boot, n)) if n_boot else np.empty((0, n), int)
        rows = []
        for v in vs:
            cv = coh[coh["variable"] == v]
            i = cv["d"].abs().idxmax()
            rv, mv_ = r[v].to_numpy(float), m[v].to_numpy(float)
            ok = np.isfinite(rv) & np.isfinite(mv_)
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
                       task_d=d_av, task_lo=np.nanpercentile(bd, 2.5) if n_boot else np.nan,
                       task_hi=np.nanpercentile(bd, 97.5) if n_boot else np.nan,
                       task_equivalent=bool((diff.mean() - h > -bound) and (diff.mean() + h < bound)),
                       task_p=stats.ttest_rel(mv_[ok], rv[ok]).pvalue, n_paired=int(ok.sum()),
                       icc=icc, icc_lo=np.nanpercentile(bi, 2.5) if n_boot else np.nan,
                       icc_hi=np.nanpercentile(bi, 97.5) if n_boot else np.nan)
            for c in r["cohort"].unique():
                s = (r["cohort"] == c).to_numpy()
                row[f"icc_{c}"] = _icc_c1(rv[s], mv_[s])
            rows.append(row)
        out = pd.DataFrame(rows)
        out["task_p_fdr"] = stats.false_discovery_control(out["task_p"].fillna(1))
        out["stable"] = out["between_equivalent"] & out["task_equivalent"]
        a = out["task_d"].abs()
        out["tier"] = np.select([a < .2, a < .5], ["stable", "moderate"], "context-dependent")
        return out

    def replication(self, kind="movie", n_split=20, min_occ=0.01, seed=0):
        """Does the transition STRUCTURE replicate across cohorts, whatever their levels?

        r between cohorts' mean p(j | i) (rows of states occupied >= min_occ), divided by the
        noise ceiling: the r between two random halves of the same cohort (mean of n_split
        splits), geometric mean of the two cohorts' ceilings. ~1 = two cohorts agree as well as a
        cohort agrees with itself. Returns (pairs, ceilings)."""
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
    def films(self):
        """{(cohort, film): one row per subject} for every movie task (film, episode, run)."""
        return {(c, t): d for (c, k, t), d in sorted(self.by_task.items()) if k == "movie"}

    def film_summary(self, variables=None):
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
        F = self.films()
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
                elif pooled is not None:
                    y = pooled[v].dropna()
                    va, vb = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
                    se = np.sqrt(va + vb)
                    df = (va + vb) ** 2 / (va ** 2 / max(len(x) - 1, 1) + vb ** 2 / max(len(y) - 1, 1))
                    diff = x.mean() - y.mean()
                    hh = stats.t.ppf(.95, df) * se
                    eq = bool((diff - hh > -bound) and (diff + hh < bound))
                    pv = stats.ttest_ind(x, y, equal_var=False).pvalue
                rows.append(dict(cohort=c, film=t, variable=v, n=n, mean=x.mean(), lo=x.mean() - h,
                                 hi=x.mean() + h, within=within, diff=diff, equivalent=eq, p=pv))
        out = pd.DataFrame(rows)
        ok = out["p"].notna()
        out["p_fdr"] = np.nan
        out.loc[ok, "p_fdr"] = stats.false_discovery_control(out.loc[ok, "p"])
        out["differs"] = out["p_fdr"] < self.cfg.alpha
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

    def occupancy_table(self):
        """Mean time in each state (%), one row per (cohort, rest|movie)."""
        return pd.DataFrame({f"{c} {k}": 100 * v[self.occ].mean().to_numpy() for (c, k), v in self.data.items()},
                            index=[f"S{k}" for k in range(self.cfg.K)]).T


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
        for label in self.cfg.cohorts:
            for kind, files in self._files(label).items():
                if not files:
                    continue
                d = pd.concat(map(pd.read_parquet, files), ignore_index=True)
                d["sub"] = d["sub"].astype(str).str.strip().str.upper()
                if self.cfg.entropy_estimator is not None and "entropy_rate_bits" in d:
                    # before as_p_cond: the recompute needs the p_joint cells
                    from fmri_decomposition.unsupervised_model.transitions import entropy_from_table
                    d["entropy_rate_bits"] = entropy_from_table(d, K, self.cfg.entropy_estimator)
                self.cells = ht.cell_cols(d)
                d[self.cells] = ht.as_p_cond(d, K)
                P = d[self.cells].to_numpy(float).reshape(len(d), K, K)        # each subject's row entropies
                with np.errstate(divide="ignore", invalid="ignore"):
                    H = -np.where(P > 0, P * np.log2(P), 0).sum(2)
                H[P.sum(2) == 0] = np.nan                                      # a state the subject never left
                d[self.row_h] = H
                cols = [*self.cells, *self.occ, *self.row_h, *[v for v in self.cfg.dynamics if v in d]]
                # the scan time used: windows x their stride (provenance written by `transitions`), summed
                d["scan_s"] = (d["n_windows"] * d["stride_s"]).astype(float) if {"n_windows", "stride_s"} <= set(d) else np.nan
                per_sub = lambda t: t.groupby("sub")[cols].mean().assign(scan_s=t.groupby("sub")["scan_s"].sum(min_count=1))
                data[label, kind] = per_sub(d)
                for task, t in d.groupby(d["task"].astype(str)):              # and one table per task (film)
                    self.by_task[label, kind, task] = per_sub(t)
        if not data:
            raise SystemExit(f"no transition tables under {self.base}")
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
        T = means[self.cells].to_numpy(float).reshape(K, K).copy()
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
        by="cohort": each cohort's movie (the default).
        by="task", tasks="kind": rest vs movie; tasks="film": every movie task (film, episode) apart.
        cohort = one cohort label to stay within; None pools every cohort (a film is then "cohort:film").
        Rest vs movie are often the same people: the tests treat the groups as independent, which is
        conservative for the difference and makes equivalence harder to show, not easier."""
        if by == "cohort":
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
