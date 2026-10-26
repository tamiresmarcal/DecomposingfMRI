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

import hads_tables as ht

DYNAMICS = ["switch_rate", "mean_dwell_s", "entropy_rate_bits", "dispersion", "n_distinct_transitions"]


@dataclass
class Settings:
    root: Path = Path(os.environ.get("FMRIDECOMP_OUTPUTS", "/project/6008063/tamires/DecomposingfMRI/outputs"))
    # one state set, as in notebook 10's SPACE
    space: dict = field(default_factory=lambda: dict(atlas="yeo7", window_s=30.0, states="HMM1_pca3_8"))
    # label -> the cohort= directories it reads, and its role in the state fit (train / projected)
    cohorts: dict = field(default_factory=lambda: {
        "ds002837": dict(dirs=["ds002837"], role="train"),
        "cneuromod": dict(dirs=["cneuromod"], role="train"),
        "camcan": dict(dirs=["camcan", "camcan_rest"], role="projected")})
    rest_pattern: str = r"rest"                       # a task= value that IS this (whole name, any case) is rest;
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

    @property
    def K(self):
        return ht.n_states(self.space["states"])


class Cohorts:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or Settings()
        self.base = (self.cfg.root / "transitions" / f"atlas={self.cfg.space['atlas']}"
                     / f"window_s={ht.fmt_window(self.cfg.space['window_s'])}" / f"states={self.cfg.space['states']}")
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


def load_latents(co, embedding="umap3", max_windows=150, seed=0):
    """Every cohort's windows in the latent space (stage 4, latents/atlas=/window_s=/cohort=/data.parquet):
    sub, task, the embedding's 3 coordinates and, when the latents carry it, the state of cfg.space.
    At most `max_windows` windows per (cohort, subject), drawn at random, so a cohort with long scans
    (cneuromod) does not bury the others."""
    import pyarrow.parquet as pq
    c = co.cfg
    xyz = [f"{embedding[:-1]}{k}/{embedding[-1]}" for k in range(3)]
    base = c.root / "latents" / f"atlas={c.space['atlas']}" / f"window_s={ht.fmt_window(c.space['window_s'])}"
    frames = []
    for label, spec in c.cohorts.items():
        for d in spec["dirs"]:
            for f in sorted((base / f"cohort={d}").rglob("*.parquet")):
                have = set(pq.read_schema(f).names)
                if not set(xyz) <= have:
                    raise SystemExit(f"{f} has no {embedding} columns ({xyz}); was it fitted?")
                cols = ["sub", "task", *xyz, *([c.space["states"]] if c.space["states"] in have else [])]
                frames.append(pd.read_parquet(f, columns=cols).assign(cohort=label))
    if not frames:
        raise SystemExit(f"no latents under {base}")
    lat = pd.concat(frames, ignore_index=True)
    lat["sub"] = lat["sub"].astype(str).str.strip().str.upper()
    lat["kind"] = np.where(lat["task"].astype(str).str.fullmatch(c.rest_pattern, case=False), "rest", "movie")
    lat = lat.sample(frac=1, random_state=seed)                   # shuffled, then the first max_windows each
    lat = lat[lat.groupby(["cohort", "sub"]).cumcount() < max_windows].sort_index()
    return lat.rename(columns=dict(zip(xyz, ["x", "y", "z"]))).reset_index(drop=True)


def available_sets(root, cohorts):
    """Every state set with a table for every cohort label (any of its dirs)."""
    found = {}
    for p in (Path(root) / "transitions").rglob("subjects.parquet"):
        key = (ht.hive(p, "atlas"), float(ht.hive(p, "window_s")), ht.hive(p, "states"))
        found.setdefault(key, set()).add(ht.hive(p, "cohort"))
    rows = [dict(atlas=a, window_s=w, states=s) for (a, w, s), have in sorted(found.items())
            if all(set(c["dirs"]) & have for c in cohorts.values())]
    return pd.DataFrame(rows)
