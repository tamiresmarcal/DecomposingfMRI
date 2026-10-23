"""Tables behind 10_hads_report.ipynb: brain-state transitions vs Cam-CAN HADS.

Everything the report computes is here; the notebook only chooses and draws. `Study(cfg)` reads
the selection rankings, the phenotype and the `transitions/` tables, and each method returns one
table. The slow ones (baselines, both multiverses, the SHAP of many models, the deep dive) are
cached in `Settings.cache_dir`, keyed by what they are built from, so a re-run reads instead of
refitting. Delete the cache folder to force a rebuild. They can be built on the cluster first:

    python notebooks/hads_tables.py
    python notebooks/hads_tables.py --space networks,30,HMM1_pca3_8 --model hgb --arm "all [cond]"

The paths come from $FMRIDECOMP_OUTPUTS and $CAMCAN_PHENO, as for the notebooks.

The statistical rule is the same everywhere: an effect passes when its BH-FDR < `alpha` AND
|log2 fold change| >= `fc_line`. The fold change is log2((mean b + c) / (mean a + c)), with the
pseudocount c = `shrink` x the block's average for probabilities (cells, occupancy) and
`shrink` x the variable's own mean for the dynamics.
"""
from __future__ import annotations

import functools
import hashlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                          # fmri_decomposition, for the refits
    sys.path.insert(0, str(REPO))

# What each select-bstm arm hands the model, as in bstm_selection.build.
DYNAMICS = ["n_states_visited", "n_distinct_transitions", "switch_rate", "self_transition_rate",
            "switches_per_min", "mean_dwell_s", "entropy_rate_bits", "dispersion"]
ARM_BLOCKS = {"cells": ["cells"], "occupancy": ["occupancy"], "dynamics": ["dynamics"],
              "summary": ["occupancy", "dynamics"], "all": ["cells", "occupancy", "dynamics"]}
PROBABILITY = {"cells", "occupancy"}                   # blocks that take MIN_PRESENT and the block pseudocount
GRAPH = {"cells", "occupancy"}                         # blocks a state diagram can draw
ALL = "all [cond]"                                     # the chosen space's variables: p(j | i), occupancy, dynamics


def _cores():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


@dataclass
class Settings:
    # data
    root: Path = Path(os.environ.get("FMRIDECOMP_OUTPUTS", "/project/6008063/tamires/DecomposingfMRI/outputs"))
    camcan: Path = Path(os.environ.get("CAMCAN_PHENO", "/project/6008063/tamires/cohorts/camcan/dataman/"
                                       "useraccess/opendata/paule_toussaint_camcan01870"))
    pheno_id: str = "CCID"
    targets: dict = field(default_factory=lambda: {"anxiety": "additional_HADS_anx_category",
                                                   "depression": "additional_HADS_dep_category"})
    ordinal_levels: list = field(default_factory=lambda: ["Normal", "Mild", "Moderate", "Severe"])
    tasks: dict = field(default_factory=lambda: {"rest": "rest", "movie": "movie"})   # first = the base
    cohorts: list = field(default_factory=lambda: ["camcan", "camcan_rest"])
    covariates: list = field(default_factory=lambda: ["Age", "Sex", "n_transitions"])
    categorical: list = field(default_factory=lambda: ["Sex"])
    # the statistical rule
    case_from: str = "Mild"
    alpha: float = 0.10
    fc_line: float = 0.25
    shrink: float = 1.0
    min_present: float = 0.10
    # A: the multiverses
    multiverse_arm: str = "all [cond]"
    # B: performance
    models: list | None = None                         # None = every model but logistic
    arms: list | None = None                           # None = every arm
    baseline_covariates: list = field(default_factory=lambda: ["Age", "Sex"])
    baseline_arm: str = "(covariates only)"
    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    margin_sd: float = 0
    # C: SHAP shares
    share_arms: list = field(default_factory=lambda: ["all [joint]", "all [cond]"])
    share_top: int | None = 10
    # D / E: the choices
    graph_arms: list = field(default_factory=lambda: ["all [joint]", "all [cond]"])
    shap_seed: int = 0
    seeds_stab: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    highlight_top: int = 8
    stable_min: float = 0.8
    min_d: float = 0.2
    label_pp: int | None = None
    # drawing
    xlim: float | None = 2
    label_top: int = 3
    label_max: int = 25
    highlight_max: int = 10
    factors: dict = field(default_factory=lambda: {"window": "window size", "atlas": "atlas", "model": "model",
                                                   "method": "state method", "arm": "arm"})
    factor_trees: list = field(default_factory=lambda: ["bstm"])
    top_vars: int = 15
    edge_top: int = 12
    save_dir: Path | None = None
    # running
    cache_dir: Path | None = Path(__file__).resolve().parent / "cache"
    n_jobs: int = field(default_factory=_cores)

    @property
    def case(self):
        return self.ordinal_levels.index(self.case_from)

    @property
    def pass_text(self):
        return f"FDR<{self.alpha:g}" + (f" & |log2FC|≥{self.fc_line:g}" if self.fc_line else "")


# ------------------------------------------------------------------ small helpers ---
def hive(path, key):
    """One `key=value` partition from a path, by name."""
    return next((p.split("=", 1)[1] for p in Path(path).parts if p.startswith(f"{key}=")), None)

def fmt_window(w):
    return str(int(w)) if float(w).is_integer() else str(w)

def window_name(w):
    return None if pd.isna(w) else "1 TR" if float(w) < 0 else f"{float(w):g}s"

def set_label(r):
    """A readable name for one state set (or the atlas of a static-FC row)."""
    if r.get("tree") == "fcm":
        return r["atlas"]
    return f"{r['states']} · {r['atlas']} · {window_name(r['window_s'])}"

def n_states(states):
    return int(str(states).rsplit("_", 1)[1])

def space_of(r):
    return dict(atlas=r["atlas"], window_s=float(r["window_s"]), states=r["states"])

def cell_cols(d):
    return [c for c in d.columns if "->" in c]

def as_p_cond(d, K):
    """p_joint -> Markov p(j | i): back to counts, then renormalise each row."""
    cols = cell_cols(d)
    M = d[cols].to_numpy(float).reshape(len(d), K, K) * d["n_transitions"].to_numpy(float)[:, None, None]
    s = M.sum(2, keepdims=True)
    return pd.DataFrame(np.divide(M, s, out=np.zeros_like(M), where=s > 0).reshape(len(d), -1),
                        columns=cols, index=d.index)

def conditional(d, K):
    """A copy of a transition table with the cells as p(j | i)."""
    d = d.copy()
    d[cell_cols(d)] = as_p_cond(d, K)
    return d

def arm_columns(t, arm):
    """{block: its columns in table t} for one arm, e.g. "all [cond]" -> cells, occupancy, dynamics."""
    cols = {"cells": cell_cols(t), "occupancy": [c for c in t.columns if c.startswith("occ_")],
            "dynamics": [c for c in DYNAMICS if c in t.columns]}
    return {b: cols[b] for b in ARM_BLOCKS[arm.split(" [")[0]]}

def design(cov, categorical, intercept=True):
    """Covariates -> numeric matrix: categoricals one-hot, gaps mean-filled."""
    X = pd.get_dummies(cov, columns=[c for c in categorical if c in cov], drop_first=True).astype(float)
    X = X.fillna(X.mean()).to_numpy()
    return np.column_stack([np.ones(len(X)), X]) if intercept else X

def direction(X, S):
    """Spearman of each variable's value with its SHAP: > 0 = higher values push the prediction up."""
    rx = np.apply_along_axis(stats.rankdata, 0, X)
    rs = np.apply_along_axis(stats.rankdata, 0, S)
    rx, rs = rx - rx.mean(0), rs - rs.mean(0)
    den = np.sqrt((rx ** 2).sum(0) * (rs ** 2).sum(0))
    return np.divide((rx * rs).sum(0), den, out=np.zeros(X.shape[1]), where=den > 0)

def pp(change):
    """A change in probability as a pp label; '' when it rounds to 0."""
    return f"{100 * change:+.0f} pp" if round(100 * change) else ""

def as_choice(row, keys):
    """One options row as the dict to paste into the choice cell."""
    return "dict(" + ", ".join(f"{k}={row[k]!r}" for k in keys) + ")"


# ------------------------------------------------------------------- the tests ---
def group_effects(A, g, cov, c, categorical):
    """Group (g = 1) vs the rest (g = 0) on every column of A, adjusted for cov by regression.
    x = log2 fold change of the adjusted group mean over the reference mean, p = t-test of g."""
    X = np.column_stack([g, design(cov, categorical)])
    B = np.linalg.lstsq(X, A, rcond=None)[0]
    df = len(A) - X.shape[1]
    se = np.sqrt(((A - X @ B) ** 2).sum(0) / df * np.linalg.pinv(X.T @ X)[0, 0])
    p = 2 * stats.t.sf(np.abs(B[0] / se), df)
    mu0 = A[g == 0].mean(0)
    mu1 = np.clip(mu0 + B[0], 0, None)                 # a mean cannot go below 0
    return np.log2((mu1 + c + 1e-12) / (mu0 + c + 1e-12)), p

def paired_effects(A, B, c):
    """B vs A in the same people (rows aligned): log2 fold change of the means, paired t-test."""
    x = np.log2((B.mean(0) + c + 1e-12) / (A.mean(0) + c + 1e-12))
    return x, stats.ttest_rel(B, A).pvalue


# --------------------------------------------------------------- the refits (one core each) ---
def _fit(X, y, model, seed):
    """One covariates-only score on ONE core: hgb otherwise spreads every tiny fit over all cores."""
    from threadpoolctl import threadpool_limits
    from fmri_decomposition.bstm_selection import score_once
    with threadpool_limits(1):
        return score_once(X, y, model, seed)

def oof_shap(X, y, model, seed):
    """Out-of-fold SHAP and predictions on one core, with select-bstm's folds (KFold(5), seed):
    each fold's model explains and predicts only the subjects it did not see. Ridge: exact SHAP =
    coefficient x standardised value. Trees: shap.TreeExplainer, also exact."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    from threadpoolctl import threadpool_limits
    from fmri_decomposition.bstm_selection import make_model
    est = make_model(model, seed)[0]
    S, pred = np.zeros_like(X, dtype=float), np.zeros(len(y))
    with threadpool_limits(1):
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            m = clone(est).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
            if model == "ridge":
                S[te] = m[-1].coef_ * m[0].transform(X[te])
            else:
                import shap
                S[te] = shap.TreeExplainer(m).shap_values(X[te])
    return S, pred

def _mean_abs(X, y, model, seed):
    return np.abs(oof_shap(X, y, model, seed)[0]).mean(0)


# ----------------------------------------------------------------------- the study ---
class Study:
    """The rankings, the phenotype and the transition tables, and every table of the report."""

    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or Settings()
        self.summ = self._summaries()
        self.pheno = self._pheno()

    # -- reading -------------------------------------------------------------------------
    def _summaries(self):
        c, frames = self.cfg, []
        for tree in ("bstm", "fcm"):
            files = sorted((c.root / f"{tree}_selection").glob("task=*/target=*/summary.csv"))
            if not files:
                raise SystemExit(f"no summary.csv under {c.root / f'{tree}_selection'}")
            frames += [pd.read_csv(p).assign(tree=tree, task=hive(p, "task"), target=hive(p, "target"))
                       for p in files]
        label_of = {col: lab for lab, col in c.targets.items()}
        s = pd.concat(frames, ignore_index=True)
        s = s[s["target"].isin(label_of) & s["task"].isin(c.tasks)].assign(phenotype=lambda d: d["target"].map(label_of))
        s["label"] = s.apply(set_label, axis=1)
        s["window"] = s["window_s"].map(window_name)   # the axes of figures B2 (fcm rows have none)
        s[["method", "embedding", "K_"]] = s["states"].str.rsplit("_", n=2, expand=True).reindex(columns=range(3))
        return s.reset_index(drop=True)

    def _pheno(self):
        c = self.cfg
        covariates = list(dict.fromkeys(c.covariates + c.baseline_covariates))
        files = [(c.camcan / "approved_data.tsv", "\t"), (c.camcan / "standard_data.csv", ",")]   # HADS; Age, Sex
        frames = [pd.read_csv(p, sep=s, dtype={c.pheno_id: str}) for p, s in files]
        ph = frames[0]
        for f in frames[1:]:
            ph = ph.merge(f.drop(columns=[x for x in f if x in ph and x != c.pheno_id]), on=c.pheno_id, how="outer")
        lower = {x.lower(): x for x in ph.columns}                  # Age vs age
        ph = ph.rename(columns={lower[x.lower()]: x for x in covariates if x.lower() in lower})
        ph["sub"] = ph[c.pheno_id].astype(str).str.strip().str.upper()
        code = {lvl: k for k, lvl in enumerate(c.ordinal_levels)}
        for lab, col in c.targets.items():
            ph[lab] = ph[col].astype("string").str.strip().map(code).astype(float)
        for x in set(covariates) - set(c.categorical) - {"n_transitions"}:
            ph[x] = pd.to_numeric(ph[x], errors="coerce")
        return ph[["sub", *c.targets, *(x for x in covariates if x in ph)]]

    def numeric(self, covariates=None):
        return [x for x in (covariates or self.cfg.covariates) if x not in self.cfg.categorical]

    def load_set(self, space, task):
        """One state set's transition table for one task, one row per subject (cells = p_joint)."""
        c = self.cfg
        base = (c.root / "transitions" / f"atlas={space['atlas']}" / f"window_s={fmt_window(space['window_s'])}"
                / f"states={space['states']}")
        files = [p for p in base.rglob("subjects.parquet")
                 if hive(p, "cohort") in c.cohorts and (hive(p, "task") or task).lower() == c.tasks[task].lower()]
        if not files:
            raise SystemExit(f"no transition table for {base} task={task}")
        d = pd.concat(map(pd.read_parquet, files), ignore_index=True)
        d = d[d["task"].astype(str).str.lower() == c.tasks[task].lower()]
        d["sub"] = d["sub"].astype(str).str.strip().str.upper()
        if d["sub"].duplicated().any():
            raise SystemExit(f"{space['states']} {task}: a subject appears twice -- narrow cohorts")
        return d

    def table(self, space, task):
        """One set and task joined to both phenotypes: everyone with both scores and the covariates."""
        t = self.load_set(space, task)
        return t.merge(self.pheno, on="sub", suffixes=("", "_pheno")).dropna(
            subset=[*self.cfg.targets, *self.numeric()]).reset_index(drop=True)

    def groups(self, d):
        """Normal = Normal on BOTH scales (the reference); each phenotype = case_from or above on it."""
        c = self.cfg
        return {"Normal": (d[list(c.targets)] == 0).all(axis=1).to_numpy(),
                **{lab: (d[lab] >= c.case).to_numpy() for lab in c.targets}}

    def sets(self):
        """Every (atlas, window_s, states) with a table under transitions/, for the cohorts."""
        found = {(hive(p, "atlas"), float(hive(p, "window_s")), hive(p, "states"))
                 for p in (self.cfg.root / "transitions").rglob("subjects.parquet")
                 if hive(p, "cohort") in self.cfg.cohorts}
        return pd.DataFrame(sorted(found), columns=["atlas", "window_s", "states"]).assign(
            K=lambda d: d["states"].map(n_states))

    def models(self):
        return self.cfg.models or sorted(set(self.summ["model"]) - {"logistic"})

    # -- the cache ---------------------------------------------------------------------
    @functools.cached_property
    def stamp(self):
        """What every cached table is built from: the rankings, the sets on disk, the phenotype."""
        s = self.summ[["tree", "task", "target", "model", "arm", "atlas", "window_s", "states", "mean"]]
        return hashlib.md5((s.to_csv(index=False) + self.sets().to_csv(index=False)
                            + self.pheno.to_csv(index=False)).encode()).hexdigest()

    def _cached(self, name, key, compute, pickle=False):
        c = self.cfg
        if not c.cache_dir:
            return compute()
        p = Path(c.cache_dir) / f"{name}_{hashlib.md5(repr((self.stamp, key)).encode()).hexdigest()[:12]}"
        p = p.with_suffix(".joblib" if pickle else ".parquet")
        if p.exists():
            print(f"{name}: read {p}")
            return joblib.load(p) if pickle else pd.read_parquet(p)
        t0 = time.time()
        out = compute()
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(out, p) if pickle else out.to_parquet(p)
        print(f"{name}: built in {time.time() - t0:.0f}s, cached in {p}")
        return out

    def _rule(self):
        c = self.cfg
        return (c.case_from, c.shrink, c.min_present, tuple(c.covariates), tuple(c.cohorts))

    # -- the tests, on one table ---------------------------------------------------------
    def _pseudocount(self, block, A):
        return self.cfg.shrink * (A.mean() if block in PROBABILITY else A.mean(0))

    def _usable(self, block, A, B=None):
        """Columns worth testing: not flat, and for probabilities not rare (in either table)."""
        keep = A.std() > 0 if B is None else (A.std() > 0) | (B.std() > 0)
        if block in PROBABILITY:
            present = (A > 0).mean() >= self.cfg.min_present
            keep &= present if B is None else present | ((B > 0).mean() >= self.cfg.min_present)
        return keep

    def _finish(self, v, by=()):
        """BH-FDR (per `by` group) and the pass rule."""
        v = v.dropna(subset=["p"]).reset_index(drop=True)
        fdr = lambda p: stats.false_discovery_control(p, method="bh")
        v["p_fdr"] = v.groupby(list(by))["p"].transform(fdr) if by else fdr(v["p"])
        v["hit"] = (v["p_fdr"] < self.cfg.alpha) & (v["x"].abs() >= (self.cfg.fc_line or 0))
        return v

    def test_blocks(self, d, arm, g):
        """Every variable of `arm` in table d: group g = 1 vs g = 0, adjusted for the covariates."""
        out = []
        for block, cols in arm_columns(d, arm).items():
            A = d[cols].astype(float)
            A = A.loc[:, self._usable(block, A)]
            if A.shape[1]:
                x, p = group_effects(A.to_numpy(), g, d[self.cfg.covariates],
                                     self._pseudocount(block, A.to_numpy()), self.cfg.categorical)
                out.append(pd.DataFrame({"block": block, "variable": A.columns, "x": x, "p": p, "n": len(d)}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def paired_blocks(self, a, b, arm):
        """Every variable of `arm`, table b vs table a in the same people (rows aligned)."""
        out = []
        for block, cols in arm_columns(a, arm).items():
            cols = [x for x in cols if x in b]
            A, B = a[cols].astype(float), b[cols].astype(float)
            keep = self._usable(block, A, B)
            A, B = A.loc[:, keep].to_numpy(), B.loc[:, keep].to_numpy()
            if A.shape[1]:
                c = self.cfg.shrink * (np.concatenate([A, B]).mean() if block in PROBABILITY
                                       else (A.mean(0) + B.mean(0)) / 2)
                x, p = paired_effects(A, B, c)
                out.append(pd.DataFrame({"block": block, "variable": keep.index[keep], "x": x, "p": p,
                                         "mean_a": A.mean(0), "mean_b": B.mean(0), "n": len(A)}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def normal_pairs(self, space):
        """The Normal-on-both people with both scans: (base task table, other task table), aligned."""
        t0, t1 = self.cfg.tasks
        t = {}
        for task in (t0, t1):
            d = self.table(space, task)
            t[task] = d[self.groups(d)["Normal"]].set_index("sub")
        subs = t[t0].index.intersection(t[t1].index)
        return t[t0].loc[subs].reset_index(), t[t1].loc[subs].reset_index()

    # -- A: the multiverses ----------------------------------------------------------------
    def _over_sets(self, scan):
        sets = self.sets()
        parts = Parallel(n_jobs=self.cfg.n_jobs, prefer="threads")(   # parquet + numpy: threads are enough
            delayed(scan)(dict(s)) for _, s in sets.iterrows())
        v = pd.concat([p for part in parts for p in part], ignore_index=True)
        v["window"] = v["window_s"].map(window_name)
        v["method"] = v["states"].str.rsplit("_", n=2).str[0]
        v["set"] = v["atlas"] + " · " + v["window"] + " · " + v["states"]
        return v

    def multiverse_tasks(self):
        """A1: every variable of every state set, the second task vs the first, paired, in the Normal group."""
        c = self.cfg
        t0, t1 = c.tasks

        def scan(s):
            try:
                a, b = self.normal_pairs(s)
            except SystemExit:                         # this set lacks one of the tasks
                return []
            if "[cond]" in c.multiverse_arm:
                a, b = conditional(a, s["K"]), conditional(b, s["K"])
            v = self.paired_blocks(a, b, c.multiverse_arm)
            return [v.assign(**s, task=f"{t1} vs {t0}", phenotype="Normal")] if len(v) else []

        return self._cached("multiverse_tasks", (self._rule(), c.multiverse_arm),
                            lambda: self._finish(self._over_sets(scan)))

    def multiverse(self):
        """A2: every variable of every state set vs each phenotype (case_from+ vs below), adjusted."""
        c = self.cfg

        def scan(s):
            out = []
            for task in c.tasks:
                try:
                    t = self.load_set(s, task)
                except SystemExit:
                    continue
                if "[cond]" in c.multiverse_arm:
                    t = conditional(t, s["K"])
                d0 = t.merge(self.pheno, on="sub", suffixes=("", "_pheno"))
                for lab in c.targets:
                    d = d0.dropna(subset=[lab, *self.numeric()]).reset_index(drop=True)
                    v = self.test_blocks(d, c.multiverse_arm, (d[lab].to_numpy(float) >= c.case).astype(float))
                    if len(v):
                        out.append(v.assign(**s, task=task, phenotype=lab))
            return out

        return self._cached("multiverse", (self._rule(), c.multiverse_arm),
                            lambda: self._finish(self._over_sets(scan), by=["task", "phenotype"]))

    # -- B: performance --------------------------------------------------------------------
    def baselines(self):
        """The covariates-only score per (phenotype, model), with the pipeline's own CV (score_once),
        on everyone the bstm runs scored. One row per (phenotype, model): mean, std over seeds, n."""
        c, models = self.cfg, self.models()

        def compute():
            data, b = {}, self.summ[(self.summ["tree"] == "bstm") & (self.summ["arm"] != c.baseline_arm)]
            for lab in c.targets:
                subs = set()
                for task in c.tasks:                   # the best set of each task: everyone it scored
                    r = b[(b["task"] == task) & (b["phenotype"] == lab)].sort_values("mean").iloc[-1]
                    subs |= set(self.load_set(space_of(r), task)["sub"])
                d = self.pheno[self.pheno["sub"].isin(subs)].dropna(subset=[lab, *self.numeric(c.baseline_covariates)])
                data[lab] = design(d[c.baseline_covariates], c.categorical, intercept=False), d[lab].to_numpy(float)
            jobs = [(lab, m, s) for lab in c.targets for m in models for s in c.seeds]
            scores = Parallel(n_jobs=min(c.n_jobs, len(jobs)))(delayed(_fit)(*data[lab], m, s) for lab, m, s in jobs)
            out = pd.DataFrame(jobs, columns=["phenotype", "model", "seed"]).assign(score=scores)
            out = out.groupby(["phenotype", "model"])["score"].agg(mean="mean", std=lambda s: np.nanstd(s)).reset_index()
            return out.assign(n=out["phenotype"].map({lab: len(v[1]) for lab, v in data.items()}))

        return self._cached("baselines", (models, tuple(c.seeds), tuple(c.baseline_covariates)), compute)

    def performance_rows(self, base):
        """Every summary row of B1 / B2, with its own model's baseline (each row already carries the
        covariates, so a shared bar would mostly select whichever model fits Age + Sex best)."""
        c, s = self.cfg, self.summ
        rows = s[(s["arm"] != c.baseline_arm) & s["model"].isin(self.models())
                 & (s["arm"].isin(c.arms) if c.arms else True)]
        rows = rows.assign(entity=rows["tree"] + " · " + rows["label"] + " · " + rows["arm"] + " · " + rows["model"])
        return rows.merge(base.rename(columns={"mean": "base_mean", "std": "base_sd", "n": "base_n"}),
                          on=["phenotype", "model"], how="left")

    def above_both(self, d):
        """Rows above their model's baseline + sd in EVERY task, best worst-task margin first."""
        margin = d["mean"] - self.cfg.margin_sd * d["std"] - (d["base_mean"] + d["base_sd"])
        worst = d.assign(margin=margin)[margin > 0].groupby("entity").agg(
            tasks=("task", "nunique"), worst=("margin", "min"))
        return worst[worst["tasks"] == len(self.cfg.tasks)].sort_values("worst", ascending=False).index.tolist()

    def above_table(self, rows):
        """Every row above its model's baseline at both tasks, by phenotype and model."""
        out = {}
        for (lab, m), d in rows.groupby(["phenotype", "model"]):
            t = (d[d["entity"].isin(self.above_both(d))]
                 .pivot_table(index=["tree", "arm", "label"], columns="task", values="mean")
                 .reindex(columns=list(self.cfg.tasks)))
            out[lab, m] = t.loc[t.min(axis=1).sort_values(ascending=False).index].assign(
                baseline=d["base_mean"].iloc[0], baseline_sd=d["base_sd"].iloc[0])
        return pd.concat(out, names=["phenotype", "model"]).round(3) if out else pd.DataFrame()

    # -- C: where the models' contribution comes from ---------------------------------------
    def ranked(self, arms, top):
        """The `top` best state sets per (task, phenotype, model, arm) of the bstm rankings."""
        s = self.summ
        b = s[(s["tree"] == "bstm") & (s["arm"] != self.cfg.baseline_arm) & s["model"].isin(self.models())
              & s["arm"].isin(arms)].sort_values("mean", ascending=False)
        d = b.groupby(["task", "phenotype", "model", "arm"], sort=False).head(top) if top else b
        d = d.rename(columns={"mean": "cv"}).reset_index(drop=True)
        d["label"] = d["model"] + " · " + d["arm"] + " · " + d.apply(set_label, axis=1)
        return d[["task", "phenotype", "model", "arm", "atlas", "window_s", "states", "cv", "label"]]

    def matrix(self, space, task, lab, arm):
        """One model's data as select-bstm builds it: (X, y, names, blocks), covariates first."""
        c = self.cfg
        t = self.load_set(space, task)
        if "[cond]" in arm:
            t = conditional(t, n_states(space["states"]))
        d = t.merge(self.pheno, on="sub", suffixes=("", "_pheno")).dropna(subset=[lab, *self.numeric()])
        # select-bstm drops a subject missing ANY covariate (Sex too) before the folds are drawn;
        # keeping one would shift every fold after it and the refit would no longer be its model
        d = d.dropna(subset=[x for x in c.covariates if x != "n_transitions"]).reset_index(drop=True)
        cov = pd.get_dummies(d[list(c.covariates)], drop_first=True).astype(float)
        cov = cov.fillna(cov.mean())
        cols = arm_columns(d, arm)
        names = list(cov.columns) + [x for cs in cols.values() for x in cs]
        blocks = ["covariates"] * cov.shape[1] + [b for b, cs in cols.items() for _ in cs]
        X = np.column_stack([cov.to_numpy(), d[names[cov.shape[1]:]].to_numpy(float)])
        return X, d[lab].to_numpy(float), names, blocks

    def shares(self):
        """C: mean |SHAP| (out-of-fold, shap_seed) of every variable of the share_top best models of
        each share_arm, and its share of the model's total. One core per model, all cores busy."""
        c = self.cfg
        models = self.ranked(c.share_arms, c.share_top)

        def compute():
            names = []

            def jobs():                                # lazily: few matrices in memory at once
                for _, r in models.iterrows():
                    X, y, n, b = self.matrix(space_of(r), r["task"], r["phenotype"], r["arm"])
                    names.append((n, b))
                    yield delayed(_mean_abs)(X, y, r["model"], c.shap_seed)

            n_boost = int((models["model"] != "ridge").sum())
            print(f"explaining {len(models)} models ({n_boost} boosted) on {c.n_jobs} core(s): "
                  f"~{n_boost * 12 / 60 / c.n_jobs:.0f}-{n_boost * 21 / 60 / c.n_jobs:.0f} min at most")
            imp = Parallel(n_jobs=c.n_jobs)(jobs())
            out = pd.concat([pd.DataFrame({"variable": n, "block": b, "mean_abs": v}).assign(
                                 **models.loc[i, ["task", "phenotype", "model", "arm", "label", "cv"]].to_dict())
                             for i, ((n, b), v) in enumerate(zip(names, imp))], ignore_index=True)
            out["share"] = out["mean_abs"] / out.groupby(["task", "phenotype", "label"])["mean_abs"].transform("sum")
            return out

        return self._cached("shares", (models.to_csv(index=False), c.shap_seed, tuple(c.covariates)), compute)

    # -- D: one variable space ---------------------------------------------------------------
    def space_options(self, base, multi=None, top=10):
        """The best state sets to choose from: for each set, its best (model, arm) among graph_arms
        by CV score averaged over the four panels, its worst-panel margin over that model's
        baseline + sd, and (with the multiverse) how many of its variables pass per phenotype."""
        c, s = self.cfg, self.summ
        b = s[(s["tree"] == "bstm") & s["arm"].isin(c.graph_arms) & s["model"].isin(self.models())]
        b = b.merge(base.rename(columns={"mean": "base_mean", "std": "base_sd"})[["phenotype", "model", "base_mean", "base_sd"]],
                    on=["phenotype", "model"], how="left")
        b = b.assign(margin=b["mean"] - (b["base_mean"] + b["base_sd"]))
        keys = ["atlas", "window_s", "states", "model", "arm"]
        m = b.groupby(keys).agg(cv=("mean", "mean"), worst_margin=("margin", "min"), panels=("mean", "size")).reset_index()
        m = m[m["panels"] == len(c.tasks) * len(c.targets)].drop(columns="panels")
        m = m.sort_values("cv", ascending=False).drop_duplicates(["atlas", "window_s", "states"])
        m["above_all"] = m["worst_margin"] > 0
        if multi is not None:
            hits = multi[multi["hit"]].groupby(["atlas", "window_s", "states", "phenotype"]).size().unstack(fill_value=0)
            m = m.merge(hits.add_prefix("hits_").reset_index(), on=["atlas", "window_s", "states"], how="left")
            m[[x for x in m if x.startswith("hits_")]] = m[[x for x in m if x.startswith("hits_")]].fillna(0).astype(int)
        return m.head(top).reset_index(drop=True).round(3)

    def choose_space(self, chosen, options):
        r = chosen or options.iloc[0]
        return dict(space_of(r), K=n_states(r["states"]))

    def paired(self, space):
        """D1: every variable of the space, the second task vs the first, paired, Normal group."""
        t0, t1 = self.cfg.tasks
        a, b = self.normal_pairs(space)
        v = self.paired_blocks(conditional(a, space["K"]), conditional(b, space["K"]), ALL)
        return self._finish(v).assign(task=f"{t1} vs {t0}", phenotype="Normal")

    def group_tests(self, space):
        """D2: every variable of the space, each phenotype group vs Normal (on both scales), per
        task, adjusted for the covariates. FDR per (task, phenotype) over all its variables."""
        out = []
        for task in self.cfg.tasks:
            d = self.table(space, task)
            m = self.groups(d)
            d = conditional(d, space["K"])
            for lab in self.cfg.targets:
                rows = m[lab] | m["Normal"]
                v = self.test_blocks(d[rows].reset_index(drop=True), ALL, m[lab][rows].astype(float))
                out.append(v.assign(task=task, phenotype=lab, n_group=int(m[lab].sum()), n_normal=int(m["Normal"].sum())))
        return self._finish(pd.concat(out, ignore_index=True), by=["task", "phenotype"])

    def pooled(self, space):
        """{(task, group): (p(j | i), occupancy, n)} for the diagrams: p(j | i) pooled over the group's
        transition counts, occupancy averaged over its subjects."""
        K, out = space["K"], {}
        for task in self.cfg.tasks:
            d = self.table(space, task)
            for g, m in self.groups(d).items():
                cnt = (d.loc[m, cell_cols(d)].to_numpy(float) * d.loc[m, "n_transitions"].to_numpy(float)[:, None]).sum(0)
                cnt = cnt.reshape(K, K)
                rows = cnt.sum(1, keepdims=True)
                out[task, g] = (cnt / np.where(rows > 0, rows, 1),
                                d.loc[m, [f"occ_{k}" for k in range(K)]].to_numpy(float).mean(0), int(m.sum()))
        return out

    def _change(self, pooled, panel, ref, block, variable):
        T, occ, _ = pooled[panel]
        Tr, orr, _ = pooled[ref]
        if block == "cells":
            i, j = map(int, variable.split("->"))
            return T[i, j] - Tr[i, j]
        k = int(variable.split("_")[1])
        return occ[k] - orr[k]

    def _notes(self, marks):
        """The pp label of each highlighted item; with label_pp, only the largest per panel."""
        if marks.empty:
            return marks.assign(note="")
        marks["note"] = [pp(x) if h else "" for x, h in zip(marks["change"], marks["hit"])]
        if self.cfg.label_pp:
            big = marks[marks["hit"]].groupby(["task", "group"])["change"].transform(
                lambda x: x.abs().rank(ascending=False, method="first") <= self.cfg.label_pp)
            marks.loc[big.index[~big], "note"] = ""
        return marks

    def stat_marks(self, pooled, paired, grp, task_alpha=None, task_fc=None, task_top=None):
        """D3: what is highlighted on the graph and its change in pp. The second task's Normal panel:
        the paired test vs the first task's Normal. Each phenotype panel: its group test vs the Normal
        panel of the same task. The task difference passes far more easily (hundreds of people, each
        their own control), so its rule can be stricter: task_alpha (FDR), task_fc (|log2FC|),
        task_top (only the N largest |log2FC| that pass). None = the usual rule."""
        t0, t1 = self.cfg.tasks
        rows = []
        if task_alpha is not None or task_fc is not None:
            paired = paired.assign(hit=(paired["p_fdr"] < (self.cfg.alpha if task_alpha is None else task_alpha))
                                   & (paired["x"].abs() >= ((self.cfg.fc_line if task_fc is None else task_fc) or 0)))
        if task_top:
            keep = paired[paired["hit"] & paired["block"].isin(GRAPH)]["x"].abs().nlargest(task_top).index
            paired = paired.assign(hit=paired.index.isin(keep))

        def add(panel, ref, v):
            for r in v[v["hit"] & v["block"].isin(GRAPH)].itertuples():
                rows.append(dict(task=panel[0], group=panel[1], block=r.block, variable=r.variable,
                                 x=r.x, p_fdr=r.p_fdr, change=self._change(pooled, panel, ref, r.block, r.variable), hit=True))

        add((t1, "Normal"), (t0, "Normal"), paired)
        for task in self.cfg.tasks:
            for lab in self.cfg.targets:
                add((task, lab), (task, "Normal"), grp[(grp["task"] == task) & (grp["phenotype"] == lab)])
        cols = ["task", "group", "block", "variable", "x", "p_fdr", "change", "hit"]
        return self._notes(pd.DataFrame(rows, columns=cols))

    # -- E: one model ---------------------------------------------------------------------------
    def model_options(self, space):
        """Every (model, arm) scored on the space in all four panels, best mean CV first. The arms
        must hold graph variables (cells or occupancy) for the diagram to have anything to draw."""
        c, s = self.cfg, self.summ
        b = s[(s["tree"] == "bstm") & (s["atlas"] == space["atlas"]) & np.isclose(s["window_s"], space["window_s"])
              & (s["states"] == space["states"]) & s["model"].isin(self.models())]
        b = b[b["arm"].map(lambda a: bool(GRAPH & set(ARM_BLOCKS.get(a.split(" [")[0], []))))]
        t = b.pivot_table(index=["model", "arm"], columns=["task", "phenotype"], values="mean")
        t = t[t.notna().all(axis=1)]
        t.columns = [f"{a} · {b}" for a, b in t.columns]
        return t.assign(cv=t.mean(axis=1)).sort_values("cv", ascending=False).reset_index().round(3)

    def choose_model(self, chosen, options):
        if chosen:
            return dict(chosen)
        best = options[options["arm"].isin(self.cfg.graph_arms)]
        return dict((best if len(best) else options).iloc[0][["model", "arm"]])

    def _scores(self, task, lab, space, model):
        """select-bstm's own per-seed scores for one model, from its scores.csv (empty if absent)."""
        p = self.cfg.root / "bstm_selection" / f"task={task}" / f"target={self.cfg.targets[lab]}" / "scores.csv"
        if not p.exists():
            return pd.DataFrame(columns=["seed", "score"])
        s = pd.read_csv(p)
        return s[(s["model"] == model["model"]) & (s["arm"] == model["arm"]) & (s["atlas"] == space["atlas"])
                 & np.isclose(s["window_s"], space["window_s"]) & (s["states"] == space["states"])][["seed", "score"]]

    def performance_choice(self, space, model, base):
        """E1: the chosen model's score per panel (mean, sd, and each seed when scores.csv has them),
        its baseline, its rank among every bstm row of the same model in that panel, and the best
        static-FC (fcm) row of the same model for comparison."""
        s, out = self.summ, []
        for task in self.cfg.tasks:
            for lab in self.cfg.targets:
                panel = s[(s["tree"] == "bstm") & (s["task"] == task) & (s["phenotype"] == lab)
                          & (s["model"] == model["model"]) & (s["arm"] != self.cfg.baseline_arm)]
                me = panel[(panel["arm"] == model["arm"]) & (panel["atlas"] == space["atlas"])
                           & np.isclose(panel["window_s"], space["window_s"]) & (panel["states"] == space["states"])]
                if me.empty:
                    continue
                b = base[(base["phenotype"] == lab) & (base["model"] == model["model"])].iloc[0]
                fc = s[(s["tree"] == "fcm") & (s["task"] == task) & (s["phenotype"] == lab)
                       & (s["model"] == model["model"]) & (s["arm"] != self.cfg.baseline_arm)]
                fc = fc.loc[fc["mean"].idxmax()] if len(fc) else None
                row = dict(task=task, phenotype=lab, mean=me["mean"].iloc[0], std=me["std"].iloc[0],
                           rank=int((panel["mean"] > me["mean"].iloc[0]).sum()) + 1, of=len(panel),
                           base_mean=b["mean"], base_sd=b["std"], base_n=b["n"],
                           fc_mean=np.nan if fc is None else fc["mean"], fc_std=np.nan if fc is None else fc["std"],
                           fc_label="none" if fc is None else f"{fc['atlas']} · {fc['arm']}")
                seeds = self._scores(task, lab, space, model)
                out += [dict(row, seed=r.seed, score=r.score) for r in seeds.itertuples()] or [dict(row, seed=np.nan, score=np.nan)]
        return pd.DataFrame(out)

    def deep(self, space, model):
        """E2 / E3: the chosen model refit in each panel, out-of-fold SHAP at every seed of
        seeds_stab (and shap_seed). Per panel: X, y, names, blocks; S and pred at shap_seed;
        by_seed / sign_by_seed = mean |SHAP| / direction per seed (seeds x variables)."""
        c = self.cfg
        seeds = list(dict.fromkeys([c.shap_seed, *c.seeds_stab]))

        def compute():
            panels = {}
            for task in c.tasks:
                for lab in c.targets:
                    X, y, names, blocks = self.matrix(space, task, lab, model["arm"])
                    panels[task, lab] = dict(X=X, y=y, names=names, blocks=blocks, by_seed={}, sign_by_seed={})
            jobs = [(k, s) for k in panels for s in seeds]
            print(f"refitting {len(jobs)} models ({model['model']}) on {min(c.n_jobs, len(jobs))} core(s)")
            res = Parallel(n_jobs=min(c.n_jobs, len(jobs)))(
                delayed(oof_shap)(panels[k]["X"], panels[k]["y"], model["model"], s) for k, s in jobs)
            for (k, s), (S, pred) in zip(jobs, res):
                p = panels[k]
                p["by_seed"][s], p["sign_by_seed"][s] = np.abs(S).mean(0), direction(p["X"], S)
                if s == c.shap_seed:
                    p["S"], p["pred"] = S, pred
            for (task, lab), p in panels.items():
                p["by_seed"] = np.array([p["by_seed"][s] for s in c.seeds_stab])
                p["sign_by_seed"] = np.array([p["sign_by_seed"][s] for s in c.seeds_stab])
                p["rho"] = stats.spearmanr(p["pred"], p["y"]).statistic
                own = self._scores(task, lab, space, model)
                own = own.loc[own["seed"] == c.shap_seed, "score"]
                p["rho_selection"] = float(own.mean()) if len(own) else np.nan
            return panels

        key = (space["atlas"], space["window_s"], space["states"], model["model"], model["arm"],
               tuple(seeds), tuple(c.covariates))
        return self._cached("deep", key, compute, pickle=True)

    def check(self, deep):
        """The refit must BE select-bstm's model: its out-of-fold rho equals the one in scores.csv."""
        rows = [(t, lab, len(d["y"]), round(d["rho"], 4), round(d["rho_selection"], 4),
                 "no scores.csv" if np.isnan(d["rho_selection"]) else
                 "same" if abs(d["rho"] - d["rho_selection"]) < 1e-6 else "DIFFERENT")
                for (t, lab), d in deep.items()]
        check = pd.DataFrame(rows, columns=["task", "phenotype", "n", "rho_here", "rho_select_bstm", "reproduced"])
        if (check["reproduced"] == "DIFFERENT").any():
            print("WARNING: the refit does not reproduce select-bstm's score in some panel. The subjects or their\n"
                  "order differ (e.g. --restrict-subjects in the selection run, or another phenotype file), so\n"
                  "these SHAP values describe a close cousin of the ranked model, not the model itself.")
        return check

    def model_marks(self, space, deep, pooled):
        """E3: the model's top `highlight_top` graph variables by mean |SHAP| (shap_seed). Red when
        it is also in the top for >= stable_min of the seeds, and the group differs from Normal
        by |Cohen's d| >= min_d and at least 1 pp. Every candidate is a row, with why."""
        c, K, rows = self.cfg, space["K"], []
        for task in c.tasks:
            d = self.table(space, task)
            m = self.groups(d)
            per_subject = pd.concat([as_p_cond(d, K), d[[f"occ_{k}" for k in range(K)]]], axis=1)
            for lab in c.targets:
                dd = deep[task, lab]
                order = np.argsort(np.abs(dd["S"]).mean(0))[::-1]
                rank = {dd["names"][v]: r + 1 for r, v in enumerate(order)}
                graph = [v for v in range(len(dd["names"])) if dd["blocks"][v] in GRAPH]
                M = dd["by_seed"][:, graph]
                stable = dict(zip(graph, (np.argsort(np.argsort(-M, axis=1), axis=1) < c.highlight_top).mean(0)))
                sign = direction(dd["X"], dd["S"])
                for v in [v for v in order if v in stable][:c.highlight_top]:
                    name = dd["names"][v]
                    a, b = per_subject.loc[m[lab], name], per_subject.loc[m["Normal"], name]
                    sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
                    cohen = (a.mean() - b.mean()) / sd if sd > 0 else 0.0
                    change = self._change(pooled, (task, lab), (task, "Normal"), dd["blocks"][v], name)
                    rows.append(dict(task=task, group=lab, block=dd["blocks"][v], variable=name, rank=rank[name],
                                     mean_abs_shap=np.abs(dd["S"][:, v]).mean(), direction="↑" if sign[v] > 0 else "↓",
                                     seeds_in_top=stable[v], cohen_d=cohen, change=change,
                                     hit=stable[v] >= c.stable_min and abs(cohen) >= c.min_d and round(100 * change) != 0))
        cols = ["task", "group", "block", "variable", "rank", "mean_abs_shap", "direction", "seeds_in_top",
                "cohen_d", "change", "hit"]
        return self._notes(pd.DataFrame(rows, columns=cols))


def dynamics_consistency(v):
    """The dynamics mean the same thing in every set: how consistent is each one across sets?
    share_up = share of sets where it is higher; hits = sets where it passes."""
    d = v[v["block"] == "dynamics"]
    return (d.groupby(["phenotype", "variable", "task"])
             .agg(sets=("x", "size"), share_up=("x", lambda x: (x > 0).mean()), median_x=("x", "median"),
                  hits=("hit", "sum"))
             .unstack("task").round(3))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the cached tables of 10_hads_report.ipynb.")
    ap.add_argument("--space", help="atlas,window_s,states: also build that space's deep dive")
    ap.add_argument("--model", help="with --space: the model of the deep dive (e.g. hgb)")
    ap.add_argument("--arm", default="all [cond]", help="with --space: the arm of the deep dive")
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args()
    cfg = Settings(**({"n_jobs": a.n_jobs} if a.n_jobs else {}))
    st = Study(cfg)
    st.multiverse_tasks()
    st.multiverse()
    st.baselines()
    st.shares()
    if a.space:
        atlas, w, states = a.space.split(",")
        space = dict(atlas=atlas, window_s=float(w), states=states, K=n_states(states))
        st.deep(space, dict(model=a.model or st.choose_model(None, st.model_options(space))["model"], arm=a.arm))
