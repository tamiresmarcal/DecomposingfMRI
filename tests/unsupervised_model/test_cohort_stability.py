"""Stability analyses of cohort_tables (notebook 14) on a toy Cohorts."""
import numpy as np
import pandas as pd

from fmri_decomposition.unsupervised_model import cohort_tables as ct


def _toy(seed=0):
    K, rng = 3, np.random.default_rng(seed)
    cfg = ct.Settings(space=dict(atlas="x", window_s=30.0, states="HMM2_umap3_3"),
                      cohorts={"a": dict(dirs=["a"], role="train"), "b": dict(dirs=["b"], role="projected")})
    co = ct.Cohorts.__new__(ct.Cohorts)
    co.cfg, co.by_task = cfg, {}
    co.cells = [f"{i}->{j}" for i in range(K) for j in range(K)]
    trait = rng.normal(0, 1, 50)

    def table(n, shift, subs, tr):
        P = rng.dirichlet(np.ones(K) * 3, size=(n, K)) + np.eye(K)
        P /= P.sum(2, keepdims=True)
        d = pd.DataFrame(P.reshape(n, -1), columns=co.cells, index=subs)
        d[[f"occ_{k}" for k in range(K)]] = rng.dirichlet(np.ones(K) * 5, size=n)
        d["switch_rate"] = .4 + shift + .1 * tr + rng.normal(0, .01, n)      # shifts, keeps rank
        d["entropy_rate_bits"] = 1 + .1 * tr + rng.normal(0, .01, n)         # stable trait
        return d
    subs = [f"s{i}" for i in range(50)]
    co.data = {("a", "movie"): table(50, 0, [f"a{i}" for i in range(50)], rng.normal(0, 1, 50)),
               ("b", "movie"): table(50, 0, subs, trait), ("b", "rest"): table(50, .2, subs, trait)}
    units = []
    for (c, k), d in co.data.items():                       # one scan per person: counts and minutes
        u = pd.DataFrame(d[co.cells].to_numpy() * 100, columns=co.cells)
        u["sub"], u["task"], u["cohort"], u["kind"] = d.index, k, c, k
        u["minutes"] = 120.0 if c == "a" else 8.0           # long films vs short scans
        units.append(u)
    co.units = pd.concat(units, ignore_index=True)
    return co


def test_stability_separates_a_stable_trait_from_a_shifted_one():
    st = _toy().stability(["entropy_rate_bits", "switch_rate"], n_boot=200).set_index("variable")
    assert st.loc["entropy_rate_bits", "task_equivalent"]
    assert not st.loc["switch_rate", "task_equivalent"]
    assert st.loc["switch_rate", "task_d"] < 0                         # movie - rest, rest is higher
    assert st.loc["entropy_rate_bits", "icc"] > .8 and st.loc["switch_rate", "icc"] > .8


def test_icc_c1_is_consistency_not_agreement():
    x = np.arange(20.0)
    assert np.isclose(ct._icc_c1(x, x + 5), 1.0)                       # a constant shift keeps rank
    assert ct._icc_c1(x, -x) < 0


def test_similarity_shape():
    S = _toy().similarity()
    assert S.shape == (3, 3) and np.allclose(np.diag(S), 1)


def _toy_films(seed=1):
    co = _toy(seed)
    rng = np.random.default_rng(seed)
    K = co.cfg.K
    def film(subs, shift):
        P = rng.dirichlet(np.ones(K) * 3, size=(len(subs), K)) + np.eye(K)
        P /= P.sum(2, keepdims=True)
        d = pd.DataFrame(P.reshape(len(subs), -1), columns=co.cells, index=subs)
        d[[f"occ_{k}" for k in range(K)]] = rng.dirichlet(np.ones(K) * 5, size=len(subs))
        d["switch_rate"] = .4 + shift + rng.normal(0, .02, len(subs))
        d["entropy_rate_bits"] = 1 + rng.normal(0, .02, len(subs))
        return d
    same = [f"s{i}" for i in range(30)]
    co.by_task = {("b", "movie", "clip1"): film(same, 0), ("b", "movie", "clip2"): film(same, 0),
                  ("b", "movie", "clip3"): film(same, .3),
                  ("a", "movie", "f1"): film([f"a{i}" for i in range(20)], 0),
                  ("a", "movie", "f2"): film([f"x{i}" for i in range(20)], 0)}
    return co


def test_film_summary_within_and_between_person():
    co = _toy_films()
    co.cfg.alpha = 0.05
    fs = co.film_summary(["entropy_rate_bits", "switch_rate"]).set_index(["film", "variable"])
    assert fs.loc[("clip1", "switch_rate"), "within"] and not fs.loc[("f1", "switch_rate"), "within"]
    assert fs.loc[("clip3", "switch_rate"), "differs"]                    # the shifted clip
    assert not fs.loc[("clip1", "entropy_rate_bits"), "differs"]
    assert not fs.loc[("f1", "entropy_rate_bits"), "differs"]


def test_tiers_and_replication():
    co = _toy()
    st = co.stability(["entropy_rate_bits", "switch_rate"], n_boot=0).set_index("variable")
    assert st.loc["entropy_rate_bits", "tier"] == "stable"
    assert st.loc["switch_rate", "tier"] == "context-dependent"
    assert np.isnan(st.loc["switch_rate", "task_lo"])                     # n_boot=0: no CI
    pairs, ceil = co.replication("movie", n_split=5)
    assert len(pairs) == 1 and ceil.notna().all()


def test_sweep_summary_runs_on_a_fresh_cohorts():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import stability_sweep as sw
    r = sw.summarize(_toy())                       # no transition_fit() beforehand
    assert r["n_stable"] + r["n_context"] <= 3 and "d_transition_fit" in r


def test_replication_balanced_by_time():
    co = _toy()
    pairs, ceil = co.replication(balance="time", hours=2, n_boot=20)
    assert set(pairs["kind"]) == {"between cohorts", "rest vs movie"}
    assert (pairs["lo"] <= pairs["r_norm"]).all() and (pairs["r_norm"] <= pairs["hi"]).all()
    assert ceil.notna().all()


def test_figure3_draws():
    import matplotlib
    matplotlib.use("Agg")
    from fmri_decomposition.figures import cohorts as cp
    co = _toy()
    co.cfg.dynamics = ["switch_rate", "entropy_rate_bits"]
    marks, _ = co.task_change()
    st = co.stability(["transition_fit", "switch_rate", "entropy_rate_bits"], n_boot=20)
    pairs, _ = co.replication(balance="time", hours=2, n_boot=10)
    fig = cp.fig_figure3(co, co.graphs(), marks, st, pairs, co.agreement(co.dyn),
                         ["transition_fit", "switch_rate", "entropy_rate_bits"])
    assert len(fig.axes) > 6
