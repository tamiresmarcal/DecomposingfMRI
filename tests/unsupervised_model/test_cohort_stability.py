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
    st = co.stability(["transition_fit", "switch_rate", "entropy_rate_bits"], n_boot=20)
    fig = cp.fig_figure3(co, st, ["transition_fit", "switch_rate", "entropy_rate_bits"])
    assert len(fig.axes) > 6
    st2 = co.stability(["transition_fit", "occupancy_fit", "switch_rate"], n_boot=0)
    assert "occupancy_fit" in set(st2["variable"])
    fig = cp.fig_figure3(co, st2, occupancy="fit")
    labels = [t.get_text() for t in fig.axes[-1].get_yticklabels()]
    assert "Occupancy profile (fit to reference)" in labels and not any(l.startswith("Occupancy S") for l in labels)
    pairs, _ = co.replication(balance="time", hours=2, n_boot=10)
    assert len(cp.fig_replication(co, pairs).axes) == 1


def test_films_with_rest_first():
    co = _toy_films()
    co.by_task[("b", "rest", "Rest")] = next(iter(co.by_task.values())).copy()
    F = co.films(rest=True)
    assert ("b", "rest") in F and list(F)[0] == ("b", "rest") or [k for k in F if k[0] == "b"][0] == ("b", "rest")
    assert ("b", "rest") not in co.films()
    fs = co.film_summary(["switch_rate"], rest=True)
    assert "rest" in set(fs["film"])


def test_context_summary_three_comparisons():
    co = _toy_films()
    st = co.stability(["entropy_rate_bits", "switch_rate"], n_boot=0)
    cs = co.context_summary(st, co.film_summary(["entropy_rate_bits", "switch_rate"])).set_index("variable")
    assert cs.loc["switch_rate", "tier_rest_vs_movie"] == "context-dependent"
    assert cs.loc["switch_rate", "film_vs_film_same_people"] > cs.loc["entropy_rate_bits", "film_vs_film_same_people"]


def test_episode_halves_are_one_film():
    co = _toy_films()
    d = co.by_task.pop(("b", "movie", "clip1"))
    co.by_task[("b", "movie", "s01e01a")] = d
    co.by_task[("b", "movie", "s01e01b")] = d + 1
    F = co.films()
    assert ("b", "s01e01") in F and ("b", "s01e01a") not in F
    assert np.allclose(F[("b", "s01e01")].loc[d.index, "switch_rate"], d["switch_rate"] + .5)


def test_expected_distinct_matches_simulation():
    from math import comb
    counts = np.array([10, 5, 1, 1, 0, 3])
    n = 6
    N = counts.sum()
    exact = sum(1 - comb(N - c, n) / comb(N, n) for c in counts if c > 0)
    assert np.isclose(ct.expected_distinct(counts, n), exact)
    rng = np.random.default_rng(0)
    pool = np.repeat(np.arange(len(counts)), counts)
    sim = np.mean([len(set(rng.choice(pool, n, replace=False))) for _ in range(20000)])
    assert abs(sim - exact) < .05
    assert ct.expected_distinct(counts, N) == (counts > 0).sum()      # all drawn: the observed count
    assert np.isnan(ct.expected_distinct(counts, N + 1))


def test_rarefied_distinct_removes_length_effect():
    K = 3
    p = np.array([.5, .2, .1, .08, .05, .03, .02, .015, .005])
    cells = [f"{i}->{j}" for i in range(K) for j in range(K)]
    rows = []
    for minutes in (100, 1000):
        n = minutes * 4                                           # 4 transitions / min, stride 15 s
        rows.append({**dict(zip(cells, p)), "n_transitions": n, "n_windows": n + 1, "stride_s": 15.0})
    d = pd.DataFrame(rows)
    r = ct.rarefied_distinct(d, K, 5)                             # 20 transitions from each
    expected = (1 - (1 - p) ** 20).sum()                          # 20 draws from p itself
    assert np.allclose(r, expected, atol=.15)                     # whatever the scan length


def test_duration_check_runs_on_toy():
    co = _toy_films()
    rng = np.random.default_rng(0)
    for d in co.by_task.values():
        d["scan_s"] = rng.uniform(300, 3000, len(d))
    v = co.dyn[0]
    for d in co.by_task.values():                         # a metric that IS scan length
        d[v] = d["scan_s"] / 60
    out = co.duration_check([v])
    assert {"cohort", "kind", "variable", "rho", "p_fdr"} <= set(out.columns)
    assert (out["rho"].dropna() > .99).all() and out["rho"].notna().any()


def test_replication_versus_mean_one_row_per_cohort():
    import pytest
    co = _toy()
    if not hasattr(co, "units"):
        pytest.skip("toy has no units")
    pairs, ceil = co.replication("movie", balance="time", hours=.5, n_boot=5, versus="mean")
    vm = pairs[pairs["kind"] == "vs mean"]
    assert len(vm) == len({c for c, k in co.data if k == "movie"})
    assert (vm["b"] == "mean of the others").all()


def test_aggregation_check_draws():
    import matplotlib
    matplotlib.use("Agg")
    from fmri_decomposition.figures import cohorts as cp
    co = _toy()
    fig, st = cp.fig_aggregation_check(co, n_boot=0, measure="rest_movie")
    assert len(fig.axes) == 3
    fig, st2 = cp.fig_aggregation_check(_toy_films(), n_boot=20)          # omega^2 of task
    assert st2.attrs["scale"] == "omega2" and len(fig.axes) == 3
    assert {"transition_fit", "occupancy_fit", "dynamics_fit"} <= set(st["variable"])


def test_p_value_columns():
    import matplotlib
    matplotlib.use("Agg")
    from fmri_decomposition.figures import cohorts as cp
    co = _toy()
    st = co.stability(["transition_fit", "switch_rate", "entropy_rate_bits"], n_boot=0)
    assert {"task_p_tost", "task_p_tost_fdr"} <= set(st.columns)
    assert st["task_p_tost"].between(0, 1).all()
    import matplotlib.pyplot as plt
    for pv, test, n_cols in (("stars", "both", 2), ("exact", "tost", 1)):
        fig, ax = plt.subplots()
        cp._tier_panel(ax, st, ["transition_fit", "switch_rate", "entropy_rate_bits"], pv, test, headers=True)
        heads = [t for t in ax.texts if t.get_text().startswith("p\n")]
        assert len(heads) == n_cols
    fig = cp.fig_figure3(co, st, ["transition_fit", "switch_rate", "entropy_rate_bits"], pvalue="stars")
    assert not [t for t in fig.axes[-1].texts if t.get_text().startswith("p\n") or t.get_text() == "stable"]
    assert cp.stars(.0005) == "***" and cp.stars(.2) == "" and cp.fmt_p(.0004) == "<.001"



def test_tier_probability():
    import matplotlib
    matplotlib.use("Agg")
    from fmri_decomposition.figures import cohorts as cp
    co = _toy()
    st = co.stability(["transition_fit", "switch_rate", "entropy_rate_bits"], n_boot=200)
    assert st["task_p_tier"].between(0, 1).all()
    assert (ct._tier_index(np.array([.1, .3, .9])) == [0, 1, 2]).all()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp._tier_panel(ax, st, ["transition_fit", "switch_rate", "entropy_rate_bits"], "exact", headers=True)
    assert [t for t in ax.texts if t.get_text() == "P(tier)"]
    assert cp.tier_stars(.96) == "**" and cp.tier_stars(.5) == ""
    assert co.stability(["switch_rate"], n_boot=0)["task_p_tier"].isna().all()



def test_stability_survives_a_variable_missing_in_a_cohort():
    co = _toy()
    v = co.dyn[0]
    first = next(iter(co.data))
    for key in co.data:
        if key[0] == first[0]:
            co.data[key][v] = np.nan                       # e.g. a row no one in this cohort leaves
    st = co.stability([v, "transition_fit"], n_boot=20)
    assert len(st) == 2


def test_load_pools_counts_per_person(tmp_path):
    """p(j | i) from counts summed over a person's scans; a row they never leave is NaN, not 0."""
    K = 2
    cells = [f"{i}->{j}" for i in range(K) for j in range(K)]

    def scan(sub, task, C):
        C = np.asarray(C, float)
        n = C.sum()
        return {"sub": sub, "task": task, **dict(zip(cells, C.ravel() / n)), "n_transitions": n,
                "occ_0": .5, "occ_1": .5, "switch_rate": .2, "n_windows": n + 1, "stride_s": 6.0}

    rest = pd.DataFrame([scan("A", "rest", [[9, 1], [0, 0]]),        # A never leaves 1 in run 1
                         scan("A", "rest", [[0, 0], [2, 8]]),
                         scan("B", "rest", [[10, 0], [0, 0]])])       # B never visits 1 at all
    movie = pd.DataFrame([scan("A", "movie", [[5, 5], [5, 5]]), scan("B", "movie", [[5, 5], [5, 5]])])
    files = {"rest": [tmp_path / "r.parquet"], "movie": [tmp_path / "m.parquet"]}
    rest.to_parquet(files["rest"][0])
    movie.to_parquet(files["movie"][0])
    co = ct.Cohorts.__new__(ct.Cohorts)
    co.cfg = ct.Settings(space=dict(atlas="x", window_s=30.0, states="HMM2_umap3_2"),
                         cohorts={"c": dict(dirs=["c"], role="train")}, dynamics=["switch_rate"],
                         min_out_transitions=5)
    co._files = lambda label: files
    data = co._load()
    r = data["c", "rest"]
    assert np.isclose(r.loc["A", "0->1"], .1) and np.isclose(r.loc["A", "1->0"], .2)   # pooled, not averaged
    assert np.isnan(r.loc["B", "1->0"]) and np.isnan(r.loc["B", "1->1"])              # never left 1: NaN
    assert np.isclose(r.loc["B", "0->0"], 1.0)
    m = data["c", "movie"]
    assert np.isclose(m.loc["B", "1->0"], .5)          # B left 1 ten times in the movie: >= 5 over all
    pooled = co.pooled["c"]                            # every scan of a person together, no rest/movie split
    assert np.isclose(pooled.loc["A", "0->1"], 6 / 20) and np.isclose(pooled.loc["A", "1->0"], 7 / 20)
    assert co.groups(tasks="all")["c"] is co.pooled["c"]


def test_load_threshold_over_both_conditions(tmp_path):
    """Eligibility is per person over rest and movie together, not per condition."""
    K = 2
    cells = [f"{i}->{j}" for i in range(K) for j in range(K)]

    def scan(sub, task, C):
        C = np.asarray(C, float)
        n = C.sum()
        return {"sub": sub, "task": task, **dict(zip(cells, C.ravel() / n)), "n_transitions": n,
                "occ_0": .5, "occ_1": .5, "switch_rate": .2, "n_windows": n + 1, "stride_s": 6.0}

    files = {"rest": [tmp_path / "r.parquet"], "movie": [tmp_path / "m.parquet"]}
    pd.DataFrame([scan("A", "rest", [[10, 1], [1, 2]])]).to_parquet(files["rest"][0])     # 3 exits from 1
    pd.DataFrame([scan("A", "movie", [[10, 1], [1, 2]])]).to_parquet(files["movie"][0])   # 3 more: 6 in all
    co = ct.Cohorts.__new__(ct.Cohorts)
    co.cfg = ct.Settings(space=dict(atlas="x", window_s=30.0, states="HMM2_umap3_2"),
                         cohorts={"c": dict(dirs=["c"], role="train")}, dynamics=["switch_rate"],
                         min_out_transitions=5)
    co._files = lambda label: files
    data = co._load()
    assert np.isclose(data["c", "rest"].loc["A", "1->1"], 2 / 3)       # 3 < 5 here, but 6 over both
    assert np.isclose(data["c", "movie"].loc["A", "1->1"], 2 / 3)


def test_tier_panel_sorted_within_family():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fmri_decomposition.figures import cohorts as cp
    st = pd.DataFrame({"variable": ["occ_1", "switch_rate", "occ_0", "transition_fit", "entropy_rate_bits"],
                       "task_d": [.1, .9, 1.5, .3, .2], "task_lo": np.nan, "task_hi": np.nan,
                       "tier": ["stable", "context-dependent", "context-dependent", "moderate", "moderate"]})
    fig, ax = plt.subplots()
    cp._tier_panel(ax, st, ["transition_fit", "switch_rate", "entropy_rate_bits", "occ_1", "occ_0"])
    ticks = sorted(zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]), reverse=True)
    labels = [l for _, l in ticks]                                        # top to bottom
    assert labels == [cp.nice(v) for v in ["transition_fit", "entropy_rate_bits", "switch_rate", "occ_1", "occ_0"]]



def test_transitions_grouped_by_source_then_sorted():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fmri_decomposition.figures import cohorts as cp
    vs = ["transition_fit", "1->0", "0->1", "0->0", "1->1"]
    st = pd.DataFrame({"variable": vs, "task_d": [.1, .2, .9, .3, .05], "task_lo": np.nan, "task_hi": np.nan,
                       "tier": ["stable"] * 5})
    fig, ax = plt.subplots()
    cp._tier_panel(ax, st, vs)
    ticks = sorted(zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]), reverse=True)
    assert [l for _, l in ticks] == [cp.nice(v) for v in ["transition_fit", "0->0", "0->1", "1->1", "1->0"]]


def test_film_summary_survives_an_empty_variable():
    co = _toy_films()
    v = co.dyn[0]
    first = next(iter(co.by_task))
    co.by_task[first][v] = np.nan
    fs = co.film_summary([v])
    assert len(fs)


def test_task_variation_omega2():
    co = _toy_films()
    v = co.dyn[0]
    rng = np.random.default_rng(0)
    for (c, k, t), d in co.by_task.items():           # no task effect: values are noise
        d[v] = rng.normal(0, 1, len(d))
    st = co.task_variation([v], n_boot=50)
    assert st.attrs["scale"] == "omega2"
    assert abs(st.loc[0, "task_d"]) < .1
    shift = {t: k for k, t in enumerate(sorted({t for _, _, t in co.by_task}))}
    for (c, k, t), d in co.by_task.items():           # a large task effect
        d[v] = rng.normal(0, .3, len(d)) + 3 * shift[t]
    st = co.task_variation([v], n_boot=50)
    assert st.loc[0, "task_d"] > .5 and st.loc[0, "tier"] == "context-dependent"
    assert 0 <= st.loc[0, "task_p_tier"] <= 1 and st.loc[0, "task_mean_hi"] > st.loc[0, "task_mean_lo"]


def test_task_variation_is_the_mean_of_cohort_omegas():
    co = _toy_films()
    v = co.dyn[0]
    rng = np.random.default_rng(1)
    for (c, k, t), d in co.by_task.items():
        d[v] = rng.normal(0, 1, len(d)) + (0.8 if t.endswith("1") else 0)
    st = co.task_variation([v], n_boot=0)
    F = co.films(rest=True)
    oms = []
    for c in sorted({c for c, _ in F}):
        gs = [F[k][v].dropna().to_numpy() for k in F if k[0] == c]
        if len(gs) < 2:
            continue
        allv = np.concatenate(gs)
        ssb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in gs)
        ssw = sum(((g - g.mean()) ** 2).sum() for g in gs)
        msw = ssw / (len(allv) - len(gs))
        oms.append((ssb - (len(gs) - 1) * msw) / (ssb + ssw + msw))
    assert np.isclose(st.loc[0, "task_d"], np.mean(oms))
