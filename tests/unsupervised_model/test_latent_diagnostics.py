"""Notebook 08's latent-space diagnostics: the per-person z-score, the variance shares, and the
sampling and scaling of tools/latent_diagnostics.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from fmri_decomposition.unsupervised_model import cohort_tables as ct

TOOL = Path(__file__).resolve().parents[2] / "tools" / "latent_diagnostics.py"
spec = importlib.util.spec_from_file_location("latent_diagnostics", TOOL)
ld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ld)


def toy(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c, off in (("a", 0.0), ("b", 5.0)):
        for s in range(4):
            centre = rng.normal(off, 2, 3)
            for t in ("m1", "m2"):
                shift = 1.0 if t == "m2" else 0.0
                for _ in range(30):
                    rows.append([c, f"S{s}", t, *(centre + shift + rng.normal(0, .3, 3))])
    E = pd.DataFrame(rows, columns=["cohort", "sub", "task", "x", "y", "z"])
    E["kind"] = "movie"
    return E


def test_zscore_within_leaves_every_person_centred_and_unit():
    Z = ct.zscore_within(toy())
    g = Z.groupby(["cohort", "sub"])[["x", "y", "z"]]
    assert np.allclose(g.mean().to_numpy(), 0, atol=1e-9)
    assert np.allclose(g.std().to_numpy(), 1, atol=1e-9)


def test_variance_shares_see_cohort_person_and_task():
    sh = ct.variance_shares(toy()).iloc[0]
    assert sh["cohort"] > .5                       # the cohorts sit 5 apart
    assert sh["person"] > .5                       # people spread by 2 inside a cohort
    assert .3 < sh["task | person"] < 1            # m2 moved by 1 against noise .3
    after = ct.variance_shares(ct.zscore_within(toy())).iloc[0]
    assert after["cohort"] < 1e-9 and after["person"] < 1e-9
    assert after["task | person"] > .3             # the task survives the per-person z-score


def test_sample_caps_each_person_and_is_repeatable():
    ident = pd.DataFrame({"sub": np.repeat(["A", "B", "C"], [10, 100, 3])})
    s = ld.sample(ident, 20, seed=1)
    n = ident.iloc[s]["sub"].value_counts()
    assert n["A"] == 10 and n["B"] == 20 and n["C"] == 3
    assert np.array_equal(s, ld.sample(ident, 20, seed=1))


def test_zscore_by_scales_each_group_on_its_own():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (50, 4)), rng.normal(10, 5, (50, 4))]).astype(np.float32)
    g = np.repeat(["p", "q"], 50)
    ld.zscore_by(g, X)
    for k in ("p", "q"):
        assert np.allclose(X[g == k].mean(0), 0, atol=1e-4)
        assert np.allclose(X[g == k].std(0), 1, atol=1e-3)


def test_with_states_marks_unlabelled_windows():
    ident = pd.DataFrame({"task": ["m", "m", "m"], "sub": ["A", "A", "A"], "window_id": [0, 1, 2]})
    lab = pd.DataFrame({"task": ["m", "m"], "sub": ["A", "A"], "window_id": [0, 2], "state": [3, 5]})
    assert ld.with_states(ident, lab)["state"].tolist() == [3, -1, 5]
    assert ld.with_states(ident, None)["state"].tolist() == [-1, -1, -1]
