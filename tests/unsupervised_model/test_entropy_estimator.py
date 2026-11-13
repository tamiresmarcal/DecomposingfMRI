"""Miller-Madow entropy rate: the formula, the table recompute, idempotence."""
import numpy as np
import pandas as pd
import pytest

from fmri_decomposition.unsupervised_model import transitions as tr


def _counts():
    return np.array([[5, 2, 0], [1, 3, 1], [0, 0, 4]], dtype=float)


def test_miller_madow_adds_m_minus_r_over_2n_ln2():
    c = _counts()
    plug = tr._entropy_rate(c)
    mm = tr._entropy_rate(c, "miller_madow")
    m, r, n = (c > 0).sum(), (c.sum(1) > 0).sum(), c.sum()
    assert mm - plug == pytest.approx((m - r) / (2 * n * np.log(2)))
    assert mm > plug


def test_unknown_estimator_refused():
    with pytest.raises(ValueError):
        tr._entropy_rate(_counts(), "shrinkage")


def test_table_recompute_matches_and_is_idempotent():
    c = _counts()
    K = 3
    row = dict(zip(tr.cell_names(K), (c.ravel() / c.sum()).tolist()))
    row["n_transitions"] = c.sum()
    row["entropy_rate_bits"] = tr._entropy_rate(c)
    d = pd.DataFrame([row, {**row}])
    once = tr.entropy_from_table(d, K, "miller_madow")
    assert np.allclose(once, tr._entropy_rate(c, "miller_madow"))
    d["entropy_rate_bits"] = once
    assert np.allclose(tr.entropy_from_table(d, K, "miller_madow"), once)
    assert np.allclose(tr.entropy_from_table(d, K, "plugin"), tr._entropy_rate(c))
