"""Life-table functions against closed-form results."""

import numpy as np
import pytest

from longevity_risk import life_table as lt


def test_constant_force_life_expectancy():
    # Constant force mu at every age (open group closed with the same mu): e = 1 / mu.
    assert lt.period_life_expectancy(np.full(60, 0.04)) == pytest.approx(25.0, rel=1e-12)


def test_annuity_due_geometric_closed_form():
    q, i, H = 0.03, 0.02, 60
    qs = np.full(H, q)
    qs[-1] = 1.0
    v = (1 + i) ** -np.arange(H)
    x = (1 - q) / (1 + i)
    assert lt.annuity_due(qs, v) == pytest.approx((1 - x**H) / (1 - x), rel=1e-13)


def test_annuity_variance_by_enumeration():
    rng = np.random.default_rng(0)
    q = np.concatenate([rng.uniform(0.01, 0.3, 30), [1.0]])
    v = 1.03 ** -np.arange(q.size)
    S = lt.survival_from_q(q)
    prob = S[:-1] * q
    assert prob.sum() == pytest.approx(1.0, abs=1e-14)
    pv = np.cumsum(v)
    mean = np.sum(prob * pv)
    assert mean == pytest.approx(lt.annuity_due(q, v), rel=1e-13)
    assert lt.annuity_due_variance(q, v) == pytest.approx(np.sum(prob * (pv - mean) ** 2), rel=1e-10)


def test_curtate_life_expectancy():
    q = np.array([0.5, 0.5, 1.0])
    # S = 1, 0.5, 0.25, 0 -> e = 0.75
    assert lt.curtate_life_expectancy(q) == pytest.approx(0.75)


def test_period_life_expectancy_is_vectorised():
    rng = np.random.default_rng(1)
    m = rng.uniform(0.01, 0.4, size=(5, 40))
    np.testing.assert_allclose(lt.period_life_expectancy(m), [lt.period_life_expectancy(row) for row in m], rtol=1e-14)
