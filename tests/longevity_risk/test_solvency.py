"""Solvency II longevity shock."""

import numpy as np
import pytest

from longevity_risk import solvency


def test_longevity_shock_closed_form():
    q, i, H = 0.02, 0.03, 80
    qs = np.full(H, q)
    qs[-1] = 1.0
    v = (1 + i) ** -np.arange(H)

    def geometric(qq):
        x = (1 - qq) / (1 + i)
        return (1 - x**H) / (1 - x)

    out = solvency.longevity_scr(qs, v)
    assert out["best_estimate"] == pytest.approx(geometric(q), rel=1e-12)
    assert out["shocked_best_estimate"] == pytest.approx(geometric(0.8 * q), rel=1e-12)
    assert out["scr"] > 0


def test_closing_age_is_not_shocked():
    qs = np.array([0.1, 0.2, 1.0])
    out = solvency.longevity_scr(qs, np.ones(3))
    # S = 1, 0.92, 0.92 * 0.84 -> annuity-due of 1 with v = 1
    assert out["shocked_best_estimate"] == pytest.approx(1 + 0.92 + 0.92 * 0.84)
