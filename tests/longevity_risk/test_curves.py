"""Svensson and Smith-Wilson curves."""

import numpy as np
import pytest

from longevity_risk import curves

SVENSSON = curves.SvenssonCurve(2.1, 1.6, -2.5, 3.0, 1.2, 9.5)


def test_smith_wilson_fits_liquid_prices_exactly():
    sw = curves.smith_wilson_from_svensson(SVENSSON, ufr=0.033)
    u = np.arange(1, 21)
    np.testing.assert_allclose(sw.discount(u), SVENSSON.discount(u), rtol=1e-12)
    assert sw.discount(0.0)[0] == pytest.approx(1.0, abs=1e-14)


def test_smith_wilson_convergence_criterion_and_minimal_alpha():
    sw = curves.smith_wilson_from_svensson(SVENSSON, ufr=0.033)
    assert sw.convergence_point == 60.0
    assert sw.convergence_gap() <= 1e-4 + 1e-12
    assert sw.alpha >= 0.05
    if sw.alpha > 0.05:
        slower = curves.SmithWilson(sw.maturities, sw.prices, sw.ufr, sw.alpha - 1e-4)
        assert slower.convergence_gap() > 1e-4
    # Far beyond the convergence point the forward rate is the UFR.
    assert sw.forward_rate(150.0)[0] == pytest.approx(0.033, abs=1e-6)


def test_flat_curve_at_ufr_is_reproduced():
    # If market prices already follow the UFR, Smith-Wilson adds nothing.
    u = np.arange(1, 21, dtype=float)
    sw = curves.SmithWilson(u, 1.033 ** -u, 0.033, 0.1)
    # Rounding in the right-hand side (~1e-16) is amplified by the conditioning of the Wilson matrix.
    np.testing.assert_allclose(sw.zeta, 0.0, atol=1e-10)
    np.testing.assert_allclose(sw.zero_rate([5, 50, 100]), 0.033, rtol=1e-12)


def test_svensson_short_and_long_limits():
    assert SVENSSON.zero_rate(1e-9) == pytest.approx((2.1 + 1.6) / 100, abs=1e-8)
    assert SVENSSON.zero_rate(1e5) == pytest.approx(2.1 / 100, abs=1e-4)
