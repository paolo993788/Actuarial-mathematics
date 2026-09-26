"""Lee-Carter estimation: exact recovery, identifiability constraints,
goodness of fit under the true model and the random walk with drift."""

import numpy as np
import pandas as pd
import pytest

from longevity_risk import lee_carter as lc
from longevity_risk import synthetic


def noise_free():
    D, E, true = synthetic.deaths_and_exposures(seed=3)
    rates = np.exp(true["ax"][:, None] + true["bx"][:, None] * true["kt"][None, :])
    return pd.DataFrame(E.to_numpy() * rates, index=D.index, columns=D.columns), E, true


def test_poisson_fit_recovers_noise_free_parameters():
    D, E, true = noise_free()
    fit = lc.fit_poisson(D, E)
    # Expected deaths equal to the model: the MLE reproduces the parameters
    # up to the convergence tolerance on the log-rates (1e-10).
    np.testing.assert_allclose(fit.ax, true["ax"], atol=1e-8)
    np.testing.assert_allclose(fit.bx, true["bx"], atol=1e-8)
    np.testing.assert_allclose(fit.kt, true["kt"], atol=1e-6)


def test_svd_fit_is_exact_on_rank_one_log_rates():
    D, E, true = noise_free()
    fit = lc.fit_svd(np.log(D / E))
    np.testing.assert_allclose(fit.ax, true["ax"], atol=1e-10)
    np.testing.assert_allclose(fit.bx, true["bx"], atol=1e-10)
    np.testing.assert_allclose(fit.kt, true["kt"], atol=1e-8)


def test_identifiability_constraints():
    D, E, _ = synthetic.deaths_and_exposures(seed=5)
    for fit in (lc.fit_poisson(D, E), lc.fit_svd(np.log(D / E))):
        assert fit.bx.sum() == pytest.approx(1.0, abs=1e-12)
        assert fit.kt.sum() == pytest.approx(0.0, abs=1e-9)


def test_deviance_matches_degrees_of_freedom_under_true_model():
    D, E, _ = synthetic.deaths_and_exposures(seed=11)
    fit = lc.fit_poisson(D, E)
    n_ages, n_years = D.shape
    dof = n_ages * n_years - (2 * n_ages + n_years - 2)
    # Deviance ~ chi-square(dof): mean dof, standard deviation sqrt(2 dof) ~ 60 here.
    assert abs(fit.deviance - dof) < 4 * np.sqrt(2 * dof)


def test_excluded_years_have_no_period_index():
    D, E, _ = synthetic.deaths_and_exposures(seed=13)
    excluded = {2020, 2021}
    fit = lc.fit_poisson(D, E, fit_years=[y for y in D.columns if y not in excluded])
    assert np.isnan(fit.kt[np.isin(fit.years, list(excluded))]).all()
    assert np.nansum(fit.kt) == pytest.approx(0.0, abs=1e-9)


def test_random_walk_estimates():
    years = np.arange(1950, 2024)
    rng = np.random.default_rng(2)
    k = np.concatenate([[10.0], 10.0 + np.cumsum(-1.3 + 0.9 * rng.standard_normal(years.size - 1))])
    rw = lc.RandomWalkDrift.from_kt(years, k)
    assert rw.drift == pytest.approx((k[-1] - k[0]) / (years.size - 1))
    assert rw.sigma == pytest.approx(np.std(np.diff(k), ddof=1), rel=1e-12)
    assert rw.drift_se == pytest.approx(rw.sigma / np.sqrt(years.size - 1))
    assert abs(rw.drift + 1.3) < 4 * rw.drift_se
    np.testing.assert_allclose(rw.central_path(3), k[-1] + rw.drift * np.array([1, 2, 3]))


def test_random_walk_with_gap_uses_endpoints():
    years = np.arange(2000, 2011)
    k = -1.0 * (years - 2000.0) + np.array([0, .3, -.2, .1, 0, 0, 0, .2, -.1, .4, 0])
    k_gap = k.copy()
    k_gap[[4, 5]] = np.nan
    rw = lc.RandomWalkDrift.from_kt(years, k_gap)
    assert rw.drift == pytest.approx((k[-1] - k[0]) / 10)
    assert rw.n_increments == 10
