"""Li-Lee multi-population model and index hedging: offsets, constraints, dynamics, exact deterministic
limits and hedge-effectiveness bounds."""

import numpy as np
import pandas as pd
import pytest

from longevity_risk import lee_carter as lc
from longevity_risk import multipopulation as mp
from longevity_risk import synthetic
from longevity_risk.life_table import annuity_due, q_from_m

COUNTRIES = ("AA", "BB", "CC", "DD")


@pytest.fixture(scope="module")
def panel():
    D, E = mp.load_panel(COUNTRIES, ages=range(50, 90), first_year=1980, last_year=2024, official=False)
    D["BB+CC+DD"], E["BB+CC+DD"] = mp.aggregate(D, E, COUNTRIES[1:])
    return D, E


@pytest.fixture(scope="module")
def fitted(panel):
    D, E = panel
    fit = mp.fit_li_lee(D, E, COUNTRIES, fit_years=[y for y in D["AA"].columns if y not in (2020, 2021)])
    return fit, mp.fit_dynamics(fit)


def test_offset_fit_shifts_the_level_exactly():
    D, E, _ = synthetic.deaths_and_exposures(seed=5)
    plain = lc.fit_poisson(D, E)
    shifted = lc.fit_poisson(D, E, offset=np.full(D.shape, 0.3))
    np.testing.assert_allclose(shifted.ax, plain.ax - 0.3, atol=1e-8)
    np.testing.assert_allclose(shifted.bx, plain.bx, atol=1e-8)
    np.testing.assert_allclose(shifted.kt, plain.kt, atol=1e-6)
    assert shifted.deviance == pytest.approx(plain.deviance, rel=1e-8)


def test_li_lee_constraints_and_deviance_ordering(panel, fitted):
    D, _ = panel
    fit, _ = fitted
    assert fit.B.sum() == pytest.approx(1.0) and np.nansum(fit.K) == pytest.approx(0.0, abs=1e-8)
    assert np.isnan(fit.K[list(fit.years).index(2020)])
    for name in D:
        assert fit.b[name].sum() == pytest.approx(1.0)
        assert np.nansum(fit.k[name]) == pytest.approx(0.0, abs=1e-8)
    dev = fit.deviance
    assert (dev["Li-Lee"] <= dev["common factor only"] + 1e-6).all()
    assert ((dev["R common"] > 0) & (dev["R Li-Lee"] <= 1) & (dev["R Li-Lee"] >= dev["R common"])).all()


def test_log_rates_reproduce_the_crude_rates_closely(panel, fitted):
    D, E = panel
    fit, _ = fitted
    mask = fit.mask
    resid = (np.log(D["AA"] / E["AA"]) - fit.log_rates("AA")).to_numpy()[:, mask]
    assert np.abs(resid.mean()) < 0.01 and resid.std() < 0.1   # Poisson noise on about 10^3 deaths per cell


def test_dynamics_recover_mean_reversion_and_cap_phi(fitted):
    fit, dyn = fitted
    phis = np.array([dyn.phi_unrestricted[c] for c in COUNTRIES])
    assert np.all((phis > 0.5) & (phis < 1.05))                # simulated with phi = 0.8, 42 increments
    assert all(dyn.phi[c] <= 0.98 for c in dyn.phi)
    assert dyn.n_obs == len(fit.years) - 1 - 3                 # pairs 2019-20, 2020-21 and 2021-22 are dropped
    rw = mp.fit_dynamics(fit, random_walk_deviations=True)
    assert all(v == 1.0 for v in rw.phi.values())
    np.testing.assert_allclose(np.diag(dyn.corr.to_numpy()), 1.0)


def test_simulation_without_noise_equals_the_central_projection(fitted):
    fit, dyn = fitted
    quiet = mp.LiLeeDynamics(dyn.drift, 0.0, dyn.phi, dyn.cov * 0.0, dyn.K_last, dyn.k_last, dyn.last_year, dyn.n_obs)
    states = mp.simulate_states(quiet, ["AA"], 30, 5, seed=1)
    K, k = mp.central_states(quiet, "AA", 30)
    np.testing.assert_allclose(states["K"], np.broadcast_to(K, (5, 30)), atol=1e-12)
    np.testing.assert_allclose(states["AA"], np.broadcast_to(k, (5, 30)), atol=1e-12)


def test_split_liability_equals_the_annuity_on_the_central_path(fitted):
    fit, dyn = fitted
    quiet = mp.LiLeeDynamics(dyn.drift, 0.0, dyn.phi, dyn.cov * 0.0, dyn.K_last, dyn.k_last, dyn.last_year, dyn.n_obs)
    v = 1.03 ** -np.arange(0, 80.0)
    states = mp.simulate_states(quiet, ["AA"], 60, 3, seed=2)
    value = mp.liability_values(fit, quiet, "AA", states, 65, 10, v)
    ages, a, B, b = mp.extended_parameters(fit, "AA", 110)
    K, k = mp.central_states(quiet, "AA", 110 - 65 + 1)
    idx = 65 - ages[0] + np.arange(K.size)
    q = q_from_m(np.exp(a[idx] + B[idx] * K + b[idx] * k))
    q[-1] = 1.0
    np.testing.assert_allclose(value, annuity_due(q, v), rtol=1e-12)


def test_index_rates_and_hedge_effectiveness_bounds(fitted):
    fit, dyn = fitted
    states = mp.simulate_states(dyn, ["AA", "BB"], 12, 4000, seed=3)
    q = mp.index_death_rates(fit, "BB", states, [70, 80], 10)
    i = np.searchsorted(fit.ages, [70, 80])
    manual = q_from_m(np.exp(fit.a["BB"][i] + fit.B[i] * states["K"][:, 9, None] + fit.b["BB"][i] * states["BB"][:, 9, None]))
    np.testing.assert_allclose(q, manual, rtol=1e-12)
    rng = np.random.default_rng(4)
    X1, X2 = rng.standard_normal((2, 5000, 2))
    exact = mp.hedge_effectiveness(3 - X1 @ [2, 1], X1, 3 - X2 @ [2, 1], X2)
    assert exact["variance reduction"] == pytest.approx(1.0) and np.allclose(exact["notionals"], [-2, -1])
    noise = mp.hedge_effectiveness(rng.standard_normal(5000), X1, rng.standard_normal(5000), X2)
    assert abs(noise["variance reduction"]) < 0.01


def test_temporary_life_expectancy():
    assert mp.temporary_life_expectancy(np.full(25, 1e-12)) == pytest.approx(25.0, rel=1e-9)
    m = np.full(25, 0.05)
    assert mp.temporary_life_expectancy(m) == pytest.approx((1 - np.exp(-0.05 * 25)) / 0.05, rel=1e-12)


def test_sampling_risk_lowers_hedge_effectiveness(fitted):
    fit, dyn = fitted
    v = 1.03 ** -np.arange(0, 80.0)
    cal, test = (mp.simulate_states(dyn, ["AA"], 60, 3000, seed=s) for s in (5, 6))
    X = [mp.index_death_rates(fit, "AA", st, [70, 75, 80, 85], 10) for st in (cal, test)]
    large = mp.hedge_effectiveness(*(mp.liability_values(fit, dyn, "AA", st, 65, 10, v) for st in (cal,)), X[0],
                                   mp.liability_values(fit, dyn, "AA", test, 65, 10, v), X[1])
    small = mp.hedge_effectiveness(mp.liability_values(fit, dyn, "AA", cal, 65, 10, v, n_lives=500, seed=1), X[0],
                                   mp.liability_values(fit, dyn, "AA", test, 65, 10, v, n_lives=500, seed=2), X[1])
    assert large["variance reduction"] > 0.9 > small["variance reduction"]
