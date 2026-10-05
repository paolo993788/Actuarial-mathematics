"""Experience rating: maximum likelihood, Gamma-Poisson credibility, multipliers from life expectancy and the
simulated experience of a fund."""

import numpy as np
import pandas as pd
import pytest

from longevity_risk import data, experience, lee_carter as lc, synthetic
from longevity_risk.life_table import period_life_expectancy


def test_actual_to_expected_is_the_poisson_mle():
    out = experience.actual_to_expected([30, 50, 20], [40.0, 50.0, 35.0])
    assert out["ratio"] == pytest.approx(100 / 125)
    assert out["se"] == pytest.approx(np.sqrt(100) / 125) and out["relative_se"] == pytest.approx(0.1)
    # The score of the Poisson likelihood sum(D log(theta E) - theta E) vanishes at the estimate.
    assert 100 / out["ratio"] - 125 == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        experience.actual_to_expected([1], [0.0])


def test_credibility_is_the_gamma_poisson_posterior():
    mean, sd, D, E = 0.9, 0.06, 140, 170.0
    a, b = experience.gamma_prior(mean, sd)
    assert a / b == pytest.approx(mean) and np.sqrt(a) / b == pytest.approx(sd)
    out = experience.credibility(D, E, mean, sd)
    # Buhlmann-Straub weight Z = E / (E + mu / tau^2) and the conjugate posterior mean coincide exactly.
    assert out["Z"] == pytest.approx(E / (E + mean / sd**2), rel=1e-14)
    assert out["estimate"] == pytest.approx(out["Z"] * D / E + (1 - out["Z"]) * mean, rel=1e-14)
    assert out["estimate"] == pytest.approx((a + D) / (b + E), rel=1e-14)
    assert out["posterior_sd"] == pytest.approx(np.sqrt(a + D) / (b + E), rel=1e-14)
    none = experience.credibility(0, 0.0, mean, sd)
    assert none["Z"] == 0.0 and none["estimate"] == pytest.approx(mean) and none["posterior_sd"] == pytest.approx(sd)
    large = experience.credibility(0.8e7, 1e7, mean, sd)
    assert large["Z"] > 0.999 and large["estimate"] == pytest.approx(0.8, abs=1e-3)


def test_multiplier_for_life_expectancy():
    # Constant force: the period life expectancy is 1 / (theta mu) (open last age), so theta = 1 / (mu e).
    m = np.full(35, 0.04)
    assert experience.multiplier_for_life_expectancy(m, 20.0) == pytest.approx(1 / (0.04 * 20), rel=1e-10)
    rates = np.exp(-4.5 + 0.1 * np.arange(35))
    e = period_life_expectancy(rates)
    assert experience.multiplier_for_life_expectancy(rates, e) == pytest.approx(1.0, rel=1e-10)
    assert experience.multiplier_for_life_expectancy(rates, e + 1.0) < 1.0     # longer life, lower rates


def test_education_multipliers_from_a_sample_document(monkeypatch):
    D, E, _ = synthetic.deaths_and_exposures(seed=3)
    fit = lc.fit_poisson(D, E)
    i = int(np.searchsorted(fit.ages, 65))
    base = {y: period_life_expectancy(np.exp(fit.ax[i:] + fit.bx[i:] * k)) for y, k in zip(fit.years, fit.kt)}
    # A JSON-stat document as returned by the Eurostat API for two years and three groups.
    doc = {"id": ["isced11", "time"], "size": [3, 2],
           "dimension": {"isced11": {"category": {"index": {"TOTAL": 0, "ED0-2": 1, "ED5-8": 2}}},
                         "time": {"category": {"index": {"2010": 0, "2011": 1}}}},
           # flat index = position of isced11 x 2 + position of time (the last dimension varies fastest)
           "value": {"0": base[2010], "1": base[2011], "2": base[2010] - 0.6,
                     "3": base[2011] - 0.6, "4": base[2010] + 1.2, "5": base[2011] + 1.2}}
    monkeypatch.setattr(data, "load_eurostat", lambda dataset, refresh=False, **f: data.parse_jsonstat(doc))
    table = experience.load_education_life_expectancy("IT", "M")
    assert list(table.index) == [2010, 2011] and set(table.columns) == {"TOTAL", "ED0-2", "ED5-8"}
    mult = experience.education_multipliers(fit, table)
    np.testing.assert_allclose(mult["TOTAL"], 1.0, rtol=1e-12)
    assert (mult["ED0-2"] > 1).all() and (mult["ED5-8"] < 1).all()
    for year in (2010, 2011):    # the multiplier reproduces the group's life expectancy on the fitted rates
        k = fit.kt[list(fit.years).index(year)]
        e = period_life_expectancy(mult.loc[year, "ED5-8"] * np.exp(fit.ax[i:] + fit.bx[i:] * k))
        assert e == pytest.approx(base[year] + 1.2, rel=1e-10)


def test_simulated_experience_recovers_the_multiplier():
    D, E, _ = synthetic.deaths_and_exposures(seed=3)
    fit = lc.fit_poisson(D, E)
    ages = synthetic.pensioners(n=1500, seed=11)["age"]
    years = list(fit.years[-5:])
    one = experience.simulate_experience(fit, ages, years, 0.85, seed=1)
    assert list(one.index) == years and (one["deaths"] >= 0).all()
    pd.testing.assert_frame_equal(one, experience.simulate_experience(fit, ages, years, 0.85, seed=1))
    ratios = np.array([experience.actual_to_expected(*experience.simulate_experience(fit, ages, years, 0.85, seed=s)[
        ["deaths", "expected"]].to_numpy().T)["ratio"] for s in range(400)])
    # Each ratio has a standard error of about 0.85 / sqrt(deaths); the mean of 400 is within 4 of its errors.
    se = np.sqrt(0.85 / one["expected"].sum() / 400)
    assert abs(ratios.mean() - 0.85) < 4 * se
    assert ratios.std() == pytest.approx(np.sqrt(0.85 / one["expected"].sum()), rel=0.15)
    with pytest.raises(ValueError):
        experience.simulate_experience(fit, [40], years, 0.85, seed=1)
