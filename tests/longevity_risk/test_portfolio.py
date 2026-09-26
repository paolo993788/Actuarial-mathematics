"""Portfolio valuation, Solvency II interest-rate shocks and the risk margin."""

import numpy as np
import pandas as pd
import pytest

from longevity_risk import lee_carter as lc
from longevity_risk import life_table as lt
from longevity_risk import portfolio, require_cpp, simulation as sim, solvency, synthetic

core = require_cpp()


@pytest.fixture(scope="module")
def model():
    D, E, _ = synthetic.deaths_and_exposures(seed=31)
    return sim.ProjectionModel.from_fit(lc.fit_poisson(D, E))


@pytest.fixture(scope="module")
def discount():
    return 1.03 ** -np.arange(0, 100.0)


def test_single_member_portfolio_reproduces_single_annuity(model, discount):
    single = sim.simulate_annuity(model, 70, discount, n_scenarios=500, seed=4)["pv_systematic"]
    port = core.simulate_portfolio(*model._args(True), np.array([70], np.int32), np.array([12_000.0]), discount, 500, False, 4, 0)
    np.testing.assert_allclose(port["pv_systematic"], 12_000.0 * single, rtol=1e-13)


def test_portfolio_one_year_is_the_weighted_sum_of_members(model, discount):
    ages, amounts = np.array([65, 72, 72, 88], np.int32), np.array([10_000.0, 5_000.0, 7_000.0, 20_000.0])
    port = core.portfolio_one_year(*model._args(True), model.rw.k_first, model.rw.n_increments, ages, amounts, discount, 300, 9, 0)
    expected = np.zeros(300)
    for age, amount in zip(ages, amounts):
        expected += amount * sim.one_year_recalibration(model, int(age), discount, 300, seed=9)["value"]
    np.testing.assert_allclose(port["value"], expected, rtol=1e-12)


def test_cash_flows_reproduce_best_estimates(model, discount):
    members = pd.DataFrame({"sex": ["M", "M", "M"], "age": [60, 75, 75], "pension": [10_000.0, 8_000.0, 4_000.0]})
    cf = portfolio.expected_cash_flows({"M": model}, members, horizon=70)
    be = sum(p * model.best_estimate(a, discount) for a, p in zip(members["age"], members["pension"]))
    assert float(cf["total"].to_numpy() @ discount[:70]) == pytest.approx(be, rel=1e-12)


def test_idiosyncratic_moments_for_a_heterogeneous_portfolio(model, discount):
    rw = lc.RandomWalkDrift(model.rw.drift, 0.0, 0.0, model.rw.k_first, model.rw.k_last, model.rw.first_year, model.rw.last_year)
    det = sim.ProjectionModel(model.age_min, model.ax, model.bx, rw)
    members = synthetic.pensioners(n=300, seed=2).assign(sex="M")
    out = portfolio.simulate({"M": det}, members, discount, n_scenarios=20_000, seed=3)
    variance = sum(p**2 * lt.annuity_due_variance(det.central_q(a), discount) for a, p in zip(members["age"], members["pension"]))
    assert out["systematic"].std() == pytest.approx(0.0, abs=1e-6 * out["systematic"].mean())
    se = np.sqrt(variance / out.shape[0])
    assert abs(out["realised"].mean() - out["systematic"].mean()) < 4 * se
    assert out["realised"].var(ddof=1) == pytest.approx(variance, rel=0.05)


def test_interest_rate_shock_table():
    assert solvency.relative_shock(10, solvency.IR_UP) == pytest.approx(0.42)
    assert solvency.relative_shock(55, solvency.IR_UP) == pytest.approx(0.26 - 0.06 * 35 / 70)
    assert solvency.relative_shock(120, solvency.IR_DOWN) == pytest.approx(0.20)
    up = solvency.shocked_zero_rates([1, 10, 30], [0.005, 0.03, -0.01], "up")
    np.testing.assert_allclose(up, [0.015, 0.03 * 1.42, 0.0])  # at least +1 pp; negative rates get +1 pp
    down = solvency.shocked_zero_rates([1, 10, 30], [0.02, 0.03, -0.01], "down")
    np.testing.assert_allclose(down, [0.02 * 0.25, 0.03 * 0.69, -0.01])  # negative rates are not shocked


def test_interest_rate_scr_single_cash_flow():
    out = solvency.interest_rate_scr([0.0] * 10 + [100.0], np.arange(11), np.full(11, 0.03))
    assert out["base"] == pytest.approx(100 * 1.03**-10)
    assert out["scr"] == pytest.approx(out["down"] - out["base"])
    assert out["down"] == pytest.approx(100 * (1 + 0.03 * 0.69) ** -10)


def test_risk_margin_and_aggregation():
    v = 1.02 ** -np.arange(0, 12.0)
    be = np.full(10, 50.0)
    assert solvency.risk_margin(8.0, be, v) == pytest.approx(0.06 * 8.0 * np.sum(v[1:11]))
    assert solvency.basic_scr(3.0, 4.0) == pytest.approx(np.sqrt(9 + 16 + 2 * 0.25 * 12))
    assert solvency.operational_scr(10.0, 1000.0) == pytest.approx(3.0)   # capped at 30% of the BSCR
    assert solvency.operational_scr(100.0, 1000.0) == pytest.approx(4.5)  # 0.45% of technical provisions
    np.testing.assert_allclose(solvency.best_estimate_runoff([1.0, 1.0, 1.0], [1.0, 0.5, 0.25]), [1.75, 1.5, 1.0])
