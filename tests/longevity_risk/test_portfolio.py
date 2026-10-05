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


def test_two_sexes_share_the_trend_but_not_the_lifetimes(model, discount):
    # Same deterministic model for both sexes (no trend risk): the systematic values coincide, while the
    # idiosyncratic residuals of men and women are uncorrelated and the variance of the total equals the sum of
    # the individual annuity variances. Before the lifetimes were given separate streams, member i of each sex
    # received the same uniform, the residuals correlated 0.6-1.0 and the variance was overstated by about 60%.
    rw = lc.RandomWalkDrift(model.rw.drift, 0.0, 0.0, model.rw.k_first, model.rw.k_last, model.rw.first_year, model.rw.last_year)
    det = sim.ProjectionModel(model.age_min, model.ax, model.bx, rw)
    members = synthetic.pensioners(n=400, seed=5)
    members["sex"] = np.where(np.arange(len(members)) % 2 == 0, "M", "F")
    out = portfolio.simulate({"M": det, "F": det}, members, discount, n_scenarios=20_000, seed=3)
    resid_m, resid_f = out["M realised"] - out["M systematic"], out["F realised"] - out["F systematic"]
    assert abs(np.corrcoef(resid_m, resid_f)[0, 1]) < 0.03
    variance = sum(p**2 * lt.annuity_due_variance(det.central_q(a), discount) for a, p in zip(members["age"], members["pension"]))
    assert out["realised"].var(ddof=1) == pytest.approx(variance, rel=0.05)
    assert abs(out["realised"].mean() - out["systematic"].mean()) < 4 * np.sqrt(variance / out.shape[0])
    # The first sex (alphabetically) keeps the scenario's own stream, so a single-sex portfolio is unchanged.
    both = portfolio.simulate({"M": det, "F": det}, members, discount, n_scenarios=200, seed=3)
    women = portfolio.simulate({"F": det}, members[members["sex"] == "F"], discount, n_scenarios=200, seed=3)
    np.testing.assert_array_equal(women["F realised"], both["F realised"])
    np.testing.assert_array_equal(women["F systematic"], both["F systematic"])


def test_indexation_and_shock_identities_with_two_sexes(model, discount):
    rw = lc.RandomWalkDrift(model.rw.drift * 0.8, model.rw.sigma, model.rw.drift_se, model.rw.k_first, model.rw.k_last,
                            model.rw.first_year, model.rw.last_year)
    models = {"F": sim.ProjectionModel(model.age_min, model.ax - 0.3, model.bx, rw), "M": model}
    members = pd.DataFrame({"sex": ["M", "F", "F", "M"], "age": [60, 75, 75, 88], "pension": [10_000.0, 8_000.0, 4_000.0, 1_000.0]})
    g, horizon = 0.02, 70
    v = discount[:horizon]
    np.testing.assert_allclose(portfolio.indexed_discount(v, g), v * (1 + g) ** np.arange(horizon), rtol=1e-15)
    # Indexed cash flows discounted at v equal the members' annuities-due at the indexed discount factors.
    cf = portfolio.expected_cash_flows(models, members, horizon, indexation=g)
    be = sum(p * lt.annuity_due(models[s].central_q(a), portfolio.indexed_discount(discount, g))
             for s, a, p in zip(members["sex"], members["age"], members["pension"]))
    assert float(cf["total"].to_numpy() @ v) == pytest.approx(be, rel=1e-12)
    assert (cf["F"] + cf["M"]).equals(cf["total"])
    # The 0.8 multiplier reproduces the standard-formula shocked best estimate member by member.
    shocked = portfolio.expected_cash_flows(models, members, horizon, q_multiplier=0.8)
    be_shocked = sum(p * solvency.longevity_scr(models[s].central_q(a), discount)["shocked_best_estimate"]
                     for s, a, p in zip(members["sex"], members["age"], members["pension"]))
    assert float(shocked["total"].to_numpy() @ v) == pytest.approx(be_shocked, rel=1e-12)
    # One-year values: best estimate and scenario values are the pension-weighted sums over sexes and members
    # of the single-life one-year view at the indexed discount factors (same seed, hence the same streams).
    best, values = portfolio.one_year_values(models, members, discount, n_scenarios=300, seed=11, indexation=g)
    expected_best, expected_values = 0.0, np.zeros(300)
    for s, a, p in zip(members["sex"], members["age"], members["pension"]):
        single = sim.one_year_recalibration(models[s], int(a), portfolio.indexed_discount(discount, g), 300, seed=11)
        expected_best += p * single["best_estimate"]
        expected_values += p * single["value"]
    assert best == pytest.approx(expected_best, rel=1e-12)
    np.testing.assert_allclose(values, expected_values, rtol=1e-12)


def test_argument_validation(model, discount):
    members = pd.DataFrame({"sex": ["M"], "age": [45], "pension": [1.0]})
    with pytest.raises(ValueError):
        portfolio.expected_cash_flows({"M": model}, members, 70)
    with pytest.raises(ValueError):
        portfolio.simulate({"M": model}, members.assign(age=65), discount, n_scenarios=0)
    with pytest.raises(ValueError):
        portfolio.one_year_values({"M": model}, members.assign(age=65), discount, n_scenarios=0)


def test_explicit_longevity_scr_projection(model, discount):
    members = pd.DataFrame({"sex": ["M", "M", "M"], "age": [60, 75, 75], "pension": [10_000.0, 8_000.0, 4_000.0]})
    horizon = 70
    scr = portfolio.longevity_scr_runoff({"M": model}, members, discount, horizon)
    # At t = 0 the projection is the standard-formula shock on the whole book.
    base = portfolio.expected_cash_flows({"M": model}, members, horizon)["total"].to_numpy() @ discount[:horizon]
    shocked = portfolio.expected_cash_flows({"M": model}, members, horizon, q_multiplier=0.8)["total"].to_numpy() @ discount[:horizon]
    assert scr[0] == pytest.approx(shocked - base, rel=1e-12)
    # At t > 0, each age group contributes its expected survivors times the shock on the remaining annuity,
    # computed independently with solvency.longevity_scr on the tail of the cohort's death probabilities.
    for t in (1, 12, 30):
        expected = 0.0
        for age, amount in ((60, 10_000.0), (75, 12_000.0)):
            q = model.central_q(age)
            if t < q.size:
                S_t = np.prod(1.0 - q[:t])
                expected += amount * S_t * solvency.longevity_scr(q[t:], discount[t:] / discount[t])["scr"]
        assert scr[t] == pytest.approx(expected, rel=1e-12)
    assert scr[model.omega - 60 + 1:].sum() == 0.0          # nobody is in force after the closing age
    # The shock bites harder on older lives, so SCR(t)/BE(t) rises: the proxy SCR(0) BE(t)/BE(0) understates it.
    be = solvency.best_estimate_runoff(portfolio.expected_cash_flows({"M": model}, members, horizon)["total"], discount[:horizon])
    ratio = scr[:30] / be[:30]
    assert np.all(np.diff(ratio) > 0)
    rm_proxy = solvency.risk_margin(scr[0], be, discount[:horizon + 1])
    rm_explicit = solvency.risk_margin_from_path(scr, discount[:horizon + 1])
    assert rm_explicit > rm_proxy
    # With SCR(t) proportional to BE(t) by construction the two risk margins coincide.
    assert solvency.risk_margin_from_path(scr[0] * be / be[0], discount[:horizon + 1]) == pytest.approx(rm_proxy, rel=1e-14)
    with pytest.raises(ValueError):
        solvency.risk_margin_from_path(scr, discount[:horizon])


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


def test_interest_rate_scr_is_zero_when_both_shocks_lower_the_liability():
    # Negative zero rates: the down shock leaves them unchanged and the up shock adds one percentage point, so
    # both shocked values are at or below the base value and the SCR is floored at zero.
    out = solvency.interest_rate_scr([0.0] * 10 + [100.0], np.arange(11), np.full(11, -0.005))
    assert out["down"] == pytest.approx(out["base"]) and out["up"] < out["base"] and out["scr"] == 0.0
    assert out["up"] == pytest.approx(100 * 1.005 ** -10)


def test_risk_margin_and_aggregation():
    v = 1.02 ** -np.arange(0, 12.0)
    be = np.full(10, 50.0)
    assert solvency.risk_margin(8.0, be, v) == pytest.approx(0.06 * 8.0 * np.sum(v[1:11]))
    assert solvency.risk_margin(8.0, be, v[:11]) == pytest.approx(0.06 * 8.0 * np.sum(v[1:11]))   # exactly N + 1 factors
    with pytest.raises(ValueError):   # a shorter curve used to drop terms silently (1.83 instead of 4.31 here)
        solvency.risk_margin(8.0, be, v[:8])
    # Solvency II review (Delegated Regulation (EU) 2026/269, from 30 January 2027): time-dependent factor
    # max(lambda^t, floor) and a lower cost of capital; lambda = 1 and floor = 0 reproduce the current formula.
    # Amended Article 37(1): max(0.96^t; 50%), reached after 17 years (0.96^17 = 0.4996); CoC 4.75%.
    factor = np.maximum(0.96 ** np.arange(10), 0.5)
    assert solvency.REVIEW_LAMBDA ** 17 < solvency.REVIEW_FLOOR < solvency.REVIEW_LAMBDA ** 16
    review = solvency.risk_margin(8.0, be, v, coc=solvency.REVIEW_COST_OF_CAPITAL, lambda_=solvency.REVIEW_LAMBDA,
                                  floor=solvency.REVIEW_FLOOR)
    assert review == pytest.approx(0.0475 * 8.0 * np.sum(factor * v[1:11]), rel=1e-12)
    assert review < solvency.risk_margin(8.0, be, v)
    assert solvency.risk_margin(8.0, be, v, lambda_=1.0, floor=0.0) == solvency.risk_margin(8.0, be, v)
    with pytest.raises(ValueError):
        solvency.risk_margin(8.0, be, v, lambda_=1.5)
    assert solvency.basic_scr(3.0, 4.0) == pytest.approx(np.sqrt(9 + 16 + 2 * 0.25 * 12))
    assert solvency.operational_scr(10.0, 1000.0) == pytest.approx(3.0)   # capped at 30% of the BSCR
    assert solvency.operational_scr(100.0, 1000.0) == pytest.approx(4.5)  # 0.45% of technical provisions
    np.testing.assert_allclose(solvency.best_estimate_runoff([1.0, 1.0, 1.0], [1.0, 0.5, 0.25]), [1.75, 1.5, 1.0])
