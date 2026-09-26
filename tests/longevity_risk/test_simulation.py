"""C++ Monte Carlo engine against deterministic values, exact moments and
the NumPy reference implementation."""

import numpy as np
import pytest

from longevity_risk import lee_carter as lc
from longevity_risk import life_table as lt
from longevity_risk import require_cpp, simulation as sim, synthetic

core = require_cpp()


@pytest.fixture(scope="module")
def model():
    D, E, _ = synthetic.deaths_and_exposures(seed=21)
    fit = lc.fit_poisson(D, E)
    return sim.ProjectionModel.from_fit(fit)


@pytest.fixture(scope="module")
def discount():
    return 1.025 ** -np.arange(0, 100.0)


def deterministic(model):
    rw = lc.RandomWalkDrift(model.rw.drift, 0.0, 0.0, model.rw.k_first, model.rw.k_last,
                            model.rw.first_year, model.rw.last_year)
    return sim.ProjectionModel(model.age_min, model.ax, model.bx, rw)


def test_extension_to_closing_age(model):
    assert model.omega == 120
    assert model.ax.size == model.bx.size == 120 - 50 + 1
    assert np.all(np.diff(model.ax[50:]) > 0)  # Gompertz extrapolation increases with age


def test_deterministic_scenarios_equal_best_estimate(model, discount):
    det = deterministic(model)
    for age in (55, 65, 85, 119):
        out = sim.simulate_annuity(det, age, discount, n_scenarios=5)
        be = det.best_estimate(age, discount)
        np.testing.assert_allclose(out["pv_systematic"], be, rtol=1e-13)
        assert out["life_expectancy"][0] == pytest.approx(lt.curtate_life_expectancy(det.central_q(age)), rel=1e-13)


def test_cohort_survival_matches_python(model):
    k_path = model.rw.central_path(model.horizon(70))
    S = core.cohort_survival(model.age_min, model.ax, model.bx, 70, k_path)
    np.testing.assert_allclose(S, lt.survival_from_q(model.cohort_q(70, k_path)), rtol=1e-13, atol=1e-300)
    assert S[-1] == 0.0


def test_idiosyncratic_risk_has_exact_mean_and_variance(model, discount):
    det = deterministic(model)
    q = det.central_q(65)
    n_lives, n_scen = 200, 20_000
    out = sim.simulate_annuity(det, 65, discount, n_scenarios=n_scen, n_lives=n_lives, seed=4)
    mean, var = lt.annuity_due(q, discount), lt.annuity_due_variance(q, discount)
    sample = out["pv_portfolio"]
    se = np.sqrt(var / n_lives / n_scen)
    assert abs(sample.mean() - mean) < 4 * se
    # Variance of an average of n_lives independent lives; the sample variance of
    # 20,000 scenarios has a relative standard error of about 1%.
    assert sample.var(ddof=1) == pytest.approx(var / n_lives, rel=0.05)


def test_cpp_matches_numpy_reference_in_distribution(model, discount):
    cpp = sim.simulate_annuity(model, 65, discount, n_scenarios=40_000, seed=8)["pv_systematic"]
    ref = sim.simulate_annuity_numpy(model, 65, discount, n_scenarios=40_000, seed=9)["pv_systematic"]
    se = np.sqrt(cpp.var() / cpp.size + ref.var() / ref.size)
    assert abs(cpp.mean() - ref.mean()) < 4 * se
    assert cpp.std() == pytest.approx(ref.std(), rel=0.03)
    for level in (0.05, 0.5, 0.95):
        assert np.quantile(cpp, level) == pytest.approx(np.quantile(ref, level), rel=2e-3)


def test_parameter_uncertainty_widens_distribution(model, discount):
    with_pu = sim.simulate_annuity(model, 65, discount, 20_000, seed=1, parameter_uncertainty=True)["pv_systematic"]
    without = sim.simulate_annuity(model, 65, discount, 20_000, seed=1, parameter_uncertainty=False)["pv_systematic"]
    assert with_pu.std() > without.std()


def test_reproducible_across_thread_counts(model, discount):
    a = sim.simulate_annuity(model, 65, discount, 500, n_lives=50, seed=3, n_threads=1)
    b = sim.simulate_annuity(model, 65, discount, 500, n_lives=50, seed=3, n_threads=4)
    np.testing.assert_array_equal(a["pv_systematic"], b["pv_systematic"])
    np.testing.assert_array_equal(a["pv_portfolio"], b["pv_portfolio"])


def test_one_year_recalibration_without_risk_returns_best_estimate(model, discount):
    det = deterministic(model)
    out = sim.one_year_recalibration(det, 65, discount, n_scenarios=10)
    assert out["best_estimate"] == pytest.approx(det.best_estimate(65, discount), rel=1e-13)
    np.testing.assert_allclose(out["value"], out["best_estimate"], rtol=1e-12)
    np.testing.assert_allclose(out["drift_next"], det.rw.drift, rtol=1e-12)


def test_one_year_recalibration_matches_numpy(model, discount):
    cpp = sim.one_year_recalibration(model, 65, discount, 100_000, seed=5)
    ref = sim.one_year_recalibration_numpy(model, 65, discount, 100_000, seed=6)
    assert cpp["best_estimate"] == pytest.approx(ref["best_estimate"], rel=1e-13)
    for level in (0.005, 0.5, 0.995):
        assert np.quantile(cpp["value"], level) == pytest.approx(np.quantile(ref["value"], level), rel=1e-3)
