"""Multi-population mortality and index-based longevity hedging.

Li-Lee augmented common factor model (Li and Lee, 2005), estimated by Poisson maximum likelihood in two
stages (as in Li, 2013 and Enchev, Kleinow and Cairns, 2017):

1. a Lee-Carter model for the aggregate of a group of populations, ln m^G(x, t) = A_x + B_x K_t;
2. for each population i, with B_x K_t as an offset,
   ln m_i(x, t) = a_{x,i} + B_x K_t + b_{x,i} k_{t,i},  sum_x b_{x,i} = 1, sum_t k_{t,i} = 0.

Dynamics: K_t is a random walk with drift and each k_{t,i} a zero-mean AR(1), so population-specific
deviations revert and long-term projections are coherent (no indefinite divergence between countries).
The innovations of K and of the k_i are jointly Gaussian with a covariance estimated from consecutive
fitted years.

Hedging: a pension liability on one population is hedged with q-forwards on the mortality of an index
population. Payoffs and liability values are simulated jointly; hedge notionals are estimated by least
squares on one set of scenarios and evaluated on an independent set (Coughlan et al., 2011; Li and Hardy,
2011). Conventions follow ``simulation``: valuation at the end of the last calibration year T; a life aged
x0 at valuation experiences in year T + h the rate of age x0 + h - 1; q = 1 - exp(-m).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, synthetic
from . import lee_carter as lc
from .lee_carter import RandomWalkDrift
from .life_table import period_life_expectancy, q_from_m

PEERS = ("FR", "DE", "ES", "NL", "BE", "AT", "SE", "DK", "PT", "CH", "FI", "NO")


# --------------------------------------------------------------------------- data


def load_panel(countries, sex="M", ages=range(50, 90), first_year=1998, last_year=None, official=True, seed=31):
    """Deaths and exposures (age x year) for several countries on their common years.

    Offline (`official=False`) each country is simulated from a Lee-Carter model sharing a common period
    index plus a mean-reverting country deviation (illustrative, not official statistics).
    """
    D, E = {}, {}
    if official:
        for geo in countries:
            D[geo], E[geo] = data.load_deaths_exposures(geo, sex, ages)
    else:
        D, E = synthetic_panel(countries, ages, range(first_year, (last_year or 2024) + 1), seed=seed)
    common = sorted(set.intersection(*(set(d.columns) for d in D.values())))
    common = [y for y in common if y >= first_year and (last_year is None or y <= last_year)]
    for geo in D:
        D[geo], E[geo] = D[geo].loc[:, common], E[geo].loc[:, common]
    return D, E


def synthetic_panel(countries, ages, years, seed=31, common_drift=-1.2, common_sigma=0.8, phi=0.8, sigma_i=1.0):
    rng = np.random.default_rng(seed)
    ages, years = np.asarray(list(ages)), np.asarray(list(years))
    ax, bx = synthetic.lee_carter_parameters(ages)
    K = np.concatenate([[0.0], np.cumsum(common_drift + common_sigma * rng.standard_normal(years.size - 1))])
    K -= K.mean()
    D, E = {}, {}
    for j, geo in enumerate(countries):
        level = 0.25 * rng.standard_normal()
        b_i = bx * (1 + 0.3 * rng.standard_normal(ages.size) * 0.1)
        b_i /= b_i.sum()
        k = np.zeros(years.size)
        for t in range(1, years.size):
            k[t] = phi * k[t - 1] + sigma_i * rng.standard_normal()
        scale = 50_000 * (1 + 3 * rng.random())
        exposures = scale * np.exp(-0.07 * (ages - ages[0]))[:, None] * np.ones(years.size)[None, :]
        rates = np.exp(ax[:, None] + level + bx[:, None] * K[None, :] + b_i[:, None] * k[None, :])
        D[geo] = pd.DataFrame(rng.poisson(exposures * rates).astype(float), index=pd.Index(ages, name="age"),
                              columns=pd.Index(years, name="year"))
        E[geo] = pd.DataFrame(exposures, index=D[geo].index, columns=D[geo].columns)
        D[geo].attrs["source"] = E[geo].attrs["source"] = f"synthetic multi-population data (seed {seed}), not official"
    return D, E


def aggregate(D: dict, E: dict, members) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum deaths and exposures over the member populations."""
    members = list(members)
    return sum(D[m] for m in members), sum(E[m] for m in members)


# --------------------------------------------------------------------------- estimation


@dataclass
class LiLeeFit:
    ages: np.ndarray
    years: np.ndarray
    mask: np.ndarray                 # fitted years (False for years given zero weight)
    A: np.ndarray
    B: np.ndarray
    K: np.ndarray                    # NaN in excluded years
    a: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    k: dict = field(default_factory=dict)
    members: tuple = ()
    deviance: pd.DataFrame | None = None

    @property
    def names(self) -> list[str]:
        return list(self.a)

    def log_rates(self, name) -> pd.DataFrame:
        """Fitted ln m(x, t) of a population (ages x years)."""
        values = (self.a[name][:, None] + self.B[:, None] * self.K[None, :]
                  + self.b[name][:, None] * self.k[name][None, :])
        return pd.DataFrame(values, index=self.ages, columns=self.years)


def _common_only_deviance(D, E, offset, mask):
    """Deviance of ln m = a_x + offset with a_x fitted in closed form (Poisson score equation); a zero offset
    gives the static age-only model."""
    Dm, Em, off = D[:, mask], E[:, mask], offset[:, mask]
    a = np.log(Dm.sum(axis=1) / (Em * np.exp(off)).sum(axis=1))
    Dhat = Em * np.exp(a[:, None] + off)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(Dm > 0, Dm * np.log(Dm / Dhat), 0.0)
    return float(np.sum(2 * (term - (Dm - Dhat))))


def fit_li_lee(D: dict, E: dict, members, fit_years=None) -> LiLeeFit:
    """Two-stage Poisson Li-Lee fit; the common factor is estimated on the aggregate of `members`.

    Every population in D (members or not, including aggregates added by the caller) gets its own
    a_{x,i}, b_{x,i} and k_{t,i}. The deviance table compares, for each population, a static age-only model,
    the common factor alone, the Li-Lee model and an independent Lee-Carter fit; the explanation ratios are
    the Poisson-deviance analogue of the R ratios of Li and Lee (2005): the share of the deviance of the
    static model removed by the common factor alone and by the full Li-Lee model.
    """
    Dg, Eg = aggregate(D, E, members)
    common = lc.fit_poisson(Dg, Eg, fit_years)
    mask = ~np.isnan(common.kt)
    offset = common.bx[:, None] * np.where(mask, common.kt, 0.0)[None, :]
    fit = LiLeeFit(common.ages, common.years, mask, common.ax, common.bx, common.kt, members=tuple(members))
    rows = {}
    for name in D:
        own = lc.fit_poisson(D[name], E[name], fit_years, offset=offset)
        fit.a[name], fit.b[name], fit.k[name] = own.ax, own.bx, own.kt
        independent = lc.fit_poisson(D[name], E[name], fit_years)
        Dn, En = D[name].to_numpy(float), E[name].to_numpy(float)
        rows[name] = {"static (age only)": _common_only_deviance(Dn, En, np.zeros_like(offset), mask),
                      "common factor only": _common_only_deviance(Dn, En, offset, mask),
                      "Li-Lee": own.deviance, "independent Lee-Carter": independent.deviance}
    dev = pd.DataFrame(rows).T
    dev["R common"] = 1 - dev["common factor only"] / dev["static (age only)"]
    dev["R Li-Lee"] = 1 - dev["Li-Lee"] / dev["static (age only)"]
    fit.deviance = dev
    return fit


@dataclass
class LiLeeDynamics:
    drift: float
    drift_se: float
    phi: dict                        # AR(1) coefficients used (capped at phi_max)
    cov: pd.DataFrame                # innovations of K and of each k_i (index "K", names...)
    K_last: float
    k_last: dict
    last_year: int
    n_obs: int
    phi_unrestricted: dict = field(default_factory=dict)

    @property
    def corr(self) -> pd.DataFrame:
        s = np.sqrt(np.diag(self.cov.to_numpy()))
        return self.cov / np.outer(s, s)


def fit_dynamics(fit: LiLeeFit, names=None, phi_max: float = 0.98, random_walk_deviations: bool = False) -> LiLeeDynamics:
    """Random walk with drift for K, zero-mean AR(1) for each k_i, joint innovation covariance.

    Innovations use pairs of consecutive fitted years only (a gap left by excluded years is skipped).
    The least-squares AR(1) coefficient is capped at `phi_max`: an estimate of one or more means the
    population kept diverging from the group in the sample, which a coherent projection does not
    extrapolate indefinitely. The innovations are the residuals at the capped coefficient. With
    `random_walk_deviations` every k_i is a driftless random walk (phi = 1): the non-coherent alternative in
    which country deviations are permanent.
    """
    names = list(names or fit.names)
    years = fit.years.astype(int)
    rw = RandomWalkDrift.from_kt(years, fit.K)
    ok = fit.mask
    pairs = [t for t in range(1, years.size) if ok[t] and ok[t - 1] and years[t] - years[t - 1] == 1]
    innov = {"K": np.array([fit.K[t] - fit.K[t - 1] - rw.drift for t in pairs])}
    phi, phi_hat = {}, {}
    for name in names:
        k = fit.k[name]
        prev, cur = np.array([k[t - 1] for t in pairs]), np.array([k[t] for t in pairs])
        phi_hat[name] = float(prev @ cur / (prev @ prev))
        phi[name] = 1.0 if random_walk_deviations else min(phi_hat[name], phi_max)
        innov[name] = cur - phi[name] * prev
    X = pd.DataFrame(innov)
    cov = X.T @ X / (len(pairs) - 1)
    last = int(np.flatnonzero(ok)[-1])
    return LiLeeDynamics(rw.drift, rw.drift_se, phi, cov, float(fit.K[last]), {n: float(fit.k[n][last]) for n in names},
                         int(years[last]), len(pairs), phi_hat)


# --------------------------------------------------------------------------- projection and simulation


def extended_parameters(fit: LiLeeFit, name, omega=110, slope_ages=15, b_ages=10):
    """a_{x,i} extrapolated linearly in age (Gompertz) and B_x, b_{x,i} held at the average of the oldest
    fitted ages, up to `omega` (same rule as ``simulation.extend_to_closing_age``)."""
    ages = fit.ages.astype(int)
    extra = np.arange(ages.max() + 1, omega + 1)
    slope, intercept = np.polyfit(ages[-slope_ages:], fit.a[name][-slope_ages:], 1)
    a = np.concatenate([fit.a[name], intercept + slope * extra])
    B = np.concatenate([fit.B, np.full(extra.size, fit.B[-b_ages:].mean())])
    b = np.concatenate([fit.b[name], np.full(extra.size, fit.b[name][-b_ages:].mean())])
    return np.arange(ages.min(), omega + 1), a, B, b


def central_states(dyn: LiLeeDynamics, name, horizon: int, K0=None, k0=None):
    """Median paths K_{T+h} = K0 + h d and k_{T+h} = phi^h k0, h = 1..horizon (vectorised over K0, k0)."""
    h = np.arange(1, horizon + 1)
    K0 = dyn.K_last if K0 is None else np.asarray(K0, dtype=float)[..., None]
    k0 = dyn.k_last[name] if k0 is None else np.asarray(k0, dtype=float)[..., None]
    return K0 + dyn.drift * h, k0 * dyn.phi[name] ** h


def simulate_states(dyn: LiLeeDynamics, names, horizon: int, n_scenarios: int, seed: int,
                    parameter_uncertainty=True) -> dict:
    """Joint paths of K and of the k_i for h = 1..horizon: arrays of shape (n_scenarios, horizon)."""
    rng = np.random.default_rng(seed)
    keys = ["K"] + list(names)
    cov = dyn.cov.loc[keys, keys].to_numpy()
    values, vectors = np.linalg.eigh(cov)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))      # robust square root of a near-singular matrix
    eps = rng.standard_normal((n_scenarios, horizon, len(keys))) @ root.T
    drift = dyn.drift + dyn.drift_se * rng.standard_normal(n_scenarios) * float(parameter_uncertainty)
    out = {"K": dyn.K_last + np.cumsum(drift[:, None] + eps[:, :, 0], axis=1)}
    for j, name in enumerate(names, start=1):
        k = np.empty((n_scenarios, horizon))
        prev = np.full(n_scenarios, dyn.k_last[name])
        for h in range(horizon):
            prev = dyn.phi[name] * prev + eps[:, h, j]
            k[:, h] = prev
        out[name] = k
    return out


def period_e(fit: LiLeeFit, name, K, k, age=65, omega=110) -> np.ndarray:
    """Period life expectancy at `age` for period indices (K, k) of any matching shape."""
    ages, a, B, b = extended_parameters(fit, name, omega)
    i = int(np.searchsorted(ages, age))
    K, k = np.asarray(K, dtype=float), np.asarray(k, dtype=float)
    m = np.exp(a[i:] + B[i:] * K[..., None] + b[i:] * k[..., None])
    return period_life_expectancy(m)


def temporary_life_expectancy(m) -> np.ndarray:
    """Expected years lived between the first age of `m` and the end of its last age (for example e(65:25)
    from rates at ages 65-89), constant force within each year of age; `m` has ages on its last axis."""
    m = np.asarray(m, dtype=float)
    q = -np.expm1(-m)                        # accurate for small rates
    ones = np.ones(m.shape[:-1] + (1,))
    l = np.concatenate([ones, np.cumprod(1.0 - q, axis=-1)[..., :-1]], axis=-1)
    return (l * q / m).sum(axis=-1)


def independent_forecast_e(D: pd.DataFrame, E: pd.DataFrame, fit_years, horizon: int, age=65, omega=110) -> np.ndarray:
    """Central period life expectancy of an independent Poisson Lee-Carter model with random-walk drift."""
    from .simulation import extend_to_closing_age
    f = lc.fit_poisson(D, E, fit_years)
    rw = RandomWalkDrift.from_kt(f.years, f.kt)
    ages, ax, bx = extend_to_closing_age(f, omega)
    i = int(np.searchsorted(ages, age))
    k = rw.central_path(horizon)
    return period_life_expectancy(np.exp(ax[i:] + bx[i:] * k[:, None]))


# --------------------------------------------------------------------------- hedging


def liability_values(fit: LiLeeFit, dyn: LiLeeDynamics, name, states: dict, age0: int, horizon: int, discount,
                     n_lives: int = 0, seed: int = 7, omega: int = 110) -> np.ndarray:
    """Present value per initial life of an annuity-due of 1 a year to a cohort aged `age0`, split at `horizon`:
    payments at t = 0..horizon-1 on the simulated survivors plus the discounted best-estimate annuity at the
    horizon, re-projected from the simulated state (K, k) at that date (a market-consistent value hedge).

    With `n_lives` > 0 deaths are binomial (sampling risk of a finite portfolio); otherwise the portfolio is
    infinitely large. `discount[t]` is the discount factor for time t.
    """
    ages, a, B, b = extended_parameters(fit, name, omega)
    v = np.asarray(discount, dtype=float)
    K, k = states["K"][:, :horizon], states[name][:, :horizon]
    n = K.shape[0]
    idx = age0 - ages[0] + np.arange(horizon)
    q = q_from_m(np.exp(a[idx] + B[idx] * K + b[idx] * k))            # (n, horizon)
    if n_lives:
        rng = np.random.default_rng(seed)
        alive = np.full(n, n_lives)
        surv = np.empty((n, horizon + 1))
        surv[:, 0] = 1.0
        for h in range(horizon):
            alive = alive - rng.binomial(alive, q[:, h])
            surv[:, h + 1] = alive / n_lives
    else:
        surv = np.concatenate([np.ones((n, 1)), np.cumprod(1 - q, axis=1)], axis=1)
    paid = surv[:, :horizon] @ v[:horizon]
    # best-estimate annuity at the horizon for the survivors, now aged age0 + horizon
    age_h = age0 + horizon
    H = omega - age_h + 1
    Kc, kc = central_states(dyn, name, H, K[:, -1], k[:, -1])
    j = age_h - ages[0] + np.arange(H)
    qf = q_from_m(np.exp(a[j] + B[j] * Kc + b[j] * kc))
    qf[:, -1] = 1.0
    sf = np.concatenate([np.ones((n, 1)), np.cumprod(1 - qf, axis=1)[:, :-1]], axis=1)
    future = sf @ (v[horizon:horizon + H] / v[horizon])
    return paid + v[horizon] * surv[:, horizon] * future


def index_death_rates(fit: LiLeeFit, name, states: dict, ages, year_index: int) -> np.ndarray:
    """Realised one-year death probabilities of an index population at `ages` in projection year `year_index`
    (1-based), shape (n_scenarios, len(ages))."""
    i = np.searchsorted(fit.ages, np.asarray(ages))
    K, k = states["K"][:, year_index - 1], states[name][:, year_index - 1]
    return q_from_m(np.exp(fit.a[name][i] + fit.B[i] * K[:, None] + fit.b[name][i] * k[:, None]))


def hedge_effectiveness(L_cal, X_cal, L_test, X_test, level=0.995) -> dict:
    """Least-squares hedge of L with instruments X (payoffs linear in X), fitted on calibration scenarios and
    evaluated on independent test scenarios.

    Returns notionals (units of the instruments to sell), and the reductions in variance, standard deviation
    and in the `level` value at risk (quantile minus mean) of the hedged position on the test scenarios.
    """
    X_cal, X_test = np.atleast_2d(np.asarray(X_cal).T).T, np.atleast_2d(np.asarray(X_test).T).T
    Z = np.column_stack([np.ones(len(L_cal)), X_cal])
    beta = np.linalg.lstsq(Z, L_cal, rcond=None)[0][1:]
    hedged = L_test - (X_test - X_cal.mean(axis=0)) @ beta
    var_u, var_h = np.var(L_test), np.var(hedged)
    var_at_risk = lambda x: np.quantile(x, level) - np.mean(x)   # noqa: E731 - liabilities: high values are losses
    return {"notionals": beta, "variance reduction": 1 - var_h / var_u,
            "sd reduction": 1 - np.sqrt(var_h / var_u),
            "VaR reduction": 1 - var_at_risk(hedged) / var_at_risk(L_test),
            "unhedged sd": float(np.sqrt(var_u)), "hedged sd": float(np.sqrt(var_h)),
            "unhedged VaR": float(var_at_risk(L_test)), "hedged VaR": float(var_at_risk(hedged))}
