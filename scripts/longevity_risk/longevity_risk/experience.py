"""Experience rating of a pension fund's mortality basis.

A pension fund's mortality differs from that of the national population (socio-economic selection). The
basis is a multiplier theta on the population's central death rates, m_fund(x, t) = theta m_pop(x, t), and
the fund's deaths are Poisson with mean theta times the deaths expected on population rates
(Brouhns, Denuit and Vermunt, 2002, for the Poisson assumption; Plat, 2009, and Villegas and Haberman, 2014,
for portfolio-specific mortality relative to a reference population).

* ``actual_to_expected``: the maximum-likelihood estimate of theta, D / E, with standard error sqrt(D) / E,
  i.e. a relative standard error of 1 / sqrt(D).
* ``credibility``: the fund's estimate combined with a prior on theta. With a Gamma prior (mean mu, standard
  deviation tau) the posterior mean is the Buhlmann-Straub credibility estimate
  Z D / E + (1 - Z) mu with Z = E / (E + mu / tau^2), exactly (Gamma-Poisson conjugacy).
* ``multiplier_for_life_expectancy``: the multiplier of a set of rates that produces a given period life
  expectancy, used to translate the socio-economic gradient of life expectancy (Eurostat demo_mlexpecedu,
  by educational attainment) into a range of plausible multipliers.

Units: deaths and expected deaths in persons; rates per person-year; life expectancies in years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from . import data
from .lee_carter import LeeCarterFit
from .life_table import period_life_expectancy


def actual_to_expected(deaths, expected) -> dict:
    """Maximum-likelihood multiplier of the expected deaths and its standard error (Poisson deaths)."""
    D = float(np.sum(deaths))
    E = float(np.sum(expected))
    if not (E > 0 and D >= 0):
        raise ValueError("expected deaths must be positive and deaths non-negative")
    ratio = D / E
    return {"deaths": D, "expected": E, "ratio": ratio, "se": np.sqrt(D) / E,
            "relative_se": 1.0 / np.sqrt(D) if D > 0 else np.inf}


def gamma_prior(mean: float, sd: float) -> tuple[float, float]:
    """Shape and rate of the Gamma distribution with the given mean and standard deviation."""
    if not (mean > 0 and sd > 0):
        raise ValueError("the prior mean and standard deviation must be positive")
    return mean**2 / sd**2, mean / sd**2


def credibility(deaths, expected, prior_mean: float, prior_sd: float) -> dict:
    """Credibility estimate of the multiplier from the fund's deaths and a Gamma prior.

    Returns the credibility weight Z, the estimate (posterior mean) and the posterior standard deviation.
    """
    a, b = gamma_prior(prior_mean, prior_sd)
    D = float(np.sum(deaths))
    E = float(np.sum(expected))
    if not (E >= 0 and D >= 0):
        raise ValueError("deaths and expected deaths must be non-negative")
    Z = E / (E + b)
    return {"Z": Z, "estimate": (a + D) / (b + E), "posterior_sd": np.sqrt(a + D) / (b + E),
            "ratio": D / E if E > 0 else np.nan}


def multiplier_for_life_expectancy(m, target: float) -> float:
    """theta such that the period life expectancy of theta * m equals `target` (see life_table)."""
    m = np.asarray(m, dtype=float)

    def gap(log_theta):
        return period_life_expectancy(np.exp(log_theta) * m) - target

    return float(np.exp(optimize.brentq(gap, np.log(0.05), np.log(20.0), xtol=1e-14)))


def load_education_life_expectancy(geo="IT", sex="M", age=65, refresh=False) -> pd.DataFrame:
    """Life expectancy at `age` by educational attainment (Eurostat demo_mlexpecedu): years x ISCED groups."""
    frame = data.load_eurostat("demo_mlexpecedu", refresh=refresh, geo=geo, sex=sex, age=f"Y{age}")
    table = frame.pivot_table(index="time", columns="isced11", values="value").rename(index=int).sort_index()
    table.index.name = "year"
    table.attrs["source"] = f"Eurostat demo_mlexpecedu (life expectancy by educational attainment), geo={geo}, sex={sex}"
    return table


def education_multipliers(fit: LeeCarterFit, life_expectancy: pd.DataFrame, age=65) -> pd.DataFrame:
    """Mortality multipliers of each education group relative to the whole population, by year.

    For each year, theta_g solves e(theta_g m_t) = e_g on the fitted period rates m_t at ages `age`..max (the
    last age open) and the result is divided by theta for the 'TOTAL' column, so that differences between the
    fitted model and Eurostat's life table cancel to first order.
    """
    i = int(np.searchsorted(fit.ages, age))
    rows = {}
    for year, row in life_expectancy.iterrows():
        j = np.flatnonzero(fit.years == year)
        if j.size == 0 or np.isnan(fit.kt[j[0]]) or row.isna().any():
            continue
        m = np.exp(fit.ax[i:] + fit.bx[i:] * fit.kt[j[0]])
        theta = {g: multiplier_for_life_expectancy(m, e) for g, e in row.items()}
        rows[year] = {g: theta[g] / theta["TOTAL"] for g in row.index}
    out = pd.DataFrame(rows).T
    out.index.name = "year"
    return out


def simulate_experience(fit: LeeCarterFit, ages, years, multiplier: float, seed: int) -> pd.DataFrame:
    """Deaths of a fund with the age profile `ages` (one member per entry, a stationary membership) in each
    calendar year of `years`, simulated as Poisson with mean multiplier x the deaths expected on the fitted
    population rates of that year (one year of exposure per member). Illustrative: not observed data."""
    rng = np.random.default_rng(seed)
    counts = pd.Series(np.asarray(ages, dtype=int)).value_counts().sort_index()
    i = np.searchsorted(fit.ages, counts.index.to_numpy())
    if not np.isin(counts.index.to_numpy(), fit.ages).all():
        raise ValueError("member ages must be fitted ages")
    rows = []
    for year in years:
        j = np.flatnonzero(fit.years == year)
        if j.size == 0 or np.isnan(fit.kt[j[0]]):
            raise ValueError(f"{year} is not a fitted year")
        expected = float(counts.to_numpy() @ np.exp(fit.ax[i] + fit.bx[i] * fit.kt[j[0]]))
        rows.append({"year": int(year), "expected": expected, "deaths": int(rng.poisson(multiplier * expected))})
    return pd.DataFrame(rows).set_index("year")
