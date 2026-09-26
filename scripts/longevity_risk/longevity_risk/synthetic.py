"""Synthetic stand-ins for the official data, used by the tests and by the
notebooks' offline mode (LONGEVITY_RISK_DATA_MODE=synthetic).

The values are generated from a known Lee-Carter model and are not
official statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Illustrative Svensson parameters (not an official observation).
ILLUSTRATIVE_SVENSSON = {"BETA0": 2.6, "BETA1": -0.6, "BETA2": -1.2, "BETA3": 1.5, "TAU1": 1.8, "TAU2": 12.0}


def svensson_parameters(date="2024-12-31") -> pd.DataFrame:
    frame = pd.DataFrame([ILLUSTRATIVE_SVENSSON], index=pd.to_datetime([date]))
    frame.attrs["source"] = "synthetic (illustrative parameters, not official data)"
    return frame


def lee_carter_parameters(ages):
    """Gompertz-like a_x and b_x decreasing with age (sum b_x = 1)."""
    ages = np.asarray(ages, dtype=float)
    ax = -10.3 + 0.095 * ages
    bx = (110.0 - ages) / np.sum(110.0 - ages)
    return ax, bx


def deaths_and_exposures(ages=range(50, 100), years=range(1980, 2024), drift=-1.2, sigma=1.0, seed=7,
                         population_scale=400_000.0, log_level=0.0):
    """Poisson deaths from a Lee-Carter model with a random-walk period index.

    Returns deaths and exposures (age x year DataFrames) and the true
    parameters (a_x, b_x, k_t) normalised with sum(k_t) = 0. `log_level`
    shifts a_x (for example -0.4 for a population with lower mortality).
    """
    rng = np.random.default_rng(seed)
    ages, years = np.asarray(list(ages)), np.asarray(list(years))
    ax, bx = lee_carter_parameters(ages)
    ax = ax + log_level
    kt = np.concatenate([[0.0], np.cumsum(drift + sigma * rng.standard_normal(years.size - 1))])
    ax = ax + bx * kt.mean()
    kt = kt - kt.mean()
    exposures = population_scale * np.exp(-0.07 * (ages - ages[0]))[:, None] * np.ones(years.size)[None, :]
    rates = np.exp(ax[:, None] + bx[:, None] * kt[None, :])
    deaths = rng.poisson(exposures * rates).astype(float)
    D = pd.DataFrame(deaths, index=pd.Index(ages, name="age"), columns=pd.Index(years, name="year"))
    E = pd.DataFrame(exposures, index=D.index, columns=D.columns)
    D.attrs["source"] = E.attrs["source"] = f"synthetic Lee-Carter population (seed {seed}), not official data"
    return D, E, {"ax": ax, "bx": bx, "kt": kt}


def pensioners(n=1500, share_male=0.6, seed=11):
    """Synthetic membership of a pension fund in payment (illustrative, not real data).

    Ages 60-95 with a mode around 70; annual pensions lognormal (median EUR 18,000
    for men and EUR 14,000 for women) with a heavy right tail.
    """
    rng = np.random.default_rng(seed)
    sex = np.where(rng.random(n) < share_male, "M", "F")
    age = np.clip(np.round(60 + rng.gamma(shape=3.0, scale=4.0, size=n)), 60, 95).astype(int)
    median = np.where(sex == "M", 18_000.0, 14_000.0)
    pension = np.round(median * np.exp(0.45 * rng.standard_normal(n)), -1)
    frame = pd.DataFrame({"member": np.arange(1, n + 1), "sex": sex, "age": age, "pension": pension})
    frame.attrs["source"] = f"synthetic pension fund membership (seed {seed}), not real data"
    return frame
