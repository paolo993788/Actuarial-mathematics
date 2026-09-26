"""Solvency II standard formula: longevity risk sub-module.

Commission Delegated Regulation (EU) 2015/35, Article 138: the capital
requirement for longevity risk is the loss in basic own funds resulting from
an instantaneous permanent decrease of 20% in the mortality rates used to
calculate the best estimate. For a portfolio of annuities in payment (no
lapse or expense effects) this is the increase of the best estimate:

    SCR_longevity = BE(0.8 q) - BE(q).
"""

from __future__ import annotations

import numpy as np

from .life_table import annuity_due

LONGEVITY_SHOCK = 0.20


def longevity_scr(q, discount, shock=LONGEVITY_SHOCK):
    """Best estimate, shocked best estimate and SCR for one life annuity-due of 1.

    `q` is the vector of one-year death probabilities along the policy's
    future (the last value 1 closes the table and is not shocked).
    """
    q = np.asarray(q, dtype=float)
    shocked = q * (1.0 - shock)
    shocked[q >= 1.0] = 1.0
    be = annuity_due(q, discount)
    be_shocked = annuity_due(shocked, discount)
    return {"best_estimate": be, "shocked_best_estimate": be_shocked, "scr": be_shocked - be,
            "scr_ratio": be_shocked / be - 1.0}


# --------------------------------------------------------------------------- interest-rate risk

# Delegated Regulation (EU) 2015/35, Articles 166 and 167: relative shocks to the
# risk-free rates by maturity (years). Between 20 and 90 years the shocks are
# interpolated linearly; from 90 years on they are 20%.
IR_UP = {1: 0.70, 2: 0.70, 3: 0.64, 4: 0.59, 5: 0.55, 6: 0.52, 7: 0.49, 8: 0.47, 9: 0.44, 10: 0.42,
         11: 0.39, 12: 0.37, 13: 0.35, 14: 0.34, 15: 0.33, 16: 0.31, 17: 0.30, 18: 0.29, 19: 0.27, 20: 0.26, 90: 0.20}
IR_DOWN = {1: 0.75, 2: 0.65, 3: 0.56, 4: 0.50, 5: 0.46, 6: 0.42, 7: 0.39, 8: 0.36, 9: 0.33, 10: 0.31,
           11: 0.30, 12: 0.29, 13: 0.28, 14: 0.28, 15: 0.27, 16: 0.28, 17: 0.28, 18: 0.28, 19: 0.29, 20: 0.29, 90: 0.20}


def relative_shock(maturity, table):
    """Relative shock for a maturity in years (the 1-year value applies below one year)."""
    keys = np.array(sorted(table), dtype=float)
    values = np.array([table[k] for k in sorted(table)])
    return np.interp(np.clip(np.asarray(maturity, dtype=float), 1.0, 90.0), keys, values)


def shocked_zero_rates(maturities, zero_rates, direction):
    """Standard-formula shocked zero rates (annual compounding).

    Up: r (1 + s_up), with an absolute increase of at least one percentage point.
    Down: r (1 - s_down) for positive rates; negative rates are not shocked.
    """
    m = np.asarray(maturities, dtype=float)
    r = np.asarray(zero_rates, dtype=float)
    if direction == "up":
        return r + np.where(r > 0, np.maximum(relative_shock(m, IR_UP) * r, 0.01), 0.01)
    if direction == "down":
        return np.where(r > 0, r * (1.0 - relative_shock(m, IR_DOWN)), r)
    raise ValueError("direction must be 'up' or 'down'")


def discount_factors(zero_rates_annual, times):
    t = np.asarray(times, dtype=float)
    return (1.0 + np.asarray(zero_rates_annual, dtype=float)) ** (-t)


def interest_rate_scr(cash_flows, times, zero_rates_annual):
    """Liability-side interest-rate SCR: largest increase of the present value under the two shocks."""
    cf = np.asarray(cash_flows, dtype=float)
    t = np.asarray(times, dtype=float)
    base = float(cf @ discount_factors(zero_rates_annual, t))
    up = float(cf @ discount_factors(shocked_zero_rates(t, zero_rates_annual, "up"), t))
    down = float(cf @ discount_factors(shocked_zero_rates(t, zero_rates_annual, "down"), t))
    return {"base": base, "up": up, "down": down, "scr": max(up - base, down - base, 0.0)}


# --------------------------------------------------------------------------- aggregation and risk margin

MARKET_LIFE_CORRELATION = 0.25   # Directive 2009/138/EC, Annex IV
OPERATIONAL_LIFE_FACTOR = 0.0045  # Delegated Regulation (EU) 2015/35, Article 204 (life obligations)
COST_OF_CAPITAL = 0.06            # Delegated Regulation (EU) 2015/35, Article 39


def basic_scr(market, life, correlation=MARKET_LIFE_CORRELATION):
    return float(np.sqrt(market**2 + life**2 + 2.0 * correlation * market * life))


def operational_scr(bscr, technical_provisions, factor=OPERATIONAL_LIFE_FACTOR):
    """Operational risk for life business without unit-linked: min(30% BSCR, 0.45% of technical provisions)."""
    return float(min(0.3 * bscr, factor * technical_provisions))


def best_estimate_runoff(cash_flows, discount):
    """BE(t) = sum_{u >= t} CF(u) v(u) / v(t) for payments at the start of each year."""
    cf = np.asarray(cash_flows, dtype=float)
    v = np.asarray(discount, dtype=float)[: cf.size]
    pv = np.cumsum((cf * v)[::-1])[::-1]
    return pv / v


def risk_margin(scr_0, be_runoff, discount, coc=COST_OF_CAPITAL):
    """Cost-of-capital risk margin with SCR(t) projected in proportion to BE(t)/BE(0).

    RM = CoC * sum_{t >= 0} SCR(t) v(t + 1), a simplification allowed by the EIOPA
    guidelines on the valuation of technical provisions (projection by a proxy).
    """
    be = np.asarray(be_runoff, dtype=float)
    v = np.asarray(discount, dtype=float)
    scr_path = scr_0 * be / be[0]
    n = min(be.size, v.size - 1)
    return float(coc * np.sum(scr_path[:n] * v[1:n + 1]))
