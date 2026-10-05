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
# Solvency II review: Commission Delegated Regulation (EU) 2026/269 of 29 October 2025 (C(2025) 7206), Article 1(8)
# and (9), applicable from 30 January 2027, following Directive (EU) 2025/2: RM = CoC sum_t max(0.96^t; 50%) SCR(t)
# / (1 + r(t+1))^(t+1) (amended Article 37(1)) with a cost-of-capital rate of 4.75% (amended Article 39).
REVIEW_COST_OF_CAPITAL = 0.0475
REVIEW_LAMBDA = 0.96
REVIEW_FLOOR = 0.50


def basic_scr(market, life, correlation=MARKET_LIFE_CORRELATION):
    return float(np.sqrt(market**2 + life**2 + 2.0 * correlation * market * life))


def operational_scr(bscr, best_estimate, factor=OPERATIONAL_LIFE_FACTOR):
    """Operational risk for life business without unit-linked: min(30% BSCR, 0.45% of the technical provisions
    without the risk margin, i.e. the best estimate, gross of reinsurance; Article 204(4))."""
    return float(min(0.3 * bscr, factor * best_estimate))


def best_estimate_runoff(cash_flows, discount):
    """BE(t) = sum_{u >= t} CF(u) v(u) / v(t) for payments at the start of each year."""
    cf = np.asarray(cash_flows, dtype=float)
    v = np.asarray(discount, dtype=float)[: cf.size]
    pv = np.cumsum((cf * v)[::-1])[::-1]
    return pv / v


def risk_margin(scr_0, be_runoff, discount, coc=COST_OF_CAPITAL, lambda_=1.0, floor=0.0):
    """Cost-of-capital risk margin with SCR(t) projected in proportion to BE(t)/BE(0).

    RM = CoC * sum_{t >= 0} max(lambda^t, floor) SCR(t) v(t + 1): with lambda = 1 the formula of Article 37
    of the Delegated Regulation in force until 29 January 2027 (the SCR projection by a proxy is a
    simplification allowed by the EIOPA guidelines on the valuation of technical provisions); with
    `coc=REVIEW_COST_OF_CAPITAL`, `lambda_=REVIEW_LAMBDA` and `floor=REVIEW_FLOOR` the amended formula that
    applies from 30 January 2027.

    `discount` must hold at least len(be_runoff) + 1 factors (v(0) .. v(N)); a shorter vector would drop
    terms of the sum silently, so it is rejected.
    """
    be = np.asarray(be_runoff, dtype=float)
    return risk_margin_from_path(scr_0 * be / be[0], discount, coc, lambda_, floor)


def risk_margin_from_path(scr_path, discount, coc=COST_OF_CAPITAL, lambda_=1.0, floor=0.0):
    """Cost-of-capital risk margin for an explicit projection SCR(t), t = 0..N-1:
    RM = CoC * sum_t max(lambda^t, floor) SCR(t) v(t + 1).

    `discount` must hold at least N + 1 factors (v(0) .. v(N)); a shorter vector would drop terms of the sum
    silently, so it is rejected. `risk_margin` is this formula with SCR(t) proportional to the best estimate.
    """
    scr = np.asarray(scr_path, dtype=float)
    v = np.asarray(discount, dtype=float)
    if v.size < scr.size + 1:
        raise ValueError("discount factors must cover the run-off plus one year")
    if not (0.0 < lambda_ <= 1.0 and 0.0 <= floor <= 1.0):
        raise ValueError("lambda_ must lie in (0, 1] and floor in [0, 1]")
    n = scr.size
    factor = np.maximum(lambda_ ** np.arange(n), floor)
    return float(coc * np.sum(factor * scr * v[1:n + 1]))
