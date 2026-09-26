"""Life-table functions: probabilities, life expectancies and annuity values.

Units and conventions: ages and durations in years; m is the central death
rate (deaths per person-year of exposure); q is the one-year death
probability. Within each year of age the force of mortality is assumed
constant, so q = 1 - exp(-m) and the person-years lived by a life aged x
during the year are (1 - exp(-m)) / m.
"""

from __future__ import annotations

import numpy as np


def q_from_m(m):
    return 1.0 - np.exp(-np.asarray(m, dtype=float))


def survival_from_q(q):
    """S(0..H) = 1, (1 - q_0), (1 - q_0)(1 - q_1), ... for a sequence of one-year death probabilities."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([[1.0], np.cumprod(1.0 - q)])


def period_life_expectancy(m):
    """Complete period life expectancy at the first age of the rates m.

    `m` holds consecutive ages along its last axis (a 2-D array gives one
    value per row). The last age is treated as an open age group closed with
    the constant force m (expected remaining lifetime 1 / m).
    """
    m = np.asarray(m, dtype=float)
    q = q_from_m(m[..., :-1])
    ones = np.ones(m.shape[:-1] + (1,))
    l = np.concatenate([ones, np.cumprod(1.0 - q, axis=-1)], axis=-1)  # survivors at each exact age
    person_years = l[..., :-1] * q / m[..., :-1]                      # constant force within the year
    e = person_years.sum(axis=-1) + l[..., -1] / m[..., -1]
    return float(e) if e.ndim == 0 else e


def curtate_life_expectancy(q):
    """Curtate expectation of life sum_{t>=1} S(t) along a cohort path of q (closed by q = 1)."""
    return float(survival_from_q(q)[1:].sum())


def annuity_due(q, discount):
    """Present value of a life annuity-due of 1 per year: sum_{t=0}^{H-1} S(t) v(t).

    q[h] is the death probability during year h + 1 of the policy and
    discount[t] the discount factor for time t (discount[0] = 1 normally).
    """
    q = np.asarray(q, dtype=float)
    S = survival_from_q(q)[:-1]
    v = np.asarray(discount, dtype=float)[: S.size]
    if v.size < S.size:
        raise ValueError("discount factors must cover the whole horizon")
    return float(np.sum(S * v))


def annuity_due_variance(q, discount):
    """Variance of the present value of a single life annuity-due.

    The curtate lifetime K has P(K = k) = S(k) q_k and the present value is
    sum_{t=0}^{K} v(t).
    """
    q = np.asarray(q, dtype=float)
    S = survival_from_q(q)
    prob = S[:-1] * q                        # P(K = k), k = 0..H-1
    pv = np.cumsum(np.asarray(discount, dtype=float)[: q.size])
    mean = np.sum(prob * pv)
    return float(np.sum(prob * pv**2) - mean**2)
