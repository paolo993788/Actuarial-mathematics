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
