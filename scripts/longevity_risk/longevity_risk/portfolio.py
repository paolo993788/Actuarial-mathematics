"""Portfolios of pensioners: cash flows, best estimates and Monte Carlo risk.

Members are described by sex, exact age at valuation and annual pension
(paid yearly in advance while alive, optionally indexed at a constant rate).
Each sex has its own ProjectionModel. For stochastic results the two sexes
use the same random scenarios, i.e. their mortality trends are treated as
perfectly correlated, a prudent simplification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import require_cpp
from .simulation import ProjectionModel


def indexed_discount(discount, indexation=0.0):
    """Discount factors for payments growing at a constant indexation rate."""
    v = np.asarray(discount, dtype=float)
    return v * (1.0 + indexation) ** np.arange(v.size)


def expected_cash_flows(models: dict, members: pd.DataFrame, horizon: int, indexation=0.0, q_multiplier=1.0) -> pd.DataFrame:
    """Expected pension payments at t = 0..horizon-1 by sex, under the central projections.

    `q_multiplier` scales every death probability except the closing age (0.8 gives the
    Solvency II longevity shock). With pension = 1 the result is the expected number of
    survivors, used for per-member expenses.
    """
    out = pd.DataFrame(0.0, index=pd.RangeIndex(horizon, name="year"), columns=sorted(members["sex"].unique()))
    growth = (1.0 + indexation) ** np.arange(horizon)
    for (sex, age), group in members.groupby(["sex", "age"]):
        model = models[sex]
        q = model.central_q(int(age))
        q = np.where(q >= 1.0, 1.0, q * q_multiplier)
        S = np.concatenate([[1.0], np.cumprod(1.0 - q)])[: model.horizon(int(age))]
        n = min(S.size, horizon)
        out.loc[: n - 1, sex] += group["pension"].sum() * S[:n] * growth[:n]
    out["total"] = out.sum(axis=1)
    return out


def simulate(models: dict, members: pd.DataFrame, discount, n_scenarios=20_000, idiosyncratic=True, seed=12345,
             indexation=0.0, parameter_uncertainty=True, n_threads=0) -> pd.DataFrame:
    """Present value of the whole portfolio in each scenario (C++ engine), by sex and in total."""
    core = require_cpp()
    v = indexed_discount(discount, indexation)
    result = {}
    for sex, group in members.groupby("sex"):
        m: ProjectionModel = models[sex]
        out = core.simulate_portfolio(*m._args(parameter_uncertainty), group["age"].to_numpy(np.int32),
                                      group["pension"].to_numpy(float), v, n_scenarios, idiosyncratic, seed, n_threads)
        result[f"{sex} systematic"] = out["pv_systematic"]
        if idiosyncratic:
            result[f"{sex} realised"] = out["pv_realised"]
    frame = pd.DataFrame(result)
    frame["systematic"] = frame.filter(like="systematic").sum(axis=1)
    if idiosyncratic:
        frame["realised"] = frame.filter(like="realised").sum(axis=1)
    return frame


def one_year_values(models: dict, members: pd.DataFrame, discount, n_scenarios=100_000, seed=12345, indexation=0.0,
                    parameter_uncertainty=True, n_threads=0):
    """Portfolio value after one simulated year with re-estimated trend (C++), and the best estimate."""
    core = require_cpp()
    v = indexed_discount(discount, indexation)
    values, best = np.zeros(n_scenarios), 0.0
    for sex, group in members.groupby("sex"):
        m: ProjectionModel = models[sex]
        out = core.portfolio_one_year(*m._args(parameter_uncertainty), m.rw.k_first, m.rw.n_increments,
                                      group["age"].to_numpy(np.int32), group["pension"].to_numpy(float), v,
                                      n_scenarios, seed, n_threads)
        values += out["value"]
        best += out["best_estimate"]
    return best, values
