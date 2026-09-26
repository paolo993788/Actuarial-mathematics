"""Projection model and Monte Carlo valuation of life annuities.

``ProjectionModel`` combines a fitted Lee-Carter model, its extension to the
closing age omega and the random walk with drift for k_t. The simulation
functions call the C++ engine; ``*_numpy`` functions are vectorised NumPy
reference implementations of the same algorithms (with a different random
number generator) used to validate the engine.

Timing: valuation at the end of the last calibration year T; a life aged x0
at valuation experiences in projection year T + h the rate of age x0 + h - 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import require_cpp
from .lee_carter import LeeCarterFit, RandomWalkDrift
from .life_table import annuity_due, q_from_m


def extend_to_closing_age(fit: LeeCarterFit, omega=120, slope_ages=15, b_ages=10):
    """Extend a_x and b_x beyond the highest fitted age up to omega.

    a_x is extrapolated linearly in age (Gompertz law for ln m), fitted by
    least squares on the last `slope_ages` ages; b_x is held at the average of
    the last `b_ages` fitted ages. Returns (ages, ax, bx) for ages
    min_age..omega.
    """
    ages = fit.ages.astype(int)
    if omega <= ages.max():
        return ages, fit.ax.copy(), fit.bx.copy()
    slope, intercept = np.polyfit(ages[-slope_ages:], fit.ax[-slope_ages:], 1)
    extra = np.arange(ages.max() + 1, omega + 1)
    ax = np.concatenate([fit.ax, intercept + slope * extra])
    bx = np.concatenate([fit.bx, np.full(extra.size, fit.bx[-b_ages:].mean())])
    return np.arange(ages.min(), omega + 1), ax, bx


@dataclass
class ProjectionModel:
    age_min: int
    ax: np.ndarray
    bx: np.ndarray
    rw: RandomWalkDrift

    @classmethod
    def from_fit(cls, fit: LeeCarterFit, rw: RandomWalkDrift | None = None, omega=120) -> "ProjectionModel":
        if rw is None:
            rw = RandomWalkDrift.from_kt(fit.years, fit.kt)
        ages, ax, bx = extend_to_closing_age(fit, omega)
        return cls(int(ages[0]), ax, bx, rw)

    @property
    def omega(self) -> int:
        return self.age_min + self.ax.size - 1

    @property
    def valuation_year(self) -> int:
        """Last calibration year T: the valuation date is 31 December of T."""
        return self.rw.last_year

    def horizon(self, age0: int) -> int:
        return self.omega - age0 + 1

    def cohort_q(self, age0: int, k_path) -> np.ndarray:
        """Death probabilities along k_path[h-1], h = 1..H (closed with q = 1 at omega)."""
        H = self.horizon(age0)
        idx = age0 - self.age_min + np.arange(H)
        q = q_from_m(np.exp(self.ax[idx] + self.bx[idx] * np.asarray(k_path, dtype=float)[:H]))
        q[-1] = 1.0
        return q

    def central_q(self, age0: int) -> np.ndarray:
        return self.cohort_q(age0, self.rw.central_path(self.horizon(age0)))

    def period_rates(self, k):
        """Central death rates for all ages at period index k (a period life table)."""
        return np.exp(self.ax + self.bx * k)

    def best_estimate(self, age0: int, discount) -> float:
        return annuity_due(self.central_q(age0), discount)

    def _args(self, parameter_uncertainty: bool):
        return (self.age_min, self.ax, self.bx, self.rw.k_last, self.rw.drift, self.rw.sigma,
                self.rw.drift_se if parameter_uncertainty else 0.0)


def simulate_annuity(model: ProjectionModel, age0: int, discount, n_scenarios=10_000, n_lives=0, seed=12345,
                     parameter_uncertainty=True, n_threads=0, keep_paths=False) -> dict:
    """C++ Monte Carlo of the annuity value (see cpp/mortality.hpp)."""
    core = require_cpp()
    return core.simulate_annuity(*model._args(parameter_uncertainty), age0, np.asarray(discount, dtype=float),
                                 n_scenarios, n_lives, seed, n_threads, keep_paths)


def one_year_recalibration(model: ProjectionModel, age0: int, discount, n_scenarios=100_000, seed=12345,
                           parameter_uncertainty=True, n_threads=0) -> dict:
    """C++ one-year view of longevity trend risk (Richards, Currie and Ritchie, 2014)."""
    core = require_cpp()
    return core.one_year_recalibration(*model._args(parameter_uncertainty), model.rw.k_first, model.rw.n_increments,
                                       age0, np.asarray(discount, dtype=float), n_scenarios, seed, n_threads)


# --------------------------------------------------------------------------- NumPy references


def _q_matrix(model: ProjectionModel, age0: int, k_paths):
    H = model.horizon(age0)
    idx = age0 - model.age_min + np.arange(H)
    q = q_from_m(np.exp(model.ax[idx][None, :] + model.bx[idx][None, :] * k_paths))
    q[:, -1] = 1.0
    return q


def simulate_annuity_numpy(model: ProjectionModel, age0: int, discount, n_scenarios=10_000, n_lives=0, seed=12345,
                           parameter_uncertainty=True):
    rng = np.random.default_rng(seed)
    H = model.horizon(age0)
    rw = model.rw
    drift = rw.drift + (rw.drift_se if parameter_uncertainty else 0.0) * rng.standard_normal(n_scenarios)
    k_paths = rw.k_last + np.cumsum(drift[:, None] + rw.sigma * rng.standard_normal((n_scenarios, H)), axis=1)
    q = _q_matrix(model, age0, k_paths)
    S = np.concatenate([np.ones((n_scenarios, 1)), np.cumprod(1.0 - q, axis=1)], axis=1)
    v = np.asarray(discount, dtype=float)[:H]
    out = {"pv_systematic": S[:, :H] @ v, "life_expectancy": S[:, 1:].sum(axis=1)}
    if n_lives > 0:
        cum_v = np.cumsum(v)
        u = rng.random((n_scenarios, n_lives))
        # K = number of t in 1..H with S(t) > u
        K = (S[:, None, 1:] > u[:, :, None]).sum(axis=2)
        out["pv_portfolio"] = cum_v[K].mean(axis=1)
    return out


def one_year_recalibration_numpy(model: ProjectionModel, age0: int, discount, n_scenarios=100_000, seed=12345,
                                 parameter_uncertainty=True):
    rng = np.random.default_rng(seed)
    rw = model.rw
    H = model.horizon(age0)
    v = np.asarray(discount, dtype=float)[:H]
    best = annuity_due(model.central_q(age0), v)
    drift = rw.drift + (rw.drift_se if parameter_uncertainty else 0.0) * rng.standard_normal(n_scenarios)
    k1 = rw.k_last + drift + rw.sigma * rng.standard_normal(n_scenarios)
    new_drift = (k1 - rw.k_first) / (rw.n_increments + 1)
    i0 = age0 - model.age_min
    p = np.exp(-np.exp(model.ax[i0] + model.bx[i0] * k1))
    future_k = k1[:, None] + new_drift[:, None] * np.arange(1, H)[None, :]
    q = _q_matrix(model, age0 + 1, future_k)
    S = np.concatenate([np.ones((n_scenarios, 1)), np.cumprod(1.0 - q, axis=1)], axis=1)[:, : H - 1]
    value = v[0] + p * (S @ v[1:H])
    return {"best_estimate": best, "value": value, "k_next": k1, "drift_next": new_drift}
