"""Lee-Carter mortality model: estimation and time-series model for k_t.

Model for the central death rate m(x, t) at age x (last birthday) in
calendar year t:

    ln m(x, t) = a_x + b_x k_t,   with sum_x b_x = 1 and sum_t k_t = 0.

Two estimators are provided:

* ``fit_svd``: the original least-squares fit of Lee and Carter (1992) by
  singular value decomposition of the centred log-rates;
* ``fit_poisson``: the Poisson log-bilinear maximum-likelihood fit of
  Brouhns, Denuit and Vermunt (2002), D(x, t) ~ Poisson(E(x, t) m(x, t)),
  which weights each cell by its number of deaths.

The period index is projected with a random walk with drift,
k_t = k_{t-1} + d + sigma * eps_t (``RandomWalkDrift``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LeeCarterFit:
    ages: np.ndarray
    years: np.ndarray
    ax: np.ndarray
    bx: np.ndarray
    kt: np.ndarray            # NaN for years excluded from the fit
    method: str
    loglik: float = np.nan    # Poisson log-likelihood without the constant term (Poisson fit only)
    deviance: float = np.nan
    iterations: int = 0

    @property
    def fitted_years(self) -> np.ndarray:
        return self.years[~np.isnan(self.kt)]

    def log_rates(self) -> pd.DataFrame:
        """Fitted ln m(x, t) as an age x year table."""
        return pd.DataFrame(self.ax[:, None] + self.bx[:, None] * self.kt[None, :], index=self.ages, columns=self.years)


def _normalise(ax, bx, kt, mask):
    """Impose sum(b) = 1 and sum(k) = 0 over the fitted years, leaving a + b k unchanged."""
    scale = bx.sum()
    bx = bx / scale
    kt = kt * scale
    k_mean = kt[mask].mean()
    return ax + bx * k_mean, bx, kt - k_mean


def fit_svd(log_m: pd.DataFrame) -> LeeCarterFit:
    """Classical Lee-Carter fit by SVD of the centred log death rates (ages x years)."""
    values = log_m.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("log death rates must be finite (no zero or missing rates)")
    ax = values.mean(axis=1)
    u, s, vt = np.linalg.svd(values - ax[:, None], full_matrices=False)
    bx, kt = u[:, 0], s[0] * vt[0]
    ax, bx, kt = _normalise(ax, bx, kt, np.ones(kt.size, dtype=bool))
    return LeeCarterFit(log_m.index.to_numpy(), log_m.columns.to_numpy(), ax, bx, kt, "svd")


def _poisson_loglik(D, Dhat, w):
    return float(np.sum(w * (D * np.log(Dhat) - Dhat)))


def poisson_deviance_residuals(deaths: pd.DataFrame, exposures: pd.DataFrame, fit: LeeCarterFit) -> pd.DataFrame:
    D = deaths.to_numpy(dtype=float)
    Dhat = exposures.to_numpy(dtype=float) * np.exp(fit.log_rates().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(D > 0, D * np.log(D / Dhat), 0.0)
    dev = 2.0 * (term - (D - Dhat))
    return pd.DataFrame(np.sign(D - Dhat) * np.sqrt(np.maximum(dev, 0.0)), index=deaths.index, columns=deaths.columns)


def fit_poisson(deaths: pd.DataFrame, exposures: pd.DataFrame, fit_years=None, max_iter=2000, tol=1e-10) -> LeeCarterFit:
    """Poisson maximum-likelihood Lee-Carter fit (Brouhns, Denuit and Vermunt, 2002).

    Parameters are updated one block at a time with Newton steps (a_x, then
    k_t, then b_x) until no fitted log-rate a_x + b_x k_t changes by more
    than ``tol`` in an iteration. Years not in ``fit_years`` receive zero
    weight and their k_t is reported as NaN (for example to exclude pandemic
    years).
    """
    if not (deaths.index.equals(exposures.index) and deaths.columns.equals(exposures.columns)):
        raise ValueError("deaths and exposures must have the same ages and years")
    D = deaths.to_numpy(dtype=float)
    E = exposures.to_numpy(dtype=float)
    years = deaths.columns.to_numpy()
    mask = np.ones(years.size, dtype=bool) if fit_years is None else np.isin(years, list(fit_years))
    if mask.sum() < 3:
        raise ValueError("at least three calendar years are needed")
    if not (np.isfinite(D[:, mask]).all() and np.isfinite(E[:, mask]).all() and (E[:, mask] > 0).all()):
        raise ValueError("deaths and exposures must be finite, with positive exposures, in the fitted years")
    w = np.broadcast_to(mask.astype(float), D.shape)
    D0 = np.where(mask, D, 0.0)
    E0 = np.where(mask, E, 1.0)

    # Starting values from the SVD fit of smoothed crude rates.
    start = fit_svd(pd.DataFrame(np.log((D0[:, mask] + 0.5) / E0[:, mask]), index=deaths.index, columns=years[mask]))
    ax, bx = start.ax.copy(), start.bx.copy()
    kt = np.zeros(years.size)
    kt[mask] = start.kt

    def fitted():
        return E0 * np.exp(ax[:, None] + bx[:, None] * kt[None, :])

    eta = ax[:, None] + bx[:, None] * kt[None, :]
    iterations = 0
    for iterations in range(1, max_iter + 1):
        Dhat = fitted()
        ax = ax + np.sum(w * (D0 - Dhat), axis=1) / np.sum(w * Dhat, axis=1)
        Dhat = fitted()
        with np.errstate(divide="ignore", invalid="ignore"):  # zero weight in excluded years
            step = np.sum(w * (D0 - Dhat) * bx[:, None], axis=0) / np.sum(w * Dhat * bx[:, None] ** 2, axis=0)
        kt = np.where(mask, kt + step, kt)
        k_mean = kt[mask].mean()
        kt = np.where(mask, kt - k_mean, kt)
        ax = ax + bx * k_mean
        Dhat = fitted()
        bx = bx + np.sum(w * (D0 - Dhat) * kt[None, :], axis=1) / np.sum(w * Dhat * kt[None, :] ** 2, axis=1)
        new_eta = ax[:, None] + bx[:, None] * kt[None, :]
        change = np.max(np.abs(new_eta - eta)[:, mask])
        eta = new_eta
        if change < tol:
            break
    loglik = _poisson_loglik(D0, fitted(), w)
    ax, bx, kt = _normalise(ax, bx, kt, mask)
    kt = np.where(mask, kt, np.nan)
    fit = LeeCarterFit(deaths.index.to_numpy(), years, ax, bx, kt, "poisson", loglik, np.nan, iterations)
    residuals = poisson_deviance_residuals(deaths.loc[:, mask], exposures.loc[:, mask],
                                           LeeCarterFit(fit.ages, years[mask], ax, bx, kt[mask], "poisson"))
    fit.deviance = float(np.sum(residuals.to_numpy() ** 2))
    return fit


@dataclass
class RandomWalkDrift:
    """Random walk with drift for k_t, estimated on the available years.

    With consecutive years, d = (k_last - k_first) / n and sigma^2 is the
    sample variance of the increments; gaps (excluded years) are handled by
    treating an increment over g years as N(g d, g sigma^2).
    """

    drift: float
    sigma: float
    drift_se: float
    k_first: float
    k_last: float
    first_year: int
    last_year: int

    @property
    def n_increments(self) -> int:
        return self.last_year - self.first_year

    @classmethod
    def from_kt(cls, years, kt) -> "RandomWalkDrift":
        years = np.asarray(years)
        kt = np.asarray(kt, dtype=float)
        ok = ~np.isnan(kt)
        y, k = years[ok].astype(int), kt[ok]
        gaps = np.diff(y)
        increments = np.diff(k)
        total = y[-1] - y[0]
        drift = (k[-1] - k[0]) / total
        sigma2 = np.sum((increments - gaps * drift) ** 2 / gaps) / (increments.size - 1)
        sigma = float(np.sqrt(sigma2))
        return cls(float(drift), sigma, sigma / np.sqrt(total), float(k[0]), float(k[-1]), int(y[0]), int(y[-1]))

    def central_path(self, horizon: int) -> np.ndarray:
        """Median projection k_{T+h} = k_T + h d for h = 1..horizon."""
        return self.k_last + self.drift * np.arange(1, horizon + 1)

    def prediction_interval(self, horizon: int, level=0.95, parameter_uncertainty=True):
        """Gaussian prediction interval of k_{T+h}, including drift uncertainty if requested."""
        from scipy import stats

        h = np.arange(1, horizon + 1)
        var = h * self.sigma**2 + (h * self.drift_se) ** 2 * parameter_uncertainty
        z = stats.norm.ppf(0.5 + level / 2)
        centre = self.central_path(horizon)
        return centre - z * np.sqrt(var), centre + z * np.sqrt(var)
