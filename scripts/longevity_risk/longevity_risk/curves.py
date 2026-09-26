"""Discount curves: ECB Svensson curve and Smith-Wilson extrapolation.

* ``SvenssonCurve`` evaluates the ECB euro area yield curve from the daily
  Svensson (1994) parameters (betas in percent, taus in years); the spot
  rates are continuously compounded.
* ``SmithWilson`` implements the extrapolation method prescribed by EIOPA for
  the Solvency II risk-free rate: the curve fits the prices of zero-coupon
  bonds up to the last liquid point exactly and its forward rates converge
  to the ultimate forward rate (UFR). The convergence speed alpha is the
  smallest value >= 0.05 such that the forward rate at the convergence point
  (max(LLP + 40, 60) years) is within 1 basis point of the UFR.

EIOPA derives the euro risk-free curve from swap rates with a credit risk
adjustment; here the same extrapolation technique is applied to the ECB AAA
government curve for illustration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SvenssonCurve:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    @classmethod
    def from_series(cls, row) -> "SvenssonCurve":
        return cls(*(float(row[k]) for k in ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")))

    def zero_rate(self, maturity):
        """Continuously compounded spot rate (decimal)."""
        m = np.maximum(np.asarray(maturity, dtype=float), 1e-10)
        x1, x2 = m / self.tau1, m / self.tau2
        f1 = (1.0 - np.exp(-x1)) / x1
        f2 = f1 - np.exp(-x1)
        f3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
        return (self.beta0 + self.beta1 * f1 + self.beta2 * f2 + self.beta3 * f3) / 100.0

    def discount(self, maturity):
        m = np.asarray(maturity, dtype=float)
        return np.exp(-self.zero_rate(m) * m)


def _wilson(t, u, alpha, omega):
    """Wilson kernel W(t, u) for arrays t (rows) and u (columns)."""
    t = np.asarray(t, dtype=float)[:, None]
    u = np.asarray(u, dtype=float)[None, :]
    lo, hi = np.minimum(t, u), np.maximum(t, u)
    return np.exp(-omega * (t + u)) * (alpha * lo - 0.5 * np.exp(-alpha * hi) * (np.exp(alpha * lo) - np.exp(-alpha * lo)))


@dataclass
class SmithWilson:
    maturities: np.ndarray       # liquid maturities u_j (years)
    prices: np.ndarray           # zero-coupon bond prices P(u_j)
    ufr: float                   # ultimate forward rate, annual compounding (decimal)
    alpha: float
    zeta: np.ndarray = field(init=False)

    def __post_init__(self):
        self.maturities = np.asarray(self.maturities, dtype=float)
        self.prices = np.asarray(self.prices, dtype=float)
        W = _wilson(self.maturities, self.maturities, self.alpha, self.omega)
        self.zeta = np.linalg.solve(W, self.prices - np.exp(-self.omega * self.maturities))

    @property
    def omega(self) -> float:
        """UFR as a continuously compounded intensity."""
        return float(np.log1p(self.ufr))

    @property
    def convergence_point(self) -> float:
        return max(self.maturities.max() + 40.0, 60.0)

    def discount(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.exp(-self.omega * t) + _wilson(t, self.maturities, self.alpha, self.omega) @ self.zeta

    def zero_rate(self, t, compounding="annual"):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        p = self.discount(t)
        if compounding == "annual":
            return p ** (-1.0 / t) - 1.0
        if compounding == "continuous":
            return -np.log(p) / t
        raise ValueError("compounding must be 'annual' or 'continuous'")

    def forward_rate(self, t):
        """One-year forward rate between t - 1 and t, annual compounding."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.discount(t - 1.0) / self.discount(t) - 1.0

    def convergence_gap(self) -> float:
        return float(abs(self.forward_rate(self.convergence_point)[0] - self.ufr))

    @classmethod
    def calibrate(cls, maturities, prices, ufr, alpha_min=0.05, tolerance=1e-4, precision=1e-6) -> "SmithWilson":
        """Choose alpha as the smallest value >= alpha_min meeting the 1 bp convergence criterion (bisection)."""
        lower = cls(maturities, prices, ufr, alpha_min)
        if lower.convergence_gap() <= tolerance:
            return lower
        lo, hi = alpha_min, 1.0
        while cls(maturities, prices, ufr, hi).convergence_gap() > tolerance:
            hi *= 2.0
            if hi > 100:
                raise RuntimeError("no alpha satisfies the convergence criterion")
        while hi - lo > precision:
            mid = 0.5 * (lo + hi)
            if cls(maturities, prices, ufr, mid).convergence_gap() <= tolerance:
                hi = mid
            else:
                lo = mid
        return cls(maturities, prices, ufr, hi)


def smith_wilson_from_svensson(curve: SvenssonCurve, ufr: float, last_liquid_point: int = 20) -> SmithWilson:
    """Smith-Wilson curve fitted to the Svensson zero-coupon prices at 1, 2, ..., LLP years."""
    u = np.arange(1, last_liquid_point + 1, dtype=float)
    return SmithWilson.calibrate(u, curve.discount(u), ufr)
