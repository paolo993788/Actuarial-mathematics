# Longevity risk: Lee-Carter projections and C++ Monte Carlo valuation of life annuities

Python package with a C++17 engine (pybind11) to project mortality with the Lee-Carter model, value life annuities under stochastic mortality and compare internal-model measures of longevity risk with the Solvency II standard formula. It downloads official Eurostat mortality data and the ECB euro area yield curve, which is extrapolated with the Smith-Wilson method used by EIOPA.

## Requirements

- Python 3.10 or later (developed and tested with Python 3.11).
- A C++17 compiler:
  - Windows: Visual Studio 2019 or later, or the free *Build Tools for Visual Studio* with the "Desktop development with C++" workload;
  - macOS: Xcode Command Line Tools (`xcode-select --install`);
  - Linux: GCC 9 or later, or Clang 10 or later.
- Python dependencies: [`requirements.txt`](requirements.txt) (NumPy, SciPy, pandas, Matplotlib, pybind11, pytest, ipykernel).

## Usage

### Set-up (from the repository root)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/longevity_risk/requirements.txt
python -m pip install -e scripts/longevity_risk
```

The last command compiles `cpp/bindings.cpp` into `longevity_risk._core`. Run it again after editing any C++ file.

### Visual Studio Code

1. Install the *Python*, *Jupyter* and *C/C++* extensions.
2. Open the repository folder and select the `.venv` interpreter (*Python: Select Interpreter*).
3. Open a notebook and choose the same environment as kernel (*Select Kernel*).
4. For C++ IntelliSense, add the include folders printed by `python -m pybind11 --includes` to the C/C++ configuration.

### Tests and data

```bash
python -m pytest tests/longevity_risk                                  # validation suite
python -m longevity_risk.data --geo IT --sex M F --yield-curve 2024-12-01 2024-12-31   # optional pre-download
```

### Notebooks

| Notebook | Content |
| --- | --- |
| [`notebooks/mortality_projection/lee_carter_mortality_projection.ipynb`](../../notebooks/mortality_projection/lee_carter_mortality_projection.ipynb) | Poisson and SVD Lee-Carter fits on Eurostat data for men and women, residual diagnostics, random walk with drift with and without the pandemic years, out-of-sample backtest, period versus cohort life expectancy. |
| [`notebooks/longevity_risk/annuity_longevity_risk_solvency.ipynb`](../../notebooks/longevity_risk/annuity_longevity_risk_solvency.ipynb) | Smith-Wilson curve from the ECB curve, best estimates of life annuities, run-off and one-year value at risk, idiosyncratic risk and pooling, comparison with the Solvency II longevity shock and interest-rate sensitivity. |

The country (default Italy, `COUNTRY = "IT"`), sex, age range and other parameters are set in the first code cell of each notebook. Official data are downloaded by default; set `LONGEVITY_RISK_DATA_MODE=synthetic` before starting Jupyter to run offline on simulated data.

## Inputs

| Name | Format | Description |
| --- | --- | --- |
| Eurostat `demo_magec` | JSON-stat from the Eurostat dissemination API | Deaths by single age (last birthday), sex, country and calendar year (persons). |
| Eurostat `demo_pjan` | JSON-stat from the Eurostat dissemination API | Population on 1 January by single age, sex and country (persons). |
| ECB euro area yield curve (AAA) | CSV from the ECB Data Portal, series `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.{BETA0..BETA3,TAU1,TAU2}` | Daily Svensson parameters (betas in percent, taus in years). |

Downloads are cached in `data/raw/eurostat/` and `data/raw/ecb/`, which Git ignores; `LONGEVITY_RISK_DATA_DIR` changes the cache folder. Eurostat and the ECB allow reuse of their statistics with acknowledgement of the source; the data are downloaded rather than committed so that their source and vintage remain explicit.

## Outputs

| File | Description |
| --- | --- |
| `outputs/lee_carter/*.png` | Crude rates, Lee-Carter parameters, residuals, backtest and projections. |
| `outputs/longevity_scr/*.png`, `scr_comparison.csv` | Discount curve, run-off distribution and capital requirements by age. |

## Method

**Units and conventions.** Ages are ages last birthday (integer), time in years. $m_{x,t}$ is the central death rate (deaths per person-year), $E_{x,t} = \tfrac12(P_{x,t} + P_{x,t+1})$ the central exposure from the populations on 1 January, and $q = 1 - e^{-m}$ the one-year death probability (constant force within each year of age). The valuation date is 31 December of the last calibration year $T$; a life aged $x_0$ at valuation experiences in year $T+h$ the rate of age $x_0 + h - 1$ (cohort rates). Annuities-due pay 1 at $t = 0, 1, \dots$ while the annuitant is alive. The ECB Svensson rates are continuously compounded; the Smith-Wilson curve and the UFR use annual compounding.

**Lee-Carter** (`longevity_risk/lee_carter.py`): $\ln m_{x,t} = a_x + b_x k_t$ with $\sum_x b_x = 1$, $\sum_t k_t = 0$. Poisson maximum likelihood (Brouhns, Denuit and Vermunt, 2002) with Newton updates of $a_x$, $k_t$ and $b_x$ until no fitted log-rate changes by more than $10^{-10}$, starting from the SVD solution (Lee and Carter, 1992). Selected years (by default 2020-2022) can receive zero weight. $k_t$ follows a random walk with drift, $d = (k_T - k_{t_0})/(T - t_0)$, $\sigma^2$ from the increments (increments over gaps count as several annual steps), $\text{s.e.}(d) = \sigma/\sqrt{T - t_0}$.

**Closing the table** (`longevity_risk/simulation.py`): above the highest fitted age, $a_x$ is extrapolated linearly (Gompertz law, least squares on the last 15 ages) and $b_x$ is held at the average of the last 10 ages; $q_{120} = 1$.

**C++ engine** (`cpp/mortality.hpp`): each scenario draws the drift from $N(d, \text{s.e.}(d)^2)$ (parameter uncertainty, optional) and simulates $k_{T+1}, k_{T+2}, \dots$; it returns the expected present value of the annuity (systematic risk), the curtate cohort life expectancy and, for a portfolio of $N$ lives, the average present value obtained by drawing each curtate lifetime $K$ by inversion ($P(K \ge t) = {}_tp_{x_0}$, binary search). The one-year view (Richards, Currie and Ritchie, 2014) simulates $k_{T+1}$, re-estimates $d' = (k_{T+1} - k_{t_0})/(T + 1 - t_0)$ and revalues the liability at the end of the year: $X = v(0) + p_{x_0}(k_{T+1}) \sum_{j \ge 0} {}_jp'_{x_0+1}\, v(1 + j)$.

**Discount curve** (`longevity_risk/curves.py`): Smith-Wilson extrapolation of the ECB Svensson zero-coupon prices at 1-20 years (last liquid point 20 years for the euro), $\omega = \ln(1 + \text{UFR})$, $\alpha$ = smallest value $\ge 0.05$ such that the one-year forward rate at the convergence point $\max(\text{LLP} + 40, 60)$ is within 1 basis point of the UFR (bisection to $10^{-6}$). EIOPA applies the method to swap rates with a credit risk adjustment; here it is applied to the ECB AAA government curve for illustration. The notebooks use UFR = 3.30%.

**Solvency II** (`longevity_risk/solvency.py`): standard-formula longevity SCR = $\text{BE}(0.8\,q) - \text{BE}(q)$ (Delegated Regulation (EU) 2015/35, Article 138), without shocking the closing age.

**Random numbers and parallelism**: xoshiro256** with SplitMix64 seeding and Box-Muller normals (portable across compilers); scenario $s$ uses its own random stream derived from `(seed, s)`, so results do not depend on the number of threads. The notebooks fix `SEED = 20240101` and report Monte Carlo standard errors.

## Verification

Run `python -m pytest tests/longevity_risk` from the repository root (31 tests, a few seconds). Main checks:

| Check | Tolerance and justification |
| --- | --- |
| Poisson Lee-Carter on noise-free expected deaths recovers the true parameters | $10^{-8}$ for $a_x, b_x$, $10^{-6}$ for $k_t$ (convergence tolerance $10^{-10}$ on log-rates) |
| SVD fit on exactly rank-one log-rates | $10^{-10}$ |
| Identifiability constraints $\sum b_x = 1$, $\sum k_t = 0$ | $10^{-12}$ and $10^{-9}$ |
| Deviance under the true model vs degrees of freedom | within 4 standard deviations of the $\chi^2$ distribution |
| Random walk estimators (drift, $\sigma$, s.e., gaps) | exact formulas |
| Life expectancy with constant force $\mu$ equals $1/\mu$; annuity-due with constant $q$ and $i$ equals the geometric sum; variance of the annuity by enumeration | relative $10^{-10}$ to $10^{-13}$ |
| C++ scenarios without risk equal the Python best estimate; cohort survival C++ vs Python | relative $10^{-13}$ |
| Idiosyncratic simulation: mean and variance vs exact moments of the curtate lifetime | mean within 4 standard errors, variance within 5% (sampling error about 1%) |
| C++ vs NumPy reference in distribution (mean, standard deviation, quantiles) | 4 standard errors, 3%, 0.2% |
| One-year recalibration without risk returns the best estimate; C++ vs NumPy quantiles | relative $10^{-12}$; 0.1% |
| Reproducibility with 1 and 4 threads | bitwise equality |
| Smith-Wilson fits the input prices; convergence gap at 60 years; minimality of $\alpha$; flat UFR input gives $\zeta = 0$ | relative $10^{-12}$; 1 bp; $\alpha - 10^{-4}$ fails; $10^{-10}$ |
| Standard-formula shock with constant $q$ | closed form, relative $10^{-12}$ |
| Eurostat JSON-stat and ECB CSV parsers | exact on sample documents |

## References

- Börger, M. (2010). Deterministic shock vs. stochastic value-at-risk: an analysis of the Solvency II standard model approach to longevity risk. *Blätter der DGVFM*, 31(2), 225-259.
- Blackman, D. and Vigna, S. (2021). Scrambled linear pseudorandom number generators. *ACM Transactions on Mathematical Software*, 47(4). Public-domain reference code: https://prng.di.unimi.it/
- Brouhns, N., Denuit, M. and Vermunt, J. K. (2002). A Poisson log-bilinear regression approach to the construction of projected lifetables. *Insurance: Mathematics and Economics*, 31(3), 373-393.
- Commission Delegated Regulation (EU) 2015/35 of 10 October 2014 supplementing Directive 2009/138/EC (Solvency II), Article 138.
- EIOPA. Technical documentation of the methodology to derive EIOPA's risk-free interest rate term structures. https://www.eiopa.europa.eu/tools-and-data/risk-free-interest-rate-term-structures_en
- Lee, R. D. and Carter, L. R. (1992). Modeling and forecasting U.S. mortality. *Journal of the American Statistical Association*, 87(419), 659-671.
- Richards, S. J., Currie, I. D. and Ritchie, G. P. (2014). A value-at-risk framework for longevity trend risk. *British Actuarial Journal*, 19(1), 116-139.
- Smith, A. and Wilson, T. (2001). Fitting yield curves with long term constraints. Research notes, Bacon and Woodrow.
- Svensson, L. E. O. (1994). Estimating and interpreting forward interest rates: Sweden 1992-1994. NBER Working Paper 4871.
- Eurostat, population and mortality statistics: https://ec.europa.eu/eurostat/data/database. European Central Bank, euro area yield curves: https://www.ecb.europa.eu/stats/
