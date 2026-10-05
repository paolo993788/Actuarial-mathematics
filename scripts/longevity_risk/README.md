# Longevity risk: Lee-Carter projections and C++ Monte Carlo valuation of life annuities

Python package with a C++17 engine (pybind11) to project mortality with the Lee-Carter model, value life annuities and portfolios of pensioners under stochastic mortality, and compute Solvency II quantities: standard-formula SCR (longevity, interest rate, operational; the expense shock of Article 140 is applied in the buy-in notebook), the cost-of-capital risk margin under the current and the amended rules, with the future SCR projected by the best-estimate proxy or explicitly, and internal-model measures of longevity risk. It downloads official Eurostat mortality data and the ECB euro area yield curve, which is extrapolated with the Smith-Wilson method used by EIOPA.

## Requirements

- Python 3.10 or later (developed and tested with Python 3.11).
- A C++17 compiler:
  - Windows: Visual Studio 2019 or later, or the free *Build Tools for Visual Studio* with the "Desktop development with C++" workload;
  - macOS: Xcode Command Line Tools (`xcode-select --install`);
  - Linux: GCC 9 or later, or Clang 10 or later.
- Python dependencies: [`requirements.txt`](requirements.txt) (NumPy, SciPy, pandas, Matplotlib, pybind11, pytest, ipykernel).
- Development tools: [`requirements-dev.txt`](requirements-dev.txt) (ruff, nbconvert).
- Exact versions: [`requirements-lock.txt`](requirements-lock.txt) pins the two files above to versions that pass the test suite and execute every notebook (Python 3.11, Linux). CI installs it on Python 3.11 and the unpinned files on 3.10 and 3.12.
- For the standalone C++ tests: CMake 3.16 or later (Ninja optional).

## Usage

### Set-up (from the repository root)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/longevity_risk/requirements.txt
python -m pip install -e scripts/longevity_risk
```

For the exact environment used by CI, install `requirements-lock.txt` instead of `requirements.txt`. The last command compiles `cpp/bindings.cpp` into `longevity_risk._core`. Run it again after editing any C++ file.

### Visual Studio Code

1. Install the *Python*, *Jupyter* and *C/C++* extensions.
2. Open the repository folder and select the `.venv` interpreter (*Python: Select Interpreter*).
3. Open a notebook and choose the same environment as kernel (*Select Kernel*).
4. For C++ IntelliSense, add the include folders printed by `python -m pybind11 --includes` to the C/C++ configuration.

### Tests and data

```bash
python -m pytest tests/longevity_risk                                  # validation suite
ruff check --select F scripts tests                                    # lint (pyflakes rules)
python -m longevity_risk.data --geo IT --sex M F --yield-curve 2024-12-01 2024-12-31   # optional pre-download

# Standalone C++ tests (strict warnings; add -DLR_SANITIZE=address,undefined or =thread)
cmake -S scripts/longevity_risk/cpp -B build/cpp -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/cpp && ctest --test-dir build/cpp --output-on-failure
```

### Notebooks

| Notebook | Content |
| --- | --- |
| [`notebooks/mortality_projection/lee_carter_mortality_projection.ipynb`](../../notebooks/mortality_projection/lee_carter_mortality_projection.ipynb) | Poisson and SVD Lee-Carter fits on Eurostat data for men and women, residual diagnostics, random walk with drift with and without the pandemic years, out-of-sample backtest, period versus cohort life expectancy. |
| [`notebooks/longevity_risk/annuity_longevity_risk_solvency.ipynb`](../../notebooks/longevity_risk/annuity_longevity_risk_solvency.ipynb) | Smith-Wilson curve from the ECB curve, best estimates of life annuities, run-off and one-year value at risk, idiosyncratic risk and pooling, comparison with the Solvency II longevity shock and interest-rate sensitivity. |
| [`notebooks/longevity_risk/multipopulation_longevity_hedging.ipynb`](../../notebooks/longevity_risk/multipopulation_longevity_hedging.ipynb) | Case study: Li-Lee model for Italy and 12 European countries, explanation ratios, mean-reverting national deviations, out-of-sample backtest against independent Lee-Carter models, coherent projections, and q-forward hedges of an Italian annuity on six indices by fund size and dynamics. |
| [`notebooks/case_studies/pension_buy_in_pricing.ipynb`](../../notebooks/case_studies/pension_buy_in_pricing.ipynb) | Case study: buy-in quote for a pension fund (cash flows, best estimate, standard-formula SCR, risk margin with the proxy and an explicit SCR projection, under the current rules and those applying from 30 January 2027, premium with hurdle rate, internal-model cross-check with Monte Carlo errors, sensitivities, board summary). An optional real membership file is read from `BUYIN_MEMBERSHIP_CSV` and must never be committed. |

The country (default Italy, `COUNTRY = "IT"`), sex, age range and other parameters are set in the first code cell of each notebook. Official data are downloaded by default; set `LONGEVITY_RISK_DATA_MODE=synthetic` before starting Jupyter to run offline on simulated data.

## Inputs

| Name | Format | Description |
| --- | --- | --- |
| Eurostat `demo_magec` | JSON-stat from the Eurostat dissemination API | Deaths by single age (last birthday), sex, country and calendar year (persons). |
| Eurostat `demo_pjan` | JSON-stat from the Eurostat dissemination API | Population on 1 January by single age, sex and country (persons). |
| Eurostat `demo_mlexpecedu` | JSON-stat from the Eurostat dissemination API | Life expectancy by age, sex and educational attainment (ISCED 2011 groups 0-2, 3-4, 5-8), years; used for the prior of the pensioner basis in the buy-in notebook. |
| ECB euro area yield curve (AAA) | CSV from the ECB Data Portal, series `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.{BETA0..BETA3,TAU1,TAU2}` | Daily Svensson parameters (betas in percent, taus in years). |

Downloads are cached in `data/raw/eurostat/` and `data/raw/ecb/`, which Git ignores; `LONGEVITY_RISK_DATA_DIR` changes the cache folder. Eurostat and the ECB allow reuse of their statistics with acknowledgement of the source; the data are downloaded rather than committed so that their source and vintage remain explicit.

## Outputs

| File | Description |
| --- | --- |
| `outputs/lee_carter/*.png` | Crude rates, Lee-Carter parameters, residuals, backtest and projections. |
| `outputs/longevity_scr/*.png`, `scr_comparison.csv` | Discount curve, run-off distribution and capital requirements by age. |
| `outputs/multipopulation/*.png` | Life expectancy by country, Li-Lee parameters and national effects, projected gap between Italy and the European index. |
| `outputs/buy_in_pricing/*.png`, `board_summary.txt` | Membership, cash flows, projected SCR, premium waterfall, sensitivities and the board summary of the case study. |
| `docs/figures/*-light.png`, `*-dark.png` | README charts drawn by `python -m longevity_risk.readme_figures` (official data; `--synthetic` offline); the only generated files committed. |

## Method

**Units and conventions.** Ages are ages last birthday (integer), time in years. $m_{x,t}$ is the central death rate (deaths per person-year), $E_{x,t} = \tfrac12(P_{x,t} + P_{x,t+1})$ the central exposure from the populations on 1 January, and $q = 1 - e^{-m}$ the one-year death probability (constant force within each year of age). The valuation date is 31 December of the last calibration year $T$; a life aged $x_0$ at valuation experiences in year $T+h$ the rate of age $x_0 + h - 1$ (cohort rates). Annuities-due pay 1 at $t = 0, 1, \dots$ while the annuitant is alive. The ECB Svensson rates are continuously compounded; the Smith-Wilson curve and the UFR use annual compounding.

**Lee-Carter** (`longevity_risk/lee_carter.py`): $\ln m_{x,t} = a_x + b_x k_t$ with $\sum_x b_x = 1$, $\sum_t k_t = 0$. Poisson maximum likelihood (Brouhns, Denuit and Vermunt, 2002) with Newton updates of $a_x$, $k_t$ and $b_x$ until no fitted log-rate changes by more than $10^{-10}$, starting from the SVD solution (Lee and Carter, 1992). Selected years (by default 2020-2022) can receive zero weight. $k_t$ follows a random walk with drift, $d = (k_T - k_{t_0})/(T - t_0)$, $\sigma^2 = \sum_j (\Delta k_j - g_j d)^2 / g_j \,/\, (m - 1)$ over the $m$ observed increments, an increment over a gap of $g_j$ years counting as $N(g_j d, g_j \sigma^2)$, and $\text{s.e.}(d) = \sigma/\sqrt{T - t_0}$; at least three fitted years are required.

**Closing the table** (`longevity_risk/simulation.py`): above the highest fitted age, $a_x$ is extrapolated linearly (Gompertz law, least squares on the last 15 ages) and $b_x$ is held at the average of the last 10 ages; $q_{120} = 1$.

**C++ engine** (`cpp/mortality.hpp`): each scenario draws the drift from $N(d, \text{s.e.}(d)^2)$ (parameter uncertainty, optional) and simulates $k_{T+1}, k_{T+2}, \dots$; it returns the expected present value of the annuity (systematic risk), the curtate cohort life expectancy and, for a portfolio of $N$ lives, the average present value obtained by drawing each curtate lifetime $K$ by inversion ($P(K \ge t) = {}_tp_{x_0}$, binary search). The one-year view (Richards, Currie and Ritchie, 2014) simulates $k_{T+1}$, re-estimates $d' = (k_{T+1} - k_{t_0})/(T + 1 - t_0)$ and revalues the liability at the end of the year: $X = v(0) + p_{x_0}(k_{T+1}) \sum_{j \ge 0} {}_jp'_{x_0+1}\, v(1 + j)$.

**Discount curve** (`longevity_risk/curves.py`): Smith-Wilson extrapolation of the ECB Svensson zero-coupon prices at 1-20 years (last liquid point 20 years for the euro), $\omega = \ln(1 + \text{UFR})$, $\alpha$ = smallest value $\ge 0.05$ such that the one-year forward rate at the convergence point $\max(\text{LLP} + 40, 60)$ is within 1 basis point of the UFR (bisection to $10^{-6}$). EIOPA applies the method to swap rates with a credit risk adjustment; here it is applied to the ECB AAA government curve for illustration. The notebooks use UFR = 3.30%.

**Solvency II** (`longevity_risk/solvency.py`): standard-formula longevity SCR = $\text{BE}(0.8\,q) - \text{BE}(q)$ (Delegated Regulation (EU) 2015/35, Article 138), without shocking the closing age; interest-rate SCR with the relative up and down shocks of Articles 166-167 (linear interpolation between 20 and 90 years, up shock of at least one percentage point, no down shock for negative rates); aggregation with the market-life correlation of 0.25 (Directive, Annex IV); operational risk $\min(30\%\,\text{BSCR}, 0.45\%\,\text{BE})$, the provisions-based charge on technical provisions without the risk margin (Article 204(4)); risk margin $\text{CoC}\sum_{t \ge 0} \max(\lambda^t, f)\,\text{SCR}(t)\,v(t+1)$ with a 6% cost of capital and $\lambda = 1$ (Articles 37 and 39), or with 4.75%, $\lambda = 0.96$ and floor $f = 50\%$ under the amended Articles 37 and 39 that apply from 30 January 2027 (Delegated Regulation (EU) 2026/269); $\text{SCR}(t)$ is projected in proportion to the best-estimate run-off (the proxy of the EIOPA guidelines) or explicitly (`portfolio.longevity_scr_runoff`: the longevity shock re-applied at each future date to the business then expected in force); the discount vector must cover the run-off plus one year.

**Portfolios** (`longevity_risk/portfolio.py`, `cpp/mortality.hpp`): members with sex, age and annual pension; expected cash flows under the central projection (optionally with shocked mortality or indexation); C++ simulation of the whole portfolio in which every member shares the scenario of the period index and, optionally, has an individually simulated lifetime; one-year trend risk of the portfolio. The two sexes use the same scenarios of the period index (perfectly correlated trends, a prudent simplification), while their lifetimes come from separate random streams and are independent.

**Experience rating** (`longevity_risk/experience.py`): the pensioner basis is a multiplier $\theta$ on the population's central death rates, with Poisson deaths; the fund's estimate is the actual-to-expected ratio $D/E$ (relative standard error $1/\sqrt{D}$), combined with a prior by Bühlmann-Straub credibility $Z\,D/E + (1-Z)\mu$, $Z = E/(E + \mu/\tau^2)$, which equals the posterior mean of a Gamma prior with mean $\mu$ and standard deviation $\tau$. The prior's spread comes from the socio-economic gradient of life expectancy at 65 (Eurostat `demo_mlexpecedu`): each education group's multiplier reproduces its life expectancy on the fitted population rates, relative to the whole population.

**Multi-population model and index hedging** (`longevity_risk/multipopulation.py`): Li-Lee augmented common factor model (Li and Lee, 2005), $\ln m_{x,t,i} = a_{x,i} + B_x K_t + b_{x,i} k_{t,i}$, fitted by Poisson maximum likelihood in two stages: a Lee-Carter model for the aggregate of the group, then for each population a Lee-Carter fit with $B_x K_t$ as an offset (the `offset` argument of `fit_poisson`). The explanation ratios are the Poisson-deviance analogue of Li and Lee's $R$ ratios. $K_t$ is a random walk with drift and each $k_{t,i}$ a zero-mean AR(1) estimated by least squares on consecutive fitted years, capped at 0.98 (or a driftless random walk for the non-coherent alternative); the innovations are jointly Gaussian with their sample covariance (simulated with the eigendecomposition factor $V\Lambda^{1/2}$, negative eigenvalues clipped to zero, which is robust when the covariance is near-singular). The hedged liability is an annuity-due split at the hedge horizon: payments on the simulated survivors up to the horizon plus the best-estimate annuity re-projected from the simulated state at the horizon, with binomial deaths for a finite fund. q-forwards pay the difference between a fixed rate and the realised index death probability at the horizon. Hedge ratios are least-squares coefficients on calibration scenarios, evaluated on independent test scenarios (variance, standard deviation and 99.5% VaR reduction; Coughlan et al., 2011; Li and Hardy, 2011).

**Random numbers and parallelism**: xoshiro256** with SplitMix64 seeding and Box-Muller normals (portable across compilers); scenario $s$ uses its own random stream derived from `(seed, s)`, so results do not depend on the number of threads. The notebooks fix their seeds (`SEED = 20240101`; 20240102 for the independent test scenarios of the hedging notebook). The Lee-Carter, Solvency II and buy-in notebooks report the Monte Carlo standard errors of their simulated headline figures; for quantiles they are estimated by bootstrapping the scenarios (`simulation.quantile_standard_error`).

## Verification

Run `python -m pytest tests/longevity_risk` from the repository root (65 tests, about 15 seconds). Main checks:

| Check | Tolerance and justification |
| --- | --- |
| Poisson Lee-Carter on noise-free expected deaths recovers the true parameters | $10^{-8}$ for $a_x, b_x$, $10^{-6}$ for $k_t$ (convergence tolerance $10^{-10}$ on log-rates) |
| SVD fit on exactly rank-one log-rates | $10^{-10}$ |
| Identifiability constraints $\sum b_x = 1$, $\sum k_t = 0$ | $10^{-12}$ and $10^{-9}$ |
| Deviance under the true model vs degrees of freedom | within 4 standard deviations of the $\chi^2$ distribution |
| Random walk estimators (drift, $\sigma$, s.e.), including the gap-weighted variance with a four-year increment; fewer than three fitted years rejected | exact formulas; relative $10^{-12}$ |
| Zero-weight years: the fit equals the fit on the remaining columns ($a_x$, $b_x$, $k_t$, deviance, iterations) | $10^{-12}$ |
| Life expectancy with constant force $\mu$ equals $1/\mu$; annuity-due with constant $q$ and $i$ equals the geometric sum; variance of the annuity by enumeration | relative $10^{-10}$ to $10^{-13}$ |
| C++ scenarios without risk equal the Python best estimate; cohort survival C++ vs Python | relative $10^{-13}$ |
| Idiosyncratic simulation: mean and variance vs exact moments of the curtate lifetime | mean within 4 standard errors, variance within 5% (sampling error about 1%) |
| C++ vs NumPy reference in distribution (mean, standard deviation, quantiles) | 4 standard errors, 3%, 0.2% |
| One-year recalibration without risk returns the best estimate; C++ vs NumPy quantiles | relative $10^{-12}$; 0.1% |
| Reproducibility with 1 and 4 threads | bitwise equality |
| Ages outside the fitted table and a closing age not above the fitted ages are rejected | exact |
| Bootstrap standard error of a 99.5% quantile vs the asymptotic formula $\sqrt{p(1-p)/n}/f(q)$ | 15% (500 replicates) |
| Smith-Wilson fits the input prices; convergence gap at 60 years; minimality of $\alpha$; flat UFR input gives $\zeta = 0$ | relative $10^{-12}$; 1 bp; $\alpha - 10^{-4}$ fails; $10^{-10}$ |
| Standard-formula shock with constant $q$ | closed form, relative $10^{-12}$ |
| Eurostat JSON-stat and ECB CSV parsers | exact on sample documents |
| Single-member portfolio equals the single-annuity engine; portfolio one-year value equals the weighted sum of members | relative $10^{-13}$ and $10^{-12}$ (same random streams) |
| Portfolio cash flows reproduce the best estimates of the members | relative $10^{-12}$ |
| Idiosyncratic risk of a heterogeneous portfolio: mean and variance against the exact sum of individual variances | 4 standard errors; 5% |
| Two sexes on the same trend scenarios: residuals of men and women uncorrelated and variance equal to the sum of individual variances; a single-sex portfolio unchanged | $|\rho| < 0.03$; 5%; bitwise |
| Indexed cash flows, the 0.8 mortality multiplier and the one-year portfolio values with two sexes equal the member-by-member identities | relative $10^{-12}$ |
| Explicit longevity SCR projection: SCR(0) equals the shocked minus the base best estimate, SCR(t) equals an independent computation, SCR(t)/BE(t) rising; risk margin from a path equals the proxy when SCR(t) is proportional to BE(t) | relative $10^{-12}$; $10^{-14}$ |
| Interest-rate shock table, interpolation, minimum up shock and negative rates; SCR of a single cash flow | exact values |
| Risk margin with constant SCR and flat curve, also under the amended formula ($\max(0.96^t, 50\%)$, 4.75%); a discount vector shorter than the run-off plus one year rejected; BSCR aggregation; operational risk cap; best-estimate run-off; interest-rate SCR zero when both shocks lower the liability (negative rates) | closed forms; exact |
| Experience rating: actual-to-expected ratio is the Poisson maximum-likelihood estimate with relative standard error $1/\sqrt{D}$; credibility estimate equals the Gamma-Poisson posterior mean, with the limits $Z = 0$ and $Z \to 1$; multiplier from a life expectancy against the constant-force closed form; education multipliers reproduce each group's life expectancy (sample JSON-stat document); simulated experience recovers the multiplier | exact; relative $10^{-14}$; $10^{-10}$; $10^{-10}$; 4 standard errors |
| Poisson fit with a constant offset shifts $a_x$ by exactly the offset and leaves $b_x$, $k_t$ and the deviance unchanged | $10^{-8}$ ($10^{-6}$ for $k_t$) |
| Li-Lee constraints ($\sum B = \sum b_i = 1$, $\sum K = \sum k_i = 0$, excluded years), deviance ordering and ratios in $[0, 1]$ | $10^{-8}$; exact inequalities |
| AR(1) recovery on a simulated panel with $\phi = 0.8$; the cap binds below the estimates and the innovations are the residuals at the capped coefficient; random-walk option; number of consecutive pairs | between 0.5 and 1.05 (41 increments); relative $10^{-12}$; exact |
| Simulated innovations: the factor reproduces the covariance; sample covariance of 200,000 draws | $10^{-12}$; 2% of the largest entry |
| Extended parameters of equal length for every closing age; ages outside the fitted range rejected by the liability and index functions | exact |
| Simulation without noise equals the central projection; split liability without noise equals the annuity on the central path | $10^{-12}$ |
| Index death rates vs the formula; hedge effectiveness 1 for an exactly linear liability and about 0 for independent noise | $10^{-12}$; 1% |
| Temporary life expectancy with constant force; sampling risk lowers hedge effectiveness | relative $10^{-9}$ to $10^{-12}$; ordering |

## C++ unit tests and sanitizers

**Tests.** The engine is header-only, so [`cpp/CMakeLists.txt`](cpp/CMakeLists.txt) also compiles it without Python. It uses `-Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Wnull-dereference -Wdouble-promotion -Werror` (`/W4 /WX` with MSVC) and bounds-checked standard containers outside Release builds. [`cpp/tests/test_core.cpp`](cpp/tests/test_core.cpp) runs 110 checks without any test framework. The Python suite compares the engine with a NumPy reference implementation; these tests add closed forms, exact identities, thread invariance and argument validation on the C++ code alone:

| Check | Tolerance |
| --- | --- |
| SplitMix64 and xoshiro256** against the published reference outputs; uniform and normal moments | exact; 4 standard errors |
| `parallel_for`: same result for 1, 3 and 8 threads; an exception in a task is rethrown | exact |
| Lee-Carter death probability $q = 1 - e^{-m}$, closure at the highest age, horizon, validation of the parameters | $10^{-15}$; exact |
| Constant force of mortality: annuity-due $(1 - (pv)^H)/(1 - pv)$ and curtate life expectancy $p(1 - p^{H-1})/(1 - p)$ in every scenario | $10^{-12}$ |
| Deterministic projection vs a hand computation along $k_{T+h} = k_T + hd$; a faster fall in $k$ raises the annuity and the life expectancy | $10^{-12}$; ordering |
| Stochastic projection: mean and variance of $k_{T+1}$ and $k_{T+H}$ ($h\sigma^2 + h^2\sigma_d^2$ with parameter uncertainty); 1 vs 5 threads | 4 standard errors; bitwise |
| Individual lifetimes: average present value of 200,000 lives vs the exact mean and variance of $\sum_{t \le K} v(t)$ | 4 standard errors |
| One-year recalibration: with a consistent drift and no noise the revalued annuity equals the best estimate; $d' = (k_{T+1} - k_{\text{first}})/(n+1)$; thread invariance | $10^{-12}$; bitwise |
| Portfolios: grouping by age; a one-member portfolio reproduces the single-life simulation stream for stream; values add up across ages; one-year portfolio value equals the amount-weighted sum of the members' values; a separate lifetime stream keeps the trend scenario and gives lifetimes uncorrelated with the scenario's own stream; fewer than one scenario or a short discount vector rejected | exact; $10^{-9}$ to $10^{-12}$; $|\rho| < 0.03$; exact |

**Mutation check.** Eight deliberate bugs were injected into the engine, and each makes the suite fail:

- $q = m$ instead of $1 - e^{-m}$;
- the cohort's age off by one year;
- the complete instead of the curtate life expectancy;
- the drift re-estimated with $n$ instead of $n + 1$ increments;
- the parameter uncertainty of the drift halved;
- the portfolio totals overwritten instead of summed;
- the one-year view discounted one year too early;
- individual lifetimes shortened by one year.

**Sanitizers.** The same tests pass under two sanitizer builds:

- AddressSanitizer with UndefinedBehaviorSanitizer (`-DLR_SANITIZE=address,undefined`): out-of-bounds access, use after free, signed overflow and similar;
- ThreadSanitizer (`-DLR_SANITIZE=thread`): data races in the parallel scenario loops.

CI runs both, plus GCC and Clang builds with warnings as errors.

## Continuous integration

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs on every pull request and on pushes to `main`. It needs no credentials, downloads no data and never reads a real membership file. Its jobs:

1. **Python:** lint (ruff, pyflakes rules) and the test suite, on Python 3.11 with the locked versions and on 3.10 and 3.12 with the newest allowed versions.
2. **C++:** the standalone tests with GCC and Clang (warnings as errors), and with GCC under ASan/UBSan and under TSan; `ctest --no-tests=error` fails if no test is registered.
3. **Notebooks:** every Python notebook executed offline on synthetic data (the job fails if the expected number of notebooks is not found), and the buy-in notebook once more with a membership CSV written from the synthetic generator, so that the `BUYIN_MEMBERSHIP_CSV` branch runs. The job checks that the data cache stays empty, i.e. that synthetic mode does not touch the network.

## References

- Börger, M. (2010). Deterministic shock vs. stochastic value-at-risk: an analysis of the Solvency II standard model approach to longevity risk. *Blätter der DGVFM*, 31(2), 225-259.
- Blackman, D. and Vigna, S. (2021). Scrambled linear pseudorandom number generators. *ACM Transactions on Mathematical Software*, 47(4). Public-domain reference code: https://prng.di.unimi.it/
- Brouhns, N., Denuit, M. and Vermunt, J. K. (2002). A Poisson log-bilinear regression approach to the construction of projected lifetables. *Insurance: Mathematics and Economics*, 31(3), 373-393.
- Coughlan, G. D., Khalaf-Allah, M., Ye, Y., Kumar, S., Cairns, A. J. G., Blake, D. and Dowd, K. (2011). Longevity hedging 101: a framework for longevity basis risk analysis and hedge effectiveness. *North American Actuarial Journal*, 15(2), 150-176.
- Commission Delegated Regulation (EU) 2015/35 of 10 October 2014 supplementing Directive 2009/138/EC (Solvency II), Articles 37, 39, 138, 140, 166, 167 and 204; Directive 2009/138/EC, Annex IV.
- Commission Delegated Regulation (EU) 2026/269 of 29 October 2025 (C(2025) 7206) amending Delegated Regulation (EU) 2015/35, Article 1(8) and (9): risk margin and cost-of-capital rate, applicable from 30 January 2027.
- EIOPA. Technical documentation of the methodology to derive EIOPA's risk-free interest rate term structures. https://www.eiopa.europa.eu/tools-and-data/risk-free-interest-rate-term-structures_en
- Enchev, V., Kleinow, T. and Cairns, A. J. G. (2017). Multi-population mortality models: fitting, forecasting and comparisons. *Scandinavian Actuarial Journal*, 2017(4), 319-342.
- Lee, R. D. and Carter, L. R. (1992). Modeling and forecasting U.S. mortality. *Journal of the American Statistical Association*, 87(419), 659-671.
- Li, J. S.-H. and Hardy, M. R. (2011). Measuring basis risk in longevity hedges. *North American Actuarial Journal*, 15(2), 177-200.
- Li, N. and Lee, R. (2005). Coherent mortality forecasts for a group of populations: an extension of the Lee-Carter method. *Demography*, 42(3), 575-594.
- Richards, S. J., Currie, I. D. and Ritchie, G. P. (2014). A value-at-risk framework for longevity trend risk. *British Actuarial Journal*, 19(1), 116-139.
- Smith, A. and Wilson, T. (2001). Fitting yield curves with long term constraints. Research notes, Bacon and Woodrow.
- Svensson, L. E. O. (1994). Estimating and interpreting forward interest rates: Sweden 1992-1994. NBER Working Paper 4871.
- Eurostat, population and mortality statistics: https://ec.europa.eu/eurostat/data/database. European Central Bank, euro area yield curves: https://www.ecb.europa.eu/stats/
