# Actuarial Mathematics

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![pybind11](https://img.shields.io/badge/bindings-pybind11-5C6BC0)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Data](https://img.shields.io/badge/data-Eurostat%20%7C%20ECB-2E7D32)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Life insurance and pension mathematics with stochastic mortality: Lee-Carter projections on official Eurostat data, a C++ Monte Carlo engine for annuity portfolios, and Solvency II valuation and capital.**

The repository covers the work of a life or pension actuary from data to decision: building a projected mortality basis, valuing annuities with a market-consistent discount curve, measuring longevity risk with an internal-model view and with the Solvency II standard formula, and pricing a bulk annuity (buy-in) for a pension fund. Simulation-heavy parts run in C++; estimation, valuation and reporting are in Python.

## Highlights

| Area | What is done |
| --- | --- |
| Pension buy-in pricing (case study) | Quote for an Italian pension fund with 1,500 pensioners: cohort mortality basis, Smith-Wilson curve, best estimate and duration, standard-formula SCR (longevity, expense, interest rate, operational), cost-of-capital risk margin, premium with a shareholder hurdle rate, internal-model cross-check, sensitivities and a board summary |
| Mortality projection | Poisson Lee-Carter by sex on Eurostat deaths and population, pandemic years excluded, deviance diagnostics, out-of-sample backtest of forecast intervals, period versus cohort life expectancy |
| Longevity capital | Run-off and one-year value at risk (re-estimation of the trend, Richards et al. 2014) against the 20% Solvency II longevity shock, by age and interest-rate level; pooling of idiosyncratic risk |
| Discounting | ECB AAA yield curve extrapolated with the Smith-Wilson method and EIOPA convergence criterion |
| Engineering | C++17 engine with portable, thread-independent random streams: 100,000 portfolio scenarios with 1,500 simulated lifetimes each in about 3 seconds on four cores; 38 automated tests against closed forms, exact moments and NumPy references |

## Catalogue

| Item | Question answered | Data |
| --- | --- | --- |
| [`scripts/longevity_risk`](scripts/longevity_risk/README.md) (library) | Reusable Lee-Carter estimation, C++ simulation of annuities and portfolios, Smith-Wilson curve, Solvency II standard formula and risk margin | Eurostat, ECB loaders |
| [Pension buy-in pricing](notebooks/case_studies/pension_buy_in_pricing.ipynb) | What premium should an insurer quote to take over a pension fund's liabilities, and how much capital does it need? | Eurostat, ECB; illustrative membership |
| [Lee-Carter mortality projection](notebooks/mortality_projection/lee_carter_mortality_projection.ipynb) | How fast is Italian mortality improving, how reliable are the forecasts, and what is the cohort life expectancy at 65? | Eurostat `demo_magec`, `demo_pjan` |
| [Longevity risk and Solvency II](notebooks/longevity_risk/annuity_longevity_risk_solvency.ipynb) | Is the standard-formula longevity shock prudent compared with stochastic value-at-risk measures? | Eurostat, ECB |

Each notebook states its assumptions and conventions, fixes its random seeds, reports Monte Carlo errors and ends with conclusions and limitations. Figures and tables are written to `outputs/`.

## Architecture

```text
notebooks/  ──►  longevity_risk (Python)                    ──►  longevity_risk._core (C++17, pybind11)
                 Eurostat and ECB loaders                        Lee-Carter scenario generator (drift uncertainty)
                 Poisson / SVD Lee-Carter, random walk           cohort survival and life annuities
                 Smith-Wilson curve                              portfolios with individual lifetimes
                 Solvency II: shocks, SCR, risk margin           one-year recalibration (trend risk)
                 NumPy reference implementations
```

## Getting started

Requirements: Python 3.10 or later and a C++17 compiler (Visual Studio Build Tools on Windows, Xcode Command Line Tools on macOS, GCC or Clang on Linux). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/longevity_risk/requirements.txt
python -m pip install -e scripts/longevity_risk
python -m pytest tests/longevity_risk
```

Open a notebook in Visual Studio Code (extensions *Python*, *Jupyter* and *C/C++*) and select the `.venv` environment as kernel. Official data are downloaded and cached on first use; set `LONGEVITY_RISK_DATA_MODE=synthetic` to work offline. The [project README](scripts/longevity_risk/README.md) documents methods, conventions and the validation table.

## Repository layout

```text
.
├── scripts/longevity_risk/   C++ engine (cpp/), Python package, build and dependency files
├── notebooks/                case_studies/, mortality_projection/, longevity_risk/
├── tests/longevity_risk/     validation suite (pytest)
├── docs/                     project README template and publishing workflow
├── data/                     download cache (ignored by Git) and small examples
└── outputs/                  generated figures and tables (ignored by Git)
```

## Scope and roadmap

Implemented: survival models and life tables, financial mathematics of annuities, stochastic mortality, Solvency II valuation and capital for annuities. Planned: premiums and reserves for term and endowment insurance with Thiele's differential equation, Cairns-Blake-Dowd and Renshaw-Haberman models, unit-linked guarantees with market-consistent valuation, IFRS 17 contractual service margin for annuity business.

## Conventions

- Each project documents its purpose, inputs, outputs and exact run command, following the [project README template](docs/script-template.md).
- Formulas, assumptions, units of measure, interest-rate conventions and sources are stated explicitly.
- Every new or modified calculation is checked against at least one known result and the relevant edge cases, with justified numerical tolerances.
- Stochastic examples use a fixed, documented random seed.
- Paths are relative to the repository root, and no credentials or confidential data (such as real membership files) are ever committed.

## Development workflow

Changes follow the [publishing workflow](docs/publishing.md). The [`CLAUDE.md`](CLAUDE.md) file provides project instructions for [Claude Code](https://claude.com/claude-code), so that AI-assisted contributions meet the same standards.

## Disclaimer

Research and educational code. Figures are indicative and depend on the stated assumptions; they are not an actuarial opinion or a price quote.

## License

Released under the [MIT License](LICENSE).
