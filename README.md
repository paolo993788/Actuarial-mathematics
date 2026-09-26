# Actuarial Mathematics

Scripts and notebooks on the actuarial mathematics of life insurance, built to be reproducible, well documented and easy to verify.

## Scope

The repository is organized around the core topics of life insurance mathematics:

- **Survival models and life tables**: survival and death probabilities, life expectancy, construction and use of life tables.
- **Financial mathematics**: interest-rate conventions, discount factors and annuities certain.
- **Insurance benefits and life annuities**: whole life, term and endowment insurance; immediate and deferred annuities; commutation functions.
- **Premiums**: the equivalence principle, net and gross premiums, expense loadings.
- **Reserves**: prospective and retrospective reserves, recursive formulas and Thiele's differential equation.
- **Modern products**: unit-linked and universal life policies, embedded guarantees and market-consistent valuation.

## Repository layout

```text
.
├── scripts/     One folder per project, each with its own README
├── notebooks/   Jupyter notebooks, grouped by topic
├── docs/        Project README template and publishing workflow
├── data/        Small synthetic or publicly redistributable datasets (data/examples/)
├── outputs/     Generated results, not tracked by Git
└── tests/       Automated checks
```

Folders are created when their first content is added.

## Catalogue

| Item | Type | Description |
| --- | --- | --- |
| [`scripts/longevity_risk`](scripts/longevity_risk/README.md) | Python + C++ library | Poisson and SVD Lee-Carter estimation, C++17 Monte Carlo engine (pybind11) for cohort survival, life annuities, idiosyncratic risk and the one-year view of longevity trend risk, Smith-Wilson extrapolation, Solvency II longevity shock, Eurostat and ECB data loaders. |
| [`notebooks/mortality_projection/lee_carter_mortality_projection.ipynb`](notebooks/mortality_projection/lee_carter_mortality_projection.ipynb) | Notebook | Lee-Carter mortality projections for men and women on Eurostat data: diagnostics, pandemic years, out-of-sample backtest, period and cohort life expectancy. |
| [`notebooks/longevity_risk/annuity_longevity_risk_solvency.ipynb`](notebooks/longevity_risk/annuity_longevity_risk_solvency.ipynb) | Notebook | Life annuity best estimates with the ECB curve extrapolated by Smith-Wilson, run-off and one-year value at risk, pooling of idiosyncratic risk and comparison with the Solvency II standard formula. |

## Getting started

The notebooks run in Visual Studio Code (with the *Python*, *Jupyter* and *C/C++* extensions) or in Jupyter. The simulation engine is written in C++ and compiled into a Python extension, so a C++17 compiler is required (Visual Studio Build Tools on Windows, Xcode Command Line Tools on macOS, GCC or Clang on Linux). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/longevity_risk/requirements.txt
python -m pip install -e scripts/longevity_risk
python -m pytest tests/longevity_risk
```

Then open a notebook and select the `.venv` environment as kernel. The notebooks download official Eurostat and ECB data on first use; see the [project README](scripts/longevity_risk/README.md) for details and for the offline mode.

## Conventions

- Each project documents its purpose, inputs, outputs and exact run command, following the [project README template](docs/script-template.md).
- Formulas, assumptions, units of measure, interest-rate conventions and sources are stated explicitly.
- Every new or modified calculation is checked against at least one known result and the relevant edge cases, with justified numerical tolerances.
- Stochastic examples use a fixed, documented random seed.
- Paths are relative to the repository root, and no credentials or confidential data are ever committed.

## Development workflow

Changes follow the [publishing workflow](docs/publishing.md). The [`CLAUDE.md`](CLAUDE.md) file provides project instructions for [Claude Code](https://claude.com/claude-code), so that AI-assisted contributions meet the same standards.

## License

Released under the [MIT License](LICENSE).
