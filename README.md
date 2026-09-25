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

No projects have been published yet. Each new script or notebook will be listed here with a short description.

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
