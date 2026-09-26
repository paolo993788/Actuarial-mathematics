"""Figures shown in the README, computed with the same data, model and seeds as the notebooks.

Run from the repository root (official data are downloaded and cached on first use):

    python -m longevity_risk.readme_figures              # writes docs/figures/*-light.png and *-dark.png
    python -m longevity_risk.readme_figures --synthetic  # offline check on simulated mortality

Each figure is saved in a light and a dark variant; the README selects one with <picture>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import curves, data, lee_carter as lc, life_table as lt, multipopulation as mp, simulation as sim, solvency, synthetic
from .figstyle import band, end_label, header, label_offsets, new_figure, render

# Settings shared with the notebooks.
COUNTRY, AGES, FIRST_YEAR, PANDEMIC_YEARS = "IT", range(50, 100), 1975, (2020, 2021, 2022)
UFR, LLP, SEED, N_SCENARIOS, PROJECTION_END = 0.033, 20, 20240101, 10_000, 2050
CAPITAL_AGES = (55, 65, 75, 85)
# Settings shared with the multi-population notebook.
MP_AGES, MP_FIRST_YEAR, MP_INDICES = range(50, 90), 1998, ["IT", "EU peers", "DE", "FR", "ES", "NL"]
MP_SHOWN = ["IT", "EU peers", "FR", "DE", "NL"]
MP_AGE0, MP_HORIZON, MP_OMEGA, MP_SCENARIOS, MP_SEEDS = 65, 10, 110, 20_000, (20240101, 20240102)


def load(official=True, n_scenarios=N_SCENARIOS) -> dict:
    D, E = {}, {}
    for sex in ("M", "F"):
        if official:
            D[sex], E[sex] = data.load_deaths_exposures(COUNTRY, sex, AGES)
        else:
            D[sex], E[sex], _ = synthetic.deaths_and_exposures(AGES, range(1975, 2024), seed=7 if sex == "M" else 8,
                                                               drift=-1.2 if sex == "M" else -1.0,
                                                               log_level=0.0 if sex == "M" else -0.4)
        keep = [y for y in D[sex].columns if y >= FIRST_YEAR]
        D[sex], E[sex] = D[sex].loc[:, keep], E[sex].loc[:, keep]
    out = {"rates": {s: D[s] / E[s] for s in D},
           "source": ("Source: Eurostat demo_magec and demo_pjan (Italy)" if official
                      else "Simulated mortality from a known Lee-Carter model, not official statistics")}

    i65 = 65 - min(AGES)
    models, e65 = {}, {}
    for sex in ("M", "F"):
        years = list(D[sex].columns)
        fit = lc.fit_poisson(D[sex], E[sex], fit_years=[y for y in years if y not in PANDEMIC_YEARS])
        model = sim.ProjectionModel.from_fit(fit, omega=120)
        models[sex] = model
        observed = pd.Series({y: lt.period_life_expectancy(out["rates"][sex].loc[65:99, y].to_numpy()) for y in years})
        H = model.horizon(65)
        k = sim.simulate_annuity(model, 65, np.ones(H), n_scenarios, seed=SEED, keep_paths=True)["k_paths"]
        T = model.valuation_year
        steps = PROJECTION_END - T
        ax65, bx65 = model.ax[i65:i65 + 35], model.bx[i65:i65 + 35]
        sims = np.array([lt.period_life_expectancy(np.exp(ax65[None, :] + bx65[None, :] * k[:, h - 1][:, None]))
                         for h in range(1, steps + 1)])
        lo, med, hi = np.percentile(sims, [2.5, 50, 97.5], axis=1)
        e65[sex] = {"observed": observed,
                    "projection": pd.DataFrame({"lo": lo, "median": med, "hi": hi}, index=np.arange(T + 1, PROJECTION_END + 1))}
    out["e65"] = e65

    model = models["M"]
    T = model.valuation_year
    if official:
        svensson = data.load_ecb_svensson_parameters(f"{T}-12-01", f"{T}-12-31")
    else:
        svensson = synthetic.svensson_parameters(f"{T}-12-31")
    sw = curves.smith_wilson_from_svensson(curves.SvenssonCurve.from_series(svensson.iloc[-1]), UFR, LLP)
    discount = sw.discount(np.arange(0, model.omega - min(AGES) + 2))
    discount[0] = 1.0
    rows = {}
    for age in CAPITAL_AGES:
        be = model.best_estimate(age, discount)
        sf = solvency.longevity_scr(model.central_q(age), discount)
        one_year = sim.one_year_recalibration(model, age, discount, n_scenarios * 10, seed=SEED)["value"]
        runoff = sim.simulate_annuity(model, age, discount, n_scenarios * 10, seed=SEED)["pv_systematic"]
        rows[age] = {"Standard formula (20% shock)": 100 * sf["scr_ratio"],
                     "Run-off VaR 99.5%": 100 * (np.quantile(runoff, 0.995) / be - 1),
                     "One-year VaR 99.5%": 100 * (np.quantile(one_year, 0.995) / be - 1)}
    out["capital"] = pd.DataFrame(rows).T
    out["valuation_year"] = T
    return out


def mortality_curves(t, d):
    fig, axes = new_figure(t, ncols=2, sharey=True)
    fig.subplots_adjust(right=0.97)
    for ax, sex, title in zip(axes, ("M", "F"), ("Men", "Women")):
        m = d["rates"][sex]
        years = list(m.columns)
        picks = sorted({years[0], years[len(years) // 4], years[len(years) // 2], years[3 * len(years) // 4], years[-1]})
        for color, y in zip(t["ordinal"], picks):
            ax.semilogy(m.index, m[y], color=color, lw=1.6, label=str(y))
        ax.set_title(title, loc="left", fontsize=10, color=t["ink2"])
        ax.set_xlabel("age")
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:g}%")
    axes[0].legend(loc="upper left", title="year", title_fontsize=9, labelcolor=t["ink2"])
    axes[0].get_legend().get_title().set_color(t["ink2"])
    first, last = min(d["rates"]["M"].columns), max(d["rates"]["M"].columns)
    header(fig, t, f"Death rates have fallen at almost every age since {first}",
           f"Central death rate by age (log scale) in selected years from {first} to {last}; {t['ordinal_note']}", d["source"])
    fig.subplots_adjust(top=0.78, bottom=0.17)
    return fig


def life_expectancy(t, d):
    fig, ax = new_figure(t)
    fig.subplots_adjust(right=0.80)
    names = {"M": "Men", "F": "Women"}
    ends = []
    for k, sex in enumerate(("M", "F")):
        c = t["series"][k]
        obs, proj = d["e65"][sex]["observed"], d["e65"][sex]["projection"]
        ax.plot(obs.index, obs.to_numpy(), color=c, lw=1.8, label=f"{names[sex]}, observed")
        x = np.r_[obs.index[-1], proj.index]
        band(ax, x, np.r_[obs.iloc[-1], proj["lo"]], np.r_[obs.iloc[-1], proj["hi"]], c, t["band"][0] + 0.04)
        ax.plot(x, np.r_[obs.iloc[-1], proj["median"]], color=c, lw=1.8, ls=(0, (3, 2)), label=f"{names[sex]}, projection")
        ends.append((sex, proj.index[-1], proj["median"].iloc[-1], c))
    offsets = label_offsets(ax, [e[2] for e in ends])
    for (sex, x_end, y_end, c), dy in zip(ends, offsets):
        end_label(ax, t, x_end, y_end, f"{names[sex]}: {y_end:.1f} years in {x_end}", c, dy=dy)
    ax.axvline(d["valuation_year"] + 0.5, color=t["axis"], lw=1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
    ax.set_ylabel("years")
    ax.legend(loc="upper left", ncol=2)
    gains = [d["e65"][s]["projection"]["median"].iloc[-1] - d["e65"][s]["observed"].iloc[-1] for s in ("M", "F")]
    header(fig, t, f"Life expectancy at 65 is projected to rise by {min(gains):.1f}-{max(gains):.1f} years by {PROJECTION_END}",
           "Period life expectancy at 65: observed, then Lee-Carter median and 95% interval (pandemic years excluded from the fit)",
           d["source"] + "; 10,000 simulated paths, seed 20240101")
    return fig


def longevity_capital(t, d):
    cap = d["capital"]
    fig, ax = new_figure(t)
    fig.subplots_adjust(right=0.97)
    x = np.arange(len(cap))
    width = 0.22
    for k, col in enumerate(cap.columns):
        bars = ax.bar(x + (k - 1) * (width + 0.03), cap[col].to_numpy(), width, color=t["series"][k], label=col)
        for b, v in zip(bars, cap[col]):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=t["ink2"])
    ax.set_xticks(x, [f"age {a}" for a in cap.index])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.axhline(0, color=t["axis"], lw=1)
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, cap.to_numpy().max() * 1.25)
    header(fig, t, "The Solvency II longevity shock is prudent for older annuitants",
           f"Longevity capital for a male life annuity, % of the best estimate, valuation at 31 December {d['valuation_year']}",
           d["source"] + "; ECB AAA curve with Smith-Wilson extrapolation; 100,000 scenarios")
    return fig


def load_multipopulation(official=True) -> dict:
    countries = ("IT",) + mp.PEERS
    D, E = mp.load_panel(countries, "M", MP_AGES, MP_FIRST_YEAR, official=official)
    D["EU peers"], E["EU peers"] = mp.aggregate(D, E, mp.PEERS)
    fit = mp.fit_li_lee(D, E, countries, [y for y in D["IT"].columns if y not in PANDEMIC_YEARS])
    sv = (data.load_ecb_svensson_parameters("2024-12-01", "2024-12-31") if official else synthetic.svensson_parameters())
    v = curves.smith_wilson_from_svensson(curves.SvenssonCurve.from_series(sv.iloc[-1]), UFR, LLP).discount(
        np.arange(0, MP_OMEGA - MP_AGE0 + 2.0))
    table = {}
    for label, dyn in (("Coherent: national deviations fade (Li-Lee)", mp.fit_dynamics(fit)),
                       ("Permanent national deviations", mp.fit_dynamics(fit, random_walk_deviations=True))):
        cal, test = (mp.simulate_states(dyn, MP_INDICES, MP_OMEGA - MP_AGE0 + 1, MP_SCENARIOS, seed=s) for s in MP_SEEDS)
        Lc = mp.liability_values(fit, dyn, "IT", cal, MP_AGE0, MP_HORIZON, v)
        Lt = mp.liability_values(fit, dyn, "IT", test, MP_AGE0, MP_HORIZON, v)
        table[label] = {idx: 100 * mp.hedge_effectiveness(Lc, mp.index_death_rates(fit, idx, cal, [75], MP_HORIZON), Lt,
                                                          mp.index_death_rates(fit, idx, test, [75], MP_HORIZON))["VaR reduction"]
                        for idx in MP_INDICES}
    names = {"IT": "Italy", "EU peers": "12 European peers", "FR": "France", "DE": "Germany", "NL": "Netherlands"}
    frame = pd.DataFrame(table).loc[MP_SHOWN].rename(index=names)
    source = ("Source: Eurostat demo_magec and demo_pjan (13 countries, men 50-89, 1998-2024); ECB AAA curve"
              if official else "Simulated mortality, not official statistics")
    return {"var_reduction": frame, "source": source}


def index_hedge(t, d):
    table = d["var_reduction"]
    fig, ax = new_figure(t, height=4.8)
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.14, top=0.72)
    y = np.arange(len(table))[::-1]
    h = 0.36
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for k, col in enumerate(table.columns):
        vals = table[col].to_numpy()
        ax.barh(y + (0.5 - k) * (h + 0.04), vals, h, color=t["series"][k], label=col)
        for yy, val in zip(y + (0.5 - k) * (h + 0.04), vals):
            ax.annotate(f"{val:.0f}%", (val, yy), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                        color=t["ink2"])
    ax.set_yticks(y, [f"{i} index" for i in table.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the 99.5% value at risk removed, %")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.4, 1.0), fontsize=8.5)
    header(fig, t, "Basis risk from a European index is small if countries stay coherent",
           "Longevity risk of an annuity to Italian men aged 65: share of the 99.5% value at risk removed by a 10-year\n"
           "q-forward at age 75 on each population index, large fund, Li-Lee model with two assumptions on national deviations",
           d["source"])
    fig.texts[-1].set_y(0.015)
    return fig


FIGURES = {"mortality_rates": (mortality_curves, "national"), "life_expectancy_65": (life_expectancy, "national"),
           "longevity_capital": (longevity_capital, "national"), "index_hedge": (index_hedge, "multipopulation")}
LOADERS = {"national": load, "multipopulation": load_multipopulation}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw the README figures (light and dark variants).")
    parser.add_argument("--out", default=str(data.repository_root() / "docs" / "figures"))
    parser.add_argument("--synthetic", action="store_true", help="use simulated mortality and an illustrative curve (offline)")
    parser.add_argument("--only", nargs="*", choices=list(FIGURES), help="draw only these figures")
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    names = args.only or list(FIGURES)
    inputs = {key: LOADERS[key](official=not args.synthetic) for key in {FIGURES[n][1] for n in names}}
    for name in names:
        builder, key = FIGURES[name]
        for path in render(builder, name, Path(args.out), inputs[key]):
            print(path)


if __name__ == "__main__":
    main()
