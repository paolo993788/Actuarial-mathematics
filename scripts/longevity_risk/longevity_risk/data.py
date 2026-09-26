"""Download and cache official data from Eurostat and the European Central Bank.

Sources (free reuse with acknowledgement of the source):

* Eurostat, deaths by age and sex (dataset ``demo_magec``) and population on
  1 January by age and sex (``demo_pjan``), through the Eurostat
  dissemination API in JSON-stat format:
  https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/<dataset>
  Eurostat copyright notice: https://ec.europa.eu/eurostat/about-us/policies/copyright
* ECB Data Portal, euro area yield curve (AAA-rated central government
  bonds), Svensson parameters, series
  YC.B.U2.EUR.4F.G_N_A.SV_C_YM.{BETA0,BETA1,BETA2,BETA3,TAU1,TAU2}.

Central exposure to risk is approximated by the average of the populations
on 1 January of years t and t + 1, the approach Eurostat uses for its
age-specific death rates. Downloads are cached in ``data/raw/`` (ignored by
Git); set ``LONGEVITY_RISK_DATA_DIR`` to use another folder. Command line:

    python -m longevity_risk.data --geo IT --sex M --yield-curve 2024-12-01 2024-12-31
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?{query}"
ECB_YC_URL = ("https://data-api.ecb.europa.eu/service/data/YC/"
              "B.U2.EUR.4F.G_N_A.SV_C_YM.BETA0+BETA1+BETA2+BETA3+TAU1+TAU2"
              "?format=csvdata&startPeriod={start}&endPeriod={end}")
USER_AGENT = "longevity-risk/0.1 (research scripts; https://github.com/paolo993788/Actuarial-mathematics)"
SVENSSON_COLUMNS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


def repository_root() -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / "scripts" / "longevity_risk").is_dir():
            return base
    return Path(__file__).resolve().parents[3]


def cache_dir(source: str) -> Path:
    env = os.environ.get("LONGEVITY_RISK_DATA_DIR")
    path = (Path(env) if env else repository_root() / "data" / "raw") / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, destination: Path, timeout: float = 120.0) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(destination)
    return destination


# --------------------------------------------------------------------------- Eurostat


def parse_jsonstat(obj: dict) -> pd.DataFrame:
    """Convert a JSON-stat 2.0 dataset into a long table (one column per dimension + value)."""
    dims = obj["id"]
    sizes = obj["size"]
    codes = []
    for dim in dims:
        index = obj["dimension"][dim]["category"]["index"]
        if isinstance(index, list):
            ordered = index
        else:
            ordered = [None] * len(index)
            for code, pos in index.items():
                ordered[pos] = code
        codes.append(ordered)
    values = obj["value"]
    if isinstance(values, dict):  # sparse form used by Eurostat: {"flat index": value}
        pairs = [(int(k), v) for k, v in values.items() if v is not None]
    else:
        pairs = [(i, v) for i, v in enumerate(values) if v is not None]
    flat = np.array([i for i, _ in pairs], dtype=np.int64)
    obs = np.array([v for _, v in pairs], dtype=float)
    strides = np.cumprod([1] + sizes[::-1])[:-1][::-1]
    table = {}
    for dim, stride, size, labels in zip(dims, strides, sizes, codes):
        table[dim] = np.asarray(labels, dtype=object)[(flat // stride) % size]
    table["value"] = obs
    return pd.DataFrame(table)


def age_to_int(code: str):
    """Eurostat age code to integer age: 'Y_LT1' -> 0, 'Y42' -> 42; aggregates and open groups -> None."""
    if code == "Y_LT1":
        return 0
    match = re.fullmatch(r"Y(\d+)", code)
    return int(match.group(1)) if match else None


def load_eurostat(dataset: str, refresh=False, **filters) -> pd.DataFrame:
    """Download (or read from cache) a Eurostat dataset filtered by dimension codes."""
    query = urllib.parse.urlencode({"format": "JSON", "lang": "EN", **filters})
    tag = "_".join(f"{k}-{v}" for k, v in sorted(filters.items()))
    path = cache_dir("eurostat") / f"{dataset}_{tag}.json"
    if refresh or not path.exists():
        download(EUROSTAT_URL.format(dataset=dataset, query=query), path)
    frame = parse_jsonstat(json.loads(path.read_text(encoding="utf-8")))
    frame.attrs["source"] = f"Eurostat, dataset {dataset} ({filters})"
    return frame


def _age_year_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.assign(age=frame["age"].map(age_to_int), year=frame["time"].astype(int)).dropna(subset=["age"])
    return frame.pivot_table(index="age", columns="year", values="value", aggfunc="sum").sort_index().rename(index=int)


def deaths_and_exposures(deaths_long: pd.DataFrame, population_long: pd.DataFrame, ages, years=None):
    """Age x year tables of deaths and central exposures E(x, t) = (P(x, t) + P(x, t + 1)) / 2."""
    D = _age_year_table(deaths_long)
    P = _age_year_table(population_long)
    common = [t for t in D.columns if t in P.columns and t + 1 in P.columns]
    if years is not None:
        common = [t for t in common if t in set(years)]
    ages = list(ages)
    D = D.reindex(index=ages, columns=common)
    E = pd.DataFrame(0.5 * (P.reindex(index=ages, columns=common).to_numpy()
                            + P.reindex(index=ages, columns=[t + 1 for t in common]).to_numpy()),
                     index=ages, columns=common)
    complete = D.notna().all() & E.notna().all() & (E > 0).all()
    D, E = D.loc[:, complete], E.loc[:, complete]
    D.index.name = E.index.name = "age"
    D.columns.name = E.columns.name = "year"
    return D, E


def load_deaths_exposures(geo="IT", sex="M", ages=range(50, 100), years=None, refresh=False):
    """Deaths and exposures from Eurostat demo_magec and demo_pjan for one country and sex."""
    deaths = load_eurostat("demo_magec", refresh=refresh, geo=geo, sex=sex, unit="NR")
    population = load_eurostat("demo_pjan", refresh=refresh, geo=geo, sex=sex, unit="NR")
    D, E = deaths_and_exposures(deaths, population, ages, years)
    D.attrs["source"] = E.attrs["source"] = f"Eurostat demo_magec and demo_pjan, geo={geo}, sex={sex}"
    return D, E


# --------------------------------------------------------------------------- ECB


def parse_ecb_yield_curve_csv(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    parameter = raw["KEY"].str.split(".").str[-1] if "KEY" in raw.columns else raw["DATA_TYPE_FM"]
    table = (raw.assign(parameter=parameter, date=pd.to_datetime(raw["TIME_PERIOD"]))
                .pivot_table(index="date", columns="parameter", values="OBS_VALUE", aggfunc="last"))
    missing = [c for c in SVENSSON_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"unexpected ECB response, missing parameters: {missing}")
    return table[SVENSSON_COLUMNS].sort_index().astype(float)


def load_ecb_svensson_parameters(start: str, end: str, refresh=False) -> pd.DataFrame:
    path = cache_dir("ecb") / f"yc_svensson_{start}_{end}.csv"
    if refresh or not path.exists():
        download(ECB_YC_URL.format(start=start, end=end), path)
    table = parse_ecb_yield_curve_csv(path.read_text(encoding="utf-8"))
    table.attrs["source"] = "European Central Bank, euro area yield curves (AAA), Svensson parameters"
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download official Eurostat and ECB data into the local cache.")
    parser.add_argument("--geo", default="IT", help="Eurostat country code, e.g. IT, DE, FR, ES")
    parser.add_argument("--sex", nargs="+", default=["M", "F"], help="M, F and/or T")
    parser.add_argument("--yield-curve", nargs=2, metavar=("START", "END"), help="ECB curve period, e.g. 2024-12-01 2024-12-31")
    args = parser.parse_args(argv)
    for sex in args.sex:
        D, _ = load_deaths_exposures(args.geo, sex, refresh=True)
        print(f"{args.geo} {sex}: deaths and exposures for {D.columns.min()}-{D.columns.max()}, cached in {cache_dir('eurostat')}")
    if args.yield_curve:
        curve = load_ecb_svensson_parameters(*args.yield_curve, refresh=True)
        print(f"ECB Svensson parameters: {len(curve)} days, cached in {cache_dir('ecb')}")


if __name__ == "__main__":
    main()
