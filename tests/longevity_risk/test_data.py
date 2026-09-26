"""Parsers for Eurostat JSON-stat and ECB CSV formats."""

import json

import numpy as np
import pytest

from longevity_risk import data

# Minimal JSON-stat 2.0 document in the sparse layout returned by Eurostat.
JSONSTAT = json.loads("""
{"version": "2.0", "class": "dataset",
 "id": ["freq", "unit", "sex", "age", "geo", "time"],
 "size": [1, 1, 1, 4, 1, 3],
 "dimension": {
   "freq": {"category": {"index": {"A": 0}}},
   "unit": {"category": {"index": {"NR": 0}}},
   "sex": {"category": {"index": {"M": 0}}},
   "age": {"category": {"index": {"TOTAL": 0, "Y_LT1": 1, "Y65": 2, "Y_OPEN": 3}}},
   "geo": {"category": {"index": {"IT": 0}}},
   "time": {"category": {"index": {"2021": 0, "2022": 1, "2023": 2}}}},
 "value": {"0": 300000, "3": 1000, "4": 1100, "5": 1200, "6": 5000, "8": 5200, "9": 800}}
""")

POPULATION = {**JSONSTAT, "value": {"3": 200000, "4": 198000, "5": 196000, "6": 300000, "7": 302000, "8": 304000}}


def test_parse_jsonstat_long_format():
    frame = data.parse_jsonstat(JSONSTAT)
    assert list(frame.columns) == ["freq", "unit", "sex", "age", "geo", "time", "value"]
    row = frame[(frame["age"] == "Y65") & (frame["time"] == "2023")]
    assert row["value"].item() == 5200
    assert frame[(frame["age"] == "Y65") & (frame["time"] == "2022")].empty  # missing observation


def test_age_codes():
    assert [data.age_to_int(c) for c in ("Y_LT1", "Y1", "Y99", "Y_OPEN", "TOTAL", "UNK")] == [0, 1, 99, None, None, None]


def test_exposure_is_average_of_january_populations():
    D, E = data.deaths_and_exposures(data.parse_jsonstat(JSONSTAT), data.parse_jsonstat(POPULATION), ages=[0, 65])
    # 2023 has no population on 1 January 2024; 2022 lacks Y65 deaths.
    assert list(D.columns) == [2021]
    assert E.loc[65, 2021] == pytest.approx((300000 + 302000) / 2)
    assert E.loc[0, 2021] == pytest.approx((200000 + 198000) / 2)
    assert D.loc[65, 2021] == 5000


def test_parse_ecb_yield_curve():
    text = "KEY,TIME_PERIOD,OBS_VALUE\n" + "\n".join(
        f"YC.B.U2.EUR.4F.G_N_A.SV_C_YM.{p},2024-12-31,{v}"
        for p, v in zip(data.SVENSSON_COLUMNS, [2.0, 1.0, -1.0, 2.0, 1.5, 10.0]))
    table = data.parse_ecb_yield_curve_csv(text)
    np.testing.assert_allclose(table.iloc[0].to_numpy(), [2.0, 1.0, -1.0, 2.0, 1.5, 10.0])
