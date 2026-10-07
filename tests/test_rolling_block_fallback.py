"""Rolling windows of one parameter must come from one station.

When the matched station lacked a rolling value, _find_closest_weather used to
search for each missing column on its own, and the instant fallback also
overwrote rolling columns with whatever the donor happened to have. A 12h
cumulative could then come from one station and the 24h cumulative from
another, which breaks 12h <= 24h <= 72h.
"""

import pandas as pd

from config.const import FMI_INSTANT_PARAMS
from src.processors.DataLoader import DataLoader

SCHEDULED = "2018-01-15T08:00:00.000Z"
P = "Precipitation amount"
BLOCK = DataLoader._rolling_columns(P)           # 12/24/72h mean + cumulative
CUM = [f"{P} ({w}h cumulative)" for w in (12, 24, 72)]


def _frame(name, values):
    row = {"timestamp": pd.to_datetime("2018-01-15T08:00:00"), "station_name": name}
    row.update(values)
    return pd.DataFrame([row])


def _block(c12, c24, c72):
    values = {f"{P} (12h mean)": 0.1, f"{P} (24h mean)": 0.1, f"{P} (72h mean)": 0.1}
    values.update(dict(zip(CUM, (c12, c24, c72))))
    return values


def _loader(weather, ranks):
    loader = DataLoader.__new__(DataLoader)
    row = {}
    for i in range(1, 11):
        row[f"ems_{i}_station"] = ranks[i - 1] if i <= len(ranks) else None
        row[f"ems_{i}_distance_km"] = float(i) if i <= len(ranks) else None
    loader.top5_ems_dict = {"HKI": pd.Series(row)}
    loader.ems_weather_dict = weather
    return loader


def test_whole_block_comes_from_one_donor():
    primary = _block(15.4, None, 20.0)               # 24h cumulative missing
    partial = {CUM[1]: 11.0}                         # rank 2 only has the missing column
    full = _block(2.0, 3.0, 4.0)                     # rank 3 has the whole block
    loader = _loader({
        "A": _frame("A", {P: 0.0, **primary}),
        "B": _frame("B", partial),
        "C": _frame("C", full),
    }, ["A", "B", "C"])

    result = loader._find_closest_weather(SCHEDULED, "HKI")

    assert [result[c] for c in CUM] == [2.0, 3.0, 4.0]
    assert all(result[c] == full[c] for c in BLOCK)


def test_instant_fallback_does_not_overwrite_rolling_columns():
    primary = _block(1.0, 2.0, 3.0)                  # complete block, instant missing
    donor = {P: 0.4, **_block(9.0, 8.0, 7.0)}
    loader = _loader({"A": _frame("A", primary), "B": _frame("B", donor)}, ["A", "B"])

    result = loader._find_closest_weather(SCHEDULED, "HKI")

    assert result[P] == 0.4                          # instant borrowed
    assert [result[c] for c in CUM] == [1.0, 2.0, 3.0]


def test_no_complete_donor_keeps_primary_values_untouched():
    primary = _block(1.0, None, 3.0)
    partial = {CUM[1]: 11.0}
    loader = _loader({"A": _frame("A", {P: 0.0, **primary}), "B": _frame("B", partial)}, ["A", "B"])

    result = loader._find_closest_weather(SCHEDULED, "HKI")

    assert result[CUM[0]] == 1.0 and result[CUM[2]] == 3.0
    assert pd.isna(result[CUM[1]])


def test_order_violations_counts_bad_rows_and_respects_tolerance():
    df = pd.DataFrame({
        CUM[0]: [15.4, 5.005, 1.0],
        CUM[1]: [11.0, 5.0, 2.0],
        CUM[2]: [20.0, 20.0, 3.0],
    })

    found = DataLoader._rolling_order_violations(df)

    assert found == {f"{P} cumulative 12h vs 24h": 1}
