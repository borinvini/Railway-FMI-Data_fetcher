"""Regression tests for duplicated hour-mark weather rows.

fetch_fmi_data used to request one hour at a time with each request starting at
the exact instant the previous one ended. FMI treats starttime and endtime as
inclusive, so every hour mark was returned by two requests and stored twice.
preprocess_fmi_rolling_features then rolled over both twins, so every hourly
Precipitation amount was summed twice. These tests pin three things: the
fetcher no longer creates the overlap, the rolling step counts each hour once,
and a file that already carries stale rolling columns is repaired.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
from unittest.mock import patch

from config.const import (
    FMI_OBSERVATION_KEY,
    FMI_ROLLING_WINDOW_HOURS,
    get_fmi_rolling_column_names,
)

_PRECIP = "Precipitation amount"
_TEMP = "Air temperature"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fetcher():
    from src.fetchers.FMI import FMIDataFetcher
    return FMIDataFetcher()


def _make_dataloader(tmp_path, weather_file):
    from src.processors.DataLoader import DataLoader
    with patch.object(DataLoader, "_check_data_folder"):
        loader = DataLoader.__new__(DataLoader)
        loader.data_folder = str(tmp_path)
        loader.weather_folder = str(tmp_path)
        loader.weather_files = [str(weather_file)]
    return loader


def _fake_download(requests):
    """Stand in for FMI: return one observation per whole hour in [start, end],
    both ends inclusive, and record the requested window."""
    def fake(_query, args):
        params = dict(a.split("=", 1) for a in args)
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        start = datetime.strptime(params["starttime"], fmt)
        end = datetime.strptime(params["endtime"], fmt)
        requests.append((start, end))

        data = {}
        t = start.replace(minute=0, second=0)
        if t < start:
            t += timedelta(hours=1)
        while t <= end:
            data[t] = {"Helsinki": {_TEMP: {"value": 1.0}}}
            t += timedelta(hours=1)
        return SimpleNamespace(data=data, location_metadata={"Helsinki": {"fmisid": 1}})
    return fake


def _weather_frame(with_duplicates=True):
    """Two days of 10-minute data for one station.

    Air temperature is the row index, so the 12 h mean at a given row is known
    in closed form. Precipitation is 1.0 mm at every :00 and NaN otherwise.
    Optionally every hour mark except the very first appears twice, exactly as
    the overlapping fetch windows produced it.
    """
    stamps = pd.date_range("2024-01-01 00:00", "2024-01-03 00:00", freq="10min")
    df = pd.DataFrame({
        "timestamp": stamps,
        "station_name": "Helsinki",
        _TEMP: np.arange(len(stamps), dtype=float),
        _PRECIP: np.where(stamps.minute == 0, 1.0, np.nan),
    })
    if with_duplicates:
        twins = df[(df.timestamp.dt.minute == 0) & (df.timestamp > df.timestamp.min())]
        df = pd.concat([df, twins], ignore_index=True)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _write_weather(tmp_path, df):
    path = tmp_path / "fmi_weather_observations_2024_01.csv"
    df.to_csv(path, index=False)
    return path


def _rolling_names():
    names = []
    for param, skip_mm, skip_cum in ((_TEMP, False, True), (_PRECIP, True, False)):
        for wh in FMI_ROLLING_WINDOW_HOURS:
            names.extend(get_fmi_rolling_column_names(
                param, wh, skip_min_max=skip_mm, skip_cumulative=skip_cum).values())
    return names


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def test_fetch_windows_do_not_overlap_and_no_duplicate_rows():
    requests = []
    fetcher = _make_fetcher()

    with patch("src.fetchers.FMI.download_stored_query", _fake_download(requests)), \
         patch("src.fetchers.FMI.time.sleep"):
        df, _ = fetcher.fetch_fmi_data("bbox", datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 3))

    assert len(requests) == 3
    for (_, prev_end), (next_start, _) in zip(requests, requests[1:]):
        assert prev_end < next_start, "a request's endtime must precede the next starttime"

    assert not df.duplicated(FMI_OBSERVATION_KEY).any()
    assert sorted(df["timestamp"]) == [pd.Timestamp(f"2024-01-01 0{h}:00") for h in range(3)]


def test_drop_duplicate_observations_guard():
    fetcher = _make_fetcher()
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 01:00"] * 2 + ["2024-01-01 02:00"]),
        "station_name": "Helsinki",
        _TEMP: [1.0, 1.0, 2.0],
    })
    assert len(fetcher._drop_duplicate_observations(df)) == 2
    assert fetcher._drop_duplicate_observations(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Rolling step
# ---------------------------------------------------------------------------

def test_rolling_sums_count_each_hour_once(tmp_path):
    path = _write_weather(tmp_path, _weather_frame())
    loader = _make_dataloader(tmp_path, path)

    loader.preprocess_fmi_rolling_features()

    out = pd.read_csv(path, parse_dates=["timestamp"])
    assert not out.duplicated(FMI_OBSERVATION_KEY).any()

    row = out[out.timestamp == "2024-01-02 12:00"].iloc[0]
    # 24 hourly totals of 1.0 mm lie in (day 1 12:00, day 2 12:00]. Duplicates gave ~48.
    assert row[f"{_PRECIP} (24h cumulative)"] == 24.0
    assert row[f"{_PRECIP} (12h cumulative)"] == 12.0

    # 72 ten-minute samples in a 12 h window, values i-71 .. i, so mean = i - 35.5.
    # With the duplicated hour marks the window held extra samples and shifted this.
    i = int(row[_TEMP])
    assert row[f"{_TEMP} (12h mean)"] == i - 35.5


def test_stale_rolling_columns_are_recomputed(tmp_path):
    df = _weather_frame()
    for name in _rolling_names():
        df[name] = 999.0
    path = _write_weather(tmp_path, df)
    loader = _make_dataloader(tmp_path, path)

    loader.preprocess_fmi_rolling_features()

    out = pd.read_csv(path, parse_dates=["timestamp"])
    assert not out.columns.duplicated().any(), "rolling columns must not be listed twice"
    assert len(out.columns) == 4 + len(_rolling_names())
    assert not out.duplicated(FMI_OBSERVATION_KEY).any()

    row = out[out.timestamp == "2024-01-02 12:00"].iloc[0]
    assert row[f"{_PRECIP} (24h cumulative)"] == 24.0


def test_clean_file_with_rolling_columns_is_left_alone(tmp_path):
    """A file with no duplicates keeps the original skip behaviour."""
    df = _weather_frame(with_duplicates=False)
    for name in _rolling_names():
        df[name] = 7.0
    path = _write_weather(tmp_path, df)
    loader = _make_dataloader(tmp_path, path)

    loader.preprocess_fmi_rolling_features()

    out = pd.read_csv(path)
    assert (out[f"{_PRECIP} (24h cumulative)"] == 7.0).all()
