# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from src.processors.DataLoader import DataLoader


SCHEDULED = "2018-01-15T08:00:00.000Z"


def _weather_frame(station_name, air_temp):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2018-01-15T08:00:00"]),
        "station_name": [station_name],
        "Air temperature": [air_temp],
    })


def _loader_with(top_row, weather):
    """A DataLoader wired with a candidate row and a live-station dictionary."""
    loader = DataLoader.__new__(DataLoader)
    loader.top5_ems_dict = {"HKI": pd.Series(top_row)}
    loader.ems_weather_dict = weather
    return loader


def _candidate_row(*pairs):
    """Build a candidate table row from (station_name, distance_km) pairs."""
    row = {}
    for rank in range(1, 11):
        name, dist = pairs[rank - 1] if rank <= len(pairs) else (None, None)
        row[f"ems_{rank}_station"] = name
        row[f"ems_{rank}_distance_km"] = dist
    return row


def test_primary_skips_dead_ranks_and_names_the_live_one():
    """Ranks 1-3 have no data this month; rank 4 does and must be recorded."""
    loader = _loader_with(
        _candidate_row(("Dead1", 1.0), ("Dead2", 2.0), ("Dead3", 3.0), ("Live4", 4.5)),
        {"Live4": _weather_frame("Live4", -7.5)},
    )

    result = loader._find_closest_weather(SCHEDULED, "HKI")

    assert result["closest_ems"] == "Live4"
    assert result["closest_ems_distance_km"] == 4.5
    assert result["Air temperature"] == -7.5


def test_rank_one_live_is_used_directly():
    loader = _loader_with(
        _candidate_row(("Live1", 0.5), ("Live2", 9.0)),
        {"Live1": _weather_frame("Live1", -3.0), "Live2": _weather_frame("Live2", -9.0)},
    )

    result = loader._find_closest_weather(SCHEDULED, "HKI")

    assert result["closest_ems"] == "Live1"
    assert result["closest_ems_distance_km"] == 0.5
    assert result["Air temperature"] == -3.0


def test_all_ranks_dead_yields_no_data_rather_than_silent_fill():
    loader = _loader_with(
        _candidate_row(("Dead1", 1.0), ("Dead2", 2.0)),
        {"Elsewhere": _weather_frame("Elsewhere", 99.0)},
    )

    result = loader._find_closest_weather(SCHEDULED, "HKI")

    assert result == {}


def test_train_station_with_no_candidates_yields_no_data():
    loader = _loader_with(_candidate_row(), {"Anywhere": _weather_frame("Anywhere", 5.0)})

    result = loader._find_closest_weather(SCHEDULED, "HKI")

    assert result == {}
