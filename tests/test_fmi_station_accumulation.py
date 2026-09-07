# -*- coding: utf-8 -*-
import pandas as pd
import pytest
from unittest.mock import patch

from src.fetchers.FMI import FMIDataFetcher


def _fetcher(tmp_path):
    """FMIDataFetcher writing into tmp_path instead of the real data folder."""
    with patch("src.fetchers.FMI.FOLDER_NAME", str(tmp_path)):
        return FMIDataFetcher()


def _stations(*triples):
    """Build a station metadata frame from (name, fmisid, lat, lon) tuples."""
    return pd.DataFrame(
        list(triples),
        columns=["station_name", "fmisid", "latitude", "longitude"],
    )


def test_interval_unions_stations_across_days(tmp_path):
    """A station appearing only on the second day must survive."""
    fetcher = _fetcher(tmp_path)

    day_one = _stations(("Alpha", 1, 60.0, 24.0))
    day_two = _stations(("Beta", 2, 61.0, 25.0))
    data = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "station_name": ["Alpha"]})

    with patch.object(FMIDataFetcher, "fetch_fmi_data",
                      side_effect=[(data, day_one), (data, day_two)]):
        result = fetcher.fetch_fmi_by_interval("18,55,35,75", "2024-01-01", "2024-01-02")

    assert set(result["fmisid"]) == {1, 2}


def test_interval_collects_metadata_when_day_has_no_observations(tmp_path):
    """Station metadata must not be gated on that day returning observation rows."""
    fetcher = _fetcher(tmp_path)

    stations = _stations(("Gamma", 3, 62.0, 26.0))

    with patch.object(FMIDataFetcher, "fetch_fmi_data",
                      return_value=(pd.DataFrame(), stations)):
        result = fetcher.fetch_fmi_by_interval("18,55,35,75", "2024-01-01", "2024-01-01")

    assert set(result["fmisid"]) == {3}


def test_interval_deduplicates_repeated_stations(tmp_path):
    """The same station on both days yields one row, not two."""
    fetcher = _fetcher(tmp_path)

    stations = _stations(("Alpha", 1, 60.0, 24.0))
    data = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "station_name": ["Alpha"]})

    with patch.object(FMIDataFetcher, "fetch_fmi_data",
                      side_effect=[(data, stations), (data, stations)]):
        result = fetcher.fetch_fmi_by_interval("18,55,35,75", "2024-01-01", "2024-01-02")

    assert len(result) == 1
