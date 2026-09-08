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


def _pool_row(name, fmisid, lat, lon, coord_source="ef", observed=True):
    return {
        "station_name": name, "fmisid": fmisid, "latitude": lat, "longitude": lon,
        "networks": "Automaattinen sääasema", "is_weather_station": True,
        "in_ef_registry": coord_source == "ef", "coord_source": coord_source,
        "observed_in_run": observed,
    }


def test_save_unions_with_the_existing_pool_file(tmp_path):
    """Two runs over disjoint months converge on one pool instead of overwriting."""
    from src.fetchers.FMI import STATION_COLUMNS
    fetcher = _fetcher(tmp_path)

    first = pd.DataFrame([_pool_row("Alpha", 1, 60.0, 24.0)])[STATION_COLUMNS]
    fetcher.save_station_metadata(first, "pool.csv")

    second = pd.DataFrame([_pool_row("Beta", 2, 61.0, 25.0)])[STATION_COLUMNS]
    fetcher.save_station_metadata(second, "pool.csv")

    result = pd.read_csv(tmp_path / "pool.csv")
    assert set(result["fmisid"]) == {1, 2}
    assert list(result.columns) == STATION_COLUMNS


def test_save_or_accumulates_observed_in_run(tmp_path):
    """A station observed in an earlier run stays observed after a run that missed it."""
    from src.fetchers.FMI import STATION_COLUMNS
    fetcher = _fetcher(tmp_path)

    fetcher.save_station_metadata(
        pd.DataFrame([_pool_row("Alpha", 1, 60.0, 24.0, observed=True)])[STATION_COLUMNS],
        "pool.csv",
    )
    fetcher.save_station_metadata(
        pd.DataFrame([_pool_row("Alpha", 1, 60.0, 24.0, observed=False)])[STATION_COLUMNS],
        "pool.csv",
    )

    result = pd.read_csv(tmp_path / "pool.csv").set_index("fmisid")
    assert bool(result.loc[1, "observed_in_run"])


def test_save_prefers_registry_coordinates_on_reunion(tmp_path):
    """A later registry coordinate correction must replace an observation coordinate."""
    from src.fetchers.FMI import STATION_COLUMNS
    fetcher = _fetcher(tmp_path)

    fetcher.save_station_metadata(
        pd.DataFrame([_pool_row("Tornio Kaakkuri", 101851, 65.8000, 24.1400,
                                coord_source="observation")])[STATION_COLUMNS],
        "pool.csv",
    )
    fetcher.save_station_metadata(
        pd.DataFrame([_pool_row("Tornio Kaakkuri", 101851, 65.8215, 24.1600,
                                coord_source="ef")])[STATION_COLUMNS],
        "pool.csv",
    )

    result = pd.read_csv(tmp_path / "pool.csv").set_index("fmisid")
    assert result.loc[101851, "latitude"] == 65.8215
    assert result.loc[101851, "coord_source"] == "ef"


def test_save_still_refuses_an_empty_table(tmp_path):
    from src.fetchers.FMI import STATION_COLUMNS
    fetcher = _fetcher(tmp_path)

    with pytest.raises(ValueError, match="Refusing to write an empty station table"):
        fetcher.save_station_metadata(pd.DataFrame(columns=STATION_COLUMNS), "pool.csv")


def test_save_backfills_observed_in_run_on_legacy_pool_file(tmp_path):
    """A pre-existing pool file written before the union (no observed_in_run column)
    must still union successfully, and its legacy rows must be treated as observed,
    since that file was built from the observation feed alone by construction."""
    from src.fetchers.FMI import STATION_COLUMNS

    fetcher = _fetcher(tmp_path)

    legacy = pd.DataFrame([{
        "station_name": "Alpha", "fmisid": 1, "latitude": 60.0, "longitude": 24.0,
        "networks": "Automaattinen sääasema", "is_weather_station": True,
        "in_ef_registry": True, "coord_source": "ef",
    }])
    assert list(legacy.columns) == STATION_COLUMNS[:-1]
    legacy.to_csv(tmp_path / "pool.csv", index=False, encoding="utf-8")

    second = pd.DataFrame([_pool_row("Beta", 2, 61.0, 25.0)])[STATION_COLUMNS]
    fetcher.save_station_metadata(second, "pool.csv")

    result = pd.read_csv(tmp_path / "pool.csv").set_index("fmisid")
    assert set(result.index) == {1, 2}
    assert bool(result.loc[1, "observed_in_run"])
    assert bool(result.loc[2, "observed_in_run"])
