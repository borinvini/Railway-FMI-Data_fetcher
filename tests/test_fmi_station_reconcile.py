# -*- coding: utf-8 -*-
import pandas as pd
import pytest


def _observed():
    return pd.DataFrame({
        "station_name": ["Porvoo Kilpilahti satama", "Pori rautatieasema"],
        "fmisid": [100683, 101064],
        "latitude": [60.30373, 61.47893],
        "longitude": [25.54916, 21.78320],
    })


def _registry():
    return pd.DataFrame({
        "fmisid": [100683, 100662, 101104],
        "station_name": ["Porvoo Kilpilahti satama", "Helsinki Kallio 2", "Tampere Siilinkari"],
        "latitude": [60.303725, 60.187390, 61.514500],
        "longitude": [25.549164, 24.950600, 23.752900],
        "networks": [
            "Automaattinen sääasema",
            "Kolmannen osapuolen ilmanlaadun havaintoasema",
            "Automaattinen sääasema|Sadeasema",
        ],
        "is_weather_station": [True, False, True],
    })


def test_reconcile_emits_the_pinned_column_schema():
    from src.fetchers.FMI import STATION_COLUMNS, reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), _registry())

    assert list(result.columns) == STATION_COLUMNS
    assert STATION_COLUMNS[:4] == ["station_name", "fmisid", "latitude", "longitude"]


def test_reconcile_retains_observed_station_absent_from_registry():
    """Pori rautatieasema is a real reporting station missing from the EF catalogue."""
    from src.fetchers.FMI import reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), _registry()).set_index("fmisid")

    assert 101064 in result.index
    assert not result.loc[101064, "in_ef_registry"]
    assert result.loc[101064, "is_weather_station"]
    assert result.loc[101064, "coord_source"] == "observation"


def test_reconcile_includes_registry_weather_stations_never_observed():
    """A registry weather station with no data this run is still a valid candidate.

    Liveness is resolved at merge time by walking the candidate ranks, so carrying
    an unobserved station costs nothing and keeps the pool stable across months.
    """
    from src.fetchers.FMI import reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), _registry()).set_index("fmisid")

    assert 101104 in result.index                      # Tampere Siilinkari, weather, unobserved
    assert result.loc[101104, "in_ef_registry"]
    assert not result.loc[101104, "observed_in_run"]
    assert result.loc[101104, "coord_source"] == "ef"


def test_reconcile_still_excludes_registry_non_weather_stations():
    """Air-quality and tide-gauge facilities must never become a weather candidate."""
    from src.fetchers.FMI import reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), _registry())

    assert 100662 not in set(result["fmisid"])         # Helsinki Kallio 2, air quality


def test_reconcile_marks_observed_stations():
    from src.fetchers.FMI import reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), _registry()).set_index("fmisid")

    assert result.loc[100683, "observed_in_run"]       # in both registry and observations
    assert result.loc[101064, "observed_in_run"]       # observation only


def test_reconcile_prefers_registry_coordinates():
    from src.fetchers.FMI import reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), _registry()).set_index("fmisid")

    assert result.loc[100683, "latitude"] == 60.303725
    assert result.loc[100683, "longitude"] == 25.549164
    assert result.loc[100683, "coord_source"] == "ef"


def test_reconcile_drops_observed_non_weather_station():
    """An observed non-weather station is excluded even though it reported data.

    The pool now unions in every registry weather station regardless of what was
    observed, so the result is not empty here — it just must never contain the
    excluded facility itself.
    """
    from src.fetchers.FMI import reconcile_station_metadata

    observed = pd.DataFrame({
        "station_name": ["Helsinki Kallio 2"],
        "fmisid": [100662],
        "latitude": [60.18739],
        "longitude": [24.95060],
    })

    result = reconcile_station_metadata(observed, _registry())

    assert 100662 not in set(result["fmisid"])


def test_reconcile_raises_on_empty_observed():
    from src.fetchers.FMI import reconcile_station_metadata

    with pytest.raises(ValueError, match="No EMS station metadata"):
        reconcile_station_metadata(pd.DataFrame(), _registry())


def test_reconcile_degrades_gracefully_without_registry():
    from src.fetchers.FMI import reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), pd.DataFrame())

    assert len(result) == 2
    assert set(result["coord_source"]) == {"observation"}
    assert result["is_weather_station"].all()
    assert result["latitude"].tolist() == [60.30373, 61.47893]


def test_reconcile_raises_on_duplicate_registry_fmisid():
    """A duplicated fmisid must stop the run, not silently fan out the join."""
    from src.fetchers.FMI import reconcile_station_metadata

    registry = _registry()
    duplicated = pd.concat([registry, registry.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate fmisid"):
        reconcile_station_metadata(_observed(), duplicated)


def test_save_station_metadata_raises_on_empty(tmp_path):
    """An empty table must never silently leave a stale CSV in place."""
    from unittest.mock import patch
    from src.fetchers.FMI import FMIDataFetcher

    with patch("src.fetchers.FMI.FOLDER_NAME", str(tmp_path)):
        fetcher = FMIDataFetcher()

    with pytest.raises(ValueError, match="Refusing to write"):
        fetcher.save_station_metadata(pd.DataFrame(), "metadata_fmi_ems_stations.csv")


def test_saved_station_file_round_trips_the_pinned_schema(tmp_path):
    """The file save_station_metadata writes is the contract match_train_with_ems reads.

    Closes the round trip: reconcile a real frame, write it with
    FMIDataFetcher.save_station_metadata, then read the CSV back and assert the
    on-disk header, dtypes, and row count are exactly what was written.
    """
    from unittest.mock import patch
    from src.fetchers.FMI import FMIDataFetcher, STATION_COLUMNS, reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), _registry())

    with patch("src.fetchers.FMI.FOLDER_NAME", str(tmp_path)):
        fetcher = FMIDataFetcher()

    filename = "metadata_fmi_ems_stations.csv"
    fetcher.save_station_metadata(result, filename)

    read_back = pd.read_csv(tmp_path / filename)

    assert list(read_back.columns) == STATION_COLUMNS
    assert read_back["is_weather_station"].dtype == bool
    assert len(read_back) == len(result)
