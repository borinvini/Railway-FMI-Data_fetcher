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


def test_reconcile_excludes_registry_only_stations():
    """A registry station that was never observed must not become a match candidate."""
    from src.fetchers.FMI import reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), _registry())

    assert 101104 not in set(result["fmisid"])
    assert 100662 not in set(result["fmisid"])
    assert len(result) == 2


def test_reconcile_prefers_registry_coordinates():
    from src.fetchers.FMI import reconcile_station_metadata

    result = reconcile_station_metadata(_observed(), _registry()).set_index("fmisid")

    assert result.loc[100683, "latitude"] == 60.303725
    assert result.loc[100683, "longitude"] == 25.549164
    assert result.loc[100683, "coord_source"] == "ef"


def test_reconcile_drops_observed_non_weather_station():
    from src.fetchers.FMI import reconcile_station_metadata

    observed = pd.DataFrame({
        "station_name": ["Helsinki Kallio 2"],
        "fmisid": [100662],
        "latitude": [60.18739],
        "longitude": [24.95060],
    })

    result = reconcile_station_metadata(observed, _registry())

    assert result.empty


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


def test_save_station_metadata_raises_on_empty(tmp_path):
    """An empty table must never silently leave a stale CSV in place."""
    from unittest.mock import patch
    from src.fetchers.FMI import FMIDataFetcher

    with patch("src.fetchers.FMI.FOLDER_NAME", str(tmp_path)):
        fetcher = FMIDataFetcher()

    with pytest.raises(ValueError, match="Refusing to write"):
        fetcher.save_station_metadata(pd.DataFrame(), "metadata_fmi_ems_stations.csv")
