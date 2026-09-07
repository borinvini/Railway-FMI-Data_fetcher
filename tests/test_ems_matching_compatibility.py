# -*- coding: utf-8 -*-
import pandas as pd
import pytest
import sys
from unittest.mock import patch, MagicMock
from enum import Enum


# Mock haversine before importing DataLoader to work around broken numba environment
class MockUnit(Enum):
    KILOMETERS = "km"


def mock_haversine(coord1, coord2, unit=None):
    """Simple Haversine implementation to avoid numba dependency."""
    import math
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


@pytest.fixture(autouse=True)
def _haversine_guard(monkeypatch):
    """Substitute a mock haversine module only when the real one can't import.

    This keeps the substitution scoped to this test module (monkeypatch
    restores sys.modules automatically at teardown) and conditional on the
    real library genuinely failing to import, so a repaired numba
    environment transparently switches these tests back to the real
    haversine library instead of silently masking it forever.
    """
    try:
        import haversine  # noqa: F401
    except Exception:
        mock_module = MagicMock()
        mock_module.haversine = mock_haversine
        mock_module.Unit = MockUnit
        monkeypatch.setitem(sys.modules, "haversine", mock_module)


def _loader(tmp_path):
    from src.processors.DataLoader import DataLoader
    with patch.object(DataLoader, "_check_data_folder"):
        loader = DataLoader.__new__(DataLoader)
        loader.data_folder = str(tmp_path)
        loader.output_folder = str(tmp_path)
        loader.train_files = []
        loader.weather_files = []
        loader.merged_metadata = pd.DataFrame()
    return loader


def _write_train_stations(tmp_path):
    pd.DataFrame({
        "stationName": ["Helsinki asema"],
        "stationShortCode": ["HKI"],
        "latitude": [60.1719],
        "longitude": [24.9414],
        "passengerTraffic": [True],
    }).to_csv(tmp_path / "metadata_train_stations.csv", index=False)


def test_legacy_four_column_ems_file_still_matches(tmp_path):
    """A station file written before this change must keep working."""
    _write_train_stations(tmp_path)
    pd.DataFrame({
        "station_name": ["Helsinki Kaisaniemi"],
        "fmisid": [100971],
        "latitude": [60.17523],
        "longitude": [24.94459],
    }).to_csv(tmp_path / "metadata_fmi_ems_stations.csv", index=False)

    result = _loader(tmp_path).match_train_with_ems()

    assert result.iloc[0]["closest_ems_station"] == "Helsinki Kaisaniemi"


def test_non_weather_station_is_never_selected(tmp_path):
    """A nearer tide gauge must lose to a farther weather station."""
    _write_train_stations(tmp_path)
    pd.DataFrame({
        "station_name": ["Helsinki Kaisaniemi", "Helsinki mareografi"],
        "fmisid": [100971, 100539],
        "latitude": [60.17523, 60.17190],
        "longitude": [24.94459, 24.94140],
        "networks": ["Automaattinen sääasema", "Mareografiasema"],
        "is_weather_station": [True, False],
        "in_ef_registry": [True, True],
        "coord_source": ["ef", "ef"],
    }).to_csv(tmp_path / "metadata_fmi_ems_stations.csv", index=False)

    result = _loader(tmp_path).match_train_with_ems()

    assert result.iloc[0]["closest_ems_station"] == "Helsinki Kaisaniemi"
