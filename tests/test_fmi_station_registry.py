# -*- coding: utf-8 -*-
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ef_stations_sample.xml")


def _fixture_bytes():
    with open(FIXTURE, "rb") as handle:
        return handle.read()


def test_parse_ef_stations_extracts_expected_columns():
    from src.fetchers.FMIStations import parse_ef_stations

    df = parse_ef_stations(_fixture_bytes())

    assert list(df.columns) == [
        "fmisid", "station_name", "latitude", "longitude",
        "networks", "is_weather_station",
    ]


def test_parse_ef_stations_skips_facilities_without_coordinates():
    from src.fetchers.FMIStations import parse_ef_stations

    df = parse_ef_stations(_fixture_bytes())

    assert len(df) == 4
    assert 999999 not in set(df["fmisid"])


def test_parse_ef_stations_reads_lat_long_axis_order():
    from src.fetchers.FMIStations import parse_ef_stations

    row = parse_ef_stations(_fixture_bytes()).set_index("fmisid").loc[100683]

    assert row["latitude"] == 60.303725
    assert row["longitude"] == 25.549164


def test_parse_ef_stations_classifies_weather_networks():
    from src.fetchers.FMIStations import parse_ef_stations

    rows = parse_ef_stations(_fixture_bytes()).set_index("fmisid")

    assert rows.loc[100683, "is_weather_station"]        # Automaattinen saaasema
    assert not rows.loc[100662, "is_weather_station"]    # air quality
    assert not rows.loc[100539, "is_weather_station"]    # tide gauge
    assert rows.loc[101104, "is_weather_station"]        # Sadeasema + weather


def test_parse_ef_stations_joins_multiple_networks_sorted():
    from src.fetchers.FMIStations import parse_ef_stations

    row = parse_ef_stations(_fixture_bytes()).set_index("fmisid").loc[101104]

    assert row["networks"] == "Automaattinen sääasema|Sadeasema"


def test_fetch_registry_returns_empty_frame_on_network_failure():
    import requests
    from src.fetchers.FMIStations import FMIStationRegistry

    with patch("src.fetchers.FMIStations.requests.get",
               side_effect=requests.exceptions.ConnectionError("boom")):
        df = FMIStationRegistry().fetch_registry()

    assert df.empty
    assert list(df.columns) == [
        "fmisid", "station_name", "latitude", "longitude",
        "networks", "is_weather_station",
    ]


def test_fetch_registry_parses_successful_response():
    from src.fetchers.FMIStations import FMIStationRegistry

    response = MagicMock()
    response.content = _fixture_bytes()
    response.raise_for_status = MagicMock()

    with patch("src.fetchers.FMIStations.requests.get", return_value=response):
        df = FMIStationRegistry().fetch_registry()

    assert len(df) == 4
    assert int(df["is_weather_station"].sum()) == 2
