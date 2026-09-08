# -*- coding: utf-8 -*-
import pandas as pd


def _pool():
    """Twelve stations strung north from 60.0N along the same meridian.

    Spacing is 0.02 deg (~2.2 km) so every station stays inside the 50 km
    radius that Task 2 introduces. At 0.05 deg spacing S9 lands at 50.04 km
    and this fixture would start failing one task later.
    """
    return pd.DataFrame({
        "station_name": [f"S{i}" for i in range(12)],
        "fmisid": list(range(12)),
        "latitude": [60.0 + 0.02 * i for i in range(12)],
        "longitude": [24.0] * 12,
    })


def test_candidate_table_is_ten_wide():
    from config.const import TOP_N_CLOSEST_EMS
    from src.processors.DataLoader import DataLoader

    assert TOP_N_CLOSEST_EMS == 10

    result = DataLoader._find_top_n_closest_ems(
        None, 60.0, 24.0, _pool(), n=TOP_N_CLOSEST_EMS
    )

    assert result["ems_1_station"] == "S0"
    assert result["ems_10_station"] == "S9"
    assert "ems_11_station" not in result.index


def test_candidate_filename_is_top10():
    from config.const import CSV_TOPN_CLOSEST_EMS_TRAIN

    assert CSV_TOPN_CLOSEST_EMS_TRAIN == "metadata_top10_closest_ems.csv"


def _radius_pool():
    """One station ~49 km north of 60.0N, one ~51 km north, one far away."""
    return pd.DataFrame({
        "station_name": ["Near", "Far", "Distant"],
        "fmisid": [1, 2, 3],
        "latitude": [60.4405, 60.4585, 65.0],
        "longitude": [24.0, 24.0, 24.0],
    })


def test_candidate_inside_radius_is_kept_and_outside_is_dropped():
    from config.const import TOP_N_CLOSEST_EMS
    from src.processors.DataLoader import DataLoader

    result = DataLoader._find_top_n_closest_ems(
        None, 60.0, 24.0, _radius_pool(), n=TOP_N_CLOSEST_EMS
    )

    assert result["ems_1_station"] == "Near"
    assert result["ems_1_distance_km"] < 50
    assert pd.isna(result["ems_2_station"])
    assert pd.isna(result["ems_2_distance_km"])


def test_all_slots_present_even_when_empty():
    from config.const import TOP_N_CLOSEST_EMS
    from src.processors.DataLoader import DataLoader

    far_only = pd.DataFrame({
        "station_name": ["Distant"], "fmisid": [1],
        "latitude": [65.0], "longitude": [24.0],
    })

    result = DataLoader._find_top_n_closest_ems(
        None, 60.0, 24.0, far_only, n=TOP_N_CLOSEST_EMS
    )

    assert len(result) == TOP_N_CLOSEST_EMS * 4
    assert result.isna().all()


def test_closest_ems_returns_nan_when_nothing_in_radius():
    from src.processors.DataLoader import DataLoader

    far_only = pd.DataFrame({
        "station_name": ["Distant"], "fmisid": [1],
        "latitude": [65.0], "longitude": [24.0],
    })

    name, lat, lon, dist = DataLoader._find_closest_ems(None, 60.0, 24.0, far_only)

    assert name is None
    assert lat is None
    assert lon is None
    assert pd.isna(dist)


def test_non_finnish_train_stations_are_excluded(tmp_path):
    from unittest.mock import patch
    from src.processors.DataLoader import DataLoader

    # DataLoader.__init__ calls _check_data_folder, which raises FileNotFoundError
    # unless the folder holds train AND weather files whose date ranges match.
    # These two stubs exist only to get past that gate.
    pd.DataFrame({"trainNumber": [1]}).to_csv(
        tmp_path / "all_trains_data_2018_01.csv", index=False)
    pd.DataFrame({"timestamp": ["2018-01-01T00:00:00Z"], "station_name": ["X"]}).to_csv(
        tmp_path / "fmi_weather_observations_2018_01.csv", index=False)

    pd.DataFrame({
        "stationName": ["Helsinki asema", "Tver"],
        "stationShortCode": ["HKI", "TVE"],
        "latitude": [60.171356, 56.835200],
        "longitude": [24.941444, 35.892800],
        "countryCode": ["FI", "RU"],
        "type": ["STATION", "STATION"],
        "stationUICCode": [1, 2],
        "passengerTraffic": [True, True],
    }).to_csv(tmp_path / "metadata_train_stations.csv", index=False)

    pd.DataFrame({
        "station_name": ["Helsinki Kaisaniemi"],
        "fmisid": [100971],
        "latitude": [60.17523],
        "longitude": [24.94459],
    }).to_csv(tmp_path / "metadata_fmi_ems_stations.csv", index=False)

    with patch("src.processors.DataLoader.FOLDER_NAME", str(tmp_path)):
        loader = DataLoader()
        result = loader.match_train_with_ems()

    assert set(result["train_station_short_code"]) == {"HKI"}
