"""Tests for the data/ folder layout.

Monthly files used to sit next to the metadata in one flat folder. They now go
to three subfolders, while metadata_*.csv stays at the top level:

    data/train/    all_trains_data_*, all_trains_data_flat_*
    data/weather/  fmi_weather_observations_*
    data/matched/  matched_data_*, matched_data_flat_*, delay tables

These tests pin where the fetchers write, where DataLoader looks, and that an
old flat layout fails with a message saying to move the files, rather than the
misleading "run DATA_FETCH first".
"""

import pandas as pd
import pytest
from unittest.mock import patch


def _frame():
    return pd.DataFrame({"a": [1]})


def test_railway_fetcher_writes_months_to_train_and_metadata_to_top_level(tmp_path):
    from src.fetchers.Railway import RailwayDataFetcher

    with patch("src.fetchers.Railway.FOLDER_NAME", str(tmp_path)):
        fetcher = RailwayDataFetcher()

    fetcher.save_monthly_data_to_csv(_frame(), "2024-03")
    fetcher.save_to_csv(_frame(), "metadata_train_stations.csv")

    assert (tmp_path / "train" / "all_trains_data_2024_03.csv").exists()
    assert (tmp_path / "metadata_train_stations.csv").exists()
    assert not (tmp_path / "all_trains_data_2024_03.csv").exists()


def test_fmi_fetcher_writes_months_to_weather_and_metadata_to_top_level(tmp_path):
    from src.fetchers.FMI import FMIDataFetcher

    with patch("src.fetchers.FMI.FOLDER_NAME", str(tmp_path)):
        fetcher = FMIDataFetcher()

    fetcher.save_monthly_data_to_csv(_frame(), "fmi_weather_observations.csv", 2024, 3)
    fetcher.save_to_csv(_frame(), "metadata_fmi_ef_registry.csv")

    assert (tmp_path / "weather" / "fmi_weather_observations_2024_03.csv").exists()
    assert (tmp_path / "metadata_fmi_ef_registry.csv").exists()
    assert not (tmp_path / "fmi_weather_observations_2024_03.csv").exists()


def _loader(tmp_path):
    from src.processors.DataLoader import DataLoader
    with patch("src.processors.DataLoader.FOLDER_NAME", str(tmp_path)):
        return DataLoader()


def test_dataloader_finds_files_in_the_subfolders(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "weather").mkdir()
    _frame().to_csv(tmp_path / "train" / "all_trains_data_2024_03.csv", index=False)
    _frame().to_csv(tmp_path / "weather" / "fmi_weather_observations_2024_03.csv", index=False)

    loader = _loader(tmp_path)

    assert [p.endswith("all_trains_data_2024_03.csv") for p in loader.train_files] == [True]
    assert [p.endswith("fmi_weather_observations_2024_03.csv") for p in loader.weather_files] == [True]
    assert (tmp_path / "matched").is_dir()


def test_old_flat_layout_says_to_move_the_files(tmp_path):
    _frame().to_csv(tmp_path / "all_trains_data_2024_03.csv", index=False)
    _frame().to_csv(tmp_path / "fmi_weather_observations_2024_03.csv", index=False)

    with pytest.raises(FileNotFoundError, match="Move them"):
        _loader(tmp_path)


def test_empty_folder_still_points_to_data_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="No train data files"):
        _loader(tmp_path)
