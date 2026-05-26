import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_train_csv(tmp_path, year=2024, month=1):
    """Write a minimal all_trains_data_flat CSV and return its path."""
    df = pd.DataFrame({
        "trainNumber": [1, 2],
        "departureDate": ["2024-01-01", "2024-01-01"],
        "stationShortCode": ["HKI", "OL"],
        "differenceInMinutes": [0, 3],
    })
    filename = f"all_trains_data_flat_{year}_{month:02d}.csv"
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return df, path


def _make_fmi_csv(tmp_path, year=2024, month=1):
    """Write a minimal fmi_weather_observations CSV and return its path."""
    df = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00Z"],
        "station_name": ["Helsinki"],
        "Air temperature": [-5.0],
    })
    filename = f"fmi_weather_observations_{year}_{month:02d}.csv"
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return df, path


def _make_matched_flat_csv(tmp_path, year=2024, month=1):
    """Write a minimal matched_data_flat CSV and return its path."""
    df = pd.DataFrame({
        "trainNumber": [1],
        "stationShortCode": ["HKI"],
        "Air temperature": [-3.0],
        "differenceInMinutes": [2],
    })
    filename = f"matched_data_flat_{year}_{month:02d}.csv"
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return df, path


def _make_dataloader(tmp_path):
    """Return a DataLoader instance with data_folder pointing to tmp_path."""
    from src.processors.DataLoader import DataLoader
    with patch.object(DataLoader, "_check_data_folder"):
        loader = DataLoader.__new__(DataLoader)
        loader.data_folder = str(tmp_path)
        loader.output_folder = str(tmp_path)
        loader.train_files = []
        loader.weather_files = []
        loader.merged_metadata = __import__("pandas").DataFrame()
        loader.ems_metadata = __import__("pandas").DataFrame()
        loader.ems_weather_dict = {}
        loader.top5_ems_dict = {}
    return loader


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_convert_flat_train_csv_to_parquet(tmp_path):
    """A flat train CSV is converted to parquet with identical content."""
    df_original, csv_path = _make_flat_train_csv(tmp_path)
    loader = _make_dataloader(tmp_path)

    loader.convert_to_parquet()

    parquet_path = tmp_path / "all_trains_data_flat_2024_01.parquet"
    assert parquet_path.exists(), "Parquet file was not created"
    df_result = pd.read_parquet(parquet_path, engine="pyarrow")
    assert list(df_result.columns) == list(df_original.columns)
    assert len(df_result) == len(df_original)


def test_convert_fmi_csv_to_parquet(tmp_path):
    """A FMI weather CSV is converted to parquet with identical content."""
    df_original, csv_path = _make_fmi_csv(tmp_path)
    loader = _make_dataloader(tmp_path)

    loader.convert_to_parquet()

    parquet_path = tmp_path / "fmi_weather_observations_2024_01.parquet"
    assert parquet_path.exists(), "Parquet file was not created"
    df_result = pd.read_parquet(parquet_path, engine="pyarrow")
    assert list(df_result.columns) == list(df_original.columns)
    assert len(df_result) == len(df_original)


def test_convert_matched_flat_csv_to_parquet(tmp_path):
    """A matched flat CSV is converted to parquet with identical content."""
    df_original, csv_path = _make_matched_flat_csv(tmp_path)
    loader = _make_dataloader(tmp_path)

    loader.convert_to_parquet()

    parquet_path = tmp_path / "matched_data_flat_2024_01.parquet"
    assert parquet_path.exists(), "Parquet file was not created"
    df_result = pd.read_parquet(parquet_path, engine="pyarrow")
    assert list(df_result.columns) == list(df_original.columns)
    assert len(df_result) == len(df_original)


def test_skips_existing_parquet(tmp_path):
    """If a parquet file already exists it is not overwritten."""
    _make_flat_train_csv(tmp_path)
    loader = _make_dataloader(tmp_path)

    # Write a sentinel parquet that should survive untouched
    parquet_path = tmp_path / "all_trains_data_flat_2024_01.parquet"
    sentinel_df = pd.DataFrame({"sentinel": [True]})
    sentinel_df.to_parquet(parquet_path, engine="pyarrow", index=False)

    loader.convert_to_parquet()

    result = pd.read_parquet(parquet_path, engine="pyarrow")
    assert list(result.columns) == ["sentinel"], "Existing parquet was overwritten"


def test_no_csvs_found_does_not_raise(tmp_path):
    """If no CSVs exist for a source type, the method completes without error."""
    loader = _make_dataloader(tmp_path)
    # No CSV files in tmp_path — should print warnings and return cleanly
    loader.convert_to_parquet()  # must not raise


def test_multiple_months_all_converted(tmp_path):
    """All monthly CSVs for a source type are converted."""
    _make_flat_train_csv(tmp_path, year=2024, month=1)
    _make_flat_train_csv(tmp_path, year=2024, month=2)
    loader = _make_dataloader(tmp_path)

    loader.convert_to_parquet()

    assert (tmp_path / "all_trains_data_flat_2024_01.parquet").exists()
    assert (tmp_path / "all_trains_data_flat_2024_02.parquet").exists()
