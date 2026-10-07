"""Tests for deleting the matched CSV files once a month's parquet exists.

The matched step writes two large CSVs per month (matched_data_YYYY_MM.csv and
matched_data_flat_YYYY_MM.csv) and then a parquet from the flat one. The CSVs are
only intermediate. convert_to_parquet(delete_matched_csv=True) removes both after
checking that the parquet holds every row of the CSV. Because re-runs used to skip
a month by looking for its CSV, a month with a parquet must now count as done too,
or every deleted month would be merged again from scratch.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

_TRAIN = {
    "trainNumber": 1, "departureDate": "2024-01-01", "operatorUICCode": 10,
    "operatorShortCode": "vr", "trainType": "IC", "trainCategory": "Long-distance",
    "commuterLineID": None, "runningCurrently": False, "cancelled": False,
    "version": 1, "timetableType": "REGULAR", "timetableAcceptanceDate": "2023-12-01",
}
_STOP = {
    "stationName": "Helsinki", "stationShortCode": "HEL", "type": "ARRIVAL", "stationUICCode": 1,
    "scheduledTime": "2024-01-01T00:00:00.000Z", "actualTime": "2024-01-01T00:01:00.000Z",
    "differenceInMinutes": 1, "cancelled": False, "causes": [], "trainReady": None,
    "commercialStop": True, "trainStopping": True, "commercialTrack": "1",
    "weather_observations": {"Air temperature": -5.0, "Pressure (msl)": 1013.0},
}


def _loader(tmp_path):
    from src.processors.DataLoader import DataLoader
    with patch.object(DataLoader, "_check_data_folder"):
        loader = DataLoader.__new__(DataLoader)
        loader.matched_folder = loader.train_folder = loader.weather_folder = str(tmp_path)
    return loader


def _matched_csv(tmp_path):
    pd.DataFrame([{**_TRAIN, "timeTableRows": str([_STOP])}]).to_csv(tmp_path / "matched_data_2024_01.csv", index=False)


def _built(tmp_path, delete):
    _matched_csv(tmp_path)
    loader = _loader(tmp_path)
    loader.convert_matched_to_flat()
    loader.convert_to_parquet(delete_matched_csv=delete)
    return loader


def test_both_csvs_are_deleted_after_a_verified_parquet(tmp_path):
    _built(tmp_path, delete=True)

    assert (tmp_path / "parquet" / "matched_data_flat_2024_01.parquet").exists()
    assert not (tmp_path / "matched_data_flat_2024_01.csv").exists()
    assert not (tmp_path / "matched_data_2024_01.csv").exists()


def test_csvs_are_kept_by_default(tmp_path):
    _built(tmp_path, delete=False)

    assert (tmp_path / "parquet" / "matched_data_flat_2024_01.parquet").exists()
    assert (tmp_path / "matched_data_flat_2024_01.csv").exists()
    assert (tmp_path / "matched_data_2024_01.csv").exists()


def test_csvs_are_kept_when_the_parquet_has_fewer_rows_than_the_csv(tmp_path, capsys):
    """A line pandas skips (too many fields) must stop the deletion."""
    _matched_csv(tmp_path)
    loader = _loader(tmp_path)
    loader.convert_matched_to_flat()
    flat = tmp_path / "matched_data_flat_2024_01.csv"
    with open(flat, "a", encoding="utf-8", newline="") as fh:
        fh.write(",".join(["x"] * 400) + "\n")

    loader.convert_to_parquet(delete_matched_csv=True)

    assert (tmp_path / "parquet" / "matched_data_flat_2024_01.parquet").exists()
    assert flat.exists() and (tmp_path / "matched_data_2024_01.csv").exists()
    assert "Keeping the CSV files" in capsys.readouterr().out


def test_leftover_csvs_of_an_older_parquet_are_not_touched(tmp_path):
    """Only months converted in this call are cleaned up."""
    _built(tmp_path, delete=False)

    _loader(tmp_path).convert_to_parquet(delete_matched_csv=True)  # parquet exists, so it is skipped

    assert (tmp_path / "matched_data_flat_2024_01.csv").exists()
    assert (tmp_path / "matched_data_2024_01.csv").exists()


def _month_inputs(tmp_path):
    train = tmp_path / "all_trains_data_2024_01.csv"
    weather = tmp_path / "fmi_weather_observations_2024_01.csv"
    pd.DataFrame({"a": [1]}).to_csv(train, index=False)
    pd.DataFrame({"a": [1]}).to_csv(weather, index=False)
    loader = _loader(tmp_path)
    loader.train_files, loader.weather_files = [str(train)], [str(weather)]
    return loader


def test_month_with_a_parquet_is_not_merged_again(tmp_path):
    loader = _month_inputs(tmp_path)
    (tmp_path / "parquet").mkdir()
    (tmp_path / "parquet" / "matched_data_flat_2024_01.parquet").write_bytes(b"")

    with patch.object(type(loader), "merge_train_weather_data", MagicMock()) as merge:
        loader.load_csv_files_by_month()

    merge.assert_not_called()


def test_month_without_csv_or_parquet_is_merged(tmp_path):
    """Control for the test above: the same setup without a parquet does merge."""
    loader = _month_inputs(tmp_path)

    with patch.object(type(loader), "merge_train_weather_data", MagicMock()) as merge, \
            patch("src.processors.DataLoader.send_email"):
        loader.load_csv_files_by_month()

    merge.assert_called_once()


def test_flat_step_skips_a_month_that_has_a_parquet(tmp_path):
    _matched_csv(tmp_path)
    (tmp_path / "parquet").mkdir()
    (tmp_path / "parquet" / "matched_data_flat_2024_01.parquet").write_bytes(b"")

    _loader(tmp_path).convert_matched_to_flat()

    assert not (tmp_path / "matched_data_flat_2024_01.csv").exists()
