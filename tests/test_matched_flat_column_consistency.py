"""Regression tests for convert_matched_to_flat() column alignment.

The flat conversion writes one 500-train chunk at a time, appending to the CSV
with the header taken from the first chunk only. If an optional stop-level key
(e.g. 'unknownTrack') is absent from the first chunk but present in a later one,
the later chunk's DataFrame gains an extra column. Appended with header=False,
this shifts every column from that row on, scattering weather values into the
wrong headers. These tests pin the file to a single, consistent column schema.
"""

import csv

import pandas as pd
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRAIN_COLS = [
    "trainNumber", "departureDate", "operatorUICCode", "operatorShortCode",
    "trainType", "trainCategory", "commuterLineID", "runningCurrently",
    "cancelled", "version", "timetableType", "timetableAcceptanceDate",
]


def _make_dataloader(tmp_path):
    from src.processors.DataLoader import DataLoader
    with patch.object(DataLoader, "_check_data_folder"):
        loader = DataLoader.__new__(DataLoader)
        # The tests only write matched files; the train and weather folders are
        # searched by convert_to_parquet and simply hold nothing here.
        loader.matched_folder = loader.train_folder = loader.weather_folder = str(tmp_path)
    return loader


def _stop(station, temp, extra=None):
    stop = {
        "stationName": station,
        "stationShortCode": station[:3].upper(),
        "type": "ARRIVAL",
        "scheduledTime": "2024-01-01T00:00:00.000Z",
        "actualTime": "2024-01-01T00:01:00.000Z",
        "differenceInMinutes": 1,
        "cancelled": False,
        "causes": [],
        "trainReady": None,
        "weather_observations": {"Air temperature": temp, "Pressure (msl)": 1013.0},
    }
    if extra:
        stop.update(extra)
    return stop


def _make_matched_csv(tmp_path, first_chunk_size=500):
    """Write matched_data_2024_01.csv where an optional key ('unknownTrack')
    appears only in the *second* 500-train chunk, forcing schema drift."""
    rows = []
    base = {c: "x" for c in _TRAIN_COLS}
    # First chunk: 500 trains whose stops lack 'unknownTrack'.
    for i in range(first_chunk_size):
        rows.append({**base, "trainNumber": i,
                     "timeTableRows": str([_stop("Helsinki", -5.0)])})
    # Second chunk: one train whose stop carries the optional 'unknownTrack' key.
    rows.append({**base, "trainNumber": first_chunk_size,
                 "timeTableRows": str([_stop("Oulu", -7.7, extra={"unknownTrack": True})])})
    path = tmp_path / "matched_data_2024_01.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_flat_csv_field_count_is_consistent_across_chunks(tmp_path):
    """Every data row must have the same field count as the header, even when a
    later chunk introduces an optional key the first chunk never had."""
    _make_matched_csv(tmp_path)
    loader = _make_dataloader(tmp_path)

    loader.convert_matched_to_flat()

    flat = tmp_path / "matched_data_flat_2024_01.csv"
    assert flat.exists(), "Flat CSV was not created"

    with open(flat, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    header_len = len(rows[0])
    mismatched = [i for i, row in enumerate(rows[1:], start=2) if len(row) != header_len]
    assert not mismatched, (
        f"{len(mismatched)} row(s) have a field count != header ({header_len}); "
        f"first offending lines: {mismatched[:5]}"
    )


def test_weather_value_not_shifted_for_late_chunk(tmp_path):
    """The weather value for a train in a later chunk must land in its own
    column, not be shifted into a neighbour by an upstream extra column."""
    _make_matched_csv(tmp_path)
    loader = _make_dataloader(tmp_path)

    loader.convert_matched_to_flat()

    flat = tmp_path / "matched_data_flat_2024_01.csv"
    df = pd.read_csv(flat)

    oulu = df[df["stationName"] == "Oulu"]
    assert len(oulu) == 1, "Expected exactly one Oulu stop"
    assert oulu.iloc[0]["Air temperature"] == -7.7
    assert oulu.iloc[0]["Pressure (msl)"] == 1013.0


# ---------------------------------------------------------------------------
# Fixed schema across months (parquet)
# ---------------------------------------------------------------------------

import pyarrow.parquet as pq

_TYPED_TRAIN = {
    "trainNumber": 1, "departureDate": "2024-01-01", "operatorUICCode": 10,
    "operatorShortCode": "vr", "trainType": "IC", "trainCategory": "Long-distance",
    "commuterLineID": None, "runningCurrently": False, "cancelled": False,
    "version": 1, "timetableType": "REGULAR", "timetableAcceptanceDate": "2023-12-01",
}


def _make_typed_matched_csv(tmp_path, year_month, track, extra=None, weather_extra=None):
    weather = {"Air temperature": -5.0, "Pressure (msl)": 1013.0}
    weather.update(weather_extra or {})
    stop = _stop("Helsinki", -5.0, extra={"commercialTrack": track, **(extra or {})})
    stop["weather_observations"] = weather
    stop.update({"stationUICCode": 1, "commercialStop": True, "trainStopping": True})
    row = {**_TYPED_TRAIN, "timeTableRows": str([stop])}
    pd.DataFrame([row]).to_csv(tmp_path / f"matched_data_{year_month}.csv", index=False)


def _build_parquets(tmp_path):
    loader = _make_dataloader(tmp_path)
    loader.convert_matched_to_flat()
    loader.convert_to_parquet()
    return loader


def test_parquet_schema_identical_across_months(tmp_path):
    from src.processors.DataLoader import DataLoader
    # Month A: digit-only track, no optional fields. Month B: letter code plus optional fields.
    _make_typed_matched_csv(tmp_path, "2024_01", "001")
    _make_typed_matched_csv(tmp_path, "2024_02", "5b",
                            extra={"unknownTrack": True, "stopSector": "A", "unknownDelay": False})
    _build_parquets(tmp_path)

    expected = DataLoader._matched_flat_schema()
    for month in ("2024_01", "2024_02"):
        got = pq.read_schema(tmp_path / "parquet" / f"matched_data_flat_{month}.parquet")
        assert got.equals(expected), f"{month} schema differs from the fixed schema"

    a = pd.read_parquet(tmp_path / "parquet" / "matched_data_flat_2024_01.parquet")
    b = pd.read_parquet(tmp_path / "parquet" / "matched_data_flat_2024_02.parquet")
    assert a["commercialTrack"].iloc[0] == "1"
    assert b["commercialTrack"].iloc[0] == "5b"
    assert a["unknownTrack"].isna().all()


def test_unknown_columns_are_dropped(tmp_path):
    _make_typed_matched_csv(tmp_path, "2024_01", "1",
                            weather_extra={"Unnamed: 15": 85.0, "newApiField": 1.0})
    _build_parquets(tmp_path)

    cols = pq.read_schema(tmp_path / "parquet" / "matched_data_flat_2024_01.parquet").names
    assert "Unnamed: 15" not in cols and "newApiField" not in cols
    flat = pd.read_csv(tmp_path / "matched_data_flat_2024_01.csv")
    assert not [c for c in flat.columns if c.startswith("Unnamed")]


def test_normalize_commercial_track():
    from src.processors.DataLoader import DataLoader
    out = DataLoader._normalize_commercial_track(
        pd.Series(["001", "1", 1.0, "5b", "IR", " 2 ", None, "000", "019b"], dtype=object))
    assert out.tolist()[:6] == ["1", "1", "1", "5b", "IR", "2"]
    assert pd.isna(out.iloc[6])
    assert out.tolist()[7:] == ["0", "019b"]
