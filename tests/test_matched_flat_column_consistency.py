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
        loader.output_folder = str(tmp_path)
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
