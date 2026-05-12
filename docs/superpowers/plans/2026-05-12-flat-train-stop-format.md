# Flat Train Stop Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `FLAT_FORMAT` flag that triggers a pipeline step converting `all_trains_data_YYYY_MM.csv` files to one-row-per-stop flat CSVs, and saves `matched_data_flat_YYYY_MM.csv` in the same format.

**Architecture:** New constants gate the feature. `DataLoader` gets a `convert_trains_to_flat()` method (run as STEP 1.5 in `main.py`) and a `_save_matched_flat()` private helper (called from the existing `merge_train_weather_data()`). Both flat files are saved alongside their originals — originals are never modified.

**Tech Stack:** Python 3.12, pandas, ast (stdlib), pytest

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `config/const.py` | Modify | Add `FLAT_FORMAT`, `CSV_ALL_TRAINS_FLAT`, `CSV_MATCHED_DATA_FLAT` |
| `src/processors/DataLoader.py` | Modify | New imports, `convert_trains_to_flat()`, `_save_matched_flat()`, one-liner in `merge_train_weather_data()` |
| `main.py` | Modify | Import `FLAT_FORMAT`, add STEP 1.5 block |
| `tests/test_flat_conversion.py` | Create | Unit tests for both new methods |

---

### Task 1: Add constants to `config/const.py`

**Files:**
- Modify: `config/const.py`

- [ ] **Step 1: Add the three new constants**

Open `config/const.py`. After the line `CSV_MATCHED_DATA = "matched_data.csv"` (around line 95), add:

```python
CSV_ALL_TRAINS_FLAT = "all_trains_data_flat.csv"
CSV_MATCHED_DATA_FLAT = "matched_data_flat.csv"

FLAT_FORMAT = False  # Set True to produce flat (one-row-per-stop) CSVs alongside originals
```

- [ ] **Step 2: Verify Python can import them**

Run:
```
python -c "from config.const import FLAT_FORMAT, CSV_ALL_TRAINS_FLAT, CSV_MATCHED_DATA_FLAT; print(FLAT_FORMAT, CSV_ALL_TRAINS_FLAT, CSV_MATCHED_DATA_FLAT)"
```

Expected output:
```
False all_trains_data_flat.csv matched_data_flat.csv
```

- [ ] **Step 3: Commit**

```bash
git add config/const.py
git commit -m "feat: add FLAT_FORMAT flag and flat CSV filename constants"
```

---

### Task 2: Write failing tests for `convert_trains_to_flat()`

**Files:**
- Create: `tests/test_flat_conversion.py`

- [ ] **Step 1: Create the test file with fixtures**

Create `tests/test_flat_conversion.py`:

```python
import ast
import os
import pandas as pd
import pytest
from unittest.mock import patch

from src.processors.DataLoader import DataLoader


STOP_1 = {
    'stationName': 'Helsinki asema',
    'stationShortCode': 'HKI',
    'stationUICCode': 1,
    'countryCode': 'FI',
    'type': 'DEPARTURE',
    'trainStopping': True,
    'commercialStop': True,
    'commercialTrack': '9',
    'cancelled': False,
    'scheduledTime': '2024-01-01T04:57:00.000Z',
    'actualTime': '2024-01-01T04:57:21.000Z',
    'differenceInMinutes': 0,
    'causes': [],
    'trainReady': {'source': 'KUPLA', 'accepted': True, 'timestamp': '2024-01-01T04:51:10.000Z'},
}

STOP_2 = {
    'stationName': 'Tampere asema',
    'stationShortCode': 'TPE',
    'stationUICCode': 160,
    'countryCode': 'FI',
    'type': 'ARRIVAL',
    'trainStopping': True,
    'commercialStop': True,
    'commercialTrack': '2',
    'cancelled': False,
    'scheduledTime': '2024-01-01T06:57:00.000Z',
    'actualTime': '2024-01-01T06:57:21.000Z',
    'differenceInMinutes': 2,
    'causes': [{'categoryCode': 'X1'}],
    'trainReady': None,
}

TRAIN_ROW = {
    'trainNumber': 1,
    'departureDate': '2024-01-01',
    'operatorUICCode': 10,
    'operatorShortCode': 'vr',
    'trainType': 'IC',
    'trainCategory': 'Long-distance',
    'commuterLineID': None,
    'runningCurrently': False,
    'cancelled': False,
    'version': 123456,
    'timetableType': 'REGULAR',
    'timetableAcceptanceDate': '2023-11-02T05:57:22.000Z',
    'timeTableRows': str([STOP_1, STOP_2]),
}


@pytest.fixture
def loader(tmp_path):
    with patch.object(DataLoader, '_check_data_folder'):
        dl = DataLoader()
    dl.output_folder = str(tmp_path)
    dl.data_folder = str(tmp_path)
    return dl


def make_train_csv(tmp_path, filename, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


class TestConvertTrainsToFlat:

    def test_output_row_count_equals_total_stops(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_path = os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv')
        assert os.path.exists(flat_path)
        flat_df = pd.read_csv(flat_path)
        assert len(flat_df) == 2  # STOP_1 + STOP_2

    def test_train_level_columns_repeated_for_each_stop(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv'))
        assert list(flat_df['trainNumber']) == [1, 1]
        assert list(flat_df['trainType']) == ['IC', 'IC']
        assert list(flat_df['departureDate']) == ['2024-01-01', '2024-01-01']

    def test_stop_cancelled_renamed_from_cancelled(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv'))
        assert 'stop_cancelled' in flat_df.columns
        assert list(flat_df['stop_cancelled']) == [False, False]

    def test_causes_stored_as_string(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv'))
        assert flat_df['causes'].dtype == object  # stored as string
        assert flat_df['causes'].iloc[0] == '[]'
        assert flat_df['causes'].iloc[1] == "[{'categoryCode': 'X1'}]"

    def test_train_ready_stored_as_string_or_none(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv'))
        # STOP_1 has trainReady dict, STOP_2 has None
        assert isinstance(flat_df['trainReady'].iloc[0], str)
        assert pd.isna(flat_df['trainReady'].iloc[1])

    def test_skips_existing_flat_file(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        # Pre-create flat file with known content
        flat_path = os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv')
        pd.DataFrame([{'sentinel': 'do_not_overwrite'}]).to_csv(flat_path, index=False)
        mtime_before = os.path.getmtime(flat_path)

        loader.convert_trains_to_flat()

        assert os.path.getmtime(flat_path) == mtime_before  # file not touched
```

- [ ] **Step 2: Run tests to confirm they all fail (method not yet implemented)**

```
python -m pytest tests/test_flat_conversion.py::TestConvertTrainsToFlat -v
```

Expected: all 6 tests fail with `AttributeError: 'DataLoader' object has no attribute 'convert_trains_to_flat'`

---

### Task 3: Implement `convert_trains_to_flat()` in `DataLoader`

**Files:**
- Modify: `src/processors/DataLoader.py`

- [ ] **Step 1: Update the import line at the top of `DataLoader.py`**

Find the existing import (line 10):
```python
from config.const import ALTERNATIVE_WEATHER_COLUMN, CSV_ALL_TRAINS, CSV_CLOSEST_EMS_TRAIN, CSV_TOP5_CLOSEST_EMS_TRAIN, CSV_DELAY_TABLE_EACH_STATION, CSV_DELAY_TABLE_OFFSET, CSV_DELAY_TABLE_ORIGINAL, CSV_FMI, CSV_FMI_EMS, CSV_MATCHED_DATA, CSV_TRAIN_STATIONS, DELAY_LONG_DISTANCE_TRAINS, FILTER_BY_ROUTE, FILTER_BY_TRAIN_CATEGORY, FMI_ROLLING_WINDOW_HOURS, FMI_ROLLING_WINDOW_PARAMS, FMI_ROLLING_SKIP_MIN_MAX, FOLDER_NAME, MANDATORY_STATIONS, TRAIN_CATEGORY_FILTER, get_fmi_rolling_column_names
```

Replace with:
```python
from config.const import ALTERNATIVE_WEATHER_COLUMN, CSV_ALL_TRAINS, CSV_ALL_TRAINS_FLAT, CSV_CLOSEST_EMS_TRAIN, CSV_MATCHED_DATA_FLAT, CSV_TOP5_CLOSEST_EMS_TRAIN, CSV_DELAY_TABLE_EACH_STATION, CSV_DELAY_TABLE_OFFSET, CSV_DELAY_TABLE_ORIGINAL, CSV_FMI, CSV_FMI_EMS, CSV_MATCHED_DATA, CSV_TRAIN_STATIONS, DELAY_LONG_DISTANCE_TRAINS, FILTER_BY_ROUTE, FILTER_BY_TRAIN_CATEGORY, FLAT_FORMAT, FMI_ROLLING_WINDOW_HOURS, FMI_ROLLING_WINDOW_PARAMS, FMI_ROLLING_SKIP_MIN_MAX, FOLDER_NAME, MANDATORY_STATIONS, TRAIN_CATEGORY_FILTER, get_fmi_rolling_column_names
```

- [ ] **Step 2: Add `convert_trains_to_flat()` method**

Add the following method to the `DataLoader` class, after `preprocess_fmi_rolling_features()` (around line 393, before `_find_closest_ems`):

```python
def convert_trains_to_flat(self):
    """
    Convert all_trains_data CSV files to flat format (one row per train stop).

    Reads each all_trains_data_YYYY_MM.csv, explodes timeTableRows into individual
    rows, and saves all_trains_data_flat_YYYY_MM.csv alongside the original.
    Skips months where the flat file already exists.
    """
    if not self.train_files:
        raise ValueError("No train files loaded.")

    print(f"\n{'='*60}")
    print("STEP 1.5: Converting train data to flat format")
    print(f"{'='*60}")

    train_level_cols = [
        'trainNumber', 'departureDate', 'operatorUICCode', 'operatorShortCode',
        'trainType', 'trainCategory', 'commuterLineID', 'runningCurrently',
        'cancelled', 'version', 'timetableType', 'timetableAcceptanceDate',
    ]

    for train_file in sorted(self.train_files):
        dates = self._extract_dates_from_filenames([train_file])
        if not dates:
            print(f"⚠️ Could not extract date from {train_file}. Skipping.")
            continue

        month_period = pd.Period(dates[0], freq='M')
        base = CSV_ALL_TRAINS_FLAT.replace('.csv', '')
        flat_filename = f"{base}_{month_period.year}_{month_period.month:02d}.csv"
        flat_filepath = os.path.join(self.output_folder, flat_filename)

        if os.path.exists(flat_filepath):
            print(f"  ℹ️ {flat_filename} already exists. Skipping.")
            continue

        print(f"📊 Converting {os.path.basename(train_file)}...")
        train_data = pd.read_csv(train_file)
        rows = []

        for _, train_row in train_data.iterrows():
            timetable_raw = train_row['timeTableRows']
            try:
                timetable = ast.literal_eval(timetable_raw) if isinstance(timetable_raw, str) else timetable_raw
            except (ValueError, SyntaxError) as e:
                print(f"⚠️ Failed to parse timeTableRows for train {train_row.get('trainNumber')}: {e}")
                continue

            if not isinstance(timetable, list):
                continue

            train_base = {col: train_row.get(col) for col in train_level_cols if col in train_row.index}

            for stop in timetable:
                row = dict(train_base)
                row['stationName'] = stop.get('stationName')
                row['stationShortCode'] = stop.get('stationShortCode')
                row['stationUICCode'] = stop.get('stationUICCode')
                row['countryCode'] = stop.get('countryCode')
                row['type'] = stop.get('type')
                row['trainStopping'] = stop.get('trainStopping')
                row['commercialStop'] = stop.get('commercialStop')
                row['commercialTrack'] = stop.get('commercialTrack')
                row['stop_cancelled'] = stop.get('cancelled')
                row['scheduledTime'] = stop.get('scheduledTime')
                row['actualTime'] = stop.get('actualTime')
                row['differenceInMinutes'] = stop.get('differenceInMinutes')
                row['causes'] = str(stop.get('causes', []))
                train_ready = stop.get('trainReady')
                row['trainReady'] = str(train_ready) if train_ready is not None else None
                rows.append(row)

        flat_df = pd.DataFrame(rows)
        flat_df.to_csv(flat_filepath, index=False)
        print(f"  ✅ Saved {len(rows)} rows to {flat_filename}")

    print(f"\n✅ Flat train conversion complete.")
    print(f"{'='*60}\n")
```

- [ ] **Step 3: Run tests — all 6 should pass**

```
python -m pytest tests/test_flat_conversion.py::TestConvertTrainsToFlat -v
```

Expected:
```
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_output_row_count_equals_total_stops
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_train_level_columns_repeated_for_each_stop
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_stop_cancelled_renamed_from_cancelled
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_causes_stored_as_string
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_train_ready_stored_as_string_or_none
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_skips_existing_flat_file
6 passed
```

- [ ] **Step 4: Commit**

```bash
git add src/processors/DataLoader.py tests/test_flat_conversion.py
git commit -m "feat: add convert_trains_to_flat() method to DataLoader"
```

---

### Task 4: Write failing tests for `_save_matched_flat()`

**Files:**
- Modify: `tests/test_flat_conversion.py`

- [ ] **Step 1: Append `TestSaveMatchedFlat` class to the test file**

Add to the bottom of `tests/test_flat_conversion.py`:

```python
STOP_WITH_WEATHER = {
    'stationName': 'Helsinki asema',
    'stationShortCode': 'HKI',
    'stationUICCode': 1,
    'countryCode': 'FI',
    'type': 'DEPARTURE',
    'trainStopping': True,
    'commercialStop': True,
    'commercialTrack': '9',
    'cancelled': False,
    'scheduledTime': '2024-01-01T04:57:00.000Z',
    'actualTime': '2024-01-01T04:57:21.000Z',
    'differenceInMinutes': 0,
    'differenceInMinutes_offset': 0,
    'differenceInMinutes_eachStation_offset': 0,
    'causes': [],
    'trainReady': {'source': 'KUPLA', 'accepted': True, 'timestamp': '2024-01-01T04:51:10.000Z'},
    'weather_observations': {
        'closest_ems': 'Helsinki Kaisaniemi',
        'Air temperature': -14.8,
        'Wind speed': 3.8,
        'Snow depth': 21.0,
    },
}

STOP_WITHOUT_WEATHER = {
    'stationName': 'Tampere asema',
    'stationShortCode': 'TPE',
    'stationUICCode': 160,
    'countryCode': 'FI',
    'type': 'ARRIVAL',
    'trainStopping': True,
    'commercialStop': True,
    'commercialTrack': '2',
    'cancelled': True,
    'scheduledTime': '2024-01-01T06:57:00.000Z',
    'actualTime': None,
    'differenceInMinutes': 5,
    'differenceInMinutes_offset': 5,
    'differenceInMinutes_eachStation_offset': 5,
    'causes': [{'categoryCode': 'X1'}],
    'trainReady': None,
    'weather_observations': {},
}

MATCHED_TRAIN_ROW = {
    **{k: v for k, v in TRAIN_ROW.items() if k != 'timeTableRows'},
    'timeTableRows': str([STOP_WITH_WEATHER, STOP_WITHOUT_WEATHER]),
}


class TestSaveMatchedFlat:

    def test_output_row_count_equals_total_stops(self, loader, tmp_path):
        matched_df = pd.DataFrame([MATCHED_TRAIN_ROW])
        loader._save_matched_flat(matched_df, '2024-01')

        flat_path = os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv')
        assert os.path.exists(flat_path)
        flat_df = pd.read_csv(flat_path)
        assert len(flat_df) == 2

    def test_weather_observations_flattened_to_top_level_columns(self, loader, tmp_path):
        matched_df = pd.DataFrame([MATCHED_TRAIN_ROW])
        loader._save_matched_flat(matched_df, '2024-01')

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv'))
        assert 'closest_ems' in flat_df.columns
        assert 'Air temperature' in flat_df.columns
        assert 'Snow depth' in flat_df.columns
        assert flat_df['Air temperature'].iloc[0] == -14.8
        assert flat_df['Snow depth'].iloc[0] == 21.0

    def test_delay_offset_columns_present(self, loader, tmp_path):
        matched_df = pd.DataFrame([MATCHED_TRAIN_ROW])
        loader._save_matched_flat(matched_df, '2024-01')

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv'))
        assert 'differenceInMinutes_offset' in flat_df.columns
        assert 'differenceInMinutes_eachStation_offset' in flat_df.columns

    def test_stop_with_no_weather_has_nan_weather_columns(self, loader, tmp_path):
        matched_df = pd.DataFrame([MATCHED_TRAIN_ROW])
        loader._save_matched_flat(matched_df, '2024-01')

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv'))
        # STOP_WITHOUT_WEATHER has empty weather_observations — those columns are NaN for row 2
        assert pd.isna(flat_df['Air temperature'].iloc[1])
        assert pd.isna(flat_df['Snow depth'].iloc[1])
```

- [ ] **Step 2: Run new tests to confirm they fail**

```
python -m pytest tests/test_flat_conversion.py::TestSaveMatchedFlat -v
```

Expected: all 4 tests fail with `AttributeError: 'DataLoader' object has no attribute '_save_matched_flat'`

---

### Task 5: Implement `_save_matched_flat()` and wire into `merge_train_weather_data()`

**Files:**
- Modify: `src/processors/DataLoader.py`

- [ ] **Step 1: Add `_save_matched_flat()` method**

Add the following private method to `DataLoader`, immediately after `convert_trains_to_flat()`:

```python
def _save_matched_flat(self, filtered_train_data, month_str):
    """
    Save matched train-weather data in flat format (one row per train stop).

    Flattens timeTableRows and weather_observations to individual columns.
    Saves as matched_data_flat_YYYY_MM.csv alongside the existing matched_data file.
    """
    month_period = pd.Period(month_str, freq='M')
    base = CSV_MATCHED_DATA_FLAT.replace('.csv', '')
    flat_filename = f"{base}_{month_period.year}_{month_period.month:02d}.csv"
    flat_filepath = os.path.join(self.output_folder, flat_filename)

    train_level_cols = [
        'trainNumber', 'departureDate', 'operatorUICCode', 'operatorShortCode',
        'trainType', 'trainCategory', 'commuterLineID', 'runningCurrently',
        'cancelled', 'version', 'timetableType', 'timetableAcceptanceDate',
    ]

    rows = []
    for _, train_row in filtered_train_data.iterrows():
        timetable = train_row['timeTableRows']
        if isinstance(timetable, str):
            try:
                timetable = ast.literal_eval(timetable)
            except (ValueError, SyntaxError) as e:
                print(f"⚠️ Failed to parse timeTableRows for train {train_row.get('trainNumber')}: {e}")
                continue

        if not isinstance(timetable, list):
            continue

        train_base = {col: train_row.get(col) for col in train_level_cols if col in train_row.index}

        for stop in timetable:
            row = dict(train_base)
            row['stationName'] = stop.get('stationName')
            row['stationShortCode'] = stop.get('stationShortCode')
            row['stationUICCode'] = stop.get('stationUICCode')
            row['countryCode'] = stop.get('countryCode')
            row['type'] = stop.get('type')
            row['trainStopping'] = stop.get('trainStopping')
            row['commercialStop'] = stop.get('commercialStop')
            row['commercialTrack'] = stop.get('commercialTrack')
            row['stop_cancelled'] = stop.get('cancelled')
            row['scheduledTime'] = stop.get('scheduledTime')
            row['actualTime'] = stop.get('actualTime')
            row['differenceInMinutes'] = stop.get('differenceInMinutes')
            row['differenceInMinutes_offset'] = stop.get('differenceInMinutes_offset')
            row['differenceInMinutes_eachStation_offset'] = stop.get('differenceInMinutes_eachStation_offset')
            row['causes'] = str(stop.get('causes', []))
            train_ready = stop.get('trainReady')
            row['trainReady'] = str(train_ready) if train_ready is not None else None

            weather = stop.get('weather_observations', {})
            if isinstance(weather, dict):
                row.update(weather)

            rows.append(row)

    flat_df = pd.DataFrame(rows)
    flat_df.to_csv(flat_filepath, index=False)
    print(f"  ✅ Saved flat matched data: {len(rows)} rows to {flat_filename}")
```

- [ ] **Step 2: Add the `_save_matched_flat` call inside `merge_train_weather_data()`**

In `merge_train_weather_data()`, find the block after the existing save call (around line 972):

```python
        # Save the merged data for the specific month
        self.save_monthly_data_to_csv(filtered_train_data, month_str)
```

Immediately after that line (before the `print` that follows), add:

```python
        if FLAT_FORMAT:
            self._save_matched_flat(filtered_train_data, month_str)
```

- [ ] **Step 3: Run all tests — all 10 should pass**

```
python -m pytest tests/test_flat_conversion.py -v
```

Expected:
```
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_output_row_count_equals_total_stops
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_train_level_columns_repeated_for_each_stop
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_stop_cancelled_renamed_from_cancelled
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_causes_stored_as_string
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_train_ready_stored_as_string_or_none
PASSED tests/test_flat_conversion.py::TestConvertTrainsToFlat::test_skips_existing_flat_file
PASSED tests/test_flat_conversion.py::TestSaveMatchedFlat::test_output_row_count_equals_total_stops
PASSED tests/test_flat_conversion.py::TestSaveMatchedFlat::test_weather_observations_flattened_to_top_level_columns
PASSED tests/test_flat_conversion.py::TestSaveMatchedFlat::test_delay_offset_columns_present
PASSED tests/test_flat_conversion.py::TestSaveMatchedFlat::test_stop_with_no_weather_has_nan_weather_columns
10 passed
```

- [ ] **Step 4: Commit**

```bash
git add src/processors/DataLoader.py tests/test_flat_conversion.py
git commit -m "feat: add _save_matched_flat() and wire into merge_train_weather_data()"
```

---

### Task 6: Wire `FLAT_FORMAT` into `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add `FLAT_FORMAT` to the import in `main.py`**

Find the existing import line (line 6):
```python
from config.const import CSV_ALL_TRAINS, CSV_FMI, CSV_FMI_EMS, CSV_TRAIN_CATEGORIES, CSV_TRAIN_CAUSES, CSV_TRAIN_CAUSES_DETAILED, CSV_TRAIN_STATIONS, CSV_TRAIN_THIRD_CAUSES, END_DATE, FMI_BBOX, FOLDER_NAME, START_DATE
```

Replace with:
```python
from config.const import CSV_ALL_TRAINS, CSV_FMI, CSV_FMI_EMS, CSV_TRAIN_CATEGORIES, CSV_TRAIN_CAUSES, CSV_TRAIN_CAUSES_DETAILED, CSV_TRAIN_STATIONS, CSV_TRAIN_THIRD_CAUSES, END_DATE, FMI_BBOX, FLAT_FORMAT, FOLDER_NAME, START_DATE
```

- [ ] **Step 2: Add STEP 1.5 block**

Find the existing STEP 1 / STEP 2 boundary in `main.py`:

```python
        # ============================================================
        # STEP 2: Match train stations with closest EMS weather stations
        # ============================================================
```

Insert the following block immediately before it:

```python
        # ============================================================
        # STEP 1.5: Convert train data to flat format (one row per stop)
        # ============================================================
        if FLAT_FORMAT:
            print("\n" + "="*60)
            print("STEP 1.5: Converting Train Data to Flat Format")
            print("="*60)
            data_loader.convert_trains_to_flat()

```

- [ ] **Step 3: Verify the pipeline imports cleanly**

```
python -c "import main" 2>&1 | head -5
```

Expected: No import errors (the script won't run because `DATA_FETCH = False` and `DataLoader` needs real files, but the import itself should be clean).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add STEP 1.5 flat format conversion to main.py pipeline"
```

---

## Done

With `FLAT_FORMAT = True` in `config/const.py`:
- Running the pipeline produces `all_trains_data_flat_YYYY_MM.csv` (one row per stop) for each month.
- After `merge_train_weather_data()`, `matched_data_flat_YYYY_MM.csv` is also saved with all weather columns as top-level columns.
- Original files are unchanged.
- Re-running is safe — flat train files are skipped if they already exist.
