# Design: Flat Train Stop Format

**Date:** 2026-05-12  
**Status:** Approved

## Problem

`all_trains_data_YYYY_MM.csv` and `matched_data_YYYY_MM.csv` store each train as a single row, with all timetable stops packed into a `timeTableRows` column as a serialized Python list of dicts. Re-using this data requires `ast.literal_eval` and manual unnesting on every load — slow and error-prone at scale.

## Goal

Introduce an opt-in pipeline step that produces flat versions of both files (one row per train stop), saved alongside the originals. A new flag controls whether this step runs and whether matched data is also saved flat.

---

## New Constants (`config/const.py`)

```python
FLAT_FORMAT = False  # Set True to enable flat conversion step and flat matched-data output

CSV_ALL_TRAINS_FLAT = "all_trains_data_flat.csv"
CSV_MATCHED_DATA_FLAT = "matched_data_flat.csv"
```

---

## Pipeline Change (`main.py`)

Only applies when `DATA_FETCH = False`.

```
STEP 1   — FMI rolling window preprocessing          (unchanged)
STEP 1.5 — Convert all_trains_data files to flat     (NEW, runs only when FLAT_FORMAT=True)
STEP 2   — Match train stations with EMS             (unchanged)
STEP 3   — Load/merge train-weather data by month    (saves flat matched_data when FLAT_FORMAT=True)
```

---

## Flat Schema

### `all_trains_data_flat_YYYY_MM.csv`

Each row = one stop of one train. Train-level columns are repeated for every stop.

**Train-level columns (repeated):**

| Column | Source |
|--------|--------|
| trainNumber | train row |
| departureDate | train row |
| operatorUICCode | train row |
| operatorShortCode | train row |
| trainType | train row |
| trainCategory | train row |
| commuterLineID | train row |
| runningCurrently | train row |
| cancelled | train row (train-level) |
| version | train row |
| timetableType | train row |
| timetableAcceptanceDate | train row |

**Stop-level columns:**

| Column | Source | Notes |
|--------|--------|-------|
| stationName | stop dict | |
| stationShortCode | stop dict | |
| stationUICCode | stop dict | |
| countryCode | stop dict | |
| type | stop dict | DEPARTURE or ARRIVAL |
| trainStopping | stop dict | |
| commercialStop | stop dict | |
| commercialTrack | stop dict | |
| stop_cancelled | stop dict `cancelled` key | renamed to avoid collision with train-level `cancelled` |
| scheduledTime | stop dict | |
| actualTime | stop dict | may be None |
| differenceInMinutes | stop dict | may be None |
| causes | stop dict | kept as `str()` (list) |
| trainReady | stop dict | kept as `str()` (dict or None); only present on DEPARTURE stops, NaN elsewhere |

### `matched_data_flat_YYYY_MM.csv`

All columns from `all_trains_data_flat` plus:

**Additional delay columns (added during matching):**

| Column |
|--------|
| differenceInMinutes_offset |
| differenceInMinutes_eachStation_offset |

**Flattened weather columns (from `weather_observations` dict, ~80 columns):**

| Column | Notes |
|--------|-------|
| closest_ems | EMS station name used |
| Air temperature | instant value |
| Wind speed | |
| Gust speed | |
| Snow depth | |
| ... | all other FMI observation fields |
| Air temperature (12h max) | rolling window features |
| Air temperature (12h min) | |
| ... | all rolling window columns |

Weather columns use their original names from the `weather_observations` dict. If a value is missing (NaN), it stays NaN in the flat file.

---

## New / Modified Code

### New method: `DataLoader.convert_trains_to_flat()`

- Iterates over `self.train_files` (all `all_trains_data_YYYY_MM.csv` files).
- For each file, checks whether the flat equivalent already exists — skips if so.
- Parses `timeTableRows` via `ast.literal_eval`.
- Explodes stops into rows, applying the schema above.
- Saves as `all_trains_data_flat_YYYY_MM.csv` via a new `save_monthly_flat_trains_to_csv()` helper (mirrors the existing `save_monthly_data_to_csv` pattern).

### New private method: `DataLoader._save_matched_flat(filtered_train_data, month_str)`

- Called from `merge_train_weather_data()` when `FLAT_FORMAT=True`.
- Iterates over `filtered_train_data` rows, parses each `timeTableRows` (already enriched with weather and offset columns).
- Flattens each stop's `weather_observations` dict to top-level columns.
- Saves as `matched_data_flat_YYYY_MM.csv`.

### Modified: `DataLoader.merge_train_weather_data()`

- After existing save call (`self.save_monthly_data_to_csv`), adds:
  ```python
  if FLAT_FORMAT:
      self._save_matched_flat(filtered_train_data, month_str)
  ```
- No other changes. Delay tracking logic is untouched.

---

## Skip-if-exists Behaviour

`convert_trains_to_flat()` checks for the flat output file before processing each month. If it exists, that month is skipped. This mirrors the rolling-features preprocessing pattern and makes reruns safe.

`_save_matched_flat()` always overwrites (same as the existing `save_monthly_data_to_csv` behaviour).

---

## Out of Scope

- No changes to FMI weather files.
- No changes to metadata files (`metadata_*.csv`).
- No changes to delay summary tables.
- No changes to the fetching pipeline (`DATA_FETCH=True` path).
- The original `all_trains_data_YYYY_MM.csv` and `matched_data_YYYY_MM.csv` files are never modified or deleted.
