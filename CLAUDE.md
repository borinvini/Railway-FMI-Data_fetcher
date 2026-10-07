# CLAUDE.md

Python pipeline that fetches Finnish railway timetables (Digitraffic) and FMI weather observations, matches each train station to nearby FMI weather stations (EMS) by haversine distance, and writes monthly train+weather files for delay research. Sibling projects in the same parent folder: `Railway-FMI-Data_viewer` (Streamlit viewer) and `Railway-FMI-Data_training-CSC` (model training on this data, run on CSC).

## Running
- Conda env `venv_rail_fmi` (`environment.yml`, Python 3.12; the file is UTF-16 encoded, so keep that encoding if you edit it).
- Run: `C:\Users\vinic\miniconda3\envs\venv_rail_fmi\python.exe -u -X utf8 main.py`
  - `-X utf8` is required: the console prints emoji and otherwise crashes on Windows. `-u` gives live logs.
- Tests: `C:\Users\vinic\miniconda3\envs\venv_rail_fmi\python.exe -X utf8 -m pytest tests` (pytest is installed in the env; tests mock the network and use `tmp_path`).
- There is no CLI. Behaviour is set by flags at the top of `main.py` (`DATA_FETCH`, `FLAT_FORMAT`, `PARQUET_FORMAT`) and by parameters in `config/const.py` (`START_DATE`/`END_DATE`, category/route filters, rolling windows, EMS radius/top-N).

## Pipeline
`DATA_FETCH = True` (network, takes hours):
1. `RailwayDataFetcher` (`src/fetchers/Railway.py`) gets station/category/cause metadata and monthly `all_trains_data_YYYY_MM.csv`. `warn_station_drift` in `main.py` warns when stations moved >100 m since the last download.
2. `FMIStationRegistry` (`src/fetchers/FMIStations.py`) gets the EF station catalogue first, so a registry outage shows up before the long download.
3. `FMIDataFetcher` (`src/fetchers/FMI.py`) gets monthly `fmi_weather_observations_YYYY_MM.csv`, then `reconcile_station_metadata` builds `metadata_fmi_ems_stations.csv`.

`DATA_FETCH = False` (local processing, `DataLoader` in `src/processors/DataLoader.py`):
1. `preprocess_fmi_rolling_features`: 12/24/72 h rolling stats added to the weather CSVs.
2. `convert_trains_to_flat` (one row per stop).
3. `match_train_with_ems`: closest EMS + top-10 candidates within `ALTERNATIVE_WEATHER_RADIUS_KM`.
4. `load_csv_files_by_month` → `merge_train_weather_data`: `matched_data_YYYY_MM.csv` + delay tables.
5. `convert_matched_to_flat`, then `convert_to_parquet`.

`DataLoader.__init__` raises if train and weather month ranges in `data/` differ. Steps skip outputs that already exist, so delete or move old outputs before rebuilding.

## APIs
Defined in `config/const.py`; change them there only.
- Digitraffic railway, base `https://rata.digitraffic.fi/api/v1`:
  - `/metadata/stations`
  - `/metadata/train-categories`
  - `/metadata/cause-category-codes`
  - `/metadata/detailed-cause-category-codes`
  - `/metadata/third-cause-category-codes`
  - `/trains` (timetables, fetched per date interval)
  - `/train-tracking`
- FMI open data WFS, base `https://opendata.fmi.fi/wfs`, selected by stored query:
  - `fmi::observations::weather::multipointcoverage` (weather observations, queried with the `FMI_BBOX` bounding box)
  - `fmi::ef::stations` (EF station registry)

## Data
- Everything is read from and written to `data/` (git-ignored, as are `*.parquet`, `*.json` and `docs/`).
- The full archive lives outside the repo in `../Railway-FMI-Data-CSV-Files-v2`. Rebuilds are done in batches: copy a period's inputs into `data/`, run, check, copy outputs back.
- Step 4 of processing (monthly merge) uses about 7 GB RAM and takes about 1h20m per 3 months.

## Invariants (each was a real bug; tests in `tests/` pin them)
- Matched flat files are written with one fixed pyarrow schema (`DataLoader._MATCHED_FLAT_HEAD_COLS` + weather cols + `_MATCHED_FLAT_TAIL_COLS`, via `_conform_matched_flat`). Add new columns there, never ad hoc, or chunked writes drift columns.
- FMI requests must not overlap: start/end are inclusive, and overlapping hour marks doubled precipitation sums. Deduplicate on `FMI_OBSERVATION_KEY`.
- Each parameter's rolling windows come from a single weather station (no mixing stations across a window).
- The EF registry (441 facilities, about 254 weather) only enriches observed stations: LEFT JOIN, never a replacement. Only `FMI_WEATHER_NETWORKS` count as weather sources.
- "Unnamed: N" columns in FMI responses are real data with lost parameter names; the fetcher retries those responses.
- Column names for rolling features always come from `get_fmi_rolling_column_names` in `config/const.py`.

## Conventions
- Constants, paths and API URLs live in `config/const.py`; import from there, don't hardcode.
- Run modes are plain flags near the top of `main.py` (edit the file, there is no CLI):
  ```python
  # Flag to control data collection
  DATA_FETCH = False
  FLAT_FORMAT = True  # Set True to produce all_trains_data_flat_*.csv (one row per stop)
  PARQUET_FORMAT = True  # Set True to convert monthly CSV files to .parquet
  ```
  `DATA_FETCH = True` downloads from the APIs, `False` processes the local files in `data/`. Check the current values before running, since a wrong `DATA_FETCH` either starts a multi-hour download or skips fetching.
- Console output uses emoji status prefixes (✅ ⚠️ ❌ ⏱️), matching the existing code.
- SMTP credentials come from `.env` (see `.env.example`), never from code.
- Commit messages are Conventional Commits (`fix:`, `feat:`, `test:`, `chore:`, `build:`), describing the effect in plain words.
- Bug fixes come with a regression test in `tests/test_<topic>.py` whose module docstring explains the bug.
