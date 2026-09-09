import os
import time
from src.processors.DataLoader import DataLoader
from src.fetchers.Railway import RailwayDataFetcher
from src.fetchers.FMI import FMIDataFetcher, reconcile_station_metadata
from src.fetchers.FMIStations import FMIStationRegistry

from config.const import CSV_ALL_TRAINS, CSV_FMI, CSV_FMI_EF_REGISTRY, CSV_FMI_EMS, CSV_TRAIN_CATEGORIES, CSV_TRAIN_CAUSES, CSV_TRAIN_CAUSES_DETAILED, CSV_TRAIN_STATIONS, CSV_TRAIN_THIRD_CAUSES, END_DATE, FMI_BBOX, FOLDER_NAME, PARQUET_ALL_TRAINS_FLAT, PARQUET_FMI, PARQUET_MATCHED_DATA_FLAT, START_DATE


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as Hh Mm S.SSs, dropping empty leading units."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"
    if minutes:
        return f"{int(minutes)}m {secs:.2f}s"
    return f"{secs:.2f}s"


program_start = time.perf_counter()

# Create data folder if it doesn't exist
os.makedirs(FOLDER_NAME, exist_ok=True)
print(f"✅ Data folder '{FOLDER_NAME}' is ready.")

# Flag to control data collection
DATA_FETCH = True
FLAT_FORMAT = True  # Set True to produce all_trains_data_flat_*.csv (one row per stop)
PARQUET_FORMAT = True  # Set True to convert monthly CSV files to .parquet

if DATA_FETCH:
    fetch_start = time.perf_counter()

    railway_fetcher = RailwayDataFetcher()
    fmi_fetcher = FMIDataFetcher()

    # Fetch station metadata
    stations_metadata = railway_fetcher.fetch_stations_metadata()
    railway_fetcher.save_to_csv(stations_metadata, CSV_TRAIN_STATIONS)

    # Fetch train categories metadata
    train_categories = railway_fetcher.fetch_train_categories_metadata()
    railway_fetcher.save_to_csv(train_categories, CSV_TRAIN_CATEGORIES)

    # Fetch cause category codes metadata
    cause_codes = railway_fetcher.fetch_cause_category_codes_metadata()
    railway_fetcher.save_to_csv(cause_codes, CSV_TRAIN_CAUSES)

    # Fetch detailed cause category codes metadata
    detailed_cause_codes = railway_fetcher.fetch_detailed_cause_category_codes_metadata()
    railway_fetcher.save_to_csv(detailed_cause_codes, CSV_TRAIN_CAUSES_DETAILED)

    # Fetch third cause category codes metadata
    third_cause_codes = railway_fetcher.fetch_third_cause_category_codes_metadata()
    railway_fetcher.save_to_csv(third_cause_codes, CSV_TRAIN_THIRD_CAUSES)

    # Fetch train data for a specific interval
    railway_fetcher.fetch_trains_by_interval(START_DATE, END_DATE, stations_metadata)

    # Fetch the EF station catalogue first, so a registry outage surfaces before
    # the multi-hour observation download rather than after it.
    ef_registry = FMIStationRegistry().fetch_registry()
    if not ef_registry.empty:
        fmi_fetcher.save_to_csv(ef_registry, CSV_FMI_EF_REGISTRY)

    # Fetch data for the specified date range
    observed_stations = fmi_fetcher.fetch_fmi_by_interval(FMI_BBOX, START_DATE, END_DATE)
    ems_data = reconcile_station_metadata(observed_stations, ef_registry)
    fmi_fetcher.save_station_metadata(ems_data, CSV_FMI_EMS)

    fetch_elapsed = time.perf_counter() - fetch_start
    print(f"\n⏱️  Fetch time: {format_duration(fetch_elapsed)}")
else:
    try:
        data_loader = DataLoader()
        print("\n✅ DataLoader initialized successfully.")

        # ============================================================
        # STEP 1: Preprocess FMI weather data to add rolling window features
        # ============================================================
        # This step adds rolling window statistics (max, min, mean, cumulative)
        # for multiple weather parameters across 12h, 24h, and 72h windows.
        # Precipitation amount only gets mean and cumulative (no min/max).
        # ============================================================
        print("\n" + "="*60)
        print("STEP 1: Preprocessing FMI Rolling Window Features")
        print("="*60)
        data_loader.preprocess_fmi_rolling_features()

        # ============================================================
        # STEP 1.5: Convert train data to flat format (one row per stop)
        # ============================================================
        if FLAT_FORMAT:
            data_loader.convert_trains_to_flat()

        # ============================================================
        # STEP 2: Match train stations with closest EMS weather stations
        # ============================================================
        print("\n" + "="*60)
        print("STEP 2: Matching Train Stations with EMS Weather Stations")
        print("="*60)
        match_start = time.perf_counter()
        merged_data = data_loader.match_train_with_ems()
        match_elapsed = time.perf_counter() - match_start
        print(merged_data.head())
        print(f"⏱️  Match time: {format_duration(match_elapsed)}")

        # ============================================================
        # STEP 3: Load and merge train-weather data by month
        # ============================================================
        print("\n" + "="*60)
        print("STEP 3: Loading and Merging Train-Weather Data by Month")
        print("="*60)
        data_loader.load_csv_files_by_month()

        # ============================================================
        # STEP 3.5: Convert matched data to flat format
        # ============================================================
        if FLAT_FORMAT:
            data_loader.convert_matched_to_flat()

        # ============================================================
        # STEP 4: Convert monthly CSV files to Parquet
        # ============================================================
        if PARQUET_FORMAT:
            data_loader.convert_to_parquet()

        print("\n" + "="*60)
        print("✅ ALL PROCESSING COMPLETE!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")

total_elapsed = time.perf_counter() - program_start
print(f"\n⏱️  Total time: {format_duration(total_elapsed)}")