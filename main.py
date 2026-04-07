import os
from src.processors.DataLoader import DataLoader
from src.fetchers.Railway import RailwayDataFetcher
from src.fetchers.FMI import FMIDataFetcher

from config.const import CSV_ALL_TRAINS, CSV_FMI, CSV_FMI_EMS, CSV_TRAIN_CATEGORIES, CSV_TRAIN_CAUSES, CSV_TRAIN_CAUSES_DETAILED, CSV_TRAIN_STATIONS, CSV_TRAIN_THIRD_CAUSES, END_DATE, FMI_BBOX, FOLDER_NAME, START_DATE

# Create data folder if it doesn't exist
os.makedirs(FOLDER_NAME, exist_ok=True)
print(f"✅ Data folder '{FOLDER_NAME}' is ready.")

# Flag to control data collection
DATA_FETCH = False

if DATA_FETCH:
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

    # Fetch data for the specified date range
    ems_data = fmi_fetcher.fetch_fmi_by_interval(FMI_BBOX, START_DATE, END_DATE)
    fmi_fetcher.save_to_csv(ems_data, CSV_FMI_EMS)
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
        # STEP 2: Match train stations with closest EMS weather stations
        # ============================================================
        print("\n" + "="*60)
        print("STEP 2: Matching Train Stations with EMS Weather Stations")
        print("="*60)
        merged_data = data_loader.match_train_with_ems()
        print(merged_data.head())

        # ============================================================
        # STEP 3: Load and merge train-weather data by month
        # ============================================================
        print("\n" + "="*60)
        print("STEP 3: Loading and Merging Train-Weather Data by Month")
        print("="*60)
        data_loader.load_csv_files_by_month()

        print("\n" + "="*60)
        print("✅ ALL PROCESSING COMPLETE!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")