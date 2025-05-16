from src.processors.DataLoader import DataLoader
from src.fetchers.Railway import RailwayDataFetcher
from src.fetchers.FMI import FMIDataFetcher

from config.const import CSV_ALL_TRAINS, CSV_FMI, CSV_FMI_EMS, CSV_TRAIN_CATEGORIES, CSV_TRAIN_CAUSES, CSV_TRAIN_STATIONS, CSV_TRAIN_THIRD_CAUSES, END_DATE, FMI_BBOX, START_DATE

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

    # Fetch detailed cause category codes metadata
    detailed_cause_codes = railway_fetcher.fetch_detailed_cause_category_codes_metadata()
    railway_fetcher.save_to_csv(detailed_cause_codes, CSV_TRAIN_CAUSES)

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

        # Call the match method only if the class creation is successful
        merged_data = data_loader.match_train_with_ems()
        print(merged_data.head())

        data_loader.load_csv_files_by_month()

    except Exception as e:
        print(f"\n❌ Error: {e}")