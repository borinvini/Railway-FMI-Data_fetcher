from src.processors.DataLoader import DataLoader
from src.fetchers.Railway import RailwayDataFetcher
from src.fetchers.FMI import FMIDataFetcher

from config.const import CSV_ALL_TRAINS, CSV_FMI, CSV_FMI_EMS, CSV_TRAIN_STATIONS, END_DATE, FMI_BBOX, START_DATE

# Flag to control data collection
DATA_FETCH = False

if DATA_FETCH:
    railway_fetcher = RailwayDataFetcher()
    fmi_fetcher = FMIDataFetcher()

    # Fetch station metadata
    stations_metadata = railway_fetcher.fetch_stations_metadata()

    # Save station metadata to a CSV file
    railway_fetcher.save_to_csv(stations_metadata, CSV_TRAIN_STATIONS)

    # Fetch train data for a specific interval
    railway_fetcher.fetch_trains_by_interval(START_DATE, END_DATE, stations_metadata)

    # Save train data to a CSV file
    # railway_fetcher.save_to_csv(train_data_df, CSV_ALL_TRAINS)

    # Fetch data for the specified date range
    ems_data = fmi_fetcher.fetch_fmi_by_interval(FMI_BBOX, START_DATE, END_DATE)

    # Save data to CSV using the class method
    # fmi_fetcher.save_to_csv(fmi_data, CSV_FMI)
    fmi_fetcher.save_to_csv(ems_data, CSV_FMI_EMS)
else:
    # Implement alternative logic here
    data_loader = DataLoader()
