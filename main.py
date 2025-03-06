from src.fetchers.Railway import RailwayDataFetcher
from src.fetchers.FMI import FMIDataFetcher

from config.const import CSV_ALL_TRAINS, CSV_TRAIN_STATIONS, END_DATE, FMI_BBOX, START_DATE


railway_fetcher = RailwayDataFetcher()
fmi_fetcher = FMIDataFetcher()

# Fetch station metadata
stations_metadata = railway_fetcher.fetch_stations_metadata()

# Save station metadata to a CSV file
railway_fetcher.save_to_csv(stations_metadata, CSV_TRAIN_STATIONS)

# Fetch train data for a specific interval
train_data_df = railway_fetcher.fetch_trains_by_interval(START_DATE, END_DATE, stations_metadata)

# Save train data to a CSV file
railway_fetcher.save_to_csv(train_data_df, CSV_ALL_TRAINS)

# Fetch data for the specified date range
#fmi_data = fmi_fetcher.fetch_fmi_by_interval(FMI_BBOX, START_DATE, END_DATE)