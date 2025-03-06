from src.fetchers.Railway import RailwayDataFetcher
from config.const import CSV_ALL_TRAINS, CSV_TRAIN_STATIONS, END_DATE, START_DATE


railway_fetcher = RailwayDataFetcher()

# Fetch station metadata
stations_metadata = railway_fetcher.fetch_stations_metadata()

# Save station metadata to a CSV file
railway_fetcher.save_to_csv(stations_metadata, CSV_TRAIN_STATIONS)

# Fetch train data for a specific interval
train_data_df = railway_fetcher.fetch_trains_by_interval(START_DATE, END_DATE, stations_metadata)

# Save train data to a CSV file
railway_fetcher.save_to_csv(train_data_df, CSV_ALL_TRAINS)