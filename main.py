from src.fetchers.Railway import RailwayDataFetcher
from config.const import CSV_TRAIN_STATIONS


railway_fetcher = RailwayDataFetcher()

# Fetch station metadata
stations_metadata = railway_fetcher.fetch_stations_metadata()

# Save station metadata to a CSV file
railway_fetcher.save_to_csv(stations_metadata, CSV_TRAIN_STATIONS)