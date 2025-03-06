import ast
from datetime import datetime, timedelta
import requests
import pandas as pd
import os

from config.const import FIN_RAILWAY_ALL_TRAINS, FIN_RAILWAY_BASE_URL, FIN_RAILWAY_STATIONS, FOLDER_NAME

class RailwayDataFetcher:
    """Class to fetch and process railway data from Digitraffic API."""

    def __init__(self, base_url=FIN_RAILWAY_BASE_URL):
        self.base_url = base_url
        self.output_folder = FOLDER_NAME

        # Ensure the output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def save_to_csv(self, df, filename):
        """
        Save a DataFrame to a CSV file inside the FOLDER_NAME directory.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            filename (str): Name of the CSV file.
        """
        if df is not None and not df.empty:
            filepath = os.path.join(self.output_folder, filename)
            df.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
        else:
            print("No data to save.")

    def get_data(self, endpoint, params=None):
        """
        Fetch data from the API endpoint with optional parameters.

        Args:
            endpoint (str): The API endpoint path.
            params (dict, optional): Query parameters for the request.

        Returns:
            dict: JSON response from the API or None if an error occurs.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during request: {e}")
            return None

    def fetch_stations_metadata(self):
        """
        Fetch and return railway station metadata as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing station metadata.
        """
        print(f"Fetching station metadata from {self.base_url}{FIN_RAILWAY_STATIONS}...")
        data = self.get_data(FIN_RAILWAY_STATIONS)

        if not data:
            print("No station metadata available. Please check the API or data source.")
            return pd.DataFrame()

        # Convert to DataFrame and keep only relevant columns
        df = pd.DataFrame(data)
        if df.empty:
            print("Fetched station metadata is empty.")
            return pd.DataFrame()

        print("Station metadata successfully loaded.")
        return df
    
    def fetch_trains_by_interval(self, start_date, end_date, stations_metadata):
        """
        Fetch train data for a given date range and return it as a DataFrame with enriched timetable data.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            stations_metadata (pd.DataFrame): DataFrame containing station metadata.

        Returns:
            pd.DataFrame: DataFrame containing all trains for the given interval.
        """
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Use 'YYYY-MM-DD'.")
            return pd.DataFrame()

        current_date = start_date
        all_train_data = []

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Fetching train data for {date_str}...")
            endpoint = f"{FIN_RAILWAY_ALL_TRAINS}/{date_str}"
            daily_data = self.get_data(endpoint)

            if daily_data:
                all_train_data.extend(daily_data)

            current_date += timedelta(days=1)

        if not all_train_data:
            print(f"No train data found for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()

        # Combine all daily data into a single DataFrame
        trains_data = pd.DataFrame(all_train_data)
        print(f"Fetched a total of {len(trains_data)} trains from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

        # Enrich `timeTableRows` with station names, processing month by month
        if "timeTableRows" in trains_data.columns:
            trains_data["departureMonth"] = pd.to_datetime(trains_data["departureDate"]).dt.to_period("M")
            months = trains_data["departureMonth"].unique()

            enriched_data = []

            for month in months:
                print(f"Processing data for: {month}. Please wait...")
                month_data = trains_data[trains_data["departureMonth"] == month].copy()

                # Apply enrichment using the separate method
                month_data["timeTableRows"] = month_data["timeTableRows"].apply(
                    lambda row: self._enrich_timetable_row(row, stations_metadata)
                )

                enriched_data.append(month_data)

            print("Data processing is complete.")
            # Combine enriched data
            trains_data = pd.concat(enriched_data, ignore_index=True)
        else:
            print("No 'timeTableRows' column found in the trains_data DataFrame.")

        return trains_data

    def _enrich_timetable_row(self, row, stations_metadata):
        """
        Adds station names to timeTableRows based on stationShortCode.

        Args:
            row (list or str): The raw timeTableRows data.
            stations_metadata (pd.DataFrame): DataFrame containing station metadata.

        Returns:
            list: Enriched timeTableRows with station names.
        """
        try:
            parsed_row = ast.literal_eval(row) if isinstance(row, str) else row
            if isinstance(parsed_row, list):  # Ensure it's a list of dictionaries
                enriched_rows = []
                for entry in parsed_row:
                    station_name = stations_metadata.loc[
                        stations_metadata["stationShortCode"] == entry["stationShortCode"], "stationName"
                    ]
                    station_name_value = station_name.iloc[0] if not station_name.empty else None
                    enriched_entry = {"stationName": station_name_value}
                    enriched_entry.update(entry)  # Add remaining keys/values
                    enriched_rows.append(enriched_entry)
                return enriched_rows
            return parsed_row
        except Exception as e:
            print(f"Error processing timeTableRows: {e}")
            return row  # Return the original row in case of an error

