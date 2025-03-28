import ast
from datetime import datetime, timedelta
import requests
import pandas as pd
import os

from config.const import CSV_ALL_TRAINS, FIN_RAILWAY_ALL_TRAINS, FIN_RAILWAY_BASE_URL, FIN_RAILWAY_STATIONS, FOLDER_NAME

class RailwayDataFetcher:
    """Class to fetch and process railway data from Digitraffic API."""

    def __init__(self):
        self.base_url = FIN_RAILWAY_BASE_URL
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


    def save_monthly_data_to_csv(self, df, month_str):
        """
        Save the train data for a specific month to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame containing train data for the month.
            month_str (str): The month in 'YYYY-MM' format.
        """
        # Convert the string 'YYYY-MM' into a Period object
        month_period = pd.Period(month_str, freq='M')

        # Get base filename from CSV_ALL_TRAINS and remove extension if it exists
        base_filename = CSV_ALL_TRAINS.replace('.csv', '')

        # Create filename using base name and month
        filename = f"{base_filename}_{month_period.year}_{month_period.month:02d}.csv"
        filepath = os.path.join(self.output_folder, filename)

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"✅ Data for {month_str} saved to {filepath}")




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
        self.preview_dataframe(df, "📍 Station Metadata Preview")
        return df
    
    def fetch_trains_by_interval(self, start_date, end_date, stations_metadata):
        """
        Fetch train data for a given date range and save it month by month while fetching.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            stations_metadata (pd.DataFrame): DataFrame containing station metadata.
        """
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Use 'YYYY-MM-DD'.")
            return

        current_date = start_date
        current_month = start_date.strftime("%Y-%m")  # Track the current processing month
        monthly_data = []

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Fetching train data for {date_str}...")
            endpoint = f"{FIN_RAILWAY_ALL_TRAINS}/{date_str}"
            daily_data = self.get_data(endpoint)

            if daily_data:
                monthly_data.extend(daily_data)

            next_date = current_date + timedelta(days=1)

            # If the next date belongs to a new month, process and save the current month's data
            if next_date.strftime("%Y-%m") != current_month or next_date > end_date:
                if monthly_data:
                    print(f"Processing and saving data for {current_month}. Please wait...")
                    
                    # Convert to DataFrame
                    month_df = pd.DataFrame(monthly_data)

                    # Enrich timetable rows with station names
                    if "timeTableRows" in month_df.columns:
                        month_df["timeTableRows"] = month_df["timeTableRows"].apply(
                            lambda row: self._enrich_timetable_row(row, stations_metadata)
                        )

                    # Save month-wise data
                    self.save_monthly_data_to_csv(month_df, current_month)

                    # Clear memory
                    del month_df
                    monthly_data = []

                # Update current processing month
                current_month = next_date.strftime("%Y-%m")

            # Move to the next day
            current_date = next_date

        print("Data processing and saving is complete.")


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


    def preview_dataframe(self, df, title, num_rows=10):
        """
        Prints a preview of the DataFrame including the header and first few rows.

        Parameters:
        df (pd.DataFrame): The DataFrame to preview.
        title (str): Title to display before the preview.
        num_rows (int): Number of rows to display (default is 5).
        """
        print(f"\n{title}:\n")
        print(df.head(num_rows).to_string(index=False))
