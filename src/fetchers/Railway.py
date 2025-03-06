import requests
import pandas as pd
import os

from config.const import FIN_RAILWAY_BASE_URL, FIN_RAILWAY_STATIONS, FOLDER_NAME

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

