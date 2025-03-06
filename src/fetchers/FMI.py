import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from fmiopendata.wfs import download_stored_query

from config.const import FMI_OBSERVATIONS, FMI_EMS, CSV_FMI, CSV_FMI_EMS, FOLDER_NAME

class FMIDataFetcher:
    def __init__(self):
        """
        Initializes the FMIDataFetcher class.
        """
        self.base_url = FMI_OBSERVATIONS
        self.ems_url = FMI_EMS
        self.output_folder = FOLDER_NAME

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