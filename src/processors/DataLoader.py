import os
import re
import pandas as pd
from glob import glob
from config.const import CSV_ALL_TRAINS, CSV_FMI, FOLDER_NAME

class DataLoader:
    def __init__(self):
        self.data_folder = FOLDER_NAME
        self.train_data = None
        self.weather_data = None

        self._check_data_folder()

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Data folder '{self.data_folder}' does not exist.")

        # Find files matching the patterns
        train_files = glob(os.path.join(self.data_folder, f"{CSV_ALL_TRAINS[:-4]}*.csv"))
        weather_files = glob(os.path.join(self.data_folder, f"{CSV_FMI[:-4]}*.csv"))

        if not train_files:
            raise FileNotFoundError(f"No train data files matching '{CSV_ALL_TRAINS}' found in the data folder.")

        if not weather_files:
            raise FileNotFoundError(f"No weather data files matching '{CSV_FMI}' found in the data folder.")

        print(f"Found {len(train_files)} train data files.")
        print(f"Found {len(weather_files)} weather data files.")

        # Extract year and month from file names using regex
        train_dates = self._extract_dates_from_filenames(train_files)
        weather_dates = self._extract_dates_from_filenames(weather_files)

        if not train_dates:
            raise ValueError("No valid dates found in train file names.")
        if not weather_dates:
            raise ValueError("No valid dates found in weather file names.")

        train_start_date = min(train_dates)
        train_end_date = max(train_dates)

        weather_start_date = min(weather_dates)
        weather_end_date = max(weather_dates)

        print(f"\nTrain Data Date Range: {train_start_date} to {train_end_date}")
        print(f"Weather Data Date Range: {weather_start_date} to {weather_end_date}")

        # Check if date ranges match
        if train_start_date != weather_start_date or train_end_date != weather_end_date:
            raise ValueError("Mismatch in date ranges between train and weather data.")

        print("\n✅ Data files detected successfully and date ranges match.")

    def _extract_dates_from_filenames(self, files):
        date_pattern = re.compile(r'(\d{4})_(\d{2})')  # Pattern to match YYYY_MM in file names
        dates = []
        for file in files:
            match = date_pattern.search(file)
            if match:
                year, month = match.groups()
                dates.append(f"{year}-{month}")
        return dates