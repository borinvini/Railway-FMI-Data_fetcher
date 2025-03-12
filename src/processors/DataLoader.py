from datetime import datetime
import json
import os
import re
import numpy as np
import pandas as pd
from glob import glob
from haversine import haversine, Unit
from config.const import CSV_ALL_TRAINS, CSV_CLOSEST_EMS_TRAIN, CSV_FMI, CSV_FMI_EMS, CSV_MATCHED_DATA, CSV_TRAIN_STATIONS, FOLDER_NAME

class DataLoader:
    def __init__(self):
        self.data_folder = FOLDER_NAME
        self.output_folder = FOLDER_NAME
        self.train_files = None
        self.weather_files = None
        self.merged_metadata = None

        self._check_data_folder()

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Data folder '{self.data_folder}' does not exist.")

        # Find files matching the patterns
        self.train_files = glob(os.path.join(self.data_folder, f"{CSV_ALL_TRAINS[:-4]}*.csv"))
        self.weather_files = glob(os.path.join(self.data_folder, f"{CSV_FMI[:-4]}*.csv"))

        if not self.train_files:
            raise FileNotFoundError(f"No train data files matching '{CSV_ALL_TRAINS}' found in the data folder.")

        if not self.weather_files:
            raise FileNotFoundError(f"No weather data files matching '{CSV_FMI}' found in the data folder.")

        print(f"Found {len(self.train_files)} train data files.")
        print(f"Found {len(self.weather_files)} weather data files.")

        # Extract year and month from file names using regex
        train_dates = self._extract_dates_from_filenames(self.train_files)
        weather_dates = self._extract_dates_from_filenames(self.weather_files)

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
        base_filename = CSV_MATCHED_DATA.replace('.csv', '')

        # Create filename using base name and month
        filename = f"{base_filename}_{month_period.year}_{month_period.month:02d}.csv"
        filepath = os.path.join(self.output_folder, filename)

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"✅ Data for {month_str} saved to {filepath}")

    def _extract_dates_from_filenames(self, files):
        date_pattern = re.compile(r'(\d{4})_(\d{2})')  # Pattern to match YYYY_MM in file names
        dates = []
        for file in files:
            match = date_pattern.search(file)
            if match:
                year, month = match.groups()
                dates.append(f"{year}-{month}")
        return dates
    
    def _find_closest_ems(self, train_lat, train_long, ems_stations):
        """
        Finds the closest EMS station based on Haversine distance.

        Parameters:
            train_lat (float): Latitude of the train station.
            train_long (float): Longitude of the train station.
            ems_stations (pd.DataFrame): DataFrame containing EMS station data.

        Returns:
            tuple: (EMS station name, EMS latitude, EMS longitude, distance in km)
        """
        min_distance = float("inf")
        closest_ems = None
        closest_lat = None
        closest_long = None

        for _, ems in ems_stations.iterrows():
            ems_coords = (ems["latitude"], ems["longitude"])
            train_coords = (train_lat, train_long)

            # Compute Haversine distance
            distance = haversine(train_coords, ems_coords, unit=Unit.KILOMETERS)

            if distance < min_distance:
                min_distance = distance
                closest_ems = ems["station_name"]
                closest_lat = ems["latitude"]
                closest_long = ems["longitude"]

        return closest_ems, closest_lat, closest_long, min_distance

    def match_train_with_ems(self) -> pd.DataFrame:
        """
        Matches each train station with the closest EMS station using the Haversine formula.

        Returns:
            pd.DataFrame: Train station DataFrame with additional columns for the closest EMS station,
                          its latitude, longitude, and distance in kilometers.
        """
        train_stations_path = os.path.join(self.data_folder, CSV_TRAIN_STATIONS)
        ems_stations_path = os.path.join(self.data_folder, CSV_FMI_EMS)

        if not os.path.exists(train_stations_path):
            raise FileNotFoundError(f"Train station metadata file '{CSV_TRAIN_STATIONS}' not found.")

        if not os.path.exists(ems_stations_path):
            raise FileNotFoundError(f"EMS metadata file '{CSV_FMI_EMS}' not found.")

        # Load metadata files
        self.merged_metadata = pd.read_csv(train_stations_path)
        ems_stations = pd.read_csv(ems_stations_path)

        # Drop unnecessary columns from train stations
        self.merged_metadata = self.merged_metadata.drop(
            columns=["type", "stationUICCode", "countryCode", "passengerTraffic"],
            errors="ignore"
        )

        # Rename columns for consistency
        self.merged_metadata = self.merged_metadata.rename(
            columns={
                "stationName": "train_station_name",
                "stationShortCode": "train_station_short_code",
                "longitude": "train_long",
                "latitude": "train_lat",
            }
        )

        # Apply the function to each train station
        self.merged_metadata[["closest_ems_station", "ems_latitude", "ems_longitude", "distance_km"]] = self.merged_metadata.apply(
            lambda row: self._find_closest_ems(row["train_lat"], row["train_long"], ems_stations),
            axis=1,
            result_type="expand"
        )

        print("\n✅ Closest EMS stations matched with train stations.")

        # Save the merged data to CSV
        self.save_to_csv(self.merged_metadata, CSV_CLOSEST_EMS_TRAIN)

        return self.merged_metadata

    def load_csv_files_by_month(self):
        """
        Load one train and one weather file for each corresponding month.
        Calls the merge_train_weather_data function to process the paired data.
        """
        if not self.train_files or not self.weather_files:
            raise ValueError("Train or weather files are not loaded.")

        # Extract the year-month from file names
        train_files_by_month = {self._extract_dates_from_filenames([file])[0]: file for file in self.train_files}
        weather_files_by_month = {self._extract_dates_from_filenames([file])[0]: file for file in self.weather_files}

        # Find common months between train and weather files
        common_months = set(train_files_by_month.keys()).intersection(set(weather_files_by_month.keys()))

        if not common_months:
            raise ValueError("No matching months found between train and weather data files.")

        for month in sorted(common_months):
            train_file = train_files_by_month[month]
            weather_file = weather_files_by_month[month]

            print(f"\n📅 Loading data for month: {month}")
            print(f"Train file: {train_file}")
            print(f"Weather file: {weather_file}")

            # Load the train and weather data for the current month
            train_data = pd.read_csv(train_file)
            weather_data = pd.read_csv(weather_file)

            # Call the merge function
            self.merge_train_weather_data(train_data, weather_data, month)


    def merge_train_weather_data(self, train_data, weather_data, month_str):
        """
        Merges train timetable data with the closest EMS weather observations for one month.

        Parameters:
            train_data (pd.DataFrame): DataFrame containing train schedule data.
            weather_data (pd.DataFrame): DataFrame containing EMS weather observations.
            month_str (str): The month in 'YYYY-MM' format.

        Returns:
            pd.DataFrame: Updated train_data DataFrame with weather observations merged into timetable records.
        """
        # Extract unique departure dates
        unique_dates = train_data["departureDate"].unique()
        print(f"🔹 Starting to process train and weather data for {len(unique_dates)} departure dates.")

        # Ensure timestamp is in datetime format (convert inplace to avoid copies)
        weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"], errors="coerce")

        # Precompute EMS weather data in a dictionary for quick lookups
        self.ems_weather_dict = {
            station: df.sort_values(by="timestamp").reset_index(drop=True)
            for station, df in weather_data.groupby("station_name")
        }

        total_trains = len(train_data)

        # Iterate over train data using itertuples for better performance
        for idx, train_row in enumerate(train_data.itertuples(index=False), start=1):
            train_number = train_row.trainNumber
            departure_date = train_row.departureDate
            
            # ✅ Print only when starting to process a new departure date
            if idx == 1 or (departure_date != train_data.iloc[idx - 2]["departureDate"]):
                print(f"📅 Processing data for departure date: {departure_date}")

            timetable = train_row.timeTableRows

            # ✅ Fix timetable format if it's a string
            if isinstance(timetable, str):
                try:
                    timetable_fixed = timetable.replace("'", '"') \
                                            .replace("True", "true") \
                                            .replace("False", "false") \
                                            .replace("None", "null")

                    timetable = json.loads(timetable_fixed)
                    if not isinstance(timetable, list):
                        raise ValueError("Decoded timetable is not a list")

                except json.JSONDecodeError as e:
                    print(f"🚨 Failed to decode timetable for train {train_number} on {departure_date}: {e}")
                    timetable = []  # Fallback to empty list

            # Iterate over each station stop in the timetable
            for train_track in timetable:
                station_short_code = train_track.get("stationShortCode")
                scheduled_time = train_track.get("scheduledTime")

                if station_short_code and scheduled_time:
                    closest_ems_row = self.merged_metadata.loc[
                        self.merged_metadata["train_station_short_code"] == station_short_code
                    ]

                    if not closest_ems_row.empty:
                        closest_ems_station = closest_ems_row.iloc[0]["closest_ems_station"]

                        # ✅ Call private method to find closest weather
                        weather_data = self._find_closest_weather(closest_ems_station, scheduled_time)

                        if not weather_data:
                            print(f"⚠️ No weather data available for {closest_ems_station} at {scheduled_time}")

                        # Merge weather data into the stop dictionary
                        train_track["weather_observations"] = weather_data

            # FIX: Reassign timetable back to the DataFrame row
            train_data.at[idx - 1, "timeTableRows"] = timetable

        # Save the merged data for the specific month
        self.save_monthly_data_to_csv(train_data, month_str)
        print(f"\n✅ Merged data for {month_str} saved successfully!")

    def _find_closest_weather(self, ems_station, scheduled_time):
        """
        Finds the closest weather observation.

        Parameters:
            ems_station (str): The closest EMS station name.
            scheduled_time (str): The scheduled time in ISO format.

        Returns:
            dict: A dictionary containing the matched weather observations.
        """
        try:
            # Convert scheduled time to datetime
            scheduled_time_dt = datetime.strptime(scheduled_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as e:
            print(f"🚨 Invalid scheduled time format: {e}")
            return {}

        if ems_station not in self.ems_weather_dict:
            print(f"🚨 No weather data available for EMS '{ems_station}'")
            return {}

        station_weather_df = self.ems_weather_dict[ems_station]

        # Convert timestamps to numpy array for fast lookup
        timestamps = station_weather_df["timestamp"].to_numpy(dtype="datetime64[ns]")

        # Convert scheduled time to numpy datetime64
        scheduled_time_np = np.datetime64(scheduled_time_dt)

        # Use np.searchsorted for fast timestamp lookup
        idx = np.searchsorted(timestamps, scheduled_time_np)

        # Handle edge cases for boundary timestamps
        if idx == 0:
            closest_idx = 0
        elif idx >= len(timestamps):
            closest_idx = len(timestamps) - 1
        else:
            before = abs(timestamps[idx - 1] - scheduled_time_np)
            after = abs(timestamps[idx] - scheduled_time_np)
            closest_idx = idx if after < before else idx - 1

        closest_row = station_weather_df.iloc[closest_idx]
        weather_dict = closest_row.drop(["station_name", "timestamp"]).to_dict()
        weather_dict = {"closest_ems": closest_row["station_name"], **weather_dict}

        return weather_dict

