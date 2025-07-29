from datetime import datetime
import json
import os
import re
import numpy as np
import pandas as pd
from glob import glob
from haversine import haversine, Unit
from collections import Counter
from config.const import ALTERNATIVE_WEATHER_COLUMN, ALTERNATIVE_WEATHER_RADIUS_KM, CSV_ALL_TRAINS, CSV_CLOSEST_EMS_TRAIN, CSV_FMI, CSV_FMI_EMS, CSV_MATCHED_DATA, CSV_TRAIN_STATIONS, DELAY_LONG_DISTANCE_TRAINS, FILTER_BY_ROUTE, FOLDER_NAME, MANDATORY_STATIONS
from config.const import send_email

class DataLoader:
    def __init__(self):
        self.data_folder = FOLDER_NAME
        self.output_folder = FOLDER_NAME
        self.train_files = None
        self.weather_files = None
        self.merged_metadata = None
        self.ems_metadata = None  # Store EMS station metadata for snow depth search

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
            columns=["type", "stationUICCode", "countryCode"],
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

            # Send email with the specific month in the subject and body
            subject = f"Code Execution Complete for {month}"
            body = f"The code has finished running successfully for {month}."
            send_email(subject, body)

    def merge_train_weather_data(self, train_data, weather_data, month_str):
        """
        Merges train timetable data with the closest EMS weather observations for one month.
        Also tracks delays for trains based on route filtering settings.
        
        Parameters:
            train_data (pd.DataFrame): DataFrame containing train schedule data.
            weather_data (pd.DataFrame): DataFrame containing EMS weather observations.
            month_str (str): The month in 'YYYY-MM' format.

        Returns:
            pd.DataFrame: Updated train_data DataFrame with weather observations merged into timetable records.
        """

        # Check if merged_metadata is populated
        if self.merged_metadata.empty:
            raise ValueError("merged_metadata is empty. Call match_train_with_ems() first.")
        
        # Load EMS metadata for snow depth alternative search
        ems_stations_path = os.path.join(self.data_folder, CSV_FMI_EMS)
        if os.path.exists(ems_stations_path):
            self.ems_metadata = pd.read_csv(ems_stations_path)
            print(f"✅ Loaded EMS metadata with {len(self.ems_metadata)} stations for snow depth search.")
        else:
            print(f"⚠️ EMS metadata file not found. Snow depth alternative search will be disabled.")
            self.ems_metadata = pd.DataFrame()

        # STEP 1: Filter trains based on route filtering setting
        if FILTER_BY_ROUTE and MANDATORY_STATIONS:
            print(f"🔍 Filtering trains that pass through mandatory stations: {MANDATORY_STATIONS}")
            filtered_train_indices = []
            
            for idx, train_row in train_data.iterrows():
                train_number = train_row.trainNumber
                timetable = train_row.timeTableRows

                # Fix timetable format if it's a string
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
                        print(f"🚨 Failed to decode timetable for train {train_number}: {e}")
                        continue  # Skip this train

                # Extract station codes from timetable
                station_codes = []
                if timetable and isinstance(timetable, list):
                    for stop in timetable:
                        if isinstance(stop, dict) and "stationShortCode" in stop:
                            station_codes.append(stop.get("stationShortCode"))
                
                # Check if train passes through all mandatory stations
                passes_through_mandatory_stations = all(station in station_codes for station in MANDATORY_STATIONS)
                
                if passes_through_mandatory_stations:
                    filtered_train_indices.append(idx)

            # Filter the train_data to only include trains that pass through mandatory stations
            filtered_train_data = train_data.loc[filtered_train_indices].copy()
            print(f"✅ Filtered from {len(train_data)} to {len(filtered_train_data)} trains that pass through mandatory stations.")
            
            # If no trains pass through mandatory stations, return empty DataFrame
            if filtered_train_data.empty:
                print(f"⚠️ No trains found that pass through all mandatory stations for {month_str}")
                return filtered_train_data
        else:
            # Include all trains (no filtering)
            filtered_train_data = train_data.copy()
            if FILTER_BY_ROUTE:
                print(f"✅ Processing all {len(filtered_train_data)} trains (no mandatory stations specified).")
            else:
                print(f"✅ Processing all {len(filtered_train_data)} trains (route filtering disabled).")

        # Extract unique departure dates from filtered data
        unique_dates = filtered_train_data["departureDate"].unique()
        print(f"🔹 Starting to process train and weather data for {len(unique_dates)} departure dates.")

        # Ensure timestamp is in datetime format (convert inplace to avoid copies)
        weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"], errors="coerce")

        # Precompute EMS weather data in a dictionary for quick lookups
        self.ems_weather_dict = {
            station: df.sort_values(by="timestamp").reset_index(drop=True)
            for station, df in weather_data.groupby("station_name")
        }

        # Initialize delay tracking
        delay_file_path = os.path.join(self.output_folder, "delay_table.csv")
        
        # Initialize or load existing delay summary data
        if os.path.exists(delay_file_path):
            delay_summary_df = pd.read_csv(delay_file_path)
            print(f"Loaded existing delay summary with {len(delay_summary_df)} records.")
        else:
            delay_summary_df = pd.DataFrame(columns=[
                "year", "month", "day_of_month", "day_of_week", 
                "delay_count_by_day", "total_schedules_by_day",
                "total_delay_minutes", "max_delay_minutes", "total_trains_on_route", 
                "avg_delay_minutes", "top_10_common_delays"
            ])
            print("Created new delay summary table.")
        
        # Extract year and month from month_str
        year, month = month_str.split("-")
        
        # Initialize daily delay tracking dictionary
        daily_delays = {}  # key: date_str, value: {'delay_count': int, 'total_schedules': int, ...}
        
        route_trains = 0

        # Group filtered train data by departure date for daily processing
        train_data_grouped = filtered_train_data.groupby('departureDate')

        for departure_date, day_trains in train_data_grouped:
            print(f"📅 Processing data for departure date: {departure_date}")
            
            # Initialize daily counters
            day_delay_count = 0
            day_total_schedules = 0
            day_total_delay_minutes = 0
            day_max_delay = 0
            day_route_trains = set()
            day_all_delays = []  # List to store all delay values for the day
            
            # Parse the departure date to get day of week
            try:
                date_obj = datetime.strptime(departure_date, "%Y-%m-%d")
                day_of_month = date_obj.day
                day_of_week = date_obj.weekday() + 1  # Convert 0-6 to 1-7
            except ValueError:
                print(f"🚨 Invalid date format: {departure_date}")
                continue

            # Process trains for this specific date
            for idx, train_row in day_trains.iterrows():
                train_number = train_row.trainNumber
                timetable = train_row.timeTableRows

                # Fix timetable format if it's a string
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

                # Count this train (whether filtered or not)
                route_trains += 1
                day_route_trains.add(train_number)
                
                # Count all station stops for these trains
                if isinstance(timetable, list):
                    day_total_schedules += len(timetable)

                # Find the differenceInMinutes of the first station
                first_station_delay = None
                if timetable and isinstance(timetable, list) and len(timetable) > 0:
                    first_station = timetable[0]
                    if "differenceInMinutes" in first_station:
                        first_station_delay = first_station.get("differenceInMinutes", 0)

                # Variable to track the previous station's differenceInMinutes for eachStation_offset calculation
                previous_station_delay = None

                # Iterate over each station stop in the timetable
                for i, train_track in enumerate(timetable):
                    station_short_code = train_track.get("stationShortCode")
                    scheduled_time = train_track.get("scheduledTime")

                    # Calculate both offset columns and reorder them
                    if "differenceInMinutes" in train_track:
                        current_delay = train_track.get("differenceInMinutes", 0)
                        
                        # Calculate differenceInMinutes_offset
                        if first_station_delay is not None:
                            if i == 0:  # This is the first station
                                # For the first station, keep the original differenceInMinutes
                                offset_value = current_delay
                            else:
                                # For other stations, calculate the offset
                                offset_value = current_delay - first_station_delay
                        else:
                            offset_value = current_delay
                        
                        # Calculate differenceInMinutes_eachStation_offset
                        if i == 0:  # First station
                            # For the first station, keep the original differenceInMinutes
                            each_station_offset_value = current_delay
                            previous_station_delay = current_delay
                        else:
                            # For other stations, calculate difference from previous station's delay
                            if previous_station_delay is not None:
                                each_station_offset_value = current_delay - previous_station_delay
                                previous_station_delay = current_delay
                            else:
                                each_station_offset_value = current_delay
                                previous_station_delay = current_delay
                        
                        # Store original train_track data
                        original_data = dict(train_track)
                        
                        # Rebuild train_track with desired column order
                        train_track.clear()
                        
                        # Add columns in desired order
                        for key, value in original_data.items():
                            train_track[key] = value
                            # Insert offset columns right after differenceInMinutes
                            if key == "differenceInMinutes":
                                train_track["differenceInMinutes_offset"] = offset_value
                                train_track["differenceInMinutes_eachStation_offset"] = each_station_offset_value

                    if station_short_code and scheduled_time:
                        closest_ems_row = self.merged_metadata.loc[
                            self.merged_metadata["train_station_short_code"] == station_short_code
                        ]

                        if not closest_ems_row.empty:
                            closest_ems_station = closest_ems_row.iloc[0]["closest_ems_station"]
                            train_lat = closest_ems_row.iloc[0]["train_lat"]
                            train_long = closest_ems_row.iloc[0]["train_long"]

                            # Call private method to find closest weather with snow depth alternative
                            weather_data_point = self._find_closest_weather(
                                closest_ems_station, 
                                scheduled_time, 
                                train_lat, 
                                train_long
                            )

                            if not weather_data_point:
                                print(f"⚠️ No weather data available for {closest_ems_station} at {scheduled_time}")

                            # Merge weather data into the stop dictionary
                            train_track["weather_observations"] = weather_data_point
                            
                            # Track delays
                            offset = train_track.get("differenceInMinutes_offset")
                            if offset is not None and offset >= DELAY_LONG_DISTANCE_TRAINS:
                                # Increment delay counter for this day
                                day_delay_count += 1
                                day_total_delay_minutes += offset
                                day_max_delay = max(day_max_delay, offset)
                                day_all_delays.append(offset)  # Store the delay value

                # Reassign timetable back to the DataFrame row
                filtered_train_data.at[idx, "timeTableRows"] = timetable

            # Calculate average delay for the day
            day_avg_delay = day_total_delay_minutes / day_delay_count if day_delay_count > 0 else 0

            # Find top 10 most common delays
            if day_all_delays:
                delay_counter = Counter(day_all_delays)
                # Get top 10 most common delays (returns list of tuples [(delay, count), ...])
                top_10_delays = delay_counter.most_common(10)
                # Extract just the delay values
                top_10_delay_values = [delay for delay, count in top_10_delays]
            else:
                top_10_delay_values = []

            # Store daily statistics
            daily_delays[departure_date] = {
                'year': year,
                'month': month,
                'day_of_month': day_of_month,
                'day_of_week': day_of_week,
                'delay_count': day_delay_count,
                'total_schedules': day_total_schedules,
                'total_delay_minutes': day_total_delay_minutes,
                'max_delay_minutes': day_max_delay,
                'total_trains_on_route': len(day_route_trains),
                'avg_delay_minutes': round(day_avg_delay, 2),
                'top_10_common_delays': str(top_10_delay_values)
            }

        # Save the merged data for the specific month
        self.save_monthly_data_to_csv(filtered_train_data, month_str)
        
        # Update print statement based on filtering
        if FILTER_BY_ROUTE and MANDATORY_STATIONS:
            print(f"\n✅ Merged data for {month_str} saved successfully! Only trains passing through {MANDATORY_STATIONS} included.")
        else:
            print(f"\n✅ Merged data for {month_str} saved successfully! All trains included.")
        
        # Update and save delay summary table with daily data
        for date_str, daily_stats in daily_delays.items():
            # Check if this date already exists in summary
            date_exists = ((delay_summary_df['year'] == daily_stats['year']) & 
                        (delay_summary_df['month'] == daily_stats['month']) &
                        (delay_summary_df['day_of_month'] == daily_stats['day_of_month'])).any()
            
            if date_exists:
                # Update existing entry
                mask = ((delay_summary_df['year'] == daily_stats['year']) & 
                    (delay_summary_df['month'] == daily_stats['month']) &
                    (delay_summary_df['day_of_month'] == daily_stats['day_of_month']))
                delay_summary_df.loc[mask, 'day_of_week'] = daily_stats['day_of_week']
                delay_summary_df.loc[mask, 'delay_count_by_day'] = daily_stats['delay_count']
                delay_summary_df.loc[mask, 'total_schedules_by_day'] = daily_stats['total_schedules']
                delay_summary_df.loc[mask, 'total_delay_minutes'] = daily_stats['total_delay_minutes']
                delay_summary_df.loc[mask, 'max_delay_minutes'] = daily_stats['max_delay_minutes']
                delay_summary_df.loc[mask, 'total_trains_on_route'] = daily_stats['total_trains_on_route']
                delay_summary_df.loc[mask, 'avg_delay_minutes'] = daily_stats['avg_delay_minutes']
                delay_summary_df.loc[mask, 'top_10_common_delays'] = daily_stats['top_10_common_delays']
            else:
                # Add new entry
                new_row = pd.DataFrame([{
                    'year': daily_stats['year'], 
                    'month': daily_stats['month'],
                    'day_of_month': daily_stats['day_of_month'],
                    'day_of_week': daily_stats['day_of_week'],
                    'delay_count_by_day': daily_stats['delay_count'],
                    'total_schedules_by_day': daily_stats['total_schedules'],
                    'total_delay_minutes': daily_stats['total_delay_minutes'],
                    'max_delay_minutes': daily_stats['max_delay_minutes'],
                    'total_trains_on_route': daily_stats['total_trains_on_route'],
                    'avg_delay_minutes': daily_stats['avg_delay_minutes'],
                    'top_10_common_delays': daily_stats['top_10_common_delays']
                }])
                delay_summary_df = pd.concat([delay_summary_df, new_row], ignore_index=True)
        
        # Sort by year, month, and day
        delay_summary_df = delay_summary_df.sort_values(by=['year', 'month', 'day_of_month']).reset_index(drop=True)
        
        # Save updated summary
        delay_summary_df.to_csv(delay_file_path, index=False)
        
        # Calculate totals for the month
        total_month_delays = sum(stats['delay_count'] for stats in daily_delays.values())
        total_month_schedules = sum(stats['total_schedules'] for stats in daily_delays.values())
        total_month_trains = sum(stats['total_trains_on_route'] for stats in daily_delays.values())
        
        print(f"✅ Updated delay summary for {month_str}: {len(daily_delays)} days processed.")
        
        # Update summary message based on filtering
        if FILTER_BY_ROUTE and MANDATORY_STATIONS:
            route_description = f"route passing through {'-'.join(MANDATORY_STATIONS)}"
        else:
            route_description = "all routes"
        
        print(f"Summary for {month_str}: Analyzed {len(filtered_train_data)} trains on {route_description} with {total_month_delays} delays out of {total_month_schedules} schedules from {total_month_trains} total trains.")
        
        return filtered_train_data

    def _find_alternative_weather_data(self, train_lat, train_long, scheduled_time, target_column, max_distance_km, exclude_station=None):
        """
        Finds weather data for a specific column from an alternative EMS station within the specified distance range.

        Parameters:
            train_lat (float): Latitude of the train station.
            train_long (float): Longitude of the train station.
            scheduled_time (str): The scheduled time in ISO format.
            target_column (str): The weather column name to search for (e.g., "Snow depth", "Air temperature").
            max_distance_km (float): Maximum distance in kilometers to search for alternative stations.
            exclude_station (str): Station to exclude from search (the primary station).

        Returns:
            dict: Dictionary containing alternative weather value and distance, or empty dict if none found.
                  Format: {'{target_column} Other': value, '{target_column} Other Distance': distance}
        """
        if self.ems_metadata.empty:
            return {}

        try:
            # Convert scheduled time to datetime
            scheduled_time_dt = datetime.strptime(scheduled_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as e:
            print(f"🚨 Invalid scheduled time format for alternative search: {e}")
            return {}

        train_coords = (train_lat, train_long)
        alternative_stations = []

        # Find all EMS stations within the distance range
        for _, ems_station in self.ems_metadata.iterrows():
            station_name = ems_station["station_name"]
            
            # Skip the excluded station (primary closest station)
            if exclude_station and station_name == exclude_station:
                continue

            # Skip if station has no weather data
            if station_name not in self.ems_weather_dict:
                continue

            ems_coords = (ems_station["latitude"], ems_station["longitude"])
            distance = haversine(train_coords, ems_coords, unit=Unit.KILOMETERS)

            if distance <= max_distance_km:
                alternative_stations.append({
                    'station_name': station_name,
                    'latitude': ems_station["latitude"],
                    'longitude': ems_station["longitude"],
                    'distance': distance
                })

        # Sort by distance (closest first)
        alternative_stations.sort(key=lambda x: x['distance'])

        # Search for weather data in alternative stations
        for station_info in alternative_stations:
            station_name = station_info['station_name']
            station_weather_df = self.ems_weather_dict[station_name]

            # Convert timestamps to numpy array for fast lookup
            timestamps = station_weather_df["timestamp"].to_numpy(dtype="datetime64[ns]")
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
            
            # Check if this station has the target weather data
            weather_value = closest_row.get(target_column)
            if pd.notna(weather_value) and weather_value is not None:
                return {
                    f'{target_column} Other': float(weather_value),
                    f'{target_column} Other Distance': round(station_info['distance'], 2)
                }

        # No alternative weather data found
        return {}

    def _find_closest_weather(self, ems_station, scheduled_time, train_lat, train_long):
        """
        Finds the closest weather observation and alternative weather data for multiple features if needed.

        Parameters:
            ems_station (str): The closest EMS station name.
            scheduled_time (str): The scheduled time in ISO format.
            train_lat (float): Latitude of the train station.
            train_long (float): Longitude of the train station.

        Returns:
            dict: A dictionary containing the matched weather observations and alternative weather data for each feature if applicable.
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

        # Ensure ALTERNATIVE_WEATHER_COLUMN is a list
        weather_features = ALTERNATIVE_WEATHER_COLUMN if isinstance(ALTERNATIVE_WEATHER_COLUMN, list) else [ALTERNATIVE_WEATHER_COLUMN]

        # Check each weather feature for missing data and search for alternatives
        for feature_name in weather_features:
            target_weather_value = weather_dict.get(feature_name)
            
            if pd.isna(target_weather_value) or target_weather_value is None:
                # Search for alternative weather data for this specific feature
                alternative_weather_data = self._find_alternative_weather_data(
                    train_lat, 
                    train_long, 
                    scheduled_time, 
                    target_column=feature_name, 
                    max_distance_km=ALTERNATIVE_WEATHER_RADIUS_KM, 
                    exclude_station=ems_station
                )
                
                if alternative_weather_data:
                    weather_dict.update(alternative_weather_data)
                else:
                    # No alternative found, set to None
                    weather_dict[f'{feature_name} Other'] = None
                    weather_dict[f'{feature_name} Other Distance'] = None
            else:
                # Primary weather data exists for this feature, set alternative columns to None
                weather_dict[f'{feature_name} Other'] = None
                weather_dict[f'{feature_name} Other Distance'] = None

        return weather_dict