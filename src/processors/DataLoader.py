import ast
from datetime import datetime
import json
import os
import re
import numpy as np
import pandas as pd
from glob import glob
from haversine import haversine, Unit
from collections import Counter
from config.const import ALTERNATIVE_WEATHER_COLUMN, CSV_ALL_TRAINS, CSV_ALL_TRAINS_FLAT, CSV_CLOSEST_EMS_TRAIN, CSV_TOP5_CLOSEST_EMS_TRAIN, CSV_DELAY_TABLE_EACH_STATION, CSV_DELAY_TABLE_OFFSET, CSV_DELAY_TABLE_ORIGINAL, CSV_FMI, CSV_FMI_EMS, CSV_MATCHED_DATA, CSV_MATCHED_DATA_FLAT, CSV_TRAIN_STATIONS, DELAY_LONG_DISTANCE_TRAINS, FILTER_BY_ROUTE, FILTER_BY_TRAIN_CATEGORY, FMI_ROLLING_WINDOW_HOURS, FMI_ROLLING_WINDOW_PARAMS, FMI_ROLLING_SKIP_MIN_MAX, FMI_ROLLING_INCLUDE_CUMULATIVE, FOLDER_NAME, MANDATORY_STATIONS, TRAIN_CATEGORY_FILTER, get_fmi_rolling_column_names
from config.const import send_email

class DataLoader:
    _TRAIN_LEVEL_COLS = [
        'trainNumber', 'departureDate', 'operatorUICCode', 'operatorShortCode',
        'trainType', 'trainCategory', 'commuterLineID', 'runningCurrently',
        'cancelled', 'version', 'timetableType', 'timetableAcceptanceDate',
    ]

    # Column descriptions written to the companion *_schema.csv for every delay table.
    _DELAY_TABLE_SCHEMA = {
        "year":                      "Calendar year of the departure date.",
        "month":                     "Calendar month of the departure date (1–12).",
        "day_of_month":              "Day within the month (1–31).",
        "day_of_week":               "Day of the week (1=Monday, 7=Sunday).",
        "total_trains_on_route":     "Number of distinct trains that departed on this day.",
        "total_schedules_by_day":    "Total individual station stops scheduled across all trains on this day.",
        "avg_stops_per_train":       "Average stops per train (total_schedules_by_day / total_trains_on_route). Higher values indicate longer or more complex routes.",
        "cancelled_trains_by_day":   "Trains that were fully cancelled at the train level and never ran.",
        "cancelled_stops_by_day":    "Individual station stops cancelled within otherwise-running trains.",
        "delay_count_by_day":        f"Station stops delayed by at least the configured threshold (DELAY_LONG_DISTANCE_TRAINS minutes).",
        "delay_rate":                "Fraction of all scheduled stops that were delayed (delay_count_by_day / total_schedules_by_day). Main normalised punctuality metric.",
        "delay_count_arrivals":      "Delayed stops where the stop type is ARRIVAL.",
        "delay_count_departures":    "Delayed stops where the stop type is DEPARTURE.",
        "max_delay_minutes":         "Maximum single delay recorded across all stops on this day (minutes).",
        "avg_delay_minutes":         "Mean delay across all delayed stops on this day (minutes). Only counts stops meeting the threshold.",
        "median_delay_minutes":      "Median delay across all delayed stops (minutes). More robust to extreme outlier delays than the mean.",
        "delays_5_15min":            "Delayed stops with delay in [5, 15) minutes.",
        "delays_15_30min":           "Delayed stops with delay in [15, 30) minutes.",
        "delays_30_60min":           "Delayed stops with delay in [30, 60) minutes.",
        "delays_over_60min":         "Delayed stops with delay >= 60 minutes.",
        "first_stop_delay_count":    "Trains already delayed at their first stop (delay >= threshold at origin). Separates origin-departure issues from en-route accumulation.",
        "delay_propagation_ratio":   "mean(differenceInMinutes_eachStation_offset) / mean(differenceInMinutes) over stops with a positive raw delay. >1: delays growing along the route; <1: crews recovering time; ~1: stable.",
        "delay_count_by_train_type": "JSON dict mapping each train type (IC, S, P, …) to the number of its delayed stops on this day.",
        "top_10_common_delays":      "The 10 most frequently occurring delay values (minutes) across all delayed stops, ordered by frequency.",
        "rolling_delay_rate_7d":     "7-row rolling mean of delay_rate computed over the full sorted history in this file. Smooths day-to-day variability to reveal trends.",
    }

    # Canonical column order for all delay table CSVs.
    _DELAY_TABLE_COLUMNS = list(_DELAY_TABLE_SCHEMA.keys())

    def __init__(self):
        self.data_folder = FOLDER_NAME
        self.output_folder = FOLDER_NAME
        self.train_files = None
        self.weather_files = None
        self.merged_metadata: pd.DataFrame = pd.DataFrame()  
        self.ems_metadata: pd.DataFrame = pd.DataFrame()
        self.ems_weather_dict: dict = {}                     # Store EMS station metadata for snow depth search
        self.top5_ems_dict: dict = {}                        # Precomputed top-5 closest EMS per train station

        self._check_data_folder()

    def _check_data_folder(self):
        # Create data folder if it doesn't exist
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            print(f"✅ Created data folder: {self.data_folder}")
        
        # Find files matching the patterns
        self.train_files = glob(os.path.join(self.data_folder, f"{CSV_ALL_TRAINS[:-4]}_[0-9]*.csv"))
        self.weather_files = glob(os.path.join(self.data_folder, f"{CSV_FMI[:-4]}*.csv"))

        if not self.train_files:
            print(f"⚠️ No train data files matching '{CSV_ALL_TRAINS}' found in the data folder.")
            print(f"   Make sure to run the script with DATA_FETCH=True first to download the data.")
            raise FileNotFoundError(f"No train data files matching '{CSV_ALL_TRAINS}' found in the data folder.")

        if not self.weather_files:
            print(f"⚠️ No weather data files matching '{CSV_FMI}' found in the data folder.")
            print(f"   Make sure to run the script with DATA_FETCH=True first to download the data.")
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
    
    def preprocess_fmi_rolling_features(self):
        """
        Preprocesses FMI weather data to add rolling window statistics for multiple weather parameters.

        For each measurement timestamp, calculates statistics using data from the
        previous rolling window (lookback). This accounts for the shifting nature
        of the rolling window for every measurement.

        For months 02-12, uses the last 72 hours of data from the previous month's file
        to ensure proper rolling window calculations at the start of each month.

        Parameters processed (defined in FMI_ROLLING_WINDOW_PARAMS):
        - Air temperature, Wind speed, Relative humidity, Precipitation intensity,
          Snow depth, Pressure (msl), Horizontal visibility, Cloud amount, Precipitation amount

        For each parameter and each window size (12h, 24h, 72h), creates new columns:
        - {parameter} ({window}h max): Highest value in the rolling window
        - {parameter} ({window}h min): Lowest value in the rolling window
        - {parameter} ({window}h mean): Mean value in the rolling window
        - {parameter} ({window}h cumulative): Sum of values in the rolling window (Precipitation amount only)

        Parameters in FMI_ROLLING_SKIP_MIN_MAX only get mean and cumulative.
        Parameters not in FMI_ROLLING_INCLUDE_CUMULATIVE skip the cumulative column.

        Returns:
            None. Updates the weather CSV files in place.
        """
        if not self.weather_files:
            raise ValueError("No weather files loaded. Cannot preprocess FMI data.")
        
        max_window = max(FMI_ROLLING_WINDOW_HOURS)

        print(f"\n{'='*60}")
        print(f"🌡️  PREPROCESSING FMI ROLLING WINDOW FEATURES")
        print(f"{'='*60}")
        print(f"Rolling windows: {FMI_ROLLING_WINDOW_HOURS} hours")
        print(f"\nParameters to process ({len(FMI_ROLLING_WINDOW_PARAMS)} total):")
        for param in FMI_ROLLING_WINDOW_PARAMS:
            skip = param in FMI_ROLLING_SKIP_MIN_MAX
            skip_cum = param not in FMI_ROLLING_INCLUDE_CUMULATIVE
            for wh in FMI_ROLLING_WINDOW_HOURS:
                col_names = get_fmi_rolling_column_names(param, wh, skip_min_max=skip, skip_cumulative=skip_cum)
                for col_name in col_names.values():
                    print(f"      → {col_name}")
        print(f"{'='*60}\n")
        
        # Sort weather files chronologically
        sorted_weather_files = sorted(self.weather_files)
        
        # Create a mapping of month to file for easy lookup of previous month
        file_by_month = {}
        for weather_file in sorted_weather_files:
            dates = self._extract_dates_from_filenames([weather_file])
            if dates:
                file_by_month[dates[0]] = weather_file
        
        # Store previous month's data for rolling window continuity
        previous_month_data = None
        previous_month_str = None
        
        for i, weather_file in enumerate(sorted_weather_files):
            print(f"📊 Processing: {os.path.basename(weather_file)}")
            
            # Extract current month string
            current_month_dates = self._extract_dates_from_filenames([weather_file])
            current_month_str = current_month_dates[0] if current_month_dates else None
            
            # Load the weather data
            weather_data = pd.read_csv(weather_file)
            original_row_count = len(weather_data)
            
            # Ensure timestamp is in datetime format
            weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"], errors="coerce")
            
            # Check which parameters exist in the data
            available_params = [p for p in FMI_ROLLING_WINDOW_PARAMS if p in weather_data.columns]
            missing_params = [p for p in FMI_ROLLING_WINDOW_PARAMS if p not in weather_data.columns]
            
            if missing_params:
                print(f"  ⚠️ Missing columns: {missing_params}")
            
            if not available_params:
                print(f"  ⚠️ No target parameters found in {weather_file}. Skipping...")
                previous_month_data = None
                previous_month_str = current_month_str
                continue
            
            # Check if rolling columns already exist (to avoid reprocessing)
            first_param = available_params[0]
            first_skip = first_param in FMI_ROLLING_SKIP_MIN_MAX
            first_skip_cum = first_param not in FMI_ROLLING_INCLUDE_CUMULATIVE
            first_col_names = get_fmi_rolling_column_names(first_param, FMI_ROLLING_WINDOW_HOURS[0], skip_min_max=first_skip, skip_cumulative=first_skip_cum)
            first_check_col = first_col_names['mean']
            if first_check_col in weather_data.columns:
                print(f"  ℹ️ Rolling features already exist. Skipping...")
                # Keep last max_window hours for the next month's rolling window continuity.
                # max() and boolean filter don't require a sorted copy — avoid sort_values+reset_index
                # which would triple peak memory (~2.5 GB file → ~7.5 GB). Free the full array
                # as soon as the small subset is extracted.
                weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"], errors="coerce")
                max_timestamp = weather_data["timestamp"].max()
                cutoff_time = max_timestamp - pd.Timedelta(hours=max_window)
                previous_month_data = weather_data[weather_data["timestamp"] > cutoff_time].copy()
                del weather_data
                previous_month_str = current_month_str
                continue
            
            # Sort by station and timestamp for proper rolling calculation
            weather_data = weather_data.sort_values(by=["station_name", "timestamp"]).reset_index(drop=True)
            
            # Mark current month's data for later filtering
            weather_data["_is_current_month"] = True
            
            # If we have previous month's data, prepend the last 72h to current month
            if previous_month_data is not None and not previous_month_data.empty:
                print(f"  🔗 Using last {max_window}h from previous month ({previous_month_str}) for window continuity")

                # Get the last max_window hours of the previous month
                prev_max_timestamp = previous_month_data["timestamp"].max()
                cutoff_time = prev_max_timestamp - pd.Timedelta(hours=max_window)
                prev_last_hour = previous_month_data[previous_month_data["timestamp"] > cutoff_time].copy()

                # Remove rolling feature columns from previous month if they exist
                cols_to_remove = ["_is_current_month"]
                for param in FMI_ROLLING_WINDOW_PARAMS:
                    skip = param in FMI_ROLLING_SKIP_MIN_MAX
                    skip_cum = param not in FMI_ROLLING_INCLUDE_CUMULATIVE
                    for wh in FMI_ROLLING_WINDOW_HOURS:
                        col_names = get_fmi_rolling_column_names(param, wh, skip_min_max=skip, skip_cumulative=skip_cum)
                        cols_to_remove.extend(col_names.values())
                
                for col in cols_to_remove:
                    if col in prev_last_hour.columns:
                        prev_last_hour = prev_last_hour.drop(columns=[col])
                
                # Mark previous month's data
                prev_last_hour["_is_current_month"] = False
                
                # Concatenate previous month's last hour with current month's data
                weather_data = pd.concat([prev_last_hour, weather_data], ignore_index=True)
                weather_data = weather_data.sort_values(by=["station_name", "timestamp"]).reset_index(drop=True)
                
                print(f"     Added {len(prev_last_hour)} rows from previous month")
            
            # Get unique stations for progress tracking
            unique_stations = weather_data["station_name"].nunique()
            print(f"  📍 Processing {unique_stations} weather stations for {len(available_params)} parameters...")
            
            # Define function to calculate rolling statistics per station
            def calculate_rolling_stats(group):
                """
                Calculate rolling window statistics for all parameters for a single station.

                For each (parameter, window_size) combination, calculates max, min, mean, and cumulative where applicable.
                Parameters in FMI_ROLLING_SKIP_MIN_MAX only get mean and cumulative.
                Parameters not in FMI_ROLLING_INCLUDE_CUMULATIVE skip cumulative.
                """
                station_name = group.name
                is_current_month = group["_is_current_month"].values

                group = group.set_index("timestamp")
                group = group.sort_index()

                new_columns = {}
                for param in available_params:
                    skip = param in FMI_ROLLING_SKIP_MIN_MAX
                    skip_cum = param not in FMI_ROLLING_INCLUDE_CUMULATIVE
                    for wh in FMI_ROLLING_WINDOW_HOURS:
                        col_names = get_fmi_rolling_column_names(param, wh, skip_min_max=skip, skip_cumulative=skip_cum)
                        window_str = f"{wh}h"

                        rolling = group[param].rolling(window=window_str, min_periods=1)

                        if not skip:
                            new_columns[col_names['max']] = rolling.max().round(2)
                            new_columns[col_names['min']] = rolling.min().round(2)

                        new_columns[col_names['mean']] = rolling.mean().round(2)
                        if not skip_cum:
                            new_columns[col_names['cumulative']] = rolling.sum().round(2)

                group = pd.concat([group, pd.DataFrame(new_columns, index=group.index)], axis=1)

                group = group.reset_index()
                group["station_name"] = station_name
                group["_is_current_month"] = is_current_month

                return group
            
            # Apply the rolling calculation to each station group
            weather_data = weather_data.groupby("station_name", group_keys=False).apply(
                calculate_rolling_stats
            )
            
            # Store last max_window hours of current month for next iteration BEFORE filtering
            weather_data_for_next = weather_data[weather_data["_is_current_month"] == True].copy()
            max_timestamp = weather_data_for_next["timestamp"].max()
            cutoff_time = max_timestamp - pd.Timedelta(hours=max_window)
            previous_month_data = weather_data_for_next[weather_data_for_next["timestamp"] > cutoff_time].copy()
            previous_month_str = current_month_str
            
            # Filter to keep only current month's data
            weather_data = weather_data[weather_data["_is_current_month"] == True].copy()
            
            # Remove the helper column
            weather_data = weather_data.drop(columns=["_is_current_month"])
            
            # Restore original column order with new columns at the end
            # First, get the original columns (excluding the new ones)
            original_cols = [col for col in pd.read_csv(weather_file, nrows=0).columns]
            
            # Build list of new columns in order (grouped by param, then by window)
            new_cols = []
            for param in available_params:
                skip = param in FMI_ROLLING_SKIP_MIN_MAX
                skip_cum = param not in FMI_ROLLING_INCLUDE_CUMULATIVE
                for wh in FMI_ROLLING_WINDOW_HOURS:
                    col_names = get_fmi_rolling_column_names(param, wh, skip_min_max=skip, skip_cumulative=skip_cum)
                    new_cols.extend(col_names.values())
            
            # Reorder columns: original columns + new columns
            final_cols = original_cols + new_cols
            
            # Only keep columns that exist in the DataFrame
            final_cols = [col for col in final_cols if col in weather_data.columns]
            weather_data = weather_data[final_cols]
            
            # Sort by timestamp and station for consistent output
            weather_data = weather_data.sort_values(by=["timestamp", "station_name"]).reset_index(drop=True)
            
            # Validate the output
            final_row_count = len(weather_data)
            if original_row_count != final_row_count:
                print(f"  ⚠️ Row count mismatch: {original_row_count} -> {final_row_count}")
            
            # Save back to CSV
            weather_data.to_csv(weather_file, index=False)
            
            # Calculate and display statistics for validation
            print(f"  ✅ Saved: {final_row_count} rows with {len(new_cols)} new columns")
            
            # Show validation stats for each parameter
            for param in available_params:
                skip = param in FMI_ROLLING_SKIP_MIN_MAX
                col_names = get_fmi_rolling_column_names(param, FMI_ROLLING_WINDOW_HOURS[0], skip_min_max=skip)
                check_col = col_names['mean']
                valid_count = weather_data[check_col].notna().sum()
                print(f"     {param}: {valid_count} valid rolling stats")

            # Show missing values and zeros for each new rolling feature
            print(f"\n  📊 Missing Values & Zeros Report for New Features:")
            print(f"     {'Feature':<55} | {'Missing':>10} | {'Zeros':>10}")
            print(f"     {'-'*55}-+-{'-'*10}-+-{'-'*10}")

            for param in available_params:
                skip = param in FMI_ROLLING_SKIP_MIN_MAX
                skip_cum = param not in FMI_ROLLING_INCLUDE_CUMULATIVE
                for wh in FMI_ROLLING_WINDOW_HOURS:
                    col_names = get_fmi_rolling_column_names(param, wh, skip_min_max=skip, skip_cumulative=skip_cum)
                    for stat_type, col_name in col_names.items():
                        if col_name in weather_data.columns:
                            missing_count = weather_data[col_name].isna().sum()
                            zero_count = (weather_data[col_name] == 0).sum()
                            print(f"     {col_name:<55} | {missing_count:>10} | {zero_count:>10}")

            print()

            # Show sample of the new columns for first available parameter
            first_param = available_params[0]
            first_skip = first_param in FMI_ROLLING_SKIP_MIN_MAX
            first_skip_cum = first_param not in FMI_ROLLING_INCLUDE_CUMULATIVE
            first_col_names = get_fmi_rolling_column_names(first_param, FMI_ROLLING_WINDOW_HOURS[0], skip_min_max=first_skip, skip_cumulative=first_skip_cum)
            sample_with_data = weather_data[weather_data[first_param].notna()].head(2)
            if not sample_with_data.empty:
                print(f"  📋 Sample data ({first_param}, {FMI_ROLLING_WINDOW_HOURS[0]}h window):")
                for _, row in sample_with_data.iterrows():
                    vals = f"Val: {row[first_param]:>7.1f}"
                    if not first_skip:
                        vals += f" | Max: {row[first_col_names['max']]:>7.1f} | Min: {row[first_col_names['min']]:>7.1f}"
                    vals += f" | Mean: {row[first_col_names['mean']]:>7.2f}"
                    if not first_skip_cum:
                        vals += f" | Cum: {row[first_col_names['cumulative']]:>7.2f}"
                    print(f"     {row['timestamp']} | {row['station_name'][:20]:<20} | {vals}")
            print()
        
        print(f"{'='*60}")
        print(f"✅ FMI ROLLING WINDOW PREPROCESSING COMPLETE")
        print(f"   Processed {len(FMI_ROLLING_WINDOW_PARAMS)} parameters x {len(FMI_ROLLING_WINDOW_HOURS)} windows across {len(sorted_weather_files)} files")
        print(f"{'='*60}\n")

    def convert_trains_to_flat(self):
        """
        Convert all_trains_data CSV files to flat format (one row per train stop).

        Reads each all_trains_data_YYYY_MM.csv, explodes timeTableRows into individual
        rows, and saves all_trains_data_flat_YYYY_MM.csv alongside the original.
        Skips months where the flat file already exists.
        """
        if not self.train_files:
            raise ValueError("No train files loaded.")

        print(f"\n{'='*60}")
        print("STEP 1.5: Converting train data to flat format")
        print(f"{'='*60}")

        train_level_cols = self._TRAIN_LEVEL_COLS

        for train_file in sorted(self.train_files):
            dates = self._extract_dates_from_filenames([train_file])
            if not dates:
                print(f"⚠️ Could not extract date from {train_file}. Skipping.")
                continue

            month_period = pd.Period(dates[0], freq='M')
            base = CSV_ALL_TRAINS_FLAT.replace('.csv', '')
            flat_filename = f"{base}_{month_period.year}_{month_period.month:02d}.csv"
            flat_filepath = os.path.join(self.output_folder, flat_filename)

            if os.path.exists(flat_filepath):
                print(f"  ℹ️ {flat_filename} already exists. Skipping.")
                continue

            print(f"📊 Converting {os.path.basename(train_file)}...")
            train_data = pd.read_csv(train_file)
            rows = []

            for _, train_row in train_data.iterrows():
                timetable_raw = train_row['timeTableRows']
                try:
                    timetable = ast.literal_eval(timetable_raw) if isinstance(timetable_raw, str) else timetable_raw
                except (ValueError, SyntaxError) as e:
                    print(f"⚠️ Failed to parse timeTableRows for train {train_row.get('trainNumber')}: {e}")
                    continue

                if not isinstance(timetable, list):
                    continue

                train_base = {col: train_row.get(col) for col in train_level_cols if col in train_row.index}

                for stop in timetable:
                    row = dict(train_base)
                    row['stationName'] = stop.get('stationName')
                    row['stationShortCode'] = stop.get('stationShortCode')
                    row['stationUICCode'] = stop.get('stationUICCode')
                    row['countryCode'] = stop.get('countryCode')
                    row['type'] = stop.get('type')
                    row['trainStopping'] = stop.get('trainStopping')
                    row['commercialStop'] = stop.get('commercialStop')
                    row['commercialTrack'] = stop.get('commercialTrack')
                    row['stop_cancelled'] = stop.get('cancelled')
                    row['scheduledTime'] = stop.get('scheduledTime')
                    row['actualTime'] = stop.get('actualTime')
                    row['differenceInMinutes'] = stop.get('differenceInMinutes')
                    row['causes'] = str(stop.get('causes', []))
                    train_ready = stop.get('trainReady')
                    row['trainReady'] = str(train_ready) if train_ready is not None else None
                    rows.append(row)

            if not rows:
                print(f"  ⚠️ No rows to save for {flat_filename}. Skipping.")
                continue

            flat_df = pd.DataFrame(rows)
            flat_df.to_csv(flat_filepath, index=False)
            print(f"  ✅ Saved {len(rows)} rows to {flat_filename}")

        print(f"\n✅ Flat train conversion complete.")
        print(f"{'='*60}\n")

    def convert_matched_to_flat(self):
        """
        Convert matched_data CSV files to flat format (one row per train stop).

        Reads each matched_data_YYYY_MM.csv, explodes timeTableRows into individual
        rows, flattens the weather_observations dict into top-level columns, and
        saves matched_data_flat_YYYY_MM.csv alongside the original.
        Skips months where the flat file already exists.
        """
        matched_files = glob(os.path.join(self.output_folder, f"{CSV_MATCHED_DATA.replace('.csv', '')}_[0-9]*.csv"))

        if not matched_files:
            print("⚠️ No matched data files found. Skipping flat conversion.")
            return

        print(f"\n{'='*60}")
        print("STEP 3.5: Converting matched data to flat format")
        print(f"{'='*60}")

        train_level_cols = self._TRAIN_LEVEL_COLS

        for matched_file in sorted(matched_files):
            dates = self._extract_dates_from_filenames([matched_file])
            if not dates:
                print(f"⚠️ Could not extract date from {matched_file}. Skipping.")
                continue

            month_period = pd.Period(dates[0], freq='M')
            base = CSV_MATCHED_DATA_FLAT.replace('.csv', '')
            flat_filename = f"{base}_{month_period.year}_{month_period.month:02d}.csv"
            flat_filepath = os.path.join(self.output_folder, flat_filename)

            if os.path.exists(flat_filepath):
                print(f"  ℹ️ {flat_filename} already exists. Skipping.")
                continue

            print(f"📊 Converting {os.path.basename(matched_file)}...")
            total_rows_written = 0
            first_chunk = True

            for chunk in pd.read_csv(matched_file, chunksize=500):
                rows = []

                for _, train_row in chunk.iterrows():
                    timetable_raw = train_row['timeTableRows']
                    try:
                        timetable = ast.literal_eval(timetable_raw) if isinstance(timetable_raw, str) else timetable_raw
                    except (ValueError, SyntaxError):
                        try:
                            timetable_fixed = timetable_raw.replace("'", '"') \
                                                            .replace("True", "true") \
                                                            .replace("False", "false") \
                                                            .replace("None", "null") \
                                                            .replace(": nan", ": null")
                            timetable = json.loads(timetable_fixed)
                        except (json.JSONDecodeError, Exception) as e:
                            print(f"⚠️ Failed to parse timeTableRows for train {train_row.get('trainNumber')}: {e}")
                            continue

                    if not isinstance(timetable, list):
                        continue

                    train_base = {col: train_row.get(col) for col in train_level_cols if col in train_row.index}

                    for stop in timetable:
                        row = dict(train_base)
                        for key, value in stop.items():
                            if key == 'cancelled':
                                row['stop_cancelled'] = value
                            elif key == 'causes':
                                row['causes'] = str(value)
                            elif key == 'trainReady':
                                row['trainReady'] = str(value) if value is not None else None
                            elif key == 'weather_observations':
                                if isinstance(value, dict):
                                    row.update(value)
                            else:
                                row[key] = value
                        rows.append(row)

                if rows:
                    pd.DataFrame(rows).to_csv(flat_filepath, mode='a', header=first_chunk, index=False)
                    total_rows_written += len(rows)
                    first_chunk = False

            if total_rows_written == 0:
                print(f"  ⚠️ No rows to save for {flat_filename}. Skipping.")
                continue

            print(f"  ✅ Saved {total_rows_written} rows to {flat_filename}")

        print(f"\n✅ Flat matched data conversion complete.")
        print(f"{'='*60}\n")

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

    def _find_top_n_closest_ems(self, train_lat, train_long, ems_stations, n=5):
        """
        Finds the top N closest EMS stations based on Haversine distance.

        Parameters:
            train_lat (float): Latitude of the train station.
            train_long (float): Longitude of the train station.
            ems_stations (pd.DataFrame): DataFrame containing EMS station data.
            n (int): Number of closest stations to return.

        Returns:
            pd.Series: Flat series with columns ems_1_station, ems_1_lat, ems_1_long, ems_1_distance_km, ..., ems_N_*.
        """
        train_coords = (train_lat, train_long)
        distances = []

        for _, ems in ems_stations.iterrows():
            ems_coords = (ems["latitude"], ems["longitude"])
            distance = haversine(train_coords, ems_coords, unit=Unit.KILOMETERS)
            distances.append((ems["station_name"], ems["latitude"], ems["longitude"], distance))

        distances.sort(key=lambda x: x[3])
        top_n = distances[:n]

        result = {}
        for i, (name, lat, lon, dist) in enumerate(top_n, start=1):
            result[f"ems_{i}_station"] = name
            result[f"ems_{i}_lat"] = lat
            result[f"ems_{i}_long"] = lon
            result[f"ems_{i}_distance_km"] = round(dist, 2)

        return pd.Series(result)

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

        # Build top-5 closest EMS stations per train station (wide format)
        top5_columns = self.merged_metadata[["train_station_name", "train_station_short_code", "train_lat", "train_long"]].copy()
        top5_ems = self.merged_metadata.apply(
            lambda row: self._find_top_n_closest_ems(row["train_lat"], row["train_long"], ems_stations, n=5),
            axis=1
        )
        top5_metadata = pd.concat([top5_columns, top5_ems], axis=1)

        print("✅ Top 5 closest EMS stations matched with train stations.")

        self.save_to_csv(top5_metadata, CSV_TOP5_CLOSEST_EMS_TRAIN)

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
            month_period = pd.Period(month, freq='M')
            base = CSV_MATCHED_DATA.replace('.csv', '')
            matched_filename = f"{base}_{month_period.year}_{month_period.month:02d}.csv"
            matched_filepath = os.path.join(self.output_folder, matched_filename)

            if os.path.exists(matched_filepath):
                print(f"  ℹ️ {matched_filename} already exists. Skipping.")
                continue

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

    def _save_delay_table_schema(self, csv_filename):
        """Save a companion *_schema.csv describing every column in the delay table."""
        schema_filename = csv_filename.replace('.csv', '_schema.csv')
        schema_path = os.path.join(self.output_folder, schema_filename)
        rows = [{'column': col, 'description': desc}
                for col, desc in self._DELAY_TABLE_SCHEMA.items()]
        pd.DataFrame(rows).to_csv(schema_path, index=False)

    def _track_delays_for_column(self, filtered_train_data, delay_column, csv_filename, month_str):
        """
        Track daily delay statistics for one delay column variant and save to CSV.

        Produces one row per calendar day with counts, rates, distributions, and
        derived metrics. A companion *_schema.csv is written alongside with
        plain-English descriptions of every column.

        Parameters:
            filtered_train_data (pd.DataFrame): Filtered train data for the month.
            delay_column (str): Delay column to aggregate ('differenceInMinutes',
                'differenceInMinutes_offset', or 'differenceInMinutes_eachStation_offset').
            csv_filename (str): Output CSV filename (basename only).
            month_str (str): Month being processed in 'YYYY-MM' format.
        """
        delay_file_path = os.path.join(self.output_folder, csv_filename)

        if os.path.exists(delay_file_path):
            delay_summary_df = pd.read_csv(delay_file_path)
            # Remove legacy column if present from older runs
            if 'total_delay_minutes' in delay_summary_df.columns:
                delay_summary_df = delay_summary_df.drop(columns=['total_delay_minutes'])
            # Add any columns introduced in this version that don't exist yet
            for col in self._DELAY_TABLE_COLUMNS:
                if col not in delay_summary_df.columns:
                    delay_summary_df[col] = pd.NA
            print(f"Loaded existing {delay_column} delay summary with {len(delay_summary_df)} records.")
        else:
            delay_summary_df = pd.DataFrame(columns=self._DELAY_TABLE_COLUMNS)
            print(f"Created new {delay_column} delay summary table.")

        year, month = month_str.split("-")
        daily_delays = {}

        for departure_date, day_trains in filtered_train_data.groupby('departureDate'):
            try:
                date_obj = datetime.strptime(departure_date, "%Y-%m-%d")
                day_of_month = date_obj.day
                day_of_week = date_obj.weekday() + 1  # 1=Monday … 7=Sunday
            except ValueError:
                print(f"🚨 Invalid date format: {departure_date}")
                continue

            day_route_trains = set()
            day_total_schedules = 0
            day_cancelled_trains = 0
            day_cancelled_stops = 0
            day_first_stop_delays = 0
            day_all_delays = []         # qualifying delay values for the tracked column
            day_delay_arrivals = 0
            day_delay_departures = 0
            day_delay_by_type = Counter()  # trainType -> delayed-stop count
            # Paired lists for propagation ratio (always uses raw differenceInMinutes)
            day_raw_for_ratio = []
            day_offset_for_ratio = []

            for _, train_row in day_trains.iterrows():
                train_number = train_row.trainNumber
                train_type = str(getattr(train_row, 'trainType', None) or 'unknown')
                train_cancelled = bool(getattr(train_row, 'cancelled', False))
                timetable = train_row.timeTableRows

                if train_cancelled:
                    day_cancelled_trains += 1

                if isinstance(timetable, str):
                    try:
                        timetable_fixed = (timetable
                                           .replace("'", '"')
                                           .replace("True", "true")
                                           .replace("False", "false")
                                           .replace("None", "null"))
                        timetable = json.loads(timetable_fixed)
                        if not isinstance(timetable, list):
                            raise ValueError("Decoded timetable is not a list")
                    except json.JSONDecodeError as e:
                        print(f"🚨 Failed to decode timetable for train {train_number} on {departure_date}: {e}")
                        timetable = []

                day_route_trains.add(train_number)

                if not isinstance(timetable, list):
                    continue

                day_total_schedules += len(timetable)

                # First-stop delay (always uses raw differenceInMinutes regardless of delay_column)
                if timetable:
                    first_raw = timetable[0].get("differenceInMinutes")
                    if first_raw is not None and first_raw >= DELAY_LONG_DISTANCE_TRAINS:
                        day_first_stop_delays += 1

                for stop in timetable:
                    # Stop-level cancellation
                    if bool(stop.get("cancelled", False)):
                        day_cancelled_stops += 1

                    # Data for propagation ratio: pair raw delay with per-stop offset
                    raw_diff = stop.get("differenceInMinutes")
                    each_offset = stop.get("differenceInMinutes_eachStation_offset")
                    if raw_diff is not None and raw_diff > 0 and each_offset is not None:
                        day_raw_for_ratio.append(raw_diff)
                        day_offset_for_ratio.append(each_offset)

                    # Tracked delay column stats
                    delay_value = stop.get(delay_column)
                    if delay_value is None or delay_value < DELAY_LONG_DISTANCE_TRAINS:
                        continue

                    day_all_delays.append(delay_value)
                    day_delay_by_type[train_type] += 1

                    stop_type = stop.get("type", "")
                    if stop_type == "ARRIVAL":
                        day_delay_arrivals += 1
                    elif stop_type == "DEPARTURE":
                        day_delay_departures += 1

            # --- Derived statistics ---
            n_delayed = len(day_all_delays)
            n_trains = len(day_route_trains)

            day_avg_delay = round(sum(day_all_delays) / n_delayed, 2) if n_delayed > 0 else 0
            day_median_delay = round(float(pd.Series(day_all_delays).median()), 2) if day_all_delays else 0
            day_max_delay = max(day_all_delays) if day_all_delays else 0
            day_delay_rate = round(n_delayed / day_total_schedules, 4) if day_total_schedules > 0 else 0
            day_avg_stops = round(day_total_schedules / n_trains, 2) if n_trains > 0 else 0

            if day_raw_for_ratio:
                mean_raw = sum(day_raw_for_ratio) / len(day_raw_for_ratio)
                mean_offset = sum(day_offset_for_ratio) / len(day_offset_for_ratio)
                propagation_ratio = round(mean_offset / mean_raw, 4) if mean_raw != 0 else None
            else:
                propagation_ratio = None

            top_10 = [d for d, _ in Counter(day_all_delays).most_common(10)]

            daily_delays[departure_date] = {
                'year': year,
                'month': month,
                'day_of_month': day_of_month,
                'day_of_week': day_of_week,
                'total_trains_on_route': n_trains,
                'total_schedules': day_total_schedules,
                'avg_stops_per_train': day_avg_stops,
                'cancelled_trains': day_cancelled_trains,
                'cancelled_stops': day_cancelled_stops,
                'delay_count': n_delayed,
                'delay_rate': day_delay_rate,
                'delay_count_arrivals': day_delay_arrivals,
                'delay_count_departures': day_delay_departures,
                'max_delay_minutes': day_max_delay,
                'avg_delay_minutes': day_avg_delay,
                'median_delay_minutes': day_median_delay,
                'delays_5_15min': sum(1 for d in day_all_delays if 5 <= d < 15),
                'delays_15_30min': sum(1 for d in day_all_delays if 15 <= d < 30),
                'delays_30_60min': sum(1 for d in day_all_delays if 30 <= d < 60),
                'delays_over_60min': sum(1 for d in day_all_delays if d >= 60),
                'first_stop_delay_count': day_first_stop_delays,
                'delay_propagation_ratio': propagation_ratio,
                'delay_count_by_train_type': json.dumps(dict(day_delay_by_type)),
                'top_10_common_delays': str(top_10),
            }

        # --- Upsert into summary DataFrame ---
        for date_str, s in daily_delays.items():
            row_data = {
                'year':                      s['year'],
                'month':                     s['month'],
                'day_of_month':              s['day_of_month'],
                'day_of_week':               s['day_of_week'],
                'total_trains_on_route':     s['total_trains_on_route'],
                'total_schedules_by_day':    s['total_schedules'],
                'avg_stops_per_train':       s['avg_stops_per_train'],
                'cancelled_trains_by_day':   s['cancelled_trains'],
                'cancelled_stops_by_day':    s['cancelled_stops'],
                'delay_count_by_day':        s['delay_count'],
                'delay_rate':                s['delay_rate'],
                'delay_count_arrivals':      s['delay_count_arrivals'],
                'delay_count_departures':    s['delay_count_departures'],
                'max_delay_minutes':         s['max_delay_minutes'],
                'avg_delay_minutes':         s['avg_delay_minutes'],
                'median_delay_minutes':      s['median_delay_minutes'],
                'delays_5_15min':            s['delays_5_15min'],
                'delays_15_30min':           s['delays_15_30min'],
                'delays_30_60min':           s['delays_30_60min'],
                'delays_over_60min':         s['delays_over_60min'],
                'first_stop_delay_count':    s['first_stop_delay_count'],
                'delay_propagation_ratio':   s['delay_propagation_ratio'],
                'delay_count_by_train_type': s['delay_count_by_train_type'],
                'top_10_common_delays':      s['top_10_common_delays'],
            }

            mask = (
                (delay_summary_df['year'].astype(str) == str(s['year'])) &
                (delay_summary_df['month'].astype(str) == str(s['month'])) &
                (delay_summary_df['day_of_month'].astype(str) == str(s['day_of_month']))
            )

            if mask.any():
                for col, val in row_data.items():
                    delay_summary_df.loc[mask, col] = val
            else:
                delay_summary_df = pd.concat(
                    [delay_summary_df, pd.DataFrame([row_data])], ignore_index=True
                )

        # Sort chronologically before computing the rolling stat
        delay_summary_df = (delay_summary_df
                            .sort_values(by=['year', 'month', 'day_of_month'])
                            .reset_index(drop=True))

        # 7-row rolling mean of delay_rate over the full sorted history
        delay_summary_df['rolling_delay_rate_7d'] = (
            delay_summary_df['delay_rate']
            .rolling(window=7, min_periods=1)
            .mean()
            .round(4)
        )

        # Enforce canonical column order
        delay_summary_df = delay_summary_df[self._DELAY_TABLE_COLUMNS]

        delay_summary_df.to_csv(delay_file_path, index=False)
        self._save_delay_table_schema(csv_filename)

        total_month_delays = sum(s['delay_count'] for s in daily_delays.values())
        total_month_schedules = sum(s['total_schedules'] for s in daily_delays.values())
        total_month_trains = sum(s['total_trains_on_route'] for s in daily_delays.values())

        print(f"✅ Updated {delay_column} delay summary for {month_str}: {len(daily_delays)} days processed.")
        print(f"   {delay_column} Summary: {total_month_delays} delays / {total_month_schedules} schedules / {total_month_trains} trains.")

    def merge_train_weather_data(self, train_data, weather_data, month_str):
        """
        Merges train timetable data with the closest EMS weather observations for one month.
        Also tracks delays for trains based on route filtering settings using all 3 delay columns.
        
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

        # Load precomputed top-5 closest EMS stations per train station
        top5_path = os.path.join(self.data_folder, CSV_TOP5_CLOSEST_EMS_TRAIN)
        if os.path.exists(top5_path):
            top5_df = pd.read_csv(top5_path)
            self.top5_ems_dict = {
                row["train_station_short_code"]: row for _, row in top5_df.iterrows()
            }
            print(f"✅ Loaded top-5 EMS lookup with {len(self.top5_ems_dict)} train stations.")
        else:
            print(f"⚠️ Top-5 EMS file not found. Alternative weather search will fall back to brute force.")
            self.top5_ems_dict = {}

        # STEP 1: Filter trains by train category (if enabled)
        if FILTER_BY_TRAIN_CATEGORY:
            print(f"🔍 Filtering trains by trainCategory = '{TRAIN_CATEGORY_FILTER}'")
            initial_count = len(train_data)
            
            # Filter for specified train category
            category_filtered_train_data = train_data[train_data['trainCategory'] == TRAIN_CATEGORY_FILTER].copy()
            category_filtered_count = len(category_filtered_train_data)
            
            print(f"✅ Filtered from {initial_count} to {category_filtered_count} {TRAIN_CATEGORY_FILTER.lower()} trains.")
            
            if category_filtered_train_data.empty:
                print(f"⚠️ No {TRAIN_CATEGORY_FILTER.lower()} trains found for {month_str}")
                return category_filtered_train_data
            
            working_train_data = category_filtered_train_data
            train_type_description = TRAIN_CATEGORY_FILTER.lower()
        else:
            # Include all train categories
            working_train_data = train_data.copy()
            train_type_description = "all"
            print(f"✅ Processing all {len(working_train_data)} trains (train category filtering disabled).")

        # STEP 2: Filter trains based on route filtering setting (if enabled)
        if FILTER_BY_ROUTE and MANDATORY_STATIONS:
            print(f"🔍 Further filtering {train_type_description} trains that pass through mandatory stations: {MANDATORY_STATIONS}")
            filtered_train_indices = []
            
            for idx, train_row in working_train_data.iterrows():
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

            # Filter the working_train_data to only include trains that pass through mandatory stations
            filtered_train_data = working_train_data.loc[filtered_train_indices].copy()
            print(f"✅ Further filtered from {len(working_train_data)} to {len(filtered_train_data)} {train_type_description} trains that pass through mandatory stations.")
            
            # If no trains pass through mandatory stations, return empty DataFrame
            if filtered_train_data.empty:
                print(f"⚠️ No {train_type_description} trains found that pass through all mandatory stations for {month_str}")
                return filtered_train_data
        else:
            # Include all trains from the category filter (no route filtering)
            filtered_train_data = working_train_data.copy()
            if FILTER_BY_ROUTE:
                print(f"✅ Processing all {len(filtered_train_data)} {train_type_description} trains (no mandatory stations specified).")
            else:
                print(f"✅ Processing all {len(filtered_train_data)} {train_type_description} trains (route filtering disabled).")

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

        # Group filtered train data by departure date for daily processing
        train_data_grouped = filtered_train_data.groupby('departureDate')

        for departure_date, day_trains in train_data_grouped:
            print(f"📅 Processing data for departure date: {departure_date}")

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

                            # Call private method to find closest weather with snow depth alternative
                            weather_data_point = self._find_closest_weather(
                                closest_ems_station,
                                scheduled_time,
                                station_short_code
                            )

                            if not weather_data_point:
                                print(f"⚠️ No weather data available for {closest_ems_station} at {scheduled_time}")

                            # Merge weather data into the stop dictionary
                            train_track["weather_observations"] = weather_data_point

                # Reassign timetable back to the DataFrame row
                filtered_train_data.at[idx, "timeTableRows"] = timetable

        # Save the merged data for the specific month
        self.save_monthly_data_to_csv(filtered_train_data, month_str)

        # Update print statement based on filtering settings
        filter_description = ""
        if FILTER_BY_TRAIN_CATEGORY:
            filter_description += f"{train_type_description} trains"
            if FILTER_BY_ROUTE and MANDATORY_STATIONS:
                filter_description += f" passing through {MANDATORY_STATIONS}"
        else:
            if FILTER_BY_ROUTE and MANDATORY_STATIONS:
                filter_description += f"trains passing through {MANDATORY_STATIONS}"
            else:
                filter_description += "all trains"
        
        print(f"\n✅ Merged data for {month_str} saved successfully! Only {filter_description} included.")
        
        # Track delays for all 3 delay columns using the helper method
        print(f"\n📊 Tracking delays for all 3 delay columns for {month_str}...")
        
        # Track delays for differenceInMinutes
        self._track_delays_for_column(
            filtered_train_data, 
            "differenceInMinutes", 
            CSV_DELAY_TABLE_ORIGINAL, 
            month_str
        )
        
        # Track delays for differenceInMinutes_offset
        self._track_delays_for_column(
            filtered_train_data, 
            "differenceInMinutes_offset", 
            CSV_DELAY_TABLE_OFFSET, 
            month_str
        )
        
        # Track delays for differenceInMinutes_eachStation_offset
        self._track_delays_for_column(
            filtered_train_data, 
            "differenceInMinutes_eachStation_offset", 
            CSV_DELAY_TABLE_EACH_STATION, 
            month_str
        )
        
        print(f"\n✅ All delay tracking completed for {month_str}!")
        
        return filtered_train_data

    def _find_alternative_weather_data(self, station_short_code, scheduled_time, target_column, exclude_station=None):
        """
        Finds weather data for a specific column from an alternative EMS station using the precomputed top-5 lookup.

        Parameters:
            station_short_code (str): The train station short code to look up in the top-5 table.
            scheduled_time (str): The scheduled time in ISO format.
            target_column (str): The weather column name to search for (e.g., "Snow depth", "Air temperature").
            exclude_station (str): Station to exclude from search (the primary station).

        Returns:
            dict: Dictionary containing the target feature's instant value and all its rolling window
                  columns from the alternative station, using original column names. Empty dict if none found.
        """
        if not self.top5_ems_dict:
            return {}

        top5_row = self.top5_ems_dict.get(station_short_code)
        if top5_row is None:
            return {}

        try:
            scheduled_time_dt = datetime.strptime(scheduled_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as e:
            print(f"🚨 Invalid scheduled time format for alternative search: {e}")
            return {}

        scheduled_time_np = np.datetime64(scheduled_time_dt)

        # Build the list of columns to extract: instant value + all rolling window columns for this feature
        columns_to_extract = [target_column]
        if target_column in FMI_ROLLING_WINDOW_PARAMS:
            skip_min_max = target_column in FMI_ROLLING_SKIP_MIN_MAX
            skip_cumulative = target_column not in FMI_ROLLING_INCLUDE_CUMULATIVE
            for window_hours in FMI_ROLLING_WINDOW_HOURS:
                rolling_names = get_fmi_rolling_column_names(target_column, window_hours, skip_min_max, skip_cumulative)
                columns_to_extract.extend(rolling_names.values())

        # Iterate through precomputed top-5 closest EMS stations (already sorted by distance)
        for rank in range(1, 6):
            station_name = top5_row.get(f"ems_{rank}_station")

            if pd.isna(station_name):
                continue

            # Skip the primary station
            if exclude_station and station_name == exclude_station:
                continue

            # Skip if station has no weather data loaded
            if station_name not in self.ems_weather_dict:
                continue

            station_weather_df = self.ems_weather_dict[station_name]

            # Convert timestamps to numpy array for fast lookup
            timestamps = station_weather_df["timestamp"].to_numpy(dtype="datetime64[ns]")

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

            # Check if this station has the target instant value
            weather_value = closest_row.get(target_column)
            if pd.notna(weather_value) and weather_value is not None:
                # Extract the instant value and all rolling window columns for this feature
                result = {}
                for col in columns_to_extract:
                    val = closest_row.get(col)
                    if val is not None and pd.notna(val):
                        result[col] = float(val)
                return result

        # No alternative weather data found
        return {}

    def _find_closest_weather(self, ems_station, scheduled_time, station_short_code):
        """
        Finds the closest weather observation and alternative weather data for multiple features if needed.

        Parameters:
            ems_station (str): The closest EMS station name.
            scheduled_time (str): The scheduled time in ISO format.
            station_short_code (str): The train station short code for top-5 EMS lookup.

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

        weather_dict = closest_row.drop(["station_name"]).to_dict()
        weather_dict = {"closest_ems": closest_row["station_name"], **weather_dict}

        # Remove the timestamp key (not needed in output)
        weather_dict.pop("timestamp", None)


        # Ensure ALTERNATIVE_WEATHER_COLUMN is a list
        weather_features = ALTERNATIVE_WEATHER_COLUMN if isinstance(ALTERNATIVE_WEATHER_COLUMN, list) else [ALTERNATIVE_WEATHER_COLUMN]

        # Check each weather feature for missing data and search for alternatives
        for feature_name in weather_features:
            target_weather_value = weather_dict.get(feature_name)

            if pd.isna(target_weather_value) or target_weather_value is None:
                # Search for alternative weather data (instant + rolling) for this feature
                alternative_weather_data = self._find_alternative_weather_data(
                    station_short_code,
                    scheduled_time,
                    target_column=feature_name,
                    exclude_station=ems_station
                )

                if alternative_weather_data:
                    weather_dict.update(alternative_weather_data)

        return weather_dict
    
