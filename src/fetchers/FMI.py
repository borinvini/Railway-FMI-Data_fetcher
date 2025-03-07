import os
import time
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

    def save_monthly_to_csv(self, df, base_filename, year, month):
        """
        Saves the DataFrame in a monthly format: `filename_YYYY_MM.csv`.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            base_filename (str): Base filename without extension.
            year (int): Year of the data.
            month (int): Month of the data.
        """
        if df is not None and not df.empty:
            filename = f"{base_filename}_{year}_{str(month).zfill(2)}.csv"
            self.save_to_csv(df, filename)

    def fetch_fmi_data(self, location, start_time, end_time, chunk_hours=1, max_retries=3):
        """
        Fetches weather observation data and station metadata from the Finnish Meteorological Institute (FMI)
        for a given time interval in chunks.

        Parameters:
            location (str): Bounding box coordinates for Finland.
            start_time (datetime): Start time for fetching data.
            end_time (datetime): End time for fetching data.
            chunk_hours (int): Number of hours per request chunk.
            max_retries (int): Maximum retries in case of API failures.

        Returns:
            tuple: (pd.DataFrame, pd.DataFrame) - DataFrame containing fetched weather observations and station metadata.
        """
        all_data = []
        station_metadata = {}
        current_time = start_time

        while current_time < end_time:
            chunk_end = min(current_time + timedelta(hours=chunk_hours), end_time)
            start_time_iso = current_time.isoformat() + "Z"
            end_time_iso = chunk_end.isoformat() + "Z"

            print(f"Fetching FMI data from {start_time_iso} to {end_time_iso}")
            time.sleep(5)  # Delay to prevent rate limits

            query_args = [
                f"bbox={location}",
                f"starttime={start_time_iso}",
                f"endtime={end_time_iso}"
            ]

            attempt = 1
            while attempt <= max_retries:
                try:
                    # Query the FMI data
                    obs = download_stored_query(FMI_OBSERVATIONS, args=query_args)

                    if not obs.data:
                        print(f"No data retrieved for {start_time_iso} - Skipping")
                        break

                    # Extract station metadata only once
                    if not station_metadata:
                        station_metadata = obs.location_metadata

                    data = []
                    for timestamp, station_data in obs.data.items():
                        for station_name, variables in station_data.items():
                            row = {"timestamp": timestamp, "station_name": station_name}
                            row.update({param: values["value"] for param, values in variables.items()})
                            data.append(row)

                    df_data = pd.DataFrame(data)
                    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"])
                    all_data.append(df_data)
                    break
                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        time.sleep(10 * attempt)
                    else:
                        print(f"Skipping {start_time_iso} after {max_retries} failed attempts.")
                attempt += 1

            current_time = chunk_end

        df_data_combined = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        df_metadata = pd.DataFrame.from_dict(station_metadata, orient="index").reset_index() if station_metadata else pd.DataFrame()
        if not df_metadata.empty:
            df_metadata.rename(columns={"index": "station_name"}, inplace=True)

        return df_data_combined, df_metadata


    def fetch_fmi_by_interval(self, location, start_date, end_date):
        """
        Fetches FMI data for a given date range by calling fetch_fmi_data for each day.
        Saves data monthly and returns only EMS metadata.

        Parameters:
            location (str): Bounding box coordinates for Finland.
            start_date (str or datetime.date): Start date for fetching data.
            end_date (str or datetime.date): End date for fetching data.

        Returns:
            pd.DataFrame: EMS metadata DataFrame.
        """
        # Ensure start_date and end_date are datetime.date objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        all_fmi_data = []
        ems_metadata = []  # Store station metadata (only once)
        current_date = start_date

        while current_date <= end_date:
            print(f"Fetching FMI data for {current_date}...")

            # Fetch data for the current date
            daily_fmi_data, daily_ems_metadata = self.fetch_fmi_data(
                location,
                datetime.combine(current_date, datetime.min.time()),  # Start of day
                datetime.combine(current_date, datetime.min.time()) + timedelta(hours=23, minutes=59, seconds=59)  # End of day
            )

            if not daily_fmi_data.empty:
                all_fmi_data.append(daily_fmi_data)  # Store weather data

                # Store metadata only once
                if not ems_metadata and not daily_ems_metadata.empty:
                    ems_metadata.append(daily_ems_metadata)

            # Check if the month has changed or if it's the last day in range
            if (
                all_fmi_data
                and (current_date.month != (current_date + timedelta(days=1)).month or current_date == end_date)
            ):
                fmi_data_combined = pd.concat(all_fmi_data, ignore_index=True)
                self.save_monthly_to_csv(fmi_data_combined, CSV_FMI, current_date.year, current_date.month)
                all_fmi_data = []  # Reset for the new month

            current_date += timedelta(days=1)

        # Combine EMS metadata into a single DataFrame
        ems_metadata_combined = pd.concat(ems_metadata, ignore_index=True) if ems_metadata else pd.DataFrame()

        return ems_metadata_combined