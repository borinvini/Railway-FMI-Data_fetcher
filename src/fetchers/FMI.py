import os
import time
import pandas as pd
from datetime import datetime, timedelta
from fmiopendata.wfs import download_stored_query

from config.const import FMI_OBSERVATIONS, FMI_EMS, CSV_FMI, CSV_FMI_EMS, FOLDER_NAME

# Pinned schema for metadata_fmi_ems_stations.csv. The first four columns are the
# file's published shape and must keep this order; the remaining five are additive
# provenance. Written once from a single DataFrame, never appended chunk by chunk.
STATION_COLUMNS = [
    "station_name", "fmisid", "latitude", "longitude",
    "networks", "is_weather_station", "in_ef_registry", "coord_source",
    "observed_in_run",
]


def reconcile_station_metadata(observed, ef_registry):
    """
    Builds the EMS candidate pool from the observation feed and the EF registry.

    This is an outer union: the pool is every EF weather station plus every station
    seen in observations, whether or not the two overlap. The pool is geography, not
    evidence of data — a station commissioned in 2024 belongs in it even when merging
    2018, because merge-time liveness resolution walks the candidate ranks and skips
    anything silent for the month.

    Non-weather facilities from the registry (tide gauges, air quality monitors,
    radiation monitors) are excluded: 187 of the registry's 441 entries. Stations the
    registry omits but observations show are kept, because catalogue absence is a
    registry gap, not evidence against a station that is demonstrably reporting.

    Args:
        observed (pd.DataFrame): Stations seen in the observation feed, with columns
            station_name, fmisid, latitude, longitude.
        ef_registry (pd.DataFrame): EF catalogue from FMIStationRegistry.fetch_registry().
            May be empty if the registry was unreachable.

    Returns:
        pd.DataFrame: Weather stations with columns STATION_COLUMNS.

    Raises:
        ValueError: If no station metadata was collected at all, or if the registry
            contains a duplicate fmisid.
    """
    if observed is None or observed.empty:
        raise ValueError(
            "No EMS station metadata was collected from the observation feed. "
            "Refusing to write an empty station table, which would silently leave "
            "a stale metadata_fmi_ems_stations.csv in place from an earlier run "
            "with a different date range or bounding box."
        )

    stations = observed.copy()

    if ef_registry is None or ef_registry.empty:
        print("⚠️ EF registry unavailable — station typing and coordinates come from observations only.")
        stations["networks"] = pd.NA
        stations["is_weather_station"] = True
        stations["in_ef_registry"] = pd.NA
        stations["coord_source"] = "observation"
        stations["observed_in_run"] = True
        return stations[STATION_COLUMNS].reset_index(drop=True)

    registry = ef_registry[["fmisid", "station_name", "latitude", "longitude", "networks", "is_weather_station"]]

    # A duplicate fmisid in the registry would fan out the join and silently
    # duplicate a station in the matching table. gml:identifier is unique per
    # facility, so a duplicate means the response is malformed — stop rather than
    # quietly dedupe and hide it.
    if registry["fmisid"].duplicated().any():
        duplicated = registry.loc[registry["fmisid"].duplicated(), "fmisid"].tolist()
        raise ValueError(
            f"EF registry contains duplicate fmisid values {duplicated}. "
            "Refusing to write a station table with duplicated stations."
        )

    merged = stations.merge(registry, on="fmisid", how="outer", suffixes=("", "_ef"), indicator=True)

    merged["observed_in_run"] = merged["_merge"].isin(["left_only", "both"])
    merged["in_ef_registry"] = merged["_merge"].isin(["right_only", "both"])
    merged["coord_source"] = merged["in_ef_registry"].map({True: "ef", False: "observation"})

    # The registry is authoritative for station position where it has one; a
    # registry-only station has no observation coordinate to fall back to.
    for axis in ("latitude", "longitude"):
        merged[axis] = merged[f"{axis}_ef"].combine_first(merged[axis])
    merged["station_name"] = merged["station_name_ef"].combine_first(merged["station_name"])

    # Stations absent from the registry are demonstrably producing weather
    # observations, so treat catalogue absence as a registry gap, not as evidence
    # against the station.
    merged["is_weather_station"] = (
        merged["is_weather_station"].astype("boolean").fillna(True).astype(bool)
    )

    excluded = merged[~merged["is_weather_station"]]
    for _, station in excluded.iterrows():
        print(f"⚠️ Excluding non-weather station '{station['station_name']}' ({station['networks']}).")

    absent = merged[merged["is_weather_station"] & ~merged["in_ef_registry"]]
    if not absent.empty:
        names = ", ".join(sorted(absent["station_name"].astype(str)))
        print(f"ℹ️ {len(absent)} observed stations absent from the EF registry: {names}")

    kept = merged[merged["is_weather_station"]]
    print(
        f"✅ Station pool: {len(kept)} stations "
        f"({int(kept['in_ef_registry'].sum())} registry-confirmed, "
        f"{int(kept['observed_in_run'].sum())} observed this run)."
    )
    return kept[STATION_COLUMNS].reset_index(drop=True)


class FMIDataFetcher:
    def __init__(self):
        """
        Initializes the FMIDataFetcher class.
        """
        self.base_url = FMI_OBSERVATIONS
        self.ems_url = FMI_EMS
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

    def save_station_metadata(self, df, filename):
        """
        Saves the station metadata table as a union with any existing file at that
        path, refusing to write an empty one.

        save_to_csv treats an empty frame as a no-op and returns, which leaves any
        previous run's file untouched. For station metadata that is a silent
        correctness failure: matching would proceed against a station table built
        from a different date range or bounding box. That empty-table refusal is
        retained here, and still fires before any file is read.

        When a pool file already exists at filepath, the new rows are unioned into
        it rather than overwriting it: two runs over disjoint months converge on one
        pool instead of leaving whichever ran last as the only survivor. This is
        safe because merge-time liveness resolution walks the candidate pool per
        month and skips any station silent for that month, so carrying a station
        that wasn't observed in a given run costs nothing.

        `observed_in_run` is OR-accumulated across the union: a station observed in
        any earlier run stays observed even if a later run didn't see it, because
        the column means "has ever produced data", not "produced data this run".
        Where a station appears with both a registry-sourced and an observation-
        sourced coordinate, the registry row wins regardless of write order, so a
        later registry coordinate correction propagates.

        A pool file written before this union existed will not have
        `observed_in_run`. Such a file was built from the observation feed alone, so
        every station in it was in fact observed by construction — that column is
        backfilled as True before the union runs.

        Args:
            df (pd.DataFrame): The station table to save.
            filename (str): Name of the CSV file.

        Raises:
            ValueError: If df is empty.
        """
        if df is None or df.empty:
            raise ValueError(
                f"Refusing to write an empty station table to '{filename}'. "
                "An earlier run's file would otherwise remain in place and be "
                "silently used for matching."
            )

        filepath = os.path.join(self.output_folder, filename)

        if os.path.exists(filepath):
            previous = pd.read_csv(filepath)
            if "observed_in_run" not in previous.columns:
                # Files written before the pool became a union came from the
                # observation feed alone, so every station in them was observed
                # by construction.
                previous["observed_in_run"] = True
            combined = pd.concat([previous, df], ignore_index=True)

            # A station observed in any earlier run stays observed; the column means
            # "has ever produced data in a fetched range", not "produced data this run".
            observed = combined.groupby("fmisid")["observed_in_run"].any()

            # Registry coordinates are authoritative, so a row sourced from the
            # registry wins over one sourced from observations regardless of order.
            combined["_ef_first"] = (combined["coord_source"] != "ef").astype(int)
            combined = combined.sort_values(["fmisid", "_ef_first"], kind="stable")
            deduped = combined.drop_duplicates("fmisid", keep="first").drop(columns="_ef_first")

            deduped["observed_in_run"] = deduped["fmisid"].map(observed)
            df = deduped.sort_values("fmisid").reset_index(drop=True)
            print(f"ℹ️ Pool union: {len(previous)} existing + this run → {len(df)} stations.")

        df[STATION_COLUMNS].to_csv(filepath, index=False, encoding="utf-8")
        print(f"Station metadata saved to {filepath} ({len(df)} stations)")

    def save_monthly_data_to_csv(self, df, base_filename, year, month):
        """
        Saves the DataFrame in a monthly format: `filename_YYYY_MM.csv`.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            base_filename (str): Base filename without extension.
            year (int): Year of the data.
            month (int): Month of the data.
        """
        if df is not None and not df.empty:
            # Remove existing ".csv" extension if present in the base filename
            if base_filename.endswith('.csv'):
                base_filename = base_filename[:-4]  # Remove last 4 characters (".csv")

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

                    # Merge station metadata from every chunk. Capturing only the
                    # first chunk made the station table a snapshot of one hour,
                    # so any station silent during that hour was permanently
                    # excluded from distance matching.
                    station_metadata.update(obs.location_metadata)

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
        ems_metadata = {}  # fmisid -> station metadata row, accumulated over the range
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

            # Accumulate station metadata independently of whether this day
            # returned observation rows, and across every day in the range.
            if not daily_ems_metadata.empty:
                for station_row in daily_ems_metadata.to_dict("records"):
                    ems_metadata[station_row["fmisid"]] = station_row

            # Check if the month has changed or if it's the last day in range
            if (
                all_fmi_data
                and (current_date.month != (current_date + timedelta(days=1)).month or current_date == end_date)
            ):
                fmi_data_combined = pd.concat(all_fmi_data, ignore_index=True)
                self.save_monthly_data_to_csv(fmi_data_combined, CSV_FMI, current_date.year, current_date.month)
                all_fmi_data = []  # Reset for the new month

            current_date += timedelta(days=1)

        # Combine EMS metadata into a single DataFrame
        ems_metadata_combined = (
            pd.DataFrame(list(ems_metadata.values()))
            if ems_metadata
            else pd.DataFrame(columns=["station_name", "fmisid", "latitude", "longitude"])
        )

        return ems_metadata_combined