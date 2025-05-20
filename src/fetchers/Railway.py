import ast
from datetime import datetime, timedelta
import requests
import pandas as pd
import os
from deep_translator import GoogleTranslator

from config.const import CSV_ALL_TRAINS, FIN_RAILWAY_ALL_TRAINS, FIN_RAILWAY_BASE_URL, FIN_RAILWAY_STATIONS, FIN_RAILWAY_TRAIN_CAT, FIN_RAILWAY_TRAIN_CAUSES, FIN_RAILWAY_TRAIN_CAUSES_DETAILED, FIN_RAILWAY_TRAIN_THIRD_CAUSES, FOLDER_NAME

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
    
    def fetch_train_categories_metadata(self):
        """
        Fetch and return train categories metadata as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing train categories metadata.
        """
        print(f"Fetching train categories metadata from {self.base_url}{FIN_RAILWAY_TRAIN_CAT}...")
        data = self.get_data(FIN_RAILWAY_TRAIN_CAT)

        if not data:
            print("No train categories metadata available. Please check the API or data source.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            print("Fetched train categories metadata is empty.")
            return pd.DataFrame()

        print("Train categories metadata successfully loaded.")
        self.preview_dataframe(df, "🚆 Train Categories Metadata Preview")
        return df
    
    def fetch_cause_category_codes_metadata(self):
        """
        Fetch and return cause category codes metadata as a DataFrame.
        Adds English translations for the Finnish category names.

        Returns:
            pd.DataFrame: DataFrame containing cause category codes metadata with translations.
        """
        print(f"Fetching cause category codes metadata from {self.base_url}{FIN_RAILWAY_TRAIN_CAUSES}...")
        data = self.get_data(FIN_RAILWAY_TRAIN_CAUSES)

        if not data:
            print("No cause category codes metadata available. Please check the API or data source.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            print("Fetched cause category codes metadata is empty.")
            return pd.DataFrame()

        # Translate the categoryName column
        df = self.translate_finnish_to_english(df, "categoryName")

        print("Cause category codes metadata successfully loaded.")
        self.preview_dataframe(df, "🚨 Cause Category Codes Metadata Preview")
        return df

    def fetch_detailed_cause_category_codes_metadata(self):
        """
        Fetch and return detailed cause category codes metadata as a DataFrame.
        Adds English translations for the Finnish detailed category names.

        Returns:
            pd.DataFrame: DataFrame containing detailed cause category codes metadata with translations.
        """
        print(f"Fetching detailed cause category codes metadata from {self.base_url}{FIN_RAILWAY_TRAIN_CAUSES_DETAILED}...")
        data = self.get_data(FIN_RAILWAY_TRAIN_CAUSES_DETAILED)

        if not data:
            print("No detailed cause category codes metadata available. Please check the API or data source.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            print("Fetched detailed cause category codes metadata is empty.")
            return pd.DataFrame()
            
        # Translate the detailedCategoryName column
        df = self.translate_finnish_to_english(df, "detailedCategoryName")

        print("Detailed cause category codes metadata successfully loaded.")
        self.preview_dataframe(df, "🚨 Detailed Cause Category Codes Metadata Preview")
        return df

    def fetch_third_cause_category_codes_metadata(self):
        """
        Fetch and return third cause category codes metadata as a DataFrame.
        Adds English translations for the Finnish third category names.

        Returns:
            pd.DataFrame: DataFrame containing third cause category codes metadata with translations.
        """
        print(f"Fetching third cause category codes metadata from {self.base_url}{FIN_RAILWAY_TRAIN_THIRD_CAUSES}...")
        data = self.get_data(FIN_RAILWAY_TRAIN_THIRD_CAUSES)

        if not data:
            print("No third cause category codes metadata available. Please check the API or data source.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            print("Fetched third cause category codes metadata is empty.")
            return pd.DataFrame()

        # Translate the thirdCategoryName column
        df = self.translate_finnish_to_english(df, "thirdCategoryName")

        print("Third cause category codes metadata successfully loaded.")
        self.preview_dataframe(df, "🔍 Third Cause Category Codes Metadata Preview")
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

    def translate_finnish_to_english(self, df, source_column, target_column=None):
        """
        Translates Finnish text in a DataFrame column to English.
        
        Args:
            df (pd.DataFrame): DataFrame containing the column to translate
            source_column (str): Name of the column containing Finnish text
            target_column (str, optional): Name of the column to store translations.
                                        If None, will use source_column + '_en'
        
        Returns:
            pd.DataFrame: DataFrame with added translation column
        """
        if target_column is None:
            target_column = f"{source_column}_en"
        
        # Check if DataFrame is empty
        if df.empty:
            return df
            
        try:
            from deep_translator import GoogleTranslator
            import time
            
            print(f"Translating '{source_column}' from Finnish to English...")
            
            # Define translation function with error handling and retry logic
            def translate_text(text):
                if pd.isna(text) or text == "":
                    return ""
                    
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Add a delay between requests to avoid rate limiting
                        time.sleep(1.5)
                        
                        translation = GoogleTranslator(source='fi', target='en').translate(text)
                        return translation
                    except Exception as e:
                        print(f"Translation attempt {attempt+1} failed for '{text}': {e}")
                        # Wait longer between retries
                        time.sleep(3.0)
                
                # If all retries fail, return a fallback
                return f"{text} [untranslated]"
            
            # Apply translation to each item one at a time
            translations = []
            for idx, row in df.iterrows():
                text = row[source_column]
                print(f"Translating [{idx+1}/{len(df)}]: {text}")
                translated = translate_text(text)
                translations.append(translated)
                
            df[target_column] = translations
            print("✅ Translation completed successfully.")
            
        except ImportError:
            print("⚠️ deep_translator package not found. Install with: pip install deep-translator")
            df[target_column] = df[source_column] + " [untranslated]"
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
            df[target_column] = df[source_column] + " [translation error]"
            
        return df
