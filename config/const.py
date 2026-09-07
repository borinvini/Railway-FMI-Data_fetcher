import os

# Load environment variables from a local .env file, if present.
# .env is git-ignored; see .env.example for the expected keys. Values already
# set in the real environment win, so exporting a variable in the shell
# overrides the file. Absent python-dotenv or .env, os.getenv still reads the
# process environment and everything below degrades gracefully.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# PARAMETERS
START_DATE = "2024-01-01" # YYYY-MM-DD
END_DATE = "2024-01-01" # YYYY-MM-DD

# Delay threshold (in minutes) for considering a stop as delayed
DELAY_LONG_DISTANCE_TRAINS = 5

# List of mandatory stations for long-distance train route analysis
FILTER_BY_ROUTE = False  # Set to True to filter trains by mandatory stations, False to include all trains
MANDATORY_STATIONS = ["HKI", "OL", "ROI"]  # Helsinki, Oulu, and Rovaniemi - trains must pass through ALL these stations

# Train filtering parameters
FILTER_BY_TRAIN_CATEGORY = True  # Set to True to filter. False to include all train categories
TRAIN_CATEGORY_FILTER = "Long-distance"
# Available train categories: "Long-distance", "Commuter", "Cargo", "Locomotive", "Test drive", "On-track machines", "Shunting"

# 
FMI_BBOX = "18,55,35,75" # Bounding box for Finland
#FMI_BBOX = "20.8,59.4,27.2,67.6"

# Alternative weather data search parameters
ALTERNATIVE_WEATHER_RADIUS_KM = 50  # Maximum radius in kilometers for alternative weather station search

# All instant (non-rolling) FMI weather parameters — used to drive per-feature top-5 fallback
FMI_INSTANT_PARAMS = [
    "Air temperature",
    "Wind speed",
    "Gust speed",
    "Wind direction",
    "Relative humidity",
    "Dew-point temperature",
    "Precipitation amount",
    "Precipitation intensity",
    "Snow depth",
    "Pressure (msl)",
    "Horizontal visibility",
    "Cloud amount",
    "Present weather (auto)",
]

# FMI Weather preprocessing parameters
FMI_ROLLING_WINDOW_HOURS = [12, 24, 72]  # Rolling window sizes in hours

# Weather parameters to apply rolling window statistics
FMI_ROLLING_WINDOW_PARAMS = [
    "Air temperature",
    "Wind speed",
    "Relative humidity",
    "Precipitation intensity",
    "Snow depth",
    "Pressure (msl)",
    "Horizontal visibility",
    "Cloud amount",
    "Precipitation amount"
]

# Parameters that skip min/max (only get mean and cumulative)
# Precipitation amount already represents 1h accumulated values
FMI_ROLLING_SKIP_MIN_MAX = ["Precipitation amount"]

# Only these parameters get a cumulative (sum) rolling feature
FMI_ROLLING_INCLUDE_CUMULATIVE = ["Precipitation amount"]

# Helper function to generate column names for rolling features
def get_fmi_rolling_column_names(param_name, window_hours, skip_min_max=False, skip_cumulative=False):
    """
    Generate standardized column names for rolling window statistics.

    Args:
        param_name (str): The base parameter name (e.g., "Air temperature")
        window_hours (int): The rolling window size in hours
        skip_min_max (bool): If True, omit max and min keys
        skip_cumulative (bool): If True, omit cumulative key

    Returns:
        dict: Dictionary with keys for each statistic and their column names
    """
    names = {}
    if not skip_min_max:
        names['max'] = f"{param_name} ({window_hours}h max)"
        names['min'] = f"{param_name} ({window_hours}h min)"
    names['mean'] = f"{param_name} ({window_hours}h mean)"
    if not skip_cumulative:
        names['cumulative'] = f"{param_name} ({window_hours}h cumulative)"
    return names


# URLs for the Finnish Meteorological Institute API
FMI_WFS_BASE = "https://opendata.fmi.fi/wfs"
FMI_OBSERVATIONS = "fmi::observations::weather::multipointcoverage"
FMI_EMS = "fmi::ef::stations"

# EF registry networks that count as a usable weather source.
# The registry holds 441 facilities, most of which are not weather stations
# (149 are third-party air quality monitors, 14 are tide gauges, 8 are
# radioactivity monitors). Without this filter a train station could be
# matched to a radiation monitor as its nearest "weather" source.
FMI_WEATHER_NETWORKS = (
    "Automaattinen sääasema",
    "IL:n hallinnoima lentosääasema",
    "Sääasema",
    "Sadeasema",
)

# URLs for the Finnish Railway API
FIN_RAILWAY_BASE_URL = "https://rata.digitraffic.fi/api/v1"
FIN_RAILWAY_STATIONS = "/metadata/stations"
FIN_RAILWAY_TRAIN_CAT = "/metadata/train-categories"
FIN_RAILWAY_TRAIN_CAUSES = "/metadata/cause-category-codes"
FIN_RAILWAY_TRAIN_CAUSES_DETAILED = "/metadata/detailed-cause-category-codes"
FIN_RAILWAY_TRAIN_THIRD_CAUSES = "/metadata/third-cause-category-codes"
FIN_RAILWAY_ALL_TRAINS = "/trains"
FIN_RAILWAY_TRAIN_TRACKING = "/train-tracking"

# CSVs
FOLDER_NAME = "data"

CSV_TRAIN_STATIONS = "metadata_train_stations.csv"
CSV_TRAIN_CATEGORIES = "metadata_train_categories.csv"
CSV_TRAIN_CAUSES = "metadata_train_causes.csv"  
CSV_TRAIN_CAUSES_DETAILED = "metadata_train_causes_detailed.csv"  
CSV_TRAIN_THIRD_CAUSES = "metadata_third_train_causes.csv"
CSV_ALL_TRAINS = "all_trains_data.csv"

CSV_FMI = "fmi_weather_observations.csv"
CSV_FMI_EMS = "metadata_fmi_ems_stations.csv"
CSV_FMI_EF_REGISTRY = "metadata_fmi_ef_registry.csv"
CSV_CLOSEST_EMS_TRAIN = "metadata_closest_ems_to_train_stations.csv"
CSV_TOP5_CLOSEST_EMS_TRAIN = "metadata_top5_closest_ems.csv"
CSV_MATCHED_DATA = "matched_data.csv"
CSV_ALL_TRAINS_FLAT = "all_trains_data_flat.csv"
CSV_MATCHED_DATA_FLAT = "matched_data_flat.csv"

# Parquet base filenames (monthly files: {base}_{YYYY}_{MM}.parquet)
PARQUET_ALL_TRAINS_FLAT   = "all_trains_data_flat.parquet"
PARQUET_FMI               = "fmi_weather_observations.parquet"
PARQUET_MATCHED_DATA_FLAT = "matched_data_flat.parquet"

CSV_DELAY_TABLE_ORIGINAL = "delay_table_differenceInMinutes.csv"
CSV_DELAY_TABLE_OFFSET = "delay_table_differenceInMinutes_offset.csv" 
CSV_DELAY_TABLE_EACH_STATION = "delay_table_differenceInMinutes_eachStation_offset.csv"

# Day of week mapping
DAY_OF_WEEK_MAPPING = {
    1: "Monday",
    2: "Tuesday", 
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday"
}

# Email notification configuration
# Credentials are read from the environment, never stored in this file.
# Set them in a local .env (git-ignored) or export them in your shell:
#   FETCHER_SMTP_SERVER   (default: smtp.gmail.com)
#   FETCHER_SMTP_PORT     (default: 587)
#   FETCHER_EMAIL_ADDRESS
#   FETCHER_EMAIL_PASSWORD
# If EMAIL_ADDRESS or EMAIL_PASSWORD is unset, notifications are skipped silently.
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_SERVER = os.getenv('FETCHER_SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('FETCHER_SMTP_PORT', '587'))
EMAIL_ADDRESS = os.getenv('FETCHER_EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('FETCHER_EMAIL_PASSWORD')


def send_email(subject, body):
    """
    Send a progress notification email.

    No-op when FETCHER_EMAIL_ADDRESS / FETCHER_EMAIL_PASSWORD are not set, so the
    pipeline runs unchanged on machines without mail configured.
    """
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("[info] Email notifications not configured (set FETCHER_EMAIL_ADDRESS "
              "and FETCHER_EMAIL_PASSWORD). Skipping.")
        return

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS  # Send to yourself or modify to any recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the server and send the email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())
        server.quit()
        print("[ok] Email sent successfully.")
    except Exception as e:
        print(f"[warn] Failed to send email: {e}")