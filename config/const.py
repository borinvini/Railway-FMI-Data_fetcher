# PARAMETERS
START_DATE = "2020-01-01" # YYYY-MM-DD
END_DATE = "2022-12-31" # YYYY-MM-DD
CATEGORY = "All Categories" 
OPERATOR = "All Operators"

# 
FMI_BBOX = "18,55,35,75" # Bounding box for Finland
#FMI_BBOX = "20.8,59.4,27.2,67.6"

# URLs for the Finnish Meteorological Institute API
FMI_OBSERVATIONS = "fmi::observations::weather::multipointcoverage"
FMI_EMS = "fmi::ef::stations"

# URLs for the Finnish Railway API
FIN_RAILWAY_BASE_URL = "https://rata.digitraffic.fi/api/v1"
FIN_RAILWAY_STATIONS = "/metadata/stations"
FIN_RAILWAY_TRAIN_CAT = "/metadata/train-categories"
FIN_RAILWAY_TRAIN_CAUSES = "/metadata/detailed-cause-category-codes"
FIN_RAILWAY_TRAIN_THIRD_CAUSES = "/metadata/third-cause-category-codes"
FIN_RAILWAY_ALL_TRAINS = "/trains"
FIN_RAILWAY_TRAIN_TRACKING = "/train-tracking"

# CSVs
FOLDER_NAME = "data"

CSV_TRAIN_STATIONS = "metadata_train_stations.csv"
CSV_TRAIN_CATEGORIES = "metadata_train_categories.csv"
CSV_TRAIN_CAUSES = "metadata_train_causes.csv"
CSV_TRAIN_THIRD_CAUSES = "metadata_third_train_causes.csv"
CSV_ALL_TRAINS = "all_trains_data.csv"

CSV_FMI = "fmi_weather_observations.csv"
CSV_FMI_EMS = "metadata_fmi_ems_stations.csv"
CSV_CLOSEST_EMS_TRAIN = "metadata_closest_ems_to_train_stations.csv"
CSV_MATCHED_DATA = "matched_data.csv"


# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'borin.vini@gmail.com'  # Your email
EMAIL_PASSWORD = 'qaon nrrc yhsq vrbg'    # Use an app password if you have 2FA enabled

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body):
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
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")