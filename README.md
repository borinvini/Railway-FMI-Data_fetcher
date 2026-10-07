<p align="center"><h1 align="center">RAILWAY-FMI-DATA_FETCHER</h1></p>
<p align="center">
	<em><code>❯ Data fetcher and merge from Finnish institutions </code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/borinvini/Railway-FMI-Data_fetcher.git?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/languages/top/borinvini/Railway-FMI-Data_fetcher.git?style=default&color=0080ff" alt="repo-top-language">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

---

## Overview  

❯ **Railway-FMI-Data-Fetcher** is a Python-based project designed to fetch and process data from two key open data sources in Finland:  
- **[Digitraffic API](https://www.digitraffic.fi/)** – Provides Finnish railway data, including real-time train timetables, station metadata, and operational status.  
- **[FMI (Finnish Meteorological Institute) API](https://github.com/pnuu/fmiopendata)** – Supplies weather observation data from environmental monitoring stations (EMS) across Finland.  

---

> **Disclaimer:**  
> There is a complementary GitHub repository for this project, designed to visualize the data fetched here using a Streamlit interface.
<br> You can find it at: ➡️ [Railway-FMI-Data-Viewer](https://github.com/borinvini/Railway-FMI-Data_viewer.git)  

---

##  Project Goal

❯ This project aims to create a unified dataset that merges railway timetable information with weather data, enabling more comprehensive analysis of how weather conditions affect train operations. The data is matched using the Haversine formula to identify the closest weather station to each train track. The train track can be a passanger station, or not.  

The project includes a data processing pipeline that:  
✔️ Collects data from the APIs.  
✔️ Merges railway and weather data based on location and time.  
✔️ Handles missing data and outliers.  
✔️ Saves the processed data to structured CSV files.  


---


##  Project Roadmap

- [X] **`Task 1`**: <strike>Railway data fetcher.</strike>
- [X] **`Task 2`**: <strike>FMI Weather data fetcher.</strike>
- [X] **`Task 3`**: <strike>Railway and weather matched data using haversine.</strike>


---


##  Project Structure

```sh
├── config
│   ├── const.py               # Configuration file for constants and paths
├── data                       # Fetched and output data (metadata_*.csv at the top level)
│   ├── train                  # Monthly train files (raw and flat)
│   │   ├── parquet            # Parquet copies of the flat train files
│   ├── weather                # Monthly FMI weather files
│   │   ├── parquet            # Parquet copies of the weather files
│   ├── matched                # Matched train+weather files and delay tables
│   │   ├── parquet            # Parquet copies of the flat matched files
├── logs                       # Directory for logs
├── src
│   ├── fetchers               # Fetchers for Railway and FMI data
│   │   ├── FMI.py             # Class Handler FMI data fetching
│   │   ├── Railway.py         # Class Handler Railway data fetching
│   ├── processors             # Data processing logic
│   │   ├── DataLoader.py      # Class for Loads and processes merged data                
├── environment.yml            # Conda environment file
├── main.py                    # Main script to execute data fetching and processing               
```


---
##  Getting Started


**Using `conda` environment** &nbsp; [<img align="center" src="https://img.shields.io/badge/conda-342B029.svg?style={badge_style}&logo=anaconda&logoColor=white" />](https://docs.conda.io/)

```sh
❯ conda venv create -f environment.yml
```

## 🚀 Data Fetcher Flag  
The `DATA_FETCH` flag in `main.py` controls whether the program will **fetch data from the APIs** or **process existing data**.

### 1. **Fetch Data from APIs**  
To fetch data from the Finnish Railway API and FMI API, set the `DATA_FETCH` flag to `True` in `main.py`:

```python
# main.py
DATA_FETCH = True
```

When `DATA_FETCH` is set to `True`, the program will:

✅ Fetch train data within the specified interval and save it to a CSV.  
✅ Fetch weather data from the FMI API within the specified interval and save it to a CSV.  

This mode is used to **download fresh data** from the APIs and **store it locally** for further processing.

### 2. Process Existing Data  
To **process and merge previously fetched data** (without making additional API calls), set the `DATA_FETCH` flag to `False` in `main.py`:

```python
# main.py
DATA_FETCH = False
```

When `DATA_FETCH` is set to `False`, the program will: 

✅ Load data from the locally stored CSV files.  
✅ Match train timetable data with the closest weather station data using the **Haversine distance** calculation.  
✅ Merge the matched data into a structured format for analysis.  

---

##  Datasets 

**Train-Level Fields**
- **trainNumber** – *Integer* – Train identifier for a specific departure date  
- **departureDate** – *String* – Date of the train's first departure  
- **operatorUICCode** – *Integer* – UIC code of the operator  
- **operatorShortCode** – *String* – Short code of the operator  
- **trainType** – *String* – Type of train (e.g., IC)  
- **trainCategory** – *String* – Category of train (e.g., Long-distance)  
- **commuterLineID** – *String* – Line identifier (e.g., Z)  
- **runningCurrently** – *Boolean* – Whether the train is currently running  
- **cancelled** – *Boolean* – Whether the train is fully cancelled  
- **deleted** – *Boolean* – Whether the train was cancelled 10 days before departure  
- **version** – *Integer* – Timestamp/version of last modification  
- **timetableType** – *String* – Type of timetable (e.g., REGULAR, ADHOC)  
- **timetableAcceptanceDate** – *String* – Date when the train was accepted to run  
- **timeTableRows** – *List of schedule entries*  
    - **stationShortCode** – *String* – Short code of the station  
    - **stationUICCode** – *Integer* – UIC code of the station  
    - **countryCode** – *String* – Country code (e.g., FI)  
    - **type** – *String* – Type of stop (e.g., ARRIVAL, DEPARTURE)  
    - **trainStopping** – *Boolean* – Whether the train stops at the station  
    - **commercialStop** – *Boolean* – Whether the stop is commercial (for passengers/cargo)  
    - **commercialTrack** – *String* – Track where the train stops  
    - **cancelled** – *Boolean* – Whether the schedule part is cancelled  
    - **scheduledTime** – *String* – Scheduled time for the stop  
    - **liveEstimateTime** – *String* – Estimated time for the stop  
    - **estimateSource** – *String* – Source of the estimate  
    - **unknownDelay** – *Boolean* – If the delay is unknown  
    - **actualTime** – *String* – Actual time the train arrived or departed  
    - **differenceInMinutes** – *Integer* – Difference between scheduled and actual time in minutes  


**Weather Dataset Fields**
- **timestamp** – *String* – Timestamp of the observation in `YYYY-MM-DD HH:MM:SS` format  
- **station_name** – *String* – Name of the weather station where the observation was recorded  
- **Air temperature** – *Float* – Air temperature at the time of observation in degrees Celsius (°C)  
- **Wind speed** – *Float* – Average wind speed during the observation period in meters per second (m/s)  
- **Gust speed** – *Float* – Maximum gust speed during the observation period in meters per second (m/s)  
- **Wind direction** – *Float* – Direction of the wind in degrees (0–360)  
- **Relative humidity** – *Float* – Percentage of humidity in the air (%)  
- **Dew-point temperature** – *Float* – Temperature at which air becomes saturated with moisture in degrees Celsius (°C)  
- **Precipitation amount** – *Float* – Total amount of precipitation (rain, snow, etc.) during the observation period in millimeters (mm)  
- **Precipitation intensity** – *Float* – Intensity of precipitation during the observation period in millimeters per hour (mm/h)  
- **Snow depth** – *Float* – Depth of snow on the ground at the time of observation in centimeters (cm)  
- **Pressure (msl)** – *Float* – Atmospheric pressure at mean sea level (MSL) in hectopascals (hPa)  
- **Horizontal visibility** – *Float* – Horizontal visibility at the observation site in meters (m)  
- **Cloud amount** – *Float* – Cloud coverage as a fraction of the sky (0–8)  




