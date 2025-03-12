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
- **[FMI (Finnish Meteorological Institute) API](https://en.ilmatieteenlaitos.fi/open-data)** – Supplies weather observation data from environmental monitoring stations (EMS) across Finland.  


---

##  Project Goal

❯ This project aims to create a unified dataset that merges railway timetable information with weather data, enabling more comprehensive analysis of how weather conditions affect train operations. The data is matched using the **Haversine formula** to identify the closest weather station to each train track. The train track can be a passanger station, or not.  

The project includes a data processing pipeline that:  
✔️ Collects data from the APIs.  
✔️ Merges railway and weather data based on location and time.  
✔️ Handles missing data and outliers.  
✔️ Saves the processed data to structured CSV files.  


---

---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Railway data fetcher.</strike>
- [X] **`Task 2`**: <strike>FMI Weather data fetcher.</strike>
- [X] **`Task 3`**: <strike>Railway and weather matched data using haversine.</strike>


##  Project Structure

```sh
├── config
│   ├── const.py               # Configuration file for constants and paths
├── data                       # Directory to store CSV files (fetched data and output data)
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



