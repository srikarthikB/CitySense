from pathlib import Path
import pickle
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "models"

grid_lookup = None
grid_history = None
grid_hour_history = None
grid_day_history = None
grid_time_history = None
grid_weekend_history = None
GRID_CATEGORIES = None
TIME_CATEGORIES = pd.Index(["Afternoon", "Evening", "Morning", "Night"])

def load_lookups():
    global grid_lookup
    global grid_history
    global grid_hour_history
    global grid_day_history
    global grid_time_history
    global grid_weekend_history
    global GRID_CATEGORIES
    if grid_lookup is None:
        with open(MODELS_DIR / "grid_lookup.pkl", "rb") as file:
            grid_lookup = pickle.load(file)
        with open(MODELS_DIR / "historical_lookups.pkl", "rb") as file:
            (
                grid_history,
                grid_hour_history,
                grid_day_history,
                grid_time_history,
                grid_weekend_history
            ) = pickle.load(file)
        GRID_CATEGORIES = grid_lookup["grid_id"].astype("category").cat.categories

def get_nearest_grid(latitude, longitude):
    load_lookups()
    distances = (
        (grid_lookup["lat_grid"] - latitude) ** 2
        + (grid_lookup["lon_grid"] - longitude) ** 2
    )
    idx = distances.idxmin()
    row = grid_lookup.loc[idx]
    return row["grid_id"], row["lat_grid"], row["lon_grid"]

def create_inference_row(latitude, longitude, date, hour):
    load_lookups()
    timestamp = pd.Timestamp(date)
    grid_id, lat_grid, lon_grid = get_nearest_grid(latitude, longitude)
    day_of_week = timestamp.dayofweek
    if hour < 5:
        time_period = "Night"
    elif hour < 12:
        time_period = "Morning"
    elif hour < 18:
        time_period = "Afternoon"
    else:
        time_period = "Evening"
    is_weekend = int(day_of_week >= 5)
    historical_grid_crime_count = grid_history.get(grid_id, 0)
    historical_grid_hour_crime_count = grid_hour_history.get((grid_id, hour), 0)
    historical_grid_day_crime_count = grid_day_history.get((grid_id, day_of_week), 0)
    historical_grid_time_period_crime_count = grid_time_history.get((grid_id, time_period), 0)
    historical_grid_weekend_crime_count = grid_weekend_history.get((grid_id, is_weekend), 0)
    row = pd.DataFrame([{
        "grid_id": grid_id,
        "hour": hour,
        "historical_grid_crime_count": historical_grid_crime_count,
        "historical_grid_hour_crime_count": historical_grid_hour_crime_count,
        "day_of_week": day_of_week,
        "time_period": time_period,
        "historical_grid_day_crime_count": historical_grid_day_crime_count,
        "historical_grid_time_period_crime_count": historical_grid_time_period_crime_count,
        "is_weekend": is_weekend,
        "historical_grid_weekend_crime_count": historical_grid_weekend_crime_count,
        "year": timestamp.year,
        "month": timestamp.month,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid
    }])
    row["grid_id"] = pd.Categorical(row["grid_id"], categories=GRID_CATEGORIES)
    row["time_period"] = pd.Categorical(row["time_period"], categories=TIME_CATEGORIES)
    return row