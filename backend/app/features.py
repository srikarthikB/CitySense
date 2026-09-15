from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

def build_historical_lookups():
    grid_history = []
    grid_hour_history = []
    grid_day_history = []
    grid_time_history = []
    grid_weekend_history = []

    for year in range(2020, 2026):
        path = DATA_DIR / f"citysense_{year}.parquet"
        df = pd.read_parquet(
            path,
            columns=[
                "grid_id",
                "crime_count",
                "hour",
                "day_of_week",
                "time_period",
                "is_weekend"
            ]
        )

        grid_history.append(
            df.groupby("grid_id")["crime_count"].sum()
        )
        grid_hour_history.append(
            df.groupby(["grid_id", "hour"])["crime_count"].sum()
        )
        grid_day_history.append(
            df.groupby(["grid_id", "day_of_week"])["crime_count"].sum()
        )
        grid_time_history.append(
            df.groupby(["grid_id", "time_period"])["crime_count"].sum()
        )
        grid_weekend_history.append(
            df.groupby(["grid_id", "is_weekend"])["crime_count"].sum()
        )

    return (
        pd.concat(grid_history).groupby(level=0).sum(),
        pd.concat(grid_hour_history).groupby(level=[0, 1]).sum(),
        pd.concat(grid_day_history).groupby(level=[0, 1]).sum(),
        pd.concat(grid_time_history).groupby(level=[0, 1]).sum(),
        pd.concat(grid_weekend_history).groupby(level=[0, 1]).sum()
    )


def build_grid_lookup():
    path = DATA_DIR / "citysense_2025.parquet"
    df = pd.read_parquet(path, columns=["grid_id", "lat_grid", "lon_grid"])
    return df.drop_duplicates("grid_id").reset_index(drop=True)


def get_nearest_grid(latitude, longitude, grid_lookup):
    distances = (
        (grid_lookup["lat_grid"] - latitude) ** 2
        + (grid_lookup["lon_grid"] - longitude) ** 2
    )
    idx = distances.idxmin()
    row = grid_lookup.loc[idx]
    return row["grid_id"], row["lat_grid"], row["lon_grid"]


grid_history, grid_hour_history, grid_day_history, grid_time_history, grid_weekend_history = build_historical_lookups()
grid_lookup = build_grid_lookup()

GRID_CATEGORIES = grid_lookup["grid_id"].astype("category").cat.categories
TIME_CATEGORIES = pd.Index(["Afternoon", "Evening", "Morning", "Night"])


def create_inference_row(latitude, longitude, date, hour):
    timestamp = pd.Timestamp(date)
    grid_id, lat_grid, lon_grid = get_nearest_grid(latitude, longitude, grid_lookup)
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