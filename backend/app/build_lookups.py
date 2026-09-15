from pathlib import Path
import pickle
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent / "models" / "historical_lookups.pkl"

grid_history = []
grid_hour_history = []
grid_day_history = []
grid_time_history = []
grid_weekend_history = []

for year in range(2020, 2026):
    path = DATA_DIR / f"citysense_{year}.parquet"
    df = pd.read_parquet(
        path,
        columns=["grid_id", "crime_count", "hour", "day_of_week", "time_period", "is_weekend"]
    )
    grid_history.append(df.groupby("grid_id")["crime_count"].sum())
    grid_hour_history.append(df.groupby(["grid_id", "hour"])["crime_count"].sum())
    grid_day_history.append(df.groupby(["grid_id", "day_of_week"])["crime_count"].sum())
    grid_time_history.append(df.groupby(["grid_id", "time_period"])["crime_count"].sum())
    grid_weekend_history.append(df.groupby(["grid_id", "is_weekend"])["crime_count"].sum())

lookups = (
    pd.concat(grid_history).groupby(level=0).sum(),
    pd.concat(grid_hour_history).groupby(level=[0, 1]).sum(),
    pd.concat(grid_day_history).groupby(level=[0, 1]).sum(),
    pd.concat(grid_time_history).groupby(level=[0, 1]).sum(),
    pd.concat(grid_weekend_history).groupby(level=[0, 1]).sum()
)

with open(OUTPUT_PATH, "wb") as file:
    pickle.dump(lookups, file)

print(f"Saved historical lookups to {OUTPUT_PATH}")