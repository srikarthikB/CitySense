from pathlib import Path
import pickle
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent / "models" / "grid_lookup.pkl"

path = DATA_DIR / "citysense_2025.parquet"
df = pd.read_parquet(path, columns=["grid_id", "lat_grid", "lon_grid"])
grid_lookup = df.drop_duplicates("grid_id").reset_index(drop=True)

with open(OUTPUT_PATH, "wb") as file:
    pickle.dump(grid_lookup, file)

print(f"Saved grid lookup to {OUTPUT_PATH}")