"""
CitySense - corrected data-preparation pipeline, 2020 ONLY.

Fixes the bug in 05_model_preparation.ipynb where `grid_dates` was derived
from `target_df` (crime records only), silently dropping any grid-date that
had ZERO crimes across all 24 hours. This script builds the true
grid x date x hour universe for 2020, joins in actual crime counts (0 where
none occurred), and recomputes the historical features on that complete,
chronologically-sorted universe so they stay leak-free.

Memory strategy: process the 946 grids in small batches. Every historical
feature groups by grid_id (optionally + hour / day_of_week / time_period /
is_weekend), so a grid's full-year timeline is self-contained -- chunking by
grid never crosses a group boundary, so no cross-chunk state needs to be
carried. Each batch (946 grids x 366 days x 24 hours, batch of ~100 grids)
is on the order of ~10^6 rows, comfortably within a 16 GB budget, and each
batch is streamed straight to a Parquet writer instead of being accumulated
in a single in-memory frame.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc

RAW_PATH = "/mnt/user-data/uploads/cleaned_crime_data.parquet"
OUT_PATH = "/home/claude/citysense/citysense_2020.parquet"
YEAR = 2020
GRID_STEP = 0.01          # 0.01-degree grid cells, inferred (see notes below)
GRID_BATCH_SIZE = 100      # ~10 batches over 946 grids

TIME_PERIODS = pd.CategoricalDtype(
    categories=["Night", "Morning", "Afternoon", "Evening"], ordered=False
)


def make_grid_id(lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
    """Reproduce the 'lat_lon' string id format used upstream (e.g. '40.49_-74.24')."""
    return np.array(
        [f"{a}_{b}" for a, b in zip(lat_grid, lon_grid)], dtype=object
    )


def floor_grid(values: np.ndarray, step: float = GRID_STEP) -> np.ndarray:
    return np.round(np.floor(values / step) * step, 2)


# ---------------------------------------------------------------------------
# 1) Determine the COMPLETE 946-grid universe from the full multi-year file.
#    We only need latitude/longitude for this -- two float columns, ~3M rows,
#    trivial memory footprint (a few tens of MB).
# ---------------------------------------------------------------------------
dataset = ds.dataset(RAW_PATH, format="parquet")

latlon_table = dataset.to_table(columns=["latitude", "longitude"])
lat_all = latlon_table.column("latitude").to_numpy(zero_copy_only=False)
lon_all = latlon_table.column("longitude").to_numpy(zero_copy_only=False)
valid = ~np.isnan(lat_all) & ~np.isnan(lon_all)
lat_grid_all = floor_grid(lat_all[valid])
lon_grid_all = floor_grid(lon_all[valid])
grid_ids_all = make_grid_id(lat_grid_all, lon_grid_all)

grid_lookup = (
    pd.DataFrame({"grid_id": grid_ids_all, "lat_grid": lat_grid_all, "lon_grid": lon_grid_all})
    .drop_duplicates(subset="grid_id")
    .sort_values("grid_id")
    .reset_index(drop=True)
)

ALL_GRIDS = grid_lookup["grid_id"].to_numpy()
N_GRIDS = len(ALL_GRIDS)
print(f"[1] Full-history unique grids found: {N_GRIDS}")

del latlon_table, lat_all, lon_all, valid, lat_grid_all, lon_grid_all, grid_ids_all
import gc
gc.collect()

# ---------------------------------------------------------------------------
# 2) Load ONLY year-2020 crime rows (predicate pushdown at the Parquet level)
#    and aggregate to (grid_id, date, hour) -> crime_count.
#    2020 is the first year in the data, so no pre-2020 history exists to
#    fold in and none is invented -- every historical feature legitimately
#    starts at 0 on 2020-01-01.
# ---------------------------------------------------------------------------
year_filter = ds.field("year") == YEAR
year_table = dataset.to_table(
    columns=["cmplnt_fr_dt", "hour", "latitude", "longitude"],
    filter=year_filter,
)
year_df = year_table.to_pandas()
del year_table
gc.collect()

n_before = len(year_df)
year_df = year_df.dropna(subset=["latitude", "longitude"])
n_dropped = n_before - len(year_df)
print(f"[2] Year-2020 crime rows: {n_before:,} ({n_dropped} dropped for missing lat/lon)")

year_df["lat_grid"] = floor_grid(year_df["latitude"].to_numpy())
year_df["lon_grid"] = floor_grid(year_df["longitude"].to_numpy())
year_df["grid_id"] = make_grid_id(year_df["lat_grid"].to_numpy(), year_df["lon_grid"].to_numpy())
year_df["cmplnt_fr_dt"] = pd.to_datetime(year_df["cmplnt_fr_dt"]).dt.normalize()

crime_agg = (
    year_df.groupby(["grid_id", "cmplnt_fr_dt", "hour"], as_index=False)
    .size()
    .rename(columns={"size": "crime_count"})
)
crime_agg["crime_count"] = crime_agg["crime_count"].astype("int32")
print(f"[2] Aggregated (grid, date, hour) rows with >=1 crime in 2020: {len(crime_agg):,}")

del year_df
gc.collect()

# Index crime_agg by grid for fast per-batch slicing.
crime_agg = crime_agg.set_index("grid_id", drop=False)
crime_agg.sort_index(inplace=True)

# ---------------------------------------------------------------------------
# 3) Full 2020 calendar (366 days -- 2020 is a leap year) and hours.
# ---------------------------------------------------------------------------
all_dates_2020 = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")
all_hours = np.arange(24, dtype="int8")
print(f"[3] 2020 calendar days: {len(all_dates_2020)}")

expected_rows = N_GRIDS * len(all_dates_2020) * 24
print(f"[3] Expected complete grid x date x hour universe: {expected_rows:,}")

# ---------------------------------------------------------------------------
# 4) Process grids in batches: build the dense universe, join real counts,
#    compute leak-free historical features, cast to compact dtypes, stream
#    to a single output Parquet file.
# ---------------------------------------------------------------------------
lat_map = dict(zip(grid_lookup["grid_id"], grid_lookup["lat_grid"]))
lon_map = dict(zip(grid_lookup["grid_id"], grid_lookup["lon_grid"]))

writer = None
total_rows = 0
positive_rows = 0
grids_seen = set()

n_batches = int(np.ceil(N_GRIDS / GRID_BATCH_SIZE))
for b in range(n_batches):
    batch_grids = ALL_GRIDS[b * GRID_BATCH_SIZE : (b + 1) * GRID_BATCH_SIZE]

    # Dense cross: batch_grids x all_dates_2020 x all_hours
    g_idx, d_idx, h_idx = np.meshgrid(
        np.arange(len(batch_grids)), np.arange(len(all_dates_2020)), all_hours, indexing="ij"
    )
    chunk = pd.DataFrame(
        {
            "grid_id": batch_grids[g_idx.ravel()],
            "cmplnt_fr_dt": all_dates_2020.values[d_idx.ravel()],
            "hour": h_idx.ravel().astype("int8"),
        }
    )

    # Join actual crime counts for just this batch's grids.
    try:
        batch_crimes = crime_agg.loc[crime_agg.index.intersection(batch_grids)]
    except KeyError:
        batch_crimes = crime_agg.iloc[0:0]
    batch_crimes = batch_crimes.reset_index(drop=True)

    chunk = chunk.merge(
        batch_crimes[["grid_id", "cmplnt_fr_dt", "hour", "crime_count"]],
        on=["grid_id", "cmplnt_fr_dt", "hour"],
        how="left",
    )
    chunk["crime_count"] = chunk["crime_count"].fillna(0).astype("int32")

    # Chronological order is required before any cumsum/shift.
    chunk = chunk.sort_values(["grid_id", "cmplnt_fr_dt", "hour"]).reset_index(drop=True)

    # --- calendar features ---
    chunk["day_of_week"] = chunk["cmplnt_fr_dt"].dt.dayofweek.astype("int8")
    chunk["is_weekend"] = (chunk["day_of_week"] >= 5).astype("int8")
    hour_vals = chunk["hour"].to_numpy()
    time_period = np.select(
        [hour_vals < 5, hour_vals < 12, hour_vals < 18],
        ["Night", "Morning", "Afternoon"],
        default="Evening",
    )
    chunk["time_period"] = pd.Categorical(time_period, dtype=TIME_PERIODS)
    chunk["year"] = YEAR
    chunk["month"] = chunk["cmplnt_fr_dt"].dt.month.astype("int8")

    # --- leak-free historical features: cumulative sum of crime_count up to
    # (but excluding) the current window, computed on the now-complete,
    # chronologically sorted universe. Each grid's timeline lives entirely
    # inside this batch, so grouping stays self-contained. ---
    def hist(group_cols):
        return (
            chunk.groupby(group_cols)["crime_count"]
            .transform(lambda x: x.cumsum().shift(1))
            .fillna(0)
            .astype("int32")
        )

    chunk["historical_grid_crime_count"] = hist(["grid_id"])
    chunk["historical_grid_hour_crime_count"] = hist(["grid_id", "hour"])
    chunk["historical_grid_day_crime_count"] = hist(["grid_id", "day_of_week"])
    chunk["historical_grid_time_period_crime_count"] = hist(["grid_id", "time_period"])
    chunk["historical_grid_weekend_crime_count"] = hist(["grid_id", "is_weekend"])

    chunk["lat_grid"] = chunk["grid_id"].map(lat_map).astype("float32")
    chunk["lon_grid"] = chunk["grid_id"].map(lon_map).astype("float32")
    chunk["grid_id"] = chunk["grid_id"].astype("category")

    chunk = chunk[
        [
            "grid_id",
            "cmplnt_fr_dt",
            "hour",
            "crime_count",
            "historical_grid_crime_count",
            "historical_grid_hour_crime_count",
            "day_of_week",
            "historical_grid_day_crime_count",
            "time_period",
            "historical_grid_time_period_crime_count",
            "is_weekend",
            "historical_grid_weekend_crime_count",
            "year",
            "month",
            "lat_grid",
            "lon_grid",
        ]
    ]

    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT_PATH, table.schema)
    writer.write_table(table)

    total_rows += len(chunk)
    positive_rows += int((chunk["crime_count"] > 0).sum())
    grids_seen.update(batch_grids.tolist())

    print(f"    batch {b + 1}/{n_batches}: {len(chunk):,} rows written (grids {b*GRID_BATCH_SIZE}-{b*GRID_BATCH_SIZE+len(batch_grids)-1})")

    del chunk, table, g_idx, d_idx, h_idx, batch_crimes
    gc.collect()

writer.close()

print("\n[4] DONE")
print(f"    Total rows written: {total_rows:,}")
print(f"    Expected rows:      {expected_rows:,}")
print(f"    Positive-crime rows:{positive_rows:,}")
print(f"    Zero-crime rows:    {total_rows - positive_rows:,}")
print(f"    Unique grids seen:  {len(grids_seen)}")
