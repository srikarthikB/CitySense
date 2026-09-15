"""
CitySense - 2021 data-preparation pipeline.

Builds the complete 946-grid x 365-date x 24-hour universe for 2021,
joining actual crime counts (0 where none occurred) and computing
leak-free historical features that CARRY OVER the cumulative state
from the end of 2020 rather than resetting to zero on 2021-01-01.

Carry-over logic
-----------------
In the 2020 pipeline, each historical feature is:
    historical_X = groupby(<grid[, subkey]>)['crime_count'].cumsum().shift(1)
i.e. the total crime_count accumulated in that (grid[, subkey]) group
STRICTLY BEFORE the current row, in chronological order.

For the last chronological row of 2020 in any group, that row's
historical_X value PLUS its own crime_count equals the FULL 2020 total
for that group. Since crime_count for every (grid, date, hour) row is
already materialized in citysense_2020.parquet (including zero-crime
rows), the full-year total for a group is simply
    sum(crime_count) over all citysense_2020.parquet rows in that group.
This is computed directly (independent of reading back the historical_*
columns) and is what is added as a constant offset to the fresh
2021-only cumsum().shift(1) for that group. It is spot-checked below
against `historical_grid_crime_count(last 2020 row) + crime_count(last
2020 row)` to confirm the two approaches agree.

Five carry-over tables are built, one per historical feature, keyed by
the grouping that feature uses:
    historical_grid_crime_count              -> key: grid_id
    historical_grid_hour_crime_count         -> key: grid_id, hour
    historical_grid_day_crime_count          -> key: grid_id, day_of_week
    historical_grid_time_period_crime_count  -> key: grid_id, time_period
    historical_grid_weekend_crime_count      -> key: grid_id, is_weekend

For each 2021 batch (a subset of grids), the within-2021 cumsum().shift(1)
is computed exactly as in the 2020 script (so no future 2021 information
leaks backward), and then the matching carry-over constant is added to
every row of that group. The very first 2021 row of a group therefore
starts from its 2020 ending total, and no in-memory full-year 2021
frame or full 2020-2025 frame is ever built.
"""

import gc

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

RAW_PATH = "/mnt/user-data/uploads/cleaned_crime_data.parquet"
PRIOR_YEAR_PATH = "/home/claude/work/citysense_2020.parquet"
OUT_PATH = "/home/claude/citysense/citysense_2021.parquet"
YEAR = 2021
GRID_BATCH_SIZE = 100

TIME_PERIODS = pd.CategoricalDtype(
    categories=["Night", "Morning", "Afternoon", "Evening"], ordered=False
)

# ---------------------------------------------------------------------------
# 1) Grid universe: reuse the EXACT grid_id / lat_grid / lon_grid set already
#    established in citysense_2020.parquet -- do not rederive a new grid
#    system.
# ---------------------------------------------------------------------------
grid_lookup = (
    pd.read_parquet(PRIOR_YEAR_PATH, columns=["grid_id", "lat_grid", "lon_grid"])
    .astype({"grid_id": "str"})
    .drop_duplicates(subset="grid_id")
    .sort_values("grid_id")
    .reset_index(drop=True)
)
ALL_GRIDS = grid_lookup["grid_id"].to_numpy()
N_GRIDS = len(ALL_GRIDS)
print(f"[1] Grids reused from citysense_2020.parquet: {N_GRIDS}")
assert N_GRIDS == 946, f"Expected 946 grids, found {N_GRIDS}"

lat_map = dict(zip(grid_lookup["grid_id"], grid_lookup["lat_grid"]))
lon_map = dict(zip(grid_lookup["grid_id"], grid_lookup["lon_grid"]))

# ---------------------------------------------------------------------------
# 2) Carry-over state: full-2020 crime_count totals per group, for each of
#    the five historical-feature groupings. Only the columns needed for
#    these aggregations are read (grid_id, hour, day_of_week, time_period,
#    is_weekend, crime_count) -- small footprint, immediately reduced to a
#    handful of tiny lookup tables.
# ---------------------------------------------------------------------------
prior = pd.read_parquet(
    PRIOR_YEAR_PATH,
    columns=[
        "grid_id",
        "cmplnt_fr_dt",
        "hour",
        "crime_count",
        "day_of_week",
        "time_period",
        "is_weekend",
        "historical_grid_crime_count",
    ],
)
prior["grid_id"] = prior["grid_id"].astype("str")

carry_grid = prior.groupby("grid_id")["crime_count"].sum().rename("carry").reset_index()
carry_grid_hour = (
    prior.groupby(["grid_id", "hour"])["crime_count"].sum().rename("carry").reset_index()
)
carry_grid_day = (
    prior.groupby(["grid_id", "day_of_week"])["crime_count"].sum().rename("carry").reset_index()
)
carry_grid_tp = (
    prior.groupby(["grid_id", "time_period"], observed=True)["crime_count"]
    .sum()
    .rename("carry")
    .reset_index()
)
carry_grid_weekend = (
    prior.groupby(["grid_id", "is_weekend"])["crime_count"].sum().rename("carry").reset_index()
)

# Spot-check: for one grid, confirm the two ways of computing the full-2020
# grid-level total agree -- (a) sum(crime_count) over the whole year (used
# above as the carry-over), vs (b) the last chronological 2020 row's
# historical_grid_crime_count + that row's own crime_count.
check_grid = ALL_GRIDS[0]
prior_sorted = prior[prior["grid_id"] == check_grid].sort_values(["cmplnt_fr_dt", "hour"])
last_row = prior_sorted.iloc[-1]
method_b = int(last_row["historical_grid_crime_count"]) + int(last_row["crime_count"])
method_a = int(carry_grid.loc[carry_grid["grid_id"] == check_grid, "carry"].iloc[0])
print(f"[2] Spot-check grid {check_grid}: full-year-sum={method_a}, "
      f"last-row-historical+crime_count={method_b}, match={method_a == method_b}")
assert method_a == method_b, "Carry-over derivation mismatch!"

del prior, prior_sorted
gc.collect()
print(f"[2] Carry-over tables built: grid={len(carry_grid)}, "
      f"grid_hour={len(carry_grid_hour)}, grid_day={len(carry_grid_day)}, "
      f"grid_time_period={len(carry_grid_tp)}, grid_weekend={len(carry_grid_weekend)}")

# ---------------------------------------------------------------------------
# 3) Load ONLY year-2021 crime rows (predicate pushdown) and aggregate to
#    (grid_id, date, hour) -> crime_count, using the same 0.01-degree grid
#    definition already used for 2020.
# ---------------------------------------------------------------------------
GRID_STEP = 0.01


def make_grid_id(lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
    return np.array([f"{a}_{b}" for a, b in zip(lat_grid, lon_grid)], dtype=object)


def floor_grid(values: np.ndarray, step: float = GRID_STEP) -> np.ndarray:
    return np.round(np.floor(values / step) * step, 2)


dataset = ds.dataset(RAW_PATH, format="parquet")
year_filter = ds.field("year") == YEAR
year_table = dataset.to_table(
    columns=["cmplnt_fr_dt", "hour", "latitude", "longitude"], filter=year_filter
)
year_df = year_table.to_pandas()
del year_table
gc.collect()

n_before = len(year_df)
year_df = year_df.dropna(subset=["latitude", "longitude"])
n_dropped = n_before - len(year_df)
print(f"[3] Year-2021 crime rows: {n_before:,} ({n_dropped} dropped for missing lat/lon)")

year_df["lat_grid"] = floor_grid(year_df["latitude"].to_numpy())
year_df["lon_grid"] = floor_grid(year_df["longitude"].to_numpy())
year_df["grid_id"] = make_grid_id(year_df["lat_grid"].to_numpy(), year_df["lon_grid"].to_numpy())
year_df["cmplnt_fr_dt"] = pd.to_datetime(year_df["cmplnt_fr_dt"]).dt.normalize()

# Restrict to the established 946-grid universe (guards against any stray
# grid outside the reused 2020 grid set).
year_df = year_df[year_df["grid_id"].isin(set(ALL_GRIDS))]

crime_agg = (
    year_df.groupby(["grid_id", "cmplnt_fr_dt", "hour"], as_index=False)
    .size()
    .rename(columns={"size": "crime_count"})
)
crime_agg["crime_count"] = crime_agg["crime_count"].astype("int32")
print(f"[3] Aggregated (grid, date, hour) rows with >=1 crime in 2021: {len(crime_agg):,}")

del year_df
gc.collect()

crime_agg = crime_agg.set_index("grid_id", drop=False)
crime_agg.sort_index(inplace=True)

# ---------------------------------------------------------------------------
# 4) Full 2021 calendar (365 days -- not a leap year) and hours.
# ---------------------------------------------------------------------------
all_dates_2021 = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")
all_hours = np.arange(24, dtype="int8")
print(f"[4] 2021 calendar days: {len(all_dates_2021)}")

expected_rows = N_GRIDS * len(all_dates_2021) * 24
print(f"[4] Expected complete grid x date x hour universe: {expected_rows:,}")

# ---------------------------------------------------------------------------
# 5) Process grids in batches: build the dense universe, join real counts,
#    compute leak-free WITHIN-2021 historical features, add the matching
#    2020 carry-over constant, cast to compact dtypes, stream to Parquet.
# ---------------------------------------------------------------------------
writer = None
total_rows = 0
positive_rows = 0
grids_seen = set()

n_batches = int(np.ceil(N_GRIDS / GRID_BATCH_SIZE))
for b in range(n_batches):
    batch_grids = ALL_GRIDS[b * GRID_BATCH_SIZE : (b + 1) * GRID_BATCH_SIZE]

    g_idx, d_idx, h_idx = np.meshgrid(
        np.arange(len(batch_grids)), np.arange(len(all_dates_2021)), all_hours, indexing="ij"
    )
    chunk = pd.DataFrame(
        {
            "grid_id": batch_grids[g_idx.ravel()],
            "cmplnt_fr_dt": all_dates_2021.values[d_idx.ravel()],
            "hour": h_idx.ravel().astype("int8"),
        }
    )

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

    # Chronological order required before any cumsum/shift.
    chunk = chunk.sort_values(["grid_id", "cmplnt_fr_dt", "hour"]).reset_index(drop=True)

    # --- calendar features (same definitions as 2020) ---
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

    # --- leak-free historical features: WITHIN-2021 cumsum().shift(1),
    # then + the matching 2020 carry-over constant for that group. ---
    def hist_with_carry(group_cols, carry_df):
        within_year = (
            chunk.groupby(group_cols)["crime_count"]
            .transform(lambda x: x.cumsum().shift(1))
            .fillna(0)
        )
        merged = chunk[group_cols].merge(carry_df, on=group_cols, how="left")
        carry_vals = merged["carry"].fillna(0).to_numpy()
        return (within_year.to_numpy() + carry_vals).astype("int32")

    chunk["historical_grid_crime_count"] = hist_with_carry(["grid_id"], carry_grid)
    chunk["historical_grid_hour_crime_count"] = hist_with_carry(["grid_id", "hour"], carry_grid_hour)
    chunk["historical_grid_day_crime_count"] = hist_with_carry(["grid_id", "day_of_week"], carry_grid_day)
    chunk["historical_grid_time_period_crime_count"] = hist_with_carry(
        ["grid_id", "time_period"], carry_grid_tp
    )
    chunk["historical_grid_weekend_crime_count"] = hist_with_carry(
        ["grid_id", "is_weekend"], carry_grid_weekend
    )

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

    print(f"    batch {b + 1}/{n_batches}: {len(chunk):,} rows written "
          f"(grids {b*GRID_BATCH_SIZE}-{b*GRID_BATCH_SIZE+len(batch_grids)-1})")

    del chunk, table, g_idx, d_idx, h_idx, batch_crimes
    gc.collect()

writer.close()

print("\n[5] DONE")
print(f"    Total rows written: {total_rows:,}")
print(f"    Expected rows:      {expected_rows:,}")
print(f"    Positive-crime rows:{positive_rows:,}")
print(f"    Zero-crime rows:    {total_rows - positive_rows:,}")
print(f"    Unique grids seen:  {len(grids_seen)}")
