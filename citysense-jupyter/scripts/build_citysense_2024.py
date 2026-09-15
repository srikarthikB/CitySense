"""
CitySense - 2024 data-preparation pipeline.

Builds the complete 946-grid x 366-date x 24-hour universe for 2024
(2024 is a leap year -> 366 days, unlike 2020-2023's builds which were
tied to 365-day years except 2020 -- see the leap-year note below),
joining actual crime counts (0 where none occurred) and computing
leak-free historical features that CARRY OVER the cumulative state
from the end of 2023 (which itself already carries the end-of-2022
state, which carries end-of-2021, which carries end-of-2020) rather
than resetting to zero on 2024-01-01.

Leap-year note
--------------
2024 is a leap year, so the dense universe is 946 x 366 x 24 =
8,309,664 rows (not 8,286,960 like 2021/2022/2023). 2020 was ALSO a
leap year (366 days), so this script's calendar-construction logic
mirrors what `build_citysense_2020.py` did, not what 2021-2023 did.
This was confirmed by checking citysense_2020.parquet's unique-date
count (366) before writing this script -- see the pre-flight check
in section 0 below.

Carry-over logic
-----------------
Each historical feature is defined as:
    historical_X = groupby(<grid[, subkey]>)['crime_count'].cumsum().shift(1)
i.e. the total crime_count accumulated in that (grid[, subkey]) group
STRICTLY BEFORE the current row, in chronological order.

For 2023 -> 2024, the carry-over constant per group must be the FULL
crime total accumulated over 2020 + 2021 + 2022 + 2023 for that group
(2023's own crime_count column only reflects 2023 activity -- it does
NOT include the 2020+2021+2022 contribution baked into 2023's
historical_* columns). Because raw crime_count sums are simple
additive quantities untouched by any cumsum/shift bookkeeping, this is
computed directly as:

    carry_2024[group] = sum(crime_count where year==2020, group)
                       + sum(crime_count where year==2021, group)
                       + sum(crime_count where year==2022, group)
                       + sum(crime_count where year==2023, group)

This is spot-checked below against the alternative derivation "last
chronological 2023 row's historical_X + that row's own crime_count"
for the grid-level key, to confirm the two approaches agree (this
mirrors exactly the check build_citysense_2023.py did against 2022,
which itself mirrored build_citysense_2022.py's check against 2021).

Five carry-over tables are built, one per historical feature, keyed by
the grouping that feature uses:
    historical_grid_crime_count              -> key: grid_id
    historical_grid_hour_crime_count         -> key: grid_id, hour
    historical_grid_day_crime_count          -> key: grid_id, day_of_week
    historical_grid_time_period_crime_count  -> key: grid_id, time_period
    historical_grid_weekend_crime_count      -> key: grid_id, is_weekend

For each 2024 batch (a subset of grids), the within-2024 cumsum().shift(1)
is computed exactly as in the 2020-2023 scripts (so no future 2024
information -- and definitely no 2025 information, since 2025 crime
rows are never even read -- leaks backward), and then the matching
carry-over constant is added to every row of that group. The very
first 2024 row of a group therefore starts from its 2023 ending total
(which already reflects 2020+2021+2022+2023), and no in-memory
full-year 2024 frame, or full 2020-2025 frame, is ever built.
"""

import gc
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

PROC = psutil.Process()
PEAK_RSS_BYTES = 0


def track_peak():
    global PEAK_RSS_BYTES
    rss = PROC.memory_info().rss
    if rss > PEAK_RSS_BYTES:
        PEAK_RSS_BYTES = rss
    return rss


RAW_PATH = "/mnt/user-data/uploads/cleaned_crime_data.parquet"
PATH_2020 = "/home/claude/work2/citysense_2020.parquet"
PATH_2021 = "/home/claude/work2/citysense_2021.parquet"
PATH_2022 = "/home/claude/work2/citysense_2022.parquet"
PATH_2023 = "/home/claude/work2/citysense_2023.parquet"
OUT_PATH = "/home/claude/work2/citysense_2024.parquet"
YEAR = 2024
GRID_BATCH_SIZE = 100

TIME_PERIODS = pd.CategoricalDtype(
    categories=["Night", "Morning", "Afternoon", "Evening"], ordered=False
)

t0 = time.time()

# ---------------------------------------------------------------------------
# 0) Pre-flight: confirm 2024's calendar length assumption (leap year) by
#    checking the previous leap year (2020) actually has 366 unique dates in
#    its own generated file, rather than assuming.
# ---------------------------------------------------------------------------
n_dates_2020 = pd.read_parquet(PATH_2020, columns=["cmplnt_fr_dt"])["cmplnt_fr_dt"].nunique()
print(f"[0] Pre-flight: citysense_2020.parquet unique dates = {n_dates_2020} (2020 was a leap year)")
assert n_dates_2020 == 366, f"Expected 2020 to have 366 dates, found {n_dates_2020}"

# ---------------------------------------------------------------------------
# 1) Grid universe: reuse the EXACT grid_id / lat_grid / lon_grid set already
#    established in citysense_2023.parquet (itself inherited unchanged from
#    citysense_2022.parquet / citysense_2021.parquet / citysense_2020.parquet)
#    -- do not rederive a new grid system.
# ---------------------------------------------------------------------------
grid_lookup = (
    pd.read_parquet(PATH_2023, columns=["grid_id", "lat_grid", "lon_grid"])
    .astype({"grid_id": "str"})
    .drop_duplicates(subset="grid_id")
    .sort_values("grid_id")
    .reset_index(drop=True)
)
ALL_GRIDS = grid_lookup["grid_id"].to_numpy()
N_GRIDS = len(ALL_GRIDS)
print(f"[1] Grids reused from citysense_2023.parquet: {N_GRIDS}")
assert N_GRIDS == 946, f"Expected 946 grids, found {N_GRIDS}"

lat_map = dict(zip(grid_lookup["grid_id"], grid_lookup["lat_grid"]))
lon_map = dict(zip(grid_lookup["grid_id"], grid_lookup["lon_grid"]))
track_peak()

# ---------------------------------------------------------------------------
# 2) Carry-over state: full 2020+2021+2022+2023 crime_count totals per
#    group, for each of the five historical-feature groupings. Only the
#    columns needed for these aggregations are read from each prior-year
#    file.
# ---------------------------------------------------------------------------
NEEDED_COLS = [
    "grid_id",
    "cmplnt_fr_dt",
    "hour",
    "crime_count",
    "day_of_week",
    "time_period",
    "is_weekend",
    "historical_grid_crime_count",
]

prior_2020 = pd.read_parquet(PATH_2020, columns=NEEDED_COLS)
prior_2020["grid_id"] = prior_2020["grid_id"].astype("str")

prior_2021 = pd.read_parquet(PATH_2021, columns=NEEDED_COLS)
prior_2021["grid_id"] = prior_2021["grid_id"].astype("str")

prior_2022 = pd.read_parquet(PATH_2022, columns=NEEDED_COLS)
prior_2022["grid_id"] = prior_2022["grid_id"].astype("str")

prior_2023 = pd.read_parquet(PATH_2023, columns=NEEDED_COLS)
prior_2023["grid_id"] = prior_2023["grid_id"].astype("str")

prior_all = pd.concat([prior_2020, prior_2021, prior_2022, prior_2023], ignore_index=True)

carry_grid = prior_all.groupby("grid_id")["crime_count"].sum().rename("carry").reset_index()
carry_grid_hour = (
    prior_all.groupby(["grid_id", "hour"])["crime_count"].sum().rename("carry").reset_index()
)
carry_grid_day = (
    prior_all.groupby(["grid_id", "day_of_week"])["crime_count"].sum().rename("carry").reset_index()
)
carry_grid_tp = (
    prior_all.groupby(["grid_id", "time_period"], observed=True)["crime_count"]
    .sum()
    .rename("carry")
    .reset_index()
)
carry_grid_weekend = (
    prior_all.groupby(["grid_id", "is_weekend"])["crime_count"].sum().rename("carry").reset_index()
)

# Spot-check: for one grid, confirm the two ways of computing the full
# 2020+2021+2022+2023 total agree -- (a) sum(crime_count) over all four
# years, vs (b) the last chronological 2023 row's historical_grid_crime_count
# + that row's own crime_count (which already reflects the 2020+2021+2022
# carry baked into the 2023 file).
check_grid = ALL_GRIDS[0]
prior_2023_sorted = prior_2023[prior_2023["grid_id"] == check_grid].sort_values(
    ["cmplnt_fr_dt", "hour"]
)
last_row = prior_2023_sorted.iloc[-1]
method_b = int(last_row["historical_grid_crime_count"]) + int(last_row["crime_count"])
method_a = int(carry_grid.loc[carry_grid["grid_id"] == check_grid, "carry"].iloc[0])
print(
    f"[2] Spot-check grid {check_grid}: 2020+2021+2022+2023-sum={method_a}, "
    f"last-2023-row-historical+crime_count={method_b}, match={method_a == method_b}"
)
assert method_a == method_b, "Carry-over derivation mismatch!"

del prior_2020, prior_2021, prior_2022, prior_2023, prior_all, prior_2023_sorted
gc.collect()
track_peak()
print(
    f"[2] Carry-over tables built: grid={len(carry_grid)}, "
    f"grid_hour={len(carry_grid_hour)}, grid_day={len(carry_grid_day)}, "
    f"grid_time_period={len(carry_grid_tp)}, grid_weekend={len(carry_grid_weekend)}"
)

# ---------------------------------------------------------------------------
# 3) Load ONLY year-2024 crime rows (predicate pushdown) and aggregate to
#    (grid_id, date, hour) -> crime_count, using the same 0.01-degree grid
#    definition already used for 2020-2023. 2025 rows are never read.
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
print(f"[3] Year-2024 crime rows: {n_before:,} ({n_dropped} dropped for missing lat/lon)")

year_df["lat_grid"] = floor_grid(year_df["latitude"].to_numpy())
year_df["lon_grid"] = floor_grid(year_df["longitude"].to_numpy())
year_df["grid_id"] = make_grid_id(year_df["lat_grid"].to_numpy(), year_df["lon_grid"].to_numpy())
year_df["cmplnt_fr_dt"] = pd.to_datetime(year_df["cmplnt_fr_dt"]).dt.normalize()

# Restrict to the established 946-grid universe (guards against any stray
# grid outside the reused 2020-2023 grid set).
year_df = year_df[year_df["grid_id"].isin(set(ALL_GRIDS))]

crime_agg = (
    year_df.groupby(["grid_id", "cmplnt_fr_dt", "hour"], as_index=False)
    .size()
    .rename(columns={"size": "crime_count"})
)
crime_agg["crime_count"] = crime_agg["crime_count"].astype("int32")
print(f"[3] Aggregated (grid, date, hour) rows with >=1 crime in 2024: {len(crime_agg):,}")

del year_df
gc.collect()
track_peak()

crime_agg = crime_agg.set_index("grid_id", drop=False)
crime_agg.sort_index(inplace=True)

# ---------------------------------------------------------------------------
# 4) Full 2024 calendar (366 days -- 2024 IS a leap year) and hours.
# ---------------------------------------------------------------------------
all_dates_2024 = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")
all_hours = np.arange(24, dtype="int8")
print(f"[4] 2024 calendar days: {len(all_dates_2024)}")
assert len(all_dates_2024) == 366, f"Expected 366 days for leap year 2024, got {len(all_dates_2024)}"
assert all_dates_2024.max() == pd.Timestamp("2024-12-31")
assert (all_dates_2024 == pd.Timestamp("2024-02-29")).any(), "Feb 29 2024 missing from calendar"

expected_rows = N_GRIDS * len(all_dates_2024) * 24
print(f"[4] Expected complete grid x date x hour universe: {expected_rows:,}")
assert expected_rows == 8_309_664, f"Expected 8,309,664 rows, got {expected_rows:,}"

# ---------------------------------------------------------------------------
# 5) Process grids in batches: build the dense universe, join real counts,
#    compute leak-free WITHIN-2024 historical features, add the matching
#    2020+2021+2022+2023 carry-over constant, cast to compact dtypes, stream
#    to Parquet.
# ---------------------------------------------------------------------------
writer = None
total_rows = 0
positive_rows = 0
grids_seen = set()

n_batches = int(np.ceil(N_GRIDS / GRID_BATCH_SIZE))
for b in range(n_batches):
    batch_grids = ALL_GRIDS[b * GRID_BATCH_SIZE : (b + 1) * GRID_BATCH_SIZE]

    g_idx, d_idx, h_idx = np.meshgrid(
        np.arange(len(batch_grids)), np.arange(len(all_dates_2024)), all_hours, indexing="ij"
    )
    chunk = pd.DataFrame(
        {
            "grid_id": batch_grids[g_idx.ravel()],
            "cmplnt_fr_dt": all_dates_2024.values[d_idx.ravel()],
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

    # --- calendar features (same definitions as 2020-2023) ---
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

    # --- leak-free historical features: WITHIN-2024 cumsum().shift(1),
    # then + the matching 2020+2021+2022+2023 carry-over constant for that
    # group. ---
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

    track_peak()
    if (b + 1) % 5 == 0 or b == n_batches - 1:
        print(
            f"    batch {b + 1}/{n_batches}: {len(chunk):,} rows written "
            f"(grids {b*GRID_BATCH_SIZE}-{b*GRID_BATCH_SIZE+len(batch_grids)-1}), "
            f"rss={track_peak() / 1e9:.2f} GB"
        )

    del chunk, table, g_idx, d_idx, h_idx, batch_crimes
    gc.collect()

writer.close()

elapsed = time.time() - t0
print("\n[5] DONE")
print(f"    Total rows written: {total_rows:,}")
print(f"    Expected rows:      {expected_rows:,}")
print(f"    Positive-crime rows:{positive_rows:,}")
print(f"    Zero-crime rows:    {total_rows - positive_rows:,}")
print(f"    Unique grids seen:  {len(grids_seen)}")
print(f"    Peak RSS:           {PEAK_RSS_BYTES / 1e9:.3f} GB")
print(f"    Elapsed:            {elapsed:.1f}s")

with open("/home/claude/work2/peak_rss.txt", "w") as f:
    f.write(str(PEAK_RSS_BYTES))
