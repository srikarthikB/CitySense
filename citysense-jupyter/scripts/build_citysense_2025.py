"""
CitySense - 2025 data-preparation pipeline (FINAL TEST YEAR).

Builds the complete 946-grid x 365-date x 24-hour universe for 2025
(2025 is NOT a leap year -> 365 days, same as 2021/2022/2023),
joining actual crime counts (0 where none occurred) and computing
leak-free historical features that CARRY OVER the cumulative state
from the end of 2024 (which itself already carries 2020+2021+2022+2023)
rather than resetting to zero on 2025-01-01.

This mirrors build_citysense_2024.py exactly, with two differences:
  1. The prior-year carry-over window is 2020+2021+2022+2023+2024
     (five years instead of four).
  2. The dense calendar is 365 days (2025 is not a leap year) instead
     of 2024's 366.

2025 is the FINAL TEST YEAR: crime_count here is the eventual target,
and this script must not be touched by any model-development step
after generation. No model is trained or tuned in this script.

Carry-over logic
-----------------
Each historical feature is defined as:
    historical_X = groupby(<grid[, subkey]>)['crime_count'].cumsum().shift(1)
i.e. the total crime_count accumulated in that (grid[, subkey]) group
STRICTLY BEFORE the current row, in chronological order.

For 2024 -> 2025, the carry-over constant per group must be the FULL
crime total accumulated over 2020 + 2021 + 2022 + 2023 + 2024 for that
group (2024's own crime_count column only reflects 2024 activity -- it
does NOT include the 2020+2021+2022+2023 contribution already baked
into 2024's historical_* columns). Because raw crime_count sums are
simple additive quantities untouched by any cumsum/shift bookkeeping,
this is computed directly as:

    carry_2025[group] = sum(crime_count where year==2020, group)
                       + sum(crime_count where year==2021, group)
                       + sum(crime_count where year==2022, group)
                       + sum(crime_count where year==2023, group)
                       + sum(crime_count where year==2024, group)

This is spot-checked below against the alternative derivation "last
chronological 2024 row's historical_X + that row's own crime_count"
for the grid-level key, to confirm the two approaches agree (this
mirrors exactly the check build_citysense_2024.py did against 2023).

Five carry-over tables are built, one per historical feature, keyed by
the grouping that feature uses:
    historical_grid_crime_count              -> key: grid_id
    historical_grid_hour_crime_count         -> key: grid_id, hour
    historical_grid_day_crime_count          -> key: grid_id, day_of_week
    historical_grid_time_period_crime_count  -> key: grid_id, time_period
    historical_grid_weekend_crime_count      -> key: grid_id, is_weekend

For each 2025 batch (a subset of grids), the within-2025 cumsum().shift(1)
is computed exactly as in the 2020-2024 scripts (so no future 2025
information leaks backward -- only earlier-in-2025 rows of the same
group can contribute), and then the matching carry-over constant is
added to every row of that group. The very first 2025 row of a group
therefore starts from its 2024 ending total (which already reflects
2020+2021+2022+2023+2024), and no in-memory full-year 2025 frame, or
multi-year frame, is ever built. Only YEAR==2025 rows are ever read
from the raw crime data -- no data after 2025-12-31 is read at all,
because the raw file's own latest year is 2025.
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
PATH_2020 = "/home/claude/work/citysense_2020.parquet"
PATH_2021 = "/home/claude/work/citysense_2021.parquet"
PATH_2022 = "/home/claude/work/citysense_2022.parquet"
PATH_2023 = "/home/claude/work/citysense_2023.parquet"
PATH_2024 = "/home/claude/work/citysense_2024.parquet"
OUT_PATH = "/home/claude/work/citysense_2025.parquet"
YEAR = 2025
GRID_BATCH_SIZE = 50

TIME_PERIODS = pd.CategoricalDtype(
    categories=["Night", "Morning", "Afternoon", "Evening"], ordered=False
)

t0 = time.time()

# ---------------------------------------------------------------------------
# 0) Pre-flight: confirm 2025's calendar length assumption (NOT a leap year)
#    by checking a previous non-leap year (2023) actually has 365 unique
#    dates in its own generated file, rather than assuming.
# ---------------------------------------------------------------------------
n_dates_2023 = pd.read_parquet(PATH_2023, columns=["cmplnt_fr_dt"])["cmplnt_fr_dt"].nunique()
print(f"[0] Pre-flight: citysense_2023.parquet unique dates = {n_dates_2023} (2023 was not a leap year)")
assert n_dates_2023 == 365, f"Expected 2023 to have 365 dates, found {n_dates_2023}"

n_dates_2024 = pd.read_parquet(PATH_2024, columns=["cmplnt_fr_dt"])["cmplnt_fr_dt"].nunique()
print(f"[0] Pre-flight: citysense_2024.parquet unique dates = {n_dates_2024} (2024 was a leap year)")
assert n_dates_2024 == 366, f"Expected 2024 to have 366 dates, found {n_dates_2024}"

# ---------------------------------------------------------------------------
# 1) Grid universe: reuse the EXACT grid_id / lat_grid / lon_grid set already
#    established in citysense_2024.parquet (itself inherited unchanged from
#    2023/2022/2021/2020) -- do not rederive a new grid system.
# ---------------------------------------------------------------------------
grid_lookup = (
    pd.read_parquet(PATH_2024, columns=["grid_id", "lat_grid", "lon_grid"])
    .astype({"grid_id": "str"})
    .drop_duplicates(subset="grid_id")
    .sort_values("grid_id")
    .reset_index(drop=True)
)
ALL_GRIDS = grid_lookup["grid_id"].to_numpy()
N_GRIDS = len(ALL_GRIDS)
print(f"[1] Grids reused from citysense_2024.parquet: {N_GRIDS}")
assert N_GRIDS == 946, f"Expected 946 grids, found {N_GRIDS}"

# Cross-check against 2020's grid set to confirm no drift anywhere in the chain.
grids_2020 = set(
    pd.read_parquet(PATH_2020, columns=["grid_id"])["grid_id"].astype(str).unique()
)
assert set(ALL_GRIDS) == grids_2020, "Grid universe drift detected between 2020 and 2024!"
print("[1] Confirmed grid universe identical to citysense_2020.parquet")

lat_map = dict(zip(grid_lookup["grid_id"], grid_lookup["lat_grid"]))
lon_map = dict(zip(grid_lookup["grid_id"], grid_lookup["lon_grid"]))
track_peak()

# ---------------------------------------------------------------------------
# 2) Carry-over state: full 2020+2021+2022+2023+2024 crime_count totals per
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

# Memory-efficient: process ONE prior year at a time (never hold all five
# years in RAM simultaneously -- the 16GB/small-RAM constraint applies here
# too), accumulating partial group-sums into running totals via repeated
# concat+groupby of the (small) partial-aggregate tables themselves, not the
# raw per-row data.
PRIOR_PATHS = [PATH_2020, PATH_2021, PATH_2022, PATH_2023, PATH_2024]
PRIOR_YEARS = [2020, 2021, 2022, 2023, 2024]

carry_grid = None
carry_grid_hour = None
carry_grid_day = None
carry_grid_tp = None
carry_grid_weekend = None

# Kept only for the spot-check below (last chronological 2024 row).
last_2024_row = None

for path, yr in zip(PRIOR_PATHS, PRIOR_YEARS):
    prior_yr = pd.read_parquet(path, columns=NEEDED_COLS)
    prior_yr["grid_id"] = prior_yr["grid_id"].astype("str")

    if yr == 2024:
        idx = prior_yr[prior_yr["grid_id"] == ALL_GRIDS[0]].sort_values(["cmplnt_fr_dt", "hour"]).index
        last_2024_row = prior_yr.loc[idx[-1]]

    def _partial_sum(df, group_cols, **kwargs):
        return df.groupby(group_cols, **kwargs)["crime_count"].sum().rename("carry").reset_index()

    def _accumulate(running, partial, group_cols):
        if running is None:
            return partial
        merged = pd.concat([running, partial], ignore_index=True)
        return merged.groupby(group_cols, observed=True)["carry"].sum().reset_index()

    carry_grid = _accumulate(carry_grid, _partial_sum(prior_yr, ["grid_id"]), ["grid_id"])
    carry_grid_hour = _accumulate(
        carry_grid_hour, _partial_sum(prior_yr, ["grid_id", "hour"]), ["grid_id", "hour"]
    )
    carry_grid_day = _accumulate(
        carry_grid_day, _partial_sum(prior_yr, ["grid_id", "day_of_week"]), ["grid_id", "day_of_week"]
    )
    carry_grid_tp = _accumulate(
        carry_grid_tp,
        _partial_sum(prior_yr, ["grid_id", "time_period"], observed=True),
        ["grid_id", "time_period"],
    )
    carry_grid_weekend = _accumulate(
        carry_grid_weekend,
        _partial_sum(prior_yr, ["grid_id", "is_weekend"]),
        ["grid_id", "is_weekend"],
    )

    print(f"    [2] accumulated carry through year {yr} ({len(prior_yr):,} rows read)")
    del prior_yr
    gc.collect()
    track_peak()

# Spot-check: for one grid, confirm the two ways of computing the full
# 2020+2021+2022+2023+2024 total agree -- (a) sum(crime_count) over all
# five years, vs (b) the last chronological 2024 row's
# historical_grid_crime_count + that row's own crime_count (which already
# reflects the 2020+2021+2022+2023 carry baked into the 2024 file).
check_grid = ALL_GRIDS[0]
method_b = int(last_2024_row["historical_grid_crime_count"]) + int(last_2024_row["crime_count"])
method_a = int(carry_grid.loc[carry_grid["grid_id"] == check_grid, "carry"].iloc[0])
print(
    f"[2] Spot-check grid {check_grid}: 2020+2021+2022+2023+2024-sum={method_a}, "
    f"last-2024-row-historical+crime_count={method_b}, match={method_a == method_b}"
)
assert method_a == method_b, "Carry-over derivation mismatch!"

del last_2024_row
gc.collect()
track_peak()
print(
    f"[2] Carry-over tables built: grid={len(carry_grid)}, "
    f"grid_hour={len(carry_grid_hour)}, grid_day={len(carry_grid_day)}, "
    f"grid_time_period={len(carry_grid_tp)}, grid_weekend={len(carry_grid_weekend)}"
)

# ---------------------------------------------------------------------------
# 3) Load ONLY year-2025 crime rows (predicate pushdown) and aggregate to
#    (grid_id, date, hour) -> crime_count, using the same 0.01-degree grid
#    definition already used for 2020-2024. No data after 2025-12-31 exists
#    in the raw file, and no such data is read.
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
print(f"[3] Year-2025 crime rows: {n_before:,} ({n_dropped} dropped for missing lat/lon)")

year_df["lat_grid"] = floor_grid(year_df["latitude"].to_numpy())
year_df["lon_grid"] = floor_grid(year_df["longitude"].to_numpy())
year_df["grid_id"] = make_grid_id(year_df["lat_grid"].to_numpy(), year_df["lon_grid"].to_numpy())
year_df["cmplnt_fr_dt"] = pd.to_datetime(year_df["cmplnt_fr_dt"]).dt.normalize()

# Guard: no 2025 crime rows should fall outside 2025-01-01..2025-12-31.
assert year_df["cmplnt_fr_dt"].min() >= pd.Timestamp("2025-01-01")
assert year_df["cmplnt_fr_dt"].max() <= pd.Timestamp("2025-12-31")

# Restrict to the established 946-grid universe (guards against any stray
# grid outside the reused 2020-2024 grid set).
year_df = year_df[year_df["grid_id"].isin(set(ALL_GRIDS))]

crime_agg = (
    year_df.groupby(["grid_id", "cmplnt_fr_dt", "hour"], as_index=False)
    .size()
    .rename(columns={"size": "crime_count"})
)
crime_agg["crime_count"] = crime_agg["crime_count"].astype("int32")
print(f"[3] Aggregated (grid, date, hour) rows with >=1 crime in 2025: {len(crime_agg):,}")

del year_df
gc.collect()
track_peak()

crime_agg = crime_agg.set_index("grid_id", drop=False)
crime_agg.sort_index(inplace=True)

# ---------------------------------------------------------------------------
# 4) Full 2025 calendar (365 days -- 2025 is NOT a leap year) and hours.
# ---------------------------------------------------------------------------
all_dates_2025 = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")
all_hours = np.arange(24, dtype="int8")
print(f"[4] 2025 calendar days: {len(all_dates_2025)}")
assert len(all_dates_2025) == 365, f"Expected 365 days for non-leap year 2025, got {len(all_dates_2025)}"
assert all_dates_2025.max() == pd.Timestamp("2025-12-31")
assert not any((d.month == 2 and d.day == 29) for d in all_dates_2025), "2025 is not a leap year; Feb 29 must not appear"

expected_rows = N_GRIDS * len(all_dates_2025) * 24
print(f"[4] Expected complete grid x date x hour universe: {expected_rows:,}")
assert expected_rows == 8_286_960, f"Expected 8,286,960 rows, got {expected_rows:,}"

# ---------------------------------------------------------------------------
# 5) Process grids in batches: build the dense universe, join real counts,
#    compute leak-free WITHIN-2025 historical features, add the matching
#    2020+2021+2022+2023+2024 carry-over constant, cast to compact dtypes,
#    stream to Parquet.
# ---------------------------------------------------------------------------
writer = None
total_rows = 0
positive_rows = 0
grids_seen = set()

n_batches = int(np.ceil(N_GRIDS / GRID_BATCH_SIZE))
for b in range(n_batches):
    batch_grids = ALL_GRIDS[b * GRID_BATCH_SIZE : (b + 1) * GRID_BATCH_SIZE]

    g_idx, d_idx, h_idx = np.meshgrid(
        np.arange(len(batch_grids)), np.arange(len(all_dates_2025)), all_hours, indexing="ij"
    )
    chunk = pd.DataFrame(
        {
            "grid_id": batch_grids[g_idx.ravel()],
            "cmplnt_fr_dt": all_dates_2025.values[d_idx.ravel()],
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

    # --- calendar features (same definitions as 2020-2024) ---
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

    # --- leak-free historical features: WITHIN-2025 cumsum().shift(1),
    # then + the matching 2020+2021+2022+2023+2024 carry-over constant for
    # that group. The shift(1) guarantees the current row's own crime_count,
    # and every later row (later hour same day, or later date) never
    # contributes to an earlier row's historical feature. ---
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

with open("/home/claude/work/peak_rss.txt", "w") as f:
    f.write(str(PEAK_RSS_BYTES))
