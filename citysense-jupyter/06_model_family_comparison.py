"""Fair CitySense model-family comparison (2020-2023 train, 2024 validation).

This experiment deliberately does not read the 2025 file, alter any source
data, or change CitySense features.  It reproduces the sampling procedure in
05_model_building.ipynb and saves the selected original row positions so every
candidate is trained on exactly the same one-million-row sample on reruns.

Run from the repository root with the project's Jupyter/Python environment:
    python 06_model_family_comparison.py
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


RANDOM_STATE = 42
TOTAL_SAMPLE = 1_000_000
PREDICTION_BATCH_SIZE = 500_000
TRAIN_YEARS = (2020, 2021, 2022, 2023)
VALIDATION_YEAR = 2024
DATA_DIR = Path("data/yearly")
ARTIFACT_DIR = Path("artifacts/model_family_comparison")
SAMPLE_INDEX_PATH = ARTIFACT_DIR / "sampled_indices_2020_2023.npz"
COMPARISON_PATH = ARTIFACT_DIR / "comparison_2024.csv"
CALIBRATION_PATH = ARTIFACT_DIR / "calibration_2024.csv"
ACTUAL_COUNT_CALIBRATION_PATH = ARTIFACT_DIR / "actual_count_calibration_2024.csv"
MANIFEST_PATH = ARTIFACT_DIR / "run_manifest.json"

TARGET = "crime_count"
DATETIME_COLUMN = "cmplnt_fr_dt"
CATEGORICAL_FEATURES = ["grid_id", "time_period"]
CALIBRATION_EDGES = np.array([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0, np.inf])


def yearly_path(year: int) -> Path:
    """Return a declared yearly input path; only 2020-2024 are allowed."""
    if year not in (*TRAIN_YEARS, VALIDATION_YEAR):
        raise ValueError(f"Year {year} is outside this experiment's allowed range")
    return DATA_DIR / f"citysense_{year}.parquet"


def model_columns() -> list[str]:
    """Derive the existing input columns without adding or removing features."""
    import pyarrow.parquet as pq

    columns = pq.ParquetFile(yearly_path(2020)).schema_arrow.names
    return [column for column in columns if column not in (TARGET, DATETIME_COLUMN)]


def current_notebook_sample_sizes() -> dict[int, int]:
    """Reproduce cell 5 of 05_model_building.ipynb exactly."""
    year_rows = {
        year: len(pd.read_parquet(yearly_path(year), columns=[TARGET]))
        for year in TRAIN_YEARS
    }
    total_rows = sum(year_rows.values())
    sizes = {year: round(TOTAL_SAMPLE * rows / total_rows) for year, rows in year_rows.items()}
    sizes[2023] += TOTAL_SAMPLE - sum(sizes.values())
    return sizes


def create_or_load_sample_indices(sample_sizes: dict[int, int]) -> dict[int, np.ndarray]:
    """Persist original positional indices from the notebook's target strata."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE_INDEX_PATH.exists():
        with np.load(SAMPLE_INDEX_PATH) as cached:
            indices = {year: cached[f"year_{year}"] for year in TRAIN_YEARS}
        if {year: len(index) for year, index in indices.items()} != sample_sizes:
            raise RuntimeError("Cached sample indices do not match the current 1M allocation")
        print(f"Reusing sample indices: {SAMPLE_INDEX_PATH}")
        return indices

    indices: dict[int, np.ndarray] = {}
    for year in TRAIN_YEARS:
        target = pd.read_parquet(yearly_path(year), columns=[TARGET])[TARGET]
        n_sample = sample_sizes[year]
        zero_indices = target[target == 0].index
        positive_indices = target[target > 0].index
        n_positive = round(n_sample * len(positive_indices) / len(target))
        n_zero = n_sample - n_positive
        # These are precisely the two DataFrame.sample calls in notebook cell 7.
        zero_sample = zero_indices.to_series().sample(n=n_zero, random_state=RANDOM_STATE)
        positive_sample = positive_indices.to_series().sample(n=n_positive, random_state=RANDOM_STATE)
        indices[year] = np.concatenate([zero_sample.to_numpy(), positive_sample.to_numpy()])
        print(f"{year}: {len(indices[year]):,} rows ({n_positive:,} positive)")

    np.savez_compressed(SAMPLE_INDEX_PATH, **{f"year_{year}": index for year, index in indices.items()})
    print(f"Saved reusable sample indices: {SAMPLE_INDEX_PATH}")
    return indices


def load_training_sample(columns: list[str], indices: dict[int, np.ndarray]) -> tuple[pd.DataFrame, pd.Series]:
    """Load the fixed sample and reproduce the notebook's final chronological sort."""
    required = [TARGET, DATETIME_COLUMN, *columns]
    samples = []
    for year in TRAIN_YEARS:
        frame = pd.read_parquet(yearly_path(year), columns=required).iloc[indices[year]].copy()
        samples.append(frame)
    train_sample = pd.concat(samples, ignore_index=True)
    train_sample = train_sample.sort_values(
        ["year", DATETIME_COLUMN, "hour", "grid_id"]
    ).reset_index(drop=True)
    return train_sample[columns], train_sample[TARGET]


def as_native_categorical(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype("category")
    return result


def as_catboost_input(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # CatBoost requires its categorical columns to be strings or integer codes.
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(str)
    return result


def as_hist_gradient_input(frame: pd.DataFrame, categories: dict[str, pd.Index]) -> pd.DataFrame:
    """Ordinal-encode only the two existing categorical columns for sklearn."""
    result = frame.copy()
    for column in CATEGORICAL_FEATURES:
        codes = pd.Categorical(result[column], categories=categories[column]).codes
        if (codes < 0).any():
            raise ValueError(f"Validation has an unseen category in {column}")
        result[column] = codes
    return result


def poisson_deviance_sum(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    prediction = np.maximum(y_predicted, 1e-12)
    positive = y_true > 0
    contribution = np.empty_like(prediction, dtype=np.float64)
    contribution[~positive] = 2.0 * prediction[~positive]
    actual = y_true[positive]
    estimated = prediction[positive]
    contribution[positive] = 2.0 * (actual * np.log(actual / estimated) - (actual - estimated))
    return float(contribution.sum())


def empty_accumulator() -> dict:
    n_buckets = len(CALIBRATION_EDGES) - 1
    return {
        "rows": 0, "absolute_error": 0.0, "squared_error": 0.0,
        "zero_rows": 0, "zero_absolute_error": 0.0,
        "positive_rows": 0, "positive_absolute_error": 0.0,
        "actual_sum": 0.0, "prediction_sum": 0.0,
        "poisson_deviance": 0.0,
        "bucket_rows": np.zeros(n_buckets, dtype=np.int64),
        "bucket_actual": np.zeros(n_buckets, dtype=np.float64),
        "bucket_prediction": np.zeros(n_buckets, dtype=np.float64),
        "actual_count_rows": {},
        "actual_count_prediction": {},
        "actual_count_absolute_error": {},
    }


def update_accumulator(accumulator: dict, y_true: np.ndarray, predicted: np.ndarray) -> None:
    predicted = np.maximum(np.asarray(predicted, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    errors = np.abs(y_true - predicted)
    zero = y_true == 0
    bucket = np.searchsorted(CALIBRATION_EDGES[1:], predicted, side="right")
    bucket = np.minimum(bucket, len(CALIBRATION_EDGES) - 2)
    accumulator["rows"] += len(y_true)
    accumulator["absolute_error"] += float(errors.sum())
    accumulator["squared_error"] += float(np.square(y_true - predicted).sum())
    accumulator["zero_rows"] += int(zero.sum())
    accumulator["zero_absolute_error"] += float(errors[zero].sum())
    accumulator["positive_rows"] += int((~zero).sum())
    accumulator["positive_absolute_error"] += float(errors[~zero].sum())
    accumulator["actual_sum"] += float(y_true.sum())
    accumulator["prediction_sum"] += float(predicted.sum())
    accumulator["poisson_deviance"] += poisson_deviance_sum(y_true, predicted)
    accumulator["bucket_rows"] += np.bincount(bucket, minlength=len(CALIBRATION_EDGES) - 1)
    accumulator["bucket_actual"] += np.bincount(bucket, weights=y_true, minlength=len(CALIBRATION_EDGES) - 1)
    accumulator["bucket_prediction"] += np.bincount(bucket, weights=predicted, minlength=len(CALIBRATION_EDGES) - 1)
    actual_counts, inverse = np.unique(
        y_true.astype(np.int64),
        return_inverse=True
    )

    for position, actual_count in enumerate(actual_counts):
        mask = inverse == position
        key = int(actual_count)

        accumulator["actual_count_rows"][key] = (
            accumulator["actual_count_rows"].get(key, 0)
            + int(mask.sum())
        )

        accumulator["actual_count_prediction"][key] = (
            accumulator["actual_count_prediction"].get(key, 0.0)
            + float(predicted[mask].sum())
        )

        accumulator["actual_count_absolute_error"][key] = (
            accumulator["actual_count_absolute_error"].get(key, 0.0)
            + float(errors[mask].sum())
        )

def actual_count_calibration(accumulator: dict, name: str) -> pd.DataFrame:
    count_rows = accumulator["actual_count_rows"]
    count_predictions = accumulator["actual_count_prediction"]
    count_errors = accumulator["actual_count_absolute_error"]

    rows = []

    for actual_count in sorted(count_rows):
        rows.append({
            "Model": name,
            "Actual count": actual_count,
            "Rows": count_rows[actual_count],
            "Mean predicted": (
                count_predictions[actual_count] / count_rows[actual_count]
            ),
            "Mean absolute error": (
                count_errors[actual_count] / count_rows[actual_count]
            ),
        })

    table = pd.DataFrame(rows)

    high_counts = [
        count for count in count_rows
        if count >= 4
    ]

    if high_counts:
        high_rows = sum(count_rows[count] for count in high_counts)

        table = table[
            table["Actual count"].astype(int) < 4
        ].copy()

        table = pd.concat([
            table,
            pd.DataFrame([{
                "Model": name,
                "Actual count": "4+",
                "Rows": high_rows,
                "Mean predicted": (
                    sum(
                        count_predictions[count]
                        for count in high_counts
                    ) / high_rows
                ),
                "Mean absolute error": (
                    sum(
                        count_errors[count]
                        for count in high_counts
                    ) / high_rows
                ),
            }])
        ], ignore_index=True)

    return table


def validation_batches(columns: list[str]):
    """Yield only 2024 in bounded Arrow batches; 2025 is never opened."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(yearly_path(VALIDATION_YEAR), format="parquet")
    scanner = dataset.scanner(columns=[TARGET, *columns], batch_size=PREDICTION_BATCH_SIZE)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        yield frame[columns], frame[TARGET].to_numpy()


def evaluate_in_batches(
    name: str, predictor: Callable[[pd.DataFrame], np.ndarray], columns: list[str]
) -> tuple[dict[str, float | str], pd.DataFrame, pd.DataFrame]:
    accumulator = empty_accumulator()
    for batch_number, (features, target) in enumerate(validation_batches(columns), start=1):
        update_accumulator(accumulator, target, predictor(features))
        print(f"{name}: evaluated batch {batch_number}")
    rows = accumulator["rows"]
    actual_mean = accumulator["actual_sum"] / rows
    predicted_mean = accumulator["prediction_sum"] / rows
    metrics = {
        "Model": name,
        "Status": "completed",
        "MAE": accumulator["absolute_error"] / rows,
        "RMSE": np.sqrt(accumulator["squared_error"] / rows),
        "Zero MAE": accumulator["zero_absolute_error"] / accumulator["zero_rows"],
        "Positive MAE": accumulator["positive_absolute_error"] / accumulator["positive_rows"],
        "Mean Actual": actual_mean,
        "Mean Predicted": predicted_mean,
        "Actual/Predicted": actual_mean / predicted_mean if predicted_mean else np.nan,
        "Poisson Deviance": accumulator["poisson_deviance"] / rows,
    }
    calibration = pd.DataFrame({
        "Model": name,
        "Predicted bucket": [
            f"[{CALIBRATION_EDGES[i]:g}, {CALIBRATION_EDGES[i + 1]:g}{')' if np.isfinite(CALIBRATION_EDGES[i + 1]) else ']'}"
            for i in range(len(CALIBRATION_EDGES) - 1)
        ],
        "Rows": accumulator["bucket_rows"],
        "Mean actual": accumulator["bucket_actual"] / np.maximum(accumulator["bucket_rows"], 1),
        "Mean predicted": accumulator["bucket_prediction"] / np.maximum(accumulator["bucket_rows"], 1),
    })
    calibration["Actual/predicted ratio"] = calibration["Mean actual"] / calibration["Mean predicted"].replace(0, np.nan)
    actual_count_table = actual_count_calibration(accumulator, name)
    return metrics, calibration, actual_count_table
    


def optional_module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    columns = model_columns()
    expected = {"grid_id", "lat_grid", "lon_grid", "year", "month", "day_of_week", "hour", "is_weekend", "time_period",
                "historical_grid_crime_count", "historical_grid_hour_crime_count", "historical_grid_day_crime_count",
                "historical_grid_time_period_crime_count", "historical_grid_weekend_crime_count"}
    if set(columns) != expected:
        raise RuntimeError(f"Unexpected model inputs; refusing to alter features: {columns}")
    if CATEGORICAL_FEATURES != ["grid_id", "time_period"]:
        raise RuntimeError("Categorical-feature definition changed")

    sample_sizes = current_notebook_sample_sizes()
    indices = create_or_load_sample_indices(sample_sizes)
    x_train, y_train = load_training_sample(columns, indices)
    print(f"Training sample: {x_train.shape}; target mean: {y_train.mean():.6f}")

    manifest = {
        "random_state": RANDOM_STATE, "train_years": list(TRAIN_YEARS),
        "validation_year": VALIDATION_YEAR, "test_year_used": False,
        "total_sample": int(len(y_train)), "sample_sizes": sample_sizes,
        "features": columns, "categorical_features": CATEGORICAL_FEATURES,
        "prediction_batch_size": PREDICTION_BATCH_SIZE,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    completed_metrics: list[dict] = []
    calibration_tables: list[pd.DataFrame] = []
    actual_count_tables: list[pd.DataFrame] = []
    unavailable: list[dict] = []

    def run(name: str, model, prepare: Callable[[pd.DataFrame], pd.DataFrame], fit_kwargs: dict | None = None) -> None:
        try:
            print(f"\\nTraining {name}...")
            model.fit(prepare(x_train), y_train, **(fit_kwargs or {}))
            metrics, calibration, actual_count_table = evaluate_in_batches(
                name,
                lambda frame: model.predict(prepare(frame)),
                columns,
            )
            completed_metrics.append(metrics)
            calibration_tables.append(calibration)
            actual_count_tables.append(actual_count_table)
        except Exception as error:  # Keep the other model families comparable if one is unsupported.
            print(f"{name} unavailable/failed: {error}")
            unavailable.append({"Model": name, "Status": f"unavailable/failed: {type(error).__name__}"})

    lightgbm = optional_module("lightgbm")
    if lightgbm is None:
        unavailable.append({"Model": "LightGBM Poisson", "Status": "unavailable: lightgbm not installed"})
    else:
        run("LightGBM Poisson", lightgbm.LGBMRegressor(
            objective="poisson", n_estimators=2000, learning_rate=0.01, num_leaves=63,
            max_depth=-1, min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_STATE, n_jobs=-1, max_bin=1024,
        ), as_native_categorical, {"categorical_feature": CATEGORICAL_FEATURES})

    xgboost = optional_module("xgboost")
    if xgboost is None:
        unavailable.extend([
            {"Model": "XGBoost Poisson", "Status": "unavailable: xgboost not installed"},
            {"Model": "XGBoost Tweedie", "Status": "unavailable: xgboost not installed"},
        ])
    else:
        common_xgb = dict(n_estimators=600, learning_rate=0.05, max_depth=8, min_child_weight=20,
                          subsample=0.9, colsample_bytree=0.9, tree_method="hist", enable_categorical=True,
                          random_state=RANDOM_STATE, n_jobs=-1)
        run("XGBoost Poisson", xgboost.XGBRegressor(objective="count:poisson", **common_xgb), as_native_categorical)
        run("XGBoost Tweedie", xgboost.XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=1.5, **common_xgb
        ), as_native_categorical)

    catboost = optional_module("catboost")
    if catboost is None:
        unavailable.append({"Model": "CatBoost Poisson", "Status": "unavailable: catboost not installed"})
    else:
        run("CatBoost Poisson", catboost.CatBoostRegressor(
            loss_function="Poisson", iterations=800, learning_rate=0.05, depth=8,
            random_seed=RANDOM_STATE, thread_count=-1, verbose=100, allow_writing_files=False,
        ), as_catboost_input, {"cat_features": CATEGORICAL_FEATURES})

    category_levels = {column: pd.Index(x_train[column].unique()) for column in CATEGORICAL_FEATURES}
    # sklearn's native categorical mode has a maximum of 255 categories while
    # grid_id has about 946.  This intentionally simple baseline therefore
    # uses the same two columns after ordinal encoding, not extra features.
    run("HistGradientBoosting Poisson (ordinal categories)", HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=300, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=0.1, random_state=RANDOM_STATE,
    ), lambda frame: as_hist_gradient_input(frame, category_levels))

    comparison = pd.DataFrame(completed_metrics + unavailable)
    if not comparison.empty:
        comparison.to_csv(COMPARISON_PATH, index=False)
        print("\\nFinal comparison (completed candidates sorted by Poisson deviance):")
        numeric = comparison[comparison["Status"] == "completed"]
        if not numeric.empty:
            print(numeric.sort_values("Poisson Deviance").to_string(index=False))
        else:
            print("No candidate completed; see statuses below.")
        if unavailable:
            print("\\nUnavailable/failed candidates:")
            print(comparison[comparison["Status"] != "completed"].to_string(index=False))
    if calibration_tables:
        pd.concat(calibration_tables, ignore_index=True).to_csv(CALIBRATION_PATH, index=False)
    if actual_count_tables:
        actual_count_result = pd.concat(actual_count_tables, ignore_index=True)
        actual_count_result.to_csv(ACTUAL_COUNT_CALIBRATION_PATH, index=False)
        print("\nActual-count calibration:")
        print(actual_count_result.to_string(index=False))
        print(f"\nSaved actual-count calibration: {ACTUAL_COUNT_CALIBRATION_PATH}")
    print(f"\\nArtifacts written under: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
