from pathlib import Path
import xgboost as xgb

MODEL_PATH = Path(__file__).parent.parent / "models" / "citysense_final_model.json"

model = None

def get_model():
    global model

    print("DEBUG: get_model() started", flush=True)

    if model is None:
        print("DEBUG: loading XGBoost model", flush=True)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        print(
            f"DEBUG: model size = {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB",
            flush=True
        )

        model = xgb.Booster()
        model.load_model(MODEL_PATH)

        print("DEBUG: XGBoost model loaded", flush=True)

    return model

def predict_probability(features):
    dmatrix = xgb.DMatrix(features, enable_categorical=True)
    probability = get_model().predict(dmatrix)[0]
    return float(probability)