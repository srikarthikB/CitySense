from pathlib import Path
import xgboost as xgb

MODEL_PATH = Path(__file__).parent.parent / "models" / "citysense_final_model.json"

model = None

def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
    return model

def predict_probability(features):
    dmatrix = xgb.DMatrix(features, enable_categorical=True)
    probability = get_model().predict(dmatrix)[0]
    return float(probability)