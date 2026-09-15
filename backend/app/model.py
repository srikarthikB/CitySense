from pathlib import Path
import xgboost as xgb

MODEL_PATH = Path(__file__).parent.parent / "models" / "citysense_final_model.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = xgb.Booster()
model.load_model(MODEL_PATH)