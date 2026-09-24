import joblib
import pandas as pd
from pathlib import Path

# Charger le modèle sauvegardé (chemin relatif au script)
model_path = Path(__file__).resolve().parent / "models" / "best_model_GradientBoosting.joblib"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

print("Model loaded successfully from:", model_path)

new_sample = pd.DataFrame([{
    "weekday": "Monday",
    "weekend": False,
    "shift": "Day",
    "ambient_temp_c": 20,
    "operational_load_pct": 75,
    "machine_state": "Normal",
    "maintenance_flag": 0,
    "production_speed_m_min": 4.6,
    "furnace_temp_c": 1050,
    "grid_emission_factor_kg_per_kwh": 0.55,
    "hour": 10,
    "dayofweek": 0,
    "month": 9
}])


prediction = model.predict(new_sample)

print(f"Predicted energy consumption: {prediction[0]:.2f} kWh / 15 min")
