import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Load `02_preprocessing.py` (module name starts with digit, so import from file)
preproc_path = Path(__file__).resolve().parent / "02_preprocessing.py"
spec = importlib.util.spec_from_file_location("preproc_module", preproc_path)
preproc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preproc_module)

# Use the reusable loader to get splits and the preprocessor
X_train, X_test, y_train, y_test, preprocessor = preproc_module.load_and_split()

# Define candidate models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}


def eval_model(name, model):
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}, pipe


results = []
trained_pipes = {}

for name, m in models.items():
    res, pipe = eval_model(name, m)
    results.append(res)
    trained_pipes[name] = pipe

results_df = pd.DataFrame(results).sort_values(by="RMSE")
print("\nModel comparison:\n")
print(results_df)

best_model = results_df.iloc[0]["model"]
best_pipe = trained_pipes[best_model]
print("\nBest model:", best_model)

# Save best model
models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(exist_ok=True)
model_path = models_dir / f"best_model_{best_model}.joblib"
joblib.dump(best_pipe, model_path)
print("Saved best model to:", model_path)

