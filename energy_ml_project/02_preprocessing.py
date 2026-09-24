import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_and_split(csv_path: str | Path | None = None, test_size=0.2, random_state=42):
    """Load dataset (path relative to this script if None) and return train/test splits.

    Returns: X_train, X_test, y_train, y_test, preprocessor
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "data" / "steel_line_energy_dataset.csv"
    df = pd.read_csv(csv_path)

    # timestamp -> temporal features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    target = "energy_consumption_kwh_15min"
    drop_cols = ["timestamp", "co2_emissions_kg_15min"]
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]

    # identify numeric / categorical
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = load_and_split()
    print("Train size:", X_train.shape, "Test size:", X_test.shape)
    print("Preprocessing setup OK.")
