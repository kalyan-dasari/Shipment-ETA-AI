"""Train baseline and ML models with a held-out test set."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from src.features import build_features

def train(data_path="data/shipments.csv", model_path="artifacts/eta_model.joblib"):
    df = pd.read_csv(data_path)
    X = build_features(df)
    y = df["actual_total_delay_hours"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    pred = np.maximum(0, model.predict(X_test))
    # A simple operational baseline: observed upstream delay plus typical residual
    # delay learned only from the training set.
    train_residual = np.maximum(0, y_train.to_numpy() - X_train["observed_delay_at_port_hours"].to_numpy())
    baseline_pred = X_test["observed_delay_at_port_hours"].to_numpy() + float(np.median(train_residual))
    metrics = {
        "test_rows": int(len(y_test)),
        "mae_hours": round(float(mean_absolute_error(y_test, pred)), 3),
        "rmse_hours": round(float(mean_squared_error(y_test, pred) ** 0.5), 3),
        "baseline_mae_hours": round(float(mean_absolute_error(y_test, baseline_pred)), 3),
        "late_threshold_hours": 24,
        "note": "Synthetic data only; metrics are illustrative and not real-world performance."
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns), "metrics": metrics}, model_path)
    Path(model_path).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return model, metrics

if __name__ == "__main__":
    train()
