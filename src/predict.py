"""Prediction helpers for a shipment observed at the port-arrival checkpoint."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features import build_features

def load_bundle(path="artifacts/eta_model.joblib"):
    if not Path(path).exists():
        raise FileNotFoundError("Model not found. Run: python -m src.data_generator && python -m src.train")
    return joblib.load(path)

def predict_dataframe(df: pd.DataFrame, bundle=None) -> pd.DataFrame:
    bundle = bundle or load_bundle()
    X = build_features(df)
    predicted_delay = np.maximum(0, bundle["model"].predict(X))
    out = pd.DataFrame(index=df.index)
    out["shipment_id"] = df.get("shipment_id", pd.Series("unknown", index=df.index))
    out["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"], utc=True)
    out["predicted_delay_hours"] = np.round(predicted_delay, 2)
    out["predicted_eta"] = out["scheduled_arrival"] + pd.to_timedelta(predicted_delay, unit="h")
    out["late_risk"] = np.where(predicted_delay > 24, "High", "Lower")
    out["observed_delay_at_port_hours"] = df["observed_delay_at_port_hours"].astype(float)
    out["predicted_remaining_delay_hours"] = np.round(
        np.maximum(0, predicted_delay - out["observed_delay_at_port_hours"]), 2
    )
    return out
