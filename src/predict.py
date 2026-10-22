"""Prediction engine for shipment ETA, delay estimation, confidence intervals, and risk scoring."""
from pathlib import Path
from typing import Dict, Any, Union, Optional
import joblib
import numpy as np
import pandas as pd

from src.features import build_features, FEATURES

def load_bundle(path: str = "artifacts/eta_model.joblib") -> Dict[str, Any]:
    """Load model bundle containing regressor, classifier, residual stats, and metrics."""
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model not found at {bundle_path}. Please run: python -m src.data_generator && python -m src.train"
        )
    return joblib.load(bundle_path)

def get_risk_category(prob: float) -> str:
    """Classify late arrival probability into intuitive operational risk categories."""
    if prob >= 0.75:
        return "Critical"
    elif prob >= 0.50:
        return "High"
    elif prob >= 0.25:
        return "Moderate"
    return "Low"

def predict_dataframe(df: pd.DataFrame, bundle: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Vectorized inference on a batch of shipments observed at the port checkpoint."""
    bundle = bundle or load_bundle()
    X = build_features(df)
    
    # 1. Point prediction of total delay in hours
    regressor = bundle["model"]
    predicted_delay = np.maximum(0.0, regressor.predict(X))

    # 2. Confidence interval bounds based on Out-Of-Bag empirical residual quantiles
    residuals = bundle.get("residual_stats", {"q10": -14.94, "q90": 16.76})
    lower_bound = np.maximum(0.0, predicted_delay + residuals["q10"])
    upper_bound = np.maximum(lower_bound, predicted_delay + residuals["q90"])

    # 3. Calibrated late probability (P(delay > 24 hours))
    classifier = bundle.get("classifier")
    if classifier is not None:
        try:
            late_probs = classifier.predict_proba(X)[:, 1]
        except Exception:
            late_probs = np.where(predicted_delay > 24, 0.85, 0.15)
    else:
        late_probs = np.where(predicted_delay > 24, 0.85, 0.15)

    out = pd.DataFrame(index=df.index)
    out["shipment_id"] = df.get("shipment_id", [f"SHP-{i+1:04d}" for i in range(len(df))])
    
    # Scheduled arrival timestamp (ISO8601 parsing safe across pandas versions)
    if pd.api.types.is_datetime64_any_dtype(df["scheduled_arrival"]):
        sched_arrival = pd.to_datetime(df["scheduled_arrival"], utc=True)
    else:
        sched_arrival = pd.to_datetime(df["scheduled_arrival"], utc=True, format="ISO8601")
    out["scheduled_arrival"] = sched_arrival
    
    out["predicted_delay_hours"] = np.round(predicted_delay, 2)
    out["predicted_delay_lower_hours"] = np.round(lower_bound, 2)
    out["predicted_delay_upper_hours"] = np.round(upper_bound, 2)
    
    # Predicted ETA timestamps
    out["predicted_eta"] = sched_arrival + pd.to_timedelta(predicted_delay, unit="h")
    out["predicted_eta_lower"] = sched_arrival + pd.to_timedelta(lower_bound, unit="h")
    out["predicted_eta_upper"] = sched_arrival + pd.to_timedelta(upper_bound, unit="h")

    out["late_probability"] = np.round(late_probs, 3)
    out["late_risk"] = [get_risk_category(p) for p in late_probs]

    # Observed vs remaining delay
    obs_delay = df.get("observed_delay_at_port_hours")
    if obs_delay is None:
        obs_delay = df["departure_delay_hours"] + df["sailing_delay_hours"]
    out["observed_delay_at_port_hours"] = np.round(obs_delay.astype(float), 2)
    out["predicted_remaining_delay_hours"] = np.round(
        np.maximum(0.0, predicted_delay - out["observed_delay_at_port_hours"]), 2
    )

    return out

def predict_single(record: Union[Dict[str, Any], pd.Series], bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Single-shipment inference with confidence intervals, risk score, and full explainability breakdown."""
    bundle = bundle or load_bundle()
    if isinstance(record, dict):
        df_row = pd.DataFrame([record])
    elif isinstance(record, pd.Series):
        df_row = pd.DataFrame([record.to_dict()])
    else:
        df_row = record

    preds = predict_dataframe(df_row, bundle).iloc[0].to_dict()

    # Calculate local Tree SHAP attributions
    from src.explain import compute_tree_attributions, get_stage_waterfall, explain_shipment
    X = build_features(df_row)
    tree_attr = compute_tree_attributions(bundle["model"], X.iloc[0].to_numpy(), bundle["features"])
    waterfall_df = get_stage_waterfall(df_row.iloc[0], float(preds["predicted_delay_hours"]))
    narrative = explain_shipment(df_row.iloc[0], float(preds["predicted_delay_hours"]), bundle=bundle)

    preds["tree_attributions"] = tree_attr
    preds["waterfall"] = waterfall_df.to_dict(orient="records")
    preds["narrative_explanations"] = narrative
    return preds
