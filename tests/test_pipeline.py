import pandas as pd
import numpy as np
import pytest
from src.data_generator import generate_shipments
from src.features import build_features, FEATURES, FORBIDDEN_LEAKAGE_COLUMNS
from src.explain import explain_shipment, compute_tree_attributions, get_stage_waterfall
from src.predict import load_bundle, predict_dataframe, predict_single, get_risk_category

def test_generator_size_and_targets():
    df = generate_shipments(50, seed=7)
    assert len(df) == 50
    assert "actual_total_delay_hours" in df
    assert df["actual_total_delay_hours"].ge(0).all()
    assert "actual_departure" in df
    assert "actual_port_arrival" in df
    assert "actual_delivery" in df

def test_features_exclude_future_target():
    df = generate_shipments(50, seed=8)
    X = build_features(df)
    assert list(X.columns) == FEATURES
    for leak in FORBIDDEN_LEAKAGE_COLUMNS:
        assert leak not in X.columns
    assert X.isna().sum().sum() == 0

def test_explanation_mentions_uncertainty():
    df = generate_shipments(50, seed=9)
    messages = explain_shipment(df.iloc[0], 20.0)
    assert any("not a causal attribution" in x.lower() or "statistical model" in x.lower() for x in messages)

def test_prediction_dataframe_confidence_intervals():
    bundle = load_bundle("artifacts/eta_model.joblib")
    df = generate_shipments(30, seed=10)
    preds = predict_dataframe(df, bundle)
    assert len(preds) == 30
    assert "predicted_delay_hours" in preds
    assert "predicted_delay_lower_hours" in preds
    assert "predicted_delay_upper_hours" in preds
    assert (preds["predicted_delay_lower_hours"] <= preds["predicted_delay_upper_hours"]).all()
    assert (preds["late_probability"] >= 0.0).all()
    assert (preds["late_probability"] <= 1.0).all()
    assert set(preds["late_risk"]).issubset({"Low", "Moderate", "High", "Critical"})

def test_single_prediction_and_attributions():
    bundle = load_bundle("artifacts/eta_model.joblib")
    df = generate_shipments(10, seed=11)
    record = df.iloc[0].to_dict()
    res = predict_single(record, bundle)
    assert "predicted_delay_hours" in res
    assert "tree_attributions" in res
    assert "waterfall" in res
    assert "narrative_explanations" in res

    # Verify tree attributions sum to prediction delta within numerical tolerance
    attr = res["tree_attributions"]
    base = attr["base_value"]
    contrib_sum = sum(a["impact_hours"] for a in attr["attributions"])
    reconstructed = base + contrib_sum
    assert abs(reconstructed - attr["prediction_from_tree"]) < 0.1

def test_stage_waterfall_structure():
    df = generate_shipments(10, seed=12)
    wf = get_stage_waterfall(df.iloc[0], 28.5)
    assert len(wf) == 5
    assert list(wf["Stage"]) == [
        "1. Departure",
        "2. Ocean Sailing",
        "3. Port Processing",
        "4. Customs Clearance",
        "5. Final Delivery",
    ]
    assert (wf["Delay Incurred (Hours)"] >= 0).all()
