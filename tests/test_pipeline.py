import pandas as pd
import numpy as np
import pytest
from src.data_generator import generate_shipments, validate_shipments_dataset
from src.features import build_features, FEATURES, FORBIDDEN_LEAKAGE_COLUMNS
from src.explain import explain_shipment, compute_tree_attributions, get_stage_waterfall
from src.predict import load_bundle, predict_dataframe, predict_single, get_risk_category
from src.train import train

def test_generator_size_and_targets():
    df = generate_shipments(50, seed=7)
    assert len(df) == 50
    assert "actual_total_delay_hours" in df
    assert df["actual_total_delay_hours"].ge(0).all()
    assert "actual_departure" in df
    assert "actual_port_arrival" in df
    assert "actual_delivery" in df
    assert "observed_delay_at_port_hours" in df

def test_features_exclude_future_target():
    df = generate_shipments(50, seed=8)
    X = build_features(df)
    assert list(X.columns) == FEATURES
    for leak in FORBIDDEN_LEAKAGE_COLUMNS:
        assert leak not in X.columns
    assert X.isna().sum().sum() == 0

def test_forbidden_columns_leakage_rejection():
    """Verify that build_features strictly strips out all forbidden columns if passed in."""
    df = generate_shipments(30, seed=9)
    # df has all the actual downstream columns
    for col in FORBIDDEN_LEAKAGE_COLUMNS:
        assert col in df.columns or col.startswith("actual_") or "delay" in col
    X = build_features(df)
    for col in FORBIDDEN_LEAKAGE_COLUMNS:
        assert col not in X.columns

def test_eta_arithmetic_consistency():
    """Verify exact scheduled ETA and actual delivery timestamps agree with mathematical definition."""
    df = generate_shipments(40, seed=10)
    sched_dep = pd.to_datetime(df["scheduled_departure"], utc=True)
    sched_arr = pd.to_datetime(df["scheduled_arrival"], utc=True)
    planned_sum = (
        df["planned_sailing_hours"] +
        df["planned_port_hours"] +
        df["planned_customs_hours"] +
        df["planned_final_hours"]
    )
    expected_sched_arr = sched_dep + pd.to_timedelta(planned_sum, unit="h")
    # Must match within 1 second rounding
    diff_sched = (sched_arr - expected_sched_arr).abs()
    assert (diff_sched <= pd.Timedelta(seconds=2)).all()

    act_deliv = pd.to_datetime(df["actual_delivery"], utc=True)
    expected_act_deliv = sched_arr + pd.to_timedelta(df["actual_total_delay_hours"], unit="h")
    diff_act = (act_deliv - expected_act_deliv).abs()
    assert (diff_act <= pd.Timedelta(seconds=2)).all()

def test_dataset_validation_rejects_negative_delays():
    """Verify validate_shipments_dataset catches negative delays."""
    df = generate_shipments(10, seed=11)
    df_bad = df.copy()
    df_bad.loc[0, "departure_delay_hours"] = -5.0
    with pytest.raises(ValueError, match="Found negative delays"):
        validate_shipments_dataset(df_bad)

def test_dataset_validation_rejects_double_counting():
    """Verify validate_shipments_dataset catches stage sum mismatch."""
    df = generate_shipments(10, seed=12)
    df_bad = df.copy()
    df_bad.loc[0, "actual_total_delay_hours"] = df_bad.loc[0, "actual_total_delay_hours"] + 50.0
    with pytest.raises(ValueError, match="Stage delay sum does not match"):
        validate_shipments_dataset(df_bad)

def test_explanation_mentions_uncertainty():
    df = generate_shipments(50, seed=13)
    messages = explain_shipment(df.iloc[0], 20.0)
    assert any("not a causal attribution" in x.lower() or "statistical model" in x.lower() for x in messages)

def test_prediction_dataframe_confidence_intervals():
    bundle = load_bundle("artifacts/eta_model.joblib")
    df = generate_shipments(30, seed=14)
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
    df = generate_shipments(5, seed=15)
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
    df = generate_shipments(5, seed=16)
    wf = get_stage_waterfall(df.iloc[0], 28.5)
    assert len(wf) == 5
    assert list(wf["Stage"]) == [
        "1. Departure",
        "2. Ocean Sailing",
        "3. Port Processing",
        "4. Customs Clearance",
        "5. Final Delivery",
    ]
    assert (wf["Incremental Delay (Hours)"] >= 0).all()
    assert (wf["Delay Incurred (Hours)"] >= 0).all()

def test_baseline_and_model_evaluation_metrics():
    bundle = load_bundle("artifacts/eta_model.joblib")
    metrics = bundle["metrics"]
    assert "regression" in metrics
    assert "classification" in metrics
    assert "prediction_intervals" in metrics
    # AI Regressor MAE must be lower than baseline heuristic MAE
    assert metrics["regression"]["model_mae_hours"] < metrics["regression"]["baseline_heuristic_mae_hours"]
    # Confusion matrix must sum to 75 test rows
    cm = np.array(metrics["classification"]["confusion_matrix"])
    assert cm.sum() == metrics["test_rows"]
    # Empirical test coverage must be positive
    assert metrics["prediction_intervals"]["empirical_test_coverage"] > 0.50
