import pandas as pd
from src.data_generator import generate_shipments
from src.features import build_features, FEATURES
from src.explain import explain_shipment

def test_generator_size_and_targets():
    df = generate_shipments(50, seed=7)
    assert len(df) == 50
    assert "actual_total_delay_hours" in df
    assert df["actual_total_delay_hours"].ge(0).all()

def test_features_exclude_future_target():
    df = generate_shipments(50, seed=8)
    X = build_features(df)
    assert list(X.columns) == FEATURES
    assert "actual_delivery" not in X.columns
    assert "actual_total_delay_hours" not in X.columns
    assert X.isna().sum().sum() == 0

def test_explanation_mentions_uncertainty():
    df = generate_shipments(50, seed=9)
    messages = explain_shipment(df.iloc[0], 20.0)
    assert any("not a causal attribution" in x for x in messages)
