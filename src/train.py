"""Train baseline, regression, and calibrated delay-risk models with held-out validation."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.features import build_features, FEATURES
from src.data_generator import generate_shipments

def train(data_path="data/shipments.csv", model_path="artifacts/eta_model.joblib", late_threshold_hours=24.0):
    """Train ML models and operational baselines on held-out test split."""
    data_file = Path(data_path)
    if not data_file.exists():
        data_file.parent.mkdir(parents=True, exist_ok=True)
        generate_shipments(300, seed=42).to_csv(data_file, index=False)

    df = pd.read_csv(data_file)
    X = build_features(df)
    y = df["actual_total_delay_hours"].astype(float)
    y_class = (y > late_threshold_hours).astype(int)

    # 25% held-out test split with fixed seed for reproducible evaluation
    X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
        X, y, y_class, test_size=0.25, random_state=42, stratify=y_class
    )

    # 1. Random Forest Regressor Pipeline
    regressor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )),
    ])
    regressor.fit(X_train, y_train)
    pred_test = np.maximum(0.0, regressor.predict(X_test))

    # 2. Operational Baselines
    # Baseline A: Naive assumption that remaining delay = 0 (pred = observed delay at port)
    observed_test = X_test["observed_delay_at_port_hours"].to_numpy()
    naive_mae = float(mean_absolute_error(y_test, observed_test))
    naive_rmse = float(mean_squared_error(y_test, observed_test) ** 0.5)

    # Baseline B: Operational heuristic (observed delay + median historical downstream residual)
    train_residual = np.maximum(0.0, y_train.to_numpy() - X_train["observed_delay_at_port_hours"].to_numpy())
    median_downstream_residual = float(np.median(train_residual))
    baseline_pred = observed_test + median_downstream_residual
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_rmse = float(mean_squared_error(y_test, baseline_pred) ** 0.5)

    # Regression metrics
    mae = float(mean_absolute_error(y_test, pred_test))
    rmse = float(mean_squared_error(y_test, pred_test) ** 0.5)
    r2 = float(r2_score(y_test, pred_test))
    mae_improvement_pct = round(((baseline_mae - mae) / baseline_mae) * 100, 2)

    # 3. Delay-Risk Classifier (predicting P(delay > late_threshold_hours))
    classifier = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
    ])
    classifier.fit(X_train, y_class_train)
    class_pred_test = classifier.predict(X_test)
    class_prob_test = classifier.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_class_test, class_pred_test))
    prec = float(precision_score(y_class_test, class_pred_test, zero_division=0))
    rec = float(recall_score(y_class_test, class_pred_test, zero_division=0))
    f1 = float(f1_score(y_class_test, class_pred_test, zero_division=0))
    roc_auc = float(roc_auc_score(y_class_test, class_prob_test))

    # 4. Residual Quantiles for Calibrated Prediction Intervals
    pred_train = regressor.predict(X_train)
    residuals_train = y_train.to_numpy() - pred_train
    q10 = float(np.percentile(residuals_train, 10))
    q25 = float(np.percentile(residuals_train, 25))
    q75 = float(np.percentile(residuals_train, 75))
    q90 = float(np.percentile(residuals_train, 90))
    res_std = float(np.std(residuals_train))

    # 5. Global Feature Importances
    rf_inner = regressor.named_steps["regressor"]
    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURES, rf_inner.feature_importances_)
    }
    # Sort descending
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "mae_hours": round(mae, 3),
        "rmse_hours": round(rmse, 3),
        "r2_score": round(r2, 3),
        "baseline_mae_hours": round(baseline_mae, 3),
        "baseline_rmse_hours": round(baseline_rmse, 3),
        "naive_baseline_mae_hours": round(naive_mae, 3),
        "mae_improvement_percent": mae_improvement_pct,
        "late_threshold_hours": late_threshold_hours,
        "classification": {
            "accuracy": round(acc, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1_score": round(f1, 3),
            "roc_auc": round(roc_auc, 3),
        },
        "feature_importances": sorted_importances,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Synthetic demonstration dataset; metrics reflect consistency with simulation mechanics."
    }

    bundle = {
        "model": regressor,
        "classifier": classifier,
        "features": FEATURES,
        "metrics": metrics,
        "baseline_median_residual": median_downstream_residual,
        "train_mean_delay": float(np.mean(y_train)),
        "residual_stats": {
            "q10": round(q10, 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "q90": round(q90, 2),
            "std": round(res_std, 2),
        },
        # Save sample of held-out test predictions for model evaluation tabs
        "test_evaluation_sample": {
            "y_test": [round(float(v), 2) for v in y_test.values],
            "pred_test": [round(float(v), 2) for v in pred_test],
            "baseline_pred": [round(float(v), 2) for v in baseline_pred],
            "observed_delay": [round(float(v), 2) for v in observed_test],
        }
    }

    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    out_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Training completed successfully. Model artifact saved to: {out_path}")
    print(json.dumps(metrics, indent=2))
    return bundle

if __name__ == "__main__":
    train()
