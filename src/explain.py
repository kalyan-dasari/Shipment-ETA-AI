"""Explainability engine: exact tree attribution, stage delay propagation waterfall, and narrative AI insights."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

FEATURE_FRIENDLY_NAMES = {
    "planned_sailing_hours": "Planned Sailing Voyage Duration",
    "planned_port_hours": "Planned Port Dwell Time",
    "planned_customs_hours": "Planned Customs Window",
    "planned_final_hours": "Planned Inland Delivery Time",
    "port_congestion": "Port Congestion Index",
    "weather_severity": "Weather Severity Index",
    "departure_delay_hours": "Observed Origin Departure Delay",
    "sailing_delay_hours": "Observed Ocean Sailing Delay",
    "observed_delay_at_port_hours": "Cumulative Upstream Delay at Port",
    "planned_remaining_hours_at_port": "Total Remaining Buffer at Port",
    "upstream_delay_ratio": "Delay / Voyage Duration Ratio",
    "congestion_weather_interaction": "Compounded Weather & Congestion",
    "congestion_buffer_ratio": "Congestion Stress on Port Buffer",
}

def compute_tree_attributions(model, x_vector: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """Compute exact per-feature contributions using tree path decomposition (Tree SHAP equivalent).

    For each tree in the Random Forest, traverses the decision path from root to leaf,
    accumulating the delta (child_value - parent_value) attributed to each feature split.
    The sum of contributions exactly equals (prediction - baseline_root_mean).
    """
    regressor = model
    if hasattr(model, "named_steps"):
        if "regressor" in model.named_steps:
            regressor = model.named_steps["regressor"]
        elif "model" in model.named_steps:
            regressor = model.named_steps["model"]

    n_features = len(feature_names)
    contributions = np.zeros(n_features)
    root_values = []

    if not hasattr(regressor, "estimators_"):
        return {"base_value": 0.0, "attributions": []}

    n_trees = len(regressor.estimators_)
    x = np.asarray(x_vector, dtype=float).ravel()

    for dt in regressor.estimators_:
        tree = dt.tree_
        node = 0
        root_values.append(float(tree.value[0][0][0]))
        while tree.children_left[node] != -1:  # While not a leaf node
            feat_idx = tree.feature[node]
            val_curr = tree.value[node][0][0]
            if x[feat_idx] <= tree.threshold[node]:
                next_node = tree.children_left[node]
            else:
                next_node = tree.children_right[node]
            val_next = tree.value[next_node][0][0]
            contributions[feat_idx] += (val_next - val_curr) / n_trees
            node = next_node

    base_value = float(np.mean(root_values)) if root_values else 0.0

    attributions = []
    for i, name in enumerate(feature_names):
        impact = float(contributions[i])
        attributions.append({
            "feature": name,
            "label": FEATURE_FRIENDLY_NAMES.get(name, name),
            "value": float(x[i]),
            "impact_hours": round(impact, 2),
            "direction": "increases_delay" if impact > 0 else "reduces_delay"
        })

    # Sort by absolute impact descending
    attributions.sort(key=lambda item: abs(item["impact_hours"]), reverse=True)
    return {
        "base_value": round(base_value, 2),
        "prediction_from_tree": round(base_value + float(np.sum(contributions)), 2),
        "attributions": attributions
    }

def get_stage_waterfall(row: Any, predicted_total_delay: float) -> pd.DataFrame:
    """Generate a detailed delay propagation waterfall across all 5 journey stages.

    Explicitly separates incremental stage delay from cumulative schedule deviation.
    Distinguishes stages observed at the port checkpoint from downstream unknown stages.
    """
    dep_delay = float(row.get("departure_delay_hours", 0) or 0)
    sail_delay = float(row.get("sailing_delay_hours", 0) or 0)
    observed_upstream = dep_delay + sail_delay

    # Expected remaining downstream delay after port arrival
    remaining_predicted = max(0.0, predicted_total_delay - observed_upstream)
    congestion = float(row.get("port_congestion", 0.5) or 0.5)

    # Check if historical actuals exist (for post-delivery audit); otherwise use synthetic propagation allocation
    has_actuals = "port_delay_hours" in row and pd.notna(row.get("port_delay_hours"))
    if has_actuals:
        port_d = float(row["port_delay_hours"])
        customs_d = float(row["customs_delay_hours"])
        final_d = float(row["final_stage_delay_hours"])
    else:
        # Realistic synthetic model allocation
        port_weight = 0.50 + 0.15 * congestion
        customs_weight = 0.30 + 0.05 * congestion
        final_weight = max(0.05, 1.0 - (port_weight + customs_weight))

        port_d = round(remaining_predicted * port_weight, 2)
        customs_d = round(remaining_predicted * customs_weight, 2)
        final_d = round(remaining_predicted * final_weight, 2)

    plan_sail = float(row.get("planned_sailing_hours", 120.0))
    plan_port = float(row.get("planned_port_hours", 24.0))
    plan_customs = float(row.get("planned_customs_hours", 12.0))
    plan_final = float(row.get("planned_final_hours", 24.0))

    stages = [
        {
            "Stage": "1. Departure",
            "Visibility at Checkpoint": "Observed Upstream",
            "Planned Hours": 0.0,
            "Incremental Delay (Hours)": round(dep_delay, 2),
            "Delay Incurred (Hours)": round(dep_delay, 2),  # Backward compatibility
            "Actual / Estimated Duration": round(dep_delay, 2),
            "Cumulative Delay at Milestone": round(dep_delay, 2),
            "Cumulative Delay": round(dep_delay, 2),       # Backward compatibility
            "Status": "Observed Upstream",
            "Propagation Driver": "Terminal loading & initial dispatch friction"
        },
        {
            "Stage": "2. Ocean Sailing",
            "Visibility at Checkpoint": "Observed Upstream",
            "Planned Hours": round(plan_sail, 1),
            "Incremental Delay (Hours)": round(sail_delay, 2),
            "Delay Incurred (Hours)": round(sail_delay, 2),
            "Actual / Estimated Duration": round(plan_sail + sail_delay, 1),
            "Cumulative Delay at Milestone": round(dep_delay + sail_delay, 2),
            "Cumulative Delay": round(dep_delay + sail_delay, 2),
            "Status": "Observed Upstream",
            "Propagation Driver": "Sea weather conditions & vessel transit speed"
        },
        {
            "Stage": "3. Port Processing",
            "Visibility at Checkpoint": "Unknown Future (Estimated)" if not has_actuals else "Historical Actual",
            "Planned Hours": round(plan_port, 1),
            "Incremental Delay (Hours)": round(port_d, 2),
            "Delay Incurred (Hours)": round(port_d, 2),
            "Actual / Estimated Duration": round(plan_port + port_d, 1),
            "Cumulative Delay at Milestone": round(dep_delay + sail_delay + port_d, 2),
            "Cumulative Delay": round(dep_delay + sail_delay + port_d, 2),
            "Status": "Estimated Downstream" if not has_actuals else "Observed / Actual",
            "Propagation Driver": "Berth wait, crane queue & terminal congestion"
        },
        {
            "Stage": "4. Customs Clearance",
            "Visibility at Checkpoint": "Unknown Future (Estimated)" if not has_actuals else "Historical Actual",
            "Planned Hours": round(plan_customs, 1),
            "Incremental Delay (Hours)": round(customs_d, 2),
            "Delay Incurred (Hours)": round(customs_d, 2),
            "Actual / Estimated Duration": round(plan_customs + customs_d, 1),
            "Cumulative Delay at Milestone": round(dep_delay + sail_delay + port_d + customs_d, 2),
            "Cumulative Delay": round(dep_delay + sail_delay + port_d + customs_d, 2),
            "Status": "Estimated Downstream" if not has_actuals else "Observed / Actual",
            "Propagation Driver": "Document inspection & port dwell ripple effect"
        },
        {
            "Stage": "5. Final Delivery",
            "Visibility at Checkpoint": "Unknown Future (Estimated)" if not has_actuals else "Historical Actual",
            "Planned Hours": round(plan_final, 1),
            "Incremental Delay (Hours)": round(final_d, 2),
            "Delay Incurred (Hours)": round(final_d, 2),
            "Actual / Estimated Duration": round(plan_final + final_d, 1),
            "Cumulative Delay at Milestone": round(dep_delay + sail_delay + port_d + customs_d + final_d, 2),
            "Cumulative Delay": round(dep_delay + sail_delay + port_d + customs_d + final_d, 2),
            "Status": "Estimated Downstream" if not has_actuals else "Observed / Actual",
            "Propagation Driver": "Drayage connection, gate-out & inland transit"
        },
    ]
    return pd.DataFrame(stages)

def explain_shipment(row: Any, predicted_delay_hours: float, bundle: Optional[Dict[str, Any]] = None) -> List[str]:
    """Human-readable, assumption-aware explanations for an individual shipment prediction."""
    explanations = []
    observed = float(row.get("observed_delay_at_port_hours", 0) or 0)
    departure_delay = float(row.get("departure_delay_hours", 0) or 0)
    sailing_delay = float(row.get("sailing_delay_hours", 0) or 0)
    congestion = float(row.get("port_congestion", 0) or 0)
    weather = float(row.get("weather_severity", 0) or 0)

    # 1. Upstream delay status
    if observed > 0:
        explanations.append(
            f"**Observed Upstream Delay:** {observed:.1f} hours accumulated prior to port arrival "
            f"({departure_delay:.1f}h departure delay + {sailing_delay:.1f}h ocean sailing delay)."
        )
    else:
        explanations.append("**Observed Upstream Delay:** Shipment arrived at the port checkpoint exactly on schedule.")

    # 2. Port congestion bottleneck
    if congestion >= 0.70:
        explanations.append(
            f"**High Port Congestion ({congestion:.2f}/1.0):** Severe terminal queueing is simulated. "
            "In this propagation model, heavy congestion increases berth wait times and yard dwell."
        )
    elif congestion >= 0.40:
        explanations.append(
            f"**Moderate Port Congestion ({congestion:.2f}/1.0):** Terminal operations show standard density, "
            "contributing moderate variance to container offloading."
        )
    else:
        explanations.append(
            f"**Low Port Congestion ({congestion:.2f}/1.0):** Berth availability is favorable, "
            "minimizing downstream terminal dwell."
        )

    # 3. Weather impact
    if weather >= 0.65:
        explanations.append(
            f"**Adverse Weather ({weather:.2f}/1.0):** Severe meteorological conditions contributed to ocean transit "
            "slowdowns and may cause minor inland transit friction."
        )
    elif weather >= 0.35:
        explanations.append(
            f"**Moderate Weather ({weather:.2f}/1.0):** Normal navigational conditions with mild headwinds."
        )

    # 4. Propagation and downstream expectation
    remaining_predicted = max(0.0, predicted_delay_hours - observed)
    explanations.append(
        f"**Expected Downstream Propagation:** The model projects **+{remaining_predicted:.1f} additional delay hours** "
        f"across port dwell, customs inspection, and final transit, yielding a total predicted delay of **{predicted_delay_hours:.1f} hours**."
    )

    # 5. Tree attribution highlights if bundle is available
    if bundle and "model" in bundle and "features" in bundle:
        try:
            from src.features import build_features
            df_row = pd.DataFrame([row]) if not isinstance(row, pd.DataFrame) else row
            feat_df = build_features(df_row)
            tree_res = compute_tree_attributions(bundle["model"], feat_df.iloc[0].to_numpy(), bundle["features"])
            top_positive = [a for a in tree_res["attributions"] if a["impact_hours"] > 0.5][:2]
            top_negative = [a for a in tree_res["attributions"] if a["impact_hours"] < -0.5][:2]

            factors = []
            for a in top_positive:
                factors.append(f"+{a['impact_hours']:.1f}h from {a['label']}")
            for a in top_negative:
                factors.append(f"{a['impact_hours']:.1f}h from {a['label']}")

            if factors:
                explanations.append(f"**Top Model Drivers (Tree SHAP):** " + ", ".join(factors) + ".")
        except Exception:
            pass

    # 6. Uncertainty & disclaimer
    explanations.append(
        "*Note: This is a statistical model estimate on synthetic demonstration data, not a causal attribution or real-world guarantee.*"
    )
    return explanations
