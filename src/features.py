"""Feature construction and target leakage prevention for predictions at the port-arrival checkpoint."""
import pandas as pd
import numpy as np
from typing import List

# Explicit feature allowlist: ONLY features available at the port-arrival checkpoint
BASE_FEATURES: List[str] = [
    "planned_sailing_hours",
    "planned_port_hours",
    "planned_customs_hours",
    "planned_final_hours",
    "port_congestion",
    "weather_severity",
    "departure_delay_hours",
    "sailing_delay_hours",
    "observed_delay_at_port_hours",
    "planned_remaining_hours_at_port",
]

# Engineered interaction features derived strictly from checkpoint-available signals
ENGINEERED_FEATURES: List[str] = [
    "upstream_delay_ratio",
    "congestion_weather_interaction",
    "congestion_buffer_ratio",
]

FEATURES: List[str] = BASE_FEATURES + ENGINEERED_FEATURES

# Comprehensive list of future downstream actuals strictly prohibited at prediction time
FORBIDDEN_LEAKAGE_COLUMNS: List[str] = [
    "actual_total_delay_hours",
    "actual_remaining_delay_hours",
    "late_delivery",
    "actual_delivery",
    "actual_customs_done",
    "actual_port_departure",
    "port_delay_hours",
    "customs_delay_hours",
    "final_stage_delay_hours",
    "actual_port_hours",
    "actual_customs_hours",
    "actual_final_hours",
    "cum_delay_port_departure_hours",
    "cum_delay_customs_done_hours",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only features available at the port-arrival checkpoint.

    Strictly excludes future outcomes (actual customs completion, port dwell delay,
    actual delivery timestamp, and final delay targets) to prevent target leakage.
    Ensures identical feature construction across training, evaluation, and inference.
    """
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns at checkpoint: {missing}")

    result = df[BASE_FEATURES].copy().astype(float)

    # Derive observed delay if not explicitly provided or to guarantee consistency
    if "observed_delay_at_port_hours" not in result.columns:
        result["observed_delay_at_port_hours"] = (
            result["departure_delay_hours"] + result["sailing_delay_hours"]
        )

    # -------------------------------------------------------------
    # Engineered interaction features (Available at Checkpoint)
    # -------------------------------------------------------------
    # 1. Upstream delay ratio: severity of delay relative to ocean voyage length
    result["upstream_delay_ratio"] = (
        result["observed_delay_at_port_hours"] / (result["planned_sailing_hours"] + 1e-5)
    )

    # 2. Compounding operational risk: congestion combined with severe weather
    result["congestion_weather_interaction"] = (
        result["port_congestion"] * result["weather_severity"]
    )

    # 3. Port congestion stress: congestion relative to scheduled port processing dwell
    result["congestion_buffer_ratio"] = (
        result["port_congestion"] / (result["planned_port_hours"] + 1e-5)
    )

    # Guard against accidental inclusion of forbidden future columns
    for col in FORBIDDEN_LEAKAGE_COLUMNS:
        if col in result.columns:
            del result[col]

    # Return exactly the allowlisted features with median fallback for any missing values
    return result[FEATURES].fillna(result[FEATURES].median(numeric_only=True))
