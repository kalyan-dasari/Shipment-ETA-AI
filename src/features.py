"""Feature construction for predictions made at the port-arrival checkpoint."""
import pandas as pd
import numpy as np

# Base features required at the port arrival checkpoint
BASE_FEATURES = [
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

# Engineered domain features capturing propagation dynamics
ENGINEERED_FEATURES = [
    "upstream_delay_ratio",
    "congestion_weather_interaction",
    "congestion_buffer_ratio",
]

FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

FORBIDDEN_LEAKAGE_COLUMNS = [
    "actual_total_delay_hours",
    "actual_remaining_delay_hours",
    "late_delivery",
    "actual_delivery",
    "actual_customs_done",
    "actual_port_departure",
    "port_delay_hours",
    "customs_delay_hours",
    "final_stage_delay_hours",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only features available at the port-arrival checkpoint.

    Excludes actual customs completion, port dwell delay, actual delivery,
    and the final target to avoid target leakage.
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

    # Engineered interaction features
    # 1. Upstream delay ratio: delay relative to ocean voyage length
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

    # Verify no leakage columns are present in output
    for col in FORBIDDEN_LEAKAGE_COLUMNS:
        if col in result.columns:
            del result[col]

    return result[FEATURES].fillna(result[FEATURES].median(numeric_only=True))
