"""Feature construction for predictions made at the port-arrival checkpoint."""
import pandas as pd

FEATURES = [
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

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only features available at the port-arrival checkpoint.

    Excludes actual customs completion, actual delivery, and the final target to
    avoid target leakage in the demo model.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    result = df[FEATURES].copy()
    return result.fillna(result.median(numeric_only=True))
