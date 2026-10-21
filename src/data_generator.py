"""Generate deterministic synthetic shipment journeys for the ETA assignment."""
from pathlib import Path
import numpy as np
import pandas as pd

STAGES = ["departure", "port_arrival", "customs", "delivery"]

def generate_shipments(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic shipment records with realistic, linked stage delays.

    Synthetic data only: effects are assumptions for demonstration, not empirical claims.
    """
    if n < 30:
        raise ValueError("n must be at least 30 to support a meaningful train/test split")
    rng = np.random.default_rng(seed)
    origins = np.array(["Shanghai", "Singapore", "Mumbai", "Rotterdam", "Los Angeles"])
    destinations = np.array(["Rotterdam", "Sydney", "Dubai", "New York", "Hamburg"])
    ports = np.array(["Port Alpha", "Port Beta", "Port Gamma", "Port Delta"])
    start = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []

    for i in range(n):
        origin = str(rng.choice(origins))
        destination = str(rng.choice(destinations))
        route = f"{origin}-{destination}"
        scheduled_departure = start + pd.Timedelta(hours=int(rng.integers(0, 24 * 365)))
        planned_sailing = float(rng.uniform(72, 360))
        planned_port = float(rng.uniform(8, 30))
        planned_customs = float(rng.uniform(4, 20))
        planned_final = float(rng.uniform(12, 60))
        scheduled_arrival = scheduled_departure + pd.Timedelta(
            hours=planned_sailing + planned_port + planned_customs + planned_final
        )

        congestion = float(rng.beta(2, 2))
        weather = float(rng.beta(1.8, 3.0))
        # Upstream delays influence later stages, with some recovery/noise.
        departure_delay = max(0.0, float(rng.normal(2 + 8 * weather, 5)))
        sailing_delay = max(0.0, float(rng.normal(3 + 12 * weather + 0.12 * departure_delay, 7)))
        port_delay = max(0.0, float(rng.normal(2 + 28 * congestion + 0.25 * sailing_delay, 8)))
        customs_delay = max(0.0, float(rng.normal(1 + 12 * congestion + 0.20 * port_delay, 5)))
        final_delay = max(0.0, float(rng.normal(1 + 0.10 * customs_delay, 4)))
        total_delay = departure_delay + sailing_delay + port_delay + customs_delay + final_delay

        actual_departure = scheduled_departure + pd.Timedelta(hours=departure_delay)
        planned_port_arrival = scheduled_departure + pd.Timedelta(hours=planned_sailing)
        actual_port_arrival = planned_port_arrival + pd.Timedelta(hours=departure_delay + sailing_delay)
        planned_customs_done = planned_port_arrival + pd.Timedelta(hours=planned_port + planned_customs)
        actual_customs_done = actual_port_arrival + pd.Timedelta(hours=planned_port + planned_customs + port_delay + customs_delay)
        actual_delivery = scheduled_arrival + pd.Timedelta(hours=total_delay)

        # Simulate an observation made at port arrival: later-stage actuals are unknown.
        rows.append({
            "shipment_id": f"SHP-{i+1:04d}",
            "origin": origin,
            "destination": destination,
            "route": route,
            "port": str(rng.choice(ports)),
            "scheduled_departure": scheduled_departure,
            "scheduled_arrival": scheduled_arrival,
            "planned_sailing_hours": round(planned_sailing, 2),
            "planned_port_hours": round(planned_port, 2),
            "planned_customs_hours": round(planned_customs, 2),
            "planned_final_hours": round(planned_final, 2),
            "actual_departure": actual_departure,
            "actual_port_arrival": actual_port_arrival,
            "planned_port_arrival": planned_port_arrival,
            "port_congestion": round(congestion, 3),
            "weather_severity": round(weather, 3),
            "departure_delay_hours": round(departure_delay, 2),
            "sailing_delay_hours": round(sailing_delay, 2),
            "port_delay_hours": round(port_delay, 2),
            "customs_delay_hours": round(customs_delay, 2),
            "final_stage_delay_hours": round(final_delay, 2),
            "actual_customs_done": actual_customs_done,
            "actual_delivery": actual_delivery,
            "actual_total_delay_hours": round(total_delay, 2),
            "late_delivery": int(total_delay > 24),
            "planned_remaining_hours_at_port": round(planned_port + planned_customs + planned_final, 2),
            "observed_delay_at_port_hours": round(departure_delay + sailing_delay, 2),
        })
    return pd.DataFrame(rows)

def save_dataset(path: str = "data/shipments.csv", n: int = 300, seed: int = 42) -> pd.DataFrame:
    df = generate_shipments(n=n, seed=seed)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df

if __name__ == "__main__":
    df = save_dataset()
    print(f"Generated {len(df)} synthetic shipments at data/shipments.csv")
