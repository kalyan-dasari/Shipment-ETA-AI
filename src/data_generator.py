"""Generate deterministic synthetic shipment journeys for the ETA assignment."""
from pathlib import Path
import numpy as np
import pandas as pd

STAGES = ["departure", "sailing", "port", "customs", "delivery"]

CARRIERS = ["Maersk Line", "MSC Mediterranean", "CMA CGM Group", "Hapag-Lloyd", "Ocean Network Express"]
VESSELS = [
    "Pacific Voyager", "Atlantic Pioneer", "Ever Apex", "Nordic Horizon",
    "Eastern Spirit", "Baltic Trader", "Ocean Titan", "Solaris Star"
]

def generate_shipments(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic shipment records with realistic, linked stage delays.

    Synthetic data only: effects are assumptions for demonstration, not empirical claims.
    Delay propagation cascade:
      Weather & origin friction -> Departure delay
      Weather & departure delay -> Sailing delay
      Sailing delay & port congestion -> Port processing / dwell delay
      Port delay & port congestion -> Customs clearance delay
      Customs delay & local weather -> Final delivery / inland transit delay
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = np.random.default_rng(seed)
    origins = np.array(["Shanghai", "Singapore", "Mumbai", "Rotterdam", "Los Angeles", "Busan"])
    destinations = np.array(["Rotterdam", "Sydney", "Dubai", "New York", "Hamburg", "Antwerp"])
    ports = np.array(["Port Alpha (Singapore)", "Port Beta (Rotterdam)", "Port Gamma (Los Angeles)", "Port Delta (Dubai)", "Port Epsilon (Busan)"])
    priorities = np.array(["Standard", "Express", "Economy"])
    start = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []

    for i in range(n):
        origin = str(rng.choice(origins))
        destination = str(rng.choice(destinations))
        while destination == origin:
            destination = str(rng.choice(destinations))
        route = f"{origin}-{destination}"
        port = str(rng.choice(ports))
        carrier = str(rng.choice(CARRIERS))
        vessel = str(rng.choice(VESSELS))
        priority = str(rng.choice(priorities, p=[0.60, 0.25, 0.15]))
        teu = int(rng.integers(1200, 18500))

        # Planned durations (hours)
        scheduled_departure = start + pd.Timedelta(hours=int(rng.integers(0, 24 * 365)))
        planned_sailing = float(rng.uniform(72, 360))
        planned_port = float(rng.uniform(8, 36))
        planned_customs = float(rng.uniform(4, 24))
        planned_final = float(rng.uniform(12, 60))
        scheduled_arrival = scheduled_departure + pd.Timedelta(
            hours=planned_sailing + planned_port + planned_customs + planned_final
        )

        # Environmental & operational signals (0.0 to 1.0)
        congestion = float(rng.beta(2.2, 2.2))
        weather = float(rng.beta(1.8, 3.2))

        # Linked delay propagation cascade (non-negative hours)
        departure_delay = max(0.0, float(rng.normal(2.0 + 8.5 * weather, 4.5)))
        sailing_delay = max(0.0, float(rng.normal(3.0 + 13.5 * weather + 0.15 * departure_delay, 6.0)))
        port_delay = max(0.0, float(rng.normal(2.0 + 26.0 * congestion + 0.22 * sailing_delay, 7.5)))
        customs_delay = max(0.0, float(rng.normal(1.5 + 13.0 * congestion + 0.18 * port_delay, 5.0)))
        final_delay = max(0.0, float(rng.normal(1.0 + 0.12 * customs_delay + 3.0 * weather, 4.0)))

        total_delay = departure_delay + sailing_delay + port_delay + customs_delay + final_delay
        observed_at_port = departure_delay + sailing_delay
        remaining_delay = port_delay + customs_delay + final_delay

        # Coherent chronological timestamps across all stages
        actual_departure = scheduled_departure + pd.Timedelta(hours=departure_delay)
        planned_port_arrival = scheduled_departure + pd.Timedelta(hours=planned_sailing)
        actual_port_arrival = planned_port_arrival + pd.Timedelta(hours=observed_at_port)

        planned_port_departure = planned_port_arrival + pd.Timedelta(hours=planned_port)
        actual_port_departure = actual_port_arrival + pd.Timedelta(hours=planned_port + port_delay)

        planned_customs_done = planned_port_departure + pd.Timedelta(hours=planned_customs)
        actual_customs_done = actual_port_departure + pd.Timedelta(hours=planned_customs + customs_delay)

        actual_delivery = scheduled_arrival + pd.Timedelta(hours=total_delay)

        rows.append({
            "shipment_id": f"SHP-{i+1:04d}",
            "carrier": carrier,
            "vessel_name": vessel,
            "priority": priority,
            "container_teu": teu,
            "origin": origin,
            "destination": destination,
            "route": route,
            "port": port,
            "scheduled_departure": scheduled_departure,
            "scheduled_arrival": scheduled_arrival,
            "planned_sailing_hours": round(planned_sailing, 2),
            "planned_port_hours": round(planned_port, 2),
            "planned_customs_hours": round(planned_customs, 2),
            "planned_final_hours": round(planned_final, 2),
            "actual_departure": actual_departure,
            "planned_port_arrival": planned_port_arrival,
            "actual_port_arrival": actual_port_arrival,
            "planned_port_departure": planned_port_departure,
            "actual_port_departure": actual_port_departure,
            "planned_customs_done": planned_customs_done,
            "actual_customs_done": actual_customs_done,
            "actual_delivery": actual_delivery,
            "port_congestion": round(congestion, 3),
            "weather_severity": round(weather, 3),
            "departure_delay_hours": round(departure_delay, 2),
            "sailing_delay_hours": round(sailing_delay, 2),
            "port_delay_hours": round(port_delay, 2),
            "customs_delay_hours": round(customs_delay, 2),
            "final_stage_delay_hours": round(final_delay, 2),
            "actual_total_delay_hours": round(total_delay, 2),
            "actual_remaining_delay_hours": round(remaining_delay, 2),
            "late_delivery": int(total_delay > 24),
            "planned_remaining_hours_at_port": round(planned_port + planned_customs + planned_final, 2),
            "observed_delay_at_port_hours": round(observed_at_port, 2),
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
