"""Generate deterministic synthetic shipment journeys with explicit stage-by-stage delay propagation."""
from pathlib import Path
from typing import Dict, List, Tuple
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
    
    The simulated shipment has five stages:
      1. Departure handling (terminal loading & initial gate-out)
      2. Ocean sailing / transit
      3. Port processing / dwell & crane unloading
      4. Customs clearance inspection
      5. Final delivery inland transit

    For every stage, the simulation explicitly distinguishes:
      - Planned duration (hours)
      - Actual duration (hours)
      - Planned start and completion timestamps
      - Actual start and completion timestamps
      - Incremental stage delay (actual duration minus planned duration)
      - Cumulative delay inherited from earlier stages (schedule deviation at milestone)

    Delay propagation cascade:
      Weather & origin friction -> Departure incremental delay
      Weather & departure delay -> Sailing incremental delay
      Sailing delay & port congestion -> Port processing incremental delay
      Port delay & port congestion -> Customs clearance incremental delay
      Customs delay & local weather -> Final delivery incremental delay
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

        # ---------------------------------------------------------
        # 1. Planned Durations (hours, rounded to 2 decimals)
        # ---------------------------------------------------------
        scheduled_departure = start + pd.Timedelta(hours=int(rng.integers(0, 24 * 365)))
        planned_sailing = round(float(rng.uniform(72, 360)), 2)
        planned_port = round(float(rng.uniform(8, 36)), 2)
        planned_customs = round(float(rng.uniform(4, 24)), 2)
        planned_final = round(float(rng.uniform(12, 60)), 2)
        
        # Scheduled Arrival = Scheduled Departure + sum of planned stage durations
        planned_total_hours = round(planned_sailing + planned_port + planned_customs + planned_final, 2)
        scheduled_arrival = scheduled_departure + pd.to_timedelta(planned_total_hours, unit="h")

        # ---------------------------------------------------------
        # 2. Environmental & Operational Signals (0.0 to 1.0)
        # ---------------------------------------------------------
        congestion = round(float(rng.beta(2.2, 2.2)), 3)
        weather = round(float(rng.beta(1.8, 3.2)), 3)

        # ---------------------------------------------------------
        # 3. Incremental Stage Delays (hours, rounded to 2 decimals)
        # Each stage delay represents incremental friction added at that stage.
        # ---------------------------------------------------------
        departure_delay = round(max(0.0, float(rng.normal(2.0 + 8.5 * weather, 4.5))), 2)
        sailing_delay = round(max(0.0, float(rng.normal(3.0 + 13.5 * weather + 0.15 * departure_delay, 6.0))), 2)
        port_delay = round(max(0.0, float(rng.normal(2.0 + 26.0 * congestion + 0.22 * sailing_delay, 7.5))), 2)
        customs_delay = round(max(0.0, float(rng.normal(1.5 + 13.0 * congestion + 0.18 * port_delay, 5.0))), 2)
        final_delay = round(max(0.0, float(rng.normal(1.0 + 0.12 * customs_delay + 3.0 * weather, 4.0))), 2)

        # Actual stage durations (hours)
        actual_sailing_hours = round(planned_sailing + sailing_delay, 2)
        actual_port_hours = round(planned_port + port_delay, 2)
        actual_customs_hours = round(planned_customs + customs_delay, 2)
        actual_final_hours = round(planned_final + final_delay, 2)

        # ---------------------------------------------------------
        # 4. Cumulative Schedule Deviations (Inherited Delays)
        # Cumulative delay at milestone = sum of incremental delays up to that point.
        # No double counting: cumulative delay reflects calendar slippage.
        # ---------------------------------------------------------
        cum_delay_departure = round(departure_delay, 2)
        cum_delay_port_arrival = round(departure_delay + sailing_delay, 2)  # Checkpoint: Observed Upstream Delay
        cum_delay_port_departure = round(cum_delay_port_arrival + port_delay, 2)
        cum_delay_customs_done = round(cum_delay_port_departure + customs_delay, 2)
        total_delay = round(cum_delay_customs_done + final_delay, 2)       # Ground Truth Total Delay

        remaining_delay_at_checkpoint = round(port_delay + customs_delay + final_delay, 2)

        # ---------------------------------------------------------
        # 5. Planned vs Actual Timestamps (Chronological Event Sequencing)
        # ---------------------------------------------------------
        # Planned timeline
        planned_port_arrival = scheduled_departure + pd.to_timedelta(planned_sailing, unit="h")
        planned_port_departure = planned_port_arrival + pd.to_timedelta(planned_port, unit="h")
        planned_customs_done = planned_port_departure + pd.to_timedelta(planned_customs, unit="h")

        # Actual timeline (using exact rounded incremental additions)
        actual_departure = scheduled_departure + pd.to_timedelta(departure_delay, unit="h")
        actual_port_arrival = planned_port_arrival + pd.to_timedelta(cum_delay_port_arrival, unit="h")
        actual_port_departure = planned_port_departure + pd.to_timedelta(cum_delay_port_departure, unit="h")
        actual_customs_done = planned_customs_done + pd.to_timedelta(cum_delay_customs_done, unit="h")
        actual_delivery = scheduled_arrival + pd.to_timedelta(total_delay, unit="h")

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
            
            # Key milestones & environmental indicators
            "scheduled_departure": scheduled_departure,
            "scheduled_arrival": scheduled_arrival,
            "port_congestion": congestion,
            "weather_severity": weather,
            
            # Planned stage durations (hours)
            "planned_sailing_hours": planned_sailing,
            "planned_port_hours": planned_port,
            "planned_customs_hours": planned_customs,
            "planned_final_hours": planned_final,
            "planned_remaining_hours_at_port": round(planned_port + planned_customs + planned_final, 2),

            # Actual stage durations (hours)
            "actual_sailing_hours": actual_sailing_hours,
            "actual_port_hours": actual_port_hours,
            "actual_customs_hours": actual_customs_hours,
            "actual_final_hours": actual_final_hours,

            # Incremental stage delays (hours)
            "departure_delay_hours": departure_delay,
            "sailing_delay_hours": sailing_delay,
            "port_delay_hours": port_delay,
            "customs_delay_hours": customs_delay,
            "final_stage_delay_hours": final_delay,

            # Cumulative schedule deviations at milestones (hours)
            "cum_delay_departure_hours": cum_delay_departure,
            "observed_delay_at_port_hours": cum_delay_port_arrival,  # Checkpoint milestone
            "cum_delay_port_departure_hours": cum_delay_port_departure,
            "cum_delay_customs_done_hours": cum_delay_customs_done,
            "actual_total_delay_hours": total_delay,                  # Target variable
            "actual_remaining_delay_hours": remaining_delay_at_checkpoint,

            # Planned event timestamps
            "planned_port_arrival": planned_port_arrival,
            "planned_port_departure": planned_port_departure,
            "planned_customs_done": planned_customs_done,

            # Actual event timestamps
            "actual_departure": actual_departure,
            "actual_port_arrival": actual_port_arrival,
            "actual_port_departure": actual_port_departure,
            "actual_customs_done": actual_customs_done,
            "actual_delivery": actual_delivery,

            # Binary classification target (late if total incremental delay > 24 hours)
            "late_delivery": int(total_delay > 24.0),
        })

    df = pd.DataFrame(rows)
    validate_shipments_dataset(df)
    return df

def validate_shipments_dataset(df: pd.DataFrame) -> None:
    """Rigorous data validation checks for shipment dataset consistency.
    
    Verifies:
      - Non-negative durations and delays
      - Exact delay aggregation (no double-counting)
      - Strict chronological event ordering
      - Non-null values across required schema
    """
    if len(df) == 0:
        raise ValueError("Dataset is empty")

    # 1. Non-negativity checks
    delay_cols = [
        "departure_delay_hours", "sailing_delay_hours", "port_delay_hours",
        "customs_delay_hours", "final_stage_delay_hours", "actual_total_delay_hours"
    ]
    for col in delay_cols:
        if (df[col] < 0).any():
            bad = df[df[col] < 0]
            raise ValueError(f"Found negative delays in {col}: {bad[col].tolist()[:3]}")

    duration_cols = ["planned_sailing_hours", "planned_port_hours", "planned_customs_hours", "planned_final_hours"]
    for col in duration_cols:
        if (df[col] <= 0).any():
            bad = df[df[col] <= 0]
            raise ValueError(f"Found non-positive planned duration in {col}: {bad[col].tolist()[:3]}")

    # 2. Mathematical delay aggregation check (Sum of incremental stage delays == Total delay)
    stage_sum = (
        df["departure_delay_hours"] +
        df["sailing_delay_hours"] +
        df["port_delay_hours"] +
        df["customs_delay_hours"] +
        df["final_stage_delay_hours"]
    ).round(2)
    mismatch = (stage_sum - df["actual_total_delay_hours"].round(2)).abs() > 0.01
    if mismatch.any():
        raise ValueError(f"Stage delay sum does not match actual_total_delay_hours for {mismatch.sum()} rows")

    # 3. Observed upstream delay at port checkpoint check
    upstream_sum = (df["departure_delay_hours"] + df["sailing_delay_hours"]).round(2)
    mismatch_up = (upstream_sum - df["observed_delay_at_port_hours"].round(2)).abs() > 0.01
    if mismatch_up.any():
        raise ValueError(f"Upstream delay sum does not match observed_delay_at_port_hours for {mismatch_up.sum()} rows")

    # 4. Chronological ordering check
    sched_dep = pd.to_datetime(df["scheduled_departure"], utc=True)
    sched_arr = pd.to_datetime(df["scheduled_arrival"], utc=True)
    act_dep = pd.to_datetime(df["actual_departure"], utc=True)
    act_arr = pd.to_datetime(df["actual_port_arrival"], utc=True)
    act_deliv = pd.to_datetime(df["actual_delivery"], utc=True)

    if (sched_arr <= sched_dep).any():
        raise ValueError("Scheduled arrival is not after scheduled departure")
    if (act_arr < act_dep).any():
        raise ValueError("Actual port arrival is earlier than actual departure")
    if (act_deliv < act_arr).any():
        raise ValueError("Actual delivery is earlier than actual port arrival")

    # 5. Label consistency check
    expected_late = (df["actual_total_delay_hours"] > 24.0).astype(int)
    if (df["late_delivery"] != expected_late).any():
        raise ValueError("late_delivery binary label inconsistent with actual_total_delay_hours > 24.0")

def save_dataset(path: str = "data/shipments.csv", n: int = 300, seed: int = 42) -> pd.DataFrame:
    df = generate_shipments(n=n, seed=seed)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df

if __name__ == "__main__":
    df = save_dataset()
    print(f"Generated {len(df)} validated synthetic shipments at data/shipments.csv")
