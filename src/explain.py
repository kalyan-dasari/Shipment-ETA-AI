"""Human-readable, assumption-aware explanations for an individual prediction."""
def explain_shipment(row, predicted_delay_hours: float) -> list[str]:
    explanations = []
    observed = float(row.get("observed_delay_at_port_hours", 0) or 0)
    congestion = float(row.get("port_congestion", 0) or 0)
    weather = float(row.get("weather_severity", 0) or 0)
    port_delay = float(row.get("port_delay_hours", 0) or 0) if "port_delay_hours" in row else None

    explanations.append(
        f"{observed:.1f} hours of upstream delay have already accumulated by the port-arrival checkpoint."
    )
    if congestion >= 0.65:
        explanations.append(
            f"Port congestion is high ({congestion:.2f} on the simulated 0–1 scale), which is associated with longer processing delays in this synthetic dataset."
        )
    elif congestion >= 0.4:
        explanations.append(
            f"Port congestion is moderate ({congestion:.2f}); it may contribute to processing-time variability."
        )
    if weather >= 0.65:
        explanations.append(
            f"Weather severity is high ({weather:.2f} on the simulated 0–1 scale), which is associated with upstream transit delays in the synthetic data."
        )
    explanations.append(
        f"The model predicts {predicted_delay_hours:.1f} hours total delay; this is a model estimate, not a causal attribution."
    )
    explanations.append(
        "The explanation uses observed features and the synthetic data-generation assumptions; it should not be interpreted as proof of real-world causation."
    )
    return explanations
