"""Streamlit dashboard for the synthetic shipment ETA demo."""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data_generator import generate_shipments
from src.train import train
from src.predict import load_bundle, predict_dataframe
from src.explain import explain_shipment

st.set_page_config(page_title="Shipment ETA Intelligence", page_icon="🚢", layout="wide")
st.title("🚢 Shipment ETA Intelligence")
st.caption("Synthetic-data prototype · Predictions are illustrative, not operational forecasts.")

DATA_PATH = Path("data/shipments.csv")
MODEL_PATH = Path("artifacts/eta_model.joblib")

if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_shipments(300).to_csv(DATA_PATH, index=False)

df = pd.read_csv(DATA_PATH, parse_dates=["scheduled_arrival", "scheduled_departure", "actual_port_arrival"])
with st.sidebar:
    st.header("Model controls")
    if st.button("Train / retrain model", type="primary"):
        with st.spinner("Training model..."):
            train(str(DATA_PATH), str(MODEL_PATH))
        st.success("Model trained.")
    st.caption("Regenerate data from the terminal with `python -m src.data_generator`.")

if not MODEL_PATH.exists():
    st.info("First run: click **Train / retrain model** in the sidebar.")
    st.stop()

bundle = load_bundle(str(MODEL_PATH))
predictions = predict_dataframe(df, bundle)
joined = df.merge(predictions, on="shipment_id", suffixes=("", "_prediction"))
joined["predicted_eta"] = pd.to_datetime(joined["predicted_eta"], utc=True)
joined["scheduled_arrival"] = pd.to_datetime(joined["scheduled_arrival"], utc=True)

selected_id = st.selectbox("Select a shipment", joined["shipment_id"].tolist())
row = joined.loc[joined["shipment_id"] == selected_id].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted ETA (UTC)", row["predicted_eta"].strftime("%d %b %Y, %H:%M"))
c2.metric("Predicted delay", f'{row["predicted_delay_hours"]:.1f} h')
c3.metric("Observed upstream delay", f'{row["observed_delay_at_port_hours_prediction"]:.1f} h')
c4.metric("Delay category", row["late_risk"])

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Stage timeline")
    stage_data = pd.DataFrame({
        "Stage": ["Departure", "Sailing", "Port", "Customs", "Final delivery"],
        "Delay (hours)": [
            row["departure_delay_hours"], row["sailing_delay_hours"],
            row["port_delay_hours"], row["customs_delay_hours"],
            row["final_stage_delay_hours"]
        ]
    })
    fig, ax = plt.subplots()
    ax.bar(stage_data["Stage"], stage_data["Delay (hours)"])
    ax.set_ylabel("Simulated delay (hours)")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig, clear_figure=True)
with right:
    st.subheader("Explanation")
    for explanation in explain_shipment(row, float(row["predicted_delay_hours"])):
        st.write("• " + explanation)

st.subheader("Model evaluation")
st.json(bundle["metrics"])
st.caption("The displayed shipment rows include synthetic ground-truth fields for demonstration. In a production system, actual future timestamps would not be available at prediction time.")

st.subheader("Shipment overview")
st.dataframe(
    joined[["shipment_id", "origin", "destination", "scheduled_arrival",
            "predicted_eta", "predicted_delay_hours", "late_risk"]],
    use_container_width=True, hide_index=True
)
