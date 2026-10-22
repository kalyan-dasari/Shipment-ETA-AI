"""Streamlit dashboard for Shipment ETA Intelligence and Delay Propagation."""
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import sys
import streamlit as st

# Automatically launch via Streamlit runner if invoked directly as `python app.py`
if not st.runtime.exists():
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

import altair as alt
from src.data_generator import generate_shipments, save_dataset
from src.train import train
from src.predict import load_bundle, predict_dataframe, predict_single, get_risk_category
from src.explain import explain_shipment, get_stage_waterfall, compute_tree_attributions
from src.features import build_features

st.set_page_config(
    page_title="Shipment ETA Intelligence | Delay Propagation AI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 4px 0;
        color: #f8fafc;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(249, 115, 22, 0.2);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-moderate {
        background-color: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .checkpoint-banner {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }
    .operator-action-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 14px;
    }
    .stage-step {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stage-done {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .stage-current {
        background: rgba(234, 179, 8, 0.25);
        color: #fde047;
        border: 1px solid rgba(234, 179, 8, 0.6);
    }
    .stage-future {
        background: rgba(148, 163, 184, 0.1);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path("data/shipments.csv")
MODEL_PATH = Path("artifacts/eta_model.joblib")

@st.cache_data
def get_dataset():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(str(DATA_PATH), n=300, seed=42)
    df = pd.read_csv(DATA_PATH)
    for col in ["scheduled_arrival", "scheduled_departure", "actual_port_arrival", "actual_departure"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df

@st.cache_resource
def get_bundle():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        train(str(DATA_PATH), str(MODEL_PATH))
    return load_bundle(str(MODEL_PATH))

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("## 🚢 System Controls")
    st.caption("AI-Powered Shipment ETA & Delay Propagation Engine")

    # Threshold slider
    delay_threshold = st.slider(
        "Late Arrival Threshold (hours)",
        min_value=12,
        max_value=72,
        value=24,
        step=6,
        help="Shipments with total delay exceeding this threshold are flagged as late."
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Maintenance")
    if st.button("🔄 Retrain ML Models", type="primary", width="stretch"):
        with st.spinner("Retraining Random Forest Ensemble and Probability Calibrator..."):
            train(str(DATA_PATH), str(MODEL_PATH), late_threshold_hours=delay_threshold)
            st.cache_resource.clear()
            st.cache_data.clear()
        st.success("Model successfully retrained and cached!")
        st.rerun()

    if st.button("🎲 Regenerate Dataset", width="stretch"):
        with st.spinner("Generating 300 validated synthetic shipments with linked stage delays..."):
            save_dataset(str(DATA_PATH), n=300, seed=int(np.random.randint(1, 9999)))
            train(str(DATA_PATH), str(MODEL_PATH), late_threshold_hours=delay_threshold)
            st.cache_resource.clear()
            st.cache_data.clear()
        st.success("Fresh synthetic journey dataset generated!")
        st.rerun()

    st.markdown("---")
    try:
        b_info = get_bundle()
        m = b_info["metrics"]
        st.markdown(f"**Prediction Checkpoint:** Port Arrival")
        st.markdown(f"**Model Type:** Random Forest Regressor")
        st.markdown(f"**Test MAE:** `{m['regression']['model_mae_hours']} h`")
        st.markdown(f"**Baseline MAE:** `{m['regression']['baseline_heuristic_mae_hours']} h`")
        st.markdown(f"**MAE Improvement:** `+{m['regression']['mae_improvement_percent_vs_heuristic']}%`")
        st.markdown(f"**Delay Risk ROC-AUC:** `{m['classification']['roc_auc']}`")
        st.markdown(f"**Empirical CI Coverage:** `{m['prediction_intervals']['empirical_test_coverage']*100:.1f}%`")
    except Exception:
        pass

    st.caption("Aiqon Take-Home Assignment · Kalyan Dasari")

# Load Data and Predictions
try:
    df_raw = get_dataset()
    bundle = get_bundle()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# Batch prediction
predictions = predict_dataframe(df_raw, bundle)
joined = df_raw.merge(predictions, on="shipment_id", suffixes=("", "_pred"))
joined["scheduled_arrival"] = pd.to_datetime(joined["scheduled_arrival"], utc=True)
joined["predicted_eta"] = pd.to_datetime(joined["predicted_eta"], utc=True)
joined["predicted_eta_lower"] = pd.to_datetime(joined["predicted_eta_lower"], utc=True)
joined["predicted_eta_upper"] = pd.to_datetime(joined["predicted_eta_upper"], utc=True)

# ----------------- MAIN HEADER -----------------
st.title("🚢 Shipment ETA & Delay Propagation Intelligence")
st.markdown(
    "Predict final delivery ETA, estimate delay risk, and uncover stage-by-stage delay propagation "
    "at the **Port Arrival Checkpoint** using machine learning on synthetic journey data."
)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Fleet Intelligence",
    "🔍 Shipment Root-Cause Deep-Dive",
    "🧪 What-If Scenario Simulator",
    "📊 Model Performance & Benchmarks"
])

# =========================================================================
# TAB 1: FLEET INTELLIGENCE & MONITORING
# =========================================================================
with tab1:
    st.subheader("Global Fleet Overview & Delay Monitoring")

    # High-level KPIs
    total_shipments = len(joined)
    avg_pred_delay = float(joined["predicted_delay_hours"].mean())
    late_count = int((joined["predicted_delay_hours"] > delay_threshold).sum())
    on_time_rate = round(((total_shipments - late_count) / total_shipments) * 100, 1)
    critical_count = int((joined["late_risk"] == "Critical").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Tracked Shipments</div>
            <div class="metric-value">{total_shipments}</div>
            <div class="metric-delta" style="color: #60a5fa;">Checkpoint: Port Arrival</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Expected Delay</div>
            <div class="metric-value">{avg_pred_delay:.1f} hrs</div>
            <div class="metric-delta" style="color: #94a3b8;">Range: {joined['predicted_delay_hours'].min():.1f}h – {joined['predicted_delay_hours'].max():.1f}h</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">On-Time Arrival Rate</div>
            <div class="metric-value">{on_time_rate}%</div>
            <div class="metric-delta" style="color: {'#4ade80' if on_time_rate > 70 else '#facc15'};">Target threshold: ≤ {delay_threshold}h</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Critical Risk Shipments</div>
            <div class="metric-value">{critical_count}</div>
            <div class="metric-delta" style="color: #f87171;">P(Late > {delay_threshold}h) ≥ 75%</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Filters
    st.markdown("#### 🔍 Filter Fleet")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        origins = ["All"] + sorted(joined["origin"].unique().tolist())
        selected_origin = st.selectbox("Origin Port", origins)
    with f2:
        dests = ["All"] + sorted(joined["destination"].unique().tolist())
        selected_dest = st.selectbox("Destination Port", dests)
    with f3:
        risks = ["All", "Critical", "High", "Moderate", "Low"]
        selected_risk = st.selectbox("Late Risk Category", risks)
    with f4:
        search_query = st.text_input("Search Shipment / Vessel", placeholder="e.g. SHP-0012 or Pacific")

    filtered = joined.copy()
    if selected_origin != "All":
        filtered = filtered[filtered["origin"] == selected_origin]
    if selected_dest != "All":
        filtered = filtered[filtered["destination"] == selected_dest]
    if selected_risk != "All":
        filtered = filtered[filtered["late_risk"] == selected_risk]
    if search_query:
        filtered = filtered[
            filtered["shipment_id"].str.contains(search_query, case=False, na=False) |
            filtered["vessel_name"].str.contains(search_query, case=False, na=False) |
            filtered["carrier"].str.contains(search_query, case=False, na=False)
        ]

    # Visual Fleet Analytics Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("**Predicted Delay Distribution Across Fleet**")
        hist_chart = alt.Chart(filtered).mark_bar(opacity=0.8, color="#3b82f6").encode(
            x=alt.X("predicted_delay_hours:Q", bin=alt.Bin(maxbins=25), title="Predicted Delay (Hours)"),
            y=alt.Y("count()", title="Shipment Count"),
            tooltip=[
                alt.Tooltip("count()", title="Count"),
                alt.Tooltip("min(predicted_delay_hours):Q", title="Min Delay (h)", format=".1f"),
                alt.Tooltip("max(predicted_delay_hours):Q", title="Max Delay (h)", format=".1f"),
            ]
        ).properties(height=260)
        rule = alt.Chart(pd.DataFrame({'threshold': [delay_threshold]})).mark_rule(
            color='#ef4444', strokeWidth=2, strokeDash=[4, 4]
        ).encode(x='threshold:Q')
        st.altair_chart(hist_chart + rule, width="stretch")

    with col_chart2:
        st.markdown("**Average Delay by Primary Trade Lane**")
        route_agg = filtered.groupby("route")["predicted_delay_hours"].mean().reset_index()
        route_agg = route_agg.sort_values(by="predicted_delay_hours", ascending=False).head(8)
        route_chart = alt.Chart(route_agg).mark_bar(color="#f97316").encode(
            x=alt.X("predicted_delay_hours:Q", title="Mean Delay (Hours)"),
            y=alt.Y("route:N", sort="-x", title="Trade Lane"),
            tooltip=[
                alt.Tooltip("route:N", title="Route"),
                alt.Tooltip("predicted_delay_hours:Q", title="Avg Delay (h)", format=".1f"),
            ]
        ).properties(height=260)
        st.altair_chart(route_chart, width="stretch")

    # Fleet Data Table
    st.markdown(f"**Matching Shipments ({len(filtered)} records)**")
    display_cols = [
        "shipment_id", "vessel_name", "carrier", "route", "port",
        "scheduled_arrival", "predicted_eta", "predicted_delay_hours",
        "late_probability", "late_risk"
    ]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "shipment_id": "Shipment ID",
            "vessel_name": "Vessel",
            "carrier": "Carrier",
            "route": "Route",
            "port": "Port Checkpoint",
            "scheduled_arrival": "Scheduled Arrival",
            "predicted_eta": "Predicted ETA",
            "predicted_delay_hours": "Expected Delay (h)",
            "late_probability": "Late Prob",
            "late_risk": "Risk Status",
        }),
        width="stretch",
        hide_index=True
    )

# =========================================================================
# TAB 2: SHIPMENT ROOT-CAUSE DEEP-DIVE & OPERATOR DECISION SUPPORT
# =========================================================================
with tab2:
    st.subheader("Individual Journey Deep-Dive & Root-Cause Attribution")

    # Select shipment
    shipment_options = joined["shipment_id"].tolist()
    selected_shp = st.selectbox(
        "Select Shipment to Inspect",
        shipment_options,
        index=0,
        format_func=lambda sid: f"{sid} — {joined.loc[joined['shipment_id'] == sid, 'route'].iloc[0]} ({joined.loc[joined['shipment_id'] == sid, 'carrier'].iloc[0]})"
    )

    row = joined.loc[joined["shipment_id"] == selected_shp].iloc[0]
    single_res = predict_single(row, bundle)

    # Checkpoint Context Banner
    checkpoint_time = pd.to_datetime(row["actual_port_arrival"]).strftime("%d %b %Y, %H:%M UTC")
    st.markdown(f"""
    <div class="checkpoint-banner">
        📍 <strong>Prediction Checkpoint:</strong> Destination Port Outer Anchorage / Pilot Station Arrival | 
        <strong>Checkpoint Event Time:</strong> <code>{checkpoint_time}</code><br>
        <span style="color:#94a3b8; font-size:0.85rem;">
        Voyage Status: In-Progress. Upstream voyage history (Departure & Sailing) is known. Downstream stages (Port dwell, Customs, Inland delivery) are unknown future outcomes.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Visual Stage Stepper
    st.markdown(f"""
    <div style="margin: 8px 0 16px 0;">
        <span class="stage-step stage-done">✓ 1. Origin Departure ({row['origin']})</span>
        <span style="color:#64748b;">➔</span>
        <span class="stage-step stage-done">✓ 2. Ocean Transit ({row['sailing_delay_hours']:.1f}h leg delay)</span>
        <span style="color:#64748b;">➔</span>
        <span class="stage-step stage-current">📍 3. Port Arrival Checkpoint ({row['port']})</span>
        <span style="color:#64748b;">➔</span>
        <span class="stage-step stage-future">⏱ 4. Customs Clearance (Pending)</span>
        <span style="color:#64748b;">➔</span>
        <span class="stage-step stage-future">🏁 5. Final Delivery ({row['destination']})</span>
    </div>
    """, unsafe_allow_html=True)

    # 1. OBSERVED CHECKPOINT FACTS vs 2. MODEL PREDICTIONS
    st.markdown("#### 1. Observed Checkpoint Facts (Known Prior to Port Operations)")
    col_obs1, col_obs2, col_obs3, col_obs4 = st.columns(4)
    sched_eta = pd.to_datetime(row["scheduled_arrival"])
    sched_dep = pd.to_datetime(row["scheduled_departure"])
    obs_delay = float(row["observed_delay_at_port_hours"])
    plan_rem = float(row["planned_remaining_hours_at_port"])

    with col_obs1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Scheduled Arrival (Original ETA)</div>
            <div class="metric-value" style="font-size:1.45rem;">{sched_eta.strftime('%d %b, %H:%M')}</div>
            <div class="metric-delta" style="color:#94a3b8;">Departure: {sched_dep.strftime('%d %b, %H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_obs2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Observed Delay So Far</div>
            <div class="metric-value" style="font-size:1.45rem;">+{obs_delay:.1f} hrs</div>
            <div class="metric-delta" style="color:#60a5fa;">Departure: {row['departure_delay_hours']:.1f}h | Sailing: {row['sailing_delay_hours']:.1f}h</div>
        </div>
        """, unsafe_allow_html=True)
    with col_obs3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Planned Remaining Duration</div>
            <div class="metric-value" style="font-size:1.45rem;">{plan_rem:.1f} hrs</div>
            <div class="metric-delta" style="color:#94a3b8;">Port: {row['planned_port_hours']:.0f}h | Cust: {row['planned_customs_hours']:.0f}h | Fin: {row['planned_final_hours']:.0f}h</div>
        </div>
        """, unsafe_allow_html=True)
    with col_obs4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Terminal & Route Signals</div>
            <div class="metric-value" style="font-size:1.45rem;">{row['port_congestion']:.2f} / 1.0</div>
            <div class="metric-delta" style="color:#cbd5e1;">Congestion: {row['port_congestion']:.2f} | Weather: {row['weather_severity']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 2. AI Model In-Flight Predictions (Estimated Future Outcomes)")
    col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
    pred_eta = pd.to_datetime(single_res["predicted_eta"])
    eta_lower = pd.to_datetime(single_res["predicted_eta_lower"])
    eta_upper = pd.to_datetime(single_res["predicted_eta_upper"])
    risk_cat = single_res["late_risk"]
    badge_class = f"badge-{risk_cat.lower()}"

    with col_pred1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Final ETA</div>
            <div class="metric-value" style="font-size:1.45rem; color:#38bdf8;">{pred_eta.strftime('%d %b, %H:%M')}</div>
            <div class="metric-delta" style="color:#94a3b8;">Band: [{eta_lower.strftime('%d %b %H:%M')} – {eta_upper.strftime('%d %b %H:%M')}]</div>
        </div>
        """, unsafe_allow_html=True)
    with col_pred2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Total Delay</div>
            <div class="metric-value" style="font-size:1.45rem;">+{single_res['predicted_delay_hours']:.1f} hrs</div>
            <div class="metric-delta" style="color:#94a3b8;">Downstream Remaining: +{single_res['predicted_remaining_delay_hours']:.1f}h</div>
        </div>
        """, unsafe_allow_html=True)
    with col_pred3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Probability of Delay > 24h</div>
            <div class="metric-value" style="font-size:1.45rem;">{single_res['late_probability'] * 100:.1f}%</div>
            <div class="metric-delta" style="color:#94a3b8;">Calibrated classification score</div>
        </div>
        """, unsafe_allow_html=True)
    with col_pred4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Operational Delay Risk Tier</div>
            <div class="metric-value" style="font-size:1.45rem;"><span class="{badge_class}">{risk_cat.upper()}</span></div>
            <div class="metric-delta" style="color:#94a3b8;">Thresholds: &lt;25% Low, 25-50% Mod, 50-75% High, &ge;75% Crit</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. OPERATOR ACTION PLAYBOOK (Practical Decision Value)
    st.markdown("#### 3. 🎯 Logistics Operator Action Playbook (Decision Support)")
    if risk_cat in ["Critical", "High"]:
        st.markdown(f"""
        <div class="operator-action-card" style="border-left: 4px solid #f87171;">
            <strong style="color:#f87171;">⚠️ High Operational Delay Risk Detected ({single_res['late_probability']*100:.1f}% probability exceeding 24h delay). Recommended Interventions:</strong>
            <ul style="margin: 8px 0 0 0; font-size: 0.9rem; color: #e2e8f0;">
                <li><strong>Proactive Consignee Notification:</strong> Alert warehouse receiving and retail consignees of expected revised delivery window (<code>{pred_eta.strftime('%d %b %Y, %H:%M UTC')}</code>) to avoid idle labour costs.</li>
                <li><strong>Terminal Berthing Intervention:</strong> Port congestion is elevated ({row['port_congestion']:.2f}). Contact local port agent to request priority crane allocation or off-dock container drayage to avoid yard dwell.</li>
                <li><strong>Pre-lodge Customs Manifests:</strong> Ensure electronic customs documents (EDI) are pre-cleared before container discharge to eliminate regulatory dwell.</li>
                <li><strong>Inland Transport Rescheduling:</strong> Reschedule rail and truck appointments to prevent costly port demurrage and container detention penalties.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif risk_cat == "Moderate":
        st.markdown(f"""
        <div class="operator-action-card" style="border-left: 4px solid #facc15;">
            <strong style="color:#facc15;">⚡ Moderate Delay Risk ({single_res['late_probability']*100:.1f}%). Recommended Monitoring:</strong>
            <ul style="margin: 8px 0 0 0; font-size: 0.9rem; color: #e2e8f0;">
                <li><strong>Terminal Queue Tracking:</strong> Monitor destination port pilot board for anchorage waiting times.</li>
                <li><strong>Carrier Verification:</strong> Confirm scheduled crane discharge window with {row['carrier']}.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="operator-action-card" style="border-left: 4px solid #4ade80;">
            <strong style="color:#4ade80;">✅ Low Delay Risk ({single_res['late_probability']*100:.1f}%). Standard Transit:</strong>
            <ul style="margin: 8px 0 0 0; font-size: 0.9rem; color: #e2e8f0;">
                <li>Shipment is progressing within acceptable tolerances. Maintain standard automated tracking updates.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 4. STAGE WATERFALL & TREE SHAP ATTRIBUTIONS
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("#### 4. 🌊 Stage-by-Stage Delay Propagation Waterfall")
        waterfall_df = pd.DataFrame(single_res["waterfall"])

        wf_chart = alt.Chart(waterfall_df).mark_bar(cornerRadius=4).encode(
            x=alt.X("Stage:N", sort=None, title="Journey Stage"),
            y=alt.Y("Incremental Delay (Hours):Q", title="Incremental Stage Delay (Hours)"),
            color=alt.Color("Status:N", scale=alt.Scale(
                domain=["Observed Upstream", "Estimated Downstream", "Observed / Actual"],
                range=["#3b82f6", "#f97316", "#10b981"]
            )),
            tooltip=[
                alt.Tooltip("Stage:N", title="Stage"),
                alt.Tooltip("Status:N", title="Checkpoint Visibility"),
                alt.Tooltip("Incremental Delay (Hours):Q", title="Incremental Delay (h)", format=".1f"),
                alt.Tooltip("Cumulative Delay at Milestone:Q", title="Cumulative Delay (h)", format=".1f"),
                alt.Tooltip("Propagation Driver:N", title="Primary Driver"),
            ]
        ).properties(height=300)

        line_chart = alt.Chart(waterfall_df).mark_line(point=True, color="#fde047", strokeWidth=2).encode(
            x=alt.X("Stage:N", sort=None),
            y=alt.Y("Cumulative Delay at Milestone:Q"),
            tooltip=[alt.Tooltip("Cumulative Delay at Milestone:Q", title="Cumulative Delay at Milestone (h)", format=".1f")]
        )
        st.altair_chart(wf_chart + line_chart, width="stretch")
        st.caption("Bar height: Incremental delay added at that stage. Yellow line: Cumulative schedule deviation (inherited delay).")

    with col_right:
        st.markdown("#### 5. 🌲 Tree SHAP Feature Attribution (Hours Contributed)")
        attributions = single_res["tree_attributions"]["attributions"]
        attr_df = pd.DataFrame(attributions).head(7)

        attr_chart = alt.Chart(attr_df).mark_bar(cornerRadius=4).encode(
            x=alt.X("impact_hours:Q", title="Impact on ETA Delay (Hours)"),
            y=alt.Y("label:N", sort=alt.EncodingSortField(field="impact_hours", op="sum", order="descending"), title=""),
            color=alt.condition(
                alt.datum.impact_hours > 0,
                alt.value("#ef4444"),  # Pushes ETA later
                alt.value("#22c55e")   # Buffers / reduces delay
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Factor"),
                alt.Tooltip("value:Q", title="Observed Value", format=".2f"),
                alt.Tooltip("impact_hours:Q", title="Impact on Prediction (h)", format="+.2f"),
                alt.Tooltip("direction:N", title="Effect")
            ]
        ).properties(height=300)
        st.altair_chart(attr_chart, width="stretch")
        st.caption("Red bars: Factors pushing arrival later. Green bars: Buffers/factors reducing delay relative to fleet average.")

    # Narrative Explanations
    st.markdown("#### 💡 AI Diagnostic Insights & Operational Explanations")
    for msg in single_res["narrative_explanations"]:
        st.markdown(f"- {msg}")

    # Ground-truth evaluation expander (Strictly separated for historical auditing)
    with st.expander("🔍 Historical Post-Delivery Ground Truth Audit (Visible only for completed journeys)"):
        st.caption("This data represents ground-truth actuals recorded after delivery. It is strictly excluded at prediction time.")
        c_gt1, c_gt2, c_gt3 = st.columns(3)
        act_deliv_ts = pd.to_datetime(row["actual_delivery"]).strftime("%d %b %Y, %H:%M UTC")
        act_total_del = float(row["actual_total_delay_hours"])
        pred_del = float(single_res["predicted_delay_hours"])
        err = round(abs(pred_del - act_total_del), 2)

        with c_gt1:
            st.metric("Ground-Truth Actual Delivery", act_deliv_ts)
        with c_gt2:
            st.metric("Ground-Truth Actual Delay", f"+{act_total_del:.1f} hrs")
        with c_gt3:
            st.metric("Model Absolute Error", f"{err:.1f} hrs", delta=f"{'Overestimated' if pred_del > act_total_del else 'Underestimated'}")

# =========================================================================
# TAB 3: WHAT-IF SCENARIO SIMULATOR & CUSTOM JOURNEY BUILDER
# =========================================================================
with tab3:
    st.subheader("🧪 What-If Scenario Simulator & Journey Builder")
    st.markdown(
        "Simulate operational interventions (e.g. priority berthing, schedule adjustments, or weather routing) "
        "or evaluate a completely custom shipment journey."
    )

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        st.markdown("##### 🎛️ Scenario Parameters")
        base_choice = st.selectbox(
            "Base Journey Template",
            ["Preset: High Congestion Transpacific", "Preset: Severe Weather Atlantic", "Preset: On-Time Baseline", "Custom Manual Entry"]
        )

        if base_choice == "Preset: High Congestion Transpacific":
            def_dep_delay = 4.0
            def_sail_delay = 8.0
            def_cong = 0.85
            def_weath = 0.40
            def_port_h = 24.0
        elif base_choice == "Preset: Severe Weather Atlantic":
            def_dep_delay = 12.0
            def_sail_delay = 20.0
            def_cong = 0.35
            def_weath = 0.90
            def_port_h = 20.0
        elif base_choice == "Preset: On-Time Baseline":
            def_dep_delay = 0.0
            def_sail_delay = 1.0
            def_cong = 0.20
            def_weath = 0.15
            def_port_h = 16.0
        else:
            def_dep_delay = 5.0
            def_sail_delay = 10.0
            def_cong = 0.50
            def_weath = 0.50
            def_port_h = 24.0

        sim_dep_delay = st.slider("Observed Departure Delay (hours)", 0.0, 48.0, float(def_dep_delay), 0.5)
        sim_sail_delay = st.slider("Observed Sailing Delay (hours)", 0.0, 72.0, float(def_sail_delay), 0.5)
        sim_cong = st.slider("Port Congestion Index (0.0 – 1.0)", 0.0, 1.0, float(def_cong), 0.05,
                             help="Higher values simulate terminal queueing and berth wait times.")
        sim_weath = st.slider("Weather Severity Index (0.0 – 1.0)", 0.0, 1.0, float(def_weath), 0.05)

        with st.expander("Planned Schedule Durations"):
            sim_plan_sail = st.number_input("Planned Sailing (hours)", 24.0, 500.0, 180.0, 12.0)
            sim_plan_port = st.number_input("Planned Port Dwell (hours)", 4.0, 72.0, float(def_port_h), 2.0)
            sim_plan_customs = st.number_input("Planned Customs (hours)", 2.0, 48.0, 12.0, 2.0)
            sim_plan_final = st.number_input("Planned Final Transit (hours)", 4.0, 72.0, 24.0, 4.0)

        sim_threshold = st.slider("Risk Evaluation Threshold (hours)", 12, 48, 24, 6)

    # Build input record
    now_utc = datetime.now(timezone.utc)
    sim_sched_arrival = now_utc + pd.Timedelta(hours=sim_plan_sail + sim_plan_port + sim_plan_customs + sim_plan_final)

    sim_record = {
        "shipment_id": "SIM-001",
        "carrier": "Simulated Carrier",
        "vessel_name": "Scenario Vessel",
        "origin": "Simulated Origin",
        "destination": "Simulated Destination",
        "route": "Simulated-Route",
        "port": "Simulated Terminal",
        "scheduled_departure": now_utc,
        "scheduled_arrival": sim_sched_arrival,
        "planned_sailing_hours": sim_plan_sail,
        "planned_port_hours": sim_plan_port,
        "planned_customs_hours": sim_plan_customs,
        "planned_final_hours": sim_plan_final,
        "planned_remaining_hours_at_port": sim_plan_port + sim_plan_customs + sim_plan_final,
        "port_congestion": sim_cong,
        "weather_severity": sim_weath,
        "departure_delay_hours": sim_dep_delay,
        "sailing_delay_hours": sim_sail_delay,
        "observed_delay_at_port_hours": sim_dep_delay + sim_sail_delay,
    }

    sim_res = predict_single(sim_record, bundle)

    with sim_col2:
        st.markdown("##### 🔮 Real-Time Simulation Results")

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.metric("Predicted Total Delay", f"{sim_res['predicted_delay_hours']:.1f} hrs",
                      delta=f"{sim_res['predicted_remaining_delay_hours']:.1f}h remaining at port")
            st.metric("Uncertainty Band",
                      f"[{sim_res['predicted_delay_lower_hours']:.1f}h — {sim_res['predicted_delay_upper_hours']:.1f}h]")
        with res_c2:
            sim_late_p = sim_res["late_probability"]
            st.metric(f"Probability Exceeding {sim_threshold}h", f"{sim_late_p * 100:.1f}%")
            st.markdown(f"**Risk Level:** <span class='badge-{sim_res['late_risk'].lower()}'>{sim_res['late_risk'].upper()}</span>", unsafe_allow_html=True)

        st.markdown(f"**Estimated Arrival:** `{pd.to_datetime(sim_res['predicted_eta']).strftime('%d %b %Y, %H:%M UTC')}`")

        # Congestion Sensitivity Curve
        st.markdown("##### 📈 Terminal Congestion Sensitivity Curve")
        st.caption("How predicted delay shifts as terminal congestion ranges from 0.0 (empty) to 1.0 (gridlock):")

        sensitivity_points = []
        for c_val in np.linspace(0.0, 1.0, 11):
            temp_rec = sim_record.copy()
            temp_rec["port_congestion"] = float(c_val)
            temp_pred = predict_single(temp_rec, bundle)
            sensitivity_points.append({
                "Congestion": round(c_val, 2),
                "Predicted Delay (Hours)": temp_pred["predicted_delay_hours"],
                "Is Current": "Current Level" if abs(c_val - sim_cong) < 0.05 else "Hypothetical"
            })
        sens_df = pd.DataFrame(sensitivity_points)

        sens_chart = alt.Chart(sens_df).mark_line(point=True, color="#60a5fa", strokeWidth=3).encode(
            x=alt.X("Congestion:Q", title="Port Congestion Index"),
            y=alt.Y("Predicted Delay (Hours):Q", title="Predicted Delay (Hours)"),
            tooltip=[
                alt.Tooltip("Congestion:Q", title="Congestion", format=".2f"),
                alt.Tooltip("Predicted Delay (Hours):Q", title="Expected Delay (h)", format=".1f")
            ]
        ).properties(height=230)
        st.altair_chart(sens_chart, width="stretch")

# =========================================================================
# TAB 4: MODEL PERFORMANCE & METHODOLOGY
# =========================================================================
with tab4:
    st.subheader("Model Performance, Benchmarking & Technical Methodology")

    metrics = bundle["metrics"]
    reg_m = metrics["regression"]
    cls_m = metrics["classification"]
    pi_m = metrics["prediction_intervals"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Model MAE", f"{reg_m['model_mae_hours']} h",
                  delta=f"-{reg_m['mae_improvement_percent_vs_heuristic']}% vs Baseline", delta_color="inverse")
    with m2:
        st.metric("Operational Baseline MAE", f"{reg_m['baseline_heuristic_mae_hours']} h",
                  help="Observed upstream delay + median training downstream residual.")
    with m3:
        st.metric("Model RMSE", f"{reg_m['model_rmse_hours']} h",
                  delta=f"Baseline: {reg_m['baseline_heuristic_rmse_hours']} h", delta_color="inverse")
    with m4:
        st.metric("Delay Risk ROC-AUC", f"{cls_m['roc_auc']}",
                  help="Evaluates accuracy in distinguishing high-delay vs on-time shipments.")

    col_eval_left, col_eval_right = st.columns(2)

    with col_eval_left:
        st.markdown("#### 🎯 Parity Plot (Actual vs Predicted Total Delay)")
        eval_sample = bundle.get("test_evaluation_sample", {})
        if eval_sample and "y_test" in eval_sample:
            eval_df = pd.DataFrame({
                "Actual Total Delay (h)": eval_sample["y_test"],
                "Predicted Delay (h)": eval_sample["pred_test"],
                "Baseline Delay (h)": eval_sample["baseline_pred"],
            })
            scatter = alt.Chart(eval_df).mark_circle(size=60, color="#3b82f6", opacity=0.7).encode(
                x=alt.X("Actual Total Delay (h):Q", title="Ground Truth Delay (Hours)"),
                y=alt.Y("Predicted Delay (h):Q", title="Model Predicted Delay (Hours)"),
                tooltip=["Actual Total Delay (h):Q", "Predicted Delay (h):Q"]
            ).properties(height=280)

            # Identity line y = x
            max_val = max(eval_df["Actual Total Delay (h)"].max(), eval_df["Predicted Delay (h)"].max())
            line_ref = alt.Chart(pd.DataFrame({'x': [0, max_val], 'y': [0, max_val]})).mark_line(
                color='#ef4444', strokeDash=[4, 4]
            ).encode(x='x:Q', y='y:Q')

            st.altair_chart(scatter + line_ref, width="stretch")
            st.caption("Points closer to the red dashed line reflect accurate predictions.")
        else:
            st.info("Retrain the model to populate the held-out parity sample.")

    with col_eval_right:
        st.markdown("#### 🏆 Global Feature Importance Ranking")
        feat_imp_df = pd.DataFrame(list(metrics["feature_importances"].items()), columns=["Feature", "Importance"])
        from src.explain import FEATURE_FRIENDLY_NAMES
        feat_imp_df["Feature Label"] = feat_imp_df["Feature"].map(lambda x: FEATURE_FRIENDLY_NAMES.get(x, x))
        feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=True).tail(8)

        imp_chart = alt.Chart(feat_imp_df).mark_bar(color="#10b981", cornerRadius=4).encode(
            x=alt.X("Importance:Q", title="Gini Feature Importance"),
            y=alt.Y("Feature Label:N", sort=None, title=""),
            tooltip=["Feature Label:N", alt.Tooltip("Importance:Q", format=".3f")]
        ).properties(height=280)
        st.altair_chart(imp_chart, width="stretch")
        st.caption("Derived from MDI across all decision trees in the ensemble.")

    # Classification Confusion Matrix & Interval Coverage
    st.markdown("#### 📋 Classification Performance & Uncertainty Calibration")
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.markdown(f"**Classification Accuracy:** `{cls_m['accuracy']*100:.1f}%`")
        st.markdown(f"**Precision:** `{cls_m['precision']*100:.1f}%` | **Recall:** `{cls_m['recall']*100:.1f}%`")
        st.markdown(f"**F1-Score:** `{cls_m['f1_score']}`")
    with c_m2:
        cm = cls_m["confusion_matrix"]
        st.markdown("**Held-Out Confusion Matrix:**")
        st.markdown(f"- True Negatives (On-Time): `{cm[0][0]}`")
        st.markdown(f"- False Positives (False Alarms): `{cm[0][1]}`")
        st.markdown(f"- False Negatives (Missed Late): `{cm[1][0]}`")
        st.markdown(f"- True Positives (Caught Late): `{cm[1][1]}`")
    with c_m3:
        st.markdown(f"**Prediction Interval Calibration:**")
        st.markdown(f"- Method: `{pi_m.get('method', 'out_of_bag_residual_quantiles')}`")
        st.markdown(f"- Nominal Target: `{pi_m['nominal_confidence']*100:.0f}%`")
        st.markdown(f"- **Empirical Test Coverage:** `{pi_m['empirical_test_coverage']*100:.1f}%`")
        st.markdown(f"- Residual Spread: `[{pi_m['q10_residual_hours']:+.1f}h to {pi_m['q90_residual_hours']:+.1f}h]`")

    st.markdown("---")
    st.markdown("#### 📘 Simulation Assumptions, Limitations & Future Roadmap")
    st.markdown("""
    - **Synthetic Delay Propagation Mechanics:** The data models non-negative delay cascades across 5 discrete stages:
      `Departure → Ocean Sailing → Port Dwell & Crane Offloading → Customs Clearance → Final Delivery`.
    - **Checkpoint Definition:** Predictions occur strictly at the **Port Arrival Checkpoint**. All downstream actuals (customs clearance time, inland transit) are withheld during training to prevent target leakage.
    - **Baselines Compared:** Evaluated against both a naive assumption (zero remaining delay) and an operational heuristic (observed upstream delay + median historical residual).
    - **Production Translation Requirements:** In an operational deployment, this prototype would be extended with real-time AIS vessel tracking feeds, terminal gate API logs, weather satellite forecasts, and time-split backtesting.
    """)
