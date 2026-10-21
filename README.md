# 🚢 Shipment ETA Intelligence & Delay Propagation AI

An advanced machine learning system and interactive decision-support dashboard for **Predicting Shipment ETA and Delay Propagation** across multi-stage maritime supply chains.

---

## 🌟 System Overview

Maritime container journeys experience cascading schedule disruptions where early origin or sea transit delays snowball into severe port congestion, customs clearance bottlenecks, and missed delivery windows.

This system provides:
1. **Final Delivery ETA Prediction:** Estimates the expected arrival timestamp at the **Port Arrival Checkpoint** using scheduled timelines, observed upstream voyage history, and terminal conditions.
2. **Total & Remaining Delay Estimation:** Predicts final delivery delay in hours with **90% calibrated confidence intervals** (`[Lower ETA — Upper ETA]`).
3. **Calibrated Delay-Risk Scoring:** Computes the probability that a shipment exceeds an operational delay threshold (default: 24 hours), assigning actionable risk tiers:
   - 🟢 **Low Risk** (<25% probability)
   - 🟡 **Moderate Risk** (25% – 50%)
   - 🟠 **High Risk** (50% – 75%)
   - 🔴 **Critical Risk** (≥75%)
4. **Stage-by-Stage Delay Propagation:** Uncovers how delays compound across 5 discrete journey stages:
   $$\text{Departure} \longrightarrow \text{Ocean Sailing} \longrightarrow \text{Port Dwell / Offloading} \longrightarrow \text{Customs Clearance} \longrightarrow \text{Final Delivery}$$
5. **Explainable AI (Exact Tree SHAP Decomposition):** Quantifies exactly how many hours each operational and environmental factor (port congestion, weather severity, ocean transit, departure friction) added or subtracted from the baseline ETA.
6. **Interactive Streamlit Dashboard:** Features real-time fleet monitoring, individual shipment root-cause deep-dives, a What-If scenario simulator, and model evaluation benchmarks.

---

## 🚀 Quick Start

### 1. Requirements & Setup
Requires Python 3.10+ (tested on Python 3.14).

```bash
# Clone or navigate to the repository
cd Shipment-ETA-AI

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data & Train Models
```bash
# Generate 300 synthetic shipment journeys with linked stage delays
python -m src.data_generator

# Train Random Forest Regressor, delay-risk classifier, and baselines
python -m src.train
```

### 3. Launch Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Automated Unit Tests
```bash
python -m pytest -v
```

---

## 📊 Evaluation Results (Held-Out Test Set)

The system was evaluated using a 25% stratified held-out test split ($N_{\text{test}} = 75$ shipments, seed 42) against two operational baselines:

| Metric | Random Forest ML Model | Operational Heuristic Baseline | Naive Baseline (Zero Remaining Delay) |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **9.885 hours** | 11.534 hours | 29.169 hours |
| **Root Mean Squared Error (RMSE)** | **12.667 hours** | 14.179 hours | 31.841 hours |
| **Coefficient of Determination ($R^2$)** | **0.425** | 0.281 | - |
| **MAE Error Reduction** | **+14.3% vs Heuristic** | Baseline | **+66.1% vs Naive** |

### Delay-Risk Classification Performance ($> 24$h Threshold)
- **Accuracy:** `84.0%`
- **Precision:** `87.3%`
- **Recall (Sensitivity):** `95.4%` (captures over 95% of severely delayed shipments)
- **F1-Score:** `0.912`
- **ROC-AUC:** `0.780`

### Dominant Predictive Factors
1. `port_congestion` (**21.4%**): Terminal berth queueing and crane capacity.
2. `observed_delay_at_port_hours` (**14.4%**): Accumulated upstream delay.
3. `congestion_weather_interaction` (**13.4%**): Compounded weather and port queueing.
4. `sailing_delay_hours` (**9.6%**): Ocean transit speed reduction.

---

## 🖥️ Dashboard Features

- **Tab 1: 🌐 Fleet Intelligence & Live Monitoring**
  - KPI metric cards (Total Shipments, On-Time Arrival Rate %, Mean Delay, Critical Risk Shipments).
  - Multi-dimensional filters (Origin, Destination, Risk Status, Free-text Search).
  - Delay distribution histogram and Route delay rankings.
  - Interactive fleet table with sortable columns and color-coded status badges.
- **Tab 2: 🔍 Shipment Root-Cause Deep-Dive**
  - Visual stage progression stepper (*Origin → Ocean → Port Checkpoint → Customs → Final Delivery*).
  - Predicted ETA with 90% confidence window and scheduled ETA comparison.
  - **Interactive Delay Propagation Waterfall Chart** showing planned vs incurred stage delays.
  - **Tree SHAP Feature Attribution Chart** showing exact hours added/subtracted by each feature.
  - Qualitative operational narrative bullets detailing delay drivers.
- **Tab 3: 🧪 What-If Scenario Simulator & Journey Builder**
  - Adjust sliders for Departure Delay, Sailing Delay, Port Congestion, and Weather Severity.
  - Dynamic Delay Threshold slider.
  - Instant real-time recalculation of Predicted ETA, Delay, and Late Probability.
  - **Congestion Sensitivity Analysis Curve** plotting delay response across terminal congestion levels.
- **Tab 4: 📊 Model Performance & Benchmarks**
  - Model vs Baseline scorecard.
  - Actual vs Predicted delay parity scatter plot.
  - Global feature importance rankings.
  - Complete documentation of assumptions and production roadmap.

---

## 📁 Repository Structure

```
├── app.py                      # Interactive Streamlit dashboard
├── requirements.txt            # Project dependencies
├── README.md                   # System documentation and instructions
├── data/
│   └── shipments.csv           # 300 synthetic shipment records
├── artifacts/
│   ├── eta_model.joblib        # Saved ML models, bundle, and residual stats
│   └── eta_model.metrics.json  # Held-out evaluation metrics
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Deterministic synthetic journey & delay cascade generator
│   ├── features.py             # Checkpoint feature engineering & leakage prevention
│   ├── train.py                # Model training, baseline benchmarking, and validation
│   ├── predict.py              # Prediction engine, confidence intervals, and risk scoring
│   └── explain.py              # Tree SHAP attributions and stage waterfall generator
├── reports/
│   ├── TECHNICAL_REPORT.md     # In-depth technical report and empirical analysis
│   └── TECHNICAL_REPORT_TEMPLATE.md
└── tests/
    └── test_pipeline.py        # Automated test suite (6 comprehensive tests)
```

---

## 📄 Deliverables & Technical Report
For detailed mathematical formulations, delay propagation dynamics, feature engineering audits, and operational limitations, refer to [reports/TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md).
