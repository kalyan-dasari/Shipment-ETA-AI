# 🚢 Shipment ETA Intelligence & Delay Propagation AI

An advanced machine learning system and interactive decision-support dashboard for **Predicting Shipment ETA and Delay Propagation** across multi-stage maritime supply chains.

---

## 🌟 System Overview

Maritime container journeys experience cascading schedule disruptions where early origin or sea transit delays snowball into severe port congestion, customs clearance bottlenecks, and missed delivery windows.

This system provides:
1. **Final Delivery ETA Prediction:** Estimates the expected arrival timestamp at the **Port Arrival Checkpoint** using scheduled timelines, observed upstream voyage history, and terminal conditions.
2. **Total & Remaining Delay Estimation:** Predicts final delivery delay in hours with **calibrated Out-Of-Bag uncertainty intervals** (`[Lower ETA — Upper ETA]`).
3. **Calibrated Delay-Risk Scoring:** Computes the probability that a shipment exceeds an operational delay threshold (default: 24 hours), assigning actionable risk tiers:
   - 🟢 **Low Risk** (<25% probability)
   - 🟡 **Moderate Risk** (25% – 50%)
   - 🟠 **High Risk** (50% – 75%)
   - 🔴 **Critical Risk** (≥75%)
4. **Stage-by-Stage Delay Propagation:** Uncovers how delays compound across 5 discrete journey stages without double-counting:
   $$\text{Departure} \longrightarrow \text{Ocean Sailing} \longrightarrow \text{Port Dwell / Offloading} \longrightarrow \text{Customs Clearance} \longrightarrow \text{Final Delivery}$$
5. **Explainable AI (Exact Tree SHAP Decomposition):** Quantifies exactly how many hours each operational and environmental factor (port congestion, weather severity, ocean transit, departure friction) added or subtracted from the baseline ETA.
6. **Interactive Streamlit Dashboard:** Features real-time fleet monitoring, individual shipment root-cause deep-dives with operator action guidance, a What-If scenario simulator, and model evaluation benchmarks.

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
# Generate 300 validated synthetic shipment journeys with linked stage delays
python -m src.data_generator

# Train Random Forest Regressor, delay-risk classifier, and baselines
python -m src.train
```

### 3. Generate PDF Technical Report
```bash
python -m src.generate_pdf_report
```
Compiles a publication-ready 3-page PDF at `reports/TECHNICAL_REPORT.pdf`.

### 4. Launch Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run Automated Unit Tests
```bash
python -m pytest -v
```

---

## 📊 Evaluation Results (Held-Out Test Set)

The system was evaluated using a 25% stratified held-out test split ($N_{\text{test}} = 75$ shipments, seed 42) against two operational baselines:

| Metric | Random Forest ML Model | Operational Heuristic Baseline | Naive Baseline (Zero Remaining Delay) |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **9.915 hours** | 11.531 hours | 29.168 hours |
| **Root Mean Squared Error (RMSE)** | **12.693 hours** | 14.175 hours | 32.163 hours |
| **Coefficient of Determination ($R^2$)** | **0.422** | 0.280 | -2.708 |
| **MAE Error Reduction** | **+14.01% vs Heuristic** | Baseline Benchmark | **+66.01% vs Naive** |

### Delay-Risk Classification Performance ($> 24$h Threshold)
- **Accuracy:** `84.0%`
- **Precision:** `87.3%`
- **Recall (Sensitivity):** `95.4%` (captures over 95% of severely delayed shipments)
- **F1-Score:** `0.912`
- **ROC-AUC:** `0.780`
- **Confusion Matrix:** `[[1, 9], [3, 62]]` (TN: 1, FP: 9, FN: 3, TP: 62)

### Prediction Interval Calibration
- **Method:** Out-of-Bag (OOB) empirical residual quantiles ($q_{10} = -14.94\text{h}, q_{90} = +16.76\text{h}$).
- **Nominal Target:** 80.0%
- **Empirical Test Coverage:** **74.7%** on held-out test data.

### Dominant Predictive Factors (Tree MDI)
1. `port_congestion` (**21.42%**): Terminal berth queueing and crane capacity.
2. `observed_delay_at_port_hours` (**14.26%**): Accumulated upstream delay floor.
3. `congestion_weather_interaction` (**13.37%**): Compounded weather and port queueing.
4. `sailing_delay_hours` (**9.79%**): Ocean transit speed reduction.
5. `upstream_delay_ratio` (**7.69%**): Relative voyage disruption ratio.

---

## 🖥️ Dashboard Features

- **Tab 1: 🌐 Fleet Intelligence & Live Monitoring**
  - KPI metric cards (Total Shipments, On-Time Arrival Rate %, Mean Delay, Critical Risk Shipments).
  - Multi-dimensional filters (Origin, Destination, Trade Lane, Risk Status, Free-text Search).
  - Delay distribution histogram and Route delay rankings.
  - Interactive fleet table with sortable columns and color-coded status badges.
- **Tab 2: 🔍 Shipment Root-Cause Deep-Dive & Operator Playbook**
  - Explicit separation of **Observed Checkpoint Facts**, **AI In-Flight Predictions**, and **Unknown Future Actuals**.
  - Visual stage progression stepper (*Origin → Ocean → Port Checkpoint → Customs → Final Delivery*).
  - Predicted ETA with uncertainty band and scheduled ETA comparison.
  - **Operator Decision Playbook** offering actionable guidance (consignee advisories, priority berthing, customs pre-clearing, drayage rescheduling).
  - **Interactive Delay Propagation Waterfall Chart** showing planned vs incremental stage delays.
  - **Tree SHAP Feature Attribution Chart** showing exact hours added/subtracted by each feature.
  - Collapsible **Post-Delivery Historical Audit** for model validation.
- **Tab 3: 🧪 What-If Scenario Simulator & Journey Builder**
  - Adjust sliders for Departure Delay, Sailing Delay, Port Congestion, and Weather Severity.
  - Dynamic Delay Threshold slider.
  - Instant real-time recalculation of Predicted ETA, Delay, and Late Probability.
  - **Terminal Congestion Sensitivity Curve** plotting delay response across terminal congestion levels.
- **Tab 4: 📊 Model Performance & Benchmarks**
  - Model vs Baseline scorecard.
  - Actual vs Predicted delay parity scatter plot.
  - Confusion matrix display and uncertainty interval calibration statistics.
  - Global feature importance rankings.
  - Complete documentation of assumptions and production roadmap.

---

## 📁 Repository Structure

```
├── app.py                      # Interactive Streamlit dashboard
├── requirements.txt            # Project dependencies
├── README.md                   # System documentation and instructions
├── data/
│   └── shipments.csv           # 300 validated synthetic shipment records
├── artifacts/
│   ├── eta_model.joblib        # Saved ML models, bundle, and residual stats
│   └── eta_model.metrics.json  # Held-out evaluation metrics
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic journey generator with stage delay propagation & validation
│   ├── features.py             # Checkpoint feature engineering & leakage prevention
│   ├── train.py                # Model training, baseline benchmarking, and OOB residual calibration
│   ├── predict.py              # Prediction engine, confidence intervals, and risk scoring
│   ├── explain.py              # Tree SHAP attributions and stage waterfall generator
│   └── generate_pdf_report.py  # Script generating publication-quality 3-page PDF report
├── reports/
│   ├── TECHNICAL_REPORT.md     # In-depth technical report (8 required sections + worked example)
│   ├── TECHNICAL_REPORT.pdf    # 3-page printable PDF deliverable
│   └── TECHNICAL_REPORT_TEMPLATE.md
└── tests/
    └── test_pipeline.py        # Automated test suite (11 comprehensive tests)
```

---

## 📄 Deliverables & Technical Report
- **Markdown Report:** [reports/TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md)
- **Printable 3-Page PDF:** [reports/TECHNICAL_REPORT.pdf](reports/TECHNICAL_REPORT.pdf)
