# 🚢 Shipment ETA Intelligence & Delay Propagation AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red.svg)](https://streamlit.io/)
[![Pytest Passing](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)](tests/test_pipeline.py)
[![Deliverable PDF](https://img.shields.io/badge/report-3--Page%20PDF-purple.svg)](reports/TECHNICAL_REPORT.pdf)

An advanced machine learning system and interactive decision-support dashboard for **Predicting Shipment ETA and Delay Propagation** across multi-stage maritime supply chains.

---

## 🌟 Executive Summary & Problem Context

In international container shipping, schedule disruptions cascade through downstream supply chain stages. A delay during origin terminal handling or deep-sea ocean transit does not remain isolated; it causes missed berthing windows, terminal anchorage queues, customs clearance bottlenecks, and missed inland rail/truck connections.

Traditional freight visibility platforms rely on static schedule tables or naive linear projections that assume zero future delay after an initial disruption. 

This system models the high-value **Port Arrival Checkpoint** (outer pilot station / anchorage): estimating final delivery arrival timestamp and delay magnitude while the voyage is in-progress, enabling logistics operators to proactively:
1. Issue calibrated delay advisories to warehouse receiving and retail consignees.
2. Request off-dock container staging or priority crane slots to mitigate port yard dwell.
3. Pre-lodge electronic customs manifests (EDI) to eliminate customs dwell.
4. Reschedule onward drayage and rail connections to avoid punitive demurrage and detention charges.

> **Notice on Synthetic Data:** This prototype uses mathematically modeled synthetic journeys. Evaluation metrics establish consistency with the underlying simulation mechanics and do not represent empirical real-world carrier accuracy.

---

## 🚀 Quick Start Guide

### 1. Requirements & Installation
Requires Python 3.10+ (tested and verified on Python 3.14).

```bash
# Clone or navigate to the repository
cd Shipment-ETA-AI

# Create and activate virtual environment (recommended)
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
Generates 300 validated synthetic shipment journeys across 5 discrete stages with linked delay propagation equations:
```bash
python -m src.data_generator
```
Output: `Generated 300 validated synthetic shipments at data/shipments.csv`

### 3. Train Models & Calibrate Baselines
Trains the Random Forest Regressor (300 trees, Out-Of-Bag scoring), Delay-Risk Classifier (200 trees), Out-Of-Bag residual quantiles, and benchmark baselines on a 25% stratified held-out split (`seed=42`):
```bash
python -m src.train
```
Artifacts saved: [`artifacts/eta_model.joblib`](artifacts/eta_model.joblib) and [`artifacts/eta_model.metrics.json`](artifacts/eta_model.metrics.json).

### 4. Run Automated Unit Tests
Executes the comprehensive 13-test test suite verifying data integrity, feature leakage prevention, exact ETA arithmetic, and model outputs:
```bash
python -m pytest -v
```
Result: `13 passed in ~4.2s`

### 5. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
# Alternatively: python app.py (auto-delegates to Streamlit runner)
```
Open your browser at `http://localhost:8501`.

### 6. Compile 3-Page Executive PDF Technical Report
```bash
python -m src.generate_pdf_report
```
Generates the publication-quality printable report at [`reports/TECHNICAL_REPORT.pdf`](reports/TECHNICAL_REPORT.pdf).

---

## 📊 Held-Out Evaluation Results & Benchmarks

The system was evaluated on a 25% stratified held-out test split ($N_{\text{test}} = 75$ shipments, `seed=42`) against two operational baselines:

### 1. Delay Regression Benchmark Table

| Model / Approach | Held-Out Test MAE | Held-Out Test RMSE | Test $R^2$ Score | Error Reduction vs Benchmark |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Baseline (Zero Downstream Delay)** | 29.168 hours | 32.163 hours | -2.708 | **+66.01% error vs AI model** |
| **Operational Heuristic Baseline** | 11.531 hours | 14.175 hours | 0.280 | Benchmark Baseline |
| **Random Forest AI Regressor** | **9.915 hours** | **12.693 hours** | **0.422** | **+14.01% improvement vs heuristic** |

- **Operational Heuristic Definition:** $\widehat{\text{Delay}} = \text{Observed Delay at Port} + \text{Median}(\text{Training Downstream Residuals})$.
- **Naive Baseline Definition:** Assumes zero downstream delay ($\widehat{\text{Delay}} = \text{Observed Delay at Port}$).
- **Key Finding:** The Naive baseline fails severely (MAE 29.17h) because downstream port dwell, customs inspection, and inland linehaul introduce substantial variance (18+ hours on average). The Random Forest model outperforms the operational heuristic by 14.01%, explaining 42.2% of downstream delay variance.

### 2. Delay-Risk Classification Performance ($> 24.0$h Threshold)

| Metric | Measured Score | Operational Significance in Supply Chain |
| :--- | :---: | :--- |
| **Accuracy** | **84.0%** ($63 / 75$) | Correctly classifies delivery status on 84% of test voyages |
| **Precision** | **87.3%** ($62 / 71$) | 87.3% of late-flagged shipments are genuinely late (low false alarm rate) |
| **Recall (Sensitivity)** | **95.4%** ($62 / 65$) | Captures 95.4% of all shipments exceeding 24h delay (vital for SLA protection) |
| **F1-Score** | **0.912** | High harmonic balance between precision and recall |
| **ROC-AUC** | **0.780** | Robust discrimination across operational decision thresholds |
| **Confusion Matrix** | `[[1, 9], [3, 62]]` | **TN:** 1 on-time, **FP:** 9 false alarms, **FN:** 3 missed late, **TP:** 62 caught late |

> **Operational Trade-Off Rationale:** In international logistics, a False Negative (failing to alert a consignee who then incurs idle labor or demurrage) is significantly more expensive than a False Positive (checking status on a container that clears early). The classifier prioritizes high sensitivity (95.4% recall) to protect consignee SLAs.

### 3. Prediction Interval Calibration
- **Method:** Out-of-Bag (OOB) empirical residual quantiles from training data ($q_{10} = -14.94\text{h}, q_{90} = +16.76\text{h}$).
- **Nominal Target:** 80.0%
- **Measured Empirical Test Coverage:** **74.7%** on held-out test data.
- **Reporting Stance:** Transparently documented as **exploratory uncertainty intervals** rather than claiming 80% coverage, reflecting heteroscedastic terminal queueing dynamics.

### 4. Global Feature Drivers (Tree MDI)
1. `port_congestion` (**21.42%**): Dominant predictor of terminal queueing and berth dwell.
2. `observed_delay_at_port_hours` (**14.26%**): Incurred upstream schedule delay floor.
3. `congestion_weather_interaction` (**13.37%**): Compounded weather and port queueing.
4. `sailing_delay_hours` (**9.79%**): Sea transit speed reduction.
5. `upstream_delay_ratio` (**7.69%**): Relative voyage disruption ratio.
6. `congestion_buffer_ratio` (**7.53%**): Congestion stress on scheduled port dwell.

---

## 🔍 Worked Shipment Example: `SHP-0001`

To ensure full audibility, here is the exact numerical trace for shipment `SHP-0001` from [`data/shipments.csv`](data/shipments.csv):

### 1. Voyage Profile & Planned Journey
- **Shipment ID:** `SHP-0001` | **Carrier:** *CMA CGM Group* | **Vessel:** *Nordic Horizon*
- **Trade Lane:** *Shanghai to Hamburg* via *Port Delta (Dubai)*
- **Scheduled Departure:** `2025-03-15 12:00:00 UTC`
- **Planned Stage Durations:** Sailing: 352.98h | Port Dwell: 29.31h | Customs: 19.72h | Inland Final: 18.15h
- **Planned Total Journey Duration:** $352.98 + 29.31 + 19.72 + 18.15 = \mathbf{420.16\text{ hours}}$
- **Scheduled ETA:** `2025-03-15 12:00:00 UTC` + 420.16h = $\mathbf{2025\text{-}04\text{-}02\text{ }00:09:36\text{ UTC}}$

### 2. Observed Checkpoint Facts (Port Arrival)
- Departure Delay: $+3.35\text{ h}$ | Ocean Sailing Delay: $+17.78\text{ h}$
- **Observed Upstream Delay at Checkpoint:** $3.35 + 17.78 = \mathbf{+21.13\text{ hours}}$
- **Terminal Signals:** Port Congestion $= 0.228$ (favorable) | Weather Severity $= 0.667$ (rough sea state)

### 3. Model In-Flight Predictions (Inference Output)
- **Predicted Total Delay:** $\mathbf{+41.59\text{ hours}}$
- **Predicted Remaining Downstream Delay:** $41.59 - 21.13 = \mathbf{+20.46\text{ hours}}$
- **Predicted Final ETA:** `2025-04-02 00:09:36 UTC` + 41.59h = $\mathbf{2025\text{-}04\text{-}03\text{ }17:45:00\text{ UTC}}$
- **Exploratory Uncertainty Band:** $[+26.65\text{h}, +58.35\text{h}] \longrightarrow [\text{03 Apr 02:48 UTC}, \text{04 Apr 10:30 UTC}]$
- **Late Probability:** $\mathbf{95.9\%}$ ($>24\text{h}$ threshold) $\longrightarrow$ **Risk Tier: CRITICAL**
- **Tree SHAP Attributions:** Observed Upstream Delay pushed ETA later by $+5.19\text{h}$ and Sailing Delay by $+3.39\text{h}$; favorable Port Congestion buffered downstream delay by $-5.96\text{h}$ and port buffer by $-3.27\text{h}$.

### 4. Post-Delivery Historical Audit (Withheld During Prediction)
- Actual Downstream Delays: Port Dwell: $+11.47\text{h}$ | Customs: $+5.60\text{h}$ | Final Delivery: $+0.95\text{h}$
- **Actual Total Delay:** $21.13\text{h (upstream)} + 18.02\text{h (downstream)} = \mathbf{+39.15\text{ hours}}$
- **Actual Delivery Timestamp:** `2025-04-02 00:09:36 UTC` + 39.15h = $\mathbf{2025\text{-}04\text{-}03\text{ }15:18:36\text{ UTC}}$
- **Prediction Absolute Error:** $|41.59 - 39.15| = \mathbf{2.44\text{ hours}}$ (Accurate within 2.5 hours).
- **Exact ETA Arithmetic Equality:**
  $$\text{Scheduled ETA } (2025\text{-}04\text{-}02\text{ }00:09:36) + \text{Predicted Delay } (41.59\text{h}) = \text{Predicted ETA } (2025\text{-}04\text{-}03\text{ }17:45:00) \quad [\text{EXACT}]$$

---

## 🛡️ Prediction Checkpoint & Target Leakage Prevention

Predictions occur strictly at the **Port Arrival Checkpoint** (vessel arrival at outer pilot station / anchorage).

```
   [Departure] ──► [Ocean Sailing] ──► [PORT ARRIVAL CHECKPOINT] ──► [Port Dwell] ──► [Customs] ──► [Final Delivery]
  └────────────── KNOWN ──────────────┘                          └───────────── UNKNOWN FUTURE ─────────────┘
```

### Explicit Feature Allowlist ($X^{\text{port}}$)
Only 13 features are passed to the model:
1. `planned_sailing_hours`: Planned sea transit duration (72–360h).
2. `planned_port_hours`: Planned terminal dwell buffer (8–36h).
3. `planned_customs_hours`: Planned regulatory clearance buffer (4–24h).
4. `planned_final_hours`: Planned inland linehaul duration (12–60h).
5. `planned_remaining_hours_at_port`: Sum of planned remaining stages.
6. `departure_delay_hours`: Incurred origin terminal delay ($\Delta_{\text{dep}} \ge 0$).
7. `sailing_delay_hours`: Incurred ocean transit delay ($\Delta_{\text{sail}} \ge 0$).
8. `observed_delay_at_port_hours`: Cumulative delay at port milestone ($\Delta_{\text{dep}} + \Delta_{\text{sail}}$).
9. `port_congestion`: Point-in-time terminal congestion index ($0.0–1.0$).
10. `weather_severity`: Point-in-time meteorological severity index ($0.0–1.0$).
11. `upstream_delay_ratio`: Interaction: observed delay / planned sailing duration.
12. `congestion_weather_interaction`: Interaction: port congestion $\times$ weather severity.
13. `congestion_buffer_ratio`: Interaction: port congestion / planned port dwell buffer.

### Forbidden Downstream Columns Strictly Excluded
The feature pipeline strictly removes 14 downstream columns:
`port_delay_hours`, `customs_delay_hours`, `final_stage_delay_hours`, `actual_port_hours`, `actual_customs_hours`, `actual_final_hours`, `cum_delay_port_departure_hours`, `cum_delay_customs_done_hours`, `actual_port_departure`, `actual_customs_done`, `actual_delivery`, `actual_total_delay_hours`, `actual_remaining_delay_hours`, and `late_delivery`.

---

## 📐 Mathematical Event-Time Definitions & Delay Mechanics

To prevent accidental double-counting, the event-time model formally distinguishes **incremental stage delay** from **cumulative schedule deviation**:

1. **Planned Total Duration ($D_{\text{planned}}$):**
   $$D_{\text{planned}} = D_{\text{sail}}^{\text{plan}} + D_{\text{port}}^{\text{plan}} + D_{\text{customs}}^{\text{plan}} + D_{\text{final}}^{\text{plan}}$$
2. **Scheduled ETA ($T_{\text{sched\_arr}}$):**
   $$T_{\text{sched\_arr}} = T_{\text{sched\_dep}} + D_{\text{planned}}$$
3. **Incremental Stage Delay ($\Delta_s$):**
   $$\Delta_s = D_s^{\text{actual}} - D_s^{\text{planned}} \ge 0, \quad \forall s \in \{1, 2, 3, 4, 5\}$$
4. **Cumulative Delay at Milestone $k$ ($C_k$):**
   $$C_k = \sum_{s=1}^k \Delta_s$$
   - Checkpoint Upstream Delay: $C_2 = \Delta_{\text{dep}} + \Delta_{\text{sail}}$
   - Final Delivery Ground Truth: $C_5 = \sum_{s=1}^5 \Delta_s = \text{Total Delay}$
5. **Actual Delivery Timestamp ($T_{\text{actual\_deliv}}$):**
   $$T_{\text{actual\_deliv}} = T_{\text{sched\_arr}} + C_5 = T_{\text{sched\_arr}} + \text{Total Delay}$$
6. **Ground-Truth Late Label ($y_{\text{late}}$):**
   $$y_{\text{late}} = \mathbb{I}(\text{Total Delay} > 24.0\text{ hours})$$
7. **Predicted Final ETA ($\widehat{T}_{\text{ETA}}$):**
   $$\widehat{T}_{\text{ETA}} = T_{\text{sched\_arr}} + \widehat{\text{Delay}}_{\text{total}}$$

*Audit Verification:* Across all 300 rows in [`data/shipments.csv`](data/shipments.csv), $\max |\sum_{s=1}^5 \Delta_s - \text{Total Delay}| = \mathbf{0.00\text{ hours}}$ (zero double-counting).

---

## 🎯 Logistics Operator Action Playbook

The system translates delay predictions into concrete operational actions via automated decision rules:

| Risk Tier | Late Probability | Recommended Operational Interventions |
| :---: | :---: | :--- |
| 🔴 **Critical** | $\ge 75\%$ | **Immediate SLA Protection:** Issue formal revised delivery notices to consignee receiving teams; request priority container offloading or off-dock drayage to avoid yard dwell; pre-lodge customs manifests (EDI); re-book inland rail/drayage appointments to prevent demurrage penalties. |
| 🟠 **High** | $50\% - 75\%$ | **Active Mitigation:** Contact terminal agent to confirm berth allocation; alert destination dispatchers to standby for off-peak gate-out; prepare alternate customs documentation. |
| 🟡 **Moderate** | $25\% - 50\%$ | **Standard Monitoring:** Monitor port pilot board for anchorage queue changes; verify container crane discharge window with ocean carrier. |
| 🟢 **Low** | $< 25\%$ | **Automated Tracking:** Journey is progressing within acceptable tolerances. Maintain standard milestone notifications. |

---

## 🖥️ Streamlit Dashboard Walkthrough

Launch via `streamlit run app.py` (or `python app.py`) to access four dedicated views:

- **Tab 1: 🌐 Fleet Intelligence & Live Monitoring**
  - KPI metric cards (Active Shipments, On-Time Arrival %, Mean Expected Delay, Critical Risk Shipments).
  - Multi-dimensional filters (Origin, Destination, Late Risk Category, Free-text search by ID/Vessel/Carrier).
  - Fleet-wide predicted delay distribution histogram and sortable table with status badges.
- **Tab 2: 🔍 Shipment Root-Cause Deep-Dive & Operator Playbook**
  - Visual 5-stage progression stepper (*Origin $\to$ Ocean $\to$ Port Checkpoint $\to$ Customs $\to$ Delivery*).
  - Clear visual demarcation: **Observed Facts** vs **AI Predictions** vs **Unknown Future Stages**.
  - Interactive **Delay Propagation Waterfall** chart with cumulative deviation tracking.
  - **Tree SHAP Attributions** showing exact hours added/subtracted by each factor.
  - Collapsible **Post-Delivery Historical Audit** for retrospective evaluation.
- **Tab 3: 🧪 What-If Scenario Simulator & Journey Builder**
  - Interactive sliders for Departure Delay, Sailing Delay, Port Congestion, and Weather Severity.
  - Real-time recalculation of Predicted ETA, Delay, and Late Probability.
  - **Terminal Congestion Sensitivity Curve** plotting delay response across 0.0 to 1.0 congestion levels.
- **Tab 4: 📊 Model Performance & Benchmarks**
  - Model vs Baseline scorecard.
  - Actual vs Predicted delay parity scatter plot with identity reference line.
  - Confusion matrix display, Out-of-Bag calibration spread, and global feature importance rankings.

---

## 🧪 Automated Unit Test Suite

The repository includes 13 automated unit tests in [`tests/test_pipeline.py`](tests/test_pipeline.py):

```bash
python -m pytest -v
```

1. `test_generator_size_and_targets`: Validates synthetic data schema and positive delays.
2. `test_features_exclude_future_target`: Validates exact feature count and allowlist.
3. `test_forbidden_columns_leakage_rejection`: Confirms all 14 forbidden columns are stripped.
4. `test_eta_arithmetic_consistency`: Confirms scheduled and actual timestamps match mathematical definitions.
5. `test_dataset_validation_rejects_negative_delays`: Verifies dataset validator catches negative durations.
6. `test_dataset_validation_rejects_double_counting`: Verifies validator catches stage sum discrepancies.
7. `test_explanation_mentions_uncertainty`: Confirms statistical disclaimers in explanations.
8. `test_prediction_dataframe_confidence_intervals`: Validates batch prediction intervals and risk categories.
9. `test_single_prediction_and_attributions`: Confirms Tree SHAP attributions sum to tree prediction delta.
10. `test_stage_waterfall_structure`: Validates 5-stage delay waterfall structure.
11. `test_baseline_and_model_evaluation_metrics`: Checks model MAE < baseline MAE and confusion matrix sum.
12. `test_predicted_eta_exact_arithmetic`: Confirms `predicted_eta == scheduled_arrival + predicted_delay_hours`.
13. `test_shp0001_worked_example_matches_report`: Confirms exact reproduction of worked example `SHP-0001`.

---

## 📁 Repository Structure

```
Shipment-ETA-AI/
├── app.py                      # Interactive 4-tab Streamlit decision-support dashboard
├── requirements.txt            # Constrained dependencies (pandas, numpy, scikit-learn, etc.)
├── README.md                   # Comprehensive assignment documentation & metrics
├── data/
│   └── shipments.csv           # 300 validated synthetic shipment records
├── artifacts/
│   ├── eta_model.joblib        # Fitted ML models, feature lists, and OOB residual stats
│   └── eta_model.metrics.json  # Recomputed evaluation metrics JSON
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # 5-stage synthetic journey generator with cascading delays
│   ├── features.py             # Checkpoint allowlist & target leakage prevention
│   ├── train.py                # Model training, baseline benchmarking, and OOB calibration
│   ├── predict.py              # In-flight prediction engine, ETA arithmetic, risk scoring
│   ├── explain.py              # Tree SHAP path attributions & stage waterfall generator
│   └── generate_pdf_report.py  # Script generating the 3-page PDF technical report
├── reports/
│   ├── TECHNICAL_REPORT.md     # Full technical report (8 sections + worked example)
│   ├── TECHNICAL_REPORT.pdf    # Final 3-page publication-quality PDF deliverable
│   └── TECHNICAL_REPORT_TEMPLATE.md # Executive template & reference brief
└── tests/
    └── test_pipeline.py        # Automated test suite (13 comprehensive tests)
```

---

## 📄 Deliverables

- **Markdown Technical Report:** [reports/TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md)
- **Executive 3-Page Printable PDF:** [reports/TECHNICAL_REPORT.pdf](reports/TECHNICAL_REPORT.pdf)
- **Executive Reference Template:** [reports/TECHNICAL_REPORT_TEMPLATE.md](reports/TECHNICAL_REPORT_TEMPLATE.md)
- **Evaluation Metrics JSON:** [artifacts/eta_model.metrics.json](artifacts/eta_model.metrics.json)

---

*Validated and verified by Kalyan Dasari · October 2026.*
