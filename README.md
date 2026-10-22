# 🚢 Shipment ETA Intelligence & Delay Propagation AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red.svg)](https://streamlit.io/)
[![Pytest Passing](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)](tests/test_pipeline.py)
[![Deliverable PDF](https://img.shields.io/badge/report-3--Page%20PDF-purple.svg)](reports/TECHNICAL_REPORT.pdf)

An advanced machine learning system and interactive decision-support dashboard for **Predicting Shipment ETA and Delay Propagation** across multi-stage maritime container supply chains.

---

## 📋 Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Intended Users & Practical Decision Support](#2-intended-users--practical-decision-support)
3. [Key System Features](#3-key-system-features)
4. [Architecture & End-to-End Data Flow](#4-architecture--end-to-end-data-flow)
5. [Five Journey Stages & Delay Propagation Cascade](#5-five-journey-stages--delay-propagation-cascade)
6. [Prediction Checkpoint & Target Leakage Prevention](#6-prediction-checkpoint--target-leakage-prevention)
7. [Dataset Description & Synthetic Data Disclaimer](#7-dataset-description--synthetic-data-disclaimer)
8. [Repository Structure](#8-repository-structure)
9. [Prerequisites & Supported Environment](#9-prerequisites--supported-environment)
10. [Local Setup & Installation](#10-local-setup--installation)
11. [Execution Commands (Step-by-Step Pipeline)](#11-execution-commands-step-by-step-pipeline)
12. [Interactive Streamlit Dashboard Usage](#12-interactive-streamlit-dashboard-usage)
13. [How to Interpret Model Predictions & Metrics](#13-how-to-interpret-model-predictions--metrics)
14. [Audited Worked Numerical Shipment Example (`SHP-0001`)](#14-audited-worked-numerical-shipment-example-shp-0001)
15. [Automated Tests & Reproducibility](#15-automated-tests--reproducibility)
16. [Comprehensive Troubleshooting Guide](#16-comprehensive-troubleshooting-guide)
17. [Realistic Staged Future Development Roadmap](#17-realistic-staged-future-development-roadmap)
18. [Optional 5-Minute Demonstration Walkthrough Plan](#18-optional-5-minute-demonstration-walkthrough-plan)
19. [Final Submission Deliverables](#19-final-submission-deliverables)

---

## 1. Project Overview & Problem Statement

In global ocean container freight, static scheduling is notoriously unreliable. Unforeseen disruptions during origin vessel loading or deep-sea ocean transit trigger compounding delays: missed berthing windows, extended outer anchorage dwell, customs clearance bottlenecks, and missed inland rail or truck drayage appointments.

Traditional freight visibility platforms rely on static schedule tables or naive linear projections that assume zero future delay once an initial disruption occurs. Predicting ETA at booking is too early (operational signals are unknown), while predicting after customs clearance is too late (no time remains for intervention).

This project focuses on the high-value **Port Arrival Checkpoint** (outer pilot station / anchorage): estimating the final delivery timestamp, delay magnitude in hours, and the likelihood of exceeding an operational threshold (24 hours) while the voyage is in-progress.

---

## 2. Intended Users & Practical Decision Support

The system is designed for **supply chain dispatchers, freight forwarders, and logistics operations managers**. It provides actionable decision intelligence:

| Risk Tier | Late Probability | Operational Decision Interventions |
| :---: | :---: | :--- |
| 🔴 **Critical** | $\ge 75\%$ | **Immediate SLA Protection:** Alert consignee receiving warehouses with revised delivery windows; contact port agent to request priority container berthing or off-dock drayage; pre-lodge customs manifests (EDI); re-book onward rail/truck connections to eliminate demurrage. |
| 🟠 **High** | $50\% - 75\%$ | **Active Mitigation:** Track terminal pilot queue; alert destination dispatchers to standby for off-peak gate-out; prepare alternate customs documentation. |
| 🟡 **Moderate** | $25\% - 50\%$ | **Standard Monitoring:** Monitor port congestion index changes; confirm scheduled container crane discharge window with ocean carrier. |
| 🟢 **Low** | $< 25\%$ | **Automated Tracking:** Journey is progressing within acceptable tolerances. Maintain standard automated tracking updates. |

*Disclaimer:* The system serves as decision support. It does not claim to automatically prevent physical carrier or terminal delays.

---

## 3. Key System Features

1. **Point Delay & Final ETA Prediction:** Estimates final delivery delay in hours and calculates predicted delivery timestamp: $\widehat{T}_{\text{ETA}} = T_{\text{sched\_arr}} + \widehat{\text{Delay}}_{\text{total}}$.
2. **Calibrated Delay-Risk Scoring:** Random Forest Classifier predicting the calibrated probability that total delay exceeds 24.0 hours, mapped to actionable risk tiers (Low, Moderate, High, Critical).
3. **Exploratory Uncertainty Intervals:** Out-of-Bag (OOB) empirical residual quantiles ($q_{10} = -14.94\text{h}, q_{90} = +16.76\text{h}$) providing realistic uncertainty spreads.
4. **Stage-by-Stage Delay Waterfall:** Disaggregates delay across 5 discrete journey stages without double-counting.
5. **Exact Tree SHAP Attributions:** Decomposes tree decision paths to quantify the exact hours contributed by port congestion, weather severity, and ocean transit friction.
6. **Interactive 4-Tab Streamlit Dashboard:** Features live fleet monitoring, single-shipment root-cause deep-dives with operator playbooks, a What-If scenario builder with congestion sensitivity curves, and held-out benchmark evaluation charts.
7. **Strict Target Leakage Guard:** Enforces an allowlist of 13 checkpoint-available features and strips 14 downstream actual outcomes.

---

## 4. Architecture & End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA GENERATION & EVENT-TIME SIMULATION (src/data_generator.py)          │
│    300 synthetic shipments • 5 stages • Chronological ISO-8601 timestamps   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. CHECKPOINT FEATURE ENGINEERING (src/features.py)                         │
│    Feature Allowlist (13 features) • Strict stripping of downstream actuals │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING & EVALUATION PIPELINE (src/train.py)                      │
│    Stratified 75/25 split (Seed 42) • Pipeline SimpleImputer (Train only)   │
│    Random Forest Regressor (300 trees, OOB) • Classifier (200 trees)        │
│    Benchmark Baselines: Operational Heuristic & Naive (Zero Future Delay)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. PREDICTION ENGINE & EXPLAINABILITY (src/predict.py, src/explain.py)      │
│    In-flight ETA • OOB confidence bands • Risk tiers • Tree SHAP attributions│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. USER INTERFACES & DELIVERABLES                                           │
│    Streamlit Dashboard (app.py) • 3-Page PDF Technical Report (PyMuPDF)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Five Journey Stages & Delay Propagation Cascade

The simulated journey comprises five discrete stages:
1. **Departure handling:** Origin terminal loading and vessel dispatch.
2. **Ocean sailing / transit:** Deep-sea navigational voyage.
3. **Port processing / dwell:** Outer anchorage waiting, berthing, and crane discharge.
4. **Customs clearance:** Regulatory document review and physical inspection.
5. **Final delivery:** Inland drayage / rail transit to final destination consignee.

### Cascading Delay Equations
$$\Delta_{\text{dep}} = \max\left(0, \mathcal{N}(2.0 + 8.5 \cdot \text{Weather}, 4.5^2)\right)$$
$$\Delta_{\text{sail}} = \max\left(0, \mathcal{N}(3.0 + 13.5 \cdot \text{Weather} + 0.15 \cdot \Delta_{\text{dep}}, 6.0^2)\right)$$
$$\Delta_{\text{port}} = \max\left(0, \mathcal{N}(2.0 + 26.0 \cdot \text{Congestion} + 0.22 \cdot \Delta_{\text{sail}}, 7.5^2)\right)$$
$$\Delta_{\text{customs}} = \max\left(0, \mathcal{N}(1.5 + 13.0 \cdot \text{Congestion} + 0.18 \cdot \Delta_{\text{port}}, 5.0^2)\right)$$
$$\Delta_{\text{final}} = \max\left(0, \mathcal{N}(1.0 + 0.12 \cdot \Delta_{\text{customs}} + 3.0 \cdot \text{Weather}, 4.0^2)\right)$$

*Environmental Indicators:* $\text{Congestion} \sim \text{Beta}(2.2, 2.2)$ and $\text{Weather} \sim \text{Beta}(1.8, 3.2)$ ($0.0–1.0$ scale).

### Mathematical Definitions & Zero Double-Counting
- **Planned Duration ($D_{\text{planned}}$):** Sum of planned sailing, port dwell, customs, and final transit hours.
- **Scheduled ETA ($T_{\text{sched\_arr}}$):** $T_{\text{sched\_dep}} + D_{\text{planned}}$.
- **Incremental Stage Delay ($\Delta_s$):** $D_s^{\text{actual}} - D_s^{\text{planned}} \ge 0$.
- **Cumulative Delay at Milestone $k$ ($C_k$):** $\sum_{s=1}^k \Delta_s$.
  - At Port Checkpoint: $C_2 = \Delta_{\text{dep}} + \Delta_{\text{sail}}$ (Observed Upstream Delay).
  - At Final Delivery: $C_5 = \sum_{s=1}^5 \Delta_s$ (Ground Truth Total Delay).
- **Actual Delivery Timestamp:** $T_{\text{actual\_deliv}} = T_{\text{sched\_arr}} + C_5$.
- **Zero Double-Counting:** Incremental stage delays and cumulative schedule deviations are never summed together. Across all 300 rows: $\max |\sum_{s=1}^5 \Delta_s - \text{Total Delay}| = 0.00\text{ h}$.

---

## 6. Prediction Checkpoint & Target Leakage Prevention

Predictions occur strictly at the **Destination Port Outer Anchorage / Pilot Station Arrival Checkpoint**.

```
   [Departure] ──► [Ocean Sailing] ──► [PORT ARRIVAL CHECKPOINT] ──► [Port Dwell] ──► [Customs] ──► [Final Delivery]
  └────────────── KNOWN ──────────────┘                          └───────────── UNKNOWN FUTURE ─────────────┘
```

### Feature Allowlist ($X^{\text{port}}$ — 13 Features)
1. `planned_sailing_hours`
2. `planned_port_hours`
3. `planned_customs_hours`
4. `planned_final_hours`
5. `planned_remaining_hours_at_port`
6. `departure_delay_hours`
7. `sailing_delay_hours`
8. `observed_delay_at_port_hours` ($= \Delta_{\text{dep}} + \Delta_{\text{sail}}$)
9. `port_congestion`
10. `weather_severity`
11. `upstream_delay_ratio` ($= \text{observed} / \text{planned\_sailing}$)
12. `congestion_weather_interaction` ($= \text{congestion} \times \text{weather}$)
13. `congestion_buffer_ratio` ($= \text{congestion} / \text{planned\_port}$)

### Forbidden Downstream Columns (14 Columns Stripped)
`port_delay_hours`, `customs_delay_hours`, `final_stage_delay_hours`, `actual_port_hours`, `actual_customs_hours`, `actual_final_hours`, `cum_delay_port_departure_hours`, `cum_delay_customs_done_hours`, `actual_port_departure`, `actual_customs_done`, `actual_delivery`, `actual_total_delay_hours`, `actual_remaining_delay_hours`, and `late_delivery`.

Preprocessing imputers are fitted inside an `sklearn.pipeline.Pipeline` on training data only.

---

## 7. Dataset Description & Synthetic Data Disclaimer

- **Volume:** 300 completed multi-stage container voyages saved in [`data/shipments.csv`](data/shipments.csv).
- **Carriers:** Maersk Line, MSC Mediterranean, CMA CGM Group, Hapag-Lloyd, Ocean Network Express.
- **Trade Lanes:** Shanghai–Rotterdam, Singapore–Sydney, Mumbai–Dubai, Los Angeles–Hamburg, Busan–Antwerp.
- **Timestamps:** ISO-8601 second-precision UTC strings (`.floor("s")`).

> **⚠️ Synthetic Data Disclaimer:** All shipment records, delay figures, and environmental indices are mathematically modeled synthetic representations. Results demonstrate compliance with the simulation mechanics and do not establish empirical real-world carrier predictive accuracy.

---

## 8. Repository Structure

```
Shipment-ETA-AI/
├── app.py                      # Interactive 4-tab Streamlit decision-support dashboard
├── requirements.txt            # Constrained dependencies
├── README.md                   # Comprehensive project documentation & usage guide
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
│   ├── TECHNICAL_REPORT.md     # Full technical report (6 sections + worked example)
│   ├── TECHNICAL_REPORT.pdf    # Final 3-page publication-quality PDF deliverable
│   └── TECHNICAL_REPORT_TEMPLATE.md # Executive technical brief & reference template
└── tests/
    └── test_pipeline.py        # Automated test suite (13 comprehensive unit tests)
```

---

## 9. Prerequisites & Supported Environment

- **Python Version:** Python 3.10, 3.11, 3.12, 3.13, or 3.14 (Verified on Python 3.14.6 64-bit on Windows).
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Dependencies:** Constrained in [`requirements.txt`](requirements.txt):
  - `pandas>=2.1`, `numpy>=1.26`, `scikit-learn>=1.4`, `streamlit>=1.32`, `joblib>=1.3`, `pytest>=8.0`, `matplotlib>=3.8`, `pymupdf>=1.24`.

---

## 10. Local Setup & Installation

### Windows PowerShell Setup

```powershell
# 1. Navigate to the repository
cd "K:\Big Things\Shipment-ETA-AI"

# 2. Create virtual environment
py -3.11 -m venv .venv
# (or use default python): python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install pinned requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **🔒 Safe Alternative if PowerShell Execution Policy Blocks Activation:**  
> If you receive an `Execution_Policies` script restriction warning, invoke the virtual environment's Python executable directly **without weakening system-wide security settings**:
> ```powershell
> .\.venv\Scripts\python.exe -m pip install --upgrade pip
> .\.venv\Scripts\python.exe -m pip install -r requirements.txt
> ```

### macOS / Linux Setup

```bash
cd Shipment-ETA-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 11. Execution Commands (Step-by-Step Pipeline)

Execute the full pipeline in the documented order:

```powershell
# Step 1: Generate 300 validated synthetic shipments
python -m src.data_generator
# Output: Generated 300 validated synthetic shipments at data/shipments.csv

# Step 2: Train models, benchmark baselines, and calibrate uncertainty
python -m src.train
# Output: Saved model artifact to artifacts/eta_model.joblib and metrics to artifacts/eta_model.metrics.json

# Step 3: Run complete automated test suite (13 tests)
python -m pytest -v
# Output: 13 passed in ~4.2s

# Step 4: Compile publication-quality 3-page PDF technical report
python -m src.generate_pdf_report
# Output: Generated PDF technical report at: reports/TECHNICAL_REPORT.pdf (Total page count: 3 pages)

# Step 5: Launch the interactive Streamlit decision-support dashboard
streamlit run app.py
# (Alternative): python app.py (auto-delegates to Streamlit runner)
# Output: Local URL: http://localhost:8501
```

---

## 12. Interactive Streamlit Dashboard Usage

Open `http://localhost:8501` to access four dedicated operational tabs:

1. **Tab 1: 🌐 Fleet Intelligence & Live Monitoring**
   - KPI cards: Active Shipments, On-Time Arrival %, Mean Expected Delay, Critical Risk Count.
   - Interactive filters: Origin Port, Destination Port, Late Risk Tier, and Free-text Search.
   - Fleet delay distribution histogram and sortable shipment table with risk badges.
2. **Tab 2: 🔍 Shipment Root-Cause Deep-Dive & Operator Playbook**
   - Visual 5-stage progression stepper (*Origin $\to$ Ocean $\to$ Port Checkpoint $\to$ Customs $\to$ Final Delivery*).
   - Strict visual demarcation: **Observed Facts** vs **AI Predictions** vs **Unknown Future Stages**.
   - **Operator Decision Playbook:** Actionable recommendations tailored to the shipment's risk tier.
   - **Interactive Delay Propagation Waterfall:** Disaggregates stage delays and tracks cumulative deviation.
   - **Tree SHAP Attributions:** Quantifies exact hours contributed or buffered by each feature.
   - Collapsible **Post-Delivery Historical Audit** for retrospective validation.
3. **Tab 3: 🧪 What-If Scenario Simulator & Journey Builder**
   - Sliders for Departure Delay, Sailing Delay, Port Congestion, and Weather Severity.
   - Dynamic Delay Threshold slider (12h to 72h).
   - Real-time recalculation of Predicted ETA, Delay, and Late Probability.
   - **Terminal Congestion Sensitivity Curve:** Visualizes delay trajectory as congestion scales from 0.0 to 1.0.
4. **Tab 4: 📊 Model Performance & Benchmarks**
   - Model vs Baseline performance scorecard.
   - Actual vs Predicted delay parity scatter plot with identity reference line.
   - Confusion matrix display, Out-of-Bag calibration spread, and global feature importance rankings.

---

## 13. How to Interpret Model Predictions & Metrics

### Regression Benchmark Table (Held-Out Test Set, $N_{\text{test}} = 75$, Seed 42)

| Model / Approach | Held-Out Test MAE | Held-Out Test RMSE | Test $R^2$ Score | Error Reduction vs Benchmark |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Baseline (Zero Future Delay)** | 29.168 hours | 32.163 hours | -2.708 | **+66.01% error vs AI model** |
| **Operational Heuristic Baseline** | 11.531 hours | 14.175 hours | 0.280 | Benchmark Baseline |
| **Random Forest AI Regressor** | **9.915 hours** | **12.693 hours** | **0.422** | **+14.01% improvement vs heuristic** |

- **Understanding MAE:** Mean Absolute Error is the average margin in hours between the predicted and actual final delivery timestamps. The Naive baseline produces an unacceptably high error of 29.17 hours because it assumes downstream stages experience zero delay. The Random Forest model reduces this error to **9.915 hours**, capturing 42.2% of downstream delay variance.

### Classification Performance ($> 24.0$h Delay Threshold)
- **Accuracy:** `84.0%` (63 of 75 test shipments correctly classified).
- **Precision:** `87.3%` (87.3% of late-flagged shipments are genuinely late).
- **Recall (Sensitivity):** `95.4%` (captures 62 of 65 severely late shipments).
- **F1-Score:** `0.912` | **ROC-AUC:** `0.780`.
- **Confusion Matrix:** `[[1, 9], [3, 62]]` (TN: 1, FP: 9, FN: 3, TP: 62).
- **Decision Trade-Off:** The classifier operating point prioritizes high sensitivity (95.4% recall) because missing a severely delayed shipment (False Negative) causes missed consignee delivery windows and demurrage, whereas checking an on-time shipment (False Positive) incurs negligible operational cost.

### Uncertainty Interval Calibration
- **Residual Quantiles:** Derived from Out-of-Bag training residuals ($q_{10} = -14.94\text{h}, q_{90} = +16.76\text{h}$).
- **Nominal Target:** 80.0% | **Measured Held-Out Test Coverage:** **74.7%**.
- **Reporting Stance:** Labeled **exploratory uncertainty intervals** due to heteroscedastic terminal queueing variance across extreme congestion levels.

---

## 14. Audited Worked Numerical Shipment Example (`SHP-0001`)

Verified directly from [`data/shipments.csv`](data/shipments.csv) and [`predict_single`](src/predict.py):

| Element | Parameter / Value | Description & Formula |
| :--- | :--- | :--- |
| **Voyage Profile** | `SHP-0001` · CMA CGM Group · *Nordic Horizon* | Route: *Shanghai to Hamburg* via *Port Delta (Dubai)* |
| **Scheduled Departure** | `2025-03-15 12:00:00 UTC` | Initial planned voyage departure |
| **Planned Stage Durations** | Sailing: 352.98h, Port: 29.31h, Customs: 19.72h, Final: 18.15h | Planned Total Journey: **420.16 hours** |
| **Scheduled ETA** | `2025-04-02 00:09:36 UTC` | $\text{Departure} + 420.16\text{h}$ |
| **Checkpoint Observations** | Dep Delay: $+3.35\text{h}$, Sailing Delay: $+17.78\text{h}$ | **Observed Upstream Delay:** $\mathbf{+21.13\text{ hours}}$ |
| **Terminal Signals** | Port Congestion: $0.228$ (favorable), Weather: $0.667$ | Environmental signals at pilot station arrival |
| **Predicted Total Delay** | $\mathbf{+41.59\text{ hours}}$ | Regressor point prediction |
| **Predicted Remaining Delay**| $\mathbf{+20.46\text{ hours}}$ | $\text{Predicted Delay} - \text{Observed Delay}$ ($41.59 - 21.13$) |
| **Predicted Final ETA** | $\mathbf{2025\text{-}04\text{-}03\text{ }17:45:00\text{ UTC}}$ | $\text{Scheduled ETA} + 41.59\text{h}$ [Exact Match] |
| **Uncertainty Interval** | $[+26.65\text{h}, +58.35\text{h}]$ | $[\text{03 Apr 02:48 UTC}, \text{04 Apr 10:30 UTC}]$ |
| **Late Risk Assessment** | Probability: **95.9%** $\to$ **CRITICAL RISK** | Exceeds 24h threshold with high confidence |
| **Tree SHAP Attributions** | Upstream Delay: $+5.19\text{h}$, Sailing Delay: $+3.39\text{h}$ | Pushed ETA later; favorable congestion ($-5.96\text{h}$) buffered delay |
| **Post-Delivery Actuals** | Port: $+11.47\text{h}$, Customs: $+5.60\text{h}$, Final: $+0.95\text{h}$ | **Actual Total Delay:** $\mathbf{+39.15\text{ hours}}$ |
| **Actual Delivery Timestamp**| $\mathbf{2025\text{-}04\text{-}03\text{ }15:18:36\text{ UTC}}$ | Recorded post-delivery for evaluation only |
| **Prediction Absolute Error**| $\mathbf{2.44\text{ hours}}$ | $|41.59 - 39.15| = 2.44\text{ hours}$ (Accurate within 2.5 hours) |

---

## 15. Automated Tests & Reproducibility

The repository maintains 13 automated unit tests in [`tests/test_pipeline.py`](tests/test_pipeline.py):

```powershell
python -m pytest -v
```

- `test_generator_size_and_targets`: Validates 50 synthetic records, target columns, positive delays.
- `test_features_exclude_future_target`: Validates exact 13 allowlisted features and zero target leakage.
- `test_forbidden_columns_leakage_rejection`: Confirms all 14 forbidden columns are stripped if present.
- `test_eta_arithmetic_consistency`: Confirms scheduled arrival and actual delivery timestamps match mathematical formulas.
- `test_dataset_validation_rejects_negative_delays`: Verifies dataset validator catches negative durations.
- `test_dataset_validation_rejects_double_counting`: Verifies validator catches stage sum discrepancies.
- `test_explanation_mentions_uncertainty`: Confirms statistical disclaimer presence in narrative explanations.
- `test_prediction_dataframe_confidence_intervals`: Validates batch prediction intervals, probability bounds, and risk tiers.
- `test_single_prediction_and_attributions`: Confirms Tree SHAP attributions sum to decision-tree prediction delta.
- `test_stage_waterfall_structure`: Validates 5-stage waterfall disaggregation.
- `test_baseline_and_model_evaluation_metrics`: Verifies AI Model MAE < Heuristic MAE, confusion matrix sum ($N=75$), and interval coverage.
- `test_predicted_eta_exact_arithmetic`: Confirms $\text{predicted\_eta} == \text{scheduled\_arrival} + \text{predicted\_delay\_hours}$.
- `test_shp0001_worked_example_matches_report`: Confirms exact reproduction of worked example `SHP-0001` metrics and error ($2.44\text{h}$).

---

## 16. Comprehensive Troubleshooting Guide

| Issue / Symptom | Probable Cause | Verified Resolution |
| :--- | :--- | :--- |
| **PowerShell script activation disabled** | Windows Execution Policy blocks `.ps1` scripts | Run using direct executable: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current session only. |
| **`python app.py` prints warning** | Running Streamlit bare | `app.py` has an auto-launcher that delegates directly. Alternatively run: `streamlit run app.py` or `python -m streamlit run app.py`. |
| **Port 8501 already in use** | An existing Streamlit server is running | Launch on an alternate port: `streamlit run app.py --server.port 8502`. |
| **Missing or stale model artifacts** | `artifacts/eta_model.joblib` not found | Regenerate fresh artifacts: `python -m src.data_generator && python -m src.train`. |
| **`ModuleNotFoundError: No module named 'src'`** | Invoking script without module flag from root | Always execute from repository root using module syntax: `python -m src.train` instead of `python src/train.py`. |
| **scikit-learn pickle version mismatch** | Environment has a different scikit-learn version | Ensure `scikit-learn>=1.4` is installed via `pip install -r requirements.txt` and retrain via `python -m src.train`. |
| **Python version incompatible** | Python < 3.10 | Install Python 3.10, 3.11, 3.12, 3.13, or 3.14 and recreate virtual environment. |

---

## 17. Realistic Staged Future Development Roadmap

The proposed future roadmap for transitioning from this verified prototype to an enterprise production deployment is structured across five sequential phases:

```
[Phase 1: Verified Prototype] ──► [Phase 2: Historical Validation] ──► [Phase 3: Operational Ingestion]
                                                                                │
[Phase 5: Production & Drift] ◄── [Phase 4: Decision Support & Conformal] ◄─────┘
```

### Phase 1 — Reliable Prototype *(Completed)*
- **Objective:** Establish clean event-time mechanics, zero-leakage checkpoint features, reproducible baseline comparisons, and unit test suite.
- **Completion Criteria:** Passing automated test suite (13/13 unit tests), verified mathematical arithmetic, and audited 3-page technical report.

### Phase 2 — Real-World Validation *(Proposed)*
- **Objective:** Ingest permitted historical completed-shipment records from carrier EDI feeds (204/214/315 carrier status messages).
- **Completion Criteria:** Establish consistent historical milestone timestamps, evaluate model on 6-month temporal rolling holdout data, and verify that error reduction over operational heuristic baselines exceeds 10% across major trade lanes.

### Phase 3 — Operational Data Feeds *(Proposed)*
- **Objective:** Connect live satellite AIS GPS feeds (Spire/MarineTraffic), ocean weather forecasts (NOAA/Copernicus), and Terminal Operating System (TOS) EDI messages.
- **Completion Criteria:** Dynamic automated feature recomputation as vessels pass intermediate waypoints, with graceful missing-data fallback and API latency under 500ms.

### Phase 4 — Decision Support & Conformal Calibration *(Proposed)*
- **Objective:** Implement split conformal prediction for distribution-free interval coverage guarantees, tune risk thresholds based on consignee demurrage cost functions, and record ETA revision histories.
- **Completion Criteria:** Empirical prediction interval coverage achieving $\ge 80.0\%$ guaranteed coverage across all congestion sub-populations, with pilot operator alert feedback.

### Phase 5 — Production Readiness & Governance *(Proposed)*
- **Objective:** Deploy containerized prediction microservices with automated model registry versioning, data drift monitoring (Evidently/MLflow), and role-based access control.
- **Completion Criteria:** Automated alert triggers upon population stability index (PSI) $> 0.20$, zero service downtime during model updates, and SOC2/ISO-compliant audit trails.

---

## 18. Optional 5-Minute Demonstration Walkthrough Plan

If presenting the project in a 5-minute screencast, follow this concise agenda:

- **0:00 – 0:30 | Problem Statement & Practical Motivation:**
  - Introduce the business cost of maritime delays (anchorage congestion, customs dwell, inland demurrage).
  - Define the prediction goal at the Port Arrival Checkpoint.
- **0:30 – 1:15 | Synthetic 5-Stage Delay Propagation:**
  - Explain the 5 journey stages (*Departure $\to$ Sailing $\to$ Port Dwell $\to$ Customs $\to$ Final Delivery*).
  - Clarify the difference between incremental stage delay ($\Delta_s$) and cumulative schedule deviation ($C_k$).
- **1:15 – 2:15 | Checkpoint Observations (Shipment Deep-Dive):**
  - Open Tab 2 in the dashboard and select shipment `SHP-0001`.
  - Highlight observed facts available at checkpoint: Scheduled Arrival (`02 Apr 00:09 UTC`), Observed Upstream Delay (`+21.13h`), Port Congestion ($0.228$), Weather Severity ($0.667$).
- **2:15 – 3:15 | Predictions, Tree SHAP & Operator Playbook:**
  - Showcase Predicted ETA (`03 Apr 17:45 UTC`), Predicted Delay (`+41.59h`), Uncertainty Interval ($[26.65\text{h}, 58.35\text{h}]$), and Critical Risk badge (95.9%).
  - Explain Tree SHAP attribution chart (upstream delay pushed ETA later; low congestion buffered delay).
  - Review the Operator Action Playbook recommendations (consignee notification, priority berthing, customs pre-clearing).
- **3:15 – 4:15 | Model Benchmarks & Parity Plot:**
  - Switch to Tab 4 to show the evaluation table: Model MAE (9.915h) vs Operational Heuristic (11.531h) vs Naive (29.168h).
  - Highlight 84% accuracy, 95.4% recall, confusion matrix, and parity scatter plot.
- **4:15 – 5:00 | Limitations & Production Roadmap:**
  - Acknowledge synthetic data assumptions.
  - Summarize Phase 2–5 roadmap (real AIS telemetry, conformal intervals, temporal backtesting).

*(Note: In accordance with academic integrity guidelines, no pre-recorded video link is fabricated; this outline provides the exact structure for recording).*

---

## 19. Final Submission Deliverables

| Deliverable | File Path | Verification Status |
| :--- | :--- | :---: |
| **Interactive Streamlit Dashboard** | [`app.py`](app.py) | Verified (`http://localhost:8501`) |
| **Comprehensive Technical Report (Markdown)** | [`reports/TECHNICAL_REPORT.md`](reports/TECHNICAL_REPORT.md) | Verified (6 exact sections) |
| **Executive 3-Page Printable PDF Report** | [`reports/TECHNICAL_REPORT.pdf`](reports/TECHNICAL_REPORT.pdf) | Verified (Exactly 3 pages) |
| **Executive Brief & Reference Template** | [`reports/TECHNICAL_REPORT_TEMPLATE.md`](reports/TECHNICAL_REPORT_TEMPLATE.md) | Verified |
| **Evaluation Metrics JSON** | [`artifacts/eta_model.metrics.json`](artifacts/eta_model.metrics.json) | Verified ($N_{\text{test}}=75$) |
| **Trained ML Models Bundle** | [`artifacts/eta_model.joblib`](artifacts/eta_model.joblib) | Verified |
| **Validated Synthetic Dataset (300 rows)** | [`data/shipments.csv`](data/shipments.csv) | Verified (Zero double-counting) |
| **Automated Unit Test Suite (13 tests)** | [`tests/test_pipeline.py`](tests/test_pipeline.py) | Verified (13/13 passing) |
| **PDF Generation Script** | [`src/generate_pdf_report.py`](src/generate_pdf_report.py) | Verified |

---

*Repository audited, verified, and submitted by Kalyan Dasari · October 2026.*
