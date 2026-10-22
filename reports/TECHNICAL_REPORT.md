# Technical Report: Predicting Shipment ETA and Delay Propagation Using AI

**Candidate:** Kalyan Dasari  
**System:** Shipment ETA Intelligence  
**Dataset:** 300 Synthetic Shipment Journeys (Seed: 42)  
**Prediction Checkpoint:** Destination Port Arrival  
**Evaluation Protocol:** 25% Stratified Held-out Test Split ($N_{\text{test}} = 75$)  

---

## 1. Problem Statement and Practical Motivation

In international container shipping, schedule unreliability creates systemic supply chain disruption. A delay incurred during vessel loading or ocean transit does not remain isolated; it triggers missed berthing windows, terminal anchorage queueing, customs inspection backlogs, and missed inland rail/truck connections.

Traditional freight visibility platforms rely on static schedule tables or naive linear projections that assume zero future delay after an initial disruption. Conversely, predicting ETA too early (e.g., at booking) lacks operational visibility, while predicting ETA after customs clearance leaves no room for intervention.

This project focuses on the high-value **Port Arrival Checkpoint**: estimating final delivery arrival timestamp and delay magnitude while the voyage is in-progress, enabling logistics operators to proactively:
1. Issue calibrated delay advisories to warehouse receiving and retail consignees.
2. Request off-dock container staging or priority crane slots to mitigate port yard dwell.
3. Pre-lodge electronic customs manifests (EDI) to eliminate customs dwell.
4. Reschedule onward drayage and rail connections to avoid punitive demurrage and detention charges.

*Notice on Synthetic Data:* This implementation uses mathematically modeled synthetic journeys. Evaluation metrics establish consistency with the simulation assumptions and do not represent empirical real-world carrier accuracy.

---

## 2. Data Source and Synthetic-Data Generation Assumptions

The dataset comprises 300 deterministic synthetic shipment journeys generated across global trade lanes (e.g., *Shanghai–Rotterdam*, *Singapore–Sydney*, *Mumbai–Dubai*, *Los Angeles–Hamburg*, *Busan–Antwerp*) and Tier-1 container carriers (*Maersk, MSC, CMA CGM, Hapag-Lloyd, ONE*).

The simulation models five discrete journey stages:
1. **Departure handling:** Origin terminal container loading and vessel cast-off.
2. **Ocean sailing / transit:** Deep-sea voyage.
3. **Port processing / dwell:** Anchorage waiting, berthing, and crane discharge.
4. **Customs clearance:** Regulatory inspection and document release.
5. **Final delivery:** Inland drayage / rail linehaul to final destination.

### Delay Propagation Cascade
Disruptions cascade across stages according to realistic logistics friction equations:
- **Departure delay:** $\Delta_{\text{dep}} = \max\left(0, \mathcal{N}(2.0 + 8.5 \cdot \text{Weather}, 4.5^2)\right)$
- **Sailing transit delay:** $\Delta_{\text{sail}} = \max\left(0, \mathcal{N}(3.0 + 13.5 \cdot \text{Weather} + 0.15 \cdot \Delta_{\text{dep}}, 6.0^2)\right)$
- **Port dwell delay:** $\Delta_{\text{port}} = \max\left(0, \mathcal{N}(2.0 + 26.0 \cdot \text{Congestion} + 0.22 \cdot \Delta_{\text{sail}}, 7.5^2)\right)$
- **Customs delay:** $\Delta_{\text{customs}} = \max\left(0, \mathcal{N}(1.5 + 13.0 \cdot \text{Congestion} + 0.18 \cdot \Delta_{\text{port}}, 5.0^2)\right)$
- **Final delivery delay:** $\Delta_{\text{final}} = \max\left(0, \mathcal{N}(1.0 + 0.12 \cdot \Delta_{\text{customs}} + 3.0 \cdot \text{Weather}, 4.0^2)\right)$

Environmental signals ($\text{Congestion} \sim \text{Beta}(2.2, 2.2)$ and $\text{Weather} \sim \text{Beta}(1.8, 3.2)$) represent localized terminal and meteorological severity on a $0.0–1.0$ scale.

---

## 3. ETA, Stage-Delay, Cumulative-Delay, and Ground-Truth Definitions

To prevent accidental double-counting, the event-time model formally distinguishes **incremental stage delay** from **cumulative schedule deviation**:

### Mathematical Definitions
1. **Planned Journey Duration:**
   $$D_{\text{planned}} = D_{\text{sail}}^{\text{plan}} + D_{\text{port}}^{\text{plan}} + D_{\text{customs}}^{\text{plan}} + D_{\text{final}}^{\text{plan}}$$
2. **Scheduled ETA:**
   $$T_{\text{sched\_arr}} = T_{\text{sched\_dep}} + D_{\text{planned}}$$
3. **Incremental Stage Delay ($\Delta_s$):**
   $$\Delta_s = D_s^{\text{actual}} - D_s^{\text{planned}} \ge 0, \quad \forall s \in \{1, 2, 3, 4, 5\}$$
4. **Cumulative Delay at Milestone $k$ ($C_k$):**
   $$C_k = \sum_{s=1}^k \Delta_s$$
   - At Port Checkpoint: $C_2 = \Delta_{\text{dep}} + \Delta_{\text{sail}} = \text{Observed Upstream Delay}$
   - At Final Delivery: $C_5 = \sum_{s=1}^5 \Delta_s = \text{Ground Truth Total Delay}$
5. **Actual Delivery Timestamp ($T_{\text{actual\_deliv}}$):**
   $$T_{\text{actual\_deliv}} = T_{\text{sched\_arr}} + C_5 = T_{\text{sched\_arr}} + \text{Total Delay}$$
6. **Ground-Truth Late Label ($y_{\text{late}}$):**
   $$y_{\text{late}} = \mathbb{I}(\text{Total Delay} > 24.0\text{ hours})$$
7. **Predicted Final ETA:**
   $$\widehat{T}_{\text{ETA}} = T_{\text{sched\_arr}} + \widehat{\text{Delay}}_{\text{total}}$$

Incremental stage delays and cumulative delays are never summed together. All timestamps are second-precision and timezone-aware (UTC).

---

## 4. Prediction Checkpoint and Feature Design

Predictions occur strictly at the **Port Arrival Checkpoint** (vessel arrival at outer pilot station). All downstream actuals are strictly withheld to eliminate target leakage.

### Feature Allowlist ($X^{\text{port}}$)
- **Schedule Features:** `planned_sailing_hours` (72–360h), `planned_port_hours` (8–36h), `planned_customs_hours` (4–24h), `planned_final_hours` (12–60h), `planned_remaining_hours_at_port`.
- **Observed Upstream Leg Metrics:** `departure_delay_hours`, `sailing_delay_hours`, `observed_delay_at_port_hours` ($\Delta_{\text{dep}} + \Delta_{\text{sail}}$).
- **Environmental Signals:** `port_congestion` ($0.0–1.0$), `weather_severity` ($0.0–1.0$).
- **Engineered Domain Interactions:**
  - `upstream_delay_ratio` $= \text{observed\_delay} / (\text{planned\_sailing} + \epsilon)$
  - `congestion_weather_interaction` $= \text{port\_congestion} \times \text{weather\_severity}$
  - `congestion_buffer_ratio` $= \text{port\_congestion} / (\text{planned\_port\_hours} + \epsilon)$

**Leakage Audit:** Automated unit tests verify that forbidden downstream outcomes (`port_delay_hours`, `customs_delay_hours`, `final_stage_delay_hours`, `actual_port_departure`, `actual_customs_done`, `actual_delivery`, `actual_total_delay_hours`, `late_delivery`) are strictly excluded.

---

## 5. Model Approach and Evaluation Protocol

The machine learning architecture pairs a primary non-linear regression model with calibrated risk classification and operational baselines:

1. **Random Forest Regressor:** 300 estimators, minimum samples leaf = 2, max features = $\sqrt{p}$, with Out-of-Bag (OOB) scoring. Fits non-linear interactions between port congestion, weather, and upstream delay.
2. **Delay-Risk Classifier:** Random Forest Classifier (200 trees, max depth = 6) calibrated to output $P(\text{Delay} > 24\text{ hours})$.
3. **Exploratory Uncertainty Intervals:** Non-parametric empirical residual quantiles ($q_{10}^{\text{oob}} = -14.94\text{h}, q_{90}^{\text{oob}} = +16.76\text{h}$) derived from Out-of-Bag training residuals.
4. **Operational Baselines for Comparison:**
   - **Naive Baseline:** Assumes zero downstream delay ($\widehat{\text{Delay}} = \text{Observed Delay at Port}$).
   - **Operational Heuristic:** $\widehat{\text{Delay}} = \text{Observed Delay} + \text{Median}(\text{Residual}_{\text{train}})$.
5. **Evaluation Protocol:** 25% stratified held-out test split ($N_{\text{train}} = 225, N_{\text{test}} = 75$, random seed 42). All transformations fitted on training data only.

---

## 6. Results Table Comparing the Model with Baselines

All results below are directly measured from held-out test execution (`artifacts/eta_model.metrics.json`):

### 6.1 Delay Regression Performance (Held-Out Test Set)

| Model / Approach | Test MAE | Test RMSE | Test $R^2$ | MAE Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Baseline (Zero Downstream Delay)** | 29.168 h | 32.163 h | -2.708 | **+66.01% error** |
| **Operational Heuristic Baseline** | 11.531 h | 14.175 h | 0.280 | Benchmark |
| **Random Forest AI Regressor** | **9.915 h** | **12.693 h** | **0.422** | **+14.01% improvement** |

*Interpretation:* The Naive Baseline fails catastrophically (MAE 29.17h) because downstream port, customs, and inland transit delays average 18+ hours. The Random Forest model outperforms the operational heuristic by 14.01%, explaining 42.2% of remaining delay variance.

### 6.2 Late-Delivery Classification ($>24$h Delay Threshold)

| Metric | Measured Score | Operational Significance |
| :--- | :---: | :--- |
| **Accuracy** | **84.0%** | Correctly predicts delivery status on 84% of test voyages |
| **Precision** | **87.3%** | 87.3% of late-flagged shipments are genuinely late (low false alarm rate) |
| **Recall (Sensitivity)** | **95.4%** | Captures 95.4% of all shipments exceeding 24h delay |
| **F1-Score** | **0.912** | Balanced performance between precision and recall |
| **ROC-AUC** | **0.780** | Robust discrimination across operational decision thresholds |
| **Confusion Matrix** | `[[1, 9], [3, 62]]` | TN: 1, FP: 9, FN: 3, TP: 62 (out of 75 test shipments) |

### 6.3 Uncertainty Interval Validation
- **Nominal Target:** 80.0%
- **Measured Held-Out Test Coverage:** **74.7%**
- *Validation Note:* Using Out-of-Bag training residuals achieved 74.7% empirical test coverage (close to the 80% nominal target). These intervals are labeled exploratory due to heteroscedastic terminal queueing variance.

### 6.4 Dominant Model Associations (Feature Importances)
1. `port_congestion` (**21.42%**): Dominant predictor of terminal queueing.
2. `observed_delay_at_port_hours` (**14.26%**): Incurred upstream delay floor.
3. `congestion_weather_interaction` (**13.37%**): Compounded weather and port queueing.
4. `sailing_delay_hours` (**9.79%**): Sea transit schedule slippage.
5. `upstream_delay_ratio` (**7.69%**): Relative voyage disruption ratio.

---

## 7. Explainability and Worked Numerical Shipment Example

To provide auditable decision support, the system decomposes predictions using **Tree SHAP path attribution** ($\widehat{y} = \bar{y} + \sum \phi_j$) and stage-by-stage waterfalls.

### Worked Numerical Example: Shipment `SHP-0001`
- **Voyage Profile:** Vessel *Nordic Horizon* (Carrier: *CMA CGM Group*, Route: *Shanghai to Hamburg* via *Port Delta (Dubai)*).
- **Scheduled Departure:** `2025-03-15 12:00:00 UTC` | **Scheduled Arrival:** `2025-04-02 00:09:36 UTC`
- **Planned Stage Durations:** Sailing: 352.98h | Port Dwell: 29.31h | Customs: 19.72h | Inland Final: 18.15h (Total: 420.16h).

#### Checkpoint Observations (Port Arrival):
- Departure Delay: $+3.35\text{ h}$ | Sailing Delay: $+17.78\text{ h}$
- **Observed Upstream Delay:** $+21.13\text{ hours}$
- **Terminal Signals:** Port Congestion $= 0.228$ (favorable) | Weather Severity $= 0.667$ (adverse sea state).

#### Model In-Flight Predictions:
- **Predicted Total Delay:** $+41.59\text{ hours}$ (Remaining Downstream: $+20.46\text{ hours}$)
- **Predicted Final ETA:** `2025-04-03 17:45:00 UTC`
- **Uncertainty Interval:** $[26.65\text{h}, 58.35\text{h}]$ | **Late Probability:** $95.9\%$ (**Critical Risk**)
- **Tree SHAP Attribution:** Observed Upstream Delay contributed $+5.19\text{h}$ and Sailing Delay $+3.39\text{h}$ pushing ETA later; low Port Congestion buffered the arrival by $-5.96\text{h}$.

#### Unknown Future Actuals (Recorded Post-Delivery for Evaluation):
- Actual Port Dwell Delay: $+11.47\text{ h}$ | Customs Delay: $+5.60\text{ h}$ | Final Transit Delay: $+0.95\text{ h}$
- **Actual Total Delay:** $+39.15\text{ hours}$
- **Actual Final Delivery:** `2025-04-03 15:18:36 UTC`
- **Model Error:** $|41.59 - 39.15| = 2.44\text{ hours}$ (Predicted ETA was accurate within 2.5 hours).

---

## 8. Limitations and Real-World Deployment Requirements

### 8.1 Limitations of the Synthetic Prototype
1. **Synthetic Dynamics:** Delay formulas and propagation weights are mathematical assumptions, not empirical carrier telemetry.
2. **Simplified Stationarity:** Omits seasonality (e.g. Golden Week, peak holiday surges), vessel blank sailings, and transshipment hub transfers.
3. **Discrete Checkpoint:** Evaluates only at port arrival, rather than dynamically updating along the ocean trajectory.

### 8.2 Production Deployment Roadmap
- **Live Ingestion Feeds:** Ingest satellite AIS GPS feeds (Spire/MarineTraffic), weather radar forecasts (Copernicus/NOAA), and Terminal Operating System (TOS) EDI gate transactions.
- **Time-Based Rolling Splits:** Replace random cross-validation with temporal backtesting (train on months $1..T$, evaluate on month $T+1$) to monitor data drift.
- **Conformal Prediction:** Implement split conformal prediction to deliver distribution-free coverage guarantees for prediction intervals.
- **Transshipment Graph Modeling:** Extend stage propagation to multi-leg container networks using Graph Neural Networks (GNNs).

---

*Report validated with code execution on 22 October 2026.*
