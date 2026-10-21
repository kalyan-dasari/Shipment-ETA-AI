# Technical Report: Predicting Shipment ETA and Delay Propagation Using AI

**Candidate:** Kalyan Dasari  
**System:** Shipment ETA Intelligence  
**Dataset:** 300 Synthetic Shipment Journeys (Seed: 42)  
**Prediction Checkpoint:** Port Arrival  
**Evaluation Methodology:** 25% Stratified Held-out Test Split  

---

## Executive Summary

Global supply chains suffer from compounded schedule disruptions where minor delays at early transit stages propagate and amplify into severe downstream delays at port terminals and customs clearance. 

This project implements an end-to-end **Shipment ETA Intelligence System** designed to:
1. **Predict Final Delivery ETA**: Estimate expected arrival timestamp using scheduled voyage timelines and journey signals observed up to the port arrival checkpoint.
2. **Estimate Total and Remaining Delay**: Accurately predict final delivery delay in hours alongside calibrated confidence intervals.
3. **Quantify Delay-Exceedance Risk**: Compute the calibrated probability that a shipment exceeds an operational delay threshold (default: 24 hours), categorizing shipments into actionable risk tiers (*Low, Moderate, High, Critical*).
4. **Explain Delay Propagation**: Reveal how upstream departure friction, ocean transit conditions, port terminal congestion, and customs bottlenecks compound across 5 distinct journey stages via exact Tree SHAP feature attributions and stage waterfalls.
5. **Interactive Operational Dashboard**: Deliver a modern decision-support tool featuring real-time fleet monitoring, root-cause diagnostics, and a What-If scenario simulator.

---

## 1. Problem Formulation and Prediction Checkpoint

### 1.1 The Prediction Checkpoint: Port Arrival
In maritime container logistics, predictions made too early (e.g., at booking) lack operational fidelity, while predictions made after customs clearance offer little mitigation value. This project models the critical **Port Arrival Checkpoint**:
- The vessel has traversed ocean sailing and arrived at the outer anchorage / pilot station of the destination port.
- Upstream historical actuals (*departure time*, *ocean transit duration*, *observed sailing delay*) are known with certainty.
- Downstream stages (*berth waiting*, *crane offloading*, *customs clearance inspection*, *final inland drayage*) have not occurred and remain uncertain.

### 1.2 Mathematical Formulation
For each shipment $i$:
$$\text{Scheduled Arrival}_i = \text{Scheduled Departure}_i + \sum_{s \in \text{Stages}} \text{Planned Duration}_{i, s}$$
$$\text{Actual Total Delay}_i = \sum_{s \in \text{Stages}} \text{Stage Delay}_{i, s}$$
$$\text{Predicted Final ETA}_i = \text{Scheduled Arrival}_i + \widehat{\text{Delay}}_i$$

Where $\widehat{\text{Delay}}_i$ is predicted by the machine learning engine from features $X_i^{\text{port}}$ available at the port checkpoint.

A shipment is classified as **Late** if:
$$y_{\text{late}, i} = \mathbb{I}(\text{Actual Total Delay}_i > \tau)$$
where $\tau = 24.0\text{ hours}$ (configurable in the operational dashboard).

---

## 2. Synthetic Data Generation and Delay Propagation Dynamics

The dataset comprises 300 synthetic shipments modeled across realistic international trade lanes (e.g., *Shanghai–Rotterdam*, *Singapore–Sydney*, *Mumbai–Dubai*, *Los Angeles–Hamburg*), major shipping carriers (*Maersk, MSC, CMA CGM, Hapag-Lloyd, ONE*), and primary container terminal hubs.

### 2.1 The 5-Stage Propagation Cascade
Disruptions do not occur independently; upstream delays compound downstream friction. The simulation models this linked cascade:

1. **Origin Departure:**
   $$\text{Delay}_{\text{dep}} = \max\left(0, \mathcal{N}(2.0 + 8.5 \cdot \text{Weather}, 4.5^2)\right)$$
2. **Ocean Sailing Transit:**
   $$\text{Delay}_{\text{sail}} = \max\left(0, \mathcal{N}(3.0 + 13.5 \cdot \text{Weather} + 0.15 \cdot \text{Delay}_{\text{dep}}, 6.0^2)\right)$$
   *Ocean transit delay is driven by heavy weather severity and exacerbated by initial departure delays.*
3. **Port Terminal Congestion & Dwell (Checkpoint):**
   $$\text{Observed Delay at Port} = \text{Delay}_{\text{dep}} + \text{Delay}_{\text{sail}}$$
   $$\text{Delay}_{\text{port}} = \max\left(0, \mathcal{N}(2.0 + 26.0 \cdot \text{Congestion} + 0.22 \cdot \text{Delay}_{\text{sail}}, 7.5^2)\right)$$
   *When a ship arrives late, it misses its reserved berthing window, triggering severe queueing when port congestion is elevated.*
4. **Customs Clearance:**
   $$\text{Delay}_{\text{customs}} = \max\left(0, \mathcal{N}(1.5 + 13.0 \cdot \text{Congestion} + 0.18 \cdot \text{Delay}_{\text{port}}, 5.0^2)\right)$$
   *Port terminal congestion creates container yard backlogs that slow physical customs inspection.*
5. **Final Inland Delivery:**
   $$\text{Delay}_{\text{final}} = \max\left(0, \mathcal{N}(1.0 + 0.12 \cdot \text{Delay}_{\text{customs}} + 3.0 \cdot \text{Weather}, 4.0^2)\right)$$
   *Final mile truck/rail connections depend on customs release timing and localized transit conditions.*

---

## 3. Feature Engineering and Leakage Prevention

### 3.1 Checkpoint Feature Set ($X^{\text{port}}$)
To prevent target leakage, all future timestamps and downstream actuals are strictly withheld:

| Feature Name | Category | Description |
| :--- | :--- | :--- |
| `planned_sailing_hours` | Schedule | Scheduled duration of ocean leg (72–360h) |
| `planned_port_hours` | Schedule | Scheduled dwell for berth and unloading (8–36h) |
| `planned_customs_hours` | Schedule | Scheduled clearance window (4–24h) |
| `planned_final_hours` | Schedule | Scheduled inland transit leg (12–60h) |
| `planned_remaining_hours_at_port` | Schedule | Total buffer remaining from port arrival to delivery |
| `departure_delay_hours` | Observed Upstream | Incurred origin terminal delay |
| `sailing_delay_hours` | Observed Upstream | Incurred sea transit delay |
| `observed_delay_at_port_hours` | Observed Upstream | Total delay accumulated upon arrival at port ($\text{dep} + \text{sail}$) |
| `port_congestion` | Environmental | Simulated terminal congestion index ($0.0–1.0$) |
| `weather_severity` | Environmental | Simulated route meteorological severity ($0.0–1.0$) |
| `upstream_delay_ratio` | Engineered | $\text{Observed Delay} / (\text{Planned Sailing} + \epsilon)$ |
| `congestion_weather_interaction` | Engineered | $\text{Port Congestion} \times \text{Weather Severity}$ |
| `congestion_buffer_ratio` | Engineered | $\text{Port Congestion} / (\text{Planned Port Hours} + \epsilon)$ |

### 3.2 Target Leakage Audit
The feature extraction pipeline (`src/features.py`) enforces strict validation against leakage:
- Excluded: `actual_port_departure`, `port_delay_hours`, `actual_customs_done`, `customs_delay_hours`, `final_stage_delay_hours`, `actual_delivery`, and `actual_total_delay_hours`.

---

## 4. Modeling Architecture and Operational Baselines

The prediction engine consists of three complementary components:

1. **Operational Baselines:**
   - **Naive Baseline:** Assumes zero downstream delay ($\widehat{\text{Delay}} = \text{Observed Delay at Port}$).
   - **Operational Heuristic Baseline:** Adds the historical median downstream residual delay learned exclusively from the training set ($\widehat{\text{Delay}} = \text{Observed Delay} + \text{Median}(\text{Residual}_{\text{train}})$).
2. **AI Delay Regressor:**
   - Random Forest Regressor (300 estimators, minimum samples leaf = 2, max features = $\sqrt{p}$) trained to predict total final delivery delay in hours.
   - Non-parametric residual quantile estimation ($q_{10}, q_{90}$) to generate 90% confidence intervals $[\widehat{\text{Delay}}_{\text{lower}}, \widehat{\text{Delay}}_{\text{upper}}]$.
3. **Calibrated Delay-Risk Classifier:**
   - Random Forest Classifier with probability calibration to estimate $P(\text{Delay} > 24\text{ hours})$.

---

## 5. Quantitative Evaluation Results

The models were evaluated on a held-out test split (25% held-out test set, $N_{\text{test}} = 75$ shipments; seed 42). Metrics are derived directly from model training artifacts (`artifacts/eta_model.metrics.json`).

### 5.1 Delay Prediction Performance (Regression)

| Model / Approach | MAE (Hours) | RMSE (Hours) | $R^2$ Score | Error Reduction vs Heuristic |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Baseline (Zero Downstream Delay)** | 29.17 h | 31.84 h | - | **-66.1% error** |
| **Operational Heuristic Baseline** | 11.53 h | 14.18 h | 0.281 | Benchmark |
| **Random Forest AI Regressor** | **9.89 h** | **12.67 h** | **0.425** | **+14.3% improvement** |

#### Key Takeaways:
- The **Naive Baseline** incurs an unacceptable MAE of **29.17 hours**, proving that ignoring downstream port and customs propagation leads to severe operational underestimation.
- The **Operational Heuristic** improves to 11.53 hours MAE by factoring in typical residual delay, but cannot distinguish between congested vs uncongested terminals.
- The **Random Forest AI Regressor** achieves **9.89 hours MAE** and **12.67 hours RMSE**, achieving a **14.3% error reduction** over the operational heuristic and explaining over 42.5% of remaining delay variance.

### 5.2 Delay-Risk Classification Performance ($> 24$h Threshold)

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **84.0%** | Correctly classifies 84 out of 100 test shipments |
| **Precision** | **87.3%** | When flagged as late, 87.3% of shipments actually arrive late |
| **Recall (Sensitivity)** | **95.4%** | Captures 95.4% of all late shipments (critical for avoiding supply disruptions) |
| **F1-Score** | **0.912** | Harmonious balance between false alarms and missed delays |
| **ROC-AUC** | **0.780** | Robust discrimination across varying decision thresholds |

### 5.3 Global Feature Importance (Top Predictors)

1. `port_congestion` (**21.37%**): The single dominant driver of downstream queueing and dwell variance.
2. `observed_delay_at_port_hours` (**14.38%**): Accumulated upstream delay forms the non-negotiable floor for total delay.
3. `congestion_weather_interaction` (**13.44%**): Highlights the compounding effect when high port congestion meets adverse weather.
4. `sailing_delay_hours` (**9.60%**): Ocean transit delay directly impacts berth slot allocations.
5. `upstream_delay_ratio` (**7.77%**): Relative severity of delay versus planned sailing duration.
6. `congestion_buffer_ratio` (**7.48%**): Congestion pressure exceeding planned terminal buffer.

---

## 6. Explainability Framework

To build trust with logistics dispatchers and supply chain planners, the system implements a three-tier explainability architecture:

### 6.1 Exact Tree SHAP Feature Attribution
For any individual shipment, the model computes the exact contribution of each feature using decision-tree path decomposition:
$$\widehat{y} = \bar{y}_{\text{train}} + \sum_{j=1}^{p} \phi_j$$
Where $\phi_j$ represents the exact hours added or subtracted by feature $j$ relative to the fleet average. For example, a shipment with $0.85$ port congestion gains $+8.4\text{ hours}$ attributable specifically to terminal queueing, while a generous planned port buffer contributes $-2.1\text{ hours}$.

### 6.2 Delay Propagation Waterfall
The interactive dashboard decomposes the journey into 5 stages, contrasting planned duration, observed delay, and expected downstream delay. Dispatchers can visually pinpoint whether an ETA slip originated at origin departure or accumulated during port dwell.

### 6.3 What-If Scenario Sensitivity
Users can simulate operational interventions in real time (e.g., *"What if priority berthing reduces congestion from 0.85 to 0.30?"*). The system updates the predicted ETA, re-computes confidence bounds, and dynamically renders the congestion sensitivity curve.

---

## 7. Limitations and Production Roadmap

### 7.1 Limitations of the Prototype
1. **Synthetic Data**: The delay distributions and propagation coefficients reflect mathematical simulation assumptions rather than empirical carrier telemetry.
2. **Simplified Stationarity**: Real-world shipping exhibits seasonal surges (e.g., Golden Week, peak holiday shipping), vessel blank sailings, and transshipment hub transfers not modeled here.
3. **No Real-Time Telemetry**: Uses static indicator values ($0.0–1.0$) rather than live AIS GPS coordinates, terminal operating systems (TOS) EDI messages, or weather radar forecasts.

### 7.2 Production Implementation Roadmap
- **Live Ingestion**: Ingest live AIS feeds (Spire / MarineTraffic), satellite weather (Copernicus / NOAA), and port berth schedules.
- **Time-Based Validation**: Replace random cross-validation with rolling temporal backtesting (train on months $1..K$, evaluate on month $K+1$) to monitor temporal drift.
- **Conformal Prediction**: Integrate inductive conformal prediction to ensure exact, distribution-free coverage guarantees on ETA prediction intervals.
- **Transshipment Graph Modeling**: Extend stage propagation to multi-leg container routing via Graph Neural Networks (GNNs) across transshipment networks.

---

*Report generated and validated with code execution on 21 October 2026.*
