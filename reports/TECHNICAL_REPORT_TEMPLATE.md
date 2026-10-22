# Shipment ETA Prediction and Delay Propagation — Executive Technical Brief

**Candidate:** Kalyan Dasari  
**System:** Shipment ETA Intelligence AI  
**Dataset:** 300 Synthetic Shipment Journeys (Seed: 42)  
**Prediction Checkpoint:** Destination Port Outer Anchorage / Pilot Station Arrival  
**Evaluation Protocol:** 25% Stratified Held-out Test Split ($N_{\text{train}} = 225, N_{\text{test}} = 75$)  

---

## 1. Problem Definition
The system predicts final delivery ETA and delay magnitude in hours at the **Port Arrival Checkpoint** while a maritime container shipment is in-progress. 
- **Scheduled ETA:** Scheduled Departure + Planned Total Journey Duration ($\sum_{s=1}^4 D_s^{\text{planned}}$).
- **Predicted ETA:** Scheduled ETA + Predicted Total Delay ($\widehat{T}_{\text{ETA}} = T_{\text{sched\_arr}} + \widehat{\text{Delay}}_{\text{total}}$).
- **Ground-Truth Delay:** Sum of actual incremental stage delays ($\sum_{s=1}^5 \Delta_s$).
- **Late Delivery Label:** $1$ if actual total delay exceeds $24.0$ hours; otherwise $0$.

---

## 2. Data and Assumptions
The simulation generates 300 synthetic shipments across 5 global trade lanes (*Shanghai–Rotterdam*, *Singapore–Sydney*, *Mumbai–Dubai*, *Los Angeles–Hamburg*, *Busan–Antwerp*) and 5 carriers (*Maersk, MSC, CMA CGM, Hapag-Lloyd, ONE*).

The journey comprises 5 discrete stages:
1. Departure handling ($\Delta_{\text{dep}} = \max(0, \mathcal{N}(2.0 + 8.5\cdot\text{Weather}, 4.5^2))$)
2. Ocean sailing ($\Delta_{\text{sail}} = \max(0, \mathcal{N}(3.0 + 13.5\cdot\text{Weather} + 0.15\cdot\Delta_{\text{dep}}, 6.0^2))$)
3. Port dwell & discharge ($\Delta_{\text{port}} = \max(0, \mathcal{N}(2.0 + 26.0\cdot\text{Congestion} + 0.22\cdot\Delta_{\text{sail}}, 7.5^2))$)
4. Customs clearance ($\Delta_{\text{customs}} = \max(0, \mathcal{N}(1.5 + 13.0\cdot\text{Congestion} + 0.18\cdot\Delta_{\text{port}}, 5.0^2))$)
5. Inland final transit ($\Delta_{\text{final}} = \max(0, \mathcal{N}(1.0 + 0.12\cdot\Delta_{\text{customs}} + 3.0\cdot\text{Weather}, 4.0^2))$)

*Distinction:* Incremental stage delay ($\Delta_s = D_s^{\text{actual}} - D_s^{\text{planned}} \ge 0$) and cumulative schedule deviation ($C_k = \sum_{s=1}^k \Delta_s$) are kept strictly distinct. They are never added together as independent quantities. All timestamps are second-precision UTC ISO-8601 strings.

*Notice on Synthetic Data:* Simulation dynamics are mathematical assumptions for demonstration and do not establish empirical carrier accuracy.

---

## 3. Features and Model
- **Prediction Checkpoint:** Port Arrival (outer pilot station / anchorage).
- **Feature Allowlist (13 features):**
  - Planned durations: `planned_sailing_hours`, `planned_port_hours`, `planned_customs_hours`, `planned_final_hours`, `planned_remaining_hours_at_port`.
  - Observed upstream history: `departure_delay_hours`, `sailing_delay_hours`, `observed_delay_at_port_hours` ($\Delta_{\text{dep}} + \Delta_{\text{sail}}$).
  - Environmental signals: `port_congestion` ($0.0–1.0$), `weather_severity` ($0.0–1.0$).
  - Domain interactions: `upstream_delay_ratio`, `congestion_weather_interaction`, `congestion_buffer_ratio`.
- **Target Leakage Prevention:** All 14 downstream actuals (`port_delay_hours`, `customs_delay_hours`, `final_stage_delay_hours`, actual timestamps, actual total delay, and late label) are strictly deleted before training and inference. Preprocessing imputers are fitted on training data only.
- **Model Architecture:**
  - *Primary Regressor:* Random Forest (300 trees, min_samples_leaf=2, max_features="sqrt", Out-of-Bag scoring).
  - *Risk Classifier:* Random Forest (200 trees, max_depth=6) predicting $P(\text{Delay} > 24\text{h})$.
  - *Uncertainty Intervals:* Non-parametric Out-of-Bag residual quantiles ($q_{10} = -14.94\text{h}, q_{90} = +16.76\text{h}$).

---

## 4. Evaluation and Benchmarking
Measured on the 25% stratified held-out test split ($N_{\text{test}} = 75$, `seed=42`) from [`artifacts/eta_model.metrics.json`](../artifacts/eta_model.metrics.json):

### Regression Performance vs Baselines
| Metric | Random Forest Model | Operational Heuristic | Naive Baseline (Zero Downstream Delay) |
| :--- | :---: | :---: | :---: |
| **MAE** | **9.915 hours** | 11.531 hours | 29.168 hours |
| **RMSE** | **12.693 hours** | 14.175 hours | 32.163 hours |
| **$R^2$ Score** | **0.422** | 0.280 | -2.708 |
| **MAE Error Reduction** | **+14.01% vs Heuristic** | Benchmark | **+66.01% vs Naive** |

### Delay-Risk Classification ($>24.0$h Threshold)
- **Accuracy:** `84.0%` ($63 / 75$) | **Precision:** `87.3%` ($62 / 71$) | **Recall:** `95.4%` ($62 / 65$)
- **F1-Score:** `0.912` | **ROC-AUC:** `0.780`
- **Confusion Matrix:** `[[1, 9], [3, 62]]` (TN: 1, FP: 9, FN: 3, TP: 62)
- **Uncertainty Interval Test Coverage:** `74.7%` empirical coverage against 80.0% nominal target (labeled exploratory due to heteroscedastic terminal variance).

---

## 5. Explainability and Worked Example (`SHP-0001`)
Predictions are decomposed using exact **Tree SHAP path attribution** and 5-stage delay waterfalls.

### Audited Numerical Example: `SHP-0001`
- **Profile:** CMA CGM Group, *Nordic Horizon*, Shanghai–Hamburg via Port Delta (Dubai).
- **Scheduled Departure:** `2025-03-15 12:00:00 UTC` | **Scheduled Arrival:** `2025-04-02 00:09:36 UTC` (Planned duration: 420.16h).
- **Checkpoint Observations:** Departure delay: $+3.35\text{h}$ | Sailing delay: $+17.78\text{h}$ | Observed delay: $+21.13\text{h}$ | Congestion: $0.228$ | Weather: $0.667$.
- **AI Predictions:**
  - Predicted Total Delay: $\mathbf{+41.59\text{ hours}}$ (Estimated downstream remaining: $+20.46\text{h}$)
  - Predicted Final ETA: $\mathbf{2025\text{-}04\text{-}03\text{ }17:45:00\text{ UTC}}$
  - Uncertainty Interval: $[26.65\text{h}, 58.35\text{h}]$ | Late Probability: $95.9\%$ (**Critical Risk**)
  - Attributions: Observed upstream delay ($+5.19\text{h}$) and sailing delay ($+3.39\text{h}$) increased ETA; favorable congestion ($-5.96\text{h}$) buffered delay.
- **Post-Delivery Historical Ground Truth:**
  - Actual Delays: Port: $+11.47\text{h}$ | Customs: $+5.60\text{h}$ | Final: $+0.95\text{h}$ $\to$ **Actual Total Delay: $+39.15\text{ hours}$**.
  - Actual Delivery Timestamp: `2025-04-03 15:18:36 UTC`.
  - **Prediction Absolute Error:** $|41.59 - 39.15| = \mathbf{2.44\text{ hours}}$.
  - Exact Equality Verified: $\text{Scheduled ETA} + \text{Predicted Delay} = \text{Predicted ETA}$ ($17:45:00\text{ UTC}$).

### Limitations
1. Simulation formulas reflect synthetic mechanics, not empirical telemetry.
2. Omits vessel blank sailings, transshipment feeder connections, and holiday seasonality.
3. Discrete checkpoint at port arrival rather than dynamic trajectory continuous tracking.

---

## 6. Production Roadmap & Future Work
1. **Live Data Ingestion:** Connect AIS satellite GPS (Spire/MarineTraffic), ocean weather forecasts (NOAA), and TOS terminal EDI logs.
2. **Temporal Backtesting:** Replace random cross-validation with rolling time-based splits to track drift.
3. **Conformal Prediction:** Implement split conformal prediction for distribution-free interval coverage guarantees.
4. **Network Graph Modeling:** Model complex multi-leg transshipment networks via Graph Neural Networks (GNNs).
