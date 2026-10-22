# 🎙️ 5-Minute Demonstration Walkthrough Plan (Private Preparation Notes)

Use this structured outline to rehearse and record your 5-minute video screencast:

---

## ⏱️ Timeline & Key Talking Points

### 0:00 – 0:30 | Problem Statement & Practical Motivation
- **The Challenge:** In maritime container shipping, multi-stage disruptions compound. A small delay during vessel loading or ocean sailing triggers missed berthing windows, port congestion, customs clearance queues, and missed inland rail/truck connections.
- **The Solution:** The Shipment ETA Intelligence system predicts final delivery ETA, delay magnitude, and delay-exceedance probability while the voyage is in-progress at the **Port Arrival Checkpoint**.

### 0:30 – 1:15 | Five Journey Stages & Synthetic Delay Propagation
- **Journey Architecture:** 5 discrete stages (*Departure handling $\to$ Ocean sailing $\to$ Port dwell & offloading $\to$ Customs clearance $\to$ Inland final delivery*).
- **Mathematical Distinction:** Differentiate between **incremental stage delay** ($\Delta_s = D_s^{\text{actual}} - D_s^{\text{planned}} \ge 0$) and **cumulative schedule deviation** ($C_k = \sum_{s=1}^k \Delta_s$). Emphasize that they are never added together as independent quantities (zero double-counting).
- **Synthetic Data Disclaimer:** State clearly that data and dynamics are simulated for demonstration and do not establish empirical real-world carrier accuracy.

### 1:15 – 2:15 | Checkpoint Observations (Shipment Root-Cause Deep-Dive)
- Open **Tab 2 (🔍 Shipment Root-Cause Deep-Dive)** in the dashboard and select shipment `SHP-0001`.
- **Observed Checkpoint Facts (Known):**
  - Scheduled Arrival: `2025-04-02 00:09:36 UTC`
  - Observed Upstream Delay: `+21.13 hours` ($3.35\text{h}$ departure delay + $17.78\text{h}$ sailing delay).
  - Terminal Indicators: Port Congestion $= 0.228$ (favorable), Weather Severity $= 0.667$ (rough sea state).
- Point out the visual stage progression stepper and clear boundary separating observed facts from unknown downstream stages.

### 2:15 – 3:15 | Predictions, Tree SHAP Attributions & Operator Action Playbook
- **Model In-Flight Predictions:**
  - Predicted Final ETA: `2025-04-03 17:45:00 UTC`
  - Predicted Total Delay: `+41.59 hours` (Estimated downstream remaining delay: `+20.46 hours`).
  - Uncertainty Interval: $[+26.65\text{h}, +58.35\text{h}]$ | Late Probability: $95.9\%$ $\to$ **CRITICAL RISK**.
- **Tree SHAP Attributions:** Show the feature impact chart—cumulative upstream delay pushed ETA later by $+5.19\text{h}$ and sailing delay by $+3.39\text{h}$, while low port congestion buffered delay by $-5.96\text{h}$.
- **Operator Decision Playbook:** Review the tailored interventions (consignee notification, priority crane berthing, customs pre-lodging, drayage re-booking).
- **Post-Delivery Audit (Collapsible Expander):** Expand to show ground truth actuals recorded after delivery: actual total delay was `+39.15 hours` and absolute error was just `2.44 hours`.

### 3:15 – 4:15 | Model Benchmarks & Parity Plot
- Switch to **Tab 4 (📊 Model Performance & Benchmarks)**.
- **Benchmark Scorecard:**
  - Random Forest Regressor: **MAE 9.915 h**, RMSE 12.693 h, $R^2$ 0.422.
  - Operational Heuristic: MAE 11.531 h (**+14.01% error reduction** by ML model).
  - Naive Baseline (Zero Future Delay): MAE 29.168 h (**+66.01% error reduction** by ML model).
- **Classification Performance:** Accuracy: 84.0%, Precision: 87.3%, Recall: 95.4%, F1: 0.912, ROC-AUC: 0.780.
- Highlight the confusion matrix (`[[1, 9], [3, 62]]`) and explain that high recall (95.4%) protects consignees from unexpected delays.
- Show the Actual vs. Predicted parity scatter plot and explain Out-of-Bag prediction interval test coverage (74.7%).

### 4:15 – 5:00 | Limitations & Production Roadmap
- **Limitations:** Acknowledge synthetic formulas and discrete port-arrival checkpointing.
- **Roadmap:** Summarize transition to production (historical carrier EDI validation, live AIS GPS satellite feeds, split conformal prediction for distribution-free uncertainty guarantees, and temporal drift monitoring).
- **Closing:** Conclude on how in-flight prediction transforms supply chain resilience.
