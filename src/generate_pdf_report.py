"""Generate a publication-quality 3-page PDF technical report from the assignment report."""
from pathlib import Path
import pymupdf

def generate_pdf():
    pdf_path = Path("reports/TECHNICAL_REPORT.pdf")
    doc = pymupdf.open()

    # CSS styles for an executive technical report
    css = """
    <style>
        @page {
            size: A4;
            margin: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            font-size: 8.5pt;
            line-height: 1.35;
            margin: 0;
            padding: 0;
        }
        h1 {
            color: #0f172a;
            font-size: 15pt;
            font-weight: 700;
            margin: 0 0 4px 0;
            padding-bottom: 3px;
            border-bottom: 2px solid #0284c7;
        }
        h2 {
            color: #0369a1;
            font-size: 10.5pt;
            font-weight: 700;
            margin: 8px 0 3px 0;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 2px;
        }
        h3 {
            color: #0f172a;
            font-size: 9pt;
            font-weight: 600;
            margin: 5px 0 2px 0;
        }
        p {
            margin: 0 0 4px 0;
            text-align: justify;
        }
        ul, ol {
            margin: 2px 0 4px 14px;
            padding: 0;
        }
        li {
            margin-bottom: 2px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 7.8pt;
            margin: 4px 0 6px 0;
        }
        th, td {
            border: 1px solid #cbd5e1;
            padding: 3px 5px;
            text-align: left;
        }
        th {
            background-color: #f1f5f9;
            color: #0f172a;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #f8fafc;
        }
        code {
            background-color: #f1f5f9;
            color: #0369a1;
            padding: 1px 3px;
            border-radius: 3px;
            font-family: Consolas, Monaco, "Courier New", monospace;
            font-size: 7.8pt;
        }
        .meta-box {
            background-color: #f8fafc;
            border-left: 3px solid #0284c7;
            padding: 5px 8px;
            margin-bottom: 6px;
            font-size: 8pt;
        }
        .callout {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 3px solid #16a34a;
            padding: 4px 8px;
            margin: 4px 0;
            font-size: 8pt;
            border-radius: 3px;
        }
        .math-box {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 4px 8px;
            margin: 4px 0;
            font-family: Consolas, Monaco, monospace;
            font-size: 7.8pt;
            border-radius: 3px;
        }
        .footer {
            font-size: 7pt;
            color: #64748b;
            text-align: center;
            border-top: 1px solid #e2e8f0;
            padding-top: 3px;
            margin-top: 6px;
        }
    </style>
    """

    # PAGE 1: Problem, Synthetic Cascade, and Definitions
    page1_html = css + """
    <body>
        <h1>Technical Report: Predicting Shipment ETA & Delay Propagation</h1>
        <div class="meta-box">
            <strong>Candidate:</strong> Kalyan Dasari &nbsp;|&nbsp; 
            <strong>Dataset:</strong> 300 Synthetic Shipment Journeys (Seed 42) &nbsp;|&nbsp; 
            <strong>Prediction Checkpoint:</strong> Port Outer Anchorage Arrival &nbsp;|&nbsp; 
            <strong>Split:</strong> 25% Stratified Test ($N=75$)
        </div>

        <h2>1. Problem Statement & Practical Motivation</h2>
        <p>
            In international maritime shipping, schedule disruptions cascade through downstream supply chain stages. A minor delay during vessel loading or ocean sailing triggers missed berthing windows, port congestion, customs clearance queues, and missed inland rail/truck connections. Traditional visibility platforms rely on static schedule tables or naive projections that assume zero future downstream delay.
        </p>
        <p>
            This system models the <strong>Port Arrival Checkpoint</strong> (outer pilot station): predicting total final delay and expected arrival timestamp while the voyage is in-progress. This enables logistics dispatchers to proactively: (1) notify consignees with calibrated revised ETAs, (2) request priority berthing or off-dock container drayage to avoid yard dwell, (3) pre-lodge customs manifests (EDI), and (4) reschedule inland drayage to prevent costly port demurrage and detention charges.
        </p>
        <div class="callout">
            <em>Notice on Synthetic Data:</em> All data and dynamics are simulated. Performance metrics measure consistency with the underlying simulation mechanics rather than empirical real-world carrier accuracy.
        </div>

        <h2>2. Data Source & Synthetic Generation Assumptions</h2>
        <p>
            The dataset models 300 journeys across realistic trade lanes (Shanghai–Rotterdam, Singapore–Sydney, Mumbai–Dubai, Los Angeles–Hamburg, Busan–Antwerp) and container carriers (Maersk, MSC, CMA CGM, Hapag-Lloyd, ONE). The journey comprises five discrete stages:
        </p>
        <ul>
            <li><strong>Stage 1 (Departure handling):</strong> Terminal loading and vessel dispatch.</li>
            <li><strong>Stage 2 (Ocean sailing):</strong> Deep-sea navigational voyage.</li>
            <li><strong>Stage 3 (Port processing):</strong> Anchorage waiting, berthing, and crane discharge.</li>
            <li><strong>Stage 4 (Customs clearance):</strong> Regulatory document and physical inspection.</li>
            <li><strong>Stage 5 (Final delivery):</strong> Inland drayage/rail transit to destination.</li>
        </ul>
        <p>
            Delays propagate via linked causal equations where upstream delays compound downstream friction:
        </p>
        <div class="math-box">
            - Dep Delay: &Delta;dep = max(0, N(2.0 + 8.5*Weather, 4.5&sup2;))<br>
            - Sailing Delay: &Delta;sail = max(0, N(3.0 + 13.5*Weather + 0.15*&Delta;dep, 6.0&sup2;))<br>
            - Port Dwell Delay: &Delta;port = max(0, N(2.0 + 26.0*Congestion + 0.22*&Delta;sail, 7.5&sup2;))<br>
            - Customs Delay: &Delta;customs = max(0, N(1.5 + 13.0*Congestion + 0.18*&Delta;port, 5.0&sup2;))<br>
            - Final Delivery Delay: &Delta;final = max(0, N(1.0 + 0.12*&Delta;customs + 3.0*Weather, 4.0&sup2;))
        </div>
        <p>
            Congestion and weather are continuous simulated signals on a 0.0 to 1.0 scale.
        </p>

        <h2>3. ETA, Stage-Delay, Cumulative-Delay & Ground-Truth Definitions</h2>
        <p>
            To prevent accidental double-counting, the event-time model formally distinguishes <strong>incremental stage delay</strong> from <strong>cumulative schedule deviation</strong>:
        </p>
        <ul>
            <li><strong>Planned Journey Duration:</strong> Sum of planned sailing, port dwell, customs, and final transit hours.</li>
            <li><strong>Scheduled ETA:</strong> <code>Scheduled Departure + Planned Journey Duration</code>.</li>
            <li><strong>Incremental Stage Delay (&Delta;s):</strong> <code>Actual Stage Duration - Planned Stage Duration &ge; 0</code> for each stage s.</li>
            <li><strong>Cumulative Delay (Ck):</strong> Sum of incremental delays up to milestone k. At the port checkpoint, observed upstream delay = &Delta;dep + &Delta;sail. At final delivery, total delay = &Sigma;&Delta;s.</li>
            <li><strong>Ground-Truth Actual Delivery:</strong> <code>Scheduled ETA + Total Delay</code>.</li>
            <li><strong>Late-Delivery Label:</strong> 1 if Total Delay &gt; 24.0 hours; otherwise 0.</li>
            <li><strong>Predicted Final ETA:</strong> <code>Scheduled ETA + Predicted Total Delay</code>.</li>
        </ul>
        <div class="footer">Page 1 of 3 | Shipment ETA Intelligence | Kalyan Dasari</div>
    </body>
    """

    # PAGE 2: Feature Design, Modeling Protocol, and Empirical Evaluation Results
    page2_html = css + """
    <body>
        <h2>4. Prediction Checkpoint & Feature Design</h2>
        <p>
            Predictions occur strictly at the <strong>Port Arrival Checkpoint</strong>. All downstream actuals are strictly withheld to eliminate target leakage. The feature pipeline enforces an explicit allowlist:
        </p>
        <ul>
            <li><strong>Schedule Baselines:</strong> <code>planned_sailing_hours</code> (72–360h), <code>planned_port_hours</code> (8–36h), <code>planned_customs_hours</code> (4–24h), <code>planned_final_hours</code> (12–60h), <code>planned_remaining_hours_at_port</code>.</li>
            <li><strong>Observed Upstream History:</strong> <code>departure_delay_hours</code>, <code>sailing_delay_hours</code>, <code>observed_delay_at_port_hours</code> (&Delta;dep + &Delta;sail).</li>
            <li><strong>Environmental Signals:</strong> <code>port_congestion</code> (0.0–1.0), <code>weather_severity</code> (0.0–1.0).</li>
            <li><strong>Domain Interaction Features:</strong> <code>upstream_delay_ratio</code> (observed delay / planned sailing), <code>congestion_weather_interaction</code> (congestion &times; weather), <code>congestion_buffer_ratio</code> (congestion / planned port hours).</li>
        </ul>
        <p>
            <em>Leakage Guard:</em> Automated tests verify that forbidden downstream columns (<code>port_delay_hours</code>, <code>customs_delay_hours</code>, <code>final_stage_delay_hours</code>, <code>actual_port_departure</code>, <code>actual_customs_done</code>, <code>actual_delivery</code>, <code>actual_total_delay_hours</code>, <code>late_delivery</code>) are strictly excluded from inputs.
        </p>

        <h2>5. Model Approach & Evaluation Protocol</h2>
        <p>
            The machine learning engine pairs regression delay estimation with calibrated delay-risk classification:
        </p>
        <ul>
            <li><strong>Random Forest Regressor:</strong> 300 decision trees, min_samples_leaf=2, max_features="sqrt", with Out-of-Bag (OOB) scoring to capture non-linear congestion interactions.</li>
            <li><strong>Delay-Risk Classifier:</strong> Random Forest Classifier (200 trees, max depth=6) predicting calibrated probability P(Delay &gt; 24 hours).</li>
            <li><strong>Exploratory Uncertainty Intervals:</strong> Non-parametric Out-of-Bag residual quantiles (q10=-14.94h, q90=+16.76h).</li>
            <li><strong>Baselines:</strong> (A) Naive Baseline: assumes zero downstream delay (&Delta;remaining = 0); (B) Operational Heuristic: observed delay + training median downstream residual.</li>
        </ul>

        <h2>6. Results Table Comparing Model with Baselines</h2>
        <p>
            Evaluated on a 25% stratified held-out test split (N=75 shipments; seed 42) from <code>artifacts/eta_model.metrics.json</code>:
        </p>
        <table>
            <thead>
                <tr>
                    <th>Model / Evaluation Approach</th>
                    <th>Held-Out MAE</th>
                    <th>Held-Out RMSE</th>
                    <th>Test R&sup2;</th>
                    <th>Error Reduction vs Benchmark</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Naive Baseline (Zero Future Delay)</strong></td>
                    <td>29.168 h</td>
                    <td>32.163 h</td>
                    <td>-2.708</td>
                    <td>+66.01% error vs AI model</td>
                </tr>
                <tr>
                    <td><strong>Operational Heuristic Baseline</strong></td>
                    <td>11.531 h</td>
                    <td>14.175 h</td>
                    <td>0.280</td>
                    <td>Baseline Benchmark</td>
                </tr>
                <tr>
                    <td><strong>Random Forest AI Regressor</strong></td>
                    <td><strong>9.915 h</strong></td>
                    <td><strong>12.693 h</strong></td>
                    <td><strong>0.422</strong></td>
                    <td><strong>+14.01% improvement vs heuristic</strong></td>
                </tr>
            </tbody>
        </table>

        <h3>Late-Delivery Classification Performance (&gt;24h Delay Threshold)</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Measured Score</th>
                    <th>Operational Interpretation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Accuracy</strong></td>
                    <td><strong>84.0%</strong></td>
                    <td>Correctly predicts delay status on 63 out of 75 test shipments</td>
                </tr>
                <tr>
                    <td><strong>Precision</strong></td>
                    <td><strong>87.3%</strong></td>
                    <td>87.3% of shipments flagged as late genuinely arrive late (low false alarms)</td>
                </tr>
                <tr>
                    <td><strong>Recall (Sensitivity)</strong></td>
                    <td><strong>95.4%</strong></td>
                    <td>Captures 62 out of 65 severely delayed shipments (critical for SLA protection)</td>
                </tr>
                <tr>
                    <td><strong>F1-Score</strong></td>
                    <td><strong>0.912</strong></td>
                    <td>Strong harmonic balance between precision and recall</td>
                </tr>
                <tr>
                    <td><strong>ROC-AUC</strong></td>
                    <td><strong>0.780</strong></td>
                    <td>Reliable risk discrimination across varying decision thresholds</td>
                </tr>
                <tr>
                    <td><strong>Confusion Matrix</strong></td>
                    <td><strong>[[1, 9], [3, 62]]</strong></td>
                    <td>TN: 1 on-time, FP: 9 false alarms, FN: 3 missed late, TP: 62 caught late</td>
                </tr>
            </tbody>
        </table>

        <h3>Uncertainty Interval Validation</h3>
        <p>
            The Out-of-Bag residual quantiles achieved <strong>74.7% empirical coverage</strong> on the held-out test set (against a nominal 80.0% target). Because random forest errors are heteroscedastic, these intervals are labeled exploratory rather than formal conformal guarantees.
        </p>

        <h3>Global Feature Importances</h3>
        <p>
            Top model drivers: <code>port_congestion</code> (21.42%), <code>observed_delay_at_port_hours</code> (14.26%), <code>congestion_weather_interaction</code> (13.37%), <code>sailing_delay_hours</code> (9.79%), <code>upstream_delay_ratio</code> (7.69%), <code>congestion_buffer_ratio</code> (7.53%).
        </p>
        <div class="footer">Page 2 of 3 | Shipment ETA Intelligence | Kalyan Dasari</div>
    </body>
    """

    # PAGE 3: Explainability, Worked Numerical Example, and Limitations / Roadmap
    page3_html = css + """
    <body>
        <h2>7. Explainability & Worked Numerical Shipment Example</h2>
        <p>
            The system provides auditable explanations via exact <strong>Tree SHAP path decomposition</strong> (&ycirc; = &yuml; + &Sigma;&phi;j) and stage-by-stage waterfalls.
        </p>
        
        <h3>Worked Numerical Example: Shipment <code>SHP-0001</code></h3>
        <div class="callout">
            <strong>Voyage Profile:</strong> Vessel <em>Nordic Horizon</em> (CMA CGM Group, Route: Shanghai to Hamburg via Port Delta (Dubai)).<br>
            <strong>Scheduled Departure:</strong> 2025-03-15 12:00:00 UTC &nbsp;|&nbsp; <strong>Scheduled Arrival:</strong> 2025-04-02 00:09:36 UTC<br>
            <strong>Planned Stage Durations:</strong> Sailing: 352.98h, Port Dwell: 29.31h, Customs: 19.72h, Final Inland: 18.15h (Total: 420.16h).
        </div>
        <p>
            <strong>1. Checkpoint Observations (Port Arrival):</strong>
        </p>
        <ul>
            <li>Departure delay: +3.35h &nbsp;|&nbsp; Sailing delay: +17.78h (severe weather: index 0.667).</li>
            <li><strong>Observed Upstream Delay at Checkpoint:</strong> +21.13 hours.</li>
            <li><strong>Terminal Indicators:</strong> Port Congestion = 0.228 (favorable) &nbsp;|&nbsp; Weather Severity = 0.667.</li>
        </ul>
        <p>
            <strong>2. Model In-Flight Predictions (Checkpoint Output):</strong>
        </p>
        <ul>
            <li><strong>Predicted Total Delay:</strong> +41.59 hours (Estimated Downstream Remaining: +20.46 hours).</li>
            <li><strong>Predicted Final ETA:</strong> 2025-04-03 17:45:00 UTC &nbsp;|&nbsp; <strong>Uncertainty Band:</strong> [26.65h, 58.35h].</li>
            <li><strong>Late Probability:</strong> 95.9% (&gt;24h threshold) &rarr; <strong>Risk Status: CRITICAL</strong>.</li>
            <li><strong>Tree SHAP Attributions:</strong> Observed Upstream Delay (+5.19h) and Sailing Delay (+3.39h) pushed ETA later, while favorable Port Congestion (-5.96h) and generous port buffer (-3.27h) cushioned downstream delay.</li>
        </ul>
        <p>
            <strong>3. Unknown Future Actuals (Recorded Post-Delivery for Auditing):</strong>
        </p>
        <ul>
            <li>Downstream Actuals: Port Dwell Delay: +11.47h, Customs Delay: +5.60h, Final Delivery Delay: +0.95h.</li>
            <li><strong>Actual Total Delay:</strong> +39.15 hours (21.13h upstream + 18.02h downstream).</li>
            <li><strong>Ground-Truth Actual Delivery:</strong> 2025-04-03 15:18:36 UTC.</li>
            <li><strong>Model Error:</strong> |41.59h - 39.15h| = <strong>2.44 hours</strong> (The model predicted delivery within 2.5 hours of actual arrival).</li>
        </ul>

        <h2>8. Limitations & Real-World Deployment Requirements</h2>
        <h3>8.1 Prototype Limitations</h3>
        <ul>
            <li><strong>Synthetic Data:</strong> Delay propagation coefficients reflect mathematical simulation assumptions rather than empirical telemetry.</li>
            <li><strong>Simplified Stationarity:</strong> Omits seasonality (e.g., Lunar New Year, peak retail shipping), vessel blank sailings, and transshipment hub connections.</li>
            <li><strong>Discrete Checkpoint:</strong> Evaluates only at port arrival, rather than continuous real-time updates along the ocean route.</li>
        </ul>
        <h3>8.2 Production Deployment Roadmap</h3>
        <ul>
            <li><strong>Live Telemetry Ingestion:</strong> Connect live AIS GPS satellite feeds (Spire/MarineTraffic), ocean weather forecasts (Copernicus/NOAA), and Terminal Operating System (TOS) EDI messages.</li>
            <li><strong>Rolling Temporal Validation:</strong> Replace random splits with temporal backtesting (train on months 1..T, test on month T+1) to detect data drift.</li>
            <li><strong>Conformal Prediction:</strong> Implement split conformal prediction to provide distribution-free, guaranteed coverage on prediction intervals.</li>
            <li><strong>Transshipment Graph Architecture:</strong> Model multi-leg container feeder networks via Graph Neural Networks (GNNs).</li>
        </ul>
        <div class="footer">Page 3 of 3 | Shipment ETA Intelligence | Kalyan Dasari | 22 October 2026</div>
    </body>
    """

    pages = [page1_html, page2_html, page3_html]
    rect = pymupdf.Rect(36, 36, 595 - 36, 842 - 36)

    for i, p_html in enumerate(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(rect, p_html)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(pdf_path))
    doc.close()
    print(f"Generated PDF technical report at: {pdf_path}")
    print(f"Total page count: {len(pymupdf.open(str(pdf_path)))} pages")

if __name__ == "__main__":
    generate_pdf()
