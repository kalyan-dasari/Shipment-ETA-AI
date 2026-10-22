import pandas as pd
from src.predict import load_bundle, predict_single

bundle = load_bundle("artifacts/eta_model.joblib")
df = pd.read_csv("data/shipments.csv")
shp1 = df[df["shipment_id"] == "SHP-0001"].iloc[0]

print("=== SHP-0001 RAW CSV VALUES ===")
print("Carrier:", shp1["carrier"])
print("Vessel:", shp1["vessel_name"])
print("Route:", shp1["route"])
print("Port:", shp1["port"])
print("Scheduled Departure:", shp1["scheduled_departure"])
print("Scheduled Arrival:", shp1["scheduled_arrival"])
print("Planned Durations: Sailing =", shp1["planned_sailing_hours"], "Port =", shp1["planned_port_hours"], "Customs =", shp1["planned_customs_hours"], "Final =", shp1["planned_final_hours"])
planned_sum = shp1["planned_sailing_hours"] + shp1["planned_port_hours"] + shp1["planned_customs_hours"] + shp1["planned_final_hours"]
print("Planned Total Duration Sum:", planned_sum)
print("Departure Delay:", shp1["departure_delay_hours"])
print("Sailing Delay:", shp1["sailing_delay_hours"])
print("Observed Upstream Delay at Port:", shp1["observed_delay_at_port_hours"])
print("Port Congestion:", shp1["port_congestion"])
print("Weather Severity:", shp1["weather_severity"])
print("Actual Port Dwell Delay:", shp1["port_delay_hours"])
print("Actual Customs Delay:", shp1["customs_delay_hours"])
print("Actual Final Delivery Delay:", shp1["final_stage_delay_hours"])
print("Actual Total Delay:", shp1["actual_total_delay_hours"])
print("Actual Delivery Timestamp:", shp1["actual_delivery"])

res = predict_single(shp1, bundle)
print("\n=== MODEL PREDICTIONS FOR SHP-0001 ===")
print("Predicted Delay Hours:", res["predicted_delay_hours"])
print("Predicted Remaining Delay Hours:", res["predicted_remaining_delay_hours"])
print("Predicted Delay Lower Hours:", res["predicted_delay_lower_hours"])
print("Predicted Delay Upper Hours:", res["predicted_delay_upper_hours"])
print("Predicted ETA:", res["predicted_eta"])
print("Predicted ETA Lower:", res["predicted_eta_lower"])
print("Predicted ETA Upper:", res["predicted_eta_upper"])
print("Late Probability:", res["late_probability"])
print("Late Risk Category:", res["late_risk"])

# Check ETA arithmetic
sched_arr = pd.to_datetime(shp1["scheduled_arrival"], utc=True)
calc_pred_eta = sched_arr + pd.to_timedelta(res["predicted_delay_hours"], unit="h")
print("\n=== ARITHMETIC VERIFICATION ===")
print("Scheduled Arrival + Predicted Delay:", calc_pred_eta)
print("Matches Predicted ETA exactly?", calc_pred_eta == pd.to_datetime(res["predicted_eta"], utc=True))

calc_act_deliv = sched_arr + pd.to_timedelta(shp1["actual_total_delay_hours"], unit="h")
print("Scheduled Arrival + Actual Total Delay:", calc_act_deliv)
print("Matches Actual Delivery exactly?", calc_act_deliv == pd.to_datetime(shp1["actual_delivery"], utc=True))

p_del = res["predicted_delay_hours"]
a_del = shp1["actual_total_delay_hours"]
err = abs(p_del - a_del)
print(f"Model Error: |{p_del} - {a_del}| = {err:.2f} hours")

print("\n--- TREE ATTRIBUTIONS ---")
for a in res["tree_attributions"]["attributions"][:6]:
    print(f"  {a['label']}: {a['impact_hours']:+.2f}h (val={a['value']})")
