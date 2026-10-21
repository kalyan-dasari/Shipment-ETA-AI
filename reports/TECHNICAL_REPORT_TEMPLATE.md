# Shipment ETA Prediction and Delay Propagation

**Candidate:** Kalyan Dasari  
**Dataset:** Synthetic shipment journeys (300 records; seed 42)  
**Prediction checkpoint:** Port arrival

## 1. Problem definition
The system estimates total final delivery delay in hours at the port-arrival checkpoint. Predicted final ETA is scheduled ETA plus predicted delay. A shipment is labelled late when actual total delay exceeds 24 hours; this threshold is an explicit prototype assumption.

## 2. Data and assumptions
Describe the generated routes, scheduled stage durations, congestion/weather signals and how delays are propagated from departure to sailing, port and customs. State clearly that the data is synthetic and does not establish real-world performance.

## 3. Features and model
List the features available at the port-arrival checkpoint. Explain why future timestamps and actual final delivery are excluded from model inputs. Compare the Random Forest model with the simple baseline.

## 4. Evaluation
After running `python -m src.data_generator` and `python -m src.train`, insert the actual held-out MAE, RMSE and baseline MAE here. Interpret the difference honestly. Do not invent results.

## 5. Explainability and limitations
Explain observed upstream delay, congestion/weather indicators and the stage timeline. Clarify that the explanation is descriptive and is not proof of causality. Discuss synthetic data, missing stages, route variation and lack of live feeds.

## 6. Future work
Use real shipment data; validate with time-based splits; add prediction intervals and calibrated late-risk probabilities; test route-specific behaviour; monitor drift.
