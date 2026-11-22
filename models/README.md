# Model artifacts

`lstm_fd001_v2/` contains the corrected model, best checkpoint, matching scaler, sequence/split metadata, history and metrics. Always use the scaler packaged with the selected model. Its input is `(batch, 30, 13)` in the sensor order recorded in metadata; multiply its scaled output by 125 to obtain capped RUL cycles.

`cox_fd001/` contains `cox_bundle.pkl` with the fitted model, StandardScaler and feature metadata. Covariates are sensor means from cycles 1–30. Its survival times are uncapped cycles since the cycle-30 landmark.

Binary models/scalers are ignored by Git. Configuration and result JSON files remain eligible for version control. Retraining overwrites the corresponding current model folder; preserve a run separately before experimenting if needed.

`anomaly_fd001/` contains `detectors.pkl` (healthy-training StandardScaler and Isolation Forest), `autoencoder.keras`, its best checkpoint, reconstruction history, metrics and `metadata.json` with sensor order, both frozen threshold policies, smoothing and sustained-alert settings. Reuse through `predict_saved` in `src/predictive_maintenance/anomaly.py`. The validation-selected primary detector is recorded even if another detector performs better on the test set. All three remain available for comparison.
