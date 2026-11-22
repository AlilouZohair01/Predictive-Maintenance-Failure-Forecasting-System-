# Project status

Updated 2026-11-22.

## Active workflows

- **Preprocessing:** 30-cycle overlapping windows, targets at the final observed cycle, final windows included; 80 training and 20 validation engines; no shared engines; scaler fitted on training engines only.
- **LSTM:** corrected training and independent final-window test evaluation; early stopping, best checkpoint, recorded history/metrics, and SHAP. Current saved arrays contain 14,241 training and 3,490 validation windows.
- **Cox:** one row per engine at a fixed cycle-30 landmark, uncapped follow-up, training-only standardization, validation-based ridge penalty selection, proportional-hazards checks and conditional survival probabilities. Training-file events are observed; the operational test table is censored. Test RUL truth is used only for evaluation. Test C-index is approximately 0.576, so this is a weak baseline.
- **ML baselines:** preserved as historical work. Their CSV is scaled using all training-file engines, their RUL target is uncapped despite an old code comment, and evaluation is per row rather than per test-engine final observation. These issues must be addressed before a fair comparison with the corrected LSTM.
- **Streaming (Step 8):** durable HTTP ingestion, SQLite history/current results, ordered worker jobs calling the model API, idempotent replay, bounded retries, in-application alert episodes and authenticated dashboard feeds. Real two-engine replay and API restart were verified. See `docs/step_08_streaming.md` and `reports/step08_validation.md`.
- **Dashboard (Step 9):** historical FD001 exploration with predictions, trends and SHAP calculated automatically on engine/cycle selection, plus a `/live/` replay workspace connected to Step 8 feeds. Live panels refresh every two seconds, show lag/freshness/alerts and preserve actual survival update cycles. Live mode includes pause and explicit cycle-pinned SHAP. Both workspaces have session-owned exports and shared login protection. Uses the existing replay script. See `docs/step_09_dashboard.md` and `reports/live_dashboard_validation.md`.

## Remaining work

Step 6 is implemented in `notebooks/06_anomaly_detection.ipynb`, with reusable inference in `src/predictive_maintenance/anomaly.py`. It compares Isolation Forest, a dense 30-cycle window autoencoder and a mean-squared z-score baseline. Normality is fitted on early training-engine data without RUL labels. Validation determines thresholds; final evaluation uses the separate FD001 test records and their true RUL-derived pseudo-labels. Failure windows of 15, 30 and 50 cycles, early-cycle and RUL>125 false alarms, and sustained-alert lead times are reported. Test records are truncated, so alert coverage is limited to observed intervals. This does not validate detection of sudden sensor faults or operating-regime changes.

1. Align ML preprocessing, target definition and evaluation with the intended comparison protocol.
2. Add a proper run tracker and systematic LSTM hyperparameter search. Saved history and metrics are not a full experiment-tracking system.
3. Improve the survival model using a justified dynamic prediction design; assess probability calibration. Optional Weibull AFT is not yet implemented.
4. Step 7 native API implementation is complete: fingerprinted startup, RUL/survival/anomaly/SHAP routes, strict contracts, bounded inference, auth, metrics, real-model tests and HTTP benchmark. Validate the supplied Docker image on a Docker-enabled machine; then perform deployment-specific load/endurance checks. See `docs/api_guide.md` and `reports/api_validation.md`. Archived deployment drafts remain historical and unused.
5. Resolve the pre-existing Git conflicts separately, without discarding uncommitted work.

The loose old SHAP images were replaced by exports of the latest notebook outputs (Engine 55 prediction 120.25 cycles). Current LSTM metrics report 11.85 validation RMSE and 13.36 test RMSE.
