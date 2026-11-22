# FD001 anomaly detector evaluation

Run date: 2026-11-19. Notebook: `notebooks/06_anomaly_detection.ipynb`.

## Protocol

80 training engines and 20 validation engines use the existing project split. Normality is fitted only on cycles 1–50 of training engines, assumed healthy. A StandardScaler uses those same training rows. Isolation Forest uses individual rows; the dense autoencoder reconstructs 30-cycle windows through an 8-dimensional bottleneck. A mean-squared z-score detector provides a simple statistical reference. Autoencoder early stopping uses healthy validation windows.

Primary thresholds are healthy-validation 98th percentiles of five-cycle causal mean scores. Alerts require three consecutive exceedances and begin on the third cycle. The first 29 cycles are warm-up for the common comparison. A separate validation-only F1 policy is frozen before test evaluation. Test labels do not influence model/threshold fitting. RUL <= 30 is the main failure-window pseudo-label; windows of 15 and 50 are also reported. These labels measure association with degradation, not annotated sensor faults.

## Independent test, primary healthy-validation threshold

| Detector | Average precision | Sustained-alert F1 | Healthy-cycle flag FPR | Healthy-cycle sustained-alert FPR |
|---|---:|---:|---:|---:|
| isolation_forest | 0.765 | 0.456 | 1.28% | 0.10% |
| autoencoder | 0.615 | 0.451 | 4.87% | 3.79% |
| zscore | 0.814 | 0.458 | 0.69% | 0.25% |

Positive prevalence on evaluated test cycles: 3.26%. AP means non-interpolated average precision, not trapezoidal PR-AUC. These are pooled cycle metrics; long records contribute more observations.

The validation-selected primary model was the autoencoder. On the independent test data it did not outperform the simpler baselines; the z-score detector had the highest AP. We retain the original selection record and do not tune on the test outcomes. Healthy validation calibration does not guarantee a 2% test false-positive rate.

## Alternative threshold selected by validation F1

| Detector | Test sustained-alert F1 | Test healthy sustained-alert FPR |
|---|---:|---:|
| isolation_forest | 0.659 | 0.00% |
| autoencoder | 0.512 | 2.11% |
| zscore | 0.702 | 0.00% |

The native Isolation Forest 2% contamination baseline is recorded separately in `tables/anomaly_fd001/native_isolation_forest.json`; it applies to unsmoothed cycle scores, so its cycle F1 is not the same metric as sustained-alert F1.

## Alert timing and interpretation

Only 25 of the 100 test engines reach RUL <= 30 within their observed records. For the primary policy, failure-window alert coverage is 96% for Isolation Forest and 100% for the autoencoder and z-score baseline. These values describe observed windows, not full-lifetime detection guarantees.

Among engines receiving any primary-policy alert, median first-alert lead time is 53 cycles for Isolation Forest, 104.5 for the autoencoder and 51 for z-score. The autoencoder issues 17 first alerts while true RUL is still above 125; the other two each issue 4. Longer lead time is therefore not automatically better. No-alert lead times remain missing, and early alerts may count as false positives under the chosen failure-window labels.

## Artifacts and limits

- Saved detectors, scaler, autoencoder, thresholds, versions and history: `models/anomaly_fd001/`.
- Every test cycle with raw/smoothed scores, flags and alerts: `reports/tables/anomaly_fd001/test_cycle_scores.csv`.
- Threshold searches, sensitivity metrics, FPR diagnostics, lead times and coverage: `reports/tables/anomaly_fd001/`.
- PR curve, training curve and engine-24 example: `reports/figures/anomaly_fd001/`.
- Reusable inference: `predict_saved` in `src/predictive_maintenance/anomaly.py`; pass complete consecutive histories to preserve rolling state.

FD001 only: no operating-regime normalization was fitted for FD002/FD004. Anomaly scores are not calibrated failure probabilities. Healthy early life is assumed, and degradation can start before the chosen label window. Genuine spike/fault detection and a deployment alert policy require additional validation.
