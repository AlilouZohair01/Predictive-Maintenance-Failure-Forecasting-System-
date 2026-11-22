# Step 9 dashboard validation

Real-model validation: 2026-11-21. Full test run and installation: 2026-11-22. Environment: Windows, Python 3.11 in `exam_env`.

## Real-model integration

`python scripts/validate_dashboard.py` passed against an isolated HTTP API using the existing saved models, release `270491b9619aa4a6`. The script stops its own API after validation and leaves any existing API untouched.

Engine 55 at observed cycle 113:

| Output | Recorded value |
|---|---:|
| Predicted RUL | 120.2535 cycles |
| Smoothed autoencoder anomaly score | 0.508362 |
| Sustained anomaly flag | False |
| Experimental failure probability within 100 cycles | 0.679046 |
| SHAP reconstruction gap | -0.05446 cycles |
| Sampled RUL/risk checkpoints | 12 |

Checks passed: actual HTTP requests, SHAP prediction parity, sampled history, inference rejection before cycle 30, cycle-30 alert warmup, session isolation, stale-selection rejection and PNG/PDF export. The PNG was visually inspected and the footer spacing corrected. The PDF uses the same verified Matplotlib layout.

Full machine-readable results and observed timings are in `dashboard_validation.json`. These timings are a local functional check, not a throughput benchmark or latency guarantee. Predictions from the weak Cox baseline can disagree with the LSTM; probability calibration is not established.

## Automated and browser checks

All **73 tests passed** in 62.329 seconds. The run includes 14 dashboard tests covering observation-prefix isolation, warmup, cache ownership/expiry, release consistency, partial survival failure, callback rendering, stale-selection clearing, basic authentication and PNG/PDF downloads. The existing model/API/streaming tests also passed. The complete output is in `step09_test_run.txt`.

Dash 4.5.1, Plotly 7.1.0, Flask 3.1.3 and Waitress 3.0.2 are installed in `exam_env`. `python -m pip check` reported no broken requirements. Packages were downloaded from official PyPI sources and verified before the final offline installation.

The launcher successfully started the real model API and dashboard. `GET http://127.0.0.1:8050/health` returned HTTP 200 with historical mode and 100 engines. Startup handling was corrected to handle connection timeouts as well as connection refusals.

**Visual browser verification remains uncompleted:** the in-app preview timed out, and the Windows browser automation tool then stopped because it could not determine the current browser URL with enough confidence to enforce its policy. No browser-policy bypass was attempted. Responsive layout and interactive chart appearance still need a manual browser review; HTTP/callback tests are not a substitute for that visual check.

## Scope

### Automatic historical analysis (2026-11-22)

The historical page now starts analysis on initial load and engine/cycle changes, then automatically calculates the sampled trend and SHAP explanation. Analyze snapshot, Build trend and Explain prediction buttons were removed from this page. Loading feedback identifies ongoing calculations. Sensor display changes do not request model inference. Live replay retains its explicit, cycle-pinned SHAP workflow.

All **25 dashboard/live-dashboard tests passed** in 30.530 seconds (`dashboard_automatic_analysis_tests.txt`). Callback tests cover automatic dependencies, both engine and cycle changes, populated trend/SHAP charts, stale-selection rejection, warmup with no inference, sensor-only changes, partial detail failures, session isolation and PNG/PDF downloads. Live dashboard regression tests also passed. These tests use model stubs; the saved models were not retrained or re-benchmarked. Browser appearance of this change still requires review after restarting the dashboard. No application server or replay process was started or stopped during this change.

### Chart sizing correction (2026-11-22)

User screenshots showed populated live metrics but collapsed chart panels on both pages. Inspection of the installed Dash 4.5.1 graph renderer confirmed that responsive graphs default to a percentage container height. Historical charts, dynamically generated live charts and live SHAP now have explicit pixel heights with responsive width; panel graphs also have a minimum height. The live pause-label contrast was corrected.

All 22 dashboard/live-dashboard tests passed after this change (`dashboard_chart_fix_tests.txt`). Serialized layouts and callback results were separately checked for explicit positive chart heights. These checks do not replace browser verification after the user restarts the dashboard and reloads the page. The user's running app and replay processes were not started or stopped for this fix.

Update: Live integration has since been implemented and tested separately; see `live_dashboard_validation.md` and `live_dashboard_validation.json`. The paragraph below describes the scope of the original historical-dashboard validation.

This validates the historical dashboard. Live playback, auto-refreshing stream feeds, corporate SSO, multi-process shared caches, external notifications and network deployment are outside this increment. No training artifacts or notebooks were changed.
