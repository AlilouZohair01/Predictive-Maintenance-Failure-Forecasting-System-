# Live dashboard integration validation

Validated 2026-11-22 in the existing Windows Python 3.11 `exam_env`.

## Outcome

The historical dashboard now mounts an authenticated live replay workspace at `/live/`. It discovers saved runs and engines and refreshes the selected engine's stored readings, predictions, survival updates and run-wide alert history every two seconds. No model inference is caused by feed polling. SHAP remains an explicit, cycle-pinned request.

## Automated checks

All **81 tests passed** in 74.480 seconds (`live_dashboard_full_tests.txt`): the previous 73 tests plus eight live integration tests. The new cases cover warm-up-to-ready transitions, a processed-cycle boundary with queued readings, survival update age, source staleness, closed/blocked runs, API outage clearing, alert rendering, run/engine discovery, pause/resume, shared authentication, pagination/bearer credentials, session/run isolation, old-release SHAP rejection and PNG/PDF downloads.

## Real replay and models

`python scripts/validate_live_dashboard.py` starts its own isolated API/database and uses the existing replay script and durable worker with real FD001 engine 55 observations. The dashboard's actual HTTP callbacks were exercised while the replay was running.

- Observed processed cycles advanced **29 → 30 → 33 → 37 → 40**. Cycle 29 was warm-up; later cycles displayed saved model results.
- The run ended as `completed` with predicted RUL **120.8134 cycles**, and a survival result from cycle **40**.
- Three additional refreshes left model-inference counters unchanged.
- Explicit SHAP succeeded and remained pinned to cycle 40.
- PNG/PDF exports succeeded with run/engine/cycle provenance. The PNG was visually inspected; the PDF uses the same Matplotlib layout.
- Creating a fresh dashboard recovered the saved run and cycle 40 from the API.
- Validation used an isolated database and stopped its own API. It did not modify the user's existing replay database or stop a user-started server.

Exact observations and timing are recorded in `live_dashboard_validation.json`. Example reports are `live_dashboard_example.png` and `live_dashboard_example.pdf`. This is a local functional validation, not a throughput/endurance benchmark.

## Verification boundary

These checks verify real replay-to-model-to-dashboard callback behavior. They do not constitute a visual browser review. Browser automation was previously unable to verify the browser URL safely; that separate visual-review limitation remains. The app's existing trained models and their methodological limitations are unchanged, including the experimental Cox probabilities.

## Run the app

First terminal, from the project folder:

```powershell
conda activate exam_env
python scripts/run_dashboard.py
```

Open `http://127.0.0.1:8050/live/`. In a second project terminal:

```powershell
conda activate exam_env
python scripts/replay_stream.py --units 1 55 --interval 0.5
```

Select the run ID printed by the simulator and an engine. If the dashboard was already running before this update, restart its process first. No new dependencies, hardware sensors or broker are required.
