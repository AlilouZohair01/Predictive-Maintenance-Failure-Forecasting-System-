# Step 9 — Historical analysis and live replay dashboard

The Dash/Plotly dashboard explores the recorded NASA C-MAPSS FD001 test set and monitors the existing Step 8 replay stream. Historical analysis calls FastAPI models automatically when the page opens or the engine/cycle selection changes. The **Live replay** workspace reads saved predictions and alerts every two seconds. Model artifacts and training notebooks are unchanged.

## Start

From the project directory in your existing environment:

```powershell
conda activate exam_env
python scripts/run_dashboard.py
```

Open **http://127.0.0.1:8050**. Keep the terminal running. The launcher reuses an API at the configured address or starts a local API when none is listening. First model startup can take about a minute. Ctrl+C stops the dashboard and only the API process this launcher created. It never stops an API you started separately.

The dashboard dependencies are installed in your existing `exam_env`. For a fresh environment, first install the serving environment as described in `api_guide.md`, then `python -m pip install -r requirements-dashboard.txt`. No Docker, MQTT, Kafka, Redis, MySQL server or physical sensors are needed for this dashboard.

If you prefer separate terminals, start the API using the README command, then run `python -m predictive_maintenance.dashboard.app`. The dashboard-only command does not start the API.

## Use

1. Select an engine and observation cycle. Sensor charts immediately show only the observed prefix, in native units. Choose up to three sensors.
2. Remaining useful life, anomaly history and conditional survival calculate automatically. At least 30 observations are required. Sustained anomaly decisions require 32 observations.
3. The trend automatically evaluates up to 12 evenly spaced checkpoints from cycle 30 through the selected cycle. Separate axes show RUL and the experimental probability of failure within 100 further cycles. Unsupported or unavailable risk estimates remain blank. Connecting lines are not predictions for every intervening cycle.
4. SHAP automatically explains the selected prediction. Positive sensor contributions increase predicted RUL relative to the reference prediction; negative contributions decrease it. The approximation gap is not prediction error. The status message shows when calculations are in progress; trend and SHAP can take longer than the main predictions.
5. Use **Export PNG** or **Export PDF** for a report of the selected, analyzed snapshot. Reports include RUL, anomaly history, survival, available SHAP, model release and analysis time. The sensor panel uses the first selected sensor. Reports are summaries, not screenshots of the entire dashboard.

Changing the engine or cycle clears displayed results that belong to another selection and calculates the new selection. The cycle slider runs analysis when released; changing only displayed sensors does not rerun models. Wait for the analysis to finish before exporting. If trend or SHAP fails, the successful results remain available and the status names the failed calculation. Reload the page to retry the current selection after an API error. An analysis is cached on the server for 30 minutes, with a maximum of 128 snapshots shared across viewers. Tokens are tied to the current browser session. Restarting the dashboard clears the cache.

## Watch the live replay

Keep `python scripts/run_dashboard.py` running. Click **Live replay** in the sidebar, or open **http://127.0.0.1:8050/live/**. In a second project terminal:

```powershell
conda activate exam_env
python scripts/replay_stream.py --units 1 55 --interval 0.5
```

The existing simulator prints a unique run ID and processes readings as it sends them. Select that run and an engine on the live page. New runs and engines are discovered automatically. If a previously selected run is still available, the dashboard keeps it selected instead of silently switching to a new run. The latest run is selected only when no valid selection exists.

- Sensor plots, RUL, anomaly history, RUL/risk trends, survival and run-wide alert history refresh every two seconds. RUL and anomaly results appear from cycle 30; sustained flags need cycle 32.
- Each view freezes at a committed **processed cycle**. Newer queued readings are excluded from its charts. Received cycle, processed cycle and lag are shown separately.
- Survival retains its actual update cycle (normally every 10 cycles). The displayed curve and probability are explicitly dated by cycle; the risk trend contains actual saved survival updates, not a new inference on each browser refresh. Custom run horizons are honored.
- Warm-up, pending work, failed jobs, stale open sources and completed/closed replays have distinct messages. API failures clear the current results and retry on the next timer tick. Closing a replay does not erase its saved results.
- **Pause updates** pauses this page's timer only. The simulator and worker continue. Changing the selected run/engine/sensors still requests a view of that selection.
- **Explain displayed snapshot** runs SHAP only on explicit request. Its panel stays pinned to the labeled run/engine/cycle as new data arrives. Changing run or engine clears the explanation. An older saved run cannot be explained using a different current model release.
- PNG/PDF exports use the displayed saved stream readings, with run ID, processed cycle, model release and the survival update cycle. SHAP is included only if it explains that same cycle. Export does not read later observations from the local NASA file.

Polling only reads `/v1/stream` feeds; it does not run models, write sensor records, change alerts or send notifications. The historical and live workspaces share the same login protection. Stream runs are shared within this single-tenant API; browser snapshot tokens and exports remain session-owned. The view is bounded to 1,000 cycles per engine and 2,000 entries per paginated catalog/alert list; exceeding a list limit produces a message rather than silently omitting entries.

Restarting the dashboard reloads saved runs from the API's SQLite history. The worker and API must still use the same configured `PM_STREAM_DB`; see the Step 8 guide for resuming interrupted replays. No MQTT, Kafka, Redis or physical sensors are required.

## Interpretation

- The LSTM uses the latest 30 observations across 13 retained sensors. RUL is capped at 125 cycles. Gauge bands (below 20, 20–50, 50+) are illustrative, not validated maintenance thresholds.
- The autoencoder receives the complete observed prefix so its five-cycle causal smoothing and three-consecutive-exceedance policy match Step 8. Its score is **not a failure probability**.
- The Cox baseline uses first-30-cycle sensor means and conditions on survival through the selected cycle. New sensor readings do not update its covariates. Its test C-index is only **0.576**, and probability calibration has not been established. A high LSTM RUL and high Cox probability can therefore disagree; the dashboard does not combine them into a single risk score.
- The 100-cycle failure estimate is omitted when that requested horizon is outside supported follow-up. A supported partial survival curve may still appear. API failures affecting survival leave RUL/anomaly results available with a message.
- **Alert review** shows historical sustained-anomaly episodes using the existing Step 8 policy, plus a current RUL-below-20 condition. Cycle numbers locate historical events; the dataset has no real acquisition timestamps. Analysis timestamps are UTC. No notification is sent.
- **Reviewed engines** on the historical page lists engines analyzed in this browser session, sorted by their last reviewed RUL. The separate Live replay page uses Step 8's persisted run history.
- Test truth is not read for inference. The dashboard never invents readings after an engine's last recorded test cycle. It is a research interface, not a validated maintenance decision system.

## Configuration and authentication

Configuration uses `.env` or environment variables; `.env.example` lists the defaults.

| Setting | Default / meaning |
|---|---|
| `PM_DASH_API_URL` | `http://127.0.0.1:8000` |
| `PM_DASH_HOST`, `PM_DASH_PORT` | `127.0.0.1`, `8050` |
| `PM_DASH_DATA_FILE` | `data/raw/cmapss/test_FD001.txt` |
| `PM_API_TOKEN` | Server-side bearer credential when the model API requires it |
| `PM_DASH_AUTH_MODE` | `local`; use `basic` for a browser login |
| `PM_DASH_USERNAME`, `PM_DASH_PASSWORD` | Required in basic mode; password at least 12 characters |
| `PM_DASH_SESSION_SECRET` | Optional stable secret, at least 32 characters; otherwise generated at startup |
| `PM_DASH_ALLOWED_HOSTS` | Explicit allowed hostnames |
| `PM_DASH_SECURE_COOKIE` | Set `true` when serving through HTTPS |

Local mode is bound to loopback by default. Basic authentication protects pages, assets, downloads and callback routes. Non-loopback binding requires basic mode. For network deployment, configure HTTPS, allowed hosts and API authentication before exposing the service. The current implementation is a single-process local dashboard; multi-process shared sessions/caches and corporate SSO are not implemented.

## Verification

```powershell
python -m unittest discover -s tests -v
python scripts/validate_dashboard.py
python scripts/validate_live_dashboard.py
```

The second command starts an isolated temporary API, exercises the real saved models, writes `reports/dashboard_validation.json` and example PNG/PDF reports, then stops only its own API. See `reports/step09_validation.md` for the recorded results and browser checks.

The third command runs the existing replay/worker against an isolated API while exercising actual live dashboard callbacks. It verifies advancing cycles, explicit SHAP, exports, recovery of saved runs after dashboard restart, and zero inference caused by feed refreshes. Results are in `reports/live_dashboard_validation.json` and `reports/live_dashboard_validation.md`.
