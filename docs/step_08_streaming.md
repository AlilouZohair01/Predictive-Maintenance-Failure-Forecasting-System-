# Step 8: durable streaming and prediction history

Implemented 2026-11-21. This is the backend streaming step; the graphical dashboard is Step 9.

Update 2026-11-22: Step 9 now includes a connected **Live replay** page at `http://127.0.0.1:8050/live/`. Start it with `python scripts/run_dashboard.py`, then use the existing replay command in a second terminal. See `step_09_dashboard.md` for the complete workflow.

The initial version uses NASA FD001 replay, authenticated HTTP ingestion, SQLite storage and a separate worker that calls the existing inference endpoints. It includes stored current results, prediction history, in-application alerts and dashboard feed endpoints. MQTT/Kafka, Redis, external email/Slack delivery and a frontend are not part of this initial local version.

## Try it now

If your API terminal is still running the Step 7 version, press **Ctrl+C** there and restart it from the project root:

```powershell
conda activate exam_env
python -m uvicorn predictive_maintenance.api.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 --no-access-log --no-proxy-headers
```

Wait for “Application startup complete.” Leave this terminal open.

Open a **second terminal** in the same project folder:

```powershell
conda activate exam_env
python scripts/replay_stream.py --units 55 --max-cycle 40 --run-id demo-step8
```

This sends engine 55's first 40 observations, runs the durable worker as it replays, and saves results. Cycles 1–29 are warm-up. RUL and anomaly inference begin at cycle 30; survival updates at cycles 30 and 40. The default 0.2-second delay is added to processing time; this is a paced simulation, not a wall-clock recreation of engine operation.

Then visit these URLs while the API is running:

- [All runs](http://127.0.0.1:8000/v1/stream/runs)
- [Demo progress](http://127.0.0.1:8000/v1/stream/runs/demo-step8)
- [Latest engine result](http://127.0.0.1:8000/v1/stream/runs/demo-step8/units/55/latest)
- [Prediction history](http://127.0.0.1:8000/v1/stream/runs/demo-step8/units/55/predictions)
- [Alert episodes](http://127.0.0.1:8000/v1/stream/runs/demo-step8/alerts?state=all)
- [Interactive endpoint documentation](http://127.0.0.1:8000/docs)

These are JSON feeds, not the dashboard. An empty alert list is a valid result: the replay does not inject faults or force an alert.

To replay the complete records of two engines as interleaved streams, use a **new** run ID:

```powershell
python scripts/replay_stream.py --units 1 55 --run-id full-demo --interval 0.2
```

If no run ID is supplied, the script generates and prints one. Save it if you might need to resume. A fresh installation needs `python -m pip install -r requirements-streaming.txt` after installing the Step 7 serving requirements and local package. Those client dependencies already exist in the current `exam_env`.

## How the data moves

1. The replay reads raw sensor observations from `data/raw/cmapss/test_FD001.txt`. It does not load RUL truth or failure labels.
2. `POST /v1/stream/runs/{run_id}/readings` validates one observation, then atomically stores the reading and a pending job. HTTP 202 means **durably accepted**, not that inference has finished.
3. A worker claims the next eligible cycle, reconstructs input history through that cycle, and calls the existing HTTP model endpoints.
4. The result, latest-result record, alert transitions and completed-job state are committed together.
5. Feed endpoints read stored results. Refreshing a dashboard does not trigger model inference.

The default database is `data/runtime/stream.sqlite3`. It survives API/worker restarts and is ignored by Git. `PM_STREAM_DB` selects another file. The API and worker must use the **same local database**; the worker checks the API's database identifier and refuses to run against a different file. There is no need to reinstall or retrain models.

The latest-result record is a durable materialized cache in SQLite. There is no Redis cache or TTL eviction in this version. Explicit receipt age, source state and processing lag describe freshness instead.

## Model cadence and provenance

| Output | When it runs | Input |
|---|---|---|
| RUL | Every accepted cycle from 30 onward | Latest exactly 30 raw observations |
| Anomaly | Every accepted cycle from 30 onward | Complete history from cycle 1 through the current job |
| Survival | Cycle 30, then every 10 cycles by default | Original cycles 1–30 and the current job's cycle |
| SHAP | On demand through the existing explanation endpoint | Explicit selected 30-cycle window |

Before cycle 30, stored snapshots say `warmup` and contain null model results. Anomaly current flags begin at 30; sustained-alert decisions remain null at 30–31. The five-cycle smoothing window is full at cycle 34, matching Step 7.

Each run pins its model release, source identifier, survival cadence and alert policy. The replay source includes a fingerprint of the raw input file. A worker refuses to commit outputs from a different model release. Restore the original release to continue an old run, or start a new run after a deliberate model rollout.

Snapshots retain the actual response model version, request ID, inference duration and observation cycle. `latest_survival` may refer to an older cycle than the latest RUL prediction; its cycle is never relabeled. After the Cox model's supported age, snapshots report `survival_status=unsupported_age`; the last supported curve remains separately dated by cycle. SHAP is not recalculated continuously.

NASA observations have cycles, not real acquisition timestamps. `observed_at` remains null unless an input supplies a timezone-aware timestamp. `received_at` and `processed_at` describe application wall-clock events, not engine measurement times.

## Durable ordering, retries and restarts

- Identity is `(run_id, unit_id, cycle)`. Identical normalized readings are acknowledged as duplicates without adding jobs or alerts. A different value for an existing identity returns 409 `READING_CONFLICT`.
- Every engine starts at cycle 1. Gaps and new out-of-order observations return 409 with the required next cycle; the processor does not invent or silently sort missing data. Engines are independent within a run.
- The default queue cap is 5,000 unfinished jobs across the database. At capacity, ingestion returns 429 and the sender must retry the same observation. Failed jobs count toward this limit. The replay uses bounded waiting and can process queued work while waiting.
- History is limited to the configured Step 7 maximum (at most 1,000 cycles per engine). There are at most 1,000 engines per run. No oldest-reading eviction silently changes anomaly behavior.
- Jobs execute in cycle order for each engine. A failed job blocks later cycles for that engine, while other engines can continue.
- Connection errors, response timeouts and selected temporary HTTP failures are retried with exponential delays, up to five failed attempts. Authentication errors, invalid responses and model-release mismatches fail without endless retries. Stored errors use safe codes, not raw response bodies or secrets.
- Worker leases last 180 seconds and are renewed before each inference request. A crashed worker's in-flight job becomes reclaimable after its lease expires. Ownership tokens prevent an older worker from committing after another worker takes over. Inference can be computed again after a crash, but only one result/alert transaction is committed.
- SQLite transactions are short, with a rollback journal and FULL synchronous durability. No model or network call holds a database transaction open. This is a local, low-volume design, not a distributed shared-filesystem database.

To resume after interruption, rerun the same replay command with the same run ID and options. Already accepted observations are acknowledged safely. A finished replay closes its run to new cycles, while existing jobs can still drain. To extend a completed demonstration or change its configuration, create a new run ID.

Inspect `next_job` in the engine's latest feed when work is blocked. After correcting the cause, retry its failed job:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/v1/stream/runs/demo-step8/units/55/retry'
```

The retry resets failed attempts but does not change the pinned policy or model release. Start the worker again if it is not running. A processing job left by an interrupted worker may need up to 180 seconds for its lease to expire.

For a separate continuously running worker:

```powershell
python -m predictive_maintenance.streaming.worker --api-url http://127.0.0.1:8000
```

Then send data with `replay_stream.py --ingest-only`, or an HTTP producer. `--run-id` can restrict a worker to a single run; `--once` attempts one eligible job. Start with one worker. More workers may hit the API's ordinary-inference capacity and do not automatically increase throughput.

## Alert rules

The default `illustrative_v1` policy is explicitly labeled **illustrative_unvalidated**. It is not a combined risk probability or an optimized maintenance policy.

| Rule | Raise | Resolve |
|---|---|---|
| `low_rul` | Predicted RUL strictly below 20 cycles | Predicted RUL at least 25 cycles |
| `sustained_anomaly` | The model's sustained alert becomes true | Five consecutive scored cycles with `is_anomaly=false` |

An active condition updates one episode rather than producing a new alert every cycle. After resolution, a new episode must also respect a 10-cycle cooldown measured from the previous raise. These cycle-based rules make replay deterministic. Unknown/warm-up results never count as a normal clearance.

Alert episodes store opening, latest and closing cycles, application timestamps, severity, triggering evidence, release and policy. A separate event log records raised/resolved transitions. Only database-backed in-application alerts are enabled: no email, Slack message or external notification is sent.

Custom thresholds/cooldown/cadence can be supplied when creating a new run through `/docs`. They are frozen for that run and remain labeled unvalidated. No learned detector thresholds are overridden.

## Dashboard feed contract

All routes use the same optional bearer authentication as the inference API. In bearer mode, browser address-bar navigation alone will not supply the token; use `/docs` with **Authorize** or an authenticated client.

| Method and route under `/v1/stream` | Purpose |
|---|---|
| `GET /status` | Database identity, job/alert counts, last worker claim time and limits |
| `POST /runs` | Create a run with pinned configuration; identical retries are safe |
| `GET /runs` | List runs with progress and queue counts |
| `GET /runs/{run_id}` | Inspect one run |
| `POST /runs/{run_id}/close` | End ingestion; drain existing jobs |
| `POST /runs/{run_id}/readings` | Accept `{unit_id, reading}` |
| `GET /runs/{run_id}/units` | Latest engine summaries/results |
| `GET /runs/{run_id}/units/{unit_id}/latest` | Latest snapshot, last survival update, active alerts, freshness and lag |
| `GET /runs/{run_id}/units/{unit_id}/readings` | Ordered raw observations with receipt timestamps |
| `GET /runs/{run_id}/units/{unit_id}/predictions` | Ordered snapshots, including warm-up records |
| `GET /runs/{run_id}/alerts` | Active, resolved or all alert episodes |
| `GET /runs/{run_id}/alert-events` | Raised/resolved event log |
| `POST /runs/{run_id}/units/{unit_id}/retry` | Reset failed jobs after their cause is fixed |

Lists have bounded `limit` parameters and cursors (`after`, `after_unit`, `after_cycle`, `after_id`). A full page returns the corresponding `next_after...` value; one final empty page may be needed. Alert episode lists are ordered by episode ID, not an invented risk score. A dashboard can prioritize the supplied severity and RUL separately.

`processing_lag_cycles` is received minus processed cycle. `is_stale` identifies an open source with no new receipt for `PM_STREAM_STALE_SECONDS` (default 30 seconds). A closed replay is identified as closed rather than presented as a stale live sensor. Last worker claim time is operational information, not proof that an in-flight native model call is healthy. `/metrics` now includes bounded job-state and alert-state gauges.

## Configuration and containers

```text
PM_STREAM_DB=data/runtime/stream.sqlite3
PM_STREAM_MAX_PENDING=5000
PM_STREAM_STALE_SECONDS=30
```

The API and worker read these values through the existing settings and `.env` support. Keep the bearer secret in environment/configuration, not command-line URLs. This is a single-tenant application: the credential grants access to the shared runs, not per-user ownership isolation. Public deployment still requires gateway TLS, rate limits and operational access controls.

Compose now gives the API a persistent `stream_state` volume at `/state`. An optional worker service shares it:

```powershell
docker compose --profile streaming up --build
```

Use replay `--ingest-only` from the host when the worker/database live inside Docker. The image installs `requirements-streaming.txt`; raw NASA data stays on the host. Docker is unavailable on this machine, so the image, worker profile and volume permissions remain unverified in containers.

Backups/retention are not automated. Use SQLite's backup facility or stop writers before making a database copy. Do not delete the runtime database to retry jobs: it contains the history and alert log. Redis, a distributed broker, PostgreSQL, external notification delivery and validated hardware integration remain later extensions.

## Validation and performance

```powershell
python -m unittest discover -s tests -v
python scripts/validate_stream.py
```

The validation script launches and stops its own API on a temporary port, uses an isolated persistent validation database, replays two real engines, compares stored results with direct inference, restarts the API and redelivers the same observations. It does not stop an API already running on port 8000. Results are in `reports/stream_validation.json`, `reports/stream_example_feed.json` and `reports/step08_validation.md`.

The initial operational target is correctness under a paced local replay, zero duplicate committed predictions, explicit backlog handling, and feed polling that performs no inference. The measured small replay is not evidence of 500–1,000 requests/sec or sub-100 ms model inference. A larger deployment needs its own load and endurance measurements.

Implementation references: [HTTPX timeout semantics](https://www.python-httpx.org/advanced/timeouts/) and [SQLite concurrency/journaling background](https://www.sqlite.org/wal.html). This implementation retains SQLite's rollback journal rather than enabling WAL.
