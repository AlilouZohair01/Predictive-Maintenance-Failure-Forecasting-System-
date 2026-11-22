# Step 7: run the FD001 API

The inference endpoints expose the existing trained models through FastAPI without retraining them or changing model selection. Use raw FD001 sensor readings in native units. Step 8 now adds separate durable ingestion/history routes under `/v1/stream`; see [the streaming guide](step_08_streaming.md).

## Start on this computer

From the project root, activate the existing environment. The local package has already been installed into `exam_env` during implementation.

```powershell
conda activate exam_env
# Needed when installing the project on another computer:
python -m pip install -r requirements-serving.txt
python -m pip install --no-deps --no-build-isolation -e .

# Create the local configuration once; do not overwrite your configured .env later.
Copy-Item .env.example .env
python -m uvicorn predictive_maintenance.api.main:create_app --factory --env-file .env --host 127.0.0.1 --port 8000 --workers 1 --no-access-log --no-proxy-headers --timeout-graceful-shutdown 150
```

Open <http://127.0.0.1:8000/docs>. Each inference route includes a complete real-data example for FD001 test engine 55: select **Try it out**, then **Execute**. `/redoc` and `/openapi.json` provide the same contract. Models load and warm once during startup, which can take tens of seconds. Wait for “Application startup complete.” Stop the server with Ctrl+C.

Local mode has no authentication and is intended for loopback access. The server is not left running automatically after tests or benchmarks.

## Endpoints

| Route | Input | Output |
|---|---|---|
| `GET /health/live` | None | Process responds |
| `GET /health/ready` | None | All four model capabilities loaded and warmed |
| `GET /v1/models` | None | Release/model fingerprints, sensor order, limits, global Cox hazard ratios and model limitations |
| `POST /v1/predict` | Exactly 30 consecutive readings in `window` | RUL at the last cycle, multiplied by 125 and capped to 0–125 cycles |
| `POST /v1/survival` | Cycles 1–30 in `initial_window`, `current_cycle`, optional `horizon_cycles` | Conditional survival from the current cycle, supported failure probability and nullable median |
| `POST /v1/anomaly` | Complete consecutive `history` starting at cycle 1 | Raw/smoothed score, current flag, sustained alert, optional per-cycle history |
| `POST /v1/explain` | Same window as prediction | 13 signed sensor SHAP contributions in cycles, baseline, reconstruction and approximation gap |
| `GET /metrics` | None | Prometheus request, latency, active-work and anomaly-count metrics |

Each request describes one engine. `unit_id` is a positive integer and may be new; it is not a learned feature or an authorization identity. `dataset` is fixed to `FD001`. Input order must already be chronological: duplicates, gaps and reversed rows fail validation. Unknown fields, missing sensors, numeric strings, booleans masquerading as numbers, NaN and infinity are rejected. Optional timestamps require a timezone; cycles determine ordering. Raw readings outside the training scaler range are allowed, with a warning for RUL, and are not clipped to training limits.

Responses carry `request_id`, `model_version`, `release_version`, cycle and engine ID. `inference_duration_ms` measures execution inside the accepted worker, including model preprocessing and result construction, excluding HTTP parsing/serialization and executor scheduling. HTTP timings are separately logged and measured in benchmarks. There is no response cache and no waiting admission queue.

## Send the supplied examples

```powershell
$base = 'http://127.0.0.1:8000'
$body = Get-Content 'src/predictive_maintenance/api/examples/predict.json' -Raw
Invoke-RestMethod -Method Post -Uri "$base/v1/predict" -ContentType 'application/json' -Body $body

$body = Get-Content 'src/predictive_maintenance/api/examples/anomaly.json' -Raw
Invoke-RestMethod -Method Post -Uri "$base/v1/anomaly" -ContentType 'application/json' -Body $body
```

Equivalent files are supplied for `survival` and `explain`. They contain measured sensor observations, not invented numbers. The benchmark saves complete working responses as `reports/api_example_*.json`; timing and request IDs vary between runs.

## Model-specific interpretation

**RUL:** the LSTM uses its own fitted MinMaxScaler and a `(1,30,13)` tensor. A prediction near 125 means the model is at its training cap, not that true remaining life is known to be exactly 125 cycles. No prediction interval, numeric risk score or maintenance band is fabricated.

**Survival:** the Cox model uses its own StandardScaler and sensor means from cycles 1–30. Later sensor readings are not model inputs. `current_cycle` asserts that the engine is still operating then. Future survival is computed from differences in the fitted Breslow cumulative hazard, preserving notebook step semantics. The maximum supported current cycle is 362 (landmark 30 plus 332 follow-up cycles). Later ages return `SURVIVAL_AGE_UNSUPPORTED`. Horizons extending beyond support are truncated; the probability at an unsupported requested horizon is null. At cycle 362 the curve is only `(0,1)`. A median is null when the returned curve never falls to 0.5. This is a weak experimental baseline (test C-index about 0.576), not a calibrated operational failure-probability guarantee.

**Anomaly:** default detector `autoencoder`, default policy `healthy_q98`, both fixed by validation metadata. Optional detectors are `isolation_forest` and `zscore`; optional policy is `validation_f1`. Numeric threshold overrides are not accepted. Scoring reuses the existing notebook utility with preloaded artifacts and the anomaly-specific StandardScaler. Full history is required to reproduce the window, trailing mean and alert state.

| Cycle | Score and current flag | Sustained alert | Smoothing samples |
|---|---|---|---|
| 1–29 | null, `status=warmup` | null | 0 |
| 30 | Available | null | 1 |
| 31 | Available | null | 2 |
| 32–33 | Available | Boolean | 3–4 |
| 34 onward | Available | Boolean | 5 |

The alert requires three consecutive **strict** threshold exceedances. Appending later readings does not change earlier scores. Scores are not probabilities. The validation-selected autoencoder performed worse than the simpler baselines on the test set; the API preserves the original selection rather than selecting a new winner using test labels.

**SHAP:** a frozen sample of 100 scaled training windows, seed 42, supports a reusable GradientExplainer. Each request uses 500 samples and seed 42; explanation calls are serialized. Contributions are multiplied by 125 and summed across time for each sensor. `raw_window_mean` gives sensor context, not the complete sequence feature. The reported gap is `prediction - (baseline + contributions)`; it is approximation error, not prediction error. Positive contributions raise predicted RUL relative to the background. SHAP values do not establish physical causation.

## Sensor identities and native units

The retained indices refer to the original C-MAPSS sensor numbering; they are not generic temperature/vibration slots. The source paper's Table 2 describes these measurements. Consult `references/cmapss_damage_propagation.pdf` and the [NASA source record](https://ntrs.nasa.gov/citations/20090029214) before mapping any external telemetry.

| Sensor | C-MAPSS measurement | Native unit |
|---|---|---|
| s_2 | T24, LPC outlet total temperature | degrees Rankine |
| s_3 | T30, HPC outlet total temperature | degrees Rankine |
| s_4 | T50, LPT outlet total temperature | degrees Rankine |
| s_7 | P30, HPC outlet total pressure | psia |
| s_8 | Nf, physical fan speed | rpm |
| s_11 | Ps30, HPC outlet static pressure | psia |
| s_12 | phi, fuel-flow to Ps30 ratio | pps/psi |
| s_13 | NRf, corrected fan speed | rpm |
| s_14 | NRc, corrected core speed | rpm |
| s_15 | BPR, bypass ratio | dimensionless |
| s_17 | htBleed, bleed enthalpy indicator | source table gives no unit |
| s_20 | W31, HPT coolant bleed | lbm/s |
| s_21 | W32, LPT coolant bleed | lbm/s |

The dataset's units are retained without conversion. This API does not justify applying FD001 models to other C-MAPSS subsets or real equipment.

## Releases and reproducibility

`models/serving_manifest.json` fingerprints each required model, scaler, metadata file and SHAP background, plus the tested scientific-library versions. The registry checks all hashes **before** loading trusted pickle/joblib files, validates sensor order, model/scaler shapes and temporal policies, and warms inference. Hashes detect accidental drift; they are not a signature from an independent trusted publisher. Never load model bundles from API callers or untrusted uploads.

The scientific-library versions must exactly match the manifest. `requirements-serving.txt` pins the installed dependency closure; it is not a package-hash lock or proof of Linux compatibility. Missing, changed or incompatible files fail startup rather than yielding misleading readiness. `/health/ready` can return 503 while a running app is unready, but a startup failure normally means no listening server.

The release and examples already exist locally. To explicitly prepare a **new release after a deliberate model change**, run:

```powershell
python scripts/prepare_serving_artifacts.py
```

This exports the frozen background, records source/training-index provenance, fingerprints artifacts, and refreshes examples and evaluation fixtures. It requires prepared training windows and the saved notebook evaluation reports. It does not fit models. Do not run it simply to hide a failed regression test, and never run it automatically on server startup. Serving needs the release, not source datasets. Binary artifacts are ignored by Git, so distribute the full release separately when cloning or deploying.

## Limits, errors and security

Defaults: 1 MiB body, 1,000 history rows, two ordinary inference jobs and one explanation job per worker. Body receipt has a 15-second deadline; inference responses have a 120-second deadline. There is no waiting admission queue: exhausted capacity returns 503. If HTTP times out or disconnects, the underlying native job remains counted until it finishes. Threaded native ML cannot be force-cancelled safely. Shutdown stops admission and waits up to 130 seconds; a process supervisor must enforce a hard outer deadline for hung native work.

Application-controlled errors use:

```json
{"error_code":"INVALID_REQUEST","message":"Request does not match the endpoint contract.","details":[],"request_id":"..."}
```

Validation/semantic errors return 422, excessive bodies 413, slow bodies 408, invalid hosts 400, invalid credentials 401, unavailable/busy/timed-out inference 503 and unexpected failures a generic 500. Unknown routes and methods return 404/405 in the same envelope. Input values and traces are not returned. A caller's bounded `X-Request-ID` is retained, otherwise a UUID is generated; it is echoed on controlled responses. JSON access logs and metric labels exclude full sensor bodies and engine IDs.

For access beyond localhost:

- Set `PM_AUTH_MODE=bearer` and a random `PM_API_TOKEN` of at least 24 characters. Use the **Authorize** button in `/docs`, or send `Authorization: Bearer <token>`. Health endpoints remain public; business routes and `/metrics` require the credential. Interactive schema pages contain no secret but remain accessible unless restricted by the gateway.
- Restrict `PM_ALLOWED_HOSTS` and `PM_CORS_ORIGINS` (JSON arrays). CORS permits no cookies/credentials and exposes `X-Request-ID`; it does not replace authentication.
- Put TLS, external rate limiting, request-size/time limits and restricted access to docs/metrics at the gateway. Application capacity protection is not a per-user rate limiter. Gateway-generated 429/error bodies are configured there.
- Keep a single worker initially. Every additional process loads another full model release. Do not overwrite a mounted release under a running worker; deploy a new release and restart.

## Container

```powershell
docker compose up --build
```

The CPU-oriented Python 3.11 image runs as UID/GID 10001, binds port 8000, includes serving code only, and mounts `models/` read-only. Compose publishes to localhost, gives libraries a writable `/tmp`, and mounts a persistent `/state` volume for Step 8's SQLite database; the rest of the container filesystem is read-only. An optional streaming worker profile is available. There are no separate database-server, Redis or dashboard services. The image uses readiness for its Docker health check; Kubernetes startup/readiness/liveness probes would be configured separately.

**Docker was not available on the implementation machine.** Image build, Linux dependency compatibility, non-root execution and the read-only container runtime remain unverified. The configurations are supplied for this next deployment check; native Windows HTTP inference is tested. If a platform needs a different dependency set, validate the artifacts there and prepare a separately versioned manifest rather than disabling version checks.

## Tests and benchmarks

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python scripts/test_api_coverage.py
python scripts/benchmark_api.py --requests 30 --explanations 5 --soak-seconds 30
```

Fast tests cover strict schemas, model transformations, error envelopes, body-size enforcement without a Content-Length header, authentication, startup failures, capacity and timeout/cancellation accounting. Integration tests load the actual saved release and compare against frozen notebook RUL, Cox and anomaly outputs, exercise warm-up/prefix invariance, and run a real 500-sample explanation. Fixtures are regression references, not threshold-tuning data.

The benchmark launches its own one-worker Uvicorn process on a temporary localhost port and stops it afterward. Run it without other training or test processes competing for CPU. It records real HTTP latency, status counts, throughput, startup time, sampled server memory and a short soak in `reports/api_benchmark.json`. Serial stages, admitted concurrency and deliberate overload are separated. Results count successful requests separately from fast busy rejections. This is not a production SLA, a 500-requests/second claim, or a long endurance test. No response caching is implemented.

See `reports/api_validation.md` for the measured implementation results and remaining deployment checks.
