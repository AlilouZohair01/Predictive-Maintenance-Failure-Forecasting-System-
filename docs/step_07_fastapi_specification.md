# Step 7 — FastAPI inference service: revised project specification

Date: 2026-11-20. Status: implemented and tested on the native Windows environment; container execution remains unverified. See `api_guide.md` and `../reports/api_validation.md` for current instructions and evidence. Future-tense passages below preserve the reviewed design specification.
This replaces the proposed Step 7 description for this repository, based on the current trained artifacts and inference code. It does not modify training or select new models using test results.

## 1. Scope and boundaries

Expose the existing FD001 LSTM, Cox baseline, anomaly detectors and LSTM explanations through typed HTTP endpoints. Reproduce notebook outputs before adding maintenance policy or performance optimizations.

The initial service is stateless: each request supplies the observations needed for its operation. Repeated requests are allowed and do not mutate a unit's state. A new positive unit ID is valid; IDs are identifiers, not model features. Training/test unit IDs are not an authorization registry.

The model domain is NASA C-MAPSS FD001 with its retained sensor definitions and native units. It is not a validated generic industrial IoT service. A `dataset` field fixed to `FD001` documents the contract; it cannot prove that supplied sensor data actually belong to that domain. FD002/FD004 require different preprocessing and calibration. FD003 also has different fault-mode coverage.

The current LSTM test RMSE is 13.36 capped-RUL cycles. The Cox test C-index is 0.576 and its probabilities remain experimental. The autoencoder was selected by validation under the primary anomaly policy but underperformed simpler baselines on test. Serving these models does not establish clinical, industrial or maintenance-policy validity.

## 2. Actual artifacts and preprocessing contracts

| Capability | Artifacts under `models/` | Input and transformation |
|---|---|---|
| LSTM RUL | `lstm_fd001_v2/model.keras`, `scaler.pkl`, `metadata.json` | Exactly 30 consecutive raw cycles, 13 sensors, LSTM MinMaxScaler; tensor `(1,30,13)` |
| Cox survival | `cox_fd001/cox_bundle.pkl`, `metadata.json` | Means of the first 30 cycles, Cox StandardScaler from the bundle; current age used for conditioning |
| Anomaly | `anomaly_fd001/detectors.pkl`, `autoencoder.keras`, `metadata.json` | Anomaly-specific healthy-training StandardScaler; per-cycle IF/z-score and ordered 30-cycle autoencoder windows |
| SHAP | LSTM artifacts plus a new frozen background artifact | Same scaled LSTM tensor; existing GradientExplainer approach |

The three scalers are distinct and must never be interchanged. Feature order comes from each model's metadata, not dictionary insertion order. There is no existing global `feature_cols.json` or global anomaly threshold file to load.

Retained sensor order:
`s_2, s_3, s_4, s_7, s_8, s_11, s_12, s_13, s_14, s_15, s_17, s_20, s_21`.

The LSTM output is a normalized value. Postprocessing must be:

```python
rul_cycles = float(np.clip(prediction_scaled * 125.0, 0.0, 125.0))
```

Clipping a 0–1 output without multiplying by 125 is wrong. The cap is part of training: a result near 125 does not identify the true remaining life beyond that cap. Cox durations are uncapped.

Training invariants such as feature order, sequence length, output scale, anomaly smoothing and fitted thresholds come from versioned metadata. Environment variables must not silently alter them. If deployment supplies expected values, mismatch is a startup error.

## 3. Architecture and layout

Extend the existing package rather than creating a second independent inference implementation:

```text
src/predictive_maintenance/
    anomaly.py                 # Existing scoring/window/alert functions
    rul.py                     # LSTM preprocessing and postprocessing
    survival.py                # Cox feature construction and conditioning
    explanations.py            # GradientExplainer and attribution aggregation
    api/
        main.py                # App factory and lifespan
        config.py              # Environment-based serving settings
        registry.py            # Immutable loaded artifacts and release metadata
        schemas.py             # Typed requests, responses and errors
        errors.py
        middleware.py
        routers/               # Health, metadata, predict, survival, anomaly, explain
scripts/
    prepare_serving_artifacts.py
    benchmark_api.py
tests/
    test_api_schemas.py
    test_api_inference.py
    test_api_endpoints.py
    test_api_startup.py
    fixtures/api/
requirements-serving.txt       # Tested serving dependency lock
pyproject.toml                 # Install the existing src package
Dockerfile
compose.yaml                   # API only initially
.env.example
```

Routes validate and delegate; no arbitrary 5–10-line rule. Services accept validated arrays/plain records and return plain results without FastAPI imports. The registry is created once per application worker during lifespan and injected through app state/dependencies, not a mutable process-global singleton shared by tests.

Reuse `score_cycles` and `apply_threshold` from `anomaly.py` with loaded objects. Do not call `predict_saved` per request: that convenience function currently reloads models each time. Schema validation must reject out-of-order input before the scoring utility's internal sorting can hide it.

## 4. Configuration and release preparation

Environment-configured settings: model release root, log level, allowed hosts, CORS origins, request-size cap, history-length cap, executor limits, explanation capacity, application version, authentication mode and optional maintenance-policy selection.

Proposed initial resource limits, subject to measurement: 1 MiB request body; 1,000 readings per full-history request; one engine per request; explanation concurrency 1 per worker. These are serving limits, not model parameters.

Before serving, create a release manifest with hashes and version IDs for all model/scaler/metadata files, supported capabilities, dataset scope and library versions. API version and model-release version are distinct. A folder name alone is insufficient provenance. The model release is mounted read-only and not changed while a worker is running.

For SHAP, export a fixed representative sample of 100 already-scaled training windows using seed 42, consistent with the corrected LSTM training split. Save `shap_background.npy` and its feature order, scaler/model fingerprints and selection metadata beside the LSTM release. Do not ship the entire training dataset or sample a background at request time. This artifact does not exist yet and is a Step 7 preparation deliverable.

Use existing `autoencoder` and `healthy_q98` anomaly defaults from metadata. The named `validation_f1` alternative and the other detectors may be explicitly requested for comparison; return their identities. Do not silently switch the default to the test-set winner or accept arbitrary per-request numeric thresholds.

## 5. Common validation contract

`SensorReading` contains cycle (positive integer), the 13 required finite numeric sensor values, and optional timezone-aware timestamp. `unit_id` is declared once on the enclosing request. Optional source tags have bounded length. Reject unknown fields at every nesting level.

Reject missing sensors, NaN/infinity, booleans as numeric readings, malformed numeric strings, duplicate/gapped/reversed cycles, mixed units, and non-finite outputs. Preserve explicit feature ordering. Timestamps are optional metadata; cycles define sequence ordering. Do not require a fixed timestamp interval without dataset evidence.

Do not copy invented limits such as temperature -50 to 200 or vibration >=0: these are not this model's feature definitions and units. Document native C-MAPSS units from the reference documentation. Add hard physical bounds only when justified. Training min/max are not hard physics: an observation outside scaler training bounds can be valid degradation and should not automatically be clipped or rejected. Return an out-of-training-range warning where useful; domain checks remain limited.

Input observations are raw, not client-normalized. Unsupported dataset/schema versions fail validation. Caller-generated unit IDs do not need to appear among training IDs.

## 6. Endpoints and response semantics

Business endpoints use `/v1`. Public health routes omit that prefix.

| Method and route | Input | Main output |
|---|---|---|
| GET `/health/live` | None | Process liveness |
| GET `/health/ready` | None | Required capabilities loaded and warmed |
| GET `/v1/models` | None | Model versions, inputs, policy IDs and limitations |
| POST `/v1/predict` | Latest exactly 30 consecutive readings | Capped RUL in cycles |
| POST `/v1/survival` | Exactly cycles 1–30 plus current cycle | Conditional survival and horizon support |
| POST `/v1/anomaly` | Complete consecutive history from cycle 1 | Latest score/flag/alert, optional per-cycle results |
| POST `/v1/explain` | Latest exactly 30 consecutive readings | LSTM SHAP contributions in RUL cycles |

All successful business responses include unit ID, relevant cycle, dataset, model version(s), request ID and clearly defined inference duration. `/docs`, `/redoc` and `/openapi.json` describe all endpoints, examples, defaults, null states and error cases.

### 6.1 RUL prediction

Request: `unit_id`, `dataset=FD001`, `window` of exactly 30 readings; cycles may start later than 1 but must be increasing and consecutive. Prediction refers to the last cycle. Never average across engines or silently pad missing history.

Pipeline: ordered raw matrix -> LSTM scaler.transform -> `(1,30,13)` -> trained model in inference mode -> multiply by 125 -> clip to [0,125]. Validate returned shape and finiteness. A valid input that produces an invalid model result is an internal inference error, not a fabricated zero.

Response: `rul_cycles`, `rul_cap_cycles=125`, `cycle`, optional maintenance band/policy, versions and timing. This endpoint does not promise a sustained anomaly flag from a 30-row payload: it lacks the additional score history required by the anomaly policy. It does not return a calibrated failure probability or prediction interval.

A future `/v1/assess` can combine RUL and anomaly from full history. Do not silently substitute an unsmoothed final-row IF flag for the calibrated anomaly result.

### 6.2 Conditional survival

Request: `unit_id`, `dataset`, `initial_window` containing exactly cycles 1–30, `current_cycle >= 30`, and `horizon_cycles` (positive integer; default 100). `current_cycle` states that the engine is still operating at that cycle. It need not include later sensor readings because this fitted Cox model does not use them.

Pipeline:
1. Compute the 13 sensor means over the initial window.
2. Apply the Cox bundle's own StandardScaler and covariate names/order.
3. Let `a = current_cycle - 30`. Compute future survival as `S(a+k|x)/S(a|x)`.
4. Match notebook Breslow step-function semantics using cumulative-hazard differences, including H(0)=0; avoid changing to interpolated/extrapolated curves accidentally.
5. Return conditional failure probability `1 - S_conditional(K)` for supported requested horizons.

The current model supports at most 332 cycles since its landmark, so the supported future horizon is `332 - a`, read from metadata rather than hard-coded. If `a > 332`, return a structured 422 `SURVIVAL_AGE_UNSUPPORTED`. If the requested horizon extends beyond support, return only supported curve points, the requested/evaluated horizons, `horizon_truncated=true`, and null for unsupported horizon probabilities. At zero supported future horizon, return the point `(0,1)` and no forward probability.

Return `median_remaining_cycles=null` with a reason if the conditional curve does not reach 0.5 within the returned horizon. Never serialize infinity or extrapolate an unsupported median. Curve time is future cycles from the current cycle, not engine age or time from cycle 30.

Return model limitation metadata indicating the fixed early-life covariates and weak measured test ranking. Global sensor hazard ratios (per +1 training SD) belong in `/v1/models` or an optional metadata endpoint, not as new per-engine explanations. A relative partial hazard for one engine, if exposed, is separately named and is not its failure probability.

### 6.3 Anomaly detection

Request: `unit_id`, `dataset`, `history` with consecutive cycles starting at 1, optional detector choice (`autoencoder`, `isolation_forest`, `zscore`), named threshold policy and `return_history=false`. Bound row count and body size before expensive inference. No stored cross-request unit buffer in v1.

Full history is a deliberate correctness-first contract. One reading cannot reproduce a window autoencoder, a five-score rolling mean and three-consecutive-flag alert. Exactly 30 readings at an arbitrary late cycle are also insufficient. A future bounded-context/stateful interface must prove equivalent output before replacing this contract.

Pipeline matches saved code: anomaly StandardScaler; raw detector scores; common evaluation warm-up; five-cycle trailing mean with `min_periods=1`; strict `score > threshold`; three consecutive valid exceedances before issuing an alert. Every operation resets by engine and uses no future observations. IF uses negative `score_samples`, AE uses window reconstruction MSE, and z-score uses mean squared standardized sensor values.

Response distinguishes `raw_score`, `anomaly_score` (smoothed), `threshold`, `is_anomaly` (current exceedance), `sustained_alert`, detector, policy and model version. The score is not constrained to [0,1].

Warm-up semantics: before cycle 30, the common smoothed score and classification are null with status `warmup`. At cycles 30–31, a current flag may be available but a three-flag alert decision lacks sufficient valid history; return a null alert with `alert_history_ready=false`. From cycle 32 onward, return the sustained-alert boolean. Include the count of available smoothing samples. This makes unavailable decisions explicit instead of calling warm-up healthy. Preserve notebook behavior after warm-up.

The earliest current flag is cycle 30 and earliest sustained alert is cycle 32; a complete five-score smoothing window is available at cycle 34. No alert is backdated. The optional history response has one entry per supplied cycle, with nulls where unavailable. No RUL/failure labels are required for inference.

### 6.4 LSTM explanation

Request: the same exactly-30-cycle window as prediction. Use the existing `shap.GradientExplainer` strategy with the frozen training background; KernelExplainer/DeepExplainer are not automatic drop-in replacements.

The input remains `(1,30,13)`, not a single 13-feature row. Multiply attributions and the background mean model output by 125 to express them in cycles. Sum signed contributions across time for each sensor; taking their mean would break the additive accounting.

Response: actual `predicted_rul_cycles`, `base_value_cycles`, all 13 signed sensor contributions sorted by magnitude, top positive/negative contributors, `reconstructed_rul_cycles`, `explanation_gap_cycles = prediction - reconstruction`, explainer method, sample budget, background version and model version. If feature values are shown, explicitly label them as raw window means or last values; the mean is context, not the full sequence input. An optional 30-by-13 attribution map may be added with a bounded response size.

Expected-gradient explanations are approximate. Report the gap; do not force it to zero, rescale contributions to hide it, or call it model prediction error. Set fixed sampling settings consistent with the notebook (initially 500 samples, seed 42). Floating-point differences still require tolerances.

Cache the reusable explainer per worker/model/background configuration, not per unit ID. If caching responses, key by canonical full input, model/background versions, sample budget and configuration; unit ID alone is unsafe. Attach a fresh request ID/timing to every response, including cache hits.

## 7. Maintenance-policy output

Keep predictive evidence separate from a policy recommendation. Do not label a sigmoid of RUL or a band lookup as a calibrated failure probability.

Initial `/predict` may return `maintenance_band` and `policy_version`; omit a numeric `risk_score`. Proposed example bands are red below 20 cycles, amber from 20 up to but excluding 50, green at least 50. If enabled, clearly label these thresholds `illustrative_unvalidated`; they have not been optimized for maintenance costs. Policy thresholds may be deployment-configured and versioned, unlike trained preprocessing invariants.

Do not automatically bump a band for an anomaly in v1. That fusion policy needs validation, especially given the autoencoder's observed false alarms. A future combined endpoint can return independent RUL and anomaly evidence first. A green band does not mean the machine is certified safe.

## 8. Startup, readiness and lifecycle

Use FastAPI's lifespan context manager. On startup, resolve the release manifest, verify configured paths and artifact hashes, load trusted model files once per worker, check library compatibility, validate shapes/scaler feature counts/order/metadata consistency, load the frozen SHAP background and warm required inference paths with representative fixtures.

Pickle/joblib artifacts must be trusted local release files, never user uploads or caller-specified paths. Do not fit scalers, regenerate thresholds, train models, or mutate the mounted release at startup.

Load `.keras` models for inference with compilation disabled where appropriate. Store registry objects in app state and inject them into services. Metadata and threshold policies are read-only during serving. Ensure SHAP execution cannot race over shared random/model state; initially serialize explanation work per worker.

All advertised capabilities are required in the default profile. Missing/corrupt required artifacts cause startup to fail. If a future profile deliberately disables a capability, document it and omit it from required-readiness checks; do not silently claim it is loaded.

`/health/live` reports that the serving process can respond. `/health/ready` returns 200 only when required capabilities have initialized successfully, otherwise 503 when the process is still serving but unready. If startup fails before HTTP serving, expect a startup failure/connection failure, not a guaranteed HTTP 503. Health responses expose safe capability names and versions, not local paths or traces.

Shutdown stops accepting work, drains bounded active tasks within the configured grace period and releases resources. It does not delete model artifacts.

## 9. Concurrency, timeouts and latency

Loading at startup avoids repeated deserialization; it does not guarantee single-millisecond inference. Declaring a CPU-heavy SHAP call `async def` does not make it nonblocking.

Use a bounded thread/executor strategy for synchronous inference, with explicit admission limits. Start with one server worker and benchmark TensorFlow/BLAS thread counts to avoid oversubscription. Each extra process loads its own model copies; worker count is not automatically CPU-core count.

SHAP receives a separate small capacity limit. Protect ordinary predictions and health endpoints from an explanation queue. If measured explanation times are too long for synchronous requests, a later job API may use 202/polling with an external queue. FastAPI BackgroundTasks are not a durable job system.

Define queue, request and shutdown timeouts. Cancelling an HTTP wait does not necessarily stop an underlying ML call; account for work until it actually exits. Hard execution cancellation requires a suitable process/job design. Return a documented busy/unavailable response when capacity is exhausted rather than queueing without bounds.

## 10. Validation errors, middleware and observability

Application-controlled errors use `{error_code, message, details, request_id}`. Use 422 for invalid schema/semantics, 413 for excessive body size, 401/403 for authentication/authorization failures, 404/405 for missing routes/methods, 429 for rate limits, 503 for unavailable capability/capacity, and generic 500 for unexpected faults. Malformed JSON may follow FastAPI's normalized 422 path. Proxy-generated errors may need separate proxy configuration to match the envelope.

Sanitize error details: do not echo Pydantic's entire rejected input, secrets, huge sensor arrays, filesystem paths or stack traces. Unexpected traces go to internal logs. NaN/infinity in outputs is an error except documented missing quantities, which use JSON null plus a reason.

Generate request IDs, or accept only bounded validated IDs supplied by clients. Echo `X-Request-ID` on normal and controlled-error responses. Log method, route template, status, elapsed time, model versions, request ID and appropriate error code, without full sensor bodies. Trust forwarded client information only from configured proxies.

Use configured hosts and CORS origins. Permit the required request headers, including Authorization if auth is enabled, and expose `X-Request-ID` to browser clients. CORS is not authentication. Make credentials/origin behavior explicit.

Bound request bodies even when Content-Length is absent or misleading. Enforce size limits before parsing/building a large tensor, with equivalent reverse-proxy protection for public deployment. Use bounded route/engine metadata rather than per-unit/request Prometheus labels to avoid unbounded metric cardinality.

Collect request/error counts, endpoint latency, inference/queue duration, active jobs and anomaly rates. Treat `/metrics` and detailed model metadata as internal or protected if publicly deployed. A simple trusted bearer credential is sufficient for the first protected service; it does not require implementing OAuth2 password flow. Secrets come from environment or the deployment secret store.

## 11. Tests and acceptance evidence

Schema tests cover finite raw values, required/extra sensors, exact RUL/explanation window length, consecutive ascending cycles, survival initial cycles 1–30, valid current age/horizon, bounded full anomaly history, and unsupported datasets. Use real FD001 example ranges, not invented sensor ranges.

Inference tests cover: feature order; correct scaler per model; LSTM output multiplied by 125; no fitting during inference; Cox conditioning, support, step semantics and null medians; anomaly sign, rolling resets, strict threshold, warm-up and third-flag timing; SHAP signed sums, cycles and gap accounting. Check prefix invariance: appending future readings must not change earlier anomaly scores.

Separate fast tests with controlled doubles from smoke tests using the actual saved artifacts. Golden examples use a frozen evaluation fixture and declared tolerances, not exact floating-point equality. Test-set fixture replay verifies software parity only; it must not become an excuse to retune model/threshold selection on the test set.

Endpoint tests run TestClient within a lifespan context and check successful responses, ID propagation, missing/malformed/oversized inputs, disabled/unavailable paths, internal failures, docs/schema, bounded concurrency and executor saturation. Startup tests verify missing files, wrong feature/scaler order, wrong shapes, threshold policy mismatch and corrupted manifests.

Target at least 85% inference/schema coverage as a useful engineering goal, but prioritize the semantic cases above. Do not waive SHAP tests entirely: test aggregation/shape handling with controlled values and run a small real explanation smoke test. Benchmark cold/warm starts and cached/uncached paths separately.

## 12. Container and dependency plan

Use a pinned Python 3.11 slim runtime to match the current Python 3.11 model environment, then test artifact compatibility in Linux. Lock a serving dependency set compatible with current artifacts (including TensorFlow 2.21.0, sklearn 1.8.0, lifelines 0.30.3, SHAP and Pandas <3). Include FastAPI, Uvicorn, Pydantic and pydantic-settings explicitly. Record the actually tested versions rather than guessing them.

Install the src package; copy only serving code and required manifests. Use a CPU-compatible TensorFlow build whose package/version exists for the selected platform. Do not promise a fixed image size or a 50% reduction. A multi-stage image is optional if compilation is actually needed; deleting a compiler does not remove TensorFlow's runtime weight.

Run as a non-root user on port 8000, mount a complete versioned `models/` release read-only, log to stdout/stderr, and permit a small writable temp/cache area where libraries require it. Do not bake source datasets or notebooks into the serving image. Updating a mounted release should be an explicit rollout/restart, not an in-place overwrite under active workers.

Compose initially runs only the API. Redis, Postgres, a dashboard service, prediction history and `/dashboard/feed` are future features requiring explicit persistence/state contracts. A browser cannot resolve a Compose service name such as `api`; a browser-facing dashboard needs an exposed URL or same-origin proxy. Internal service-to-service traffic can use Compose DNS names.

Docker HEALTHCHECK may call readiness with a startup grace period using an installed tool such as Python urllib. Kubernetes deployments must separately define startup/readiness/liveness probes; do not assume the Dockerfile health check configures them. Use TLS termination and appropriate access controls at the production boundary.

## 13. Benchmarks and definition of done

The original 500 requests/second at 100 concurrent clients, sub-50 ms prediction p99 and sub-500 ms SHAP claims are unmeasured. Anomaly now includes a window autoencoder and history processing, so a universal 5 ms target is also unjustified. Treat performance figures as goals to investigate, not established behavior or arbitrary pass/fail promises.

Record hardware, OS/container resources, dependency versions, worker/thread counts, input-history lengths, SHAP budget, request mix, payload sizes, concurrency, test duration, warm-up, timeouts/errors, throughput, p50/p95/p99 latency and peak/steady memory. Run staged load tests, then a longer soak when feasible. Set service-level targets from measured capacity and dashboard needs; document unmet goals without claiming success.

Step 7 is complete when:
1. A packaged service starts with the current release and validates all required artifacts.
2. The endpoints reproduce the notebook's semantics and saved-model golden fixtures.
3. Strict request/response schemas, null states, support limits and model-domain limits are documented.
4. Missing/corrupt artifacts fail startup; liveness/readiness semantics are verified.
5. Tests cover normal, invalid, overloaded and internal-failure paths with consistent controlled errors.
6. All endpoint examples work through `/docs`, with generated real-data fixtures rather than hand-invented observations.
7. Container execution as non-root and read-only model mounting are tested if Docker is available; otherwise explicitly recorded as unverified.
8. Repeatable benchmark results and a bounded-concurrency configuration are saved, without unsupported throughput or image-size claims.
9. The API exposes experimental model/policy limitations accurately. A future dashboard can consume its documented contract; an end-to-end dashboard claim waits until a dashboard exists.

## 14. Recommended implementation sequence

1. Export the release manifest, SHAP background, representative examples and golden fixtures.
2. Package reusable RUL/survival/explanation services alongside existing anomaly functions.
3. Implement schemas, registry/lifespan, errors and health/model metadata routes.
4. Add prediction, survival and history-based anomaly endpoints and semantic parity tests.
5. Add bounded SHAP execution and explanation tests.
6. Add request IDs, logging, input-size protection and documented auth/CORS settings.
7. Build and verify the CPU serving container, then benchmark and document measured limits.
8. Consider stateful streaming, Redis/database storage and dashboard integration only after the stateless contract is stable.

## Sources and project evidence

Project evidence: `models/*/metadata.json`, `models/*/metrics.json`, `src/predictive_maintenance/anomaly.py`, notebooks 04–06, and `references/cmapss_readme.txt`.

- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/): startup/shutdown resource management.
- [FastAPI concurrency](https://fastapi.tiangolo.com/async/): async versus blocking work.
- [FastAPI deployment concepts](https://fastapi.tiangolo.com/deployment/concepts/): process replication and memory.
- [FastAPI error handling](https://fastapi.tiangolo.com/tutorial/handling-errors/): request validation and exception handlers.
- [Pydantic types](https://docs.pydantic.dev/latest/concepts/types/): validated field types.
- [SHAP GradientExplainer](https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html): approximate expected-gradient attributions.
- [Kubernetes probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/): separate startup, readiness and liveness behavior.
