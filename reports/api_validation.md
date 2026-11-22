# Step 7 validation report

Completed 2026-11-21 on the native Windows development machine.

## Result

The packaged FastAPI service starts with the existing FD001 model release and exposes RUL, conditional survival, full-history anomaly scoring and SHAP explanations. Release `270491b9619aa4a6` includes a frozen 100-window training background and artifact/library fingerprints. Startup validates the release before loading trusted model files and warms every advertised capability.

All **42 tests passed**. The independent notebook regression references agree with the API within their declared tolerances:

- Engine 55 RUL: **120.253466 cycles** at cycle 113, tolerance 0.001 cycles.
- Conditional Cox survival: all returned points match the saved notebook curve, absolute tolerance 1e-10.
- Anomaly: all three detectors and both named policies match saved scores and flags; score tolerance rtol=1e-5, atol=1e-6. Early-cycle null states and causal prefix invariance are checked separately.
- SHAP: a real 500-sample explanation completes, returns all 13 contributions, and preserves signed-sum/reconstruction/gap accounting. The gap is retained rather than hidden.

Tests also cover finite inputs, ordered cycles, missing/extra fields, unsupported datasets/ages, survival support boundaries, malformed JSON, body-size and slow-body protection, authentication, CORS, generic internal errors, readiness, corrupted/missing artifacts, wrong scaler/model shapes and feature order, capacity rejection, and retention of active-work capacity after HTTP timeout or cancellation. The existing notebook/project checks pass too.

## Coverage and reproducibility

`python scripts/test_api_coverage.py` measured **90.71% executable-line coverage** (732/807 lines) across the reusable source package, including the older anomaly module's training/evaluation utilities. This is standard-library line tracing, including worker threads, not branch coverage. The coverage target is met; it does not prove every possible failure path.

Evidence: `api_test_run.txt`, `api_coverage_summary.json`, and annotated files under `api_line_coverage/`. `python -m pip check` reported no broken requirements. Serving dependency versions are pinned in `requirements-serving.txt`; HTTP testing/benchmark dependencies are in `requirements-test.txt`.

## Real HTTP benchmark

The final benchmark ran after the tests completed, using a separate one-worker Uvicorn server on localhost. Configuration: Python 3.11.15, Windows 10, 4 physical/8 logical CPU cores, approximately 16 GiB RAM; TensorFlow intra/inter-op and OMP threads set to 1; two ordinary inference jobs and one explanation job admitted per worker. Requests use engine 55, a 30-cycle LSTM window, 113-cycle anomaly history, and 500 SHAP samples. No response cache.

| Serial endpoint | Requests | Median | p95 | p99 | Successful requests/sec |
|---|---:|---:|---:|---:|---:|
| RUL prediction | 30 | 320 ms | 411 ms | 488 ms | 3.01 |
| Survival | 30 | 10 ms | 14 ms | 15 ms | 93.79 |
| Anomaly | 30 | 202 ms | 260 ms | 295 ms | 4.97 |
| SHAP | 5 | 1,220 ms | 1,393 ms | 1,418 ms | 0.84 |

All serial requests succeeded. Startup to readiness took **21.03 seconds**. Sampled server memory was approximately **495 MiB after readiness** and **524 MiB at peak**. A 30.1-second prediction smoke soak completed **99 requests with no errors**.

At concurrency 2, the mixed predict/survival/anomaly stage accepted all 60 requests and completed about 7.12 requests/second. Intentional concurrency-4 overload produced 5 successful requests and 55 busy responses (503). The mixed explanation overload stage returned 3 successes and 7 busy responses. Busy rejections are reported separately from successful throughput; they are expected capacity protection, not accelerated inference.

Full measurements, payload sizes, first-request times, status counts and hardware metadata are in `api_benchmark.json`. Actual endpoint responses are saved in `api_example_predict.json`, `api_example_survival.json`, `api_example_anomaly.json` and `api_example_explain.json`. The benchmark stops its server on completion.

These are small local samples, especially for SHAP, so tail percentiles are unstable. This is not a long endurance test or a production SLA. The original sub-50 ms RUL/sub-500 ms SHAP and 500 requests/second aspirations are **not achieved by this configuration**. Future optimization should be measured against the same parity tests.

## Remaining deployment checks

- Docker is unavailable here. The supplied non-root Dockerfile, read-only model mount and Compose configuration have **not** been built or executed on Linux. Test platform dependency compatibility and container behavior on a Docker-enabled host.
- Public deployment still requires gateway TLS, rate limiting and access restrictions, plus the supplied bearer-token mode. Per-worker admission limits are implemented; per-user rate limiting belongs at the deployment gateway in this version.
- The short soak does not replace a sustained production load test. Native threaded inference cannot be force-cancelled safely; the supervisor owns the hard shutdown deadline.
- Model limitations remain: FD001-only domain, capped LSTM targets, weak fixed-landmark Cox ranking, and the autoencoder's documented test performance. Serving does not resolve those modeling limitations.

No training notebooks or learned models were changed for Step 7. New serving background/manifest files, packaged inference code, examples, tests, deployment configuration and documentation were added. Pre-existing Git merge conflicts were left untouched.

Start and usage instructions: [API guide](../docs/api_guide.md).
