# Step 8 validation

Completed 2026-11-21 on the native Windows environment.

## Automated checks

**59 tests passed**: the previous 42 checks plus 17 streaming checks. Evidence: `step08_test_run.txt`.

The new checks cover duplicate/conflicting observations, gaps and closed runs, concurrent duplicate delivery, persisted run identity, queue/history limits, worker leases and stale-owner rejection, per-engine failure blocking, manual retries, atomic result/alert rollback, alert hysteresis/cooldown/clearance, freshness/lag, cursor reads, authenticated endpoints, input errors, causal input selection, API failure retries, and model-release mismatch protection.

The old Step 7 coverage report is a historical measurement of the earlier source tree; it is not a coverage percentage for the added streaming code. No new overall coverage percentage is claimed here.

## Native HTTP replay and restart

`python scripts/validate_stream.py` ran a separate one-worker Uvicorn process on a temporary localhost port with the real saved model release. The user's server was not stopped. TensorFlow intra/inter-op and OMP thread counts were each set to one.

- Replayed **66 raw observations** from test engines 1 and 55, through cycles 31 and 35 respectively.
- Stored **58 warm-up snapshots and 8 cycles with model predictions**. No failure labels were loaded by the replay.
- Stored RUL, anomaly and survival results matched direct calls to the same API. RUL tolerance was 0.001 cycles; anomaly score tolerance was 1e-6; survival probability tolerance was 1e-10.
- Engine 1 at cycle 31: RUL **115.674570 cycles**, anomaly score **0.424264**.
- Engine 55 at cycle 35: RUL **122.849114 cycles**, anomaly score **0.407989**.
- Both latest survival curves correctly retained their cycle-30 timestamp instead of being relabeled as current-cycle estimates.
- Stopped and restarted the validation API with the same database. Previously committed results remained available.
- Redelivered all **66 observations** after restart. All were acknowledged as duplicates, with no new jobs, predictions or alert episodes.
- Final queue: **66 done, 0 pending, 0 processing, 0 failed**.
- Twenty feed reads caused **no additional inference calls**.

These early-life records did not trigger alerts. The real replay therefore validates history/inference/feed integration; synthetic controlled tests separately validate alert opening, repeated-condition suppression, resolution, cooldown and transactional rollback. No external notifications were sent.

## Measurements

| Measurement | Observed |
|---|---:|
| API startup to readiness | 18.99 seconds |
| API restart to readiness | 18.22 seconds |
| Replay plus processing, excluding startup and restart | 6.86 seconds |
| Observations/sec, including inexpensive warm-up records | 9.62 |
| Ready prediction cycles/sec over that same complete replay | 1.17 |
| Two-engine feed median latency | 10.11 ms |
| Two-engine feed p95 / p99 | 13.01 / 13.40 ms |

These figures describe a short local smoke test, not production capacity. In particular, the observation throughput includes 58 records requiring no inference, and the feed's latency does not represent model execution time. The earlier Step 7 benchmark remains the separate model-endpoint performance reference.

Machine: Windows 10, Python 3.11.15. Persistent validation data is in `data/runtime/stream_validation.sqlite3` (ignored by Git). Exact run ID, timings and result checks are in `stream_validation.json`; a sample dashboard feed is in `stream_example_feed.json`.

## Scope and remaining work

Step 8's local replay, durable HTTP intake, model worker, history, current-result storage, in-application alerts and feed endpoints are implemented. The graphical dashboard is still Step 9.

MQTT/Kafka, Redis, external email/Slack notification delivery and real hardware integration were deliberately deferred in the agreed initial design. Docker/Compose configuration now includes persistent state and an optional worker, but container execution remains unverified because Docker is unavailable. Long-running endurance, multi-host operation, automated retention/backups and policy calibration remain deployment work.

No trained models, scalers, learned thresholds or training notebooks were changed. Pre-existing Git conflicts were left untouched.
