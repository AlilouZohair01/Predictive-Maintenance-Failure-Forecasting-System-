"""Real HTTP replay, parity, duplicate-delivery and API-restart validation.

Starts its own server on a temporary port; never stops the user's API process.
Keeps the validation database under data/runtime and writes a compact report.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import uuid

import httpx
import numpy as np

from replay_stream import ROOT, observations, replay
from predictive_maintenance.streaming.store import Store


def main():
    database = ROOT / "data/runtime/stream_validation.sqlite3"
    store = Store(database)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PM_STREAM_DB": str(database), "PM_MODEL_ROOT": str(ROOT / "models"),
           "PYTHONPATH": str(ROOT / "src"), "PM_AUTH_MODE": "local", "PM_ALLOWED_HOSTS": '["127.0.0.1"]',
           "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "uvicorn", "predictive_maintenance.api.main:create_app", "--factory",
               "--host", "127.0.0.1", "--port", str(port), "--workers", "1", "--no-access-log", "--no-proxy-headers"]
    log = (ROOT / "reports/stream_server_validation.log").open("w", encoding="utf-8")
    process = None

    def start_server():
        nonlocal process
        started = time.perf_counter()
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        with httpx.Client(base_url=url, timeout=1, trust_env=False) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Validation API failed startup; inspect the validation server log")
                try:
                    if client.get("/health/ready").status_code == 200:
                        return time.perf_counter() - started
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 150:
                    raise TimeoutError("Validation API did not start")
                time.sleep(0.2)

    def stop_server():
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    try:
        startup = start_server()
        path = ROOT / "data/raw/cmapss/test_FD001.txt"
        rows = observations(path, {1, 55}, max_cycle=35)
        source = "FD001-test:" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        run_id = "validation-" + uuid.uuid4().hex[:12]
        with httpx.Client(base_url=url, timeout=130, trust_env=False) as client:
            started = time.perf_counter()
            result = replay(client, store, run_id, rows, source)
            replay_seconds = time.perf_counter() - started
            assert result["run"]["progress"] == "completed"
            assert result["run"]["jobs"]["done"] == len(rows)
            print(json.dumps({"phase": "replay_complete", "readings": len(rows), "seconds": replay_seconds}), flush=True)
            parity = []
            for unit in (1, 55):
                history = [r["reading"] for r in rows if r["unit_id"] == unit]
                current = client.get(f"/v1/stream/runs/{run_id}/units/{unit}/latest").json()
                direct = client.post("/v1/predict", json={"unit_id": unit, "window": history[-30:]}).json()
                direct_anomaly = client.post("/v1/anomaly", json={"unit_id": unit, "history": history}).json()
                saved = current["latest"]
                np.testing.assert_allclose(saved["prediction"]["rul_cycles"], direct["rul_cycles"], atol=1e-3)
                np.testing.assert_allclose(saved["anomaly"]["latest"]["anomaly_score"], direct_anomaly["latest"]["anomaly_score"], atol=1e-6)
                assert saved["anomaly"]["latest"]["sustained_alert"] == direct_anomaly["latest"]["sustained_alert"]
                survival = client.post("/v1/survival", json={"unit_id": unit, "initial_window": history[:30],
                                       "current_cycle": current["latest_survival"]["cycle"]}).json()
                np.testing.assert_allclose([p["survival_probability"] for p in current["latest_survival"]["curve"]],
                                           [p["survival_probability"] for p in survival["curve"]], atol=1e-10)
                parity.append({"unit_id": unit, "cycle": len(history), "rul_cycles": saved["prediction"]["rul_cycles"],
                               "anomaly_score": saved["anomaly"]["latest"]["anomaly_score"], "survival_cycle": current["latest_survival"]["cycle"], "parity_passed": True})
            inference_counts = lambda: [line for line in client.get("/metrics").text.splitlines() if line.startswith("pm_inference_seconds_count")]
            before = inference_counts()
            times = []
            for _ in range(20):
                tick = time.perf_counter()
                response = client.get(f"/v1/stream/runs/{run_id}/units")
                response.raise_for_status()
                times.append((time.perf_counter() - tick) * 1000)
            assert before == inference_counts(), "Feed reads must not run inference"
            alerts_before = store.alerts(run_id, "all")
        stop_server()
        restart = start_server()
        with httpx.Client(base_url=url, timeout=130, trust_env=False) as client:
            resumed = replay(client, Store(database), run_id, rows, source)
            assert resumed["new_readings"] == 0 and resumed["duplicate_readings"] == len(rows)
            assert resumed["run"]["jobs"]["done"] == len(rows)
            assert store.alerts(run_id, "all") == alerts_before
            feed = client.get(f"/v1/stream/runs/{run_id}/units").json()
            assert all(r["processing_lag_cycles"] == 0 for r in feed["items"])
        ready_cycles = sum(max(0, len([r for r in rows if r["unit_id"] == u]) - 29) for u in (1, 55))
        report = {"validated_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                  "database": database.relative_to(ROOT).as_posix(), "os": platform.platform(), "python": platform.python_version(),
                  "workers": 1, "tf_intra_threads": 1, "tf_inter_threads": 1, "omp_threads": 1,
                  "startup_seconds": startup, "restart_seconds": restart, "accepted_readings": len(rows), "ready_cycle_predictions": ready_cycles,
                  "replay_seconds": replay_seconds, "observations_per_second_including_warmup": len(rows) / replay_seconds,
                  "end_to_end_ready_cycles_per_second_including_warmup": ready_cycles / replay_seconds,
                  "parity": parity, "feed_latency_ms": dict(zip(("p50", "p95", "p99"), np.percentile(times, [50, 95, 99]).tolist())),
                  "feed_triggered_inference": False, "duplicate_deliveries": resumed["duplicate_readings"],
                  "api_restart_preserved_results": True, "final_jobs": resumed["run"]["jobs"],
                  "alert_episodes": len(alerts_before), "external_notifications_sent": 0,
                  "limits": "Short two-engine native HTTP smoke test, not a production throughput or hardware-stream validation."}
        (ROOT / "reports/stream_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (ROOT / "reports/stream_example_feed.json").write_text(json.dumps(feed, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        stop_server()
        log.close()


if __name__ == "__main__":
    main()
