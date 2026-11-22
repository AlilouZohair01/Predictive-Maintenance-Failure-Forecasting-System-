"""Repeatable localhost HTTP benchmark; launches and stops its own one-worker server.

Not a production SLA or a long soak test. No model fitting or test-set tuning.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import threading
import time

import httpx
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--explanations", type=int, default=5)
    parser.add_argument("--soak-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/api_benchmark.json")
    args = parser.parse_args()
    if args.requests < 1 or args.explanations < 1 or args.soak_seconds < 0:
        parser.error("Counts must be positive and soak duration nonnegative")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PM_MODEL_ROOT": str(ROOT / "models"),
           "PM_AUTH_MODE": "local", "PM_ALLOWED_HOSTS": '["127.0.0.1"]', "PM_INFERENCE_CAPACITY": "2",
           "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "uvicorn", "predictive_maintenance.api.main:create_app", "--factory", "--host", "127.0.0.1",
               "--port", str(port), "--workers", "1", "--no-access-log", "--no-proxy-headers"]
    log = (ROOT / "reports/api_server_benchmark.log").open("w", encoding="utf-8")
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    stop = threading.Event()
    memory = []
    def sample_memory():
        while not stop.wait(0.1):
            try:
                memory.append(psutil.Process(process.pid).memory_info().rss)
            except psutil.Error:
                return
    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    try:
        with httpx.Client(base_url=url, timeout=150, trust_env=False) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Benchmark server failed; inspect reports/api_server_benchmark.log")
                try:
                    if client.get("/health/ready", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - start > 150:
                    raise TimeoutError("Server did not become ready")
                time.sleep(0.2)
            cold_start = time.perf_counter() - start
            baseline_memory = psutil.Process(process.pid).memory_info().rss
            models = client.get("/v1/models").json()
            examples = {name: json.loads((ROOT / f"src/predictive_maintenance/api/examples/{name}.json").read_text())
                        for name in ("predict", "survival", "anomaly", "explain")}
            stages = []

            def run_stage(name, routes, count, concurrency):
                def request(i):
                    route = routes[i % len(routes)]
                    tick = time.perf_counter()
                    try:
                        response = client.post("/v1/" + route, json=examples[route])
                        return (time.perf_counter() - tick) * 1000, response.status_code, route
                    except httpx.HTTPError:
                        return (time.perf_counter() - tick) * 1000, 0, route
                tick = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(pool.map(request, range(count)))
                elapsed = time.perf_counter() - tick
                all_latency = [r[0] for r in results]
                ok_latency = [r[0] for r in results if r[1] == 200]
                stage = {"name": name, "routes": routes, "requests": count, "concurrency": concurrency,
                         "duration_seconds": elapsed, "status_counts": dict(Counter(str(r[1]) for r in results)),
                         "throughput_requests_per_second": count / elapsed,
                         "successful_requests_per_second": len(ok_latency) / elapsed,
                         "latency_ms_all": dict(zip(["p50", "p95", "p99"], np.percentile(all_latency, [50, 95, 99]).tolist())),
                         "latency_ms_success": dict(zip(["p50", "p95", "p99"], np.percentile(ok_latency, [50, 95, 99]).tolist())) if ok_latency else None}
                stages.append(stage)
                print(json.dumps(stage), flush=True)

            first_requests = {}
            for name in examples:
                tick = time.perf_counter()
                response = client.post("/v1/" + name, json=examples[name])
                response.raise_for_status()
                first_requests[name] = (time.perf_counter() - tick) * 1000
                (ROOT / f"reports/api_example_{name}.json").write_text(json.dumps(response.json(), indent=2) + "\n")
            for name in examples:
                run_stage(name + "_serial", [name], args.explanations if name == "explain" else args.requests, 1)
            run_stage("mixed_capacity", ["predict", "survival", "anomaly"], args.requests * 2, 2)
            run_stage("mixed_overload", ["predict", "survival", "anomaly"], args.requests * 2, 4)
            run_stage("explanation_overload", ["explain", "predict"], 10, 4)
            soak_start, soak_count, soak_errors = time.perf_counter(), 0, Counter()
            while time.perf_counter() - soak_start < args.soak_seconds:
                response = client.post("/v1/predict", json=examples["predict"])
                soak_count += 1
                if response.status_code != 200:
                    soak_errors[str(response.status_code)] += 1
            final_memory = psutil.Process(process.pid).memory_info().rss
            output = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "transport": "Real localhost HTTP / Uvicorn",
                      "python": platform.python_version(), "os": platform.platform(), "processor": platform.processor(),
                      "physical_cpu_count": psutil.cpu_count(logical=False), "logical_cpu_count": psutil.cpu_count(),
                      "total_ram_bytes": psutil.virtual_memory().total, "workers": 1,
                      "tf_intra_threads": 1, "tf_inter_threads": 1, "omp_threads": 1,
                      "inference_capacity": 2, "explanation_capacity": 1, "cache": "No response cache",
                      "release_version": models["release_version"], "cold_start_seconds": cold_start,
                      "first_request_latency_ms_after_startup_warmup": first_requests,
                      "history_rows": len(examples["anomaly"]["history"]), "shap_samples": 500,
                      "payload_bytes": {k: len(json.dumps(v).encode()) for k, v in examples.items()},
                      "memory_bytes": {"after_ready": baseline_memory, "after_load": final_memory, "sampled_peak": max(memory + [final_memory])},
                      "short_soak": {"duration_seconds": time.perf_counter() - soak_start, "requests": soak_count, "errors": dict(soak_errors)},
                      "stages": stages,
                      "limitations": ["One machine; small samples, especially SHAP, make p99 unstable.",
                                      "Short bounded smoke soak, not a production endurance or security test.",
                                      "Overload intentionally returns 503; accepted-work capacity is bounded."]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n")
    finally:
        stop.set()
        monitor.join(timeout=2)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
