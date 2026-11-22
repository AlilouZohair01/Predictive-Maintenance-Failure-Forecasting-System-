"""Replay raw FD001 test observations, optionally processing the durable queue.

Run the API in another terminal first. Uses no RUL/failure labels.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictive_maintenance.api.config import Settings
from predictive_maintenance.api.schemas import SENSORS
from predictive_maintenance.streaming.store import Store
from predictive_maintenance.streaming.worker import Worker, client_for


def observations(path, units, max_cycle=None):
    rows = []
    for line in path.read_text().splitlines():
        values = line.split()
        if not values:
            continue
        unit, cycle = int(values[0]), int(values[1])
        if unit not in units or (max_cycle is not None and cycle > max_cycle):
            continue
        reading = {"cycle": cycle, **{s: float(values[4 + int(s[2:])]) for s in SENSORS}}
        rows.append({"unit_id": unit, "reading": reading})
    present = {r["unit_id"] for r in rows}
    if present != set(units):
        raise ValueError("Some selected engine IDs are absent from the selected FD001 data")
    return sorted(rows, key=lambda r: (r["reading"]["cycle"], r["unit_id"]))


def replay(client, store, run_id, rows, source, interval=0, ingest_only=False, drain_timeout=600):
    response = client.post("/v1/stream/runs", json={"run_id": run_id, "source": source})
    response.raise_for_status()
    worker = None if ingest_only else Worker(store, client, run_id)
    if worker:
        worker.verify_database()
    accepted, duplicates = 0, 0
    for message in rows:
        deadline = time.monotonic() + drain_timeout
        while True:
            response = client.post(f"/v1/stream/runs/{run_id}/readings", json=message)
            if response.status_code != 429:
                response.raise_for_status()
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Backlog remains full. Start the worker or repair failed jobs, then resume this run ID.")
            if worker:
                worker.run_once()
            time.sleep(0.5)
        disposition = response.json()["disposition"]
        accepted += disposition == "accepted"
        duplicates += disposition == "duplicate"
        if worker:
            worker.run_once()
        if interval:
            time.sleep(interval)
    client.post(f"/v1/stream/runs/{run_id}/close").raise_for_status()
    if worker:
        deadline = time.monotonic() + drain_timeout
        while True:
            run = store.run(run_id)
            if run["jobs"]["failed"]:
                raise RuntimeError("Run has failed jobs. Inspect the latest feed error, correct its cause, and use the retry endpoint.")
            if not run["jobs"]["pending"] + run["jobs"]["processing"]:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Drain deadline reached. Durable jobs remain; restart the worker to continue.")
            if not worker.run_once():
                time.sleep(0.25)
    result = client.get(f"/v1/stream/runs/{run_id}")
    result.raise_for_status()
    return {"run": result.json()["run"], "new_readings": accepted, "duplicate_readings": duplicates}


def main():
    parser = argparse.ArgumentParser(description="Replay FD001 sensor data through Step 8")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-id", default=None, help="Reuse after interruption; use a new ID for a new experiment")
    parser.add_argument("--units", nargs="+", type=int, default=[55])
    parser.add_argument("--max-cycle", type=int, default=None, help="Optional bounded demonstration")
    parser.add_argument("--interval", type=float, default=0.2, help="Delay after each observation; processing also takes time")
    parser.add_argument("--ingest-only", action="store_true", help="Use a separately running worker")
    args = parser.parse_args()
    if args.interval < 0 or (args.max_cycle is not None and not 1 <= args.max_cycle <= 1000):
        parser.error("Interval must be nonnegative and max-cycle must be between 1 and 1000")
    settings = Settings()
    path = ROOT / "data/raw/cmapss/test_FD001.txt"
    rows = observations(path, set(args.units), args.max_cycle)
    source = "FD001-test:" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    run_id = args.run_id or "replay-" + uuid.uuid4().hex[:12]
    print(f"Run ID: {run_id} (save this to resume after an interruption)", flush=True)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    store = None if args.ingest_only else Store(settings.stream_db, settings.stream_max_pending, settings.max_history_rows)
    try:
        with client_for(args.api_url, settings) as client:
            result = replay(client, store, run_id, rows, source, args.interval, args.ingest_only)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print(f"Stopped. Accepted readings are saved. Resume with --run-id {run_id} and the same options.")


if __name__ == "__main__":
    main()
