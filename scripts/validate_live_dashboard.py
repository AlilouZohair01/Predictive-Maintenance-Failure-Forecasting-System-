"""Replay real FD001 observations while polling the real dashboard callbacks.

Uses its own API port/database. Never stops a user's existing API or dashboard.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import httpx

from replay_stream import observations, replay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictive_maintenance.dashboard.app import create_app
from predictive_maintenance.dashboard.config import DashboardSettings
from predictive_maintenance.streaming.store import Store
from predictive_maintenance.streaming.worker import Worker


def main():
    os.chdir(ROOT)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    database = ROOT / "data/runtime/live_dashboard_validation.sqlite3"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PM_MODEL_ROOT": str(ROOT / "models"),
           "PM_STREAM_DB": str(database), "PM_AUTH_MODE": "local", "PM_ALLOWED_HOSTS": '["127.0.0.1"]',
           "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1"}
    started = time.perf_counter()
    with (ROOT / "reports/dashboard_api_live_validation.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "predictive_maintenance.api.main:create_app", "--factory",
            "--host", "127.0.0.1", "--port", str(port), "--no-access-log", "--no-proxy-headers"], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            with httpx.Client(base_url=url, timeout=130, trust_env=False) as api:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Validation API failed startup; inspect its log")
                    try:
                        if api.get("/health/ready", timeout=1).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.perf_counter() - started > 180:
                        raise TimeoutError("API startup timed out")
                    time.sleep(.25)
                print("Real-model API ready", flush=True)
                app = create_app(DashboardSettings(api_url=url, _env_file=None))
                browser = app.server.test_client()
                assert browser.get("/live/").status_code == 200
                assert browser.get("/live/_dash-layout").status_code == 200
                live = app.live_app

                def callback(marker, inputs, states=(), changed=None):
                    output = next(k for k in live.callback_map if marker in k)
                    out = live.callback_map[output]["output"]
                    encode = lambda o: {"id": o.component_id, "property": o.component_property}
                    outputs = [encode(o) for o in out] if isinstance(out, list) else encode(out)
                    values = lambda xs: [{"id": i, "property": p, "value": v} for i, p, v in xs]
                    response = browser.post("/live/_dash-update-component", json={"output": output, "outputs": outputs,
                        "inputs": values(inputs), "state": values(states), "changedPropIds": [changed or f"{inputs[0][0]}.{inputs[0][1]}"]})
                    assert response.status_code == 200, response.text[:500]
                    return response.json["response"]

                run_id = "live-check-" + uuid.uuid4().hex[:10]
                path = ROOT / "data/raw/cmapss/test_FD001.txt"
                source = "FD001-test:" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                rows = observations(path, {55}, 40)
                api.post("/v1/stream/runs", json={"run_id": run_id, "source": source}).raise_for_status()
                store = Store(database)
                worker = Worker(store, api, run_id)
                worker.verify_database()
                for row in rows[:29]:
                    api.post(f"/v1/stream/runs/{run_id}/readings", json=row).raise_for_status()
                    worker.run_once()

                observed = []
                def poll():
                    output = callback("live-ref.data", [("live-run", "value", run_id), ("live-unit", "value", 55),
                        ("live-tick", "n_intervals", len(observed)), ("live-sensors", "value", ["s_11", "s_12"])])
                    ref = output["live-ref"]["data"]
                    assert ref and "token" in ref, str(output)[:500]
                    observed.append(ref["cycle"])
                    print(f"Dashboard refresh: cycle {ref['cycle']}", flush=True)
                    return ref

                assert poll()["cycle"] == 29
                runs = callback("live-run.options", [("live-tick", "n_intervals", 0)], [("live-run", "value", None)])
                assert any(r["value"] == run_id for r in runs["live-run"]["options"])
                def produce():
                    with httpx.Client(base_url=url, timeout=130, trust_env=False) as client:
                        return replay(client, store, run_id, rows, source, interval=.1)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(produce)
                    while not future.done():
                        poll()
                        time.sleep(2)
                    result = future.result()
                ref = poll()
                assert ref["cycle"] == 40 and len(set(observed)) >= 3
                assert result["run"]["progress"] == "completed"
                count = lambda: [line for line in api.get("/metrics").text.splitlines() if line.startswith("pm_inference_seconds_count")]
                before = count()
                for _ in range(3):
                    ref = poll()
                assert before == count(), "Polling must not run inference"
                explained = callback("live-explanation.data", [("live-explain", "n_clicks", 1), ("live-run", "value", run_id),
                    ("live-unit", "value", 55)], [("live-ref", "data", ref)])
                explanation_ref = explained["live-explanation"]["data"]
                assert explanation_ref and explanation_ref["cycle"] == 40
                for kind, signature in (("png", b"\x89PNG"), ("pdf", b"%PDF")):
                    exported = callback("live-download.data", [("live-png", "n_clicks", 1), ("live-pdf", "n_clicks", 1)],
                        [("live-ref", "data", ref), ("live-explanation", "data", explanation_ref), ("live-run", "value", run_id),
                         ("live-unit", "value", 55), ("live-sensors", "value", ["s_11"])], changed=f"live-{kind}.n_clicks")
                    content = base64.b64decode(exported["live-download"]["data"]["content"])
                    assert content.startswith(signature)
                    (ROOT / f"reports/live_dashboard_example.{kind}").write_bytes(content)
                new_app = create_app(DashboardSettings(api_url=url, _env_file=None))
                _, resumed = new_app.live_app.live_service.poll("new-session", run_id, 55)
                assert resumed["cycle"] == 40
                report = {"validated_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                    "observed_dashboard_cycles": observed, "final_progress": result["run"]["progress"],
                    "rul_cycles": resumed["snapshot"]["prediction"]["rul_cycles"], "survival_cycle": resumed["snapshot"]["survival"]["cycle"],
                    "polling_triggered_inference": False, "shap_requested_explicitly": True, "png_pdf_exports": True,
                    "dashboard_restart_recovered_saved_run": True, "elapsed_seconds": time.perf_counter() - started,
                    "verification": "Real HTTP API and saved models, actual replay/worker, Dash callback HTTP tests; not a browser visual test."}
                (ROOT / "reports/live_dashboard_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(report, indent=2), flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
