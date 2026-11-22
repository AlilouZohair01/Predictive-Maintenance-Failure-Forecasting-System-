"""Validate dashboard integration against real models on an isolated API port."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    os.chdir(ROOT)
    from predictive_maintenance.dashboard.config import DashboardSettings
    from predictive_maintenance.dashboard.data import Dataset
    from predictive_maintenance.dashboard.service import DashboardService, ModelClient, DashboardError
    from predictive_maintenance.dashboard.export import render_report
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PM_MODEL_ROOT": str(ROOT / "models"),
           "PM_STREAM_DB": str(ROOT / "data/runtime/dashboard_validation.sqlite3"),
           "PM_AUTH_MODE": "local", "PM_ALLOWED_HOSTS": '["127.0.0.1"]',
           "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1"}
    started = time.perf_counter()
    with (ROOT / "reports/dashboard_api_validation.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "predictive_maintenance.api.main:create_app", "--factory",
            "--host", "127.0.0.1", "--port", str(port), "--no-access-log", "--no-proxy-headers"], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            with httpx.Client(base_url=url, timeout=1, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Validation API stopped; inspect reports/dashboard_api_validation.log")
                    try:
                        if client.get("/health/ready").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.perf_counter() - started > 180:
                        raise TimeoutError("API startup timed out")
                    time.sleep(.25)
            print("Real-model API ready", flush=True)
            service = DashboardService(Dataset(ROOT / "data/raw/cmapss/test_FD001.txt"),
                                       ModelClient(DashboardSettings(api_url=url, _env_file=None)))
            owner = "validation"
            t = time.perf_counter()
            token = service.analyze(owner, 55, 113)
            ref = {"token": token}
            snapshot = service.resolve(owner, 55, 113, ref)
            assert snapshot["survival"] and not snapshot["warnings"]
            analysis_seconds = time.perf_counter() - t
            print(f"Engine 55 cycle 113: RUL {snapshot['prediction']['rul_cycles']:.2f}", flush=True)
            detail = {"token": service.enrich(owner, 55, 113, ref, None, "explain")}
            detail = {"token": service.enrich(owner, 55, 113, ref, detail, "trend")}
            snapshot = service.resolve(owner, 55, 113, ref, detail)
            assert len(snapshot["rul_history"]) == 12 and snapshot["explanation"]
            assert abs(snapshot["explanation"]["predicted_rul_cycles"] - snapshot["prediction"]["rul_cycles"]) < .001
            assert service.resolve(owner, 55, 112, ref, detail) is None
            assert service.resolve("other-session", 55, 113, ref, detail) is None
            warm = {"token": service.analyze(owner, 55, 30)}
            assert service.resolve(owner, 55, 30, warm)["anomaly"]["latest"]["sustained_alert"] is None
            try:
                service.analyze(owner, 55, 29)
                raise AssertionError("Warmup must be rejected")
            except DashboardError:
                pass
            for kind in ("png", "pdf"):
                content = render_report(snapshot, service.dataset, ["s_11"], kind)
                assert content.startswith(b"%PDF" if kind == "pdf" else b"\x89PNG")
                (ROOT / f"reports/dashboard_example.{kind}").write_bytes(content)
            report = {"validated_at": datetime.now(timezone.utc).isoformat(), "release": snapshot["prediction"]["release_version"],
                "unit": 55, "cycle": 113, "rul_cycles": snapshot["prediction"]["rul_cycles"],
                "failure_probability_100": snapshot["survival"]["failure_probability_at_requested_horizon"],
                "anomaly_score": snapshot["anomaly"]["latest"]["anomaly_score"],
                "sustained_alert": snapshot["anomaly"]["latest"]["sustained_alert"],
                "shap_gap_cycles": snapshot["explanation"]["explanation_gap_cycles"],
                "rul_checkpoints": len(snapshot["rul_history"]), "analysis_seconds": analysis_seconds,
                "elapsed_seconds": time.perf_counter() - started, "checks": ["real HTTP inference", "SHAP prediction parity",
                    "12-checkpoint trend", "cycle-30 alert warmup", "cycle-29 inference rejected", "stale selection rejected",
                    "session isolation", "PNG and PDF reports"]}
            (ROOT / "reports/dashboard_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
