"""Start the local historical dashboard and reuse or start its model API."""
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    os.chdir(ROOT)
    import httpx
    from waitress import serve
    from predictive_maintenance.dashboard.app import create_app
    from predictive_maintenance.dashboard.config import DashboardSettings

    settings = DashboardSettings()
    address = urlsplit(settings.api_url)
    process, log = None, None
    headers = {"Authorization": "Bearer " + settings.api_token.get_secret_value()} if settings.api_token else {}
    try:
        with httpx.Client(base_url=settings.api_url, headers=headers, timeout=2, trust_env=False) as client:
            try:
                response = client.get("/health/ready")
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if address.hostname not in ("127.0.0.1", "localhost") or address.scheme != "http":
                    raise RuntimeError("Start the configured remote model API before launching the dashboard.") from None
                log_path = ROOT / "data/runtime/dashboard_api.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a", encoding="utf-8")
                env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "TF_NUM_INTRAOP_THREADS": "1",
                       "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1"}
                process = subprocess.Popen([sys.executable, "-m", "uvicorn", "predictive_maintenance.api.main:create_app",
                    "--factory", "--host", address.hostname, "--port", str(address.port or 80), "--workers", "1",
                    "--no-access-log", "--no-proxy-headers"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                print("Loading the trained models. First startup can take about a minute…", flush=True)
                response = None
            except httpx.ReadTimeout:
                # An existing API may still be initializing; do not start a second one.
                response = None
            deadline = time.monotonic() + 180
            while response is None or response.status_code == 503:
                if process and process.poll() is not None:
                    raise RuntimeError("Model API startup failed; inspect data/runtime/dashboard_api.log.")
                if time.monotonic() > deadline:
                    raise RuntimeError("Model API startup timed out; inspect data/runtime/dashboard_api.log.")
                time.sleep(.5)
                try:
                    response = client.get("/health/ready")
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                    response = None
            if response.status_code != 200:
                raise RuntimeError(f"Model API is not ready (HTTP {response.status_code}); check its address and credentials.")
        app = create_app(settings)
        print(f"Open http://{settings.host}:{settings.port} - Historical analysis and Live replay - Ctrl+C to stop", flush=True)
        serve(app.server, host=settings.host, port=settings.port, threads=4, max_request_body_size=1_048_576)
    except KeyboardInterrupt:
        print("Dashboard stopped.")
    finally:
        # Never stop an API process that was already running before this launcher.
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log:
            log.close()


if __name__ == "__main__":
    main()
