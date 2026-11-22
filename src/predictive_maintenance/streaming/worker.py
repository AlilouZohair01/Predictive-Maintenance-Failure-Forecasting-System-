"""Separate synchronous durable worker calling the existing inference HTTP API."""
import argparse
import json
import logging
import time

import httpx
from pydantic import ValidationError
from ..api.schemas import PredictionResponse, AnomalyResponse, SurvivalResponse
from ..api.config import Settings
from .store import Store, StreamError

LOG = logging.getLogger("predictive_maintenance.stream")


class InferenceFailure(Exception):
    def __init__(self, code, retryable):
        self.code, self.retryable = code, retryable


class Worker:
    def __init__(self, store, client, run_id=None, max_attempts=5):
        self.store, self.client, self.run_id, self.max_attempts = store, client, run_id, max_attempts

    def verify_database(self):
        response = self.client.get("/v1/stream/status")
        response.raise_for_status()
        if response.json()["store_id"] != self.store.store_id:
            raise RuntimeError("Worker and API use different stream databases. Set PM_STREAM_DB to the same local file.")

    def infer(self, job, endpoint, payload, schema, release):
        self.store.renew(job)
        try:
            response = self.client.post("/v1/" + endpoint, json=payload)
        except httpx.HTTPError:
            raise InferenceFailure("INFERENCE_CONNECTION_ERROR", True) from None
        if response.status_code != 200:
            raise InferenceFailure("INFERENCE_HTTP_" + str(response.status_code), response.status_code in (408, 429, 500, 502, 503, 504))
        try:
            data = schema.model_validate(response.json()).model_dump(mode="json")
        except (ValueError, ValidationError):
            raise InferenceFailure("INVALID_INFERENCE_RESPONSE", False) from None
        if data["release_version"] != release:
            raise InferenceFailure("MODEL_RELEASE_CHANGED", False)
        if data["unit_id"] != job["unit_id"] or data["cycle"] != job["cycle"]:
            raise InferenceFailure("INFERENCE_IDENTITY_MISMATCH", False)
        if endpoint == "anomaly" and data["latest"]["cycle"] != job["cycle"]:
            raise InferenceFailure("INFERENCE_IDENTITY_MISMATCH", False)
        return data

    def process(self, job):
        start = time.perf_counter()
        run = self.store.run(job["run_id"])
        rows = self.store.history(job["run_id"], job["unit_id"], through=job["cycle"])
        history = [r["reading"] for r in rows]
        if len(history) != job["cycle"]:
            raise InferenceFailure("INCOMPLETE_STORED_HISTORY", False)
        cycle, release = job["cycle"], run["release_version"]
        snapshot = {"run_id": job["run_id"], "unit_id": job["unit_id"], "cycle": cycle, "dataset": "FD001",
                    "release_version": release, "status": "warmup" if cycle < 30 else "ready",
                    "observed_at": history[-1].get("timestamp"), "received_at": rows[-1]["received_at"],
                    "prediction": None, "anomaly": None, "survival": None, "survival_status": "warmup"}
        if cycle >= 30:
            common = {"unit_id": job["unit_id"], "dataset": "FD001"}
            snapshot["prediction"] = self.infer(job, "predict", {**common, "window": history[-30:]}, PredictionResponse, release)
            snapshot["anomaly"] = self.infer(job, "anomaly", {**common, "history": history, "return_history": False,
                "detector": "autoencoder", "threshold_policy": "healthy_q98"}, AnomalyResponse, release)
            snapshot["survival_status"] = "not_scheduled"
            if cycle > run["contract"]["survival_last_cycle"]:
                snapshot["survival_status"] = "unsupported_age"
            elif (cycle - 30) % run["config"]["survival_every_cycles"] == 0:
                snapshot["survival"] = self.infer(job, "survival", {**common, "initial_window": history[:30],
                    "current_cycle": cycle, "horizon_cycles": run["config"]["survival_horizon_cycles"]}, SurvivalResponse, release)
                snapshot["survival_status"] = "updated"
        snapshot["processing_duration_ms"] = (time.perf_counter() - start) * 1000
        self.store.complete(job, snapshot)

    def run_once(self):
        job = self.store.claim(self.run_id)
        if job is None:
            return False
        failure = None
        try:
            self.process(job)
            LOG.info(json.dumps({"event": "processed", "run_id": job["run_id"], "unit_id": job["unit_id"], "cycle": job["cycle"]}))
        except InferenceFailure as exc:
            failure = (exc.code, exc.retryable)
            LOG.warning(json.dumps({"event": "inference_failed", "run_id": job["run_id"], "cycle": job["cycle"], "code": exc.code}))
        except StreamError as exc:
            if exc.code != "LEASE_LOST":
                raise
            LOG.warning("Result discarded after worker lease changed")
        except Exception:
            # A failed write rolls back the entire result/alert transaction.
            LOG.exception("Stream job failed unexpectedly")
            failure = ("WORKER_INTERNAL_ERROR", True)
        if failure:
            try:
                self.store.fail(job, *failure, max_attempts=self.max_attempts)
            except StreamError as exc:
                if exc.code != "LEASE_LOST":
                    raise
                LOG.warning("Failure ignored after worker lease changed")
        return True


def client_for(api_url, settings):
    headers = {}
    if settings.api_token and settings.api_token.get_secret_value():
        headers["Authorization"] = "Bearer " + settings.api_token.get_secret_value()
    # Lease is 180 seconds and renewed before each request; API response timeout is <=120 by default.
    return httpx.Client(base_url=api_url.rstrip("/"), timeout=httpx.Timeout(130, connect=5, write=10, pool=5),
                        headers=headers, trust_env=False)


def main():
    parser = argparse.ArgumentParser(description="Process persisted readings through the running FD001 API")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--once", action="store_true", help="Attempt one available job, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    store = Store(settings.stream_db, settings.stream_max_pending, settings.max_history_rows)
    with client_for(args.api_url, settings) as client:
        worker = Worker(store, client, args.run_id)
        worker.verify_database()
        try:
            while True:
                worked = worker.run_once()
                if args.once:
                    break
                if not worked:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            LOG.info("Worker stopped; uncompleted jobs remain in SQLite for restart")


if __name__ == "__main__":
    main()
