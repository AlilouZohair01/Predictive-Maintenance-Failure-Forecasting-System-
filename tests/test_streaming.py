import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api_helpers import fixture, FakeRegistry
from predictive_maintenance.api.config import Settings
from predictive_maintenance.api.main import create_app
from predictive_maintenance.streaming.alerts import transition
from predictive_maintenance.streaming.schemas import CreateRun, IngestReading, AlertPolicy
from predictive_maintenance.streaming.store import Store, StreamError
from predictive_maintenance.streaming.worker import Worker


class StreamCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.now = 1000.0
        self.path = Path(self.directory.name) / "stream.sqlite3"
        self.store = Store(self.path, clock=lambda: self.now)
        self.config = CreateRun(run_id="test-run").model_dump(mode="json")
        self.contract = {"window_cycles": 30, "survival_last_cycle": 362, "max_history_cycles": 1000}
        self.store.create_run(self.config, "release-test", self.contract)

    def tearDown(self):
        self.directory.cleanup()

    def message(self, cycle, unit=55):
        row = {**fixture("anomaly")["history"][0], "cycle": cycle}
        return IngestReading(unit_id=unit, reading=row).model_dump(mode="json")

    def enqueue(self, count, unit=55):
        for cycle in range(1, count + 1):
            self.store.ingest("test-run", self.message(cycle, unit))

    def snapshot(self, job, rul=None, flagged=None):
        return {"unit_id": job["unit_id"], "cycle": job["cycle"], "release_version": "release-test",
                "prediction": {"rul_cycles": rul} if rul is not None else None,
                "anomaly": {"latest": {"sustained_alert": flagged, "is_anomaly": flagged}} if flagged is not None else None,
                "survival": None}


class StoreTests(StreamCase):
    def test_duplicate_conflict_gap_and_closed_run(self):
        first = self.message(1)
        self.assertEqual(self.store.ingest("test-run", first)["disposition"], "accepted")
        self.assertEqual(self.store.ingest("test-run", first)["disposition"], "duplicate")
        changed = copy.deepcopy(first)
        changed["reading"]["s_2"] += 1
        for value, code in ((changed, "READING_CONFLICT"), (self.message(3), "CYCLE_OUT_OF_ORDER")):
            with self.assertRaises(StreamError) as exc:
                self.store.ingest("test-run", value)
            self.assertEqual(exc.exception.code, code)
        self.store.close_run("test-run")
        self.assertEqual(self.store.ingest("test-run", first)["disposition"], "duplicate")
        with self.assertRaises(StreamError):
            self.store.ingest("test-run", self.message(2))
        self.assertEqual(self.store.run("test-run")["jobs"]["pending"], 1)

    def test_concurrent_duplicate_has_one_job(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.store.ingest("test-run", self.message(1)), range(8)))
        self.assertEqual(sum(r["disposition"] == "accepted" for r in results), 1)
        self.assertEqual(self.store.run("test-run")["jobs"]["pending"], 1)

    def test_restart_run_identity_and_isolation(self):
        self.enqueue(1)
        resumed = Store(self.path)
        self.assertEqual(resumed.store_id, self.store.store_id)
        self.assertEqual(resumed.latest("test-run", 55)["received_cycle"], 1)
        resumed.create_run(self.config, "release-test", self.contract)
        for config, release in (({**self.config, "source": "changed"}, "release-test"), (self.config, "new-release")):
            with self.assertRaises(StreamError):
                resumed.create_run(config, release, self.contract)
        resumed.create_run({**self.config, "run_id": "second"}, "release-test", self.contract)
        resumed.ingest("second", self.message(1))
        self.assertEqual(len(resumed.units("second")), 1)

    def test_queue_and_history_limits(self):
        limited = Store(self.path, max_pending=1, max_history=30)
        limited.ingest("test-run", self.message(1))
        with self.assertRaises(StreamError) as exc:
            limited.ingest("test-run", self.message(2))
        self.assertEqual(exc.exception.status, 429)
        self.assertEqual(limited.ingest("test-run", self.message(1))["disposition"], "duplicate")
        with self.assertRaises(StreamError) as exc:
            limited.ingest("test-run", self.message(31))
        self.assertEqual(exc.exception.code, "HISTORY_LIMIT")

    def test_leases_preserve_unit_order_and_fence_old_worker(self):
        self.enqueue(2, 1)
        self.enqueue(2, 2)
        first, other = self.store.claim(), self.store.claim()
        self.assertEqual((first["unit_id"], first["cycle"]), (1, 1))
        self.assertEqual((other["unit_id"], other["cycle"]), (2, 1))
        self.assertIsNone(self.store.claim())
        self.now += 181
        reclaimed = self.store.claim()
        self.assertEqual(reclaimed["id"], first["id"])
        with self.assertRaises(StreamError) as exc:
            self.store.complete(first, self.snapshot(first))
        self.assertEqual(exc.exception.code, "LEASE_LOST")
        self.store.complete(reclaimed, self.snapshot(reclaimed))
        self.assertEqual(self.store.claim()["cycle"], 2)

    def test_failed_job_blocks_only_its_engine_and_manual_retry(self):
        self.enqueue(2, 1)
        self.enqueue(1, 2)
        job = self.store.claim()
        self.store.fail(job, "PERMANENT", False)
        self.assertEqual(self.store.claim()["unit_id"], 2)
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.store.retry_failed("test-run", 1)["retried_jobs"], 1)
        self.assertEqual(self.store.claim()["id"], job["id"])

    def test_transaction_rolls_back_result_and_alerts(self):
        self.enqueue(1)
        job = self.store.claim()
        with patch("predictive_maintenance.streaming.store.transition", side_effect=RuntimeError("injected")):
            with self.assertRaises(RuntimeError):
                self.store.complete(job, self.snapshot(job, 10, True))
        self.assertEqual(self.store.predictions("test-run", 55), [])
        self.assertEqual(self.store.alerts("test-run"), [])
        self.assertEqual(self.store.latest("test-run", 55)["processed_cycle"], 0)
        self.store.complete(job, self.snapshot(job, 10, True))
        self.assertEqual(len(self.store.alerts("test-run")), 2)
        with self.assertRaises(StreamError):
            self.store.complete(job, self.snapshot(job, 10, True))
        self.assertEqual(len(self.store.events("test-run")), 2)

    def test_alert_lifecycle_no_duplicate_notifications(self):
        self.enqueue(10)
        for index in range(1, 11):
            job = self.store.claim()
            self.store.complete(job, self.snapshot(job, 15 if index < 5 else 30, index < 5))
        alerts = self.store.alerts("test-run", "all")
        self.assertEqual(len(alerts), 2)
        self.assertTrue(all(r["state"] == "resolved" for r in alerts))
        self.assertEqual(len(self.store.events("test-run")), 4)
        self.assertEqual(next(a for a in alerts if a["rule"] == "low_rul")["closed_cycle"], 5)
        self.assertEqual(next(a for a in alerts if a["rule"] == "sustained_anomaly")["closed_cycle"], 9)

    def test_staleness_lag_and_pagination(self):
        self.enqueue(3)
        job = self.store.claim()
        self.store.complete(job, self.snapshot(job))
        self.now += 31
        latest = self.store.latest("test-run", 55)
        self.assertTrue(latest["is_stale"])
        self.assertEqual(latest["processing_lag_cycles"], 2)
        self.assertEqual([x["reading"]["cycle"] for x in self.store.history("test-run", 55, after=1, limit=1)], [2])
        self.store.close_run("test-run")
        self.assertFalse(self.store.latest("test-run", 55)["is_stale"])


class PolicyTests(unittest.TestCase):
    def test_policy_is_explicit_and_validated(self):
        with self.assertRaises(ValidationError):
            AlertPolicy(rul_trigger_below=25, rul_clear_at=20)
        with self.assertRaises(ValidationError):
            CreateRun(run_id="../escape")
        self.assertEqual(AlertPolicy().validation_status, "illustrative_unvalidated")

    def test_hysteresis_cooldown_and_unknown_state(self):
        policy = AlertPolicy().model_dump()
        state, event = transition("low_rul", None, 30, 20, None, policy)
        self.assertIsNone(event)  # strict boundary
        state, event = transition("low_rul", state, 31, 19, None, policy)
        self.assertEqual(event, "raised")
        for cycle, rul in ((32, 22), (33, None)):
            state, event = transition("low_rul", state, cycle, rul, None, policy)
            self.assertTrue(state["active"])
            self.assertIsNone(event)
        state, event = transition("low_rul", state, 34, 25, None, policy)
        self.assertEqual(event, "resolved")
        state, event = transition("low_rul", state, 35, 19, None, policy)
        self.assertIsNone(event)
        state, event = transition("low_rul", state, 41, 19, None, policy)
        self.assertEqual(event, "raised")


class WorkerTests(StreamCase):
    def setUp(self):
        super().setUp()
        self.calls, self.failures = [], []
        self.response_release = "release-test"
        self.client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(self.handler))
        self.worker = Worker(self.store, self.client, "test-run")

    def tearDown(self):
        self.client.close()
        super().tearDown()

    def handler(self, request):
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"store_id": self.store.store_id})
        payload = json.loads(request.content)
        kind = request.url.path.rsplit("/", 1)[-1]
        self.calls.append((kind, payload))
        if self.failures:
            code = self.failures.pop(0)
            if code:
                return httpx.Response(code, json={"message": "simulated failure"})
        cycle = payload.get("current_cycle") or payload.get("window", payload.get("history"))[-1]["cycle"]
        common = {"unit_id": payload["unit_id"], "cycle": cycle, "dataset": "FD001", "model_version": "model-test",
                  "release_version": self.response_release, "request_id": "test-request", "inference_duration_ms": 1.0}
        if kind == "predict":
            result = {"rul_cycles": 19.0, "rul_cap_cycles": 125.0, "warnings": []}
        elif kind == "anomaly":
            result = {"detector": "autoencoder", "threshold_policy": "healthy_q98", "threshold": 0.5,
                      "latest": {"cycle": cycle, "raw_score": 0.8, "anomaly_score": 0.8, "is_anomaly": True,
                                 "sustained_alert": True if cycle >= 32 else None, "status": "ready",
                                 "alert_history_ready": cycle >= 32, "smoothing_samples": min(5, cycle - 29)}, "history": None}
        else:
            result = {"requested_horizon_cycles": 100, "evaluated_horizon_cycles": 100, "horizon_truncated": False,
                      "failure_probability_at_requested_horizon": 0.1, "median_remaining_cycles": None,
                      "median_unavailable_reason": "Not reached", "curve": [{"future_cycles": 0, "survival_probability": 1}], "limitations": []}
        return httpx.Response(200, json={**common, **result})

    def drain(self):
        while self.worker.run_once():
            pass

    def test_full_flow_warmup_no_future_leak_and_restart(self):
        self.enqueue(41)
        self.worker.verify_database()
        self.drain()
        predictions = self.store.predictions("test-run", 55)
        self.assertEqual(len(predictions), 41)
        self.assertIsNone(predictions[28]["prediction"])
        self.assertEqual(predictions[29]["prediction"]["rul_cycles"], 19)
        self.assertIsNone(predictions[29]["anomaly"]["latest"]["sustained_alert"])
        windows = [body["window"] for kind, body in self.calls if kind == "predict"]
        self.assertTrue(all(len(rows) == 30 for rows in windows))
        self.assertEqual([w[-1]["cycle"] for w in windows], list(range(30, 42)))
        histories = [body["history"] for kind, body in self.calls if kind == "anomaly"]
        self.assertEqual([len(h) for h in histories], list(range(30, 42)))
        survival = [body for kind, body in self.calls if kind == "survival"]
        self.assertEqual([r["current_cycle"] for r in survival], [30, 40])
        self.assertTrue(all([r["cycle"] for r in x["initial_window"]] == list(range(1, 31)) for x in survival))
        self.assertNotIn("explain", [k for k, _ in self.calls])
        self.assertEqual(len(self.store.alerts("test-run")), 2)
        latest = Store(self.path).latest("test-run", 55)
        self.assertEqual(latest["latest_survival"]["cycle"], 40)
        self.assertEqual(latest["processed_cycle"], 41)
        before = len(self.calls)
        self.assertFalse(Worker(Store(self.path), self.client, "test-run").run_once())
        self.assertEqual(before, len(self.calls))

    def test_transient_failure_retry_does_not_advance_or_partially_commit(self):
        self.enqueue(31)
        for _ in range(29):
            self.worker.run_once()
        self.failures = [None, 503]  # prediction succeeds; anomaly fails
        self.worker.run_once()
        self.assertEqual(self.store.latest("test-run", 55)["processed_cycle"], 29)
        self.assertEqual(self.store.alerts("test-run"), [])
        self.assertFalse(self.worker.run_once())
        self.now += 3
        self.drain()
        self.assertEqual(self.store.latest("test-run", 55)["processed_cycle"], 31)
        self.assertEqual(len(self.store.events("test-run")), 1)

    def test_release_mismatch_blocks_without_mixing_versions(self):
        self.enqueue(31)
        self.response_release = "new-release"
        self.drain()
        latest = self.store.latest("test-run", 55)
        self.assertEqual(latest["processed_cycle"], 29)
        self.assertEqual(latest["next_job"]["error_code"], "MODEL_RELEASE_CHANGED")
        self.assertEqual(latest["next_job"]["status"], "failed")

    def test_permanent_auth_failure_and_database_mismatch(self):
        self.enqueue(30)
        self.failures = [401]
        self.drain()
        self.assertEqual(self.store.run("test-run")["jobs"]["failed"], 1)
        other = Store(Path(self.directory.name) / "other.sqlite3")
        with self.assertRaises(RuntimeError):
            Worker(other, self.client).verify_database()

    def test_bounded_retries(self):
        self.enqueue(30)
        self.failures = [503] * 5
        for _ in range(5):
            self.drain()
            self.now += 40
        self.assertEqual(self.store.run("test-run")["jobs"]["failed"], 1)
        self.assertEqual(self.store.latest("test-run", 55)["next_job"]["attempts"], 5)


class FeedRegistry(FakeRegistry):
    def describe(self):
        return {"rul": {"window_cycles": 30}, "survival": {"support_since_landmark": 332}}


class StreamEndpointTests(unittest.TestCase):
    def test_ingestion_feeds_auth_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(stream_db=Path(directory) / "stream.sqlite3", auth_mode="bearer", api_token="x" * 32, _env_file=None)
            app = create_app(settings, FeedRegistry)
            with TestClient(app) as client:
                self.assertEqual(client.get("/v1/stream/status").status_code, 401)
                client.headers["Authorization"] = "Bearer " + "x" * 32
                self.assertEqual(client.post("/v1/stream/runs", json={"run_id": "api-demo"}).status_code, 200)
                payload = {"unit_id": 55, "reading": fixture("anomaly")["history"][0]}
                self.assertEqual(client.post("/v1/stream/runs/api-demo/readings", json=payload).status_code, 202)
                for suffix in ("", "/units", "/units/55/latest", "/units/55/readings", "/units/55/predictions", "/alerts", "/alert-events"):
                    response = client.get("/v1/stream/runs/api-demo" + suffix)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertIn("request_id", response.json())
                self.assertEqual(client.get("/v1/stream/runs").status_code, 200)
                self.assertEqual(client.get("/v1/stream/runs/api-demo/units?limit=1001").status_code, 422)
                bad = copy.deepcopy(payload)
                bad["reading"]["s_2"] = "123"
                self.assertEqual(client.post("/v1/stream/runs/api-demo/readings", json=bad).status_code, 422)
                bad["reading"]["s_2"] = 123
                response = client.post("/v1/stream/runs/api-demo/readings", json=bad)
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()["error_code"], "READING_CONFLICT")
                self.assertEqual(client.get("/v1/stream/runs/missing").status_code, 404)
                self.assertEqual(client.post("/v1/stream/runs/api-demo/units/55/retry").status_code, 200)
                self.assertEqual(client.post("/v1/stream/runs/api-demo/close").status_code, 200)
                # Feed reads cannot invoke the models.
                app.state.registry.predict = lambda _: self.fail("Feed should never run inference")
                self.assertEqual(client.get("/v1/stream/runs/api-demo/units/55/latest").status_code, 200)


if __name__ == "__main__":
    unittest.main()
