import asyncio
import json
import threading
import time
import unittest
from fastapi.testclient import TestClient

from api_helpers import fixture, FakeRegistry
from predictive_maintenance.api.config import Settings
from predictive_maintenance.api.main import create_app, WorkPool
from predictive_maintenance.api.middleware import Metrics
from predictive_maintenance.api.errors import ServiceError


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(_env_file=None), FakeRegistry)
        self.context = TestClient(self.app)
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_health_models_docs_metrics_and_ids(self):
        for path in ("/health/live", "/health/ready", "/v1/models", "/docs", "/redoc", "/openapi.json", "/metrics"):
            response = self.client.get(path, headers={"X-Request-ID": "example-123"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["X-Request-ID"], "example-123")
        response = self.client.post("/v1/predict", json=fixture("predict"))
        self.assertEqual(response.json()["rul_cycles"], 62.5)
        self.assertGreaterEqual(response.json()["inference_duration_ms"], 0)
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])
        response = self.client.get("/health/live", headers={"X-Request-ID": "a" * 100})
        self.assertEqual(len(response.headers["X-Request-ID"]), 36)
        self.assertIn("pm_http_duration_seconds_bucket", self.client.get("/metrics").text)

    def test_invalid_requests_use_scrubbed_envelope(self):
        requests = [("post", "/v1/predict", {"json": {"secret": "never-echo-this"}}, 422),
                    ("post", "/v1/predict", {"content": "{bad", "headers": {"Content-Type": "application/json"}}, 422),
                    ("get", "/not-a-route", {}, 404), ("get", "/v1/predict", {}, 405)]
        for method, path, options, status in requests:
            response = getattr(self.client, method)(path, **options)
            self.assertEqual(response.status_code, status, response.text)
            self.assertEqual(set(response.json()), {"error_code", "message", "details", "request_id"})
            self.assertNotIn("never-echo-this", response.text)

    def test_host_size_and_readiness_failures(self):
        self.assertEqual(self.client.get("/health/live", headers={"Host": "untrusted.example"}).status_code, 400)
        self.assertEqual(self.client.post("/v1/predict", content=b"x" * 1_048_577).status_code, 413)
        self.assertEqual(self.client.post("/v1/predict", content=b"x", headers={"Content-Length": "bad"}).status_code, 400)
        # Actual bytes are checked even when the declared length is smaller.
        self.assertEqual(self.client.post("/v1/predict", content=b"x" * 1_048_577, headers={"Content-Length": "1"}).status_code, 413)
        self.app.state.ready = False
        self.assertEqual(self.client.get("/health/ready").status_code, 503)
        self.assertEqual(self.client.get("/health/live").status_code, 200)

    def test_unexpected_error_and_invalid_output_are_generic(self):
        def broken(raw):
            raise RuntimeError("private-model-path")
        self.app.state.registry.predict = broken
        response = self.client.post("/v1/predict", json=fixture("predict"))
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private-model-path", response.text)
        self.app.state.registry.predict = lambda raw: {"rul_cycles": float("nan"), "warnings": []}
        self.assertEqual(self.client.post("/v1/predict", json=fixture("predict")).status_code, 500)

    def test_history_limit(self):
        app = create_app(Settings(max_history_rows=30, _env_file=None), FakeRegistry)
        with TestClient(app) as client:
            response = client.post("/v1/anomaly", json=fixture("anomaly"))
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error_code"], "HISTORY_TOO_LONG")

    def test_auth_and_cors(self):
        app = create_app(Settings(auth_mode="bearer", api_token="x" * 32, cors_origins=["http://localhost:3000"], _env_file=None), FakeRegistry)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health/live").status_code, 200)
            for path in ("/v1/models", "/metrics"):
                self.assertEqual(client.get(path).status_code, 401)
                self.assertEqual(client.get(path, headers={"Authorization": "Bearer " + "x" * 32}).status_code, 200)
            response = client.options("/v1/predict", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,x-request-id"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:3000")


class WorkPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_body_is_bounded(self):
        from predictive_maintenance.api.middleware import BoundaryMiddleware
        sent = []
        async def unused(scope, receive, send):
            self.fail("Slow body must be rejected before the application")
        async def receive():
            await asyncio.sleep(1)
        async def send(message):
            sent.append(message)
        boundary = BoundaryMiddleware(unused, Settings(body_timeout_seconds=0.01, _env_file=None), Metrics())
        await boundary({"type": "http", "method": "POST", "headers": [(b"host", b"localhost")]}, receive, send)
        self.assertEqual(sent[0]["status"], 408)

    async def test_timeout_keeps_capacity_until_native_work_finishes(self):
        metrics = Metrics()
        pool = WorkPool(Settings(inference_capacity=1, request_timeout_seconds=0.05, _env_file=None), metrics)
        finish = threading.Event()
        try:
            with self.assertRaises(ServiceError) as error:
                await pool.run("predict", lambda: finish.wait(2))
            self.assertEqual(error.exception.code, "INFERENCE_TIMEOUT")
            self.assertEqual(pool.active["inference"], 1)
            with self.assertRaises(ServiceError) as error:
                await pool.run("predict", lambda: 1)
            self.assertEqual(error.exception.code, "CAPACITY_EXHAUSTED")
            # Explanation has an independent executor and admission limit.
            self.assertEqual((await pool.run("explain", lambda: 3))[0], 3)
        finally:
            finish.set()
            await pool.close()
        self.assertEqual(pool.active["inference"], 0)
        with self.assertRaises(ServiceError):
            await pool.run("predict", lambda: 1)

    async def test_cancellation_does_not_release_capacity(self):
        pool = WorkPool(Settings(inference_capacity=1, _env_file=None), Metrics())
        finish, started = threading.Event(), threading.Event()
        def work():
            started.set()
            finish.wait(2)
        task = asyncio.create_task(pool.run("predict", work))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(pool.active["inference"], 1)
        finish.set()
        await pool.close()

    async def test_unlabelled_stream_size_bound(self):
        from predictive_maintenance.api.middleware import BoundaryMiddleware
        called, sent = [], []
        async def app(scope, receive, send):
            called.append(True)
        chunks = iter([{"type": "http.request", "body": b"x" * 800, "more_body": True},
                       {"type": "http.request", "body": b"x" * 800, "more_body": False}])
        async def receive():
            return next(chunks)
        async def send(message):
            sent.append(message)
        boundary = BoundaryMiddleware(app, Settings(max_body_bytes=1024, _env_file=None), Metrics())
        await boundary({"type": "http", "method": "POST", "headers": [(b"host", b"localhost")]}, receive, send)
        self.assertFalse(called)
        self.assertEqual(sent[0]["status"], 413)
