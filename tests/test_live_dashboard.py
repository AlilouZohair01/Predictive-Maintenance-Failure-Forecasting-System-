import base64
import copy
from pathlib import Path
import tempfile
import unittest

import httpx

from api_helpers import ROOT
from test_dashboard import FakeModelClient
from predictive_maintenance.dashboard.app import create_app
from predictive_maintenance.dashboard.config import DashboardSettings
from predictive_maintenance.dashboard.data import Dataset
from predictive_maintenance.dashboard.service import DashboardService, DashboardError
from predictive_maintenance.dashboard.live_service import FeedClient, LiveService
from predictive_maintenance.api.schemas import PredictionResponse, AnomalyResponse, SurvivalResponse
from predictive_maintenance.streaming.store import Store
from predictive_maintenance.streaming.schemas import CreateRun


class StoreFeed(FeedClient):
    def __init__(self, store):
        self.store = store
        self.models = FakeModelClient()
        self.requests = []
        self.fail = False

    def get(self, path, params=None):
        self.requests.append(("GET", path))
        if self.fail:
            raise DashboardError("Stream API offline")
        p, parts = params or {}, path.split("/")
        if path == "runs":
            return {"items": self.store.runs()}
        run = parts[1]
        if len(parts) == 2:
            return {"run": self.store.run(run)}
        if parts[2] == "alerts":
            return {"items": self.store.alerts(run, p.get("state", "active"))}
        if len(parts) == 3:
            return {"items": self.store.units(run)}
        unit = int(parts[3])
        if parts[4] == "latest":
            return self.store.latest(run, unit)
        if parts[4] == "readings":
            return {"items": self.store.history(run, unit, p["through_cycle"], limit=p["limit"])}
        return {"items": self.store.predictions(run, unit, limit=p["limit"])}

    def call(self, endpoint, payload, schema):
        self.requests.append(("POST", endpoint))
        return self.models.call(endpoint, payload, schema)


class LiveDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(ROOT / "data/raw/cmapss/test_FD001.txt")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = [1000.]
        self.store = Store(Path(self.temp.name) / "stream.sqlite3", clock=lambda: self.clock[0])
        self.store.create_run(CreateRun(run_id="live-test").model_dump(mode="json"), "release-test",
            {"window_cycles": 30, "survival_last_cycle": 362, "max_history_cycles": 1000})
        self.feed = StoreFeed(self.store)
        self.service = LiveService(self.feed)
        self.app = create_app(DashboardSettings(_env_file=None), DashboardService(self.dataset, FakeModelClient()), self.service)
        self.client = self.app.server.test_client()
        self.client.get("/live/")

    def tearDown(self):
        self.temp.cleanup()

    def advance(self, received, processed=None, low_rul=False):
        processed = received if processed is None else processed
        for row in self.dataset.readings(55, received):
            self.store.ingest("live-test", {"unit_id": 55, "reading": row})
        while self.store.latest("live-test", 55)["processed_cycle"] < processed:
            job = self.store.claim("live-test")
            cycle = job["cycle"]
            history = self.dataset.readings(55, cycle)
            snapshot = {"run_id": "live-test", "unit_id": 55, "cycle": cycle, "dataset": "FD001", "release_version": "release-test",
                        "status": "ready" if cycle >= 30 else "warmup", "prediction": None, "anomaly": None, "survival": None,
                        "survival_status": "not_scheduled" if cycle >= 30 else "warmup"}
            if cycle >= 30:
                snapshot["prediction"] = self.feed.models.call("predict", {"unit_id": 55, "window": history[-30:]}, PredictionResponse)
                if low_rul:
                    snapshot["prediction"]["rul_cycles"] = 10
                snapshot["anomaly"] = self.feed.models.call("anomaly", {"unit_id": 55, "history": history}, AnomalyResponse)
                snapshot["anomaly"]["history"] = None
                if (cycle - 30) % 10 == 0:
                    snapshot["survival"] = self.feed.models.call("survival", {"unit_id": 55, "current_cycle": cycle}, SurvivalResponse)
            self.store.complete(job, snapshot)

    def callback(self, marker, inputs, states=(), changed=None, client=None):
        live = self.app.live_app
        output = next(k for k in live.callback_map if marker in k)
        out = live.callback_map[output]["output"]
        encode = lambda o: {"id": o.component_id, "property": o.component_property}
        outputs = [encode(o) for o in out] if isinstance(out, list) else encode(out)
        values = lambda xs: [{"id": i, "property": p, "value": v} for i, p, v in xs]
        response = (client or self.client).post("/live/_dash-update-component", json={"output": output, "outputs": outputs,
            "inputs": values(inputs), "state": values(states), "changedPropIds": [changed or f"{inputs[0][0]}.{inputs[0][1]}"]})
        self.assertEqual(response.status_code, 200, response.text[:600])
        return response

    def refresh(self):
        return self.callback("live-ref.data", [("live-run", "value", "live-test"), ("live-unit", "value", 55),
            ("live-tick", "n_intervals", 1), ("live-sensors", "value", ["s_11"])])

    def test_live_layout_pause_and_auth(self):
        layout = self.client.get("/live/_dash-layout")
        self.assertEqual(layout.status_code, 200)
        self.assertIn('"interval":2000', layout.text)
        paused = self.callback("live-tick.disabled", [("live-pause", "value", ["pause"])])
        self.assertTrue(paused.json["response"]["live-tick"]["disabled"])
        self.assertFalse(self.callback("live-tick.disabled", [("live-pause", "value", [])]).json["response"]["live-tick"]["disabled"])
        guarded = create_app(DashboardSettings(auth_mode="basic", username="test", password="long-test-password", _env_file=None),
                            DashboardService(self.dataset, FakeModelClient()), self.service).server.test_client()
        for route in ("/live/", "/live/_dash-layout", "/live/_dash-dependencies", "/live/assets/style.css"):
            with guarded.get(route) as r:
                self.assertEqual(r.status_code, 401)
            with guarded.get(route, auth=("test", "long-test-password")) as r:
                self.assertEqual(r.status_code, 200)
        self.assertEqual(guarded.post("/live/_dash-update-component", json={}).status_code, 401)

    def test_warmup_to_ready_auto_refresh_without_inference(self):
        self.advance(29)
        self.assertIn("Warming up", self.refresh().text)
        self.advance(32)
        response = self.refresh()
        self.assertIn("Cycle 32", response.text)
        self.assertIn("No sustained alert", response.text)
        self.assertIn("Estimate from cycle 30", response.text)
        self.assertEqual({method for method, _ in self.feed.requests}, {"GET"})

    def test_freezes_processed_boundary_and_survival_age(self):
        self.advance(35, processed=32)
        ref, view = self.service.poll("owner", "live-test", 55)
        self.assertEqual(view["cycle"], 32)
        self.assertEqual(view["feed"]["processing_lag_cycles"], 3)
        self.assertEqual(len(view["readings"]), 32)
        self.assertEqual(view["snapshot"]["survival"]["cycle"], 30)
        self.assertEqual([p["cycle"] for p in view["risk_history"]], [30])
        self.assertIsNone(self.service.resolve("other", ref, "live-test", 55))
        self.assertIsNone(self.service.resolve("owner", ref, "another-run", 55))

    def test_stale_closed_blocked_and_api_failure_are_explicit(self):
        self.advance(32, processed=30)
        self.clock[0] += 40
        self.assertIn("Source stale", self.refresh().text)
        job = self.store.claim("live-test")
        self.store.fail(job, "INFERENCE_HTTP_401", False)
        response = self.refresh()
        self.assertIn("blocked", response.text)
        self.assertIn("INFERENCE_HTTP_401", response.text)
        self.store.close_run("live-test")
        self.assertIn("Replay closed", self.refresh().text)
        self.feed.fail = True
        failed = self.refresh()
        self.assertIsNone(failed.json["response"]["live-ref"]["data"])
        self.assertIn("Stream API offline", failed.text)
        self.assertNotIn("Saved prediction", failed.text)

    def test_alerts_discovery_and_selection(self):
        self.advance(32, low_rul=True)
        runs = self.callback("live-run.options", [("live-tick", "n_intervals", 0)], [("live-run", "value", None)])
        self.assertEqual(runs.json["response"]["live-run"]["value"], "live-test")
        units = self.callback("live-unit.options", [("live-run", "value", "live-test"), ("live-tick", "n_intervals", 0)], [("live-unit", "value", None)])
        self.assertEqual(units.json["response"]["live-unit"]["value"], 55)
        response = self.refresh()
        self.assertIn("low rul", response.text)
        self.assertIn("active", response.text)

    def test_shap_is_explicit_pinned_and_export_uses_stream(self):
        self.advance(32)
        ref = self.refresh().json["response"]["live-ref"]["data"]
        response = self.callback("live-explanation.data", [("live-explain", "n_clicks", 1), ("live-run", "value", "live-test"),
            ("live-unit", "value", 55)], [("live-ref", "data", ref)])
        explained = response.json["response"]["live-explanation"]["data"]
        self.assertIn("cycle 32", response.text)
        self.assertEqual([p for m, p in self.feed.requests if m == "POST"], ["explain"])
        for kind, signature in (("png", b"\x89PNG"), ("pdf", b"%PDF")):
            result = self.callback("live-download.data", [("live-png", "n_clicks", 1), ("live-pdf", "n_clicks", 1)],
                [("live-ref", "data", ref), ("live-explanation", "data", explained), ("live-run", "value", "live-test"),
                 ("live-unit", "value", 55), ("live-sensors", "value", ["s_11"])], changed=f"live-{kind}.n_clicks")
            download = result.json["response"]["live-download"]["data"]
            self.assertTrue(base64.b64decode(download["content"]).startswith(signature))
            self.assertIn("live-test_engine055_cycle32", download["filename"])
        cleared = self.callback("live-explanation.data", [("live-explain", "n_clicks", 1), ("live-run", "value", "another-run"),
            ("live-unit", "value", 55)], [("live-ref", "data", ref)], changed="live-run.value")
        self.assertIsNone(cleared.json["response"]["live-explanation"]["data"])

    def test_old_release_rejected_for_shap(self):
        self.advance(30)
        ref, _ = self.service.poll("owner", "live-test", 55)
        self.feed.models.release = "new-release"
        with self.assertRaisesRegex(DashboardError, "older model release"):
            self.service.explain("owner", ref, "live-test", 55)

    def test_feed_pagination_auth_and_invalid_response(self):
        calls = []
        def response(request):
            calls.append(request)
            if "after" not in request.url.params:
                return httpx.Response(200, json={"items": [1], "next_after": "one"})
            return httpx.Response(200, json={"items": [2], "next_after": None})
        client = FeedClient(DashboardSettings(api_token="test-secret", _env_file=None), httpx.MockTransport(response))
        self.assertEqual(client.collect("runs", "next_after", "after"), [1, 2])
        self.assertEqual(calls[0].headers["Authorization"], "Bearer test-secret")
        for status, body in ((401, {}), (503, {}), (200, [])):
            client.transport = httpx.MockTransport(lambda r: httpx.Response(status, json=body))
            with self.assertRaises(DashboardError):
                client.get("runs")


if __name__ == "__main__":
    unittest.main()
