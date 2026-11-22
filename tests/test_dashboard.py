import base64
import copy
import json
from pathlib import Path
import unittest

import httpx
from pydantic import ValidationError

from api_helpers import ROOT
from predictive_maintenance.dashboard.config import DashboardSettings
from predictive_maintenance.dashboard.data import Dataset
from predictive_maintenance.dashboard.service import DashboardService, ModelClient, SnapshotStore, DashboardError


class FakeModelClient:
    def __init__(self):
        self.calls = []
        self.release = "release-test"

    def call(self, endpoint, payload, schema):
        self.calls.append((endpoint, copy.deepcopy(payload)))
        cycle = payload.get("current_cycle") or payload.get("window", payload.get("history"))[-1]["cycle"]
        common = {"unit_id": payload["unit_id"], "dataset": "FD001", "cycle": cycle,
                  "model_version": "model-test", "release_version": self.release, "request_id": "test", "inference_duration_ms": 1.0}
        if endpoint == "predict":
            result = {"rul_cycles": 70.0, "rul_cap_cycles": 125, "warnings": []}
        elif endpoint == "anomaly":
            points = [{"cycle": i, "raw_score": .3 if i >= 30 else None, "anomaly_score": .3 if i >= 30 else None,
                       "is_anomaly": False if i >= 30 else None, "sustained_alert": False if i >= 32 else None,
                       "status": "ready" if i >= 30 else "warmup", "alert_history_ready": i >= 32,
                       "smoothing_samples": min(5, max(0, i - 29))} for i in range(1, cycle + 1)]
            result = {"detector": "autoencoder", "threshold_policy": "healthy_q98", "threshold": .5,
                      "latest": points[-1], "history": points}
        elif endpoint == "survival":
            result = {"requested_horizon_cycles": 100, "evaluated_horizon_cycles": 100, "horizon_truncated": False,
                      "failure_probability_at_requested_horizon": .2, "median_remaining_cycles": None,
                      "median_unavailable_reason": "Not reached", "curve": [{"future_cycles": 0, "survival_probability": 1},
                          {"future_cycles": 100, "survival_probability": .8}], "limitations": []}
        else:
            from predictive_maintenance.api.schemas import SENSORS
            rows = [{"sensor": s, "contribution_cycles": (i - 6) * .2, "raw_window_mean": 1.0} for i, s in enumerate(SENSORS)]
            result = {"predicted_rul_cycles": 70., "base_value_cycles": 70., "reconstructed_rul_cycles": 70., "explanation_gap_cycles": 0.,
                      "sensor_contributions": rows, "top_positive": rows[-3:], "top_negative": rows[:3],
                      "method": "shap.GradientExplainer", "sample_budget": 500, "seed": 42, "background_version": "background-test", "warnings": []}
        return schema.model_validate({**common, **result}).model_dump(mode="json")


class DashboardServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Dataset(ROOT / "data/raw/cmapss/test_FD001.txt")

    def setUp(self):
        self.client = FakeModelClient()
        self.service = DashboardService(self.data, self.client)

    def test_dataset_prefix_and_invalid_selection(self):
        self.assertEqual(len(self.data.engines), 100)
        self.assertEqual(self.data.length(55), 113)
        self.assertEqual([r["cycle"] for r in self.data.readings(55, 30)], list(range(1, 31)))
        for unit, cycle in ((999, 30), (True, 30), ([], 30), (55, 0), (55, 114), (55, 30.5)):
            with self.assertRaises(ValueError):
                self.data.prefix(unit, cycle)

    def test_analysis_inputs_never_include_future_observations(self):
        token = self.service.analyze("viewer", 55, 40)
        self.assertEqual([k for k, _ in self.client.calls], ["predict", "anomaly", "survival"])
        prediction, anomaly, survival = [p for _, p in self.client.calls]
        self.assertEqual([r["cycle"] for r in prediction["window"]], list(range(11, 41)))
        self.assertEqual(len(anomaly["history"]), 40)
        self.assertEqual([r["cycle"] for r in survival["initial_window"]], list(range(1, 31)))
        self.assertEqual(survival["current_cycle"], 40)
        self.assertIsNotNone(self.service.snapshots.get(token, "viewer", 55, 40))
        self.assertIsNone(self.service.snapshots.get(token, "viewer", 55, 41))
        self.assertIsNone(self.service.snapshots.get(token, "other-viewer", 55, 40))

    def test_warmup_does_not_request_inference(self):
        with self.assertRaises(DashboardError):
            self.service.analyze("viewer", 55, 29)
        self.assertEqual(self.client.calls, [])

    def test_survival_failure_preserves_other_models(self):
        original = self.client.call
        def failing_survival(endpoint, payload, schema):
            if endpoint == "survival":
                raise DashboardError("Unsupported follow-up")
            return original(endpoint, payload, schema)
        self.client.call = failing_survival
        ref = {"token": self.service.analyze("viewer", 55, 113)}
        snapshot = self.service.resolve("viewer", 55, 113, ref)
        self.assertIsNone(snapshot["survival"])
        self.assertEqual(snapshot["prediction"]["rul_cycles"], 70)
        self.assertIn("Survival unavailable", snapshot["warnings"][0])
        detail = {"token": self.service.enrich("viewer", 55, 113, ref, None, "trend")}
        self.assertTrue(all(p["failure_probability_100"] is None for p in self.service.resolve("viewer", 55, 113, ref, detail)["rul_history"]))

    def test_trend_is_bounded_and_explanation_preserves_context(self):
        analysis = {"token": self.service.analyze("viewer", 55, 113)}
        trend = {"token": self.service.enrich("viewer", 55, 113, analysis, None, "trend")}
        snapshot = self.service.resolve("viewer", 55, 113, analysis, trend)
        self.assertEqual(len(snapshot["rul_history"]), 12)
        self.assertEqual(snapshot["rul_history"][-1]["cycle"], 113)
        self.assertTrue(all(p["failure_probability_100"] == .2 for p in snapshot["rul_history"]))
        explained = {"token": self.service.enrich("viewer", 55, 113, analysis, trend, "explain")}
        enriched = self.service.resolve("viewer", 55, 113, analysis, explained)
        self.assertIsNotNone(enriched["explanation"])
        self.assertIsNotNone(enriched["rul_history"])
        newer = {"token": self.service.analyze("viewer", 55, 113)}
        self.assertIsNone(self.service.resolve("viewer", 55, 113, newer, explained)["explanation"])
        self.assertIsNone(self.service.resolve("viewer", 55, 112, analysis, explained))

    def test_release_changes_do_not_mix_explanations(self):
        analysis = {"token": self.service.analyze("viewer", 55, 40)}
        self.client.release = "new-release"
        with self.assertRaises(DashboardError):
            self.service.enrich("viewer", 55, 40, analysis, None, "explain")

    def test_cache_expiry_and_bound(self):
        now = [0]
        cache = SnapshotStore(capacity=2, ttl=10, clock=lambda: now[0])
        snapshot = {"unit": 55, "cycle": 40}
        first = cache.put("a", snapshot)
        second = cache.put("a", snapshot)
        cache.put("a", snapshot)
        self.assertIsNone(cache.get(first, "a", 55, 40))
        now[0] = 11
        self.assertIsNone(cache.get(second, "a", 55, 40))

    def test_api_errors_and_invalid_output_are_safe(self):
        from predictive_maintenance.api.schemas import PredictionResponse
        payload = {"unit_id": 55, "window": self.data.readings(55, 30)}
        for status, body in ((401, {}), (503, {}), (500, {"private": "secret"}), (200, {"bad": True})):
            client = ModelClient(DashboardSettings(_env_file=None), transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body)))
            with self.assertRaises(DashboardError) as error:
                client.call("predict", payload, PredictionResponse)
            self.assertNotIn("secret", str(error.exception))

    def test_capacity_and_policy_labels(self):
        self.service.capacity.acquire()
        self.service.capacity.acquire()
        with self.assertRaises(DashboardError):
            self.service.analyze("a", 55, 30)
        self.service.capacity.release()
        self.service.capacity.release()


class DashboardUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Dataset(ROOT / "data/raw/cmapss/test_FD001.txt")

    def setUp(self):
        from predictive_maintenance.dashboard.app import create_app
        self.service = DashboardService(self.data, FakeModelClient())
        self.app = create_app(DashboardSettings(api_token="server-only-secret", _env_file=None), self.service)
        self.client = self.app.server.test_client()
        self.client.get("/")

    def callback(self, output, inputs, states=(), changed=None, client=None):
        meta = self.app.callback_map[output]
        out = meta["output"]
        as_dict = lambda o: {"id": o.component_id, "property": o.component_property}
        outputs = [as_dict(o) for o in out] if isinstance(out, list) else as_dict(out)
        values = lambda rows: [{"id": i, "property": p, "value": v} for i, p, v in rows]
        return (client or self.client).post("/_dash-update-component", json={"output": output, "outputs": outputs,
            "inputs": values(inputs), "state": values(states), "changedPropIds": [changed or f"{inputs[0][0]}.{inputs[0][1]}"]})

    def analyze(self, cycle=113, unit=55):
        response = self.callback("analysis-ref.data", [("engine", "value", unit), ("cycle", "value", cycle)])
        self.assertEqual(response.status_code, 200, response.text)
        return response.json["response"]["analysis-ref"]["data"]

    def details(self, ref, unit=55, cycle=113):
        response = self.callback("detail-ref.data", [("analysis-ref", "data", ref)],
            [("engine", "value", unit), ("cycle", "value", cycle)])
        self.assertEqual(response.status_code, 200, response.text)
        return response.json["response"]["detail-ref"]["data"]

    def test_layout_has_no_timer_or_secret(self):
        response = self.client.get("/_dash-layout")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Historical analysis", response.text)
        self.assertNotIn("server-only-secret", response.text)
        self.assertNotIn('"type":"Interval"', response.text)
        for label in ("Analyze snapshot", "Build trend", "Explain prediction"):
            self.assertNotIn(label, response.text)
        dependencies = self.client.get("/_dash-dependencies").json
        analysis = next(d for d in dependencies if d["output"] == "analysis-ref.data")
        self.assertFalse(analysis["prevent_initial_call"])
        self.assertEqual(analysis["inputs"], [{"id": "engine", "property": "value"}, {"id": "cycle", "property": "value"}])
        detail = next(d for d in dependencies if d["output"] == "detail-ref.data")
        self.assertEqual(detail["inputs"], [{"id": "analysis-ref", "property": "data"}])
        self.assertEqual(self.client.get("/health").json["engines"], 100)

    def test_callback_analysis_and_selection_clear(self):
        ref = self.analyze()
        detail = self.details(ref)
        output = next(k for k in self.app.callback_map if "rul-gauge.figure" in k)
        response = self.callback(output, [("analysis-ref", "data", ref), ("detail-ref", "data", detail), ("engine", "value", 55), ("cycle", "value", 113)])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Analysis ready for engine 055", response.text)
        self.assertIn("No sustained alert", response.text)
        self.assertEqual(len(response.json["response"]["rul-trend"]["figure"]["data"][0]["x"]), 12)
        self.assertEqual(len(response.json["response"]["shap-chart"]["figure"]["data"][0]["y"]), 13)
        changed = self.callback(output, [("analysis-ref", "data", ref), ("detail-ref", "data", detail), ("engine", "value", 55), ("cycle", "value", 40)])
        self.assertIn("Awaiting analysis", changed.text)
        self.assertNotIn("Analysis ready", changed.text)
        self.assertEqual(changed.json["response"]["shap-chart"]["figure"]["data"], [])

    def test_auto_details_skip_stale_and_failed_selections(self):
        warmup = self.analyze(20)
        self.assertIsNone(self.details(warmup, cycle=20))
        self.assertEqual(self.service.client.calls, [])
        ref = self.analyze()
        calls = len(self.service.client.calls)
        self.assertIsNone(self.details(ref, cycle=40))
        self.assertIsNone(self.details(ref, unit=1))
        self.assertEqual(len(self.service.client.calls), calls)

    def test_auto_analysis_follows_engine_cycle_but_not_sensor_selection(self):
        for unit, cycle in ((55, 40), (1, 30)):
            ref = self.analyze(cycle, unit)
            detail = self.details(ref, unit, cycle)
            with self.client.session_transaction() as session:
                snapshot = self.service.resolve(session["viewer"], unit, cycle, ref, detail)
            self.assertEqual(snapshot["explanation"]["cycle"], cycle)
            self.assertEqual(snapshot["explanation"]["unit_id"], unit)
            self.assertEqual(snapshot["rul_history"][-1]["cycle"], cycle)
        calls = len(self.service.client.calls)
        output = next(k for k in self.app.callback_map if "sensor-chart.figure" in k)
        response = self.callback(output, [("engine", "value", 1), ("cycle", "value", 30), ("sensors", "value", ["s_12"])], changed="sensors.value")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.service.client.calls), calls)

    def test_auto_detail_failure_preserves_successful_panels(self):
        original = self.service.client.call
        for failed in ("trend", "explain"):
            with self.subTest(failed=failed):
                self.service.client.call = original
                ref = self.analyze()
                def fail_one(endpoint, payload, schema):
                    if endpoint == failed or (failed == "trend" and endpoint == "predict" and payload["window"][-1]["cycle"] < 113):
                        raise DashboardError("Temporarily unavailable")
                    return original(endpoint, payload, schema)
                self.service.client.call = fail_one
                detail = self.details(ref)
                self.assertIn("Temporarily unavailable", detail["error"])
                with self.client.session_transaction() as session:
                    snapshot = self.service.resolve(session["viewer"], 55, 113, ref, detail)
                self.assertEqual(snapshot["prediction"]["rul_cycles"], 70)
                self.assertEqual(snapshot["rul_history"] is None, failed == "trend")
                self.assertEqual(snapshot["explanation"] is None, failed == "explain")

    def test_warmup_callback_and_session_isolation(self):
        self.assertIn("cycle 30 or later", self.analyze(20)["error"])
        ref = self.analyze()
        second = self.app.server.test_client()
        second.get("/")
        out = next(k for k in self.app.callback_map if "report-download.data" in k)
        response = self.callback(out, [("export-png", "n_clicks", 1), ("export-pdf", "n_clicks", 0)],
            [("analysis-ref", "data", ref), ("detail-ref", "data", None), ("engine", "value", 55), ("cycle", "value", 113), ("sensors", "value", ["s_11"])], client=second)
        self.assertIn("Wait for analysis of the selected engine", response.text)
        self.assertNotIn("report-download", response.json["response"])

    def test_explanation_and_downloads(self):
        ref = self.analyze()
        detail_ref = self.details(ref)
        output = next(k for k in self.app.callback_map if "report-download.data" in k)
        for kind, signature in (("png", b"\x89PNG"), ("pdf", b"%PDF")):
            response = self.callback(output, [("export-png", "n_clicks", 1), ("export-pdf", "n_clicks", 1)],
                [("analysis-ref", "data", ref), ("detail-ref", "data", detail_ref), ("engine", "value", 55), ("cycle", "value", 113), ("sensors", "value", ["s_11"])],
                changed=f"export-{kind}.n_clicks")
            self.assertEqual(response.status_code, 200, response.text[:500])
            data = response.json["response"]["report-download"]["data"]
            self.assertTrue(base64.b64decode(data["content"]).startswith(signature))
            self.assertEqual(data["filename"], f"FD001_engine055_cycle113.{kind}")

    def test_authentication_protects_callbacks_and_assets(self):
        from predictive_maintenance.dashboard.app import create_app
        with self.assertRaises(ValidationError):
            DashboardSettings(host="0.0.0.0", _env_file=None)
        app = create_app(DashboardSettings(auth_mode="basic", username="tester", password="test-password-123", _env_file=None), self.service)
        client = app.server.test_client()
        for route in ("/", "/_dash-layout", "/_dash-dependencies", "/assets/style.css", "/health"):
            self.assertEqual(client.get(route).status_code, 401)
            self.assertEqual(client.get(route, auth=("tester", "test-password-123")).status_code, 200)
        self.assertEqual(client.post("/_dash-update-component", json={}).status_code, 401)
        self.assertEqual(client.get("/", headers={"Host": "untrusted.example"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
