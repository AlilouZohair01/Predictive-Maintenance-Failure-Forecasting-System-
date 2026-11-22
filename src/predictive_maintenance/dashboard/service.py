"""HTTP-only model client and bounded, session-owned analysis snapshots."""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import threading
import time
import uuid

import httpx
import numpy as np
from ..api.schemas import PredictionResponse, AnomalyResponse, SurvivalResponse, ExplanationResponse


class DashboardError(Exception):
    pass


class SnapshotStore:
    def __init__(self, capacity=128, ttl=1800, clock=time.monotonic):
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self.items, self.lock = OrderedDict(), threading.Lock()

    def put(self, owner, snapshot):
        token = uuid.uuid4().hex
        with self.lock:
            self.items[token] = (owner, self.clock(), deepcopy(snapshot))
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)
        return token

    def get(self, token, owner, unit, cycle):
        with self.lock:
            item = self.items.get(token or "")
            if not item or item[0] != owner:
                return None
            if self.clock() - item[1] > self.ttl:
                del self.items[token]
                return None
            snapshot = item[2]
            if (snapshot["unit"], snapshot["cycle"]) != (unit, cycle):
                return None
            return deepcopy(snapshot)

    def reviewed(self, owner):
        with self.lock:
            latest = {}
            for who, created, snapshot in self.items.values():
                if who == owner and self.clock() - created <= self.ttl:
                    if snapshot["unit"] not in latest or snapshot["analyzed_at"] >= latest[snapshot["unit"]]["analyzed_at"]:
                        latest[snapshot["unit"]] = deepcopy(snapshot)
        return sorted(latest.values(), key=lambda s: s["prediction"]["rul_cycles"])


class ModelClient:
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport

    def call(self, endpoint, payload, schema):
        headers = {}
        if self.settings.api_token:
            headers["Authorization"] = "Bearer " + self.settings.api_token.get_secret_value()
        try:
            with httpx.Client(base_url=self.settings.api_url, headers=headers, transport=self.transport,
                              timeout=httpx.Timeout(130, connect=3), trust_env=False) as client:
                response = client.post("/v1/" + endpoint, json=payload)
        except httpx.HTTPError:
            raise DashboardError("Cannot reach the model API. Start FastAPI on its configured address, then try again.") from None
        if response.status_code in (401, 403):
            raise DashboardError("The model API requires valid credentials. Configure PM_API_TOKEN on the dashboard server.")
        if response.status_code == 503:
            raise DashboardError("The model API is busy or still starting. Wait a moment and try again.")
        if response.status_code != 200:
            raise DashboardError(f"The model API could not complete this request (HTTP {response.status_code}).")
        try:
            result = schema.model_validate(response.json()).model_dump(mode="json")
        except ValueError:
            raise DashboardError("The model API returned an unexpected result. No result has been displayed.") from None
        expected_cycle = payload.get("current_cycle") or payload.get("window", payload.get("history"))[-1]["cycle"]
        if result["unit_id"] != payload["unit_id"] or result["cycle"] != expected_cycle:
            raise DashboardError("The API result did not match the selected engine and cycle.")
        return result


class DashboardService:
    def __init__(self, dataset, client, snapshots=None):
        self.dataset, self.client = dataset, client
        self.snapshots = snapshots or SnapshotStore()
        self.capacity = threading.BoundedSemaphore(2)

    def analyze(self, owner, unit, cycle):
        history = self.dataset.readings(unit, cycle)
        if cycle < 30:
            raise DashboardError(f"This selection has {cycle} observations. Select cycle 30 or later to run the models.")
        if not self.capacity.acquire(blocking=False):
            raise DashboardError("Two analyses are already running. Please try again shortly.")
        try:
            common = {"unit_id": unit, "dataset": "FD001"}
            prediction = self.client.call("predict", {**common, "window": history[-30:]}, PredictionResponse)
            anomaly = self.client.call("anomaly", {**common, "history": history, "return_history": True}, AnomalyResponse)
            errors, survival = [], None
            try:
                survival = self.client.call("survival", {**common, "initial_window": history[:30], "current_cycle": cycle, "horizon_cycles": 100}, SurvivalResponse)
            except DashboardError as exc:
                errors.append("Survival unavailable: " + str(exc))
            releases = {r["release_version"] for r in (prediction, anomaly, survival) if r}
            if len(releases) != 1:
                raise DashboardError("The model release changed during analysis. Analyze the selection again.")
            if anomaly["latest"]["cycle"] != cycle or anomaly["history"] is None or [p["cycle"] for p in anomaly["history"]] != list(range(1, cycle + 1)):
                raise DashboardError("The anomaly history does not match the selected observations.")
            snapshot = {"analysis_id": uuid.uuid4().hex, "unit": unit, "cycle": cycle,
                        "dataset_fingerprint": self.dataset.fingerprint, "analyzed_at": datetime.now(timezone.utc).isoformat(),
                        "prediction": prediction, "anomaly": anomaly, "survival": survival,
                        "explanation": None, "rul_history": None, "warnings": errors}
            return self.snapshots.put(owner, snapshot)
        finally:
            self.capacity.release()

    def resolve(self, owner, unit, cycle, analysis_ref, detail_ref=None):
        base = self.snapshots.get((analysis_ref or {}).get("token"), owner, unit, cycle)
        detail = self.snapshots.get((detail_ref or {}).get("token"), owner, unit, cycle)
        return detail if base and detail and detail["analysis_id"] == base["analysis_id"] else base

    def enrich(self, owner, unit, cycle, analysis_ref, detail_ref, action):
        snapshot = self.resolve(owner, unit, cycle, analysis_ref, detail_ref)
        if not snapshot:
            raise DashboardError("Analyze this engine and cycle first. Previous analyses may expire after 30 minutes.")
        if not self.capacity.acquire(blocking=False):
            raise DashboardError("Analysis capacity is busy. Please try again shortly.")
        try:
            history = self.dataset.readings(unit, cycle)
            release = snapshot["prediction"]["release_version"]
            if action == "explain":
                result = self.client.call("explain", {"unit_id": unit, "window": history[-30:]}, ExplanationResponse)
                if result["release_version"] != release or result["model_version"] != snapshot["prediction"]["model_version"]:
                    raise DashboardError("The model changed since analysis. Analyze this selection again before explaining it.")
                snapshot["explanation"] = result
            elif action == "trend":
                cycles = sorted(set(np.linspace(30, cycle, min(12, cycle - 29), dtype=int).tolist()))
                points = []
                for end in cycles:
                    result = snapshot["prediction"] if end == cycle else self.client.call(
                        "predict", {"unit_id": unit, "window": history[end - 30:end]}, PredictionResponse)
                    if result["release_version"] != release:
                        raise DashboardError("The model changed during trend calculation. Analyze this selection again.")
                    failure_probability = None
                    try:
                        survival = snapshot["survival"] if end == cycle else self.client.call("survival", {
                            "unit_id": unit, "initial_window": history[:30], "current_cycle": end,
                            "horizon_cycles": 100}, SurvivalResponse)
                    except DashboardError:
                        survival = None
                    if survival:
                        if survival["release_version"] != release:
                            raise DashboardError("The model changed during trend calculation. Analyze this selection again.")
                        failure_probability = survival["failure_probability_at_requested_horizon"]
                    points.append({"cycle": end, "rul_cycles": result["rul_cycles"],
                                   "failure_probability_100": failure_probability})
                snapshot["rul_history"] = points
            else:
                raise DashboardError("Unknown analysis action")
            return self.snapshots.put(owner, snapshot)
        finally:
            self.capacity.release()
