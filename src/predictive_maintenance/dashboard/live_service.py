"""Read-only stream feeds and immutable, session-owned replay snapshots."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
import threading

import httpx
import pandas as pd

from .service import DashboardError, ModelClient, SnapshotStore
from ..api.schemas import PredictionResponse, AnomalyResponse, SurvivalResponse, ExplanationResponse, SensorReading


class FeedClient(ModelClient):
    def get(self, path, params=None):
        headers = {"Authorization": "Bearer " + self.settings.api_token.get_secret_value()} if self.settings.api_token else {}
        try:
            with httpx.Client(base_url=self.settings.api_url, headers=headers, transport=self.transport,
                              timeout=httpx.Timeout(5, connect=2), trust_env=False) as client:
                response = client.get("/v1/stream/" + path, params=params, headers=headers)
            if response.status_code in (401, 403):
                raise DashboardError("Stream access requires valid API credentials on the dashboard server.")
            if response.status_code != 200:
                raise DashboardError(f"Stream feed unavailable (HTTP {response.status_code}). Check the API and selected run.")
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected object")
            return result
        except httpx.HTTPError:
            raise DashboardError("Cannot reach the stream API. Updates will resume when the API is available.") from None
        except ValueError:
            raise DashboardError("The stream API returned an invalid response.") from None

    def collect(self, path, cursor_key, parameter, **params):
        items, previous = [], None
        for _ in range(20):
            page = self.get(path, {"limit": 100, **params, **({parameter: previous} if previous is not None else {})})
            if not isinstance(page.get("items"), list):
                raise DashboardError("The stream list response was invalid.")
            items.extend(page["items"])
            cursor = page.get(cursor_key)
            if cursor is None:
                return items
            if cursor == previous or not page["items"]:
                raise DashboardError("The stream feed returned an invalid pagination cursor.")
            previous = cursor
        raise DashboardError("This feed exceeds the dashboard's 2,000-item view limit.")


def valid_run(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise DashboardError("Select an available replay run.")
    return run_id


class LiveBusy(DashboardError):
    pass


class LiveService:
    def __init__(self, client, snapshots=None):
        self.client = client
        self.snapshots = snapshots or SnapshotStore()
        self.capacity = threading.BoundedSemaphore(2)
        self.explain_capacity = threading.BoundedSemaphore(1)

    def runs(self):
        return sorted(self.client.collect("runs", "next_after", "after"), key=lambda r: r["created_at"], reverse=True)

    def units(self, run_id):
        return self.client.collect(f"runs/{valid_run(run_id)}/units", "next_after_unit", "after_unit")

    def poll(self, owner, run_id, unit):
        valid_run(run_id)
        if isinstance(unit, bool) or not isinstance(unit, int) or unit < 1:
            raise DashboardError("Select an engine in this replay run.")
        if not self.capacity.acquire(blocking=False):
            raise LiveBusy("A refresh is already in progress.")
        try:
            root = f"runs/{run_id}"
            run = self.client.get(root)["run"]
            feed = self.client.get(f"{root}/units/{unit}/latest")
            end = feed["processed_cycle"]
            if feed["run_id"] != run_id or feed["unit_id"] != unit or not 0 <= end <= 1000:
                raise DashboardError("The stream response does not match this selection.")
            release = run["release_version"]
            if feed["release_version"] != release:
                raise DashboardError("The stream contains mismatched model releases.")
            # Latest is read first; freeze history at that committed boundary even if the worker advances.
            readings = self.client.get(f"{root}/units/{unit}/readings", {"through_cycle": end, "limit": 1000})["items"] if end else []
            raw = [SensorReading.model_validate(row["reading"]).model_dump(mode="json") for row in readings]
            records = self.client.get(f"{root}/units/{unit}/predictions", {"limit": 1000})["items"] if end else []
            records = [p for p in records if p["cycle"] <= end]
            if [r["cycle"] for r in raw] != list(range(1, end + 1)) or [r["cycle"] for r in records] != list(range(1, end + 1)):
                raise DashboardError("Stream history is incomplete. Waiting for a consistent refresh.")

            def checked(value, schema, cycle):
                result = schema.model_validate(value).model_dump(mode="json")
                if (result["unit_id"], result["cycle"], result["release_version"]) != (unit, cycle, release):
                    raise DashboardError("A stored model result does not match its stream snapshot.")
                return result

            scores, trend, survival_points = [], [], []
            for record in records:
                cycle = record["cycle"]
                if (record["run_id"], record["unit_id"], record["release_version"]) != (run_id, unit, release):
                    raise DashboardError("Stored history mixes different runs or model releases.")
                if record["prediction"] is not None:
                    record["prediction"] = checked(record["prediction"], PredictionResponse, cycle)
                    record["anomaly"] = checked(record["anomaly"], AnomalyResponse, cycle)
                    scores.append(record["anomaly"]["latest"])
                    trend.append({"cycle": cycle, "rul_cycles": record["prediction"]["rul_cycles"]})
                if record["survival"] is not None:
                    result = checked(record["survival"], SurvivalResponse, cycle)
                    survival_points.append({"cycle": cycle, "probability": result["failure_probability_at_requested_horizon"]})
            latest = records[-1] if records else None
            if latest and latest["cycle"] != feed["latest"]["cycle"]:
                raise DashboardError("Latest prediction and stored history disagree.")
            survival = feed["latest_survival"]
            if survival:
                if not 30 <= survival["cycle"] <= end:
                    raise DashboardError("The survival result is ahead of the committed snapshot.")
                survival = checked(survival, SurvivalResponse, survival["cycle"])
            snapshot = None
            if latest and latest["prediction"]:
                anomaly = deepcopy(latest["anomaly"])
                anomaly["history"] = scores
                snapshot = {"unit": unit, "cycle": end, "run_id": run_id, "mode": "stream",
                    "dataset_fingerprint": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16],
                    "analyzed_at": latest["processed_at"], "prediction": latest["prediction"], "anomaly": anomaly,
                    "survival": survival, "explanation": None, "rul_history": trend, "warnings": []}
            alerts = self.client.collect(f"{root}/alerts", "next_after_id", "after_id", state="all")
            view = {"unit": unit, "cycle": end, "run_id": run_id, "run": run, "feed": feed,
                    "readings": raw, "snapshot": snapshot, "risk_history": survival_points, "alerts": alerts,
                    "refreshed_at": datetime.now(timezone.utc).isoformat()}
            token = self.snapshots.put(owner, view)
            return {"token": token, "unit": unit, "cycle": end, "run_id": run_id}, view
        except (KeyError, TypeError, ValueError):
            raise DashboardError("The stream feed is incomplete or invalid. No new results were displayed.") from None
        finally:
            self.capacity.release()

    def resolve(self, owner, ref, run_id, unit):
        if not isinstance(ref, dict) or ref.get("run_id") != run_id or ref.get("unit") != unit:
            return None
        view = self.snapshots.get(ref.get("token"), owner, unit, ref.get("cycle"))
        return view if view and view["run_id"] == run_id else None

    def explain(self, owner, ref, run_id, unit):
        view = self.resolve(owner, ref, run_id, unit)
        if not view or not view["snapshot"]:
            raise DashboardError("Wait for a scored snapshot at cycle 30 or later. Pause updates to inspect one cycle.")
        if not self.explain_capacity.acquire(blocking=False):
            raise DashboardError("An explanation is already running. Please try again shortly.")
        try:
            result = self.client.call("explain", {"unit_id": unit, "window": view["readings"][-30:]}, ExplanationResponse)
            prediction = view["snapshot"]["prediction"]
            if (result["release_version"], result["model_version"]) != (prediction["release_version"], prediction["model_version"]):
                raise DashboardError("The running model differs from this saved run. SHAP cannot explain an older model release.")
            view["snapshot"]["explanation"] = result
            token = self.snapshots.put(owner, view)
            return {**ref, "token": token}, view
        finally:
            self.explain_capacity.release()


class ObservedDataset:
    """Export exactly the stream's saved readings, not a local dataset look-up."""
    def __init__(self, view):
        self.view = view

    def prefix(self, unit, cycle):
        if (unit, cycle) != (self.view["unit"], self.view["cycle"]):
            raise ValueError("Snapshot selection changed")
        return pd.DataFrame(self.view["readings"])
