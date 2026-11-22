"""Trusted, fingerprinted model release, loaded once during each worker's lifespan."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading

import joblib
import numpy as np
import pandas as pd

from .schemas import SENSORS
from ..anomaly import ALGORITHMS, score_cycles, apply_threshold
from ..rul import predict_rul
from ..survival import conditional_survival, LIMITATIONS
from ..explanations import explain

REQUIRED_FILES = {
    "lstm_fd001_v2/model.keras", "lstm_fd001_v2/scaler.pkl", "lstm_fd001_v2/metadata.json",
    "lstm_fd001_v2/shap_background.npy", "lstm_fd001_v2/shap_background.json",
    "cox_fd001/cox_bundle.pkl", "cox_fd001/metadata.json", "anomaly_fd001/detectors.pkl",
    "anomaly_fd001/autoencoder.keras", "anomaly_fd001/metadata.json"}


def verify_release(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "serving_manifest.json").read_text(encoding="utf-8"))
    fingerprint = dict(manifest)
    version = fingerprint.pop("release_version")
    expected = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    if version != expected or manifest["schema_version"] != 1 or manifest["dataset"] != "FD001":
        raise ValueError("Invalid release manifest")
    if manifest["sensors"] != SENSORS or set(manifest["files"]) != REQUIRED_FILES:
        raise ValueError("Unexpected release feature or file contract")
    if set(manifest["capabilities"]) != {"predict", "survival", "anomaly", "explain"}:
        raise ValueError("Incomplete release capabilities")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Artifact fingerprint mismatch")
    for package, version in manifest["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Release dependency mismatch: {package}")
    return manifest


def check_scaler(scaler, names):
    if scaler.n_features_in_ != len(names) or list(scaler.feature_names_in_) != names:
        raise ValueError("Scaler feature order mismatch")


class Registry:
    def __init__(self, root):
        root = Path(root)
        self.manifest = verify_release(root)
        self.release_version = self.manifest["release_version"]
        read = lambda p: json.loads((root / p).read_text(encoding="utf-8"))
        self.lstm_meta = read("lstm_fd001_v2/metadata.json")
        self.cox_meta = read("cox_fd001/metadata.json")
        self.anomaly_meta = read("anomaly_fd001/metadata.json")
        self.background_meta = read("lstm_fd001_v2/shap_background.json")
        self.sensors = SENSORS.copy()
        self.validate_metadata()
        import tensorflow as tf
        import shap
        self.lstm = tf.keras.models.load_model(root / "lstm_fd001_v2/model.keras", compile=False)
        self.scaler = joblib.load(root / "lstm_fd001_v2/scaler.pkl")
        self.cox = joblib.load(root / "cox_fd001/cox_bundle.pkl")
        self.detectors = joblib.load(root / "anomaly_fd001/detectors.pkl")
        self.autoencoder = tf.keras.models.load_model(root / "anomaly_fd001/autoencoder.keras", compile=False)
        check_scaler(self.scaler, self.sensors)
        check_scaler(self.cox["scaler"], self.cox_meta["features"])
        check_scaler(self.detectors["scaler"], self.sensors)
        if list(self.cox["model"].params_.index) != self.cox_meta["features"]:
            raise ValueError("Cox feature order mismatch")
        if self.cox["metadata"] != self.cox_meta:
            raise ValueError("Cox embedded metadata mismatch")
        if tuple(self.lstm.input_shape) != (None, 30, 13) or tuple(self.lstm.output_shape) != (None, 1):
            raise ValueError("Unexpected LSTM shape")
        if tuple(self.autoencoder.input_shape) != (None, 390) or tuple(self.autoencoder.output_shape) != (None, 390):
            raise ValueError("Unexpected autoencoder shape")
        if self.detectors["forest"].n_features_in_ != 13:
            raise ValueError("Unexpected forest feature count")
        base = self.cox["model"].baseline_cumulative_hazard_
        if float(base.index.max()) != self.cox_meta["maximum_supported_time_since_landmark"]:
            raise ValueError("Survival support mismatch")
        self.background = np.load(root / "lstm_fd001_v2/shap_background.npy", allow_pickle=False)
        if self.background.shape != (100, 30, 13) or not np.isfinite(self.background).all():
            raise ValueError("Invalid frozen background")
        self.background_version = self.manifest["files"]["lstm_fd001_v2/shap_background.npy"][:16]
        self.explainer = shap.GradientExplainer(self.lstm, self.background)
        self.explanation_lock = threading.Lock()
        self.base_cycles = float(np.asarray(self.lstm(self.background, training=False)).mean() * 125)
        # Warm real transformations using a training-background window, not test labels.
        raw = self.scaler.inverse_transform(self.background[0])
        self.predict(raw)
        self.survival(raw, 30, 10)
        self.anomaly(raw, 1, "autoencoder", "healthy_q98", False)
        with self.explanation_lock:
            smoke = self.explainer.shap_values(self.background[:1], nsamples=2, rseed=42)
            if not np.isfinite(np.asarray(smoke)).all():
                raise ValueError("Explanation warm-up failed")

    def validate_metadata(self):
        a, c, l, b = self.anomaly_meta, self.cox_meta, self.lstm_meta, self.background_meta
        if l["sensor_names"] != SENSORS or c["sensor_names"] != SENSORS or a["sensors"] != SENSORS:
            raise ValueError("Metadata sensor order mismatch")
        if l["sequence_length"] != 30 or l["max_rul"] != 125 or l["dataset"] != "FD001":
            raise ValueError("Unsupported LSTM contract")
        if c["landmark_cycle"] != 30 or c["features"] != [s + "_mean_first30" for s in SENSORS]:
            raise ValueError("Unsupported Cox contract")
        if (a["window_length"], a["smoothing"], a["consecutive"]) != (30, 5, 3):
            raise ValueError("Unsupported anomaly temporal contract")
        if a["selected_model"] != "autoencoder" or a["primary_policy"] != "healthy_q98":
            raise ValueError("Anomaly default mismatch")
        if set(a["thresholds"]) != set(ALGORITHMS):
            raise ValueError("Incomplete detector thresholds")
        for values in a["thresholds"].values():
            if set(values) != {"healthy_q98", "validation_f1"} or not np.isfinite(list(values.values())).all():
                raise ValueError("Invalid threshold policies")
        if b["sensor_names"] != SENSORS or not b["scaled"] or b["samples"] != 100 or b["seed"] != 42:
            raise ValueError("Invalid background provenance")
        for name in ("model", "scaler"):
            filename = "model.keras" if name == "model" else "scaler.pkl"
            if b[name + "_sha256"] != self.manifest["files"]["lstm_fd001_v2/" + filename]:
                raise ValueError("Background belongs to a different model/scaler")

    def model_version(self, capability):
        prefix = {"predict": "lstm_fd001_v2", "explain": "lstm_fd001_v2", "survival": "cox_fd001", "anomaly": "anomaly_fd001"}[capability]
        files = {k: v for k, v in self.manifest["files"].items() if k.startswith(prefix + "/")}
        return prefix + ":" + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]

    def predict(self, raw):
        return predict_rul(raw, self.sensors, self.scaler, self.lstm, self.lstm_meta["max_rul"])

    def survival(self, raw, current_cycle, horizon):
        return conditional_survival(raw, current_cycle, horizon, self.cox, self.cox_meta)

    def anomaly(self, raw, unit_id, detector, policy, return_history):
        frame = pd.DataFrame(raw, columns=self.sensors)
        frame["unit_number"] = unit_id
        frame["time_in_cycles"] = np.arange(1, len(frame) + 1)
        meta = self.anomaly_meta
        scored = score_cycles(frame, self.sensors, self.detectors["scaler"], self.detectors["forest"],
                              self.autoencoder, meta["window_length"], meta["smoothing"])
        threshold = meta["thresholds"][detector][policy]
        flags, alerts = apply_threshold(scored, detector, threshold, meta["consecutive"])
        rows = []
        for i, row in scored.iterrows():
            cycle = i + 1
            ready = cycle >= 30
            alert_ready = cycle >= 32
            raw_score, score = float(row[detector + "_raw"]), float(row[detector + "_score"])
            if ready and not np.isfinite([raw_score, score]).all():
                raise ValueError("Invalid anomaly score")
            rows.append({"cycle": cycle, "raw_score": raw_score if ready else None,
                         "anomaly_score": score if ready else None, "is_anomaly": bool(flags[i]) if ready else None,
                         "sustained_alert": bool(alerts[i]) if alert_ready else None,
                         "status": "ready" if ready else "warmup", "alert_history_ready": alert_ready,
                         "smoothing_samples": min(5, max(0, cycle - 29))})
        return {"detector": detector, "threshold_policy": policy, "threshold": threshold,
                "latest": rows[-1], "history": rows if return_history else None}

    def explain(self, raw):
        with self.explanation_lock:
            return explain(raw, self.sensors, self.scaler, self.lstm, self.explainer, self.base_cycles, self.background_version)

    def describe(self):
        return {"dataset": "FD001", "release_version": self.release_version,
                "capabilities": {k: self.model_version(k) for k in self.manifest["capabilities"]},
                "sensor_order": self.sensors, "input_units": "Raw native C-MAPSS units; see API guide sensor table.",
                "rul": {"window_cycles": 30, "cap_cycles": 125, "test_rmse_cycles": 13.361126950231279},
                "survival": {"initial_cycles": [1, 30], "support_since_landmark": self.cox_meta["maximum_supported_time_since_landmark"],
                             "limitations": LIMITATIONS, "hazard_ratios_per_training_sd": self.cox["model"].hazard_ratios_.to_dict()},
                "anomaly": {"default_detector": self.anomaly_meta["selected_model"], "default_policy": self.anomaly_meta["primary_policy"],
                            "thresholds": self.anomaly_meta["thresholds"], "smoothing_cycles": 5, "consecutive_flags": 3,
                            "limitation": "Validation-selected autoencoder underperformed simpler baselines on test; anomaly scores are not probabilities."},
                "explanation": {"method": "GradientExplainer", "samples": 500, "seed": 42, "background_version": self.background_version},
                "limitations": ["FD001 research model, not validated for generic IoT or maintenance safety.", "No calibrated RUL interval or automated maintenance policy."]}
