import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from fastapi.testclient import TestClient
from api_helpers import ROOT, FakeRegistry
from predictive_maintenance.api.registry import Registry, verify_release, check_scaler
from predictive_maintenance.api.main import create_app
from predictive_maintenance.api.config import Settings
from predictive_maintenance.api.schemas import SENSORS


class StartupTests(unittest.TestCase):
    def test_missing_manifest_fails_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                with TestClient(create_app(Settings(model_root=directory, _env_file=None))):
                    pass

    def test_manifest_and_artifact_corruption(self):
        manifest = json.loads((ROOT / "models/serving_manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "serving_manifest.json"
            altered = copy.deepcopy(manifest)
            altered["dataset"] = "FD004"
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                verify_release(directory)
            path.write_text(json.dumps(manifest))
            # A tiny file cannot satisfy the trusted manifest's SHA-256.
            first = next(iter(manifest["files"]))
            artifact = Path(directory) / first
            artifact.parent.mkdir(parents=True)
            artifact.write_text("corrupt")
            with self.assertRaises(ValueError):
                verify_release(directory)

    def test_manifest_wrong_files_and_features(self):
        original = json.loads((ROOT / "models/serving_manifest.json").read_text())
        for field, value in (("sensors", SENSORS[::-1]), ("files", {"../untrusted.pkl": "0"}), ("capabilities", ["predict"])):
            manifest = {**original, field: value}
            manifest.pop("release_version")
            manifest["release_version"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "serving_manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    verify_release(directory)

    def test_library_compatibility_is_checked(self):
        with patch("predictive_maintenance.api.registry.importlib.metadata.version", return_value="0.0"):
            with self.assertRaisesRegex(ValueError, "dependency mismatch"):
                verify_release(ROOT / "models")

    def test_scaler_order_and_width(self):
        for count, names in ((12, SENSORS), (13, SENSORS[::-1])):
            with self.assertRaises(ValueError):
                check_scaler(SimpleNamespace(n_features_in_=count, feature_names_in_=names), SENSORS)

    def test_wrong_neural_model_shapes_fail_startup(self):
        valid_lstm = SimpleNamespace(input_shape=(None, 30, 13), output_shape=(None, 1))
        valid_ae = SimpleNamespace(input_shape=(None, 390), output_shape=(None, 390))
        bad = SimpleNamespace(input_shape=(None, 50, 13), output_shape=(None, 1))
        for models in ([bad, valid_ae], [valid_lstm, bad]):
            with patch("tensorflow.keras.models.load_model", side_effect=models):
                with self.assertRaisesRegex(ValueError, "shape"):
                    Registry(ROOT / "models")

    def test_metadata_contracts(self):
        def fresh():
            r = Registry.__new__(Registry)
            r.manifest = json.loads((ROOT / "models/serving_manifest.json").read_text())
            for attr, path in (("lstm_meta", "lstm_fd001_v2/metadata.json"), ("cox_meta", "cox_fd001/metadata.json"),
                               ("anomaly_meta", "anomaly_fd001/metadata.json"), ("background_meta", "lstm_fd001_v2/shap_background.json")):
                setattr(r, attr, json.loads((ROOT / "models" / path).read_text()))
            return r
        for attr, key, value in (("lstm_meta", "sensor_names", SENSORS[::-1]), ("lstm_meta", "max_rul", 1),
                                 ("cox_meta", "landmark_cycle", 50), ("anomaly_meta", "smoothing", 1),
                                 ("anomaly_meta", "selected_model", "zscore"), ("anomaly_meta", "thresholds", {}),
                                 ("background_meta", "seed", 1), ("background_meta", "model_sha256", "wrong")):
            r = fresh()
            getattr(r, attr)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_metadata()
        r = fresh()
        r.anomaly_meta["thresholds"]["autoencoder"]["healthy_q98"] = float("nan")
        with self.assertRaises(ValueError):
            r.validate_metadata()
