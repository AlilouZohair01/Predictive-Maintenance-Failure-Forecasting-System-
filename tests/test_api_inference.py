import json
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api_helpers import ROOT, fixture
from predictive_maintenance.api.config import Settings
from predictive_maintenance.api.main import create_app
from predictive_maintenance.api.registry import Registry
from predictive_maintenance.api.schemas import SENSORS
from predictive_maintenance.rul import predict_rul, scaled_window
from predictive_maintenance.explanations import aggregate
from predictive_maintenance.survival import conditional_survival
from predictive_maintenance.anomaly import apply_threshold


class TransformationTests(unittest.TestCase):
    def test_rul_scaling_feature_order_and_invalid_output(self):
        scaler = Mock()
        scaler.transform.return_value = np.zeros((30, 13))
        scaler.data_min_, scaler.data_max_ = np.zeros(13), np.ones(13)
        model = Mock(return_value=np.array([[0.5]]))
        result = predict_rul(np.zeros((30, 13)), SENSORS, scaler, model)
        self.assertEqual(result["rul_cycles"], 62.5)
        self.assertEqual(list(scaler.transform.call_args.args[0].columns), SENSORS)
        scaler.fit.assert_not_called()
        model.assert_called_once()
        self.assertEqual(model.call_args.args[0].shape, (1, 30, 13))
        for value, expected in ((2, 125), (-1, 0)):
            model.return_value = np.array([[value]])
            self.assertEqual(predict_rul(np.zeros((30, 13)), SENSORS, scaler, model)["rul_cycles"], expected)
        for value in (np.array([[np.nan]]), np.zeros((2, 1))):
            model.return_value = value
            with self.assertRaises(ValueError):
                predict_rul(np.zeros((30, 13)), SENSORS, scaler, model)
        with self.assertRaises(ValueError):
            scaled_window(np.zeros((1, 13)), SENSORS, scaler)
        scaler.transform.return_value[:] = np.inf
        with self.assertRaises(ValueError):
            scaled_window(np.zeros((30, 13)), SENSORS, scaler)

    def test_shap_signed_sum_shape_and_gap(self):
        values = np.zeros((1, 30, 13, 1))
        values[0, :, 0, 0] = 0.001
        values[0, :, 1, 0] = -0.002
        for shape in (values, values[..., 0], [values[..., 0]]):
            result = aggregate(shape, np.ones((30, 13)), SENSORS, 125, 80, 77)
            self.assertAlmostEqual(result["reconstructed_rul_cycles"], 76.25)
            self.assertAlmostEqual(result["explanation_gap_cycles"], 0.75)
            self.assertEqual(result["top_negative"][0]["sensor"], "s_3")
            self.assertAlmostEqual(result["top_positive"][0]["contribution_cycles"], 3.75)
        for bad in (np.zeros(13), [values, values], np.full((1, 30, 13), np.nan)):
            with self.assertRaises(ValueError):
                aggregate(bad, np.zeros((30, 13)), SENSORS, 125, 80, 77)

    def test_breslow_steps_conditioning_and_median(self):
        scaler, model = Mock(), Mock()
        scaler.transform.return_value = np.zeros((1, 13))
        model.predict_partial_hazard.return_value = pd.Series([1.0])
        model.baseline_cumulative_hazard_ = pd.DataFrame({"hazard": [0.2, 1.0]}, index=[2.0, 5.0])
        metadata = {"landmark_cycle": 30, "maximum_supported_time_since_landmark": 5, "features": SENSORS}
        result = conditional_survival(np.zeros((30, 13)), 32, 3, {"scaler": scaler, "model": model}, metadata)
        np.testing.assert_allclose([p["survival_probability"] for p in result["curve"]], [1, 1, 1, np.exp(-0.8)])
        self.assertEqual(result["median_remaining_cycles"], 3)
        scaler.fit.assert_not_called()
        model.predict_partial_hazard.return_value = pd.Series([np.inf])
        with np.errstate(invalid="ignore"), self.assertRaises(ValueError):
            conditional_survival(np.zeros((30, 13)), 32, 3, {"scaler": scaler, "model": model}, metadata)

    def test_strict_threshold_and_third_flag(self):
        scores = pd.DataFrame({"unit_number": [1] * 5, "zscore_score": [1, 2, 2, 2, 1]})
        flags, alerts = apply_threshold(scores, "zscore", 1, 3)
        np.testing.assert_array_equal(flags, [False, True, True, True, False])
        np.testing.assert_array_equal(alerts, [False, False, False, True, False])


class RealArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = Registry(ROOT / "models")
        cls.app = create_app(Settings(_env_file=None), lambda root: cls.registry)
        cls.context = TestClient(cls.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_prediction_matches_frozen_notebook_computation(self):
        response = self.client.post("/v1/predict", json=fixture("predict"))
        self.assertEqual(response.status_code, 200, response.text)
        expected = json.loads((ROOT / "tests/fixtures/api/prediction_engine55.json").read_text())
        self.assertAlmostEqual(response.json()["rul_cycles"], expected["rul_cycles"], delta=expected["absolute_tolerance_cycles"])

    def test_survival_matches_saved_notebook_curve(self):
        response = self.client.post("/v1/survival", json=fixture("survival"))
        self.assertEqual(response.status_code, 200, response.text)
        expected = pd.read_csv(ROOT / "tests/fixtures/api/survival_engine55.csv")
        np.testing.assert_allclose([p["survival_probability"] for p in response.json()["curve"]], expected["55"], atol=1e-10)

    def test_survival_support_boundaries(self):
        base = fixture("survival")
        for current, expected_horizon in ((361, 1), (362, 0)):
            response = self.client.post("/v1/survival", json={**base, "current_cycle": current})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result["horizon_truncated"])
            self.assertEqual(result["evaluated_horizon_cycles"], expected_horizon)
            self.assertIsNone(result["failure_probability_at_requested_horizon"])
            self.assertEqual(result["curve"][0]["survival_probability"], 1)
        response = self.client.post("/v1/survival", json={**base, "current_cycle": 363})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "SURVIVAL_AGE_UNSUPPORTED")

    def test_all_detectors_match_notebook_and_warmup(self):
        expected = pd.read_csv(ROOT / "tests/fixtures/api/anomaly_engine55.csv")
        for detector in ("autoencoder", "isolation_forest", "zscore"):
            for policy in ("healthy_q98", "validation_f1"):
                response = self.client.post("/v1/anomaly", json={**fixture("anomaly"), "detector": detector, "threshold_policy": policy})
                self.assertEqual(response.status_code, 200, response.text)
                rows = response.json()["history"]
                self.assertIsNone(rows[28]["anomaly_score"])
                self.assertIsNone(rows[29]["sustained_alert"])
                self.assertIsNone(rows[30]["sustained_alert"])
                self.assertIsInstance(rows[31]["sustained_alert"], bool)
                self.assertEqual(rows[33]["smoothing_samples"], 5)
                np.testing.assert_allclose([r["anomaly_score"] for r in rows[29:]], expected[detector + "_score"].iloc[29:], rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal([r["is_anomaly"] for r in rows[29:]], expected[detector + "_" + policy + "_flag"].iloc[29:])
                np.testing.assert_array_equal([r["sustained_alert"] for r in rows[31:]], expected[detector + "_" + policy + "_alert"].iloc[31:])

    def test_prefix_invariance_and_short_history(self):
        base = fixture("anomaly")
        full = self.client.post("/v1/anomaly", json=base).json()["history"]
        for length in (1, 29, 30, 31, 32, 34, 50):
            response = self.client.post("/v1/anomaly", json={**base, "history": base["history"][:length], "return_history": False})
            self.assertEqual(response.status_code, 200, response.text)
            latest = response.json()["latest"]
            target = full[length - 1]
            if target["anomaly_score"] is not None:
                self.assertAlmostEqual(latest["anomaly_score"], target["anomaly_score"], delta=1e-5)
            self.assertEqual(latest["sustained_alert"], target["sustained_alert"])
            self.assertIsNone(response.json()["history"])

    def test_real_shap_explanation_accounting(self):
        response = self.client.post("/v1/explain", json=fixture("explain"))
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(result["sensor_contributions"]), 13)
        self.assertEqual(result["sample_budget"], 500)
        summed = sum(row["contribution_cycles"] for row in result["sensor_contributions"])
        self.assertAlmostEqual(result["base_value_cycles"] + summed, result["reconstructed_rul_cycles"], places=6)
        self.assertAlmostEqual(result["predicted_rul_cycles"] - result["reconstructed_rul_cycles"], result["explanation_gap_cycles"], places=6)
        self.assertTrue(np.isfinite(result["explanation_gap_cycles"]))

    def test_metadata_and_readiness(self):
        self.assertEqual(self.client.get("/health/ready").status_code, 200)
        data = self.client.get("/v1/models").json()
        self.assertEqual(data["anomaly"]["default_detector"], "autoencoder")
        self.assertEqual(len(data["survival"]["hazard_ratios_per_training_sd"]), 13)
        schema = self.client.get("/openapi.json").json()
        for route in ("predict", "survival", "anomaly", "explain"):
            self.assertIn("FD001_engine55", schema["paths"]["/v1/" + route]["post"]["requestBody"]["content"]["application/json"]["examples"])
