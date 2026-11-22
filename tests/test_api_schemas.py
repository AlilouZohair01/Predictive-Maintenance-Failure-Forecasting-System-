import copy
import unittest
from pydantic import ValidationError
from api_helpers import fixture
from predictive_maintenance.api.schemas import WindowRequest, SurvivalRequest, AnomalyRequest
from predictive_maintenance.api.config import Settings


class SchemaTests(unittest.TestCase):
    def test_valid_real_examples_and_new_unit_id(self):
        for name, schema in [("predict", WindowRequest), ("survival", SurvivalRequest), ("anomaly", AnomalyRequest)]:
            payload = fixture(name)
            payload["unit_id"] = 99999
            self.assertEqual(schema.model_validate(payload).unit_id, 99999)

    def test_sensor_types_and_missing_extra(self):
        for value in (True, "123.4", "bad", None, float("nan"), float("inf"), -float("inf")):
            payload = fixture("predict")
            payload["window"][0]["s_2"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                WindowRequest.model_validate(payload)
        for action in ("missing", "extra", "nested_unit"):
            payload = fixture("predict")
            if action == "missing":
                del payload["window"][0]["s_2"]
            else:
                payload["window"][0]["unexpected" if action == "extra" else "unit_id"] = 1
            with self.subTest(action=action), self.assertRaises(ValidationError):
                WindowRequest.model_validate(payload)

    def test_lengths_order_and_dataset(self):
        base = fixture("predict")
        variants = [base["window"][:-1], base["window"] + [base["window"][-1]], base["window"][::-1]]
        for index in (1, 2):
            rows = copy.deepcopy(base["window"])
            rows[index]["cycle"] = rows[0]["cycle"]
            variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValidationError):
                WindowRequest.model_validate({**base, "window": rows})
        for field, value in (("dataset", "FD002"), ("unit_id", True), ("unit_id", 0), ("unit_id", "1"), ("source", "x" * 129), ("extra", 1)):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                WindowRequest.model_validate({**base, field: value})

    def test_timestamp_timezone(self):
        payload = fixture("predict")
        payload["window"][0]["timestamp"] = "2026-11-20T10:00:00+01:00"
        WindowRequest.model_validate(payload)
        payload["window"][0]["timestamp"] = "2026-11-20T10:00:00"
        with self.assertRaises(ValidationError):
            WindowRequest.model_validate(payload)

    def test_survival_contract(self):
        base = fixture("survival")
        for key, value in (("current_cycle", 29), ("current_cycle", True), ("horizon_cycles", 0), ("horizon_cycles", "100")):
            with self.assertRaises(ValidationError):
                SurvivalRequest.model_validate({**base, key: value})
        for row in base["initial_window"]:
            row["cycle"] += 1
        with self.assertRaises(ValidationError):
            SurvivalRequest.model_validate(base)

    def test_anomaly_complete_history_and_choices(self):
        base = fixture("anomaly")
        for key, value in (("history", []), ("history", base["history"][1:]), ("history", base["history"] * 10),
                           ("detector", "unknown"), ("threshold_policy", "custom"), ("threshold", 0.2), ("return_history", "yes")):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                AnomalyRequest.model_validate({**base, key: value})

    def test_secure_configuration(self):
        with self.assertRaises(ValidationError):
            Settings(auth_mode="bearer", api_token="short", _env_file=None)
        self.assertEqual(Settings(auth_mode="bearer", api_token="a" * 32, _env_file=None).auth_mode, "bearer")
