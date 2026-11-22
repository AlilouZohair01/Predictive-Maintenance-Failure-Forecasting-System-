"""Explicit release preparation; never run automatically in the serving process.

Run from the project root using the training environment. No models are fitted.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictive_maintenance.api.schemas import SENSORS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    models = ROOT / "models"
    lstm = models / "lstm_fd001_v2"
    source = ROOT / "data/processed/lstm_fd001/X_train.npy"
    x = np.load(source, mmap_mode="r", allow_pickle=False)
    indices = np.random.default_rng(42).choice(len(x), 100, replace=False)
    np.save(lstm / "shap_background.npy", np.asarray(x[indices], dtype=np.float32))
    write_json(lstm / "shap_background.json", {
        "seed": 42, "samples": 100, "sensor_names": SENSORS, "scaled": True,
        "source": "data/processed/lstm_fd001/X_train.npy", "source_sha256": digest(source),
        "training_indices": indices.tolist(), "model_sha256": digest(lstm / "model.keras"),
        "scaler_sha256": digest(lstm / "scaler.pkl")})
    columns = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]
    frame = pd.read_csv(ROOT / "data/raw/cmapss/test_FD001.txt", sep=r"\s+", header=None, names=columns)
    engine = frame[frame.unit_id == 55]
    readings = engine[["cycle"] + SENSORS].to_dict(orient="records")
    for row in readings:
        row["cycle"] = int(row["cycle"])
    examples = {
        "predict": {"unit_id": 55, "dataset": "FD001", "window": readings[-30:]},
        "explain": {"unit_id": 55, "dataset": "FD001", "window": readings[-30:]},
        "survival": {"unit_id": 55, "dataset": "FD001", "initial_window": readings[:30],
                     "current_cycle": int(engine.cycle.max()), "horizon_cycles": 100},
        "anomaly": {"unit_id": 55, "dataset": "FD001", "history": readings, "return_history": True},
    }
    for name, payload in examples.items():
        write_json(ROOT / f"src/predictive_maintenance/api/examples/{name}.json", payload)
    # Freeze expected software parity values from notebook reports, independently of API code.
    scores = pd.read_csv(ROOT / "reports/tables/anomaly_fd001/test_cycle_scores.csv")
    scores[scores.unit_number == 55].to_csv(ROOT / "tests/fixtures/api/anomaly_engine55.csv", index=False)
    curves = pd.read_csv(ROOT / "reports/tables/survival_fd001/conditional_survival.csv", index_col=0)
    curves[["55"]].to_csv(ROOT / "tests/fixtures/api/survival_engine55.csv")
    # Direct notebook LSTM computation as a regression fixture (no API service imports).
    import joblib
    import tensorflow as tf
    model = tf.keras.models.load_model(lstm / "model.keras", compile=False)
    scaler = joblib.load(lstm / "scaler.pkl")
    values = scaler.transform(engine[SENSORS].iloc[-30:]).astype(np.float32)[None]
    prediction = float(np.clip(np.asarray(model(values, training=False)).item() * 125, 0, 125))
    write_json(ROOT / "tests/fixtures/api/prediction_engine55.json", {
        "unit_id": 55, "rul_cycles": prediction, "absolute_tolerance_cycles": 0.001,
        "source": "Direct saved-model notebook computation; evaluation only, no model selection"})
    files = ["lstm_fd001_v2/" + p for p in ("model.keras", "scaler.pkl", "metadata.json", "shap_background.npy", "shap_background.json")]
    files += ["cox_fd001/cox_bundle.pkl", "cox_fd001/metadata.json"]
    files += ["anomaly_fd001/" + p for p in ("detectors.pkl", "autoencoder.keras", "metadata.json")]
    packages = ["tensorflow", "numpy", "pandas", "scipy", "scikit-learn", "lifelines", "shap", "joblib"]
    manifest = {"schema_version": 1, "dataset": "FD001", "sensors": SENSORS,
                "capabilities": ["predict", "survival", "anomaly", "explain"],
                "files": {p: digest(models / p) for p in files},
                "versions": {p: importlib.metadata.version(p) for p in packages}}
    manifest["release_version"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    write_json(models / "serving_manifest.json", manifest)
    print(json.dumps({"release_version": manifest["release_version"], "prediction_engine55": prediction,
                      "history_cycles": len(readings), "background_windows": len(indices)}))


if __name__ == "__main__":
    main()
