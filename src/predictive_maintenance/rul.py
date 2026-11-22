"""LSTM serving transformations, independent of HTTP."""
import numpy as np
import pandas as pd


def scaled_window(raw, sensors, scaler):
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (30, len(sensors)) or not np.isfinite(raw).all():
        raise ValueError("Expected a finite 30-cycle sensor window")
    scaled = scaler.transform(pd.DataFrame(raw, columns=sensors)).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite scaled window")
    return scaled[np.newaxis]


def predict_rul(raw, sensors, scaler, model, cap=125.0):
    x = scaled_window(raw, sensors, scaler)
    prediction = np.asarray(model(x, training=False))
    if prediction.shape != (1, 1) or not np.isfinite(prediction).all():
        raise ValueError("Invalid model prediction")
    outside = ((np.asarray(raw) < scaler.data_min_) | (np.asarray(raw) > scaler.data_max_)).any()
    return {"rul_cycles": float(np.clip(prediction.item() * cap, 0, cap)),
            "rul_cap_cycles": cap,
            "warnings": ["Some readings are outside the training scaler range; they were not clipped."] if outside else []}
