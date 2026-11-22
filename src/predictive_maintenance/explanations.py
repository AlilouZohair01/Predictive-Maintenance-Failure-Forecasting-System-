"""Expected-gradient attribution accounting in RUL cycles."""
import numpy as np
from .rul import scaled_window, predict_rul


def aggregate(values, raw, sensors, cap, base, prediction):
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected one model output")
        values = values[0]
    values = np.asarray(values)
    if values.shape == (1, 30, 13, 1):
        values = values[..., 0]
    if values.shape != (1, 30, 13) or not np.isfinite(values).all():
        raise ValueError("Unexpected SHAP shape or non-finite contributions")
    contributions = values[0].sum(axis=0) * cap
    means = np.asarray(raw).mean(axis=0)
    rows = [{"sensor": s, "contribution_cycles": float(v), "raw_window_mean": float(m)}
            for s, v, m in zip(sensors, contributions, means)]
    rows.sort(key=lambda r: abs(r["contribution_cycles"]), reverse=True)
    reconstructed = float(base + contributions.sum())
    return {"predicted_rul_cycles": prediction, "base_value_cycles": base,
            "reconstructed_rul_cycles": reconstructed, "explanation_gap_cycles": prediction - reconstructed,
            "sensor_contributions": rows, "top_positive": [r for r in rows if r["contribution_cycles"] > 0][:3],
            "top_negative": [r for r in rows if r["contribution_cycles"] < 0][:3]}


def explain(raw, sensors, scaler, model, explainer, base, background_version):
    prediction = predict_rul(raw, sensors, scaler, model)
    x = scaled_window(raw, sensors, scaler)
    values = explainer.shap_values(x, nsamples=500, rseed=42)
    result = aggregate(values, raw, sensors, 125.0, base, prediction["rul_cycles"])
    return {**result, "method": "shap.GradientExplainer", "sample_budget": 500, "seed": 42,
            "background_version": background_version,
            "warnings": prediction["warnings"] + ["Approximate explanation: the gap is not model prediction error."]}
