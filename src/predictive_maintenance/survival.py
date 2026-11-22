"""Conditional Breslow step-function survival, matching notebook 05."""
import numpy as np
import pandas as pd

LIMITATIONS = ["Experimental Cox model: test concordance index 0.576.",
               "Covariates are frozen means of cycles 1–30; later sensor changes are not used."]


class UnsupportedAge(ValueError):
    pass


def conditional_survival(raw, current_cycle, horizon, bundle, metadata):
    landmark = metadata["landmark_cycle"]
    age = current_cycle - landmark
    support = int(metadata["maximum_supported_time_since_landmark"])
    if age > support:
        raise UnsupportedAge("Current age exceeds the fitted survival time support")
    features = metadata["features"]
    means = pd.DataFrame([np.asarray(raw).mean(axis=0)], columns=features)
    x = pd.DataFrame(bundle["scaler"].transform(means), columns=features)
    model = bundle["model"]
    risk = float(model.predict_partial_hazard(x).iloc[0])
    base = model.baseline_cumulative_hazard_.iloc[:, 0]
    times = np.r_[0.0, base.index.to_numpy(dtype=float)]
    hazards = np.r_[0.0, base.to_numpy(dtype=float)]
    evaluate = lambda t: hazards[np.searchsorted(times, t, side="right") - 1]
    evaluated = min(horizon, support - age)
    future = np.arange(evaluated + 1)
    survival = np.exp(-(evaluate(age + future) - evaluate(age)) * risk)
    if not np.isfinite(survival).all() or not np.isfinite(risk):
        raise ValueError("Invalid conditional survival result")
    crossing = np.flatnonzero(survival <= 0.5)
    return {"requested_horizon_cycles": horizon, "evaluated_horizon_cycles": evaluated,
            "horizon_truncated": evaluated < horizon,
            "failure_probability_at_requested_horizon": float(1 - survival[-1]) if evaluated == horizon else None,
            "median_remaining_cycles": int(future[crossing[0]]) if len(crossing) else None,
            "median_unavailable_reason": None if len(crossing) else "Survival does not reach 0.5 within the returned supported horizon.",
            "curve": [{"future_cycles": int(t), "survival_probability": float(s)} for t, s in zip(future, survival)],
            "limitations": LIMITATIONS}
