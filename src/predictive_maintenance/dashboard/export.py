"""Standalone PNG/PDF analysis reports without a browser or Kaleido dependency."""
from io import BytesIO
import threading
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .data import SENSOR_INFO

LOCK = threading.Lock()


def render_report(snapshot, dataset, selected, kind):
    if kind not in ("png", "pdf"):
        raise ValueError("Unsupported report format")
    sensor = next((s for s in (selected or []) if s in SENSOR_INFO), "s_11")
    frame = dataset.prefix(snapshot["unit"], snapshot["cycle"])
    teal, ink, orange = "#087f8c", "#203143", "#d8932b"
    with LOCK:
        fig = Figure(figsize=(14, 10), facecolor="white")
        FigureCanvasAgg(fig)
        fig.subplots_adjust(left=.08, right=.96, bottom=.18, top=.78, hspace=.60, wspace=.25)
        fig.text(.08, .94, f"ENGINE LAB  /  Engine {snapshot['unit']:03d} · Cycle {snapshot['cycle']}", fontsize=22, color=ink, weight="bold")
        mode = f"Replay snapshot  |  Run {snapshot['run_id']}" if snapshot.get("mode") == "stream" else "Historical analysis  |  No future observations used"
        fig.text(.08, .90, "NASA C-MAPSS FD001  |  " + mode, fontsize=10, color="#68798a")
        rul = snapshot["prediction"]["rul_cycles"]
        latest = snapshot["anomaly"]["latest"]
        flag = "Alert" if latest["sustained_alert"] else "Not yet available" if latest["sustained_alert"] is None else "No sustained alert"
        survival = snapshot["survival"]
        probability = survival["failure_probability_at_requested_horizon"] if survival else None
        chance = f"{probability:.1%}" if probability is not None else "Unsupported / unavailable"
        fig.text(.08, .85, f"Predicted RUL: {rul:.2f} cycles (cap 125)     |     {flag}", fontsize=11, color=ink)
        risk_label = f"Failure within {survival['requested_horizon_cycles']} future cycles, estimated at observation cycle {survival['cycle']}: {chance}" if survival else "Survival estimate unavailable"
        fig.text(.08, .815, risk_label, fontsize=10, color=ink)
        axes = fig.subplots(2, 2)
        ax = axes[0, 0]
        ax.plot(frame.cycle, frame[sensor], color=teal, linewidth=1.2)
        ax.set(title=f"Sensor history · {sensor}", xlabel="Observed cycle", ylabel=SENSOR_INFO[sensor][1])
        ax = axes[0, 1]
        history = snapshot["anomaly"]["history"]
        ax.plot([p["cycle"] for p in history], [np.nan if p["anomaly_score"] is None else p["anomaly_score"] for p in history], color=teal)
        ax.axhline(snapshot["anomaly"]["threshold"], color=orange, linestyle="--", label="Threshold")
        ax.set(title="Anomaly score · autoencoder", xlabel="Observed cycle", ylabel="Smoothed score")
        ax.legend(frameon=False, fontsize=9)
        ax = axes[1, 0]
        if survival:
            ax.step([p["future_cycles"] for p in survival["curve"]], [p["survival_probability"] for p in survival["curve"]], where="post", color=teal)
            ax.set_ylim(0, 1.03)
        else:
            ax.text(.5, .5, "Survival unavailable", ha="center", transform=ax.transAxes)
        ax.set(title="Conditional survival · experimental Cox", xlabel=f"Future cycles from cycle {survival['cycle']}" if survival else "Future cycles", ylabel="Survival probability")
        ax = axes[1, 1]
        explanation = snapshot["explanation"]
        if explanation:
            rows = sorted(explanation["sensor_contributions"], key=lambda r: abs(r["contribution_cycles"]))
            values = [r["contribution_cycles"] for r in rows]
            ax.barh([r["sensor"] for r in rows], values, color=[teal if v >= 0 else orange for v in values])
            ax.set_title(f"SHAP · gap {explanation['explanation_gap_cycles']:+.2f} cycles")
            ax.set_xlabel("Signed RUL contribution (cycles)")
        else:
            ax.text(.5, .5, "SHAP not available for this snapshot", ha="center", transform=ax.transAxes)
            ax.set_title("Prediction explanation")
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
            ax.tick_params(labelsize=9)
        fig.text(.08, .075, f"Analyzed: {snapshot['analyzed_at']}\nModel release: {snapshot['prediction']['release_version']}  |  Dataset: {snapshot['dataset_fingerprint']}", fontsize=9, color="#68798a")
        fig.text(.08, .025, "Research estimates, not maintenance certification. Cox test C-index 0.576; fixed first-30-cycle covariates.\nSensor panel shows the first selected sensor. RUL bands are illustrative; anomaly scores are not probabilities.", fontsize=9, color="#68798a")
        output = BytesIO()
        fig.savefig(output, format=kind, dpi=160)
        return output.getvalue()
