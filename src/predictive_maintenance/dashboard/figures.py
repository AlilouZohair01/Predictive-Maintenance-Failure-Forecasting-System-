"""Consistent Plotly charts; all x axes stop at the selected observation."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textwrap import wrap
from .data import SENSOR_INFO

TEAL, ORANGE, RED, INK, MUTED = "#087f8c", "#d8932b", "#c24b43", "#203143", "#68798a"


def style(fig, height=300):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=48, r=20, t=24, b=42),
                      font=dict(family="Segoe UI, Arial, sans-serif", size=13, color=MUTED),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hovermode="x unified", showlegend=False)
    fig.update_xaxes(gridcolor="#edf1f4", zeroline=False)
    fig.update_yaxes(gridcolor="#edf1f4", zeroline=False)
    return fig


def empty(message, height=300):
    fig = style(go.Figure(), height)
    fig.add_annotation(text="<br>".join(wrap(message, 30)), x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def sensors(frame, selected):
    selected = [s for s in (selected or []) if s in SENSOR_INFO][:3]
    if not selected:
        return empty("Choose up to three sensors in the sidebar.", 350)
    fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True, vertical_spacing=.13)
    colors = [TEAL, "#587bb8", "#b57b43"]
    for i, (sensor, color) in enumerate(zip(selected, colors), 1):
        label, unit = SENSOR_INFO[sensor]
        fig.add_trace(go.Scatter(x=frame.cycle, y=frame[sensor], name=sensor, mode="lines",
                                line=dict(color=color, width=1.6), hovertemplate=f"{sensor}: %{{y:.3f}} {unit}<extra></extra>"), row=i, col=1)
        fig.update_yaxes(title_text=f"{sensor} · {unit}", row=i, col=1)
    style(fig, 350)
    fig.update_xaxes(title_text="Observed cycle", row=len(selected), col=1)
    return fig


def gauge(snapshot):
    if not snapshot:
        return empty("Not analyzed", 165)
    value = snapshot["prediction"]["rul_cycles"]
    fig = go.Figure(go.Indicator(mode="number+gauge", value=value,
        number=dict(valueformat=".1f", font=dict(size=37, color=INK)),
        gauge=dict(axis=dict(range=[0, 125], tickvals=[0, 50, 125], tickfont=dict(size=11)),
                   bar=dict(color=TEAL, thickness=.25), bgcolor="white", borderwidth=0,
                   steps=[dict(range=[0, 20], color="#f5d9d5"), dict(range=[20, 50], color="#f8e9c9"),
                          dict(range=[50, 125], color="#d7eeeb")],
                   threshold=dict(line=dict(color=INK, width=2), thickness=.65, value=value))))
    fig.update_layout(height=165, margin=dict(l=24, r=24, t=30, b=12), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Segoe UI"))
    return fig


def anomaly(snapshot):
    if not snapshot:
        return empty("Anomaly scores appear after analysis.", 350)
    result = snapshot["anomaly"]
    history = result["history"]
    fig = go.Figure(go.Scatter(x=[p["cycle"] for p in history], y=[p["anomaly_score"] for p in history],
                              mode="lines", name="Smoothed score", line=dict(color=TEAL, width=2)))
    flagged = [p for p in history if p["sustained_alert"]]
    fig.add_trace(go.Scatter(x=[p["cycle"] for p in flagged], y=[p["anomaly_score"] for p in flagged],
                            name="Sustained alert", mode="markers", marker=dict(color=RED, size=5)))
    fig.add_hline(y=result["threshold"], line_dash="dash", line_color=ORANGE,
                  annotation_text="Threshold", annotation_position="top left")
    style(fig, 350)
    fig.update_xaxes(title_text="Observed cycle")
    fig.update_yaxes(title_text="Anomaly score")
    return fig


def survival(snapshot):
    if not snapshot or not snapshot["survival"]:
        return empty("Conditional survival appears after analysis.")
    result = snapshot["survival"]
    fig = go.Figure(go.Scatter(x=[p["future_cycles"] for p in result["curve"]],
        y=[p["survival_probability"] for p in result["curve"]], mode="lines", fill="tozeroy",
        fillcolor="rgba(8,127,140,.07)", line=dict(color=TEAL, width=2, shape="hv"), name="Survival"))
    style(fig)
    fig.update_yaxes(range=[0, 1.03], tickformat=".0%", title_text="Survival probability")
    fig.update_xaxes(title_text="Future cycles from selection")
    return fig


def rul_trend(snapshot):
    if not snapshot or not snapshot["rul_history"]:
        return empty("RUL and failure-risk trends appear after calculation.")
    points = snapshot["rul_history"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.18)
    fig.add_trace(go.Scatter(x=[p["cycle"] for p in points], y=[p["rul_cycles"] for p in points],
                            mode="lines+markers", line=dict(color=TEAL, width=2), marker=dict(size=5), name="Predicted RUL"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[p["cycle"] for p in points], y=[p.get("failure_probability_100") for p in points],
                            mode="lines+markers", connectgaps=False, line=dict(color=ORANGE, width=2), marker=dict(size=5),
                            name="Failure within 100 cycles · Cox", hovertemplate="100-cycle failure probability: %{y:.1%}<extra>Experimental Cox</extra>"), row=2, col=1)
    style(fig, 350)
    fig.update_yaxes(range=[0, 130], title_text="RUL · cycles", row=1, col=1)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="100-cycle risk", row=2, col=1)
    fig.update_xaxes(title_text="Observed cycle · sampled checkpoints", row=2, col=1)
    return fig


def shap(snapshot):
    if not snapshot or not snapshot["explanation"]:
        return empty("Sensor contributions appear when the explanation is available.", 365)
    rows = sorted(snapshot["explanation"]["sensor_contributions"], key=lambda r: abs(r["contribution_cycles"]))
    values = [r["contribution_cycles"] for r in rows]
    fig = go.Figure(go.Bar(x=values, y=[r["sensor"] for r in rows], orientation="h",
                          marker_color=[TEAL if v >= 0 else ORANGE for v in values],
                          text=[f"{v:+.2f}" for v in values], textposition="auto", name="Contribution"))
    style(fig, 365)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Contribution to predicted RUL · cycles", zeroline=True, zerolinecolor="#8794a1")
    return fig
