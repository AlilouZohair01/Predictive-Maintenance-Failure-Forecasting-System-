"""Auto-refreshing stream workspace mounted at /live/ on the dashboard server."""
from pathlib import Path

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from flask import session
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import figures
from .data import SENSOR_INFO
from .export import render_report
from .live_service import LiveService, FeedClient, ObservedDataset, LiveBusy
from .service import DashboardError


def trends(view):
    snapshot = view["snapshot"]
    if not snapshot:
        return figures.empty("Predictions begin at cycle 30.", 350)
    points = snapshot["rul_history"]
    risk = view["risk_history"]
    horizon = view["run"]["config"]["survival_horizon_cycles"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.18)
    fig.add_trace(go.Scatter(x=[p["cycle"] for p in points], y=[p["rul_cycles"] for p in points],
                            name="Saved RUL", mode="lines", line_color=figures.TEAL), row=1, col=1)
    fig.add_trace(go.Scatter(x=[p["cycle"] for p in risk], y=[p["probability"] for p in risk],
                            name=f"Failure within {horizon} cycles", mode="lines+markers", connectgaps=False,
                            line_color=figures.ORANGE, hovertemplate="%{y:.1%}<extra>Experimental Cox</extra>"), row=2, col=1)
    figures.style(fig, 350)
    fig.update_yaxes(title_text="RUL · cycles", range=[0, 130], row=1, col=1)
    fig.update_yaxes(title_text=f"{horizon}-cycle risk", tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Processed observation cycle", row=2, col=1)
    return fig


def create_live_app(server, settings, service=None):
    from .app import panel, table, GRAPH_CONFIG
    service = service or LiveService(FeedClient(settings))
    app = Dash(__name__, server=server, url_base_pathname="/live/", title="Engine Lab · Live replay",
               assets_folder=str(Path(__file__).parent / "assets"), update_title=None)
    app.live_service = service

    def chart(fig):
        height = fig.layout.height or 300
        return dcc.Graph(figure=fig, config=GRAPH_CONFIG, responsive=True,
                         style={"width": "100%", "height": f"{height}px", "flexShrink": 0})

    def message(text):
        return html.Div(text, className="status-message", role="status")

    def render(view, selected):
        feed, run, snapshot = view["feed"], view["run"], view["snapshot"]
        cycle = view["cycle"]
        progress = run["progress"]
        source = "Replay closed" if feed["source_state"] == "closed" else "Source stale" if feed["is_stale"] else "Receiving replay"
        status = f"{source} · {progress} · received {feed['received_cycle']} / processed {cycle} · lag {feed['processing_lag_cycles']} cycles"
        job = feed["next_job"]
        if job:
            status += f" · Next job: cycle {job['cycle']} ({job['status']})"
            if job.get("error_code"):
                status += " · " + job["error_code"]
        status += f" · Refreshed {view['refreshed_at'][11:19]} UTC"
        if snapshot and snapshot["prediction"]["warnings"]:
            status += " · Some sensor values are outside the training range; inputs were not clipped."
        survival = snapshot["survival"] if snapshot else None
        probability = survival["failure_probability_at_requested_horizon"] if survival else None
        horizon = run["config"]["survival_horizon_cycles"]
        anomaly = snapshot["anomaly"] if snapshot else None
        state = "Warming up" if not anomaly or anomaly["latest"]["sustained_alert"] is None else "Sustained alert" if anomaly["latest"]["sustained_alert"] else "No sustained alert"
        note = "30 observations required for inference; 32 for a sustained-alert decision."
        if anomaly:
            note = f"Score {anomaly['latest']['anomaly_score']:.3f} / threshold {anomaly['threshold']:.3f}. Not a probability."
        survival_note = "No survival result has been stored yet."
        if survival:
            survival_note = f"Updated at cycle {survival['cycle']}; current processed cycle {cycle}. The curve starts at that update cycle."
            if survival["horizon_truncated"]:
                survival_note += " Requested horizon exceeds fitted support; its probability is unavailable."
            if feed["latest"].get("survival_status") == "unsupported_age":
                survival_note += " Current age is unsupported; this is the last supported curve."
        raw_chart = figures.sensors(pd.DataFrame(view["readings"]), selected) if view["readings"] else figures.empty("Waiting for the worker to process readings.", 350)
        alerts = sorted(view["alerts"], key=lambda a: (a["state"] != "active", {"critical": 0, "warning": 1}.get(a["severity"], 2), -a["id"]))
        alert_table = table(["Engine", "Condition", "Severity", "State", "Opened / last cycle", "Opened (UTC)"], [
            [str(a["unit_id"]), a["rule"].replace("_", " "), a["severity"], a["state"], f"{a['opened_cycle']} / {a['last_cycle']}", a["opened_at"]]
            for a in alerts]) if alerts else message("No alert episodes have been recorded for this run.")
        return html.Div([
            html.Div(f"Run {view['run_id']} / Engine {view['unit']:03d} / Cycle {cycle}", className="selection-title"),
            message(status),
            html.Div([
                html.Section([html.Div("REMAINING USEFUL LIFE", className="eyebrow"), chart(figures.gauge(snapshot)),
                              html.P("Saved prediction · cap 125 cycles", className="metric-footnote")], className="metric-card"),
                html.Section([html.Div("ANOMALY STATUS", className="eyebrow"), html.Div(state, className="metric-word"),
                              html.P(note, className="metric-description")], className="metric-card"),
                html.Section([html.Div(f"FAILURE WITHIN {horizon} CYCLES", className="eyebrow"),
                              html.Div(f"{probability:.1%}" if probability is not None else "Unavailable", className="metric-number"),
                              html.P(f"Estimate from cycle {survival['cycle']}" if survival else "Awaiting a supported estimate", className="metric-description"),
                              html.P("Experimental Cox · test C-index 0.576", className="metric-footnote")], className="metric-card"),
                html.Section([html.Div("PROCESSING", className="eyebrow"), html.Div(progress.title(), className="metric-word"),
                              html.P(f"Pending {run['jobs']['pending']} · processing {run['jobs']['processing']} · failed {run['jobs']['failed']}", className="metric-description"),
                              html.P(f"Last processed: {feed['last_processed_at'] or 'not yet'}", className="metric-footnote")], className="metric-card"),
            ], className="metric-grid"),
            html.Div([
                panel("Received sensor history", "Readings through the processed cycle; queued observations are excluded.", [chart(raw_chart)]),
                panel("Anomaly trajectory", "Stored scores and sustained flags. Refreshing does not trigger inference.", [chart(figures.anomaly(snapshot))]),
                panel("Remaining life & failure risk", "RUL at every scored cycle; risk only at actual survival updates. Missing estimates stay blank.", [chart(trends(view))]),
                panel("Conditional survival", survival_note, [chart(figures.survival(snapshot))]),
            ], className="chart-grid"),
            panel("Run alert history", "All engines in the selected run. Active episodes first. No external notifications are sent.", [alert_table]),
            html.P(f"Model release {run['release_version']} · source {run['config']['source']}. NASA cycles are not wall-clock acquisition timestamps. Alert thresholds are illustrative.", className="panel-note")
        ])

    app.layout = html.Div([
        dcc.Interval(id="live-tick", interval=2000, n_intervals=0), dcc.Store(id="live-ref"), dcc.Store(id="live-explanation"), dcc.Download(id="live-download"),
        html.Aside([
            html.Div([html.Div("EL", className="brand-mark"), html.Div([html.Strong("ENGINE LAB"), html.Span("Replay monitoring")])], className="brand"),
            html.A("Historical analysis", href="/", className="nav-link"), html.A("Live replay", href="/live/", className="nav-link active"),
            html.Div(className="sidebar-divider"), html.Div([
                html.Label("Replay run", htmlFor="live-run"), dcc.Dropdown(id="live-run", options=[], clearable=False),
                html.Div(id="live-runs-status", className="control-help", role="status"),
                html.Label("Engine", htmlFor="live-unit"), dcc.Dropdown(id="live-unit", options=[], clearable=False),
                html.Div(id="live-units-status", className="control-help", role="status"),
                html.Label("Sensors", htmlFor="live-sensors"), dcc.Dropdown(id="live-sensors", multi=True,
                    options=[{"label": s + " · " + label[0], "value": s} for s, label in SENSOR_INFO.items()], value=["s_11", "s_12", "s_14"]),
                html.P("Up to three sensor charts are shown.", className="control-help"),
                dcc.Checklist(id="live-pause", options=[{"label": " Pause updates", "value": "pause"}], value=[]),
                html.P("Updates every 2 seconds. Pausing this view does not stop the replay or worker.", className="control-help"),
            ], className="controls"),
        ], className="sidebar"),
        html.Main([
            html.Header([html.Div([html.Div("STREAM WORKSPACE / FD001", className="eyebrow"), html.H1("Watch the replay unfold"),
                                  html.P("Stored predictions and alerts, updated as new observations are processed.", className="page-subtitle")]),
                         html.Span("Simulated sensor stream", className="mode-badge")], className="page-header"),
            html.Details([html.Summary("Start a dataset replay"), html.P("Keep the app running. In a second project terminal, activate exam_env and run:"),
                          html.Pre("python scripts/replay_stream.py --units 1 55 --interval 0.5"),
                          html.P("The script prints a new run ID. Choose it in the list. Completed runs remain available; no sensors or broker are required.")], className="replay-help"),
            dcc.Loading(html.Div(id="live-panels"), type="dot", delay_show=800),
            panel("Explain a saved prediction", "SHAP runs only on request. The explanation stays pinned to its labeled cycle as the stream advances.", [
                html.Button("Explain displayed snapshot", id="live-explain", n_clicks=0, className="button small"),
                html.Div(id="live-explain-status", className="status-message", role="status"),
                dcc.Graph(id="live-shap", figure=figures.empty("Pause updates, then request an explanation.", 365),
                          config=GRAPH_CONFIG, responsive=True, style={"width": "100%", "height": "365px"}),
            ]),
            html.Div([html.Button("Export PNG", id="live-png", n_clicks=0, className="button secondary"),
                      html.Button("Export PDF", id="live-pdf", n_clicks=0, className="button secondary")], className="live-export"),
            html.Div(id="live-export-status", role="status", className="status-message"),
            html.Footer("Research models · historical dataset replay · no hardware sensor connection", className="page-footer"),
        ], className="workspace")
    ], className="app-shell")

    @app.callback(Output("live-tick", "disabled"), Input("live-pause", "value"))
    def pause(values):
        return "pause" in (values or [])

    @app.callback(Output("live-run", "options"), Output("live-run", "value"), Output("live-runs-status", "children"),
                  Input("live-tick", "n_intervals"), State("live-run", "value"))
    def discover_runs(tick, selected):
        try:
            runs = service.runs()
            options = [{"label": f"{r['run_id']} · {r['progress']}", "value": r["run_id"]} for r in runs]
            value = no_update if selected in [r["run_id"] for r in runs] else runs[0]["run_id"] if runs else None
            return options, value, f"{len(runs)} saved runs" if runs else "No replay yet. Use the replay command on this page."
        except DashboardError as exc:
            return no_update, no_update, str(exc)

    @app.callback(Output("live-unit", "options"), Output("live-unit", "value"), Output("live-units-status", "children"),
                  Input("live-run", "value"), Input("live-tick", "n_intervals"), State("live-unit", "value"))
    def discover_units(run_id, tick, selected):
        if not run_id:
            return [], None, "Choose a replay run."
        try:
            units = service.units(run_id)
            ids = [u["unit_id"] for u in units]
            value = no_update if selected in ids else 55 if 55 in ids else ids[0] if ids else None
            return [{"label": f"Engine {u['unit_id']:03d} · processed {u['processed_cycle']}", "value": u["unit_id"]} for u in units], value, f"{len(units)} engines" if units else "Waiting for the first reading."
        except DashboardError as exc:
            return [], None, str(exc)

    @app.callback(Output("live-ref", "data"), Output("live-panels", "children"), Input("live-run", "value"),
                  Input("live-unit", "value"), Input("live-tick", "n_intervals"), Input("live-sensors", "value"))
    def refresh(run_id, unit, tick, selected):
        if not run_id or unit is None:
            return None, message("Select a run and engine to monitor its saved results.")
        try:
            ref, view = service.poll(session["viewer"], run_id, unit)
            return ref, render(view, selected)
        except LiveBusy:
            return None, message("Refresh capacity is busy. Waiting for the next update.")
        except DashboardError as exc:
            return None, message(str(exc) + " Display cleared; retrying on the next refresh.")

    @app.callback(Output("live-explanation", "data"), Output("live-shap", "figure"), Output("live-explain-status", "children"),
                  Input("live-explain", "n_clicks"), Input("live-run", "value"), Input("live-unit", "value"), State("live-ref", "data"),
                  prevent_initial_call=True, running=[(Output("live-explain", "disabled"), True, False)])
    def explain(clicks, run_id, unit, ref):
        if ctx.triggered_id != "live-explain":
            return None, figures.empty("Request an explanation for this engine.", 365), ""
        try:
            explained_ref, view = service.explain(session["viewer"], ref, run_id, unit)
            result = view["snapshot"]["explanation"]
            rows = sorted(result["sensor_contributions"], key=lambda r: abs(r["contribution_cycles"]), reverse=True)[:3]
            summary = ", ".join(f"{r['sensor']} {r['contribution_cycles']:+.2f} cycles" for r in rows)
            return explained_ref, figures.shap(view["snapshot"]), f"Pinned to run {run_id}, engine {unit}, cycle {view['cycle']}. Top contributions: {summary}. Approximation gap {result['explanation_gap_cycles']:+.2f} cycles."
        except DashboardError as exc:
            return None, figures.empty("Explanation unavailable.", 365), str(exc)

    @app.callback(Output("live-download", "data"), Output("live-export-status", "children"), Input("live-png", "n_clicks"), Input("live-pdf", "n_clicks"),
                  State("live-ref", "data"), State("live-explanation", "data"), State("live-run", "value"), State("live-unit", "value"), State("live-sensors", "value"),
                  prevent_initial_call=True, running=[(Output("live-png", "disabled"), True, False), (Output("live-pdf", "disabled"), True, False)])
    def export(png, pdf, ref, explanation_ref, run_id, unit, selected):
        view = service.resolve(session["viewer"], ref, run_id, unit)
        if not view or not view["snapshot"]:
            return no_update, "Wait for a scored snapshot at cycle 30 or later, then export."
        explained = service.resolve(session["viewer"], explanation_ref, run_id, unit)
        if explained and explained["cycle"] == view["cycle"]:
            view["snapshot"]["explanation"] = explained["snapshot"]["explanation"]
        kind = "pdf" if ctx.triggered_id == "live-pdf" else "png"
        content = render_report(view["snapshot"], ObservedDataset(view), selected, kind)
        return dcc.send_bytes(content, f"{run_id}_engine{unit:03d}_cycle{view['cycle']}.{kind}"), f"Exported run {run_id}, engine {unit}, cycle {view['cycle']}."

    return app
