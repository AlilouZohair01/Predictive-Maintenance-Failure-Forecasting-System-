"""Historical dashboard. Run with python -m predictive_maintenance.dashboard.app."""
import hmac
from pathlib import Path
import secrets

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from flask import Flask, request, session, Response

from .config import DashboardSettings
from .data import Dataset, SENSOR_INFO
from .service import DashboardService, ModelClient, DashboardError
from . import figures
from .export import render_report
from ..streaming.alerts import transition
from ..streaming.schemas import AlertPolicy

GRAPH_CONFIG = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def graph(identifier, height=300):
    return dcc.Graph(id=identifier, figure=figures.empty("Select a snapshot to begin.", height),
                     config=GRAPH_CONFIG, responsive=True,
                     style={"width": "100%", "height": f"{height}px", "flexShrink": 0})


def panel(title, subtitle, children, identifier=None, action=None):
    return html.Section([
        html.Div([html.Div([html.H2(title), html.P(subtitle, className="panel-subtitle")]), action], className="panel-heading"),
        *children], className="panel", **({"id": identifier} if identifier else {}))


def table(headers, rows):
    return html.Div(html.Table([html.Thead(html.Tr([html.Th(h) for h in headers])),
        html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows])]), className="table-scroll")


def anomaly_episodes(snapshot):
    if not snapshot:
        return []
    policy, state, rows = AlertPolicy().model_dump(), None, []
    for point in snapshot["anomaly"]["history"]:
        state, event = transition("sustained_anomaly", state, point["cycle"], None, point, policy)
        if event == "raised":
            rows.append([f"Cycle {point['cycle']}", "Sustained anomaly", "Active at selection"])
        elif event == "resolved" and rows:
            rows[-1][2] = f"Resolved at {point['cycle']}"
    if snapshot["prediction"]["rul_cycles"] < 20:
        rows.append([f"Cycle {snapshot['cycle']}", "RUL below 20 cycles", "Current selection"])
    return rows[-8:]


def create_app(settings=None, service=None, live_service=None):
    settings = settings or DashboardSettings()
    service = service or DashboardService(Dataset(settings.data_file), ModelClient(settings))
    dataset = service.dataset
    server = Flask(__name__)
    server.secret_key = settings.session_secret.get_secret_value() if settings.session_secret else secrets.token_hex(32)
    server.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict", SESSION_COOKIE_SECURE=settings.secure_cookie,
                         MAX_CONTENT_LENGTH=1_048_576)

    @server.before_request
    def access():
        if request.host.split(":", 1)[0].lower() not in settings.allowed_hosts:
            return Response("Host is not allowed", status=400)
        if settings.auth_mode == "basic":
            credentials = request.authorization
            valid = credentials is not None and credentials.type.lower() == "basic"
            if valid:
                valid = hmac.compare_digest((credentials.username or "").encode(), settings.username.encode()) and hmac.compare_digest(
                    (credentials.password or "").encode(), settings.password.get_secret_value().encode())
            if not valid:
                return Response("Authentication required", status=401, headers={"WWW-Authenticate": 'Basic realm="Engine Lab"'})
        if "viewer" not in session:
            session["viewer"] = secrets.token_hex(24)

    @server.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        if not request.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @server.get("/health")
    def health():
        return {"status": "ready", "mode": "historical", "modes": ["historical", "live_replay"], "engines": len(dataset.engines)}

    app = Dash(__name__, server=server, title="Engine Lab · FD001", update_title="Analyzing…",
               assets_folder=str(Path(__file__).parent / "assets"))
    app.dashboard_service = service
    default_unit = 55 if 55 in dataset.engines else next(iter(dataset.engines))
    default_cycle = dataset.length(default_unit)
    app.layout = html.Div([
        dcc.Download(id="report-download"),
        html.A("Skip to analysis", href="#workspace", className="skip-link"),
        html.Aside([
            html.Div([html.Div("EL", className="brand-mark"), html.Div([html.Strong("ENGINE LAB"), html.Span("Predictive maintenance")])], className="brand"),
            html.Div("WORKSPACE", className="eyebrow sidebar-eyebrow"),
            html.A([html.Span("◫", className="nav-icon"), "Engine overview"], href="#workspace", className="nav-link active"),
            html.A("Live replay", href="/live/", className="nav-link"),
            html.A([html.Span("∿", className="nav-icon"), "Sensor history"], href="#sensor-panel", className="nav-link"),
            html.A([html.Span("≡", className="nav-icon"), "Model explanation"], href="#explanation-panel", className="nav-link"),
            html.Div(className="sidebar-divider"),
            html.Div([
                html.Label("Engine", htmlFor="engine"),
                dcc.Dropdown(id="engine", options=[{"label": f"Engine {u:03d} · {dataset.length(u)} cycles", "value": u} for u in dataset.engines],
                             value=default_unit, clearable=False, searchable=True),
                html.Div([html.Label("Observation cycle", htmlFor="cycle"), html.Span(id="cycle-caption", className="cycle-caption")], className="control-label-row"),
                dcc.Slider(id="cycle", min=1, max=default_cycle, step=1, value=default_cycle,
                           marks={1: "1", default_cycle: str(default_cycle)}, tooltip={"placement": "bottom", "always_visible": False}),
                html.P("Only observations through this cycle are used.", className="control-help"),
                html.Label("Sensors to display", htmlFor="sensors"),
                dcc.Dropdown(id="sensors", options=[{"label": f"{s} · {v[0]}", "value": s} for s, v in SENSOR_INFO.items()],
                             value=["s_11", "s_12", "s_14"], multi=True, clearable=True),
                html.P("Up to three charts, each in its native units.", className="control-help"),
                html.P("Predictions, trends and SHAP update automatically when you select an engine or cycle. Calculations may take a moment.", className="control-help")
            ], className="controls"),
            html.Div([html.Div("DATA SOURCE", className="eyebrow"), html.Strong("NASA C-MAPSS"),
                      html.P(f"FD001 · test split · {len(dataset.engines)} engines"), html.Span("Recorded observations", className="source-pill")], className="source-card"),
        ], className="sidebar"),
        html.Main([
            html.Header([
                html.Div([html.Div("CONDITION MONITORING / FD001", className="eyebrow"), html.H1("Understand engine health"),
                          html.P("Explore recorded sensor behavior and the evidence behind each prediction.", className="page-subtitle")]),
                html.Div([html.Span("Historical analysis", className="mode-badge"),
                    html.Div([html.Button("Export PNG", id="export-png", n_clicks=0, className="button secondary"),
                              html.Button("Export PDF", id="export-pdf", n_clicks=0, className="button secondary")], className="export-buttons")], className="header-actions")
            ], className="page-header"),
            html.Div(id="export-status", className="export-status", role="status"),
            html.Div([html.Div(id="selection-title", className="selection-title"), html.Span("13 retained sensors · 30-cycle model window", className="selection-detail")], className="selection-bar"),
            dcc.Loading([
                dcc.Store(id="analysis-ref"), dcc.Store(id="detail-ref"),
                html.Div(id="analysis-status", className="status-message", role="status", **{"aria-live": "polite"})
            ], target_components={"analysis-ref": "data", "detail-ref": "data"}, delay_show=300,
                custom_spinner=html.Div("Calculating predictions, trend and SHAP for your selection…",
                                        className="status-message", role="status")),
            html.Div([
                html.Section([html.Div("REMAINING USEFUL LIFE", className="eyebrow"), graph("rul-gauge", 165),
                              html.P("Cycles remaining · prediction capped at 125", className="metric-footnote")], className="metric-card gauge-card"),
                html.Section([html.Div("ANOMALY STATUS", className="eyebrow"), html.Div(id="anomaly-status", className="metric-word"),
                              html.P(id="anomaly-detail", className="metric-description"), html.Div("Autoencoder · sustained-alert policy", className="metric-footnote")], className="metric-card"),
                html.Section([html.Div("FAILURE WITHIN 100 CYCLES", className="eyebrow"), html.Div(id="failure-value", className="metric-number"),
                              html.P(id="failure-detail", className="metric-description"), html.Div("Experimental Cox estimate", className="metric-footnote")], className="metric-card"),
                html.Section([html.Div("ANALYSIS CONTEXT", className="eyebrow"), html.Div(id="context-value", className="metric-word"),
                              html.P(id="context-detail", className="metric-description"), html.Div("No live sensor connection", className="metric-footnote")], className="metric-card"),
            ], className="metric-grid"),
            html.Div([
                panel("Sensor history", "Raw readings through the selected cycle. Each sensor has its own scale.", [graph("sensor-chart", 350)], "sensor-panel"),
                panel("Anomaly trajectory", "Smoothed scores, with sustained alerts highlighted. Not a probability.", [graph("anomaly-chart", 350)]),
                panel("Remaining life & failure risk", "Up to 12 checkpoints. Cox risk is experimental; unsupported estimates remain blank.", [graph("rul-trend", 350)]),
                panel("Conditional survival", "Probability of surviving beyond a future cycle, given survival to the selection.",
                      [graph("survival-chart"), html.P(id="survival-note", className="panel-note")]),
            ], className="chart-grid"),
            html.Div([
                panel("What shaped this prediction?", "Signed SHAP contributions, summed across the 30-cycle input window.",
                      [graph("shap-chart", 365), html.Div([html.Span("Raises predicted RUL", className="legend-positive"), html.Span("Lowers predicted RUL", className="legend-negative")], className="legend-row"),
                       html.Div(id="shap-summary", className="shap-summary")], "explanation-panel"),
                panel("Alert review", "Illustrative policy applied to the selected history. No notifications are sent.",
                      [html.Div(id="alert-review"), html.Div([
                          html.H3("Read the evidence in context"),
                          html.P("RUL is capped at 125 cycles. A high estimate does not certify that an engine is safe."),
                          html.P("The Cox baseline uses the first 30 cycles only. Its test C-index is 0.576; probabilities remain experimental."),
                          html.P("Gauge bands are illustrative: below 20, 20–50, and at least 50 cycles. They are not validated maintenance thresholds.")
                      ], className="context-note")]),
            ], className="chart-grid"),
            panel("Reviewed engines", "Only engines analyzed in this browser session. Sorted by last reviewed RUL, not a fleet-wide risk score.", [html.Div(id="reviewed-engines")]),
            html.Footer([html.Span("ENGINE LAB · FD001 RESEARCH WORKSPACE"), html.A("Open live replay", href="/live/")], className="page-footer")
        ], id="workspace", className="workspace")
    ], className="app-shell")

    @app.callback(Output("cycle", "max"), Output("cycle", "value"), Output("cycle", "marks"), Input("engine", "value"))
    def engine_changed(unit):
        count = dataset.length(unit)
        return count, count, {1: "1", max(1, count // 2): str(max(1, count // 2)), count: str(count)}

    @app.callback(Output("sensors", "options"), Input("sensors", "value"))
    def sensor_options(selected):
        selected = selected or []
        return [{"label": f"{s} · {v[0]}", "value": s, "disabled": len(selected) >= 3 and s not in selected}
                for s, v in SENSOR_INFO.items()]

    @app.callback(Output("sensor-chart", "figure"), Output("selection-title", "children"), Output("cycle-caption", "children"),
                  Input("engine", "value"), Input("cycle", "value"), Input("sensors", "value"))
    def selection_changed(unit, cycle, selected):
        try:
            frame = dataset.prefix(unit, cycle)
        except ValueError:
            return figures.empty("Choose an observed cycle for this engine.", 350), "Choose an engine and cycle", ""
        return figures.sensors(frame, selected), f"Engine {unit:03d} / Cycle {cycle}", f"{cycle} of {dataset.length(unit)}"

    @app.callback(Output("analysis-ref", "data"), Input("engine", "value"), Input("cycle", "value"))
    def analyze_selection(unit, cycle):
        try:
            token = service.analyze(session["viewer"], unit, cycle)
            return {"token": token, "unit": unit, "cycle": cycle}
        except (DashboardError, ValueError) as exc:
            return {"error": str(exc), "unit": unit, "cycle": cycle}

    @app.callback(Output("detail-ref", "data"), Input("analysis-ref", "data"),
                  State("engine", "value"), State("cycle", "value"), prevent_initial_call=True)
    def add_detail(analysis_ref, unit, cycle):
        # A completed request may belong to a selection the viewer has already left.
        if not analysis_ref or not analysis_ref.get("token") or (
                analysis_ref.get("unit"), analysis_ref.get("cycle")) != (unit, cycle):
            return None
        detail_ref = {"unit": unit, "cycle": cycle, "analysis_token": analysis_ref["token"]}
        errors = []
        for action, label in (("trend", "Trend"), ("explain", "SHAP explanation")):
            try:
                token = service.enrich(session["viewer"], unit, cycle, analysis_ref, detail_ref, action)
                detail_ref["token"] = token
            except (DashboardError, ValueError) as exc:
                # Keep any successful result and still attempt the other panel.
                errors.append(f"{label} unavailable: {exc}")
        if errors:
            detail_ref["error"] = " ".join(errors)
        return detail_ref

    render_outputs = [("rul-gauge", "figure"), ("anomaly-status", "children"), ("anomaly-detail", "children"),
        ("failure-value", "children"), ("failure-detail", "children"), ("context-value", "children"), ("context-detail", "children"),
        ("anomaly-chart", "figure"), ("survival-chart", "figure"), ("rul-trend", "figure"), ("shap-chart", "figure"),
        ("shap-summary", "children"), ("analysis-status", "children"), ("survival-note", "children"),
        ("alert-review", "children"), ("reviewed-engines", "children")]

    @app.callback(*[Output(*o) for o in render_outputs], Input("analysis-ref", "data"), Input("detail-ref", "data"),
                  Input("engine", "value"), Input("cycle", "value"))
    def render_analysis(analysis_ref, detail_ref, unit, cycle):
        snapshot = service.resolve(session["viewer"], unit, cycle, analysis_ref, detail_ref)
        current_detail = detail_ref if detail_ref and detail_ref.get("analysis_token") == (analysis_ref or {}).get("token") else None
        status = "Calculating results for the selected engine and cycle. Sensor charts are available immediately."
        for ref in (analysis_ref, current_detail):
            if ref and (ref.get("unit"), ref.get("cycle")) == (unit, cycle) and ref.get("error"):
                status = ref["error"]
        anomaly_word, anomaly_detail, probability, failure_detail = "Awaiting analysis", "Waiting for the selected snapshot.", "—", "Waiting for the selected estimate."
        context_word, context_detail = f"Cycle {cycle}", "Historical observations · FD001 test split"
        survival_note, shap_summary = "Fixed early-life covariates · weak test ranking (C-index 0.576).", "Sensor contributions appear automatically after calculation."
        alert_content = html.Div("Alert conditions appear after the selected analysis completes.", className="empty-state")
        if snapshot:
            latest = snapshot["anomaly"]["latest"]
            anomaly_word = "Sustained alert" if latest["sustained_alert"] else "Alert warming up" if latest["sustained_alert"] is None else "No sustained alert"
            anomaly_detail = f"Score {latest['anomaly_score']:.3f} / threshold {snapshot['anomaly']['threshold']:.3f}"
            if latest["is_anomaly"] and not latest["sustained_alert"]:
                anomaly_detail += " · Current score exceeds threshold."
            survival = snapshot["survival"]
            if survival:
                value = survival["failure_probability_at_requested_horizon"]
                probability = f"{value:.1%}" if value is not None else "Unavailable"
                failure_detail = "Conditional on survival through the selected cycle."
                if survival["horizon_truncated"]:
                    failure_detail = f"Requested 100-cycle horizon exceeds support; curve stops at {survival['evaluated_horizon_cycles']} future cycles."
                median = survival["median_remaining_cycles"]
                survival_note = (f"Median remaining life within returned curve: {median} cycles." if median is not None else "Median not reached within the returned supported horizon.") + " Fixed early-life covariates; experimental baseline."
            context_word = f"Cycle {snapshot['cycle']}"
            context_detail = f"Analyzed {snapshot['analyzed_at'][11:19]} UTC · release {snapshot['prediction']['release_version'][:8]}"
            if not any(ref and ref.get("error") and (ref.get("unit"), ref.get("cycle")) == (unit, cycle) for ref in (analysis_ref, current_detail)):
                status = (f"Analysis ready for engine {unit:03d}, cycle {cycle}. Results use observations available at that cycle."
                          if current_detail else "Predictions ready; calculating the trend and SHAP explanation…")
            if snapshot["warnings"]:
                status += " " + " ".join(snapshot["warnings"])
            if snapshot["prediction"]["warnings"]:
                status += " Some sensor values lie outside the training range; values were not clipped."
            rows = anomaly_episodes(snapshot)
            alert_content = table(["Observed at", "Condition", "State"], rows) if rows else html.Div([
                html.Div("No alert condition in this view", className="empty-title"),
                html.P("No sustained anomaly episode or current RUL-below-20 condition was found. This is not a safety certification.")], className="empty-state")
            explanation = snapshot["explanation"]
            if explanation:
                describe = lambda name: ", ".join(f"{r['sensor']} ({r['contribution_cycles']:+.2f})" for r in explanation[name]) or "None"
                shap_summary = html.Div([html.P([html.Strong("Raises RUL: "), describe("top_positive")]),
                    html.P([html.Strong("Lowers RUL: "), describe("top_negative")]),
                    html.P(f"Baseline {explanation['base_value_cycles']:.2f} + contributions = {explanation['reconstructed_rul_cycles']:.2f} cycles. Approximation gap: {explanation['explanation_gap_cycles']:+.2f} cycles. This gap is not prediction error.", className="panel-note")])
        reviewed = service.snapshots.reviewed(session["viewer"])
        review_content = table(["Engine", "Observed cycle", "Predicted RUL", "Anomaly", "Analyzed (UTC)"], [
            [f"{s['unit']:03d}", s["cycle"], f"{s['prediction']['rul_cycles']:.1f} cycles",
             "Sustained alert" if s["anomaly"]["latest"]["sustained_alert"] else "Warming up" if s["anomaly"]["latest"]["sustained_alert"] is None else "No sustained alert",
             s["analyzed_at"][11:19]] for s in reviewed]) if reviewed else html.Div("Your reviewed engines will appear here after analysis.", className="empty-state compact")
        return (figures.gauge(snapshot), anomaly_word, anomaly_detail, probability, failure_detail, context_word, context_detail,
                figures.anomaly(snapshot), figures.survival(snapshot), figures.rul_trend(snapshot), figures.shap(snapshot), shap_summary,
                status, survival_note, alert_content, review_content)

    @app.callback(Output("report-download", "data"), Output("export-status", "children"),
                  Input("export-png", "n_clicks"), Input("export-pdf", "n_clicks"),
                  State("analysis-ref", "data"), State("detail-ref", "data"), State("engine", "value"), State("cycle", "value"), State("sensors", "value"),
                  prevent_initial_call=True, running=[(Output("export-png", "disabled"), True, False), (Output("export-pdf", "disabled"), True, False)])
    def export_selection(png, pdf, analysis_ref, detail_ref, unit, cycle, selected):
        snapshot = service.resolve(session["viewer"], unit, cycle, analysis_ref, detail_ref)
        if not snapshot:
            return no_update, "Wait for analysis of the selected engine and cycle before exporting. If it expired, reload the page."
        kind = "pdf" if ctx.triggered_id == "export-pdf" else "png"
        content = render_report(snapshot, dataset, selected, kind)
        return dcc.send_bytes(content, f"FD001_engine{unit:03d}_cycle{cycle}.{kind}"), f"Exported engine {unit:03d}, cycle {cycle} as {kind.upper()}."

    from .live import create_live_app
    app.live_app = create_live_app(server, settings, live_service)
    return app


def main():
    from waitress import serve
    settings = DashboardSettings()
    app = create_app(settings)
    print(f"Engine Lab: http://{settings.host}:{settings.port} · historical mode", flush=True)
    serve(app.server, host=settings.host, port=settings.port, threads=4, max_request_body_size=1_048_576)


if __name__ == "__main__":
    main()
