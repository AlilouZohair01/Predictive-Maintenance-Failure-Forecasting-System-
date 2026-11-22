"""Run: python -m uvicorn predictive_maintenance.api.main:create_app --factory."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import json
import logging
from pathlib import Path
import time

from fastapi import Body, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.responses import PlainTextResponse

from .config import Settings
from .errors import ServiceError, register_handlers
from .middleware import BoundaryMiddleware, Metrics
from .registry import Registry
from .schemas import (SENSORS, WindowRequest, SurvivalRequest, AnomalyRequest, ErrorResponse,
                      PredictionResponse, SurvivalResponse, AnomalyResponse, ExplanationResponse)
from ..survival import UnsupportedAge
from ..streaming.store import Store, StreamError
from ..streaming.routes import router as stream_router


class WorkPool:
    """No waiting queue. Capacity remains occupied after HTTP cancellation/timeout."""
    def __init__(self, settings, metrics):
        self.settings, self.metrics = settings, metrics
        self.limits = {"inference": settings.inference_capacity, "explain": 1}
        self.active = {"inference": 0, "explain": 0}
        self.executors = {k: ThreadPoolExecutor(max_workers=n, thread_name_prefix="pm-" + k) for k, n in self.limits.items()}
        self.pending = set()
        self.accepting = True

    async def run(self, operation, function):
        kind = "explain" if operation == "explain" else "inference"
        if not self.accepting or self.active[kind] >= self.limits[kind]:
            raise ServiceError(503, "CAPACITY_EXHAUSTED", "Service is busy; retry later.")
        # Runs on the event loop: admission and counter increment have no await between them.
        self.active[kind] += 1
        loop = asyncio.get_running_loop()

        def measured():
            start = time.perf_counter()
            value = function()
            return value, (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        future = loop.run_in_executor(self.executors[kind], measured)
        self.pending.add(future)

        def completed(done):
            self.pending.discard(done)
            self.active[kind] -= 1
            self.metrics.inference_seconds[operation] += time.perf_counter() - start
            self.metrics.inference_count[operation] += 1
            if not done.cancelled():
                done.exception()  # Observe exceptions even if the HTTP client has gone away.

        future.add_done_callback(completed)
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.settings.request_timeout_seconds)
        except TimeoutError:
            raise ServiceError(503, "INFERENCE_TIMEOUT", "Inference exceeded the response time limit; execution may still be draining.") from None

    async def close(self):
        self.accepting = False
        if self.pending:
            _, pending = await asyncio.wait(self.pending, timeout=self.settings.shutdown_timeout_seconds)
            if pending:
                logging.getLogger("predictive_maintenance.api").warning("Native inference is still draining after shutdown grace period")
        for executor in self.executors.values():
            # Native calls cannot safely be killed in a thread; the process supervisor owns the hard deadline.
            executor.shutdown(wait=False, cancel_futures=True)


def example(name):
    path = Path(__file__).parent / "examples" / (name + ".json")
    return {"FD001_engine55": {"summary": "Real FD001 test engine 55 (raw sensors)", "value": json.loads(path.read_text(encoding="utf-8"))}} if path.exists() else {}


def create_app(settings=None, registry_factory=None):
    settings = settings or Settings()
    registry_factory = registry_factory or Registry
    metrics = Metrics()
    logger = logging.getLogger("predictive_maintenance.api")
    logger.setLevel(settings.log_level)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.propagate = False

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        # Loading can take tens of seconds; startup fails closed on any artifact error.
        app.state.registry = await asyncio.to_thread(registry_factory, settings.model_root)
        app.state.stream_store = await asyncio.to_thread(Store, settings.stream_db, settings.stream_max_pending, settings.max_history_rows)
        app.state.stream_stale_seconds = settings.stream_stale_seconds
        app.state.pool = WorkPool(settings, metrics)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await app.state.pool.close()

    errors = {status: {"model": ErrorResponse} for status in (400, 401, 403, 404, 405, 408, 409, 413, 422, 429, 500, 503)}
    app = FastAPI(title="FD001 Predictive Maintenance API", version=settings.app_version,
                  description="FD001 research inference, durable local stream ingestion and dashboard feeds. Models are experimental; no maintenance safety certification.",
                  lifespan=lifespan, responses=errors)
    app.state.ready = False
    app.state.metrics = metrics
    register_handlers(app)

    @app.exception_handler(StreamError)
    async def stream_error(request, exc):
        from .errors import error_response
        request.state.error_code = exc.code
        return error_response(request.state.request_id, exc.status, exc.code, exc.message)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
                       expose_headers=["X-Request-ID"])
    app.add_middleware(BoundaryMiddleware, settings=settings, metrics=metrics)

    bearer = HTTPBearer(auto_error=False, description="Required in bearer mode. Local mode allows unauthenticated loopback use.")

    async def authorize(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if settings.auth_mode == "bearer":
            supplied = request.headers.get("authorization", "")
            expected = "Bearer " + settings.api_token.get_secret_value()
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                raise ServiceError(401, "UNAUTHORIZED", "A valid bearer token is required.")

    app.include_router(stream_router(authorize))

    def registry(request):
        if not request.app.state.ready:
            raise ServiceError(503, "NOT_READY", "Required models are unavailable.")
        value = request.app.state.registry
        request.state.release_version = value.release_version
        return value

    async def inference(request, payload, operation, readings, call):
        reg = registry(request)
        raw = [[getattr(row, s) for s in SENSORS] for row in readings]
        try:
            result, duration = await app.state.pool.run(operation, lambda: call(reg, raw))
        except UnsupportedAge:
            raise ServiceError(422, "SURVIVAL_AGE_UNSUPPORTED", "Current cycle exceeds the fitted survival support.") from None
        cycle = payload.current_cycle if operation == "survival" else readings[-1].cycle
        return {**result, "unit_id": payload.unit_id, "dataset": payload.dataset, "cycle": cycle,
                "request_id": request.state.request_id, "model_version": reg.model_version(operation),
                "release_version": reg.release_version, "inference_duration_ms": duration}

    @app.get("/health/live", tags=["Health"])
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready", tags=["Health"])
    async def ready(request: Request):
        reg = registry(request)
        return {"status": "ready", "release_version": reg.release_version, "capabilities": ["predict", "survival", "anomaly", "explain"]}

    @app.get("/v1/models", dependencies=[Depends(authorize)], tags=["Models"])
    async def models(request: Request):
        return {**registry(request).describe(), "request_id": request.state.request_id, "api_version": settings.app_version}

    @app.post("/v1/predict", response_model=PredictionResponse, dependencies=[Depends(authorize)], tags=["Inference"])
    async def predict(request: Request, payload: WindowRequest = Body(openapi_examples=example("predict"))):
        """Predict capped remaining life at the last of exactly 30 consecutive cycles."""
        return await inference(request, payload, "predict", payload.window, lambda r, x: r.predict(x))

    @app.post("/v1/survival", response_model=SurvivalResponse, dependencies=[Depends(authorize)], tags=["Inference"])
    async def survival(request: Request, payload: SurvivalRequest = Body(openapi_examples=example("survival"))):
        """Use cycles 1–30 and condition on survival through current_cycle. Unsupported horizons are truncated, not extrapolated."""
        return await inference(request, payload, "survival", payload.initial_window,
                               lambda r, x: r.survival(x, payload.current_cycle, payload.horizon_cycles))

    @app.post("/v1/anomaly", response_model=AnomalyResponse, dependencies=[Depends(authorize)], tags=["Inference"])
    async def anomaly(request: Request, payload: AnomalyRequest = Body(openapi_examples=example("anomaly"))):
        """Full history from cycle 1. Score/flag unavailable before 30; sustained alert unavailable before 32. Scores are not probabilities."""
        if len(payload.history) > settings.max_history_rows:
            raise ServiceError(422, "HISTORY_TOO_LONG", "History exceeds the configured row limit.")
        result = await inference(request, payload, "anomaly", payload.history,
                                 lambda r, x: r.anomaly(x, payload.unit_id, payload.detector, payload.threshold_policy, payload.return_history))
        state = result["latest"]["is_anomaly"]
        metrics.anomaly_flags["warmup" if state is None else "flagged" if state else "normal"] += 1
        return result

    @app.post("/v1/explain", response_model=ExplanationResponse, dependencies=[Depends(authorize)], tags=["Inference"])
    async def explanation(request: Request, payload: WindowRequest = Body(openapi_examples=example("explain"))):
        """500-sample approximate GradientExplainer; signed time contributions sum to each sensor's contribution. One concurrent explanation per worker."""
        return await inference(request, payload, "explain", payload.window, lambda r, x: r.explain(x))

    @app.get("/metrics", dependencies=[Depends(authorize)], response_class=PlainTextResponse, tags=["Operations"])
    async def prometheus(request: Request):
        registry(request)
        stream = await asyncio.to_thread(app.state.stream_store.status)
        lines = [metrics.render(app.state.pool.active).rstrip()]
        for state, count in stream["jobs"].items():
            lines.append(f'pm_stream_jobs{{state="{state}"}} {count}')
        for state, count in stream["alerts"].items():
            lines.append(f'pm_stream_alerts{{state="{state}"}} {count}')
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app
