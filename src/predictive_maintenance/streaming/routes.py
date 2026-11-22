"""Authenticated ingestion and read-only dashboard feeds; no model calls on feed reads."""
from typing import Annotated, Literal
from fastapi import APIRouter, Depends, Query, Request
from .schemas import CreateRun, IngestReading, IngestResponse, RunId, UnitPath
from ..api.errors import ServiceError


def router(authorize):
    routes = APIRouter(prefix="/v1/stream", tags=["Stream and dashboard feed"], dependencies=[Depends(authorize)])

    def store(request):
        if not request.app.state.ready:
            raise ServiceError(503, "NOT_READY", "Stream storage and required models are not ready.")
        return request.app.state.stream_store

    def envelope(request, **values):
        return {**values, "request_id": request.state.request_id}

    @routes.get("/status")
    def status(request: Request):
        value = store(request)
        return envelope(request, **value.status(), max_history_cycles=value.max_history,
                        max_pending_jobs=value.max_pending, processing="separate durable worker", external_notifications=False)

    @routes.post("/runs")
    def create_run(request: Request, body: CreateRun):
        metadata = request.app.state.registry.describe()
        contract = {"window_cycles": metadata["rul"]["window_cycles"],
                    "survival_last_cycle": 30 + int(metadata["survival"]["support_since_landmark"]),
                    "max_history_cycles": store(request).max_history}
        run = store(request).create_run(body.model_dump(mode="json"), request.app.state.registry.release_version, contract)
        return envelope(request, run=run)

    @routes.get("/runs")
    def runs(request: Request, limit: Annotated[int, Query(ge=1, le=100)] = 100,
             after: Annotated[str, Query(max_length=64)] = ""):
        items = store(request).runs(limit, after)
        return envelope(request, items=items, next_after=items[-1]["run_id"] if len(items) == limit else None)

    @routes.get("/runs/{run_id}")
    def run(request: Request, run_id: RunId):
        return envelope(request, run=store(request).run(run_id))

    @routes.post("/runs/{run_id}/close")
    def close_run(request: Request, run_id: RunId):
        """Stop accepting new cycles; already accepted jobs continue to drain."""
        return envelope(request, run=store(request).close_run(run_id))

    @routes.post("/runs/{run_id}/readings", response_model=IngestResponse, status_code=202)
    def ingest(request: Request, run_id: RunId, body: IngestReading):
        """Persist one raw observation. Identical duplicates are acknowledged, conflicting repeats/gaps are rejected."""
        return envelope(request, **store(request).ingest(run_id, body.model_dump(mode="json")))

    @routes.get("/runs/{run_id}/units")
    def units(request: Request, run_id: RunId, limit: Annotated[int, Query(ge=1, le=100)] = 100,
              after_unit: Annotated[int, Query(ge=0, le=9223372036854775807)] = 0):
        items = store(request).units(run_id, limit, after_unit, request.app.state.stream_stale_seconds)
        return envelope(request, items=items, next_after_unit=items[-1]["unit_id"] if len(items) == limit else None)

    @routes.get("/runs/{run_id}/units/{unit_id}/latest")
    def latest(request: Request, run_id: RunId, unit_id: UnitPath):
        return envelope(request, **store(request).latest(run_id, unit_id, request.app.state.stream_stale_seconds))

    @routes.get("/runs/{run_id}/units/{unit_id}/readings")
    def readings(request: Request, run_id: RunId, unit_id: UnitPath,
                 after_cycle: Annotated[int, Query(ge=0, le=1000)] = 0,
                 through_cycle: Annotated[int, Query(ge=1, le=1000)] = 1000,
                 limit: Annotated[int, Query(ge=1, le=1000)] = 100):
        items = store(request).history(run_id, unit_id, through_cycle, after_cycle, limit)
        return envelope(request, items=items, next_after_cycle=items[-1]["reading"]["cycle"] if len(items) == limit else None)

    @routes.get("/runs/{run_id}/units/{unit_id}/predictions")
    def predictions(request: Request, run_id: RunId, unit_id: UnitPath,
                    after_cycle: Annotated[int, Query(ge=0, le=1000)] = 0,
                    limit: Annotated[int, Query(ge=1, le=1000)] = 100):
        items = store(request).predictions(run_id, unit_id, after_cycle, limit)
        return envelope(request, items=items, next_after_cycle=items[-1]["cycle"] if len(items) == limit else None)

    @routes.get("/runs/{run_id}/alerts")
    def alerts(request: Request, run_id: RunId, state: Literal["active", "resolved", "all"] = "active",
               after_id: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=100)] = 100):
        items = store(request).alerts(run_id, state, after_id, limit)
        return envelope(request, items=items, next_after_id=items[-1]["id"] if len(items) == limit else None)

    @routes.get("/runs/{run_id}/alert-events")
    def events(request: Request, run_id: RunId, after_id: Annotated[int, Query(ge=0)] = 0,
               limit: Annotated[int, Query(ge=1, le=100)] = 100):
        items = store(request).events(run_id, after_id, limit)
        return envelope(request, items=items, next_after_id=items[-1]["id"] if len(items) == limit else None)

    @routes.post("/runs/{run_id}/units/{unit_id}/retry")
    def retry(request: Request, run_id: RunId, unit_id: UnitPath):
        """Retry failed work after correcting its cause; policy and model release remain pinned."""
        return envelope(request, **store(request).retry_failed(run_id, unit_id))

    return routes
