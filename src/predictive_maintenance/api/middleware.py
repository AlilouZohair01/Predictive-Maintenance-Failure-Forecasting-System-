"""Bound bodies before JSON parsing, preserve IDs, emit bounded metrics and logs."""
from collections import Counter, defaultdict
import asyncio
from fnmatch import fnmatchcase
import json
import logging
import re
import time
import uuid

from .errors import error_response

LOG = logging.getLogger("predictive_maintenance.api")
BUCKETS = (0.01, 0.05, 0.1, 0.5, 1, 5, 30, 120, float("inf"))


class Metrics:
    def __init__(self):
        self.counts = Counter()
        self.duration = defaultdict(float)
        self.buckets = Counter()
        self.inference_seconds = Counter()
        self.inference_count = Counter()
        self.anomaly_flags = Counter()

    def observe(self, route, method, status, duration):
        # No arbitrary URL, ID or caller method becomes a metric label.
        method = method if method in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"} else "OTHER"
        self.counts[(route, method, status)] += 1
        self.duration[(route, method)] += duration
        for bound in BUCKETS:
            if duration <= bound:
                self.buckets[(route, method, bound)] += 1

    def render(self, active):
        lines = ["# TYPE pm_http_requests_total counter"]
        for (route, method, status), count in sorted(self.counts.items()):
            lines.append(f'pm_http_requests_total{{route="{route}",method="{method}",status="{status}"}} {count}')
        lines.append("# TYPE pm_http_duration_seconds histogram")
        for (route, method), total in sorted(self.duration.items()):
            labels = f'route="{route}",method="{method}"'
            count = sum(c for (r, m, _), c in self.counts.items() if (r, m) == (route, method))
            lines += [f"pm_http_duration_seconds_sum{{{labels}}} {total}", f"pm_http_duration_seconds_count{{{labels}}} {count}"]
            for bound in BUCKETS:
                label = "+Inf" if bound == float("inf") else str(bound)
                lines.append(f'pm_http_duration_seconds_bucket{{{labels},le="{label}"}} {self.buckets[(route, method, bound)]}')
        for kind, count in active.items():
            lines.append(f'pm_active_jobs{{kind="{kind}"}} {count}')
        for kind, total in self.inference_seconds.items():
            lines.append(f'pm_inference_seconds_sum{{kind="{kind}"}} {total}')
            lines.append(f'pm_inference_seconds_count{{kind="{kind}"}} {self.inference_count[kind]}')
        for state, count in self.anomaly_flags.items():
            lines.append(f'pm_anomaly_latest_total{{state="{state}"}} {count}')
        return "\n".join(lines) + "\n"


class BoundaryMiddleware:
    def __init__(self, app, settings, metrics):
        self.app, self.settings, self.metrics = app, settings, metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        headers = {k.lower(): v for k, v in scope["headers"]}
        supplied = headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied) else str(uuid.uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        status = 500
        started = False

        async def wrapped_send(message):
            nonlocal status, started
            if message["type"] == "http.response.start":
                status, started = message["status"], True
                message["headers"] = [(k, v) for k, v in message["headers"] if k.lower() != b"x-request-id"] + [(b"x-request-id", request_id.encode())]
            await send(message)

        async def reject(status_code, code, message):
            state["error_code"] = code
            await error_response(request_id, status_code, code, message)(scope, receive, wrapped_send)

        try:
            host = headers.get(b"host", b"").decode("ascii", errors="ignore").split(":")[0].lower()
            if not any(fnmatchcase(host, allowed.lower()) for allowed in self.settings.allowed_hosts):
                await reject(400, "INVALID_HOST", "Host is not allowed.")
                return
            length = headers.get(b"content-length")
            if length is not None:
                try:
                    declared = int(length)
                    if declared < 0:
                        raise ValueError()
                except ValueError:
                    await reject(400, "INVALID_CONTENT_LENGTH", "Invalid content length.")
                    return
                if declared > self.settings.max_body_bytes:
                    await reject(413, "BODY_TOO_LARGE", "Request body exceeds the configured limit.")
                    return
            chunks, size = [], 0
            while True:
                remaining = self.settings.body_timeout_seconds - (time.perf_counter() - start)
                try:
                    message = await asyncio.wait_for(receive(), timeout=max(0, remaining))
                except TimeoutError:
                    await reject(408, "BODY_TIMEOUT", "Request body was not received within the time limit.")
                    return
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > self.settings.max_body_bytes:
                    await reject(413, "BODY_TOO_LARGE", "Request body exceeds the configured limit.")
                    return
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            delivered = False

            async def replay():
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}

            await self.app(scope, replay, wrapped_send)
        except Exception:
            LOG.exception("Request failed", extra={"request_id": request_id})
            if not started:
                await reject(500, "INTERNAL_ERROR", "Inference service encountered an unexpected error.")
            else:
                raise
        finally:
            duration = time.perf_counter() - start
            route = getattr(scope.get("route"), "path", "unmatched")
            self.metrics.observe(route, scope["method"], status, duration)
            LOG.info(json.dumps({"request_id": request_id, "method": scope["method"], "route": route,
                                 "status": status, "elapsed_ms": round(duration * 1000, 3),
                                 "release_version": state.get("release_version"), "error_code": state.get("error_code")}))
