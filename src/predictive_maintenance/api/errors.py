import logging
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse


class ServiceError(Exception):
    def __init__(self, status, code, message, details=None):
        self.status, self.code, self.message = status, code, message
        self.details = details or []


def error_response(request_id, status, code, message, details=None, headers=None):
    return JSONResponse(status_code=status, headers={"X-Request-ID": request_id, **(headers or {})},
                        content={"error_code": code, "message": message, "details": details or [], "request_id": request_id})


def register_handlers(app):
    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        request.state.error_code = exc.code
        return error_response(request.state.request_id, exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        request.state.error_code = "INVALID_REQUEST"
        # Do not echo rejected bodies, parser context, tokens or user values.
        details = [{"location": list(e["loc"]), "type": e["type"]} for e in exc.errors()[:20]]
        return error_response(request.state.request_id, 422, "INVALID_REQUEST", "Request does not match the endpoint contract.", details)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        codes = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 429: "RATE_LIMITED"}
        code = codes.get(exc.status_code, "HTTP_ERROR")
        request.state.error_code = code
        return error_response(request.state.request_id, exc.status_code, code, code.replace("_", " ").capitalize(), headers=exc.headers)
