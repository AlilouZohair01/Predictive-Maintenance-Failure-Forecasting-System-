FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PM_MODEL_ROOT=/models \
    PM_STREAM_DB=/state/stream.sqlite3 \
    PM_ALLOWED_HOSTS='["localhost","127.0.0.1","api"]' \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

WORKDIR /app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app && mkdir /state && chown app:app /state
COPY requirements-serving.txt requirements-streaming.txt pyproject.toml ./
RUN python -m pip install -r requirements-serving.txt -r requirements-streaming.txt
COPY src ./src
RUN python -m pip install --no-deps --no-build-isolation .
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).read()"
CMD ["python", "-m", "uvicorn", "predictive_maintenance.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers", "--no-access-log", "--timeout-graceful-shutdown", "150"]
