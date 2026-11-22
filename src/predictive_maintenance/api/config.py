from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PM_", env_file=".env", extra="ignore")
    model_root: Path = Path("models")
    app_version: str = "0.1.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]
    cors_origins: list[str] = []
    max_body_bytes: int = Field(1_048_576, ge=1024, le=16_777_216)
    max_history_rows: int = Field(1000, ge=30, le=1000)
    inference_capacity: int = Field(2, ge=1, le=8)
    request_timeout_seconds: float = Field(120, gt=0, le=600)
    body_timeout_seconds: float = Field(15, gt=0, le=120)
    shutdown_timeout_seconds: float = Field(130, gt=0, le=900)
    auth_mode: Literal["local", "bearer"] = "local"
    api_token: SecretStr | None = None
    stream_db: Path = Path("data/runtime/stream.sqlite3")
    stream_max_pending: int = Field(5000, ge=1, le=100000)
    stream_stale_seconds: float = Field(30, gt=0, le=86400)

    @model_validator(mode="after")
    def require_token(self):
        if self.auth_mode == "bearer" and (self.api_token is None or len(self.api_token.get_secret_value()) < 24):
            raise ValueError("Bearer mode requires PM_API_TOKEN with at least 24 characters")
        return self
