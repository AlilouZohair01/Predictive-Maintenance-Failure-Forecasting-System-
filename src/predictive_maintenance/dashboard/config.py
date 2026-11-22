from pathlib import Path
from typing import Literal
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PM_DASH_", env_file=".env", extra="ignore", populate_by_name=True)
    data_file: Path = Path("data/raw/cmapss/test_FD001.txt")
    api_url: str = "http://127.0.0.1:8000"
    api_token: SecretStr | None = Field(default=None, validation_alias="PM_API_TOKEN")
    host: str = "127.0.0.1"
    port: int = Field(8050, ge=1024, le=65535)
    auth_mode: Literal["local", "basic"] = "local"
    username: str | None = None
    password: SecretStr | None = None
    session_secret: SecretStr | None = None
    secure_cookie: bool = False
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]

    @model_validator(mode="after")
    def validate_access(self):
        if self.auth_mode == "basic" and (not self.username or not self.password or len(self.password.get_secret_value()) < 12):
            raise ValueError("Basic mode requires a username and password of at least 12 characters")
        if self.host not in ("127.0.0.1", "localhost") and self.auth_mode != "basic":
            raise ValueError("Non-loopback binding requires basic authentication")
        if self.session_secret and len(self.session_secret.get_secret_value()) < 32:
            raise ValueError("Session secret must contain at least 32 characters")
        return self
