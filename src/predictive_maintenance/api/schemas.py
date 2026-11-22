"""Strict public inputs; raw native sensor units, never normalized client values."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, AwareDatetime, model_validator

SENSORS = [f"s_{i}" for i in (2, 3, 4, 7, 8, 11, 12, 13, 14, 15, 17, 20, 21)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SensorReading(StrictModel):
    cycle: PositiveInt
    s_2: Number
    s_3: Number
    s_4: Number
    s_7: Number
    s_8: Number
    s_11: Number
    s_12: Number
    s_13: Number
    s_14: Number
    s_15: Number
    s_17: Number
    s_20: Number
    s_21: Number
    timestamp: AwareDatetime | None = None


def consecutive(readings, start=None):
    cycles = [r.cycle for r in readings]
    if start is not None and cycles[0] != start:
        raise ValueError(f"History must start at cycle {start}")
    if any(b != a + 1 for a, b in zip(cycles, cycles[1:])):
        raise ValueError("Cycles must be ordered, unique and consecutive")


class EngineRequest(StrictModel):
    unit_id: PositiveInt
    dataset: Literal["FD001"] = "FD001"
    source: Annotated[str, Field(max_length=128)] | None = None


class WindowRequest(EngineRequest):
    window: list[SensorReading] = Field(min_length=30, max_length=30)

    @model_validator(mode="after")
    def validate_window(self):
        consecutive(self.window)
        return self


class SurvivalRequest(EngineRequest):
    initial_window: list[SensorReading] = Field(min_length=30, max_length=30)
    current_cycle: Annotated[int, Field(strict=True, ge=30)]
    horizon_cycles: PositiveInt = 100

    @model_validator(mode="after")
    def validate_window(self):
        consecutive(self.initial_window, start=1)
        return self


class AnomalyRequest(EngineRequest):
    history: list[SensorReading] = Field(min_length=1, max_length=1000)
    detector: Literal["autoencoder", "isolation_forest", "zscore"] = "autoencoder"
    threshold_policy: Literal["healthy_q98", "validation_f1"] = "healthy_q98"
    return_history: Annotated[bool, Field(strict=True)] = False

    @model_validator(mode="after")
    def validate_history(self):
        consecutive(self.history, start=1)
        return self


class ResponseBase(StrictModel):
    unit_id: int
    dataset: Literal["FD001"] = "FD001"
    cycle: int
    model_version: str
    release_version: str
    request_id: str
    inference_duration_ms: Number


class PredictionResponse(ResponseBase):
    rul_cycles: Annotated[float, Field(ge=0, le=125)]
    rul_cap_cycles: float = 125.0
    warnings: list[str]


class SurvivalPoint(StrictModel):
    future_cycles: int
    survival_probability: Annotated[float, Field(ge=0, le=1)]


class SurvivalResponse(ResponseBase):
    requested_horizon_cycles: int
    evaluated_horizon_cycles: int
    horizon_truncated: bool
    failure_probability_at_requested_horizon: Annotated[float, Field(ge=0, le=1)] | None
    median_remaining_cycles: int | None
    median_unavailable_reason: str | None
    curve: list[SurvivalPoint]
    limitations: list[str]


class AnomalyPoint(StrictModel):
    cycle: int
    raw_score: Number | None
    anomaly_score: Number | None
    is_anomaly: bool | None
    sustained_alert: bool | None
    status: Literal["warmup", "ready"]
    alert_history_ready: bool
    smoothing_samples: int


class AnomalyResponse(ResponseBase):
    detector: str
    threshold_policy: str
    threshold: Number
    latest: AnomalyPoint
    history: list[AnomalyPoint] | None


class SensorContribution(StrictModel):
    sensor: str
    contribution_cycles: Number
    raw_window_mean: Number


class ExplanationResponse(ResponseBase):
    predicted_rul_cycles: Number
    base_value_cycles: Number
    reconstructed_rul_cycles: Number
    explanation_gap_cycles: Number
    sensor_contributions: list[SensorContribution]
    top_positive: list[SensorContribution]
    top_negative: list[SensorContribution]
    method: str
    sample_budget: int
    seed: int
    background_version: str
    warnings: list[str]


class ErrorResponse(StrictModel):
    error_code: str
    message: str
    details: list[dict] = []
    request_id: str
