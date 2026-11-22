from typing import Annotated, Literal
from pydantic import Field, model_validator
from ..api.schemas import StrictModel, SensorReading, Number

RunId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
UnitId = Annotated[int, Field(strict=True, ge=1, le=9223372036854775807)]
UnitPath = Annotated[int, Field(ge=1, le=9223372036854775807)]


class AlertPolicy(StrictModel):
    policy_version: Literal["illustrative_v1"] = "illustrative_v1"
    validation_status: Literal["illustrative_unvalidated"] = "illustrative_unvalidated"
    rul_trigger_below: Annotated[Number, Field(gt=0, le=125)] = 20.0
    rul_clear_at: Annotated[Number, Field(gt=0, le=125)] = 25.0
    anomaly_clear_cycles: Annotated[int, Field(strict=True, ge=1, le=30)] = 5
    cooldown_cycles: Annotated[int, Field(strict=True, ge=1, le=1000)] = 10

    @model_validator(mode="after")
    def hysteresis(self):
        if self.rul_clear_at <= self.rul_trigger_below:
            raise ValueError("RUL clear threshold must exceed its trigger threshold")
        return self


class CreateRun(StrictModel):
    run_id: RunId
    dataset: Literal["FD001"] = "FD001"
    source: Annotated[str, Field(min_length=1, max_length=128)] = "cmapss-test-replay"
    survival_every_cycles: Annotated[int, Field(strict=True, ge=1, le=1000)] = 10
    survival_horizon_cycles: Annotated[int, Field(strict=True, ge=1, le=332)] = 100
    alert_policy: AlertPolicy = Field(default_factory=AlertPolicy)


class IngestReading(StrictModel):
    unit_id: UnitId
    reading: SensorReading


class IngestResponse(StrictModel):
    run_id: str
    unit_id: int
    cycle: int
    disposition: Literal["accepted", "duplicate"]
    job_status: Literal["pending", "processing", "done", "failed"]
    request_id: str
