"""State and explainability API contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BuildingStateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indoor_temperature_c: float = Field(ge=-10.0, le=55.0)
    outdoor_temperature_c: float = Field(ge=-20.0, le=60.0)
    relative_humidity_pct: float = Field(ge=0.0, le=100.0)
    occupancy: int = Field(ge=0, le=80)
    co2_ppm: float = Field(ge=350.0, le=5000.0)
    electricity_price_per_kwh: float = Field(ge=0.0, le=0.70)
    time_sin: float = Field(ge=-1.0, le=1.0)
    time_cos: float = Field(ge=-1.0, le=1.0)
    hvac_action: int = Field(ge=0, le=3)

    def to_array(self) -> np.ndarray:
        return np.array(list(self.model_dump().values()), dtype=np.float32)


class DecisionExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: BuildingStateInput
    include_counterfactual: bool = True


class DecisionExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribution: dict[str, Any]
    counterfactual: dict[str, Any] | None
