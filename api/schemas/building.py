"""Building and scenario API contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ScenarioResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class BuildingConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    timestep_minutes: int
    steps_per_episode: int
    observation_names: list[str]
    actions: dict[int, str]
    comfort: dict[str, Any]
    iaq: dict[str, Any]
    hvac: dict[str, Any]
    scenario_config: dict[str, Any]
