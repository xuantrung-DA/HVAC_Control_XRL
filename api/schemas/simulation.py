"""Simulation API contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ControllerName = Literal["random", "fixed_thermostat", "rule_based", "dqn"]
ScenarioName = Literal[
    "normal",
    "hot_day",
    "high_occupancy",
    "expensive_electricity",
    "combined_stress",
]


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller: ControllerName = "dqn"
    scenario: ScenarioName = "normal"
    seed: int = Field(default=707, ge=0, le=2_147_483_647)
    include_explanations: bool = False


class SimulationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller: ControllerName
    scenario: ScenarioName
    seed: int
    summary: dict[str, Any]
    trajectory: list[dict[str, Any]]
