"""Configuration-driven operating scenarios for the HVAC simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.utils.config import load_environment_config


SCENARIO_DESCRIPTIONS = {
    "normal": "Typical warm workday with standard occupancy and pricing.",
    "hot_day": "Elevated outdoor temperature throughout the day.",
    "high_occupancy": "Higher building capacity and near-peak attendance.",
    "expensive_electricity": "Increased shoulder and peak electricity rates.",
    "combined_stress": "Hot weather, high occupancy, and expensive electricity.",
}


@dataclass(frozen=True)
class ScenarioDefinition:
    """Metadata and overrides for one configured scenario."""

    name: str
    description: str
    overrides: Mapping[str, Any]


def list_scenarios(
    config_path: str | Path | None = None,
) -> list[ScenarioDefinition]:
    """Return every scenario declared in the environment configuration."""

    config = load_environment_config(config_path=config_path)
    return [
        ScenarioDefinition(
            name=name,
            description=SCENARIO_DESCRIPTIONS.get(name, name.replace("_", " ").title()),
            overrides=overrides,
        )
        for name, overrides in config.get("scenarios", {}).items()
    ]


def build_scenario_config(
    scenario: str = "normal",
    *,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete environment config for a named scenario."""

    return load_environment_config(
        config_path=config_path,
        scenario=scenario,
        overrides=overrides,
    )
