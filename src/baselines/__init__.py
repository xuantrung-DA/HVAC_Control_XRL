"""Traditional HVAC controllers and their small factory."""

from __future__ import annotations

from typing import Any, Mapping

from src.agents.base_agent import BaseAgent
from src.baselines.rule_based import RuleBasedController
from src.baselines.thermostat import FixedThermostat


BASELINE_NAMES = ("fixed_thermostat", "rule_based")


def create_baseline(
    name: str, config: Mapping[str, Any] | None = None
) -> BaseAgent:
    """Create a configured traditional controller by stable public name."""

    factories = {
        "fixed_thermostat": FixedThermostat,
        "rule_based": RuleBasedController,
    }
    try:
        factory = factories[name]
    except KeyError as exc:
        available = ", ".join(BASELINE_NAMES)
        raise ValueError(
            f"Unknown baseline '{name}'. Available baselines: {available}"
        ) from exc
    return factory(config=config)


__all__ = [
    "BASELINE_NAMES",
    "FixedThermostat",
    "RuleBasedController",
    "create_baseline",
]
from src.baselines.v2_controllers import V2RandomController, V2RuleBasedController

__all__ = ["V2RandomController", "V2RuleBasedController"]
