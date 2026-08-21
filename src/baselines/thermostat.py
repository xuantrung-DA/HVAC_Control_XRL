"""Fixed multi-stage thermostat baseline."""

from __future__ import annotations

from typing import Any, Mapping

from src.agents.base_agent import BaseAgent, ObservationView
from src.utils.config import load_environment_config


class FixedThermostat(BaseAgent):
    """Map indoor temperature to HVAC stages using fixed thresholds."""

    name = "fixed_thermostat"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        project_config = config if config is not None else load_environment_config()
        settings = project_config["baselines"]["fixed_thermostat"]
        self.thresholds = tuple(
            float(value) for value in settings["temperature_thresholds_c"]
        )
        self.hysteresis_c = float(settings["hysteresis_c"])
        self._validate_settings()

    def _predict(
        self, observation: ObservationView, *, deterministic: bool
    ) -> int:
        del deterministic
        temperature = observation.indoor_temperature_c
        current_action = observation.hvac_action
        target_action = sum(temperature >= threshold for threshold in self.thresholds)

        if target_action > current_action:
            boundary = self.thresholds[target_action - 1]
            if temperature < boundary + self.hysteresis_c:
                return current_action
        elif target_action < current_action:
            boundary = self.thresholds[current_action - 1]
            if temperature > boundary - self.hysteresis_c:
                return current_action
        return target_action

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "temperature_thresholds_c": list(self.thresholds),
            "hysteresis_c": self.hysteresis_c,
        }

    def _validate_settings(self) -> None:
        if len(self.thresholds) != self.action_count - 1:
            raise ValueError("Fixed thermostat requires exactly three thresholds")
        if tuple(sorted(self.thresholds)) != self.thresholds:
            raise ValueError("Thermostat thresholds must be strictly increasing")
        if len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("Thermostat thresholds must be strictly increasing")
        if self.hysteresis_c < 0:
            raise ValueError("Thermostat hysteresis cannot be negative")
