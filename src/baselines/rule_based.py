"""Multi-signal rule-based HVAC controller baseline."""

from __future__ import annotations

from typing import Any, Mapping

from src.agents.base_agent import BaseAgent, ObservationView
from src.utils.config import load_environment_config


class RuleBasedController(BaseAgent):
    """Use temperature, occupancy, CO₂, and price to select HVAC power."""

    name = "rule_based"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        project_config = config if config is not None else load_environment_config()
        settings = project_config["baselines"]["rule_based"]
        comfort = project_config["comfort"]

        self.temperature_low_c = float(settings["temperature_low_c"])
        self.temperature_medium_c = float(settings["temperature_medium_c"])
        self.temperature_high_c = float(settings["temperature_high_c"])
        self.co2_medium_ppm = float(settings["co2_medium_ppm"])
        self.co2_high_ppm = float(settings["co2_high_ppm"])
        self.high_price_threshold = float(
            settings["high_price_threshold_per_kwh"]
        )
        self.max_unoccupied_action = int(settings["max_unoccupied_action"])
        self.max_action_change = int(settings["max_action_change_per_step"])
        self.occupancy_threshold = int(comfort["occupancy_threshold"])
        self._validate_settings()

    def _predict(
        self, observation: ObservationView, *, deterministic: bool
    ) -> int:
        del deterministic
        occupied = observation.occupancy >= self.occupancy_threshold
        temperature_action = self._temperature_action(
            observation.indoor_temperature_c
        )
        co2_action = self._co2_action(observation.co2_ppm) if occupied else 0
        target_action = max(temperature_action, co2_action)

        if not occupied:
            target_action = min(target_action, self.max_unoccupied_action)

        comfort_emergency = (
            occupied
            and observation.indoor_temperature_c >= self.temperature_high_c
        )
        iaq_emergency = occupied and observation.co2_ppm >= self.co2_high_ppm
        expensive_period = (
            observation.electricity_price_per_kwh >= self.high_price_threshold
        )
        if expensive_period and not (comfort_emergency or iaq_emergency):
            target_action = max(0, target_action - 1)

        lower = max(0, observation.hvac_action - self.max_action_change)
        upper = min(self.action_count - 1, observation.hvac_action + self.max_action_change)
        return min(max(target_action, lower), upper)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "signals": [
                "indoor_temperature_c",
                "occupancy",
                "co2_ppm",
                "electricity_price_per_kwh",
            ],
            "temperature_thresholds_c": [
                self.temperature_low_c,
                self.temperature_medium_c,
                self.temperature_high_c,
            ],
            "co2_thresholds_ppm": [self.co2_medium_ppm, self.co2_high_ppm],
        }

    def _temperature_action(self, temperature_c: float) -> int:
        if temperature_c >= self.temperature_high_c:
            return 3
        if temperature_c >= self.temperature_medium_c:
            return 2
        if temperature_c >= self.temperature_low_c:
            return 1
        return 0

    def _co2_action(self, co2_ppm: float) -> int:
        if co2_ppm >= self.co2_high_ppm:
            return 3
        if co2_ppm >= self.co2_medium_ppm:
            return 2
        return 0

    def _validate_settings(self) -> None:
        temperatures = (
            self.temperature_low_c,
            self.temperature_medium_c,
            self.temperature_high_c,
        )
        if tuple(sorted(temperatures)) != temperatures or len(set(temperatures)) != 3:
            raise ValueError("Rule-based temperature thresholds must increase")
        if self.co2_medium_ppm >= self.co2_high_ppm:
            raise ValueError("Rule-based CO2 thresholds must increase")
        if self.max_unoccupied_action not in range(self.action_count):
            raise ValueError("max_unoccupied_action must be between 0 and 3")
        if self.max_action_change not in range(1, self.action_count):
            raise ValueError("max_action_change_per_step must be between 1 and 3")
