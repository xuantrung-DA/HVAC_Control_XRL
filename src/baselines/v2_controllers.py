"""Frozen traditional and random controller contracts for V2 evaluation."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from src.envs.v2.observation import V2_OBSERVATION_NAMES


class V2RuleBasedController:
    name = "rule_based_v2"

    def __init__(self, config: Mapping) -> None:
        settings = config["rule_based_v2"]
        self.version = str(settings["version"])
        self.temperature_thresholds = tuple(
            float(value) for value in settings["temperature_thresholds_c"]
        )
        self.co2_thresholds = tuple(float(value) for value in settings["co2_thresholds_ppm"])
        self.humidity_thresholds = tuple(
            float(value) for value in settings["humidity_thresholds_pct"]
        )
        self.high_price = float(settings["high_price_threshold_per_kwh"])
        self.max_unoccupied = int(settings["maximum_unoccupied_action"])
        self.max_change = int(settings["maximum_action_change_per_step"])

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        del deterministic
        state = dict(zip(V2_OBSERVATION_NAMES, observation, strict=True))
        temperature = state["indoor_temperature_c"]
        co2 = state["co2_ppm"]
        occupied = state["occupancy"] > 0
        previous = int(round(state["hvac_action"]))
        action = 0
        for index, threshold in enumerate(self.temperature_thresholds, start=1):
            if temperature >= threshold:
                action = index
        if occupied:
            if co2 >= self.co2_thresholds[1]:
                action = max(action, 3)
            elif co2 >= self.co2_thresholds[0]:
                action = max(action, 2)
            humidity = state["indoor_relative_humidity_pct"]
            if humidity >= self.humidity_thresholds[1]:
                action = max(action, 3)
            elif humidity >= self.humidity_thresholds[0]:
                action = max(action, 2)
        else:
            action = min(action, self.max_unoccupied)
        urgent = (
            temperature >= self.temperature_thresholds[-1]
            or co2 >= self.co2_thresholds[-1]
            or state["indoor_relative_humidity_pct"] >= self.humidity_thresholds[0]
        )
        if state["electricity_price_per_kwh"] >= self.high_price and not urgent:
            action = max(action - 1, 0)
        return int(np.clip(action, max(0, previous - self.max_change), min(3, previous + self.max_change)))

    def reset(self) -> None:
        return None


class V2RandomController:
    name = "random_v2"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        del observation, deterministic
        return int(self.rng.integers(4))

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)


class HybridRuleBasedController:
    """Sensible-cooling baseline with the same hybrid actuator stack as DQN."""

    name = "hybrid_rule_based_v2"

    def __init__(self, config: Mapping) -> None:
        settings = config["hybrid_control"]["matched_rule_based"]
        self.version = str(settings["version"])
        self.medium = float(settings["temperature_medium_c"])
        self.high = float(settings["temperature_high_c"])
        self.maximum_unoccupied_action = int(settings["maximum_unoccupied_action"])

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        del deterministic
        temperature = float(observation[0])
        occupied = float(observation[3]) > 0.0
        thermal_risk = float(observation[25])
        if temperature >= self.high or thermal_risk >= 0.75:
            action = 3
        elif temperature >= self.medium or thermal_risk >= 0.45:
            action = 2
        elif occupied and temperature >= 23.0:
            action = 1
        else:
            action = 0
        if not occupied:
            action = min(action, self.maximum_unoccupied_action)
        return action

    def reset(self) -> None:
        return None
