"""Gymnasium environment for explainable HVAC control experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.building import (
    BuildingSimulator,
    BuildingState,
    BuildingTransition,
    HVACAction,
)
from src.envs.reward import HVACReward, RewardBreakdown
from src.envs.scenarios import build_scenario_config


OBSERVATION_NAMES = (
    "indoor_temperature_c",
    "outdoor_temperature_c",
    "relative_humidity_pct",
    "occupancy",
    "co2_ppm",
    "electricity_price_per_kwh",
    "time_sin",
    "time_cos",
    "hvac_action",
)


class HVACEnv(gym.Env[np.ndarray, int]):
    """A configurable single-zone, 24-hour smart-building environment."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        scenario: str = "normal",
        config_overrides: Mapping[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")

        self.render_mode = render_mode
        self.scenario = scenario
        self.config = build_scenario_config(
            scenario,
            config_path=config_path,
            overrides=config_overrides,
        )
        self._validate_config()
        self.simulator = BuildingSimulator(self.config)
        self.reward_model = HVACReward(self.config)
        self.max_steps = int(self.config["simulation"]["steps_per_episode"])
        self.action_space = spaces.Discrete(len(HVACAction))
        self.observation_space = self._build_observation_space()

        self._state: BuildingState | None = None
        self._episode_done = False
        self._totals: dict[str, float | int] = {}

    @property
    def state(self) -> BuildingState:
        """Return the current state, requiring reset to have been called."""

        if self._state is None:
            raise RuntimeError("Environment must be reset before accessing state")
        return self._state

    @property
    def observation_names(self) -> tuple[str, ...]:
        return OBSERVATION_NAMES

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return the initial observation and metadata."""

        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        self._state = self.simulator.initial_state(self.np_random)
        if options:
            self._apply_reset_options(options)
        self._episode_done = False
        self._totals = {
            "reward": 0.0,
            "energy_kwh": 0.0,
            "electricity_cost": 0.0,
            "comfort_violation_steps": 0,
            "co2_violation_steps": 0,
            "switch_count": 0,
        }
        return self._observation(), self._base_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one 15-minute control interval."""

        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step")
        if self._episode_done:
            raise RuntimeError("Episode is complete; call reset before stepping again")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be one of 0, 1, 2, 3; received {action!r}")

        action_value = int(action)
        interval_hour = self._state.hour
        previous_state = self._state
        next_state, transition = self.simulator.transition(
            previous_state,
            action_value,
            self.np_random,
        )
        reward = self.reward_model.calculate(
            indoor_temperature_c=next_state.indoor_temperature_c,
            occupancy=previous_state.occupancy,
            co2_ppm=next_state.co2_ppm,
            electricity_cost=transition.electricity_cost,
            previous_action=previous_state.hvac_action,
            action=action_value,
        )
        self._state = next_state
        self._update_totals(reward, transition.energy_kwh, transition.electricity_cost, transition.switched)

        terminated = next_state.step >= self.max_steps
        truncated = False
        self._episode_done = terminated or truncated
        info = self._step_info(
            interval_hour=interval_hour,
            reward=reward,
            transition=transition,
        )
        return self._observation(), reward.reward, terminated, truncated, info

    def render(self) -> str | None:
        """Render a compact textual state for CLI demonstrations."""

        if self.render_mode != "ansi":
            return None
        state = self.state
        return (
            f"{state.hour:05.2f}h | Tin {state.indoor_temperature_c:05.2f}°C | "
            f"Tout {state.outdoor_temperature_c:05.2f}°C | "
            f"CO₂ {state.co2_ppm:06.1f} ppm | "
            f"HVAC {HVACAction(state.hvac_action).name}"
        )

    def _build_observation_space(self) -> spaces.Box:
        capacity = float(self.config["occupancy"]["capacity"])
        prices = self.config["energy"]
        maximum_price = max(
            float(prices["off_peak_price_per_kwh"]),
            float(prices["shoulder_price_per_kwh"]),
            float(prices["peak_price_per_kwh"]),
        )
        for scenario_overrides in self.config.get("scenarios", {}).values():
            occupancy = scenario_overrides.get("occupancy", {})
            energy = scenario_overrides.get("energy", {})
            capacity = max(capacity, float(occupancy.get("capacity", capacity)))
            scenario_prices = [
                float(value) for key, value in energy.items() if "price" in key
            ]
            if scenario_prices:
                maximum_price = max(maximum_price, *scenario_prices)
        low = np.array(
            [-10.0, -20.0, 0.0, 0.0, 350.0, 0.0, -1.0, -1.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [55.0, 60.0, 100.0, capacity, 5000.0, maximum_price, 1.0, 1.0, 3.0],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        state = self.state
        angle = 2.0 * np.pi * state.hour / 24.0
        observation = np.array(
            [
                state.indoor_temperature_c,
                state.outdoor_temperature_c,
                state.relative_humidity_pct,
                state.occupancy,
                state.co2_ppm,
                state.electricity_price_per_kwh,
                np.sin(angle),
                np.cos(angle),
                state.hvac_action,
            ],
            dtype=np.float32,
        )
        return observation

    def _base_info(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "step": self.state.step,
            "hour": self.state.hour,
            "state": self.state.as_dict(),
            "episode_metrics": dict(self._totals),
        }

    def _step_info(
        self,
        *,
        interval_hour: float,
        reward: RewardBreakdown,
        transition: BuildingTransition,
    ) -> dict[str, Any]:
        return {
            **self._base_info(),
            "interval_hour": interval_hour,
            "action": transition.action,
            "action_name": HVACAction(transition.action).name,
            "energy_kwh": transition.energy_kwh,
            "electricity_cost": transition.electricity_cost,
            "switched": transition.switched,
            "switch_magnitude": transition.switch_magnitude,
            "reward_components": reward.as_dict(),
            "thermal": {
                "envelope_heat_kw": transition.thermal.envelope_heat_kw,
                "occupant_heat_kw": transition.thermal.occupant_heat_kw,
                "solar_heat_kw": transition.thermal.solar_heat_kw,
                "hvac_cooling_kw": transition.thermal.hvac_cooling_kw,
                "net_heat_kw": transition.thermal.net_heat_kw,
            },
        }

    def _update_totals(
        self,
        reward: RewardBreakdown,
        energy_kwh: float,
        electricity_cost: float,
        switched: bool,
    ) -> None:
        self._totals["reward"] = float(self._totals["reward"]) + reward.reward
        self._totals["energy_kwh"] = float(self._totals["energy_kwh"]) + energy_kwh
        self._totals["electricity_cost"] = float(self._totals["electricity_cost"]) + electricity_cost
        if reward.temperature_violation_c > 0:
            self._totals["comfort_violation_steps"] = int(self._totals["comfort_violation_steps"]) + 1
        if reward.co2_violation_ppm > 0:
            self._totals["co2_violation_steps"] = int(self._totals["co2_violation_steps"]) + 1
        if switched:
            self._totals["switch_count"] = int(self._totals["switch_count"]) + 1

    def _apply_reset_options(self, options: Mapping[str, Any]) -> None:
        allowed = {
            "indoor_temperature_c",
            "co2_ppm",
            "hvac_action",
        }
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unsupported reset options: {sorted(unknown)}")
        values = self.state.as_dict()
        values.update(options)
        if not -10.0 <= float(values["indoor_temperature_c"]) <= 55.0:
            raise ValueError(
                "indoor_temperature_c reset option must be between -10 and 55"
            )
        if not 350.0 <= float(values["co2_ppm"]) <= 5000.0:
            raise ValueError("co2_ppm reset option must be between 350 and 5000")
        if int(values["hvac_action"]) not in range(len(HVACAction)):
            raise ValueError("hvac_action reset option must be one of 0, 1, 2, 3")
        self._state = BuildingState(**values)

    def _validate_config(self) -> None:
        simulation = self.config["simulation"]
        timestep = int(simulation["timestep_minutes"])
        hours = int(simulation["episode_hours"])
        expected_steps = hours * 60 // timestep
        if timestep <= 0 or hours <= 0 or hours * 60 % timestep != 0:
            raise ValueError("Episode duration must be divisible by timestep_minutes")
        if int(simulation["steps_per_episode"]) != expected_steps:
            raise ValueError(
                "steps_per_episode does not match episode_hours and timestep_minutes"
            )
        expected_actions = set(range(len(HVACAction)))
        for section, key in (
            ("hvac", "cooling_power_kw"),
            ("hvac", "electrical_power_kw"),
            ("iaq", "hvac_air_changes_per_hour"),
        ):
            actions = {int(value) for value in self.config[section][key]}
            if actions != expected_actions:
                raise ValueError(f"{section}.{key} must define actions 0 through 3")
