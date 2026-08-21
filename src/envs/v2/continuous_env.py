"""Continuous cooling/ventilation Gymnasium environment justified by DQN evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from src.envs.v2.hvac_env import V2HVACEnv
from src.envs.v2.observation import V2_OBSERVATION_NAMES


CONTINUOUS_OBSERVATION_NAMES = V2_OBSERVATION_NAMES + (
    "cooling_command_fraction",
    "ventilation_fraction",
)


class V2ContinuousHVACEnv(V2HVACEnv):
    """Expose independently modulated cooling and ventilation in [0, 1]."""

    def __init__(self, scenario: str = "normal_v2", reward_mode: str = "dynamic") -> None:
        super().__init__(
            scenario=scenario,
            reward_mode=reward_mode,
            shield_enabled=False,
        )
        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        discrete = self.observation_space
        self.observation_space = spaces.Box(
            low=np.concatenate((discrete.low, np.zeros(2, dtype=np.float32))),
            high=np.concatenate((discrete.high, np.ones(2, dtype=np.float32))),
            dtype=np.float32,
        )

    @property
    def observation_names(self) -> tuple[str, ...]:
        return CONTINUOUS_OBSERVATION_NAMES

    def step(self, action: np.ndarray):
        if self._state is None or self._timeline is None:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is complete; call reset")
        controls = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(controls):
            raise ValueError("Continuous action must contain cooling and ventilation in [0, 1]")
        cooling_fraction = float(controls[0])
        ventilation_fraction = float(controls[1])
        interval_inputs = self._timeline.inputs[self.state.step]
        previous_action = self.state.hvac_action
        previous_controls = np.array(
            [self.state.cooling_command_fraction, self.state.ventilation_fraction]
        )
        decision_risk = self._risk
        next_state, transition = self.simulator.step_continuous(
            self.state,
            cooling_fraction,
            ventilation_fraction,
            interval_inputs,
        )
        self._state = next_state
        terminated = next_state.step >= self.max_steps
        self._done = terminated
        self._totals["whole_building_kwh"] = float(self._totals["whole_building_kwh"]) + transition.energy.whole_building_kwh
        self._totals["hvac_ventilation_kwh"] = float(self._totals["hvac_ventilation_kwh"]) + transition.energy.controllable_hvac_ventilation_kwh
        self._totals["electricity_cost"] = float(self._totals["electricity_cost"]) + transition.energy.electricity_cost
        display_step = min(next_state.step, self.max_steps - 1)
        inputs = self._timeline.inputs[display_step]
        residuals = self._residuals(display_step, inputs)
        self._monitoring = self._monitor.update(self._signal_values(inputs), residuals)
        reliability = self._reliability.update(
            residuals, anomaly_score=self._monitoring.forecast_error_score
        )
        self._forecast = self.forecaster.predict(
            inputs, display_step, planned_events=self._timeline.event_metadata
        )
        self._remember_forecast(self._forecast)
        self._risk = self.risk_analyzer.analyze(
            self.state, inputs, self._monitoring, self._forecast, reliability
        )
        change = float(np.abs(controls - previous_controls).sum() * 1.5)
        self._reward_audit = self.reward_model.calculate(
            state=next_state,
            inputs=interval_inputs,
            transition=transition,
            previous_action=previous_action,
            action=cooling_fraction * 3.0,
            decision_risk=decision_risk,
            control_change_magnitude=change,
        )
        reward = self._reward_audit.reward
        self._totals["reward"] = float(self._totals["reward"]) + reward
        if terminated:
            self.reward_model.end_episode()
        info = self._info(transition, reward)
        info["control"] = {
            "type": "continuous",
            "proposed_action": controls.tolist(),
            "executed_action": controls.tolist(),
            "cooling_fraction": cooling_fraction,
            "ventilation_fraction": ventilation_fraction,
            "shield_enabled": False,
            "shield": None,
        }
        return self._observation(), reward, terminated, False, info

    def _observation(self) -> np.ndarray:
        base = super()._observation()
        return np.concatenate(
            (
                base,
                np.array(
                    [
                        self.state.cooling_command_fraction,
                        self.state.ventilation_fraction,
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)
