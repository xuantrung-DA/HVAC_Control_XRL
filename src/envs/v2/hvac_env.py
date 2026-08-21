"""Gymnasium integration for the proactive XRL-HVAC V2 stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.building import HVACAction
from src.envs.v2.models import V2BuildingState
from src.envs.v2.observation import V2_OBSERVATION_NAMES, build_v2_observation, v2_observation_space
from src.envs.v2.physics import TwoR1CBuildingModel
from src.envs.v2.profiles import ScenarioTimeline, V2ScenarioGenerator
from src.envs.v2.reward import V2RewardModel
from src.forecasting import FORECAST_FEATURES, ForecastBundle, SeasonalProfileForecaster
from src.risk import ForecastReliabilityTracker, ObservableRiskAnalyzer, OnlineSignalMonitor
from src.shields import PredictiveSafetyShield, ShieldDecision, ShieldDecisionType
from src.utils.config import PROJECT_ROOT, load_yaml


class V2HVACEnv(gym.Env[np.ndarray, int]):
    """One-day V2 environment with auditable reward and proactive context."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        scenario: str = "normal_v2",
        render_mode: str | None = None,
        reward_mode: str = "dynamic",
        shield_enabled: bool | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.scenario = scenario
        self.render_mode = render_mode
        self.environment_config = load_yaml(PROJECT_ROOT / "configs/v2/environment.yaml")
        self.forecast_config = load_yaml(PROJECT_ROOT / "configs/v2/forecasting.yaml")
        self.risk_config = load_yaml(PROJECT_ROOT / "configs/v2/risk.yaml")
        self.action_config = load_yaml(PROJECT_ROOT / "configs/v2/action_mapping.yaml")
        self.shield_config = load_yaml(PROJECT_ROOT / "configs/v2/shield.yaml")
        self.reward_profile = json.loads(
            (PROJECT_ROOT / "configs/reward_profiles/reward_profile_v2_001.json").read_text(
                encoding="utf-8"
            )
        )
        self.generator = V2ScenarioGenerator.from_project()
        self.simulator = TwoR1CBuildingModel(self.environment_config, self.action_config)
        self.forecaster = SeasonalProfileForecaster(self.forecast_config)
        model_path = PROJECT_ROOT / "outputs/v2/forecasting/forecast_model.json"
        self.forecaster.load_model_state(json.loads(model_path.read_text(encoding="utf-8")))
        self.risk_analyzer = ObservableRiskAnalyzer(self.risk_config, self.environment_config)
        self.reward_model = V2RewardModel(
            self.reward_profile, self.environment_config, mode=reward_mode
        )
        self.shield_enabled = (
            bool(self.shield_config["shield"]["enabled_by_default"])
            if shield_enabled is None
            else bool(shield_enabled)
        )
        self.shield = PredictiveSafetyShield(
            self.shield_config, self.environment_config, self.action_config
        )
        self.action_space = spaces.Discrete(len(HVACAction))
        self.observation_space = v2_observation_space()
        self.max_steps = int(self.environment_config["simulation"]["steps_per_episode"])
        self._state: V2BuildingState | None = None
        self._timeline: ScenarioTimeline | None = None
        self._monitor: OnlineSignalMonitor | None = None
        self._reliability: ForecastReliabilityTracker | None = None
        self._forecast: ForecastBundle | None = None
        self._monitoring = None
        self._risk = None
        self._reward_audit = None
        self._shield_decision: ShieldDecision | None = None
        self._issued_forecasts: dict[int, Any] = {}
        self._done = False
        self._totals: dict[str, float | int] = {}

    @property
    def observation_names(self) -> tuple[str, ...]:
        return V2_OBSERVATION_NAMES

    @property
    def state(self) -> V2BuildingState:
        if self._state is None:
            raise RuntimeError("Environment must be reset first")
        return self._state

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options:
            unknown = set(options) - {"scenario"}
            if unknown:
                raise ValueError(f"Unsupported V2 reset options: {sorted(unknown)}")
            self.scenario = str(options.get("scenario", self.scenario))
        episode_seed = 42 if seed is None else int(seed)
        self._timeline = self.generator.generate(self.scenario, episode_seed)
        self._state = self.simulator.initial_state()
        self._monitor = OnlineSignalMonitor(self.risk_config)
        self._reliability = ForecastReliabilityTracker(self.risk_config)
        self._issued_forecasts = {}
        self.reward_model.reset_episode()
        self._reward_audit = None
        self._shield_decision = None
        self._done = False
        self._totals = {
            "reward": 0.0, "whole_building_kwh": 0.0,
            "hvac_ventilation_kwh": 0.0, "electricity_cost": 0.0,
            "shield_interventions": 0, "shield_fallbacks": 0,
        }
        inputs = self._timeline.inputs[0]
        self._monitoring = self._monitor.update(self._signal_values(inputs))
        reliability = {"weather": 1.0, "occupancy": 1.0, "price": 1.0}
        self._forecast = self.forecaster.predict(
            inputs, 0, planned_events=self._timeline.event_metadata
        )
        self._remember_forecast(self._forecast)
        self._risk = self.risk_analyzer.analyze(
            self.state, inputs, self._monitoring, self._forecast, reliability
        )
        return self._observation(), self._info(None, reward=0.0)

    def step(self, action: int):
        if self._state is None or self._timeline is None:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is complete; call reset")
        if not self.action_space.contains(action):
            raise ValueError("V2 action must be one of 0, 1, 2, 3")
        interval_inputs = self._timeline.inputs[self.state.step]
        proposed_action = int(action)
        previous_action = self.state.hvac_action
        decision_risk = self._risk
        if self.shield_enabled:
            self._shield_decision = self.shield.decide(
                state=self.state,
                inputs=interval_inputs,
                proposed_action=proposed_action,
                forecast=self._forecast,
                risk=decision_risk,
            )
        else:
            self._shield_decision = ShieldDecision(
                decision=ShieldDecisionType.ALLOW,
                proposed_action=proposed_action,
                executed_action=proposed_action,
                intervention=False,
                constraint=None,
                reason="Shield disabled for controlled ablation.",
                risk=decision_risk.as_dict(),
                projection=None,
            )
        executed_action = self._shield_decision.executed_action
        next_state, transition = self.simulator.step(
            self.state, executed_action, interval_inputs
        )
        self._state = next_state
        terminated = next_state.step >= self.max_steps
        self._done = terminated
        self._totals["whole_building_kwh"] = float(self._totals["whole_building_kwh"]) + transition.energy.whole_building_kwh
        self._totals["hvac_ventilation_kwh"] = float(self._totals["hvac_ventilation_kwh"]) + transition.energy.controllable_hvac_ventilation_kwh
        self._totals["electricity_cost"] = float(self._totals["electricity_cost"]) + transition.energy.electricity_cost
        self._totals["shield_interventions"] = int(self._totals["shield_interventions"]) + int(self._shield_decision.intervention)
        self._totals["shield_fallbacks"] = int(self._totals["shield_fallbacks"]) + int(self._shield_decision.decision is ShieldDecisionType.FALLBACK)
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
        self._reward_audit = self.reward_model.calculate(
            state=next_state,
            inputs=interval_inputs,
            transition=transition,
            previous_action=previous_action,
            action=executed_action,
            decision_risk=decision_risk,
        )
        reward = self._reward_audit.reward
        self._totals["reward"] = float(self._totals["reward"]) + reward
        if terminated:
            self.reward_model.end_episode()
        return self._observation(), reward, terminated, False, self._info(transition, reward)

    def _observation(self) -> np.ndarray:
        step = min(self.state.step, self.max_steps - 1)
        return build_v2_observation(
            self.state, self._timeline.inputs[step], self._forecast,
            self._monitoring, self._risk,
        )

    def _signal_values(self, inputs) -> dict[str, float]:
        return {
            "occupancy": inputs.occupancy,
            "indoor_temperature_c": self.state.indoor_temperature_c,
            "co2_ppm": self.state.co2_ppm,
        }

    def _remember_forecast(self, bundle: ForecastBundle) -> None:
        for forecast in bundle.forecasts:
            target = bundle.issued_at_step + forecast.horizon_steps
            if target < self.max_steps:
                self._issued_forecasts[target] = forecast

    def _residuals(self, step: int, inputs) -> dict[str, tuple[float, float]]:
        issued = self._issued_forecasts.get(step)
        if issued is None:
            return {}
        return {
            feature: (
                float(getattr(inputs, feature)) - issued.values[feature].point,
                issued.values[feature].standard_deviation,
            )
            for feature in FORECAST_FEATURES
        }

    def _info(self, transition, reward: float) -> dict[str, Any]:
        return {
            "simulator_version": "XRL-HVAC-v2",
            "observation_schema": "xrl_hvac_v2_obs_001",
            "scenario": self.scenario,
            "step": self.state.step,
            "reward": reward,
            "reward_status": "AUTHORIZED_AUDITABLE_V2_REWARD",
            "reward_profile_id": self.reward_profile["profile_id"],
            "reward_mode": self.reward_model.mode,
            "reward_audit": self._reward_audit.as_dict() if self._reward_audit else None,
            "control": {
                "shield_enabled": self.shield_enabled,
                "proposed_action": self._shield_decision.proposed_action
                if self._shield_decision
                else None,
                "executed_action": self._shield_decision.executed_action
                if self._shield_decision
                else None,
                "shield": self._shield_decision.as_dict()
                if self._shield_decision
                else None,
            },
            "state": self.state.as_dict(),
            "forecast": self._forecast.as_dict(),
            "monitoring": self._monitoring.as_dict(),
            "risk": self._risk.as_dict(),
            "transition": transition.as_dict() if transition else None,
            "episode_metrics": dict(self._totals),
        }

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        inputs = self._timeline.inputs[min(self.state.step, self.max_steps - 1)]
        return (
            f"{inputs.hour:05.2f}h | Tin {self.state.indoor_temperature_c:05.2f}C | "
            f"Tout {inputs.outdoor_temperature_c:05.2f}C | CO2 {self.state.co2_ppm:06.1f} ppm | "
            f"HVAC {HVACAction(self.state.hvac_action).name}"
        )
