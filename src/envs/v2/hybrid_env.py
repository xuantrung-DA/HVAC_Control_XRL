"""Gymnasium environment for the frozen learning-augmented V2 architecture."""

from __future__ import annotations

from typing import Any

from src.envs.v2.hvac_env import V2HVACEnv
from src.envs.v2.hybrid_physics import HybridBuildingModel
from src.shields import HybridControlDecision, HybridControlGuard
from src.utils.config import PROJECT_ROOT, load_yaml


class V2HybridHVACEnv(V2HVACEnv):
    """DQN proposes cooling; deterministic layers control IAQ and humidity."""

    simulator_version = "XRL-HVAC-v2-hybrid-001"

    def __init__(
        self,
        scenario: str = "normal_v2",
        render_mode: str | None = None,
        reward_mode: str = "dynamic",
    ) -> None:
        super().__init__(
            scenario=scenario,
            render_mode=render_mode,
            reward_mode=reward_mode,
            shield_enabled=False,
        )
        self.hybrid_config = load_yaml(PROJECT_ROOT / "configs/v2/hybrid_control.yaml")
        self.simulator = HybridBuildingModel(
            self.environment_config, self.action_config, self.hybrid_config
        )
        self.hybrid_guard = HybridControlGuard(self.hybrid_config)
        self._hybrid_decision: HybridControlDecision | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._hybrid_decision = None
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        if self._state is None or self._timeline is None:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is complete; call reset")
        if not self.action_space.contains(action):
            raise ValueError("Hybrid cooling proposal must be one of 0, 1, 2, 3")
        interval_inputs = self._timeline.inputs[self.state.step]
        proposed_action = int(action)
        previous_action = self.state.hvac_action
        decision_risk = self._risk
        self._hybrid_decision = self.hybrid_guard.decide(
            state=self.state,
            inputs=interval_inputs,
            proposed_cooling_action=proposed_action,
            risk=decision_risk,
        )
        decision = self._hybrid_decision
        next_state, transition = self.simulator.step_hybrid(
            self.state,
            cooling_action=decision.executed_cooling_action,
            cooling_fraction=decision.cooling_fraction,
            ventilation_fraction=decision.ventilation_fraction,
            dehumidification_fraction=decision.dehumidification_fraction,
            inputs=interval_inputs,
        )
        self._state = next_state
        terminated = next_state.step >= self.max_steps
        self._done = terminated
        self._totals["whole_building_kwh"] = float(
            self._totals["whole_building_kwh"]
        ) + transition.energy.whole_building_kwh
        self._totals["hvac_ventilation_kwh"] = float(
            self._totals["hvac_ventilation_kwh"]
        ) + transition.energy.controllable_hvac_ventilation_kwh
        self._totals["electricity_cost"] = float(
            self._totals["electricity_cost"]
        ) + transition.energy.electricity_cost
        self._totals["shield_interventions"] = int(
            self._totals["shield_interventions"]
        ) + int(decision.intervention)
        self._totals["dehumidification_kwh"] = float(
            self._totals.get("dehumidification_kwh", 0.0)
        ) + transition.energy.dehumidification_kwh
        self._totals["cooling_interventions"] = int(
            self._totals.get("cooling_interventions", 0)
        ) + int(decision.cooling_intervention)
        self._totals["ventilation_interventions"] = int(
            self._totals.get("ventilation_interventions", 0)
        ) + int(decision.ventilation_intervention)

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
            action=decision.executed_cooling_action,
            decision_risk=decision_risk,
        )
        reward = self._reward_audit.reward
        self._totals["reward"] = float(self._totals["reward"]) + reward
        if terminated:
            self.reward_model.end_episode()
        return self._observation(), reward, terminated, False, self._info(transition, reward)

    def _info(self, transition, reward: float) -> dict[str, Any]:
        decision = self._hybrid_decision
        return {
            "simulator_version": self.simulator_version,
            "observation_schema": "xrl_hvac_v2_obs_002",
            "scenario": self.scenario,
            "step": self.state.step,
            "reward": reward,
            "reward_status": "AUTHORIZED_AUDITABLE_V2_REWARD",
            "reward_profile_id": self.reward_profile["profile_id"],
            "reward_mode": self.reward_model.mode,
            "reward_audit": self._reward_audit.as_dict() if self._reward_audit else None,
            "control": {
                "type": "learning_augmented_hybrid",
                "proposed_action": decision.proposed_cooling_action if decision else None,
                "executed_action": decision.executed_cooling_action if decision else None,
                "hybrid_guard": decision.as_dict() if decision else None,
            },
            "state": self.state.as_dict(),
            "forecast": self._forecast.as_dict(),
            "monitoring": self._monitoring.as_dict(),
            "risk": self._risk.as_dict(),
            "transition": transition.as_dict() if transition else None,
            "episode_metrics": dict(self._totals),
        }
