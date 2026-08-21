"""Application service for deterministic full-day HVAC simulations."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.baselines import create_baseline
from src.envs.building import HVACAction
from src.envs.hvac_env import HVACEnv, OBSERVATION_NAMES
from src.services.agent_service import AgentService
from src.services.explanation_service import ExplanationService
from src.xai.trajectory import explain_episode, summarize_trajectory


class SimulationService:
    def __init__(self, agent_service: AgentService, explanation_service: ExplanationService) -> None:
        self.agent_service = agent_service
        self.explanation_service = explanation_service

    def run(self, *, controller_name: str, scenario: str, seed: int, include_explanations: bool) -> dict[str, Any]:
        if include_explanations and controller_name != "dqn":
            raise ValueError("Step-level XAI is available only for the frozen DQN")
        if include_explanations:
            records = explain_episode(
                self.agent_service.agent,
                scenario,
                seed,
                self.explanation_service.attributor,
                self.explanation_service.counterfactual,
            )
            return {
                "controller": "dqn",
                "scenario": scenario,
                "seed": seed,
                "summary": summarize_trajectory(records),
                "trajectory": [record.as_dict() for record in records],
            }
        return self._run_plain(controller_name, scenario, seed)

    def _run_plain(self, controller_name: str, scenario: str, seed: int) -> dict[str, Any]:
        env = HVACEnv(scenario=scenario)
        controller = self._controller(controller_name, env)
        observation, _ = env.reset(seed=seed)
        if controller is not None:
            controller.reset()
        rng = np.random.default_rng(seed + 10_000)
        records: list[dict[str, Any]] = []
        actions: Counter[str] = Counter()
        temperatures = [float(observation[0])]
        co2_values = [float(observation[4])]
        done = False
        while not done:
            state = {name: float(observation[index]) for index, name in enumerate(OBSERVATION_NAMES)}
            action = int(rng.integers(4)) if controller is None else controller.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            components = info["reward_components"]
            actions[HVACAction(action).name] += 1
            temperatures.append(float(observation[0]))
            co2_values.append(float(observation[4]))
            records.append(
                {
                    "step": len(records),
                    "timestamp": _timestamp(float(info["interval_hour"])),
                    "hour": float(info["interval_hour"]),
                    "state": state,
                    "action": action,
                    "action_name": HVACAction(action).name,
                    "reward": float(reward),
                    "reward_components": components,
                    "energy_kwh": float(info["energy_kwh"]),
                    "electricity_cost": float(info["electricity_cost"]),
                    "comfort_status": "comfortable" if components["temperature_violation_c"] <= 0 else "violation",
                    "co2_status": "acceptable" if components["co2_violation_ppm"] <= 0 else "violation",
                }
            )
            done = terminated or truncated
        metrics = info["episode_metrics"]
        steps = len(records)
        summary = {
            "scenario": scenario,
            "seed": seed,
            "steps": steps,
            "total_reward": float(metrics["reward"]),
            "total_energy_kwh": float(metrics["energy_kwh"]),
            "total_electricity_cost": float(metrics["electricity_cost"]),
            "comfort_violation_steps": int(metrics["comfort_violation_steps"]),
            "comfort_violation_percent": 100.0 * int(metrics["comfort_violation_steps"]) / steps,
            "co2_violation_steps": int(metrics["co2_violation_steps"]),
            "co2_violation_percent": 100.0 * int(metrics["co2_violation_steps"]) / steps,
            "hvac_switches": int(metrics["switch_count"]),
            "action_distribution": dict(actions),
            "min_indoor_temperature_c": min(temperatures),
            "max_indoor_temperature_c": max(temperatures),
            "max_co2_ppm": max(co2_values),
        }
        env.close()
        return {"controller": controller_name, "scenario": scenario, "seed": seed, "summary": summary, "trajectory": records}

    def _controller(self, name: str, env: HVACEnv) -> BaseAgent | None:
        if name == "random":
            return None
        if name == "dqn":
            return self.agent_service.agent
        if name in {"fixed_thermostat", "rule_based"}:
            return create_baseline(name, config=env.config)
        raise ValueError(f"Unsupported controller: {name}")


def _timestamp(hour: float) -> str:
    minutes = int(round(hour * 60))
    return f"Day 1 {minutes // 60:02d}:{minutes % 60:02d}"
