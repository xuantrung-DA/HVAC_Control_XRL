"""Development-only V2 simulation surface with sealed-test enforcement."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.envs.building import HVACAction
from src.envs.v2 import V2HVACEnv, V2_OBSERVATION_NAMES
from src.services.v2_agent_service import V2AgentService
from src.xai.v2_explainer import explain_shield


DEVELOPMENT_SCENARIOS = (
    "normal_v2", "hot_day_v2", "high_occupancy_v2", "high_humidity_v2",
    "expensive_electricity_v2", "meeting_surge_v2",
    "high_electronics_load_v2", "cleaning_event_v2",
)
HELD_OUT_SCENARIOS = (
    "combined_stress_v2", "unexpected_occupancy_surge_v2", "forecast_failure_v2",
    "heatwave_v2", "door_left_open_v2",
)


class V2SimulationService:
    def __init__(self, agent_service: V2AgentService) -> None:
        self.agent_service = agent_service

    def run(self, *, scenario: str, seed: int, include_explanations: bool = True) -> dict[str, Any]:
        if scenario in HELD_OUT_SCENARIOS:
            raise ValueError("Held-out V2 scenarios are sealed because no candidate passed development gates")
        if scenario not in DEVELOPMENT_SCENARIOS:
            raise ValueError(f"Unsupported V2 development scenario: {scenario}")
        env = V2HVACEnv(scenario=scenario, shield_enabled=True)
        observation, reset_info = env.reset(seed=seed)
        records: list[dict[str, Any]] = []
        proposed_counts: Counter[str] = Counter()
        executed_counts: Counter[str] = Counter()
        done = False
        while not done:
            proposed = self.agent_service.agent.predict(observation, deterministic=True)
            policy_explanation = (
                self.agent_service.explainer.explain(observation, counterfactual=True)
                if include_explanations else None
            )
            next_observation, reward, terminated, truncated, info = env.step(proposed)
            control = info["control"]
            executed = int(control["executed_action"])
            proposed_counts[HVACAction(proposed).name] += 1
            executed_counts[HVACAction(executed).name] += 1
            transition = info["transition"]
            audit = info["reward_audit"]
            state = {
                name: float(next_observation[index])
                for index, name in enumerate(V2_OBSERVATION_NAMES)
            }
            hour = _hour_from_cycle(state["time_sin"], state["time_cos"])
            records.append({
                "step": len(records),
                "timestamp": _timestamp(hour),
                "hour": hour,
                "state": state,
                "proposed_action": proposed,
                "proposed_action_name": HVACAction(proposed).name,
                "executed_action": executed,
                "executed_action_name": HVACAction(executed).name,
                "reward": float(reward),
                "reward_audit": audit,
                "energy": transition["energy"],
                "heat_flows": transition["heat_flows"],
                "air_quality": transition["air_quality"],
                "forecast": info["forecast"],
                "monitoring": info["monitoring"],
                "risk": info["risk"],
                "control": control,
                "policy_explanation": policy_explanation,
                "shield_explanation": explain_shield(control),
                "comfort_status": "violation" if audit["comfort_violation"] else "comfortable",
                "co2_status": "violation" if audit["co2_violation"] else "acceptable",
            })
            observation = next_observation
            done = terminated or truncated
        totals = info["episode_metrics"]
        summary = {
            **totals,
            "steps": len(records),
            "scenario": scenario,
            "seed": seed,
            "proposed_action_distribution": dict(proposed_counts),
            "executed_action_distribution": dict(executed_counts),
            "development_status": "FAIL",
            "held_out_used": False,
        }
        env.close()
        return {
            "controller": "v2_dqn_experimental",
            "scenario": scenario,
            "seed": seed,
            "status": "DEVELOPMENT_FAIL",
            "disclaimer": "Experimental V2 visualization only; V1 frozen DQN remains the official demo controller.",
            "summary": summary,
            "trajectory": records,
        }


def _hour_from_cycle(sine: float, cosine: float) -> float:
    import math
    return (math.atan2(sine, cosine) % (2.0 * math.pi)) * 24.0 / (2.0 * math.pi)


def _timestamp(hour: float) -> str:
    minutes = int(round(hour * 60)) % (24 * 60)
    return f"Day 1 {minutes // 60:02d}:{minutes % 60:02d}"
