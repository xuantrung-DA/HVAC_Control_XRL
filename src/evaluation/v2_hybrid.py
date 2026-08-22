"""Evaluation contracts for learning-augmented V2 controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from src.envs.v2.hybrid_env import V2HybridHVACEnv
from src.utils.config import PROJECT_ROOT, load_yaml


class HybridController(Protocol):
    name: str

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int: ...
    def reset(self) -> None: ...


@dataclass(frozen=True)
class HybridEvaluationResult:
    controller: str
    scenario: str
    seed: int
    reward: float
    whole_building_kwh: float
    hvac_cooling_kwh: float
    ventilation_fan_kwh: float
    dehumidification_kwh: float
    hvac_ventilation_kwh: float
    electricity_cost: float
    peak_power_kw: float
    comfort_violation_percent: float
    temperature_violation_percent: float
    humidity_violation_percent: float
    co2_violation_percent: float
    critical_safety_violations: int
    cooling_intervention_percent: float
    ventilation_intervention_percent: float
    dehumidifier_runtime_percent: float
    action_distribution: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action_distribution"] = list(self.action_distribution)
        return result


def evaluate_hybrid_controller(
    controller: HybridController,
    *,
    controller_name: str,
    scenarios: Sequence[str],
    seeds: Sequence[int],
) -> list[HybridEvaluationResult]:
    safety = load_yaml(PROJECT_ROOT / "configs/v2/controllers.yaml")["safety_metrics"]
    temperature_min, temperature_max = (
        float(value) for value in safety["critical_temperature_bounds_c"]
    )
    critical_co2 = float(safety["critical_co2_ppm"])
    results: list[HybridEvaluationResult] = []
    for scenario in scenarios:
        for seed in seeds:
            env = V2HybridHVACEnv(scenario=scenario)
            observation, _ = env.reset(seed=int(seed))
            controller.reset()
            totals = {
                "reward": 0.0,
                "whole": 0.0,
                "cooling": 0.0,
                "fan": 0.0,
                "dehumidification": 0.0,
                "controllable": 0.0,
                "cost": 0.0,
                "peak": 0.0,
                "occupied": 0,
                "comfort": 0,
                "temperature": 0,
                "humidity": 0,
                "co2": 0,
                "critical": 0,
                "cooling_intervention": 0,
                "ventilation_intervention": 0,
                "dehumidifier_runtime": 0,
                "steps": 0,
            }
            actions = np.zeros(4, dtype=np.int64)
            terminated = False
            while not terminated:
                proposed = int(controller.predict(observation, deterministic=True))
                observation, reward, terminated, _, info = env.step(proposed)
                audit = info["reward_audit"]
                transition = info["transition"]
                energy = transition["energy"]
                decision = info["control"]["hybrid_guard"]
                executed = int(decision["executed_cooling_action"])
                actions[executed] += 1
                totals["reward"] += reward
                totals["whole"] += energy["whole_building_kwh"]
                totals["cooling"] += energy["hvac_cooling_kwh"]
                totals["fan"] += energy["ventilation_fan_kwh"]
                totals["dehumidification"] += energy["dehumidification_kwh"]
                totals["controllable"] += energy["controllable_hvac_ventilation_kwh"]
                totals["cost"] += energy["electricity_cost"]
                totals["peak"] = max(totals["peak"], energy["interval_peak_power_kw"])
                occupied = bool(audit["occupied"])
                totals["occupied"] += int(occupied)
                totals["comfort"] += int(occupied and audit["comfort_violation"])
                totals["temperature"] += int(
                    occupied and audit["raw_components"]["temperature_violation_c"] > 0
                )
                totals["humidity"] += int(
                    occupied and audit["raw_components"]["humidity_violation_pct"] > 0
                )
                totals["co2"] += int(audit["co2_violation"])
                state = info["state"]
                totals["critical"] += int(
                    state["indoor_temperature_c"] < temperature_min
                    or state["indoor_temperature_c"] > temperature_max
                    or state["co2_ppm"] > critical_co2
                )
                totals["cooling_intervention"] += int(decision["cooling_intervention"])
                totals["ventilation_intervention"] += int(
                    decision["ventilation_intervention"]
                )
                totals["dehumidifier_runtime"] += int(
                    decision["dehumidification_fraction"] > 0
                )
                totals["steps"] += 1
            steps = max(int(totals["steps"]), 1)
            occupied_steps = max(int(totals["occupied"]), 1)
            results.append(
                HybridEvaluationResult(
                    controller=controller_name,
                    scenario=scenario,
                    seed=int(seed),
                    reward=float(totals["reward"]),
                    whole_building_kwh=float(totals["whole"]),
                    hvac_cooling_kwh=float(totals["cooling"]),
                    ventilation_fan_kwh=float(totals["fan"]),
                    dehumidification_kwh=float(totals["dehumidification"]),
                    hvac_ventilation_kwh=float(totals["controllable"]),
                    electricity_cost=float(totals["cost"]),
                    peak_power_kw=float(totals["peak"]),
                    comfort_violation_percent=100.0 * totals["comfort"] / occupied_steps,
                    temperature_violation_percent=100.0 * totals["temperature"] / occupied_steps,
                    humidity_violation_percent=100.0 * totals["humidity"] / occupied_steps,
                    co2_violation_percent=100.0 * totals["co2"] / steps,
                    critical_safety_violations=int(totals["critical"]),
                    cooling_intervention_percent=100.0 * totals["cooling_intervention"] / steps,
                    ventilation_intervention_percent=100.0 * totals["ventilation_intervention"] / steps,
                    dehumidifier_runtime_percent=100.0 * totals["dehumidifier_runtime"] / steps,
                    action_distribution=tuple((actions / steps).tolist()),
                )
            )
            env.close()
    return results


def aggregate_hybrid_results(results: Sequence[HybridEvaluationResult]) -> dict[str, Any]:
    fields = tuple(
        field
        for field in HybridEvaluationResult.__dataclass_fields__
        if field not in {"controller", "scenario", "seed", "action_distribution"}
    )
    return {
        "episodes": len(results),
        "metrics": {
            field: {
                "mean": float(np.mean([getattr(item, field) for item in results])),
                "std": float(np.std([getattr(item, field) for item in results])),
            }
            for field in fields
        },
        "action_distribution_mean": np.mean(
            [item.action_distribution for item in results], axis=0
        ).tolist(),
    }
