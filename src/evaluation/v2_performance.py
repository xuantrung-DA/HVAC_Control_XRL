"""Constraint-first, multi-objective V2 controller evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from src.envs.v2 import V2HVACEnv
from src.utils.config import PROJECT_ROOT, load_yaml


class V2Controller(Protocol):
    name: str

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int: ...
    def reset(self) -> None: ...


@dataclass(frozen=True)
class V2EvaluationResult:
    controller: str
    scenario: str
    seed: int
    shield_enabled: bool
    reward: float
    whole_building_kwh: float
    hvac_ventilation_kwh: float
    electricity_cost: float
    peak_power_kw: float
    comfort_violation_percent: float
    temperature_violation_percent: float
    humidity_violation_percent: float
    comfort_degree_hours: float
    humidity_percent_hours: float
    co2_violation_percent: float
    co2_ppm_hours: float
    co2_peak_ppm: float
    switches: int
    aggressive_transitions: int
    shield_intervention_percent: float
    shield_fallback_percent: float
    critical_safety_violations: int
    action_distribution: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action_distribution"] = list(self.action_distribution)
        return result


def evaluate_v2_controller(
    controller: V2Controller,
    *,
    controller_name: str,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    shield_enabled: bool,
    reward_mode: str = "dynamic",
) -> list[V2EvaluationResult]:
    settings = load_yaml(PROJECT_ROOT / "configs/v2/controllers.yaml")["safety_metrics"]
    temperature_min, temperature_max = settings["critical_temperature_bounds_c"]
    critical_co2 = float(settings["critical_co2_ppm"])
    results = []
    for scenario in scenarios:
        for seed in seeds:
            env = V2HVACEnv(
                scenario, shield_enabled=shield_enabled, reward_mode=reward_mode
            )
            observation, _ = env.reset(seed=int(seed))
            controller.reset()
            terminated = False
            steps = 0
            reward_total = energy_total = hvac_total = cost_total = 0.0
            peak = co2_peak = 0.0
            comfort_steps = temperature_steps = humidity_steps = 0
            occupied_steps = 0
            co2_steps = switches = aggressive = critical = 0
            comfort_degree_hours = humidity_percent_hours = co2_ppm_hours = 0.0
            interventions = fallbacks = 0
            actions = np.zeros(4, dtype=np.int64)
            previous_action = 0
            while not terminated:
                proposed = int(controller.predict(observation, deterministic=True))
                observation, reward, terminated, _, info = env.step(proposed)
                audit = info["reward_audit"]
                transition = info["transition"]
                shield = info["control"]["shield"]
                executed = int(shield["executed_action"])
                actions[executed] += 1
                reward_total += reward
                energy_total += transition["energy"]["whole_building_kwh"]
                hvac_total += transition["energy"]["controllable_hvac_ventilation_kwh"]
                cost_total += transition["energy"]["electricity_cost"]
                peak = max(peak, transition["energy"]["interval_peak_power_kw"])
                co2_peak = max(co2_peak, info["state"]["co2_ppm"])
                occupied = bool(audit["occupied"])
                occupied_steps += int(occupied)
                comfort_steps += int(audit["comfort_violation"] and occupied)
                temperature_steps += int(
                    occupied
                    and audit["raw_components"]["temperature_violation_c"] > 0.0
                )
                humidity_steps += int(
                    occupied
                    and audit["raw_components"]["humidity_violation_pct"] > 0.0
                )
                co2_steps += int(audit["co2_violation"])
                if occupied:
                    comfort_degree_hours += max(-audit["comfort_margin_c"], 0.0) * 0.25
                    humidity_percent_hours += max(-audit["humidity_margin_pct"], 0.0) * 0.25
                co2_ppm_hours += max(-audit["co2_margin_ppm"], 0.0) * 0.25
                switches += int(executed != previous_action)
                aggressive += int(abs(executed - previous_action) >= 2)
                interventions += int(shield["intervention"])
                fallbacks += int(shield["decision"] == "FALLBACK")
                state = info["state"]
                critical += int(
                    state["indoor_temperature_c"] < float(temperature_min)
                    or state["indoor_temperature_c"] > float(temperature_max)
                    or state["co2_ppm"] > critical_co2
                )
                previous_action = executed
                steps += 1
            results.append(
                V2EvaluationResult(
                    controller=controller_name,
                    scenario=scenario,
                    seed=int(seed),
                    shield_enabled=shield_enabled,
                    reward=reward_total,
                    whole_building_kwh=energy_total,
                    hvac_ventilation_kwh=hvac_total,
                    electricity_cost=cost_total,
                    peak_power_kw=peak,
                    comfort_violation_percent=100.0 * comfort_steps / max(occupied_steps, 1),
                    temperature_violation_percent=100.0 * temperature_steps / max(occupied_steps, 1),
                    humidity_violation_percent=100.0 * humidity_steps / max(occupied_steps, 1),
                    comfort_degree_hours=comfort_degree_hours,
                    humidity_percent_hours=humidity_percent_hours,
                    co2_violation_percent=100.0 * co2_steps / steps,
                    co2_ppm_hours=co2_ppm_hours,
                    co2_peak_ppm=co2_peak,
                    switches=switches,
                    aggressive_transitions=aggressive,
                    shield_intervention_percent=100.0 * interventions / steps,
                    shield_fallback_percent=100.0 * fallbacks / steps,
                    critical_safety_violations=critical,
                    action_distribution=tuple((actions / steps).tolist()),
                )
            )
            env.close()
    return results


def aggregate_v2_results(results: Sequence[V2EvaluationResult]) -> dict[str, Any]:
    scalar_fields = [
        "reward", "whole_building_kwh", "hvac_ventilation_kwh", "electricity_cost",
        "peak_power_kw", "comfort_violation_percent", "temperature_violation_percent",
        "humidity_violation_percent", "comfort_degree_hours", "humidity_percent_hours",
        "co2_violation_percent", "co2_ppm_hours", "co2_peak_ppm", "switches",
        "aggressive_transitions", "shield_intervention_percent",
        "shield_fallback_percent", "critical_safety_violations",
    ]
    return {
        "episodes": len(results),
        "metrics": {
            field: {
                "mean": float(np.mean([getattr(item, field) for item in results])),
                "std": float(np.std([getattr(item, field) for item in results])),
            }
            for field in scalar_fields
        },
        "action_distribution_mean": np.mean(
            [item.action_distribution for item in results], axis=0
        ).tolist(),
    }
