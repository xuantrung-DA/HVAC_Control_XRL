"""Multi-objective evaluation for continuous V2 policies."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.envs.v2 import V2ContinuousHVACEnv


def evaluate_continuous_controller(
    controller,
    *,
    controller_name: str,
    scenarios: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    results = []
    for scenario in scenarios:
        for seed in seeds:
            env = V2ContinuousHVACEnv(scenario)
            observation, _ = env.reset(seed=int(seed))
            controller.reset()
            totals = {
                "reward": 0.0, "whole_building_kwh": 0.0,
                "hvac_ventilation_kwh": 0.0, "electricity_cost": 0.0,
                "peak_power_kw": 0.0, "comfort_steps": 0, "occupied_steps": 0,
                "co2_steps": 0, "critical_safety_violations": 0,
            }
            controls = []
            terminated = False
            steps = 0
            while not terminated:
                action = controller.predict(observation, deterministic=True)
                observation, reward, terminated, _, info = env.step(action)
                audit = info["reward_audit"]
                energy = info["transition"]["energy"]
                totals["reward"] += reward
                totals["whole_building_kwh"] += energy["whole_building_kwh"]
                totals["hvac_ventilation_kwh"] += energy["controllable_hvac_ventilation_kwh"]
                totals["electricity_cost"] += energy["electricity_cost"]
                totals["peak_power_kw"] = max(totals["peak_power_kw"], energy["interval_peak_power_kw"])
                totals["occupied_steps"] += int(audit["occupied"])
                totals["comfort_steps"] += int(audit["occupied"] and audit["comfort_violation"])
                totals["co2_steps"] += int(audit["co2_violation"])
                state = info["state"]
                totals["critical_safety_violations"] += int(
                    state["indoor_temperature_c"] < 15.0
                    or state["indoor_temperature_c"] > 35.0
                    or state["co2_ppm"] > 2000.0
                )
                controls.append(action)
                steps += 1
            control_array = np.asarray(controls)
            results.append(
                {
                    "controller": controller_name,
                    "scenario": scenario,
                    "seed": int(seed),
                    "reward": totals["reward"],
                    "whole_building_kwh": totals["whole_building_kwh"],
                    "hvac_ventilation_kwh": totals["hvac_ventilation_kwh"],
                    "electricity_cost": totals["electricity_cost"],
                    "peak_power_kw": totals["peak_power_kw"],
                    "comfort_violation_percent": 100.0 * totals["comfort_steps"] / max(totals["occupied_steps"], 1),
                    "co2_violation_percent": 100.0 * totals["co2_steps"] / steps,
                    "critical_safety_violations": totals["critical_safety_violations"],
                    "cooling_fraction_mean": float(control_array[:, 0].mean()),
                    "cooling_fraction_std": float(control_array[:, 0].std()),
                    "ventilation_fraction_mean": float(control_array[:, 1].mean()),
                    "ventilation_fraction_std": float(control_array[:, 1].std()),
                }
            )
    return results


def aggregate_continuous_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "reward", "whole_building_kwh", "hvac_ventilation_kwh", "electricity_cost",
        "peak_power_kw", "comfort_violation_percent", "co2_violation_percent",
        "critical_safety_violations", "cooling_fraction_mean", "cooling_fraction_std",
        "ventilation_fraction_mean", "ventilation_fraction_std",
    ]
    return {
        "episodes": len(results),
        "metrics": {
            field: {
                "mean": float(np.mean([item[field] for item in results])),
                "std": float(np.std([item[field] for item in results])),
            }
            for field in fields
        },
    }
