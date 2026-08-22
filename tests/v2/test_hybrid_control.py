from __future__ import annotations

import math
from dataclasses import replace

from src.envs.v2 import V2HybridHVACEnv


def test_hybrid_energy_accounting_includes_dehumidifier_and_fan() -> None:
    env = V2HybridHVACEnv("high_humidity_v2")
    env.reset(seed=901)
    env._state = env._state.__class__(
        **{
            **env.state.as_dict(),
            "indoor_temperature_c": 23.0,
            "indoor_relative_humidity_pct": 70.0,
        }
    )
    env._timeline = replace(
        env._timeline,
        inputs=(replace(env._timeline.inputs[0], occupancy=10), *env._timeline.inputs[1:]),
    )
    _, _, _, _, info = env.step(1)
    energy = info["transition"]["energy"]
    component_sum = sum(
        energy[name]
        for name in (
            "hvac_cooling_kwh",
            "ventilation_fan_kwh",
            "dehumidification_kwh",
            "lighting_kwh",
            "electronics_kwh",
            "base_building_kwh",
            "cleaning_equipment_kwh",
        )
    )
    assert energy["dehumidification_kwh"] == 0.625
    assert energy["ventilation_fan_kwh"] >= 0.0
    assert math.isclose(component_sum, energy["whole_building_kwh"], abs_tol=1e-9)
    assert math.isclose(
        energy["electricity_cost"],
        energy["whole_building_kwh"]
        * env._timeline.inputs[0].electricity_price_per_kwh,
        abs_tol=1e-9,
    )
    env.close()


def test_hybrid_dehumidifier_removes_moisture_without_changing_cooling_proposal() -> None:
    env = V2HybridHVACEnv("high_humidity_v2")
    env.reset(seed=902)
    env._state = env._state.__class__(
        **{
            **env.state.as_dict(),
            "indoor_temperature_c": 23.0,
            "indoor_relative_humidity_pct": 70.0,
        }
    )
    env._timeline = replace(
        env._timeline,
        inputs=(replace(env._timeline.inputs[0], occupancy=10), *env._timeline.inputs[1:]),
    )
    _, _, _, _, info = env.step(0)
    decision = info["control"]["hybrid_guard"]
    assert decision["dehumidification_fraction"] == 1.0
    assert decision["executed_cooling_action"] == 0
    assert (
        info["transition"]["air_quality"]["independent_dehumidification_kg"] > 0
    )
    env.close()


def test_hybrid_episode_is_deterministic() -> None:
    traces = []
    for _ in range(2):
        env = V2HybridHVACEnv("meeting_surge_v2")
        observation, _ = env.reset(seed=903)
        trace = []
        terminated = False
        while not terminated:
            observation, reward, terminated, _, info = env.step(1)
            trace.append((reward, *observation[:5], info["episode_metrics"]["whole_building_kwh"]))
        traces.append(trace)
        env.close()
    assert traces[0] == traces[1]
