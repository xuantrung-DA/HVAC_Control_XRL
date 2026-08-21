"""Controller rollout and multi-objective episode metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import log
from typing import Any, Iterable

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.hvac_env import HVACEnv
from src.evaluation.comfort import ViolationMetrics
from src.evaluation.energy import EnergyMetrics


@dataclass(frozen=True)
class EpisodeResult:
    """Machine-readable metrics for one controller, scenario, and seed."""

    controller: str
    scenario: str
    evaluation_seed: int
    training_seed: int | None
    reward: float
    energy_kwh: float
    electricity_cost: float
    comfort_violation_percent: float
    comfort_violation_hours: float
    average_temperature_deviation_c: float
    co2_violation_percent: float
    co2_violation_hours: float
    average_co2_excess_ppm: float
    hvac_switches: int
    action_0_fraction: float
    action_1_fraction: float
    action_2_fraction: float
    action_3_fraction: float
    action_entropy: float
    unique_actions: int
    min_indoor_temperature_c: float
    max_indoor_temperature_c: float
    max_co2_ppm: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_episode(
    controller: BaseAgent | None,
    *,
    controller_name: str,
    scenario: str,
    evaluation_seed: int,
    training_seed: int | None = None,
) -> EpisodeResult:
    """Run one full day using a controller, or seeded random actions for None."""

    env = HVACEnv(scenario=scenario)
    observation, _ = env.reset(seed=evaluation_seed)
    if controller is not None:
        controller.reset()
    random_generator = np.random.default_rng(evaluation_seed + 10_000)
    action_counts: Counter[int] = Counter()
    temperature_violations = 0.0
    co2_violations = 0.0
    temperatures = [env.state.indoor_temperature_c]
    co2_values = [env.state.co2_ppm]
    done = False

    while not done:
        action = (
            int(random_generator.integers(4))
            if controller is None
            else controller.predict(observation, deterministic=True)
        )
        action_counts[action] += 1
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward_components = info["reward_components"]
        temperature_violations += float(
            reward_components["temperature_violation_c"]
        )
        co2_violations += float(reward_components["co2_violation_ppm"])
        temperatures.append(env.state.indoor_temperature_c)
        co2_values.append(env.state.co2_ppm)

    metrics = info["episode_metrics"]
    total_steps = int(info["step"])
    timestep_hours = float(env.config["simulation"]["timestep_minutes"]) / 60.0
    energy = EnergyMetrics(
        total_kwh=float(metrics["energy_kwh"]),
        electricity_cost=float(metrics["electricity_cost"]),
    )
    comfort = ViolationMetrics(
        steps=int(metrics["comfort_violation_steps"]),
        total_steps=total_steps,
        timestep_hours=timestep_hours,
        cumulative_magnitude=temperature_violations,
    )
    iaq = ViolationMetrics(
        steps=int(metrics["co2_violation_steps"]),
        total_steps=total_steps,
        timestep_hours=timestep_hours,
        cumulative_magnitude=co2_violations,
    )
    fractions = [action_counts[action] / total_steps for action in range(4)]
    entropy = -sum(value * log(value) for value in fractions if value > 0) / log(4)
    env.close()

    return EpisodeResult(
        controller=controller_name,
        scenario=scenario,
        evaluation_seed=evaluation_seed,
        training_seed=training_seed,
        reward=float(metrics["reward"]),
        energy_kwh=energy.total_kwh,
        electricity_cost=energy.electricity_cost,
        comfort_violation_percent=comfort.percentage,
        comfort_violation_hours=comfort.hours,
        average_temperature_deviation_c=comfort.average_magnitude,
        co2_violation_percent=iaq.percentage,
        co2_violation_hours=iaq.hours,
        average_co2_excess_ppm=iaq.average_magnitude,
        hvac_switches=int(metrics["switch_count"]),
        action_0_fraction=fractions[0],
        action_1_fraction=fractions[1],
        action_2_fraction=fractions[2],
        action_3_fraction=fractions[3],
        action_entropy=float(entropy),
        unique_actions=sum(count > 0 for count in action_counts.values()),
        min_indoor_temperature_c=float(min(temperatures)),
        max_indoor_temperature_c=float(max(temperatures)),
        max_co2_ppm=float(max(co2_values)),
    )


def evaluate_controller(
    controller: BaseAgent | None,
    *,
    controller_name: str,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    training_seed: int | None = None,
) -> list[EpisodeResult]:
    """Evaluate a deterministic controller over every scenario/seed pair."""

    return [
        run_episode(
            controller,
            controller_name=controller_name,
            scenario=scenario,
            evaluation_seed=seed,
            training_seed=training_seed,
        )
        for scenario in scenarios
        for seed in seeds
    ]
