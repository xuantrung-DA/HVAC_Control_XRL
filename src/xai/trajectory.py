"""Step-level DQN trajectory explanations for the HVAC simulator."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.envs.building import HVACAction
from src.envs.hvac_env import HVACEnv, OBSERVATION_NAMES
from src.xai.counterfactual import DQNCounterfactualExplainer
from src.xai.feature_attribution import DQNFeatureAttributor, DQNLikePolicy


@dataclass(frozen=True)
class TrajectoryStep:
    scenario: str
    seed: int
    step: int
    timestamp: str
    hour: float
    state: dict[str, float]
    action: int
    action_name: str
    reward: float
    reward_components: dict[str, float]
    feature_attribution: dict[str, Any]
    counterfactual: dict[str, Any]
    energy_kwh: float
    electricity_cost: float
    comfort_status: str
    co2_status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_dict(self) -> dict[str, Any]:
        """Return a stable, frontend/data-frame friendly CSV representation."""

        attribution = self.feature_attribution
        counterfactual = self.counterfactual
        contributions = {
            item["feature"]: item for item in attribution["contributions"]
        }
        row: dict[str, Any] = {
            "scenario": self.scenario,
            "seed": self.seed,
            "step": self.step,
            "timestamp": self.timestamp,
            "hour": self.hour,
            **{f"state_{name}": value for name, value in self.state.items()},
            "action": self.action,
            "action_name": self.action_name,
            "reward": self.reward,
            "energy_kwh": self.energy_kwh,
            "electricity_cost": self.electricity_cost,
            "comfort_status": self.comfort_status,
            "co2_status": self.co2_status,
            "decision_margin": attribution["decision_margin"],
            "explanation": attribution["human_readable"],
            "counterfactual_found": counterfactual["found"],
            "counterfactual_action": counterfactual.get("counterfactual_action"),
            "counterfactual_distance": counterfactual.get("normalized_l1_distance"),
            "counterfactual_explanation": counterfactual["human_readable"],
        }
        for name in OBSERVATION_NAMES:
            item = contributions[name]
            row[f"importance_{name}_pct"] = item["absolute_importance_pct"]
            row[f"signed_contribution_{name}"] = item["signed_contribution"]
        return row


def explain_episode(
    policy: DQNLikePolicy,
    scenario: str,
    seed: int,
    attributor: DQNFeatureAttributor,
    counterfactual_explainer: DQNCounterfactualExplainer,
    *,
    maximum_steps: int | None = None,
) -> list[TrajectoryStep]:
    """Simulate one deterministic episode and explain every DQN decision."""

    env = HVACEnv(scenario=scenario)
    observation, _ = env.reset(seed=seed)
    records: list[TrajectoryStep] = []
    done = False
    while not done and (maximum_steps is None or len(records) < maximum_steps):
        attribution = attributor.explain(observation)
        preferred = [
            item.feature
            for item in sorted(
                attribution.contributions,
                key=lambda value: value.absolute_importance,
                reverse=True,
            )
        ]
        counterfactual = counterfactual_explainer.explain(
            observation, preferred_features=preferred
        )
        action = int(attribution.action)
        next_observation, reward, terminated, truncated, info = env.step(action)
        components = {
            key: float(value) for key, value in info["reward_components"].items()
        }
        hour = float(info["interval_hour"])
        records.append(
            TrajectoryStep(
                scenario=scenario,
                seed=seed,
                step=len(records),
                timestamp=_format_timestamp(hour),
                hour=hour,
                state={
                    name: float(observation[index])
                    for index, name in enumerate(OBSERVATION_NAMES)
                },
                action=action,
                action_name=HVACAction(action).name,
                reward=float(reward),
                reward_components=components,
                feature_attribution=attribution.as_dict(),
                counterfactual=counterfactual.as_dict(),
                energy_kwh=float(info["energy_kwh"]),
                electricity_cost=float(info["electricity_cost"]),
                comfort_status=(
                    "comfortable"
                    if components["temperature_violation_c"] <= 0.0
                    else "violation"
                ),
                co2_status=(
                    "acceptable"
                    if components["co2_violation_ppm"] <= 0.0
                    else "violation"
                ),
            )
        )
        observation = next_observation
        done = terminated or truncated
    env.close()
    return records


def summarize_trajectory(records: list[TrajectoryStep]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty trajectory")
    actions = Counter(record.action_name for record in records)
    top_features = Counter()
    completeness_errors: list[float] = []
    correlations: list[float] = []
    for record in records:
        contributions = record.feature_attribution["contributions"]
        top = max(contributions, key=lambda item: item["absolute_importance_pct"])
        top_features[top["feature"]] += 1
        faithfulness = record.feature_attribution["faithfulness"]
        completeness_errors.append(faithfulness["completeness_relative_error"])
        correlation = faithfulness["absolute_attribution_ablation_correlation"]
        if correlation is not None:
            correlations.append(correlation)
    found = sum(record.counterfactual["found"] for record in records)
    return {
        "scenario": records[0].scenario,
        "seed": records[0].seed,
        "steps": len(records),
        "total_reward": float(sum(record.reward for record in records)),
        "total_energy_kwh": float(sum(record.energy_kwh for record in records)),
        "total_electricity_cost": float(
            sum(record.electricity_cost for record in records)
        ),
        "comfort_violation_steps": sum(
            record.comfort_status == "violation" for record in records
        ),
        "co2_violation_steps": sum(
            record.co2_status == "violation" for record in records
        ),
        "action_distribution": dict(actions),
        "top_feature_counts": dict(top_features),
        "counterfactual_found_steps": found,
        "counterfactual_found_rate": found / len(records),
        "mean_completeness_relative_error": float(np.mean(completeness_errors)),
        "mean_attribution_ablation_correlation": (
            float(np.mean(correlations)) if correlations else None
        ),
    }


def representative_step(records: list[TrajectoryStep]) -> TrajectoryStep:
    """Choose a scenario-distinctive valid-counterfactual step for reporting."""

    candidates = [
        record
        for record in records
        if record.counterfactual["found"] and record.state["occupancy"] > 0
    ]
    if not candidates:
        candidates = [record for record in records if record.counterfactual["found"]]
    if not candidates:
        candidates = records
    scenario_features = {
        "normal": ("indoor_temperature_c",),
        "hot_day": ("outdoor_temperature_c",),
        "high_occupancy": ("occupancy", "co2_ppm"),
        "combined_stress": (
            "outdoor_temperature_c",
            "occupancy",
            "electricity_price_per_kwh",
        ),
    }
    targets = scenario_features.get(records[0].scenario, ("indoor_temperature_c",))

    def distinctive_importance(record: TrajectoryStep) -> float:
        return sum(
            item["absolute_importance_pct"]
            for item in record.feature_attribution["contributions"]
            if item["feature"] in targets
        )

    return max(candidates, key=distinctive_importance)


def _format_timestamp(hour: float) -> str:
    total_minutes = int(round(hour * 60.0))
    return f"Day 1 {total_minutes // 60:02d}:{total_minutes % 60:02d}"
