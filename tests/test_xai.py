"""Tests for faithful, bounded, deterministic DQN explanations."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from src.agents.base_agent import ObservationScaler
from src.envs.hvac_env import HVACEnv
from src.xai.counterfactual import DQNCounterfactualExplainer
from src.xai.feature_attribution import DQNFeatureAttributor
from src.xai.trajectory import explain_episode, summarize_trajectory


class TemperatureNetwork(nn.Module):
    """A transparent four-action Q-network for deterministic unit tests."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        temperature = observations[:, 0]
        return torch.stack(
            (
                torch.full_like(temperature, -0.5),
                -temperature,
                temperature,
                torch.full_like(temperature, -1.0),
            ),
            dim=1,
        )


class TransparentPolicy:
    def __init__(self, env: HVACEnv) -> None:
        self.device = torch.device("cpu")
        self.scaler = ObservationScaler(env.observation_space)
        self.online_network = TemperatureNetwork()

    def action_scores(self, observation: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(observation)
        tensor = torch.as_tensor(scaled).unsqueeze(0)
        with torch.no_grad():
            return self.online_network(tensor).squeeze(0).numpy()


@pytest.fixture
def explanation_setup() -> tuple[
    TransparentPolicy,
    np.ndarray,
    DQNFeatureAttributor,
    DQNCounterfactualExplainer,
]:
    env = HVACEnv()
    policy = TransparentPolicy(env)
    observation, _ = env.reset(seed=3)
    reference = observation.copy()
    reference[0] = 22.5
    attributor = DQNFeatureAttributor(
        policy, reference, integration_steps=32
    )
    counterfactual = DQNCounterfactualExplainer(
        policy,
        {
            "indoor_temperature_c": {
                "minimum": 16.0,
                "maximum": 35.0,
                "step": 0.25,
                "decimals": 2,
            },
            "occupancy": {
                "minimum": 0,
                "maximum": 80,
                "step": 1,
                "decimals": 0,
            },
            "co2_ppm": {
                "minimum": 350,
                "maximum": 2000,
                "step": 25,
                "decimals": 0,
            },
            "electricity_price_per_kwh": {
                "minimum": 0.05,
                "maximum": 0.70,
                "step": 0.025,
                "decimals": 3,
            },
        },
    )
    env.close()
    return policy, observation, attributor, counterfactual


def test_feature_attribution_is_deterministic_and_normalized(
    explanation_setup: tuple[
        TransparentPolicy,
        np.ndarray,
        DQNFeatureAttributor,
        DQNCounterfactualExplainer,
    ],
) -> None:
    _, observation, attributor, _ = explanation_setup
    first = attributor.explain(observation)
    second = attributor.explain(observation)

    assert first.as_dict() == second.as_dict()
    assert first.action == 2
    assert first.causal_claim is False
    assert sum(item.absolute_importance_pct for item in first.contributions) == pytest.approx(
        100.0
    )
    top = max(first.contributions, key=lambda item: item.absolute_importance_pct)
    assert top.feature == "indoor_temperature_c"
    assert top.direction == "supports_selected_action"
    assert top.absolute_importance_pct == pytest.approx(100.0)
    assert first.faithfulness["completeness_relative_error"] < 1e-5
    assert first.faithfulness["top_feature_changes_margin_when_ablated"] is True


def test_counterfactual_is_bounded_sparse_and_changes_actual_action(
    explanation_setup: tuple[
        TransparentPolicy,
        np.ndarray,
        DQNFeatureAttributor,
        DQNCounterfactualExplainer,
    ],
) -> None:
    policy, observation, _, counterfactual = explanation_setup
    result = counterfactual.explain(observation)

    assert result.found is True
    assert result.within_bounds is True
    assert result.action_changed is True
    assert result.causal_claim is False
    assert len(result.changes) == 1
    assert result.changes[0].feature == "indoor_temperature_c"
    changed = observation.copy()
    changed[0] = result.changes[0].counterfactual_value
    assert int(np.argmax(policy.action_scores(changed))) == result.counterfactual_action
    assert result.counterfactual_action != result.original_action
    assert 16.0 <= changed[0] <= 35.0


def test_counterfactual_output_is_deterministic(
    explanation_setup: tuple[
        TransparentPolicy,
        np.ndarray,
        DQNFeatureAttributor,
        DQNCounterfactualExplainer,
    ],
) -> None:
    _, observation, _, counterfactual = explanation_setup
    assert counterfactual.explain(observation).as_dict() == counterfactual.explain(
        observation
    ).as_dict()


def test_trajectory_contains_requested_explanation_fields(
    explanation_setup: tuple[
        TransparentPolicy,
        np.ndarray,
        DQNFeatureAttributor,
        DQNCounterfactualExplainer,
    ],
) -> None:
    policy, _, attributor, counterfactual = explanation_setup
    records = explain_episode(
        policy,
        "normal",
        17,
        attributor,
        counterfactual,
        maximum_steps=3,
    )
    assert len(records) == 3
    first = records[0]
    assert first.timestamp == "Day 1 00:00"
    assert first.action_name == "MEDIUM"
    assert first.energy_kwh >= 0.0
    assert first.comfort_status in {"comfortable", "violation"}
    assert first.co2_status in {"acceptable", "violation"}
    assert len(first.feature_attribution["contributions"]) == 9
    assert first.counterfactual["action_changed"] is True
    summary = summarize_trajectory(records)
    assert summary["steps"] == 3
    assert summary["counterfactual_found_rate"] == 1.0


def test_counterfactual_rejects_invalid_bounds() -> None:
    env = HVACEnv()
    policy = TransparentPolicy(env)
    with pytest.raises(ValueError, match="invalid bounds"):
        DQNCounterfactualExplainer(
            policy,
            {
                "indoor_temperature_c": {
                    "minimum": 30,
                    "maximum": 20,
                    "step": 1,
                }
            },
        )
    env.close()
