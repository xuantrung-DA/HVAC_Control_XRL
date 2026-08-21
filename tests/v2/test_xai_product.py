"""Product-layer tests for honest V2 policy and shield explanations."""

from __future__ import annotations

import numpy as np

from src.envs.v2 import V2HVACEnv
from src.services.v2_agent_service import V2AgentService
from src.xai.v2_explainer import explain_shield


def test_v2_policy_explanation_is_deterministic_and_noncausal() -> None:
    service = V2AgentService()
    env = V2HVACEnv(scenario="normal_v2", shield_enabled=True)
    observation, _ = env.reset(seed=901)
    first = service.explainer.explain(observation)
    second = service.explainer.explain(observation)
    env.close()
    assert first == second
    assert first["causal_claim"] is False
    assert len(first["contributions"]) == 35
    assert np.isclose(
        sum(item["absolute_importance_pct"] for item in first["contributions"]),
        100.0,
    )


def test_v2_counterfactual_is_replayed_when_found() -> None:
    service = V2AgentService()
    env = V2HVACEnv(scenario="high_occupancy_v2", shield_enabled=False)
    observation, _ = env.reset(seed=902)
    explanation = service.explainer.explain(observation)
    counterfactual = explanation["counterfactual"]
    if counterfactual["found"]:
        edited = observation.copy()
        change = counterfactual["changes"][0]
        edited[env.observation_names.index(change["feature"])] = change["counterfactual_value"]
        assert service.agent.predict(edited, deterministic=True) == counterfactual["counterfactual_action"]
        assert counterfactual["action_changed"] is True
        assert counterfactual["within_bounds"] is True
    env.close()


def test_shield_explanation_is_distinct_from_policy_explanation() -> None:
    payload = explain_shield({
        "proposed_action": 0,
        "executed_action": 2,
        "shield": {
            "decision": "CLAMP",
            "intervention": True,
            "constraint": "co2_risk",
            "reason": "Projected CO2 exceeds the operational threshold.",
            "projection": {"co2_ppm": 1100.0},
        },
    })
    assert payload["method"] == "deterministic_predictive_constraint_check"
    assert payload["intervention"] is True
    assert payload["proposed_action_name"] == "OFF"
    assert payload["executed_action_name"] == "MEDIUM"
    assert payload["causal_claim"] is False
