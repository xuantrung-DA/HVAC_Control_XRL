"""Contract tests for the FastAPI demo surface."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.envs.hvac_env import HVACEnv, OBSERVATION_NAMES


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def _state_payload(seed: int = 42) -> dict[str, float | int]:
    env = HVACEnv()
    observation, _ = env.reset(seed=seed)
    env.close()
    payload = {
        name: float(observation[index])
        for index, name in enumerate(OBSERVATION_NAMES)
    }
    payload["occupancy"] = int(payload["occupancy"])
    payload["hvac_action"] = int(payload["hvac_action"])
    return payload


def test_health_and_openapi(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["project"] == "XRL-HVAC"
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert schema.json()["info"]["title"] == "XRL-HVAC API"


def test_building_scenarios_and_config(client: TestClient) -> None:
    scenarios = client.get("/api/v1/building/scenarios")
    assert scenarios.status_code == 200
    assert {item["name"] for item in scenarios.json()} == {
        "normal",
        "hot_day",
        "high_occupancy",
        "expensive_electricity",
        "combined_stress",
    }
    config = client.get("/api/v1/building/config/combined_stress")
    assert config.status_code == 200
    assert config.json()["steps_per_episode"] == 96
    assert config.json()["actions"]["3"] == "HIGH"
    assert client.get("/api/v1/building/config/not_a_scenario").status_code == 422


def test_frozen_agent_metadata_and_deterministic_prediction(client: TestClient) -> None:
    metadata = client.get("/api/v1/agent/demo")
    assert metadata.status_code == 200
    assert metadata.json()["role"] == "frozen_demo_controller"
    assert metadata.json()["parameters"] == 18_308

    state = _state_payload()
    first = client.post("/api/v1/agent/predict", json=state)
    second = client.post("/api/v1/agent/predict", json=state)
    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["action"] in range(4)
    assert len(first.json()["q_values"]) == 4


def test_decision_explanation_contract(client: TestClient) -> None:
    response = client.post(
        "/api/v1/explanations/decision",
        json={"state": _state_payload(), "include_counterfactual": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["attribution"]["contributions"]) == 9
    assert payload["attribution"]["causal_claim"] is False
    assert payload["counterfactual"]["action_changed"] is True
    assert payload["counterfactual"]["within_bounds"] is True


def test_simulation_and_xai_guard(client: TestClient) -> None:
    response = client.post(
        "/api/v1/simulations/run",
        json={
            "controller": "rule_based",
            "scenario": "normal",
            "seed": 12,
            "include_explanations": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["steps"] == 96
    assert len(payload["trajectory"]) == 96
    assert payload["trajectory"][0]["timestamp"] == "Day 1 00:00"

    rejected = client.post(
        "/api/v1/simulations/run",
        json={
            "controller": "rule_based",
            "scenario": "normal",
            "include_explanations": True,
        },
    )
    assert rejected.status_code == 422
    assert "only for the frozen DQN" in rejected.json()["detail"]


def test_validation_and_artifact_errors(client: TestClient) -> None:
    invalid = _state_payload()
    invalid["hvac_action"] = 9
    response = client.post("/api/v1/agent/predict", json=invalid)
    assert response.status_code == 422
    assert client.get("/api/v1/metrics/benchmark").status_code == 200
    xai = client.get("/api/v1/metrics/xai")
    assert xai.status_code == 200
    assert xai.json()["validation"]["deterministic_replay_passed"] is True
