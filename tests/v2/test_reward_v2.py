"""Reward profile, dynamic weighting, CMDP, and timestep audit tests."""

from __future__ import annotations

import json

import pytest

from src.envs.v2 import V2HVACEnv, V2RewardModel
from src.risk import RiskVector
from src.utils.config import PROJECT_ROOT, load_yaml
from src.utils.v2_manifest import file_sha256


def load_profile():
    path = PROJECT_ROOT / "configs/reward_profiles/reward_profile_v2_001.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def risk(thermal: float, co2: float) -> RiskVector:
    return RiskVector(
        thermal_risk=thermal,
        co2_risk=co2,
        occupancy_surge=0.0,
        forecast_uncertainty=0.1,
        forecast_error=0.0,
        energy_peak_risk=0.2,
        weather_reliability=1.0,
        occupancy_reliability=1.0,
        price_reliability=1.0,
    )


def test_profile_is_versioned_auditable_and_not_tuned_on_heldout() -> None:
    profile, path = load_profile()
    assert profile["profile_id"] == "reward_profile_v2_001"
    assert profile["adaptive_rules"]["deterministic"] is True
    assert profile["adaptive_rules"]["observable_context_only"] is True
    assert profile["selection_protocol"]["combined_stress_used_for_tuning"] is False
    assert len(file_sha256(path)) == 64


def test_dynamic_weights_shift_priority_from_safe_to_risky_context() -> None:
    profile, _ = load_profile()
    model = V2RewardModel(
        profile, load_yaml(PROJECT_ROOT / "configs/v2/environment.yaml"), "dynamic"
    )
    safe = model._weights(risk(0.0, 0.0))
    dangerous = model._weights(risk(1.0, 1.0))
    assert safe["energy"] > dangerous["energy"]
    assert dangerous["comfort"] > safe["comfort"]
    assert dangerous["co2"] > safe["co2"]


def test_fixed_mode_keeps_profile_weights_constant() -> None:
    profile, _ = load_profile()
    model = V2RewardModel(
        profile, load_yaml(PROJECT_ROOT / "configs/v2/environment.yaml"), "fixed"
    )
    assert model._weights(risk(0.0, 0.0)) == model._weights(risk(1.0, 1.0))
    assert model._weights(risk(0.5, 0.5)) == profile["base_weights"]


def test_reward_audit_reconstructs_exact_final_reward() -> None:
    env = V2HVACEnv("normal_v2", reward_mode="dynamic")
    env.reset(seed=42)
    _, reward_value, _, _, info = env.step(3)
    audit = info["reward_audit"]
    assert reward_value == pytest.approx(-sum(audit["weighted_penalties"].values()))
    assert sum(audit["priority_percent"].values()) == pytest.approx(100.0)
    assert audit["context"] == info["risk"] or set(audit["context"]) == set(info["risk"])
    assert audit["raw_components"]["energy"] > 0.0
    assert audit["raw_components"]["overcooling"] > 0.0


def test_reward_is_deterministic_for_same_seed_and_actions() -> None:
    first = V2HVACEnv("normal_v2", reward_mode="dynamic")
    second = V2HVACEnv("normal_v2", reward_mode="dynamic")
    first.reset(seed=123)
    second.reset(seed=123)
    for action in (0, 1, 2, 3, 2, 1):
        _, reward_a, _, _, info_a = first.step(action)
        _, reward_b, _, _, info_b = second.step(action)
        assert reward_a == reward_b
        assert info_a["reward_audit"] == info_b["reward_audit"]


def test_cmdp_multipliers_update_only_at_episode_boundary_and_are_bounded() -> None:
    profile, _ = load_profile()
    model = V2RewardModel(
        profile,
        load_yaml(PROJECT_ROOT / "configs/v2/environment.yaml"),
        "cmdp_lagrangian",
    )
    initial = model.lagrange.multipliers()
    model.episode_steps = 100
    model.episode_comfort_violations = 80
    model.episode_co2_violations = 50
    updated = model.end_episode()
    assert updated["comfort"] > initial["comfort"]
    assert updated["co2"] > initial["co2"]
    for _ in range(1000):
        model.lagrange.update(1.0, 1.0)
    assert model.lagrange.comfort_multiplier <= profile["cmdp"]["multiplier_max"]
    assert model.lagrange.co2_multiplier <= profile["cmdp"]["multiplier_max"]


@pytest.mark.parametrize("mode", ["fixed", "dynamic", "cmdp_lagrangian"])
def test_all_reward_modes_complete_an_episode_with_audit(mode: str) -> None:
    env = V2HVACEnv("normal_v2", reward_mode=mode)
    env.reset(seed=42)
    terminated = False
    while not terminated:
        _, reward_value, terminated, _, info = env.step(1)
        assert reward_value <= 0.0
        assert info["reward_audit"]["mode"] == mode
    assert info["step"] == 96
