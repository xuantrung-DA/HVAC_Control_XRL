"""Gymnasium V2 environment and frozen-V1 compatibility tests."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from src.agents.dqn import DQNAgent
from src.envs.hvac_env import HVACEnv
from src.envs.v2 import V1AgentOnV2Adapter, V1ObservationAdapter, V2HVACEnv


def test_v2_environment_passes_gymnasium_checker() -> None:
    check_env(V2HVACEnv("normal_v2"), skip_render_check=True)


def test_v2_episode_is_seeded_and_completes() -> None:
    first = V2HVACEnv("hot_day_v2")
    second = V2HVACEnv("hot_day_v2")
    obs1, info1 = first.reset(seed=123)
    obs2, info2 = second.reset(seed=123)
    assert np.array_equal(obs1, obs2)
    assert first.observation_space.contains(obs1)
    terminated = False
    steps = 0
    while not terminated:
        obs1, reward, terminated, truncated, info1 = first.step(steps % 4)
        assert first.observation_space.contains(obs1)
        assert np.isfinite(reward)
        assert not truncated
        steps += 1
    assert steps == 96
    assert info1["reward_status"] == "AUTHORIZED_AUDITABLE_V2_REWARD"
    assert info1["reward_audit"]["profile_id"] == "reward_profile_v2_001"
    assert info1["episode_metrics"]["whole_building_kwh"] > 0.0


def test_observation_schema_is_named_and_compact() -> None:
    env = V2HVACEnv()
    observation, _ = env.reset(seed=42)
    assert observation.shape == env.observation_space.shape
    assert len(env.observation_names) == observation.size == 35
    assert len(set(env.observation_names)) == 35


def test_frozen_v1_dqn_runs_via_adapter_without_checkpoint_change() -> None:
    v2_env = V2HVACEnv("normal_v2")
    v2_observation, _ = v2_env.reset(seed=42)
    v1_env = HVACEnv(scenario="normal")
    dqn = DQNAgent(v1_env.observation_space, v1_env.action_space)
    dqn.load("models/dqn/demo_best.pt")
    adapter = V1AgentOnV2Adapter(dqn, V1ObservationAdapter(v1_env.observation_space))
    action = adapter.predict(v2_observation, deterministic=True)
    assert action in range(4)
    assert adapter.metadata()["checkpoint_modified"] is False


def test_v1_adapter_clips_v2_values_to_frozen_contract() -> None:
    v2_env = V2HVACEnv("heatwave_v2")
    observation, _ = v2_env.reset(seed=42)
    v1_env = HVACEnv()
    adapter = V1ObservationAdapter(v1_env.observation_space)
    adapted = adapter.transform(observation)
    assert adapted.shape == (9,)
    assert v1_env.observation_space.contains(adapted)
