"""V2 DQN vector and curriculum compatibility tests."""

from __future__ import annotations

from copy import deepcopy

from src.agents.dqn import DQNAgent
from src.envs.hvac_env import HVACEnv
from src.envs.v2 import V2HVACEnv, V2ScenarioSamplerEnv
from src.utils.config import load_agent_config


def test_dqn_accepts_v2_observation_size_without_breaking_v1() -> None:
    v2 = V2HVACEnv(shield_enabled=False)
    observation, _ = v2.reset(seed=42)
    config = deepcopy(load_agent_config("dqn"))
    config["agent"]["device"] = "cpu"
    v2_agent = DQNAgent(v2.observation_space, v2.action_space, config=config)
    assert v2_agent.predict(observation) in range(4)
    assert v2_agent.observation_size == 35

    v1 = HVACEnv()
    v1_observation, _ = v1.reset(seed=42)
    v1_agent = DQNAgent(v1.observation_space, v1.action_space, config=config)
    assert v1_agent.predict(v1_observation) in range(4)
    assert v1_agent.observation_size == 9


def test_curriculum_never_samples_outside_training_pool() -> None:
    scenarios = ["normal_v2", "hot_day_v2", "high_occupancy_v2"]
    env = V2ScenarioSamplerEnv(
        scenarios, seed=42, normal_only_episodes=2, shield_enabled=False
    )
    seen = []
    for episode in range(12):
        _, info = env.reset(seed=42 + episode)
        seen.append(info["scenario"])
    assert seen[:2] == ["normal_v2", "normal_v2"]
    assert set(seen).issubset(scenarios)
    assert len(set(seen)) > 1


def test_short_dqn_learning_run_updates_network() -> None:
    env = V2ScenarioSamplerEnv(
        ["normal_v2", "hot_day_v2"], seed=42, normal_only_episodes=1,
        shield_enabled=False,
    )
    config = deepcopy(load_agent_config("dqn"))
    config["agent"]["device"] = "cpu"
    config["replay_buffer"].update({"capacity": 256, "batch_size": 16, "warmup_steps": 16})
    config["optimization"].update({"train_frequency": 1, "target_update_frequency": 20})
    agent = DQNAgent(env.observation_space, env.action_space, config=config)
    summary = agent.learn(env, total_steps=40, seed=42)
    assert summary.total_steps == 40
    assert agent.gradient_updates > 0
    assert agent.parameter_count() > 0
