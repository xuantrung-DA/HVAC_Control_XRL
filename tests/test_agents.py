"""Tests for RL agent interfaces, learning updates, and checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents import RL_AGENT_NAMES, create_agent
from src.agents.base_agent import ObservationScaler
from src.agents.double_dqn import DoubleDQNAgent
from src.agents.dqn import DQNAgent, ReplayBuffer
from src.agents.ppo import PPOAgent
from src.agents.q_learning import QLearningAgent
from src.envs.hvac_env import HVACEnv
from src.utils.config import deep_merge, load_agent_config


@pytest.fixture
def env() -> HVACEnv:
    return HVACEnv()


def test_observation_scaler_maps_space_bounds(env: HVACEnv) -> None:
    scaler = ObservationScaler(env.observation_space)
    np.testing.assert_allclose(scaler.transform(env.observation_space.low), -1.0)
    np.testing.assert_allclose(scaler.transform(env.observation_space.high), 1.0)


def test_replay_buffer_adds_and_samples_deterministically() -> None:
    first = ReplayBuffer(capacity=10, observation_size=9, seed=5)
    second = ReplayBuffer(capacity=10, observation_size=9, seed=5)
    for index in range(8):
        observation = np.full(9, index, dtype=np.float32)
        transition = (observation, index % 4, float(index), observation + 1, False)
        first.add(*transition)
        second.add(*transition)

    first_sample = first.sample(4)
    second_sample = second.sample(4)
    for left, right in zip(first_sample, second_sample, strict=True):
        np.testing.assert_array_equal(left, right)


def test_q_learning_update_changes_q_value(env: HVACEnv) -> None:
    agent = QLearningAgent()
    observation, _ = env.reset(seed=1)
    next_observation, reward, terminated, truncated, _ = env.step(2)
    state = agent.discretize(observation)

    agent.update(
        observation,
        2,
        reward,
        next_observation,
        terminated or truncated,
    )

    assert agent.q_table[state][2] != 0.0


def test_q_learning_checkpoint_round_trip(
    env: HVACEnv, tmp_path: Path
) -> None:
    agent = QLearningAgent()
    agent.learn(env, episodes=3, seed=9)
    observation, _ = env.reset(seed=22)
    expected = agent.predict(observation)
    checkpoint = tmp_path / "q_learning.npz"
    agent.save(checkpoint)

    loaded = QLearningAgent()
    loaded.load(checkpoint)

    assert loaded.predict(observation) == expected
    assert loaded.q_table.keys() == agent.q_table.keys()
    for state in agent.q_table:
        np.testing.assert_allclose(loaded.q_table[state], agent.q_table[state])


@pytest.mark.parametrize("agent_class", [DQNAgent, DoubleDQNAgent])
def test_dqn_variants_train_and_checkpoint(
    agent_class: type[DQNAgent], env: HVACEnv, tmp_path: Path
) -> None:
    config = deep_merge(
        load_agent_config("dqn"),
        {
            "agent": {"device": "cpu"},
            "replay_buffer": {
                "capacity": 500,
                "batch_size": 8,
                "warmup_steps": 8,
            },
            "optimization": {
                "train_frequency": 1,
                "target_update_frequency": 16,
            },
            "exploration": {"epsilon_decay_steps": 100},
        },
    )
    agent = agent_class(env.observation_space, env.action_space, config=config)
    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.online_network.named_parameters()
    }
    summary = agent.learn(env, total_steps=192, seed=4)
    assert summary.total_steps == 192
    assert agent.gradient_updates > 0
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in agent.online_network.named_parameters()
    )

    observation, _ = env.reset(seed=30)
    expected_scores = agent.action_scores(observation)
    checkpoint = tmp_path / f"{agent.name}.pt"
    agent.save(checkpoint)
    loaded = agent_class(env.observation_space, env.action_space, config=config)
    loaded.load(checkpoint)
    np.testing.assert_allclose(
        loaded.action_scores(observation), expected_scores, rtol=0.0, atol=0.0
    )
    assert loaded.predict(observation) == agent.predict(observation)


def test_ppo_short_update_and_checkpoint(env: HVACEnv, tmp_path: Path) -> None:
    config = deep_merge(
        load_agent_config("ppo"),
        {
            "agent": {"device": "cpu"},
            "training": {"n_steps": 96, "batch_size": 32, "n_epochs": 2},
        },
    )
    agent = PPOAgent(env, config=config)
    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.model.policy.named_parameters()
    }
    summary = agent.learn(total_timesteps=192)
    assert summary.total_steps >= 192
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in agent.model.policy.named_parameters()
    )

    raw_env = env.unwrapped
    observation, _ = raw_env.reset(seed=30)
    expected_scores = agent.action_scores(observation)
    checkpoint = tmp_path / "ppo_test.zip"
    agent.save(checkpoint)
    loaded = PPOAgent(HVACEnv(), config=config)
    loaded.load(checkpoint)
    np.testing.assert_allclose(
        loaded.action_scores(observation), expected_scores, rtol=1e-6, atol=1e-6
    )
    assert loaded.predict(observation) == agent.predict(observation)


def test_agent_factory_and_deterministic_inference(env: HVACEnv) -> None:
    observation, _ = env.reset(seed=42)
    for name in RL_AGENT_NAMES:
        agent = create_agent(name, HVACEnv())
        first = agent.predict(observation, deterministic=True)
        second = agent.predict(observation, deterministic=True)
        assert first == second
        assert first in range(4)

    with pytest.raises(ValueError, match="Unknown RL agent"):
        create_agent("unknown", env)
