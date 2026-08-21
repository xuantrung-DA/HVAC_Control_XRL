"""Behavioral tests for the Gymnasium HVAC environment."""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as check_sb3_env

from src.envs import ENVIRONMENT_ID
from src.envs.hvac_env import HVACEnv
from src.envs.scenario_sampler import ScenarioSamplerEnv


def test_environment_passes_gymnasium_checker() -> None:
    env = HVACEnv()
    check_env(env, skip_render_check=True)


def test_environment_passes_stable_baselines_checker() -> None:
    check_sb3_env(HVACEnv(), warn=True)


def test_environment_is_registered_with_gymnasium() -> None:
    env = gym.make(ENVIRONMENT_ID)
    observation, _ = env.reset(seed=42)
    assert env.observation_space.contains(observation)
    env.close()


def test_reset_returns_valid_observation_and_metadata() -> None:
    env = HVACEnv(scenario="normal")
    observation, info = env.reset(seed=42)

    assert observation.shape == (9,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert env.action_space.n == 4
    assert info["scenario"] == "normal"
    assert info["step"] == 0


def test_step_returns_reward_breakdown_and_physical_outputs() -> None:
    env = HVACEnv()
    env.reset(seed=42)

    observation, reward, terminated, truncated, info = env.step(2)

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["action_name"] == "MEDIUM"
    assert info["energy_kwh"] > 0
    assert info["electricity_cost"] > 0
    assert set(info["reward_components"]) == {
        "reward",
        "energy_cost_penalty",
        "comfort_penalty",
        "co2_penalty",
        "switching_penalty",
        "temperature_violation_c",
        "co2_violation_ppm",
    }


def test_episode_terminates_after_exactly_96_steps() -> None:
    env = HVACEnv()
    env.reset(seed=3)

    for step in range(96):
        _, _, terminated, truncated, _ = env.step(1)
        assert truncated is False
        assert terminated is (step == 95)

    with pytest.raises(RuntimeError, match="Episode is complete"):
        env.step(1)


def test_same_seed_and_actions_produce_identical_trajectory() -> None:
    first_env = HVACEnv(scenario="combined_stress")
    second_env = HVACEnv(scenario="combined_stress")
    first_observation, _ = first_env.reset(seed=123)
    second_observation, _ = second_env.reset(seed=123)
    np.testing.assert_array_equal(first_observation, second_observation)

    actions = [0, 1, 2, 3] * 12
    for action in actions:
        first_step = first_env.step(action)
        second_step = second_env.step(action)
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1:] == second_step[1:]


def test_high_hvac_cools_more_and_consumes_more_than_off() -> None:
    off_env = HVACEnv()
    high_env = HVACEnv()
    off_env.reset(seed=7, options={"indoor_temperature_c": 28.0})
    high_env.reset(seed=7, options={"indoor_temperature_c": 28.0})

    _, _, _, _, off_info = off_env.step(0)
    _, _, _, _, high_info = high_env.step(3)

    assert high_env.state.indoor_temperature_c < off_env.state.indoor_temperature_c
    assert high_info["energy_kwh"] > off_info["energy_kwh"]


@pytest.mark.parametrize(
    "scenario",
    [
        "normal",
        "hot_day",
        "high_occupancy",
        "expensive_electricity",
        "combined_stress",
    ],
)
def test_all_scenarios_remain_inside_observation_space(scenario: str) -> None:
    env = HVACEnv(scenario=scenario)
    observation, _ = env.reset(seed=11)
    assert env.observation_space.contains(observation)

    terminated = False
    while not terminated:
        observation, _, terminated, truncated, _ = env.step(0)
        assert env.observation_space.contains(observation)
        assert not truncated


def test_scenarios_change_conditions_without_changing_interface() -> None:
    normal = HVACEnv(scenario="normal")
    hot = HVACEnv(scenario="hot_day")
    normal.reset(seed=5)
    hot.reset(seed=5)

    normal_temperatures = []
    hot_temperatures = []
    for _ in range(60):
        normal.step(0)
        hot.step(0)
        normal_temperatures.append(normal.state.outdoor_temperature_c)
        hot_temperatures.append(hot.state.outdoor_temperature_c)

    assert max(hot_temperatures) > max(normal_temperatures) + 5.0
    assert normal.observation_space == hot.observation_space


def test_invalid_action_and_reset_options_are_rejected() -> None:
    env = HVACEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError, match="Action must be"):
        env.step(4)
    with pytest.raises(ValueError, match="Unsupported reset options"):
        env.reset(options={"outdoor_temperature_c": 99.0})
    with pytest.raises(ValueError, match="indoor_temperature_c"):
        env.reset(options={"indoor_temperature_c": 80.0})


def test_curriculum_sampler_expands_training_scenarios_deterministically() -> None:
    sampler = ScenarioSamplerEnv(
        ["normal", "hot_day", "high_occupancy"],
        normal_only_episodes=2,
        expansion_episodes=4,
        seed=7,
    )
    sequence = []
    for episode in range(7):
        _, info = sampler.reset(seed=100 + episode)
        sequence.append(info["training_scenario"])

    assert sequence[:2] == ["normal", "normal"]
    assert set(sequence[2:4]).issubset({"normal", "hot_day"})
    assert "high_occupancy" in sequence[4:]
    sampler.close()
