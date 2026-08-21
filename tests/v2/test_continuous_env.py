"""Continuous physical control interface tests."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from src.envs.v2 import V2ContinuousHVACEnv


def test_continuous_environment_passes_checker_and_episode() -> None:
    env = V2ContinuousHVACEnv("normal_v2")
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=42)
    assert observation.shape == (37,)
    assert env.observation_space.contains(observation)
    terminated = False
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(
            np.array([0.3, 0.6], dtype=np.float32)
        )
        assert np.isfinite(reward)
        assert not truncated
    assert info["step"] == 96
    assert info["control"]["type"] == "continuous"


def test_cooling_and_ventilation_are_physically_independent() -> None:
    cooling = V2ContinuousHVACEnv("normal_v2")
    ventilation = V2ContinuousHVACEnv("normal_v2")
    cooling.reset(seed=42)
    ventilation.reset(seed=42)
    _, _, _, _, cooling_info = cooling.step(np.array([0.8, 0.0], dtype=np.float32))
    _, _, _, _, ventilation_info = ventilation.step(np.array([0.0, 0.8], dtype=np.float32))
    assert cooling_info["transition"]["commanded_cooling_kw"] > 0.0
    assert cooling_info["transition"]["air_quality"]["ventilation_air_changes_per_hour"] == 0.0
    assert ventilation_info["transition"]["commanded_cooling_kw"] == 0.0
    assert ventilation_info["transition"]["air_quality"]["ventilation_air_changes_per_hour"] > 0.0
