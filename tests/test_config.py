"""Tests for project configuration and reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.utils.config import ConfigError, deep_merge, load_agent_config, load_environment_config
from src.utils.seed import seed_everything


def test_environment_config_has_consistent_episode_length() -> None:
    config = load_environment_config()
    simulation = config["simulation"]

    calculated_steps = (
        simulation["episode_hours"] * 60 // simulation["timestep_minutes"]
    )
    assert simulation["steps_per_episode"] == calculated_steps == 96


def test_scenario_is_merged_without_modifying_base_config() -> None:
    base = load_environment_config()
    hot_day = load_environment_config(scenario="hot_day")

    assert base["weather"]["daily_high_c"] == 34.0
    assert hot_day["weather"]["daily_high_c"] == 40.0
    assert hot_day["simulation"] == base["simulation"]


def test_unknown_scenario_has_descriptive_error() -> None:
    with pytest.raises(ConfigError, match="Unknown scenario"):
        load_environment_config(scenario="not-a-scenario")


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"nested": {"left": 1, "right": 2}}
    override = {"nested": {"right": 3}}

    merged = deep_merge(base, override)

    assert merged == {"nested": {"left": 1, "right": 3}}
    assert base == {"nested": {"left": 1, "right": 2}}


def test_agent_config_loads_by_name() -> None:
    config = load_agent_config("dqn")
    assert config["agent"]["name"] == "dqn"
    assert config["model"]["hidden_sizes"] == [128, 128]


def test_seed_everything_reproduces_python_and_numpy() -> None:
    seed_everything(17)
    first = (random.random(), np.random.random())

    seed_everything(17)
    second = (random.random(), np.random.random())

    assert first == second


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
