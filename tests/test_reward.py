"""Tests for decomposed HVAC reward calculation."""

from __future__ import annotations

import pytest

from src.envs.reward import HVACReward
from src.utils.config import load_environment_config


@pytest.fixture
def reward_model() -> HVACReward:
    return HVACReward(load_environment_config())


def test_reward_equals_negative_sum_of_penalties(reward_model: HVACReward) -> None:
    result = reward_model.calculate(
        indoor_temperature_c=27.0,
        occupancy=20,
        co2_ppm=1200.0,
        electricity_cost=0.50,
        previous_action=0,
        action=3,
    )
    penalty_sum = (
        result.energy_cost_penalty
        + result.comfort_penalty
        + result.co2_penalty
        + result.switching_penalty
    )
    assert result.reward == pytest.approx(-penalty_sum)


def test_occupied_comfort_violation_is_penalized(reward_model: HVACReward) -> None:
    comfortable = reward_model.calculate(
        indoor_temperature_c=24.0,
        occupancy=10,
        co2_ppm=700.0,
        electricity_cost=0.0,
        previous_action=0,
        action=0,
    )
    too_hot = reward_model.calculate(
        indoor_temperature_c=27.0,
        occupancy=10,
        co2_ppm=700.0,
        electricity_cost=0.0,
        previous_action=0,
        action=0,
    )

    assert comfortable.comfort_penalty == 0.0
    assert too_hot.temperature_violation_c == 2.0
    assert too_hot.comfort_penalty > 0.0
    assert too_hot.reward < comfortable.reward


def test_unoccupied_period_uses_wider_temperature_bounds(
    reward_model: HVACReward,
) -> None:
    result = reward_model.calculate(
        indoor_temperature_c=27.0,
        occupancy=0,
        co2_ppm=700.0,
        electricity_cost=0.0,
        previous_action=0,
        action=0,
    )
    assert result.temperature_violation_c == 0.0


def test_co2_violation_is_scaled_and_penalized(reward_model: HVACReward) -> None:
    result = reward_model.calculate(
        indoor_temperature_c=24.0,
        occupancy=30,
        co2_ppm=1200.0,
        electricity_cost=0.0,
        previous_action=1,
        action=1,
    )
    assert result.co2_violation_ppm == 200.0
    assert result.co2_penalty == 1.5


def test_switching_penalty_reflects_action_magnitude(reward_model: HVACReward) -> None:
    stable = reward_model.calculate(
        indoor_temperature_c=24.0,
        occupancy=5,
        co2_ppm=600.0,
        electricity_cost=0.0,
        previous_action=2,
        action=2,
    )
    large_switch = reward_model.calculate(
        indoor_temperature_c=24.0,
        occupancy=5,
        co2_ppm=600.0,
        electricity_cost=0.0,
        previous_action=0,
        action=3,
    )
    assert stable.switching_penalty == 0.0
    assert large_switch.switching_penalty == pytest.approx(0.30)
