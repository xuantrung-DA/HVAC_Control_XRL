"""Tests for traditional HVAC controller baselines."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.base_agent import BaseAgent, ObservationView
from src.baselines import BASELINE_NAMES, FixedThermostat, RuleBasedController, create_baseline
from src.envs.hvac_env import HVACEnv


def observation(
    *,
    indoor_temperature_c: float = 24.0,
    occupancy: int = 20,
    co2_ppm: float = 600.0,
    price: float = 0.10,
    hvac_action: int = 0,
) -> np.ndarray:
    return np.array(
        [
            indoor_temperature_c,
            32.0,
            60.0,
            occupancy,
            co2_ppm,
            price,
            0.0,
            1.0,
            hvac_action,
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    ("temperature", "expected_action"),
    [(22.0, 0), (24.0, 1), (25.0, 2), (26.0, 3)],
)
def test_fixed_thermostat_uses_temperature_stages(
    temperature: float, expected_action: int
) -> None:
    controller = FixedThermostat()
    assert controller.predict(observation(indoor_temperature_c=temperature)) == expected_action


def test_fixed_thermostat_hysteresis_prevents_boundary_chatter() -> None:
    controller = FixedThermostat()

    increasing = controller.predict(
        observation(indoor_temperature_c=23.6, hvac_action=0)
    )
    decreasing = controller.predict(
        observation(indoor_temperature_c=23.4, hvac_action=1)
    )

    assert increasing == 0
    assert decreasing == 1


def test_rule_based_responds_to_temperature_and_co2() -> None:
    controller = RuleBasedController()
    assert controller.predict(observation(indoor_temperature_c=26.0)) == 1
    assert controller.predict(
        observation(indoor_temperature_c=26.0, hvac_action=1)
    ) == 2
    assert controller.predict(
        observation(co2_ppm=1200.0, hvac_action=2)
    ) == 3


def test_rule_based_caps_unoccupied_cooling() -> None:
    controller = RuleBasedController()
    action = controller.predict(
        observation(
            indoor_temperature_c=30.0,
            occupancy=0,
            co2_ppm=1600.0,
            hvac_action=1,
        )
    )
    assert action == 1


def test_rule_based_reduces_noncritical_action_during_high_price() -> None:
    controller = RuleBasedController()
    normal_price = controller.predict(
        observation(indoor_temperature_c=25.0, price=0.18, hvac_action=1)
    )
    high_price = controller.predict(
        observation(indoor_temperature_c=25.0, price=0.32, hvac_action=1)
    )
    assert normal_price == 2
    assert high_price == 1


@pytest.mark.parametrize("name", BASELINE_NAMES)
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
def test_baselines_complete_all_scenarios(name: str, scenario: str) -> None:
    env = HVACEnv(scenario=scenario)
    controller = create_baseline(name, config=env.config)
    observation_value, _ = env.reset(seed=42)
    controller.reset()
    terminated = False

    while not terminated:
        action = controller.predict(observation_value)
        observation_value, _, terminated, truncated, info = env.step(action)
        assert action in range(4)
        assert not truncated

    assert info["step"] == 96
    assert info["episode_metrics"]["energy_kwh"] >= 0.0


def test_baseline_factory_and_metadata() -> None:
    controllers = [create_baseline(name) for name in BASELINE_NAMES]
    assert all(isinstance(controller, BaseAgent) for controller in controllers)
    assert [controller.metadata()["name"] for controller in controllers] == list(
        BASELINE_NAMES
    )
    with pytest.raises(ValueError, match="Unknown baseline"):
        create_baseline("unknown")


def test_observation_validation_rejects_bad_shape_and_nan() -> None:
    with pytest.raises(ValueError, match="shape"):
        ObservationView.from_array(np.zeros(8, dtype=np.float32))
    invalid = np.zeros(9, dtype=np.float32)
    invalid[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ObservationView.from_array(invalid)
