"""Predictive shield decisions, projections, and environment flow tests."""

from __future__ import annotations

from dataclasses import replace

from src.envs.v2 import V2HVACEnv
from src.forecasting import ForecastBundle
from src.risk import RiskVector
from src.shields import ShieldDecisionType


def make_risk(thermal=0.0, co2=0.0, reliability=1.0, error=0.0):
    return RiskVector(
        thermal_risk=thermal,
        co2_risk=co2,
        occupancy_surge=0.0,
        forecast_uncertainty=0.1,
        forecast_error=error,
        energy_peak_risk=0.2,
        weather_reliability=reliability,
        occupancy_reliability=reliability,
        price_reliability=reliability,
    )


def test_safe_low_action_is_allowed_deterministically() -> None:
    env = V2HVACEnv("normal_v2")
    env.reset(seed=42)
    inputs = env._timeline.inputs[0]
    first = env.shield.decide(
        state=env.state, inputs=inputs, proposed_action=1,
        forecast=env._forecast, risk=make_risk(),
    )
    second = env.shield.decide(
        state=env.state, inputs=inputs, proposed_action=1,
        forecast=env._forecast, risk=make_risk(),
    )
    assert first == second
    assert first.decision is ShieldDecisionType.ALLOW
    assert first.executed_action == 1
    assert first.projection is not None


def test_high_co2_risk_clamps_low_ventilation_upward() -> None:
    env = V2HVACEnv("normal_v2")
    env.reset(seed=42)
    inputs = replace(env._timeline.inputs[40], occupancy=70)
    state = replace(env.state, co2_ppm=960.0, step=40)
    decision = env.shield.decide(
        state=state, inputs=inputs, proposed_action=0,
        forecast=env.forecaster.predict(inputs, 40), risk=make_risk(co2=0.9),
    )
    assert decision.decision is ShieldDecisionType.CLAMP
    assert decision.executed_action == 3
    assert decision.constraint in {"CO2_RISK", "PROJECTED_CO2_LIMIT"}


def test_high_cooling_near_lower_bound_is_rejected() -> None:
    env = V2HVACEnv("normal_v2")
    env.reset(seed=42)
    inputs = replace(env._timeline.inputs[40], occupancy=30)
    state = replace(env.state, indoor_temperature_c=22.1, step=40)
    decision = env.shield.decide(
        state=state, inputs=inputs, proposed_action=3,
        forecast=env.forecaster.predict(inputs, 40), risk=make_risk(),
    )
    assert decision.decision is ShieldDecisionType.REJECT
    assert decision.executed_action == 1
    assert decision.constraint == "OVERCOOLING"


def test_unavailable_forecast_under_high_risk_uses_fallback() -> None:
    env = V2HVACEnv("normal_v2")
    env.reset(seed=42)
    inputs = env._timeline.inputs[40]
    unavailable = ForecastBundle(40, "fault_injector", tuple(), "unavailable")
    decision = env.shield.decide(
        state=replace(env.state, indoor_temperature_c=25.5, step=40),
        inputs=inputs,
        proposed_action=0,
        forecast=unavailable,
        risk=make_risk(thermal=0.9, reliability=0.0, error=1.0),
    )
    assert decision.decision is ShieldDecisionType.FALLBACK
    assert decision.executed_action in range(4)
    assert decision.intervention


def test_environment_logs_proposed_and_executed_actions() -> None:
    enabled = V2HVACEnv("normal_v2", shield_enabled=True)
    enabled.reset(seed=42)
    _, _, _, _, info = enabled.step(1)
    assert info["control"]["proposed_action"] == 1
    assert info["control"]["executed_action"] in range(4)
    assert info["control"]["shield"]["decision"] in {
        "ALLOW", "CLAMP", "REJECT", "FALLBACK"
    }

    disabled = V2HVACEnv("normal_v2", shield_enabled=False)
    disabled.reset(seed=42)
    _, _, _, _, disabled_info = disabled.step(3)
    assert disabled_info["control"]["executed_action"] == 3
    assert disabled_info["control"]["shield"]["reason"] == "Shield disabled for controlled ablation."
