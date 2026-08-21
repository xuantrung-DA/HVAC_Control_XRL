"""Tests for the evidence-producing V2 physics validation harness."""

from __future__ import annotations

from src.envs.v2.validation import build_validation_report


def test_physics_validation_report_passes_all_cases() -> None:
    report = build_validation_report(performance_steps=500)
    assert report["thermal_network"] == "2R1C"
    assert report["checks_total"] >= 9
    assert report["checks_passed"] == report["checks_total"]
    assert report["all_checks_passed"] is True
    assert report["training_authorized"] is True
    assert report["performance"]["steps_per_second"] > 100.0


def test_validation_report_covers_required_interventions() -> None:
    report = build_validation_report(performance_steps=10)
    names = {case["name"] for case in report["cases"]}
    assert {
        "occupancy_heat_co2_moisture",
        "internal_electrical_gains",
        "outdoor_envelope_direction",
        "door_infiltration_duration",
        "ventilation_co2_thermal_tradeoff",
        "solar_day_night",
        "hvac_action_energy_ordering",
        "hvac_inertia_and_gradual_temperature",
        "physical_bounds_stress_rollout",
    }.issubset(names)
