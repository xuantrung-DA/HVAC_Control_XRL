"""V2 protocol isolation, split, gate, and evidence tests."""

from __future__ import annotations

import json

from src.utils.config import PROJECT_ROOT
from src.utils.v2_manifest import (
    V2_CONFIG_PATHS,
    build_v1_baseline_manifest,
    build_v2_protocol_manifest,
    load_yaml,
    validate_scenario_splits,
)


def test_v1_frozen_evidence_is_intact() -> None:
    manifest = build_v1_baseline_manifest()
    assert manifest["git_tag"] == "v1-frozen"
    assert manifest["checkpoint_sha256"] == (
        "9d33a157ddb25a79a6ef03efee97c3491ef61bcb8b47d814f237d54695d5236a"
    )
    assert manifest["frozen_controller"] == "dqn"
    assert manifest["training_seed"] == 2026
    assert manifest["immutable"] is True


def test_v2_scenario_splits_are_disjoint() -> None:
    scenarios = load_yaml(PROJECT_ROOT / "configs/v2/scenarios.yaml")
    validate_scenario_splits(scenarios)
    splits = scenarios["splits"]
    held_out = {
        item
        for name, values in splits.items()
        if name.startswith("held_out")
        for item in values
    }
    assert "combined_stress_v2" in held_out
    assert "unexpected_occupancy_surge_v2" in held_out
    assert "forecast_failure_v2" in held_out
    assert not held_out.intersection(splits["train"])
    assert not held_out.intersection(splits["validation"])


def test_v2_acceptance_gates_are_locked_in_config() -> None:
    protocol = load_yaml(PROJECT_ROOT / "configs/v2/protocol.yaml")
    gates = protocol["acceptance_gates"]
    assert gates["efficiency"][
        "whole_building_energy_delta_vs_rule_based_max_pct"
    ] == 0.0
    assert gates["efficiency"][
        "hvac_ventilation_energy_delta_vs_rule_based_max_pct"
    ] == 0.0
    assert gates["constraints"][
        "comfort_violation_percent_max_exclusive"
    ] == 5.0
    assert gates["constraints"]["co2_violation_percent_max_exclusive"] == 1.0
    assert gates["constraints"]["critical_safety_violations_max"] == 0
    assert protocol["held_out_policy"]["rerun_after_observing_results"] is False


def test_generated_protocol_manifests_match_current_configs() -> None:
    manifest_path = PROJECT_ROOT / "outputs/v2/protocol/v2_protocol_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        stored = json.load(stream)
    current = build_v2_protocol_manifest()
    assert stored["protocol_bundle_sha256"] == current["protocol_bundle_sha256"]
    assert stored["config_hashes"] == current["config_hashes"]
    assert stored["v1_git_commit"] == current["v1_git_commit"]
    assert stored["final_test_opened"] is False


def test_v2_namespace_exists_without_overwriting_v1() -> None:
    for path in V2_CONFIG_PATHS:
        assert (PROJECT_ROOT / path).is_file()
    assert (PROJECT_ROOT / "models/v2").is_dir()
    assert (PROJECT_ROOT / "outputs/v2").is_dir()
    assert (PROJECT_ROOT / "models/dqn/demo_best.pt").is_file()
