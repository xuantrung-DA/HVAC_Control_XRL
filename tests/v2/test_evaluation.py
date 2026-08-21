"""V2 multi-objective evaluation and frozen baseline tests."""

from __future__ import annotations

from src.baselines import V2RandomController, V2RuleBasedController
from src.evaluation import aggregate_v2_results, evaluate_v2_controller
from src.utils.config import PROJECT_ROOT, load_yaml


def test_rule_based_and_random_produce_complete_multiobjective_metrics() -> None:
    config = load_yaml(PROJECT_ROOT / "configs/v2/controllers.yaml")
    controllers = [V2RuleBasedController(config), V2RandomController(42)]
    for controller in controllers:
        results = evaluate_v2_controller(
            controller,
            controller_name=controller.name,
            scenarios=["normal_v2"],
            seeds=[901],
            shield_enabled=False,
        )
        result = results[0]
        assert result.whole_building_kwh >= result.hvac_ventilation_kwh >= 0.0
        assert result.peak_power_kw > 0.0
        assert 0.0 <= result.comfort_violation_percent <= 100.0
        assert 0.0 <= result.co2_violation_percent <= 100.0
        assert sum(result.action_distribution) == 1.0
        assert result.shield_intervention_percent == 0.0


def test_aggregation_reports_mean_std_and_action_diversity() -> None:
    controller = V2RuleBasedController(
        load_yaml(PROJECT_ROOT / "configs/v2/controllers.yaml")
    )
    results = evaluate_v2_controller(
        controller,
        controller_name=controller.name,
        scenarios=["normal_v2"],
        seeds=[901, 902],
        shield_enabled=True,
    )
    summary = aggregate_v2_results(results)
    assert summary["episodes"] == 2
    assert set(summary["metrics"]["whole_building_kwh"]) == {"mean", "std"}
    assert abs(sum(summary["action_distribution_mean"]) - 1.0) < 1e-9
