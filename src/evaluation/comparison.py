"""Aggregation, baseline comparisons, and Pareto-style analysis."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.evaluation.performance import EpisodeResult


METRICS = (
    "reward",
    "energy_kwh",
    "electricity_cost",
    "comfort_violation_percent",
    "comfort_violation_hours",
    "average_temperature_deviation_c",
    "co2_violation_percent",
    "co2_violation_hours",
    "average_co2_excess_ppm",
    "hvac_switches",
    "action_0_fraction",
    "action_1_fraction",
    "action_2_fraction",
    "action_3_fraction",
    "action_entropy",
    "unique_actions",
)
PARETO_OBJECTIVES = (
    "energy_kwh_mean",
    "comfort_violation_percent_mean",
    "co2_violation_percent_mean",
)


def aggregate_results(
    results: Iterable[EpisodeResult],
    split_by_scenario: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate episode rows by scenario and by train/validation/test split."""

    scenario_groups: dict[tuple[str, str], list[EpisodeResult]] = defaultdict(list)
    split_groups: dict[tuple[str, str], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        scenario_groups[(result.controller, result.scenario)].append(result)
        split = split_by_scenario[result.scenario]
        split_groups[(result.controller, split)].append(result)

    scenario_summary = [
        _summarize(group, controller=controller, scenario=scenario)
        for (controller, scenario), group in sorted(scenario_groups.items())
    ]
    split_summary = [
        _summarize(group, controller=controller, split=split)
        for (controller, split), group in sorted(split_groups.items())
    ]
    return scenario_summary, split_summary


def pareto_analysis(
    split_summary: list[dict[str, Any]],
    feasibility: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Find non-dominated controllers and normalized balanced trade-off scores."""

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_summary:
        by_split[str(row["split"])].append(row)

    output: dict[str, dict[str, Any]] = {}
    for split, rows in by_split.items():
        frontier = [
            row["controller"]
            for row in rows
            if not any(
                _dominates(candidate, row)
                for candidate in rows
                if candidate is not row
            )
        ]
        scores = _balanced_scores(rows)
        feasible_rows = [
            row for row in rows if _is_feasible(row, feasibility)
        ]
        feasible_frontier = [
            row["controller"]
            for row in feasible_rows
            if not any(
                _dominates(candidate, row)
                for candidate in feasible_rows
                if candidate is not row
            )
        ]
        output[split] = {
            "pareto_front": sorted(frontier),
            "feasible_controllers": sorted(
                str(row["controller"]) for row in feasible_rows
            ),
            "feasible_pareto_front": sorted(feasible_frontier),
            "balanced_tradeoff_score": scores,
            "score_interpretation": "Lower is better; equal weight for energy/cost, comfort, and CO2.",
        }
    return output


def baseline_comparisons(
    split_summary: list[dict[str, Any]],
    rl_agents: Iterable[str],
    significance: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Compare RL reward and objective deltas with Random and traditional baselines."""

    lookup = {
        (str(row["controller"]), str(row["split"])): row for row in split_summary
    }
    comparisons: list[dict[str, Any]] = []
    splits = sorted({str(row["split"]) for row in split_summary})
    for agent in rl_agents:
        for split in splits:
            current = lookup.get((agent, split))
            if current is None:
                continue
            record: dict[str, Any] = {
                "controller": agent,
                "split": split,
                "policy_collapse": _policy_collapse(current),
                "low_action_diversity": _low_action_diversity(current),
            }
            for baseline in ("random", "fixed_thermostat", "rule_based"):
                reference = lookup[(baseline, split)]
                reward_delta = current["reward_mean"] - reference["reward_mean"]
                threshold = _significance_threshold(current, reference, significance)
                record[f"beats_{baseline}"] = reward_delta > 0
                record[f"meaningfully_beats_{baseline}"] = reward_delta > threshold
                record[f"reward_delta_vs_{baseline}"] = reward_delta
                record[f"reward_significance_threshold_vs_{baseline}"] = threshold
            rule = lookup[("rule_based", split)]
            for metric in (
                "energy_kwh_mean",
                "electricity_cost_mean",
                "comfort_violation_percent_mean",
                "co2_violation_percent_mean",
                "hvac_switches_mean",
            ):
                record[f"{metric}_delta_vs_rule_based"] = current[metric] - rule[metric]
                record[f"{metric}_pct_vs_rule_based"] = _percentage_delta(
                    current[metric], rule[metric]
                )
            comparisons.append(record)
    return comparisons


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize(
    group: list[EpisodeResult],
    *,
    controller: str,
    scenario: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "controller": controller,
        "samples": len(group),
        "training_seeds": len(
            {item.training_seed for item in group if item.training_seed is not None}
        ),
        "evaluation_seeds": len({item.evaluation_seed for item in group}),
    }
    if scenario is not None:
        record["scenario"] = scenario
    if split is not None:
        record["split"] = split
    for metric in METRICS:
        values = np.asarray([getattr(item, metric) for item in group], dtype=np.float64)
        record[f"{metric}_mean"] = float(np.mean(values))
        record[f"{metric}_std"] = float(np.std(values))
    return record


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_values = [float(left[metric]) for metric in PARETO_OBJECTIVES]
    right_values = [float(right[metric]) for metric in PARETO_OBJECTIVES]
    return all(a <= b + 1e-9 for a, b in zip(left_values, right_values)) and any(
        a < b - 1e-9 for a, b in zip(left_values, right_values)
    )


def _balanced_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    categories = {
        "energy_cost": ("energy_kwh_mean", "electricity_cost_mean"),
        "comfort": ("comfort_violation_percent_mean",),
        "co2": ("co2_violation_percent_mean",),
    }
    normalized: dict[str, dict[str, float]] = defaultdict(dict)
    for category, metrics in categories.items():
        category_values: dict[str, list[float]] = defaultdict(list)
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            minimum, maximum = min(values), max(values)
            span = maximum - minimum
            for row, value in zip(rows, values, strict=True):
                normalized_value = (value - minimum) / span if span > 1e-12 else 0.0
                category_values[str(row["controller"])].append(normalized_value)
        for controller, values in category_values.items():
            normalized[controller][category] = float(np.mean(values))
    return {
        controller: float(np.mean(list(category_scores.values())))
        for controller, category_scores in normalized.items()
    }


def _policy_collapse(row: Mapping[str, Any]) -> bool:
    maximum_fraction = max(
        float(row[f"action_{action}_fraction_mean"]) for action in range(4)
    )
    return float(row["action_entropy_mean"]) < 0.15 or maximum_fraction > 0.95


def _low_action_diversity(row: Mapping[str, Any]) -> bool:
    return (
        float(row["action_entropy_mean"]) < 0.35
        or float(row["unique_actions_mean"]) < 2.0
    )


def _is_feasible(
    row: Mapping[str, Any], feasibility: Mapping[str, float] | None
) -> bool:
    if feasibility is None:
        return True
    return (
        float(row["comfort_violation_percent_mean"])
        <= float(feasibility["maximum_comfort_violation_percent"])
        and float(row["co2_violation_percent_mean"])
        <= float(feasibility["maximum_co2_violation_percent"])
    )


def _significance_threshold(
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
    significance: Mapping[str, float] | None,
) -> float:
    if significance is None:
        return 0.0
    current_error = float(current["reward_std"]) / np.sqrt(float(current["samples"]))
    reference_error = float(reference["reward_std"]) / np.sqrt(
        float(reference["samples"])
    )
    confidence_margin = float(significance["confidence_multiplier"]) * np.sqrt(
        current_error**2 + reference_error**2
    )
    return max(
        float(significance["minimum_reward_improvement"]),
        float(confidence_margin),
    )


def _percentage_delta(value: float, baseline: float) -> float | None:
    if abs(baseline) < 1e-12:
        return None
    return 100.0 * (value - baseline) / abs(baseline)
