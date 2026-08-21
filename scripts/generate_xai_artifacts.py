"""Generate Step 6 explanations from the immutable DQN demo checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agents.dqn import DQNAgent
from src.envs.hvac_env import HVACEnv, OBSERVATION_NAMES
from src.utils.config import PROJECT_ROOT, deep_merge, load_agent_config, load_config
from src.xai.counterfactual import DQNCounterfactualExplainer
from src.xai.feature_attribution import DQNFeatureAttributor
from src.xai.trajectory import (
    TrajectoryStep,
    explain_episode,
    representative_step,
    summarize_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "models" / "demo_manifest.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "trajectories" / "xai",
    )
    parser.add_argument("--maximum-steps", type=int)
    args = parser.parse_args()

    manifest = _read_json(args.manifest)
    checkpoint = PROJECT_ROOT / Path(manifest["frozen_checkpoint"])
    actual_hash = _sha256(checkpoint)
    expected_hash = manifest["frozen_checkpoint_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Frozen demo checkpoint hash mismatch; refusing to explain a changed model"
        )

    xai_config = load_config("xai")
    env = HVACEnv()
    dqn_config = deep_merge(
        load_agent_config("dqn"),
        {"agent": {"seed": int(manifest["training_seed"])}},
    )
    agent = DQNAgent(env.observation_space, env.action_space, config=dqn_config)
    agent.load(checkpoint)
    env.close()

    reference = np.array(
        [xai_config["attribution"]["reference_state"][name] for name in OBSERVATION_NAMES],
        dtype=np.float32,
    )
    attributor = DQNFeatureAttributor(
        agent,
        reference,
        integration_steps=int(xai_config["attribution"]["integration_steps"]),
    )
    counterfactual = DQNCounterfactualExplainer(
        agent,
        xai_config["counterfactual"]["searchable_features"],
        two_feature_fallback=xai_config["counterfactual"]["two_feature_fallback"],
    )

    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    all_records: list[TrajectoryStep] = []
    summaries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    seed = int(xai_config["trajectory"]["seed"])
    for scenario in xai_config["trajectory"]["scenarios"]:
        records = explain_episode(
            agent,
            str(scenario),
            seed,
            attributor,
            counterfactual,
            maximum_steps=args.maximum_steps,
        )
        all_records.extend(records)
        summaries.append(summarize_trajectory(records))
        samples.append(representative_step(records).as_dict())
        _write_json(
            output_directory / f"{scenario}_trajectory.json",
            {"scenario": scenario, "records": [record.as_dict() for record in records]},
        )
        _write_csv(
            output_directory / f"{scenario}_trajectory.csv",
            [record.flat_dict() for record in records],
        )

    validation = _validate_explanations(
        all_records, samples, attributor, counterfactual
    )

    report = {
        "controller": "dqn",
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": actual_hash,
        "model_unchanged": True,
        "methods": {
            "feature_attribution": "integrated_gradients_decision_margin",
            "faithfulness": "completeness plus reference ablation",
            "counterfactual": "sparse bounded grid search with verified action flip",
            "causal_claims": False,
        },
        "scenario_summaries": summaries,
        "representative_explanations": samples,
        "validation": validation,
        "limitations": [
            "Attributions explain the learned DQN locally; they are not physical causal effects.",
            "Integrated Gradients depends on the configured reference state.",
            "Counterfactuals are limited to configured features, bounds, and grid resolution.",
            "A plausible state vector may not be dynamically reachable from every trajectory history.",
            "Correlated time and weather features can share or redistribute attribution.",
        ],
    }
    _write_json(output_directory / "step6_xai_report.json", report)
    _write_csv(
        output_directory / "all_scenarios_trajectory.csv",
        [record.flat_dict() for record in all_records],
    )
    print(json.dumps(summaries, indent=2))


def _validate_explanations(
    records: list[TrajectoryStep],
    samples: list[dict[str, Any]],
    attributor: DQNFeatureAttributor,
    counterfactual: DQNCounterfactualExplainer,
) -> dict[str, Any]:
    normalization_errors: list[float] = []
    completeness_absolute_errors: list[float] = []
    correlations: list[float] = []
    top_ablation_valid: list[bool] = []
    found_counterfactuals = []
    for record in records:
        attribution = record.feature_attribution
        total = sum(
            item["absolute_importance_pct"]
            for item in attribution["contributions"]
        )
        normalization_errors.append(abs(total - 100.0) if total else 0.0)
        faithfulness = attribution["faithfulness"]
        completeness_absolute_errors.append(
            faithfulness["completeness_absolute_error"]
        )
        if faithfulness["absolute_attribution_ablation_correlation"] is not None:
            correlations.append(
                faithfulness["absolute_attribution_ablation_correlation"]
            )
        top_ablation_valid.append(
            faithfulness["top_feature_changes_margin_when_ablated"]
        )
        if record.counterfactual["found"]:
            found_counterfactuals.append(record.counterfactual)

    deterministic_checks = []
    for sample in samples:
        observation = np.array(
            [sample["state"][name] for name in OBSERVATION_NAMES], dtype=np.float32
        )
        repeated = attributor.explain(observation).as_dict()
        deterministic_checks.append(repeated == sample["feature_attribution"])
        repeated_counterfactual = counterfactual.explain(observation).as_dict()
        deterministic_checks.append(
            repeated_counterfactual == sample["counterfactual"]
        )

    scenario_phrases = {
        sample["scenario"]: sample["feature_attribution"]["human_readable"]
        for sample in samples
    }
    return {
        "steps_validated": len(records),
        "deterministic_replay_checks": len(deterministic_checks),
        "deterministic_replay_passed": all(deterministic_checks),
        "maximum_importance_normalization_error_pct": float(
            max(normalization_errors, default=0.0)
        ),
        "median_completeness_absolute_error": float(
            np.median(completeness_absolute_errors)
        ),
        "p95_completeness_absolute_error": float(
            np.percentile(completeness_absolute_errors, 95)
        ),
        "mean_absolute_attribution_ablation_correlation": (
            float(np.mean(correlations)) if correlations else None
        ),
        "top_feature_ablation_effect_rate": float(np.mean(top_ablation_valid)),
        "counterfactual_found_rate": len(found_counterfactuals) / len(records),
        "counterfactual_action_flip_validity_rate": float(
            np.mean([item["action_changed"] for item in found_counterfactuals])
        ),
        "counterfactual_bounds_validity_rate": float(
            np.mean([item["within_bounds"] for item in found_counterfactuals])
        ),
        "scenario_explanations_are_distinct": len(set(scenario_phrases.values()))
        == len(scenario_phrases),
        "scenario_explanation_samples": scenario_phrases,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
