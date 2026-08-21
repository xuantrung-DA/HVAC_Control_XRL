"""Build auditable V1 baseline and V2 protocol manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.utils.config import PROJECT_ROOT


V2_CONFIG_PATHS = (
    Path("configs/v2/protocol.yaml"),
    Path("configs/v2/scenarios.yaml"),
    Path("configs/v2/evaluation.yaml"),
    Path("configs/v2/action_mapping.yaml"),
    Path("configs/v2/environment.yaml"),
    Path("configs/v2/forecasting.yaml"),
    Path("configs/v2/risk.yaml"),
)

V1_EVIDENCE_PATHS = (
    Path("models/dqn/demo_best.pt"),
    Path("models/demo_manifest.json"),
    Path("outputs/metrics/step5/benchmark_report.json"),
    Path("outputs/trajectories/xai/step6_xai_report.json"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_hash(items: Iterable[tuple[str, str]]) -> str:
    serialized = "\n".join(
        f"{name}:{value}" for name, value in sorted(items)
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def git_commit_for(reference: str, project_root: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.as_posix()}",
            "rev-list",
            "-n",
            "1",
            reference,
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_scenario_splits(scenario_config: dict[str, Any]) -> None:
    splits = scenario_config["splits"]
    seen: dict[str, str] = {}
    for split_name, scenarios in splits.items():
        for scenario in scenarios:
            if scenario in seen:
                raise ValueError(
                    f"Scenario {scenario!r} appears in both {seen[scenario]} "
                    f"and {split_name}"
                )
            seen[scenario] = split_name
    held_out = {
        scenario
        for split_name, scenarios in splits.items()
        if split_name.startswith("held_out")
        for scenario in scenarios
    }
    development = set(splits["train"]) | set(splits["validation"])
    if held_out & development:
        raise ValueError("Held-out scenarios overlap development scenarios")


def build_v1_baseline_manifest(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    evidence_hashes = {
        path.as_posix(): file_sha256(project_root / path)
        for path in V1_EVIDENCE_PATHS
    }
    demo_manifest_path = project_root / "models/demo_manifest.json"
    with demo_manifest_path.open("r", encoding="utf-8") as stream:
        demo_manifest = json.load(stream)
    checkpoint_hash = evidence_hashes["models/dqn/demo_best.pt"]
    if checkpoint_hash != demo_manifest["frozen_checkpoint_sha256"]:
        raise RuntimeError("V1 frozen checkpoint no longer matches its manifest")

    benchmark_path = project_root / "outputs/metrics/step5/benchmark_report.json"
    xai_path = project_root / "outputs/trajectories/xai/step6_xai_report.json"
    with benchmark_path.open("r", encoding="utf-8") as stream:
        benchmark = json.load(stream)
    with xai_path.open("r", encoding="utf-8") as stream:
        xai = json.load(stream)
    recommendation = benchmark["recommended_demo_controller"]
    return {
        "manifest_version": 1,
        "project_version": "XRL-HVAC-v1",
        "git_tag": "v1-frozen",
        "git_commit": git_commit_for("v1-frozen", project_root),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evidence_hashes": evidence_hashes,
        "evidence_bundle_sha256": combined_hash(evidence_hashes.items()),
        "frozen_controller": recommendation["controller"],
        "training_seed": recommendation["training_seed"],
        "checkpoint_sha256": checkpoint_hash,
        "verified_python_tests": 75,
        "xai_steps_validated": xai["validation"]["steps_validated"],
        "xai_deterministic_replay_passed": xai["validation"][
            "deterministic_replay_passed"
        ],
        "immutable": True,
    }


def build_v2_protocol_manifest(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    config_hashes = {
        path.as_posix(): file_sha256(project_root / path)
        for path in V2_CONFIG_PATHS
    }
    protocol = load_yaml(project_root / V2_CONFIG_PATHS[0])
    scenarios = load_yaml(project_root / V2_CONFIG_PATHS[1])
    validate_scenario_splits(scenarios)
    baseline_manifest_path = (
        project_root / "outputs/v2/protocol/v1_baseline_manifest.json"
    )
    return {
        "manifest_version": 1,
        "protocol_id": protocol["protocol"]["id"],
        "simulator_version": protocol["protocol"]["simulator_version"],
        "status": protocol["protocol"]["status"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "v1_git_tag": protocol["protocol"]["v1_git_tag"],
        "v1_git_commit": git_commit_for(
            protocol["protocol"]["v1_git_tag"], project_root
        ),
        "config_hashes": config_hashes,
        "protocol_bundle_sha256": combined_hash(config_hashes.items()),
        "v1_baseline_manifest_sha256": file_sha256(baseline_manifest_path),
        "scenario_splits": scenarios["splits"],
        "seeds": protocol["seeds"],
        "acceptance_gates": protocol["acceptance_gates"],
        "held_out_policy": protocol["held_out_policy"],
        "final_test_opened": False,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
