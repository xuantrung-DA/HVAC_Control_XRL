"""Freeze the passing hybrid candidate and every final-test dependency."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.v2_manifest import combined_hash, file_sha256, write_json  # noqa: E402


FROZEN_COMPONENTS = (
    "configs/v2/environment.yaml",
    "configs/v2/action_mapping.yaml",
    "configs/v2/hybrid_control.yaml",
    "configs/v2/protocol.yaml",
    "configs/v2/scenarios.yaml",
    "configs/v2/controllers.yaml",
    "configs/v2/forecasting.yaml",
    "configs/v2/risk.yaml",
    "configs/v2/training.yaml",
    "configs/reward_profiles/reward_profile_v2_001.json",
    "outputs/v2/forecasting/forecast_model.json",
    "outputs/v2/training/dqn_development_summary.json",
    "outputs/v2/hybrid/development_benchmark.json",
    "src/envs/v2/models.py",
    "src/envs/v2/physics.py",
    "src/envs/v2/hybrid_physics.py",
    "src/envs/v2/hvac_env.py",
    "src/envs/v2/hybrid_env.py",
    "src/envs/v2/profiles.py",
    "src/envs/v2/observation.py",
    "src/envs/v2/reward.py",
    "src/shields/hybrid_guard.py",
    "src/baselines/v2_controllers.py",
    "src/evaluation/v2_hybrid.py",
    "scripts/v2/benchmark_hybrid_development.py",
    "scripts/v2/freeze_hybrid_candidate.py",
    "scripts/v2/run_hybrid_combined_stress_once.py",
)


def main() -> None:
    development_path = PROJECT_ROOT / "outputs/v2/hybrid/development_benchmark.json"
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if development["held_out_used"] or not development["development_pass"]:
        raise RuntimeError("Only a passing, development-only candidate may be frozen")
    selected = development["selected_candidate"]
    checkpoint = PROJECT_ROOT / selected["checkpoint"]
    if file_sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint does not match development evidence")
    component_paths = (*FROZEN_COMPONENTS, selected["checkpoint"])
    hashes = {
        relative: file_sha256(PROJECT_ROOT / relative)
        for relative in component_paths
    }
    manifest = {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "project": "XRL-HVAC",
        "architecture": "learning_augmented_hvac_control",
        "simulator_version": development["simulator_version"],
        "candidate": selected,
        "development_benchmark_sha256": hashes[
            "outputs/v2/hybrid/development_benchmark.json"
        ],
        "component_hashes": hashes,
        "frozen_bundle_sha256": combined_hash(hashes.items()),
        "verified_python_tests": 149,
        "held_out_scenario": "combined_stress_v2",
        "held_out_seeds": [1701, 1702, 1703, 1704, 1705],
        "held_out_opened": False,
        "immutable_after_freeze": True,
        "local_only": True,
    }
    output = PROJECT_ROOT / "outputs/v2/hybrid/frozen_candidate_manifest.json"
    write_json(output, manifest)
    print(json.dumps({
        "candidate": selected,
        "components": len(hashes),
        "frozen_bundle_sha256": manifest["frozen_bundle_sha256"],
        "output": output.relative_to(PROJECT_ROOT).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
