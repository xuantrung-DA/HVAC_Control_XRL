"""Write V2 Gymnasium schema and V1 compatibility evidence."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn import DQNAgent  # noqa: E402
from src.envs.hvac_env import HVACEnv  # noqa: E402
from src.envs.v2 import (  # noqa: E402
    V1AgentOnV2Adapter,
    V1ObservationAdapter,
    V2HVACEnv,
)
from src.utils.v2_manifest import file_sha256  # noqa: E402


def main() -> None:
    v2 = V2HVACEnv("normal_v2")
    observation, info = v2.reset(seed=42)
    v1 = HVACEnv(scenario="normal")
    agent = DQNAgent(v1.observation_space, v1.action_space)
    checkpoint = PROJECT_ROOT / "models/dqn/demo_best.pt"
    checksum_before = file_sha256(checkpoint)
    agent.load(str(checkpoint))
    adapter = V1AgentOnV2Adapter(
        agent, V1ObservationAdapter(v1.observation_space)
    )
    actions = []
    started = time.perf_counter()
    terminated = False
    while not terminated:
        action = adapter.predict(observation, deterministic=True)
        actions.append(action)
        observation, _, terminated, _, info = v2.step(action)
    duration = time.perf_counter() - started
    checksum_after = file_sha256(checkpoint)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": "XRL-HVAC-v2",
        "observation_schema": info["observation_schema"],
        "observation_size": len(v2.observation_names),
        "observation_names": list(v2.observation_names),
        "physical_units_emitted": True,
        "training_reward_authorized": True,
        "v1_adapter": {
            "checkpoint": "models/dqn/demo_best.pt",
            "sha256_before": checksum_before,
            "sha256_after": checksum_after,
            "checkpoint_unchanged": checksum_before == checksum_after,
            "episode_steps": len(actions),
            "unique_actions": sorted(set(actions)),
            "deterministic_inference": True,
        },
        "performance": {
            "episode_seconds": duration,
            "steps_per_second_with_forecast_risk_v1_dqn": len(actions) / duration,
        },
        "checks": {
            "observation_size_is_34": len(v2.observation_names) == 34,
            "v1_checkpoint_unchanged": checksum_before == checksum_after,
            "v1_agent_completed_episode": len(actions) == 96,
            "reward_is_auditable_and_authorized": info["reward_status"]
            == "AUTHORIZED_AUDITABLE_V2_REWARD"
            and info["reward_audit"]["profile_id"] == "reward_profile_v2_001",
        },
    }
    report["all_checks_passed"] = all(report["checks"].values())
    output = PROJECT_ROOT / "outputs/v2/validation/environment_integration_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "all_checks_passed": report["all_checks_passed"],
                "observation_size": report["observation_size"],
                "v1_checkpoint_unchanged": checksum_before == checksum_after,
                "episode_steps": len(actions),
                "steps_per_second": report["performance"]["steps_per_second_with_forecast_risk_v1_dqn"],
            },
            indent=2,
        )
    )
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
