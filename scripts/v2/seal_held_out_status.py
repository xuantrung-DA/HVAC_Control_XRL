"""Record why the one-shot V2 held-out benchmark remains sealed."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.v2_manifest import file_sha256  # noqa: E402


def main() -> None:
    dqn_path = PROJECT_ROOT / "outputs/v2/training/dqn_development_summary.json"
    sac_path = PROJECT_ROOT / "outputs/v2/training/sac_go_no_go_summary.json"
    dqn = json.loads(dqn_path.read_text(encoding="utf-8"))
    sac = json.loads(sac_path.read_text(encoding="utf-8"))
    if dqn["development_status"] == "PASS" or sac["development_status"] == "PASS":
        raise RuntimeError("A passing candidate exists; use the one-shot held-out runner")
    receipt = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol_id": "xrl_hvac_v2_001",
        "status": "SEALED_NOT_RUN",
        "final_test_opened": False,
        "candidate_checkpoint": None,
        "reason": (
            "No DQN or SAC candidate passed locked development comfort and IAQ gates; "
            "opening held-out scenarios would violate the predeclared protocol and invite test tuning."
        ),
        "development_evidence": {
            "dqn_status": dqn["development_status"],
            "dqn_sha256": file_sha256(dqn_path),
            "sac_status": sac["development_status"],
            "sac_sha256": file_sha256(sac_path),
        },
        "sealed_scenarios": [
            "combined_stress_v2",
            "unexpected_occupancy_surge_v2",
            "forecast_failure_v2",
            "heatwave_v2",
            "door_left_open_v2"
        ],
        "acceptance_result": "NOT_ELIGIBLE_FOR_FINAL_TEST",
    }
    output = PROJECT_ROOT / "outputs/v2/protocol/held_out_status.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "final_test_opened": False,
        "acceptance_result": receipt["acceptance_result"],
    }, indent=2))


if __name__ == "__main__":
    main()
