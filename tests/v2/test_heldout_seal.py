"""Ensure the one-shot hybrid result and remaining V2 seal stay explicit."""

from __future__ import annotations

import json

from src.utils.config import PROJECT_ROOT


def test_combined_stress_is_opened_once_and_other_scenarios_remain_sealed() -> None:
    receipt = json.loads(
        (PROJECT_ROOT / "outputs/v2/protocol/held_out_status.json").read_text(
            encoding="utf-8"
        )
    )
    one_shot = json.loads(
        (PROJECT_ROOT / "outputs/v2/hybrid/combined_stress_one_shot.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "PARTIALLY_OPENED_HYBRID_COMBINED_STRESS"
    assert receipt["candidate_checkpoint"].endswith("seed_42_full_best.pt")
    assert receipt["final_test_opened"] is True
    assert one_shot["status"] == "COMPLETED_FAIL"
    assert one_shot["rerun_permitted"] is False
    assert one_shot["acceptance_pass"] is False
    assert "combined_stress_v2" not in receipt["remaining_sealed_scenarios"]
    assert len(receipt["remaining_sealed_scenarios"]) == 4
