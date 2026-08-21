"""Ensure failed development work cannot silently open held-out scenarios."""

from __future__ import annotations

import json

from src.utils.config import PROJECT_ROOT


def test_held_out_receipt_remains_sealed_without_candidate() -> None:
    receipt = json.loads(
        (PROJECT_ROOT / "outputs/v2/protocol/held_out_status.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (PROJECT_ROOT / "outputs/v2/protocol/v2_protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "SEALED_NOT_RUN"
    assert receipt["candidate_checkpoint"] is None
    assert receipt["final_test_opened"] is False
    assert manifest["final_test_opened"] is False
    assert "combined_stress_v2" in receipt["sealed_scenarios"]
