"""Read-only benchmark and XAI evidence endpoints."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.utils.config import PROJECT_ROOT

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/benchmark")
def benchmark() -> dict[str, Any]:
    return _read_artifact(PROJECT_ROOT / "outputs" / "metrics" / "step5" / "benchmark_report.json")


@router.get("/xai")
def xai_validation() -> dict[str, Any]:
    return _read_artifact(PROJECT_ROOT / "outputs" / "trajectories" / "xai" / "step6_xai_report.json")


def _read_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not generated: {path.name}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
