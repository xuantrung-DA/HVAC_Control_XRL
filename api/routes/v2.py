"""Honest development-only API for XRL-HVAC V2."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.services.v2_simulation_service import DEVELOPMENT_SCENARIOS, HELD_OUT_SCENARIOS
from src.utils.config import PROJECT_ROOT


router = APIRouter(prefix="/v2", tags=["v2-development"])


class V2SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str = "normal_v2"
    seed: int = Field(default=901, ge=0, le=2_147_483_647)
    include_explanations: bool = True


@router.get("/status")
def status(request: Request) -> dict:
    training = _artifact("outputs/v2/training/dqn_development_summary.json")
    held_out = _artifact("outputs/v2/protocol/held_out_status.json")
    baseline = _artifact("outputs/v2/baselines/development_baseline_report.json")
    return {
        "simulator_version": "XRL-HVAC-v2",
        "lifecycle": "development",
        "development_status": training["development_status"],
        "official_demo_controller": "v1_frozen_dqn",
        "v2_controller": request.app.state.v2_agent_service.metadata(),
        "held_out": held_out,
        "development_gates": training["development_gates"],
        "scenario_access": {
            "development": list(DEVELOPMENT_SCENARIOS),
            "held_out_sealed": list(HELD_OUT_SCENARIOS),
        },
        "development_evidence": {
            "dqn": training["dqn_variants"],
            "baselines": baseline["controllers"],
        },
    }


@router.get("/scenarios")
def scenarios() -> dict:
    return {
        "development": [{"name": name, "runnable": True} for name in DEVELOPMENT_SCENARIOS],
        "held_out": [
            {"name": name, "runnable": False, "status": "SEALED_NOT_RUN"}
            for name in HELD_OUT_SCENARIOS
        ],
    }


@router.post("/simulations/run")
def run_simulation(payload: V2SimulationRequest, request: Request) -> dict:
    try:
        return request.app.state.v2_simulation_service.run(
            scenario=payload.scenario,
            seed=payload.seed,
            include_explanations=payload.include_explanations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _artifact(relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"V2 artifact not generated: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))
