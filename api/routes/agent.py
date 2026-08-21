"""Frozen controller metadata and deterministic inference."""

from typing import Any

from fastapi import APIRouter, Request

from api.schemas.explanation import BuildingStateInput

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/demo")
def demo_agent(request: Request) -> dict[str, Any]:
    return request.app.state.agent_service.metadata()


@router.post("/predict")
def predict(state: BuildingStateInput, request: Request) -> dict[str, Any]:
    return request.app.state.agent_service.predict(state.to_array())
