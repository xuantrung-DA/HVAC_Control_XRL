"""Full-day deterministic simulation endpoint."""

from fastapi import APIRouter, HTTPException, Request

from api.schemas.simulation import SimulationRequest, SimulationResponse

router = APIRouter(prefix="/simulations", tags=["simulations"])


@router.post("/run", response_model=SimulationResponse)
def run_simulation(payload: SimulationRequest, request: Request) -> SimulationResponse:
    try:
        result = request.app.state.simulation_service.run(
            controller_name=payload.controller,
            scenario=payload.scenario,
            seed=payload.seed,
            include_explanations=payload.include_explanations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SimulationResponse(**result)
