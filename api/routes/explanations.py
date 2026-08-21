"""Local DQN decision explanation endpoint."""

from fastapi import APIRouter, Request

from api.schemas.explanation import DecisionExplanationRequest, DecisionExplanationResponse

router = APIRouter(prefix="/explanations", tags=["explanations"])


@router.post("/decision", response_model=DecisionExplanationResponse)
def explain_decision(payload: DecisionExplanationRequest, request: Request) -> DecisionExplanationResponse:
    result = request.app.state.explanation_service.explain(
        payload.state.to_array(), include_counterfactual=payload.include_counterfactual
    )
    return DecisionExplanationResponse(**result)
