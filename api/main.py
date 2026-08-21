"""FastAPI application for the XRL-HVAC interactive demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import agent, building, explanations, metrics, simulation
from src.services import AgentService, ExplanationService, SimulationService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    agent_service = AgentService()
    explanation_service = ExplanationService(agent_service)
    app.state.agent_service = agent_service
    app.state.explanation_service = explanation_service
    app.state.simulation_service = SimulationService(agent_service, explanation_service)
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="XRL-HVAC API",
        summary="Explainable reinforcement learning for smart-building HVAC control.",
        version="1.0.0",
        lifespan=lifespan,
    )
    origins = os.getenv("XRL_HVAC_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @application.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "healthy", "project": "XRL-HVAC", "version": "1.0.0"}

    prefix = "/api/v1"
    application.include_router(building.router, prefix=prefix)
    application.include_router(agent.router, prefix=prefix)
    application.include_router(simulation.router, prefix=prefix)
    application.include_router(metrics.router, prefix=prefix)
    application.include_router(explanations.router, prefix=prefix)
    return application


app = create_app()
