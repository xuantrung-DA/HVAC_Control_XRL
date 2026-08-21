"""Building configuration and scenario endpoints."""

from fastapi import APIRouter

from api.schemas.building import BuildingConfigResponse, ScenarioResponse
from api.schemas.simulation import ScenarioName
from src.envs.building import HVACAction
from src.envs.hvac_env import HVACEnv
from src.envs.scenarios import list_scenarios

router = APIRouter(prefix="/building", tags=["building"])


@router.get("/scenarios", response_model=list[ScenarioResponse])
def scenarios() -> list[ScenarioResponse]:
    return [ScenarioResponse(name=item.name, description=item.description) for item in list_scenarios()]


@router.get("/config/{scenario}", response_model=BuildingConfigResponse)
def building_config(scenario: ScenarioName) -> BuildingConfigResponse:
    env = HVACEnv(scenario=scenario)
    config = env.config
    response = BuildingConfigResponse(
        scenario=scenario,
        timestep_minutes=int(config["simulation"]["timestep_minutes"]),
        steps_per_episode=env.max_steps,
        observation_names=list(env.observation_names),
        actions={int(action): action.name for action in HVACAction},
        comfort=dict(config["comfort"]),
        iaq=dict(config["iaq"]),
        hvac=dict(config["hvac"]),
        scenario_config=config,
    )
    env.close()
    return response
