# XRL-HVAC

**Explainable Reinforcement Learning for Smart Building HVAC Control**

XRL-HVAC is a portfolio-grade AI engineering project that combines a seeded Gymnasium building simulator, traditional and learned HVAC controllers, practical explainability, FastAPI, and an interactive Next.js dashboard.

The repository contains two evidence tracks:

- **V1 — frozen official demo:** a compact DQN with validated held-out generalization and step-level explanations.
- **V2 — development laboratory:** a richer 2R1C physics simulator with forecasts, online risk, dynamic rewards, detailed energy accounting, and a predictive safety shield. No V2 policy passed every locked development gate, so the final held-out split remains sealed and V1 remains the official controller.

That distinction is intentional. Failed acceptance gates are reported rather than hidden or tuned away.

## Engineering story

V1 demonstrated that a learned controller can produce a stronger energy–comfort–IAQ balance than conventional control on the original simulator. On held-out Combined Stress, the frozen DQN spent about 15% more energy than Rule-Based while reducing comfort violation by about 99% and measured CO₂ violation by 100%.

V2 then made the task materially harder and more realistic:

```text
Weather + occupancy + price + events
                  |
                  v
    2R1C building physics + humidity + CO2
                  |
          current state + forecast
                  |
                  v
       trend / reliability / risk features
                  |
                  v
       DQN proposal ---> safety shield
                            |
                            v
             executed physical HVAC action
                            |
                            v
       heat flows + energy ledger + reward audit
```

The best experimental V2 DQN met energy and cost gates but missed the predeclared comfort and IAQ gates. SAC was tested only after discrete-action evidence justified decoupled continuous cooling and ventilation; it also failed the development gate and was stopped under the declared go/no-go rule.

## Evidence at a glance

### V1 frozen demo — held-out evidence

| Controller | Reward | Energy | Comfort violation | CO₂ violation | Interpretation |
|---|---:|---:|---:|---:|---|
| Rule-Based | -112.39 | 64.05 kWh | 33.33% | 6.25% | Efficient, but violates constraints |
| **DQN** | **-25.18** | 73.73 kWh | **0.35%** | **0.00%** | Best balanced V1 policy |
| Double DQN | -25.21 | 77.49 kWh | **0.00%** | **0.00%** | Strong, higher energy |
| PPO | -109.30 | **59.49 kWh** | 35.07% | **0.00%** | Energy-oriented policy |

### V2 development evidence — held-out not opened

| Controller | Whole-building energy | Cost | Comfort violation | CO₂ violation | Gate status |
|---|---:|---:|---:|---:|---|
| Rule-Based V2 | 423.32 kWh | 94.38 | 39.44% | **0.00%** | Baseline |
| Experimental DQN, no shield | **401.20 kWh** | **88.70** | **15.93%** | 3.67% | **FAIL** |
| Experimental DQN + shield | 409.36 kWh | 91.70 | 32.84% | **0.00%** | **FAIL** |

V2 acceptance was locked before training: energy and cost at or below Rule-Based, comfort violation below 5%, CO₂ violation below 1%, zero critical safety violations, reproducible checkpoints, and no action collapse. The shield improved CO₂ but worsened temperature/humidity comfort because V2 discrete actions still couple cooling and ventilation.

See [`docs/v2-results.md`](docs/v2-results.md) and the machine-readable files under `outputs/v2` for the full audit trail.

## What is implemented

- Deterministic 24-hour episodes with 96 fifteen-minute timesteps.
- V1 controllers: Random, Fixed Thermostat, Rule-Based, Q-Learning, DQN, Double DQN, and PPO.
- V2 2R1C physics: envelope/windows, thermal mass, solar, occupants, electronics, lighting, cleaning, infiltration, ventilation, humidity, CO₂, HVAC inertia, nonlinear COP, and time-of-use cost.
- Correlated weather and realistic seeded schedules/events.
- Leakage-resistant 1–4 hour forecasts with uncertainty and reliability tracking.
- EWMA/CUSUM monitoring, bounded risk features, dynamic/fixed/CMDP reward modes, and timestep reward audits.
- Predictive shield with `ALLOW`, `CLAMP`, `REJECT`, and `FALLBACK` decisions.
- Separate explanations for the learned proposal and the deterministic shield action.
- SHA-verified checkpoints, multi-seed evaluation, ablations, seal receipts, tests, Docker, API, and themed dashboard.

## Quick start on Windows

Requirements: Python 3.12+, Node.js 24+, and 16 GB RAM. CUDA is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\start_demo.py
```

In a second PowerShell terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:3000`; API docs are at `http://localhost:8000/docs`.

Using the virtual-environment executable is deliberate: it prevents Uvicorn from resolving packages from a different global Python installation. `scripts/start_demo.py` also inserts the repository root before importing `api`, fixing the common `ModuleNotFoundError: No module named 'api'` issue.

## Docker

```powershell
docker compose up --build
```

The container uses CPU inference by default. Frontend: `http://localhost:3000`; backend: `http://localhost:8000`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm.cmd run typecheck
npm.cmd run build
```

Current release verification: **146 Python tests passed**, strict TypeScript passed, and the Next.js production build passed. The remaining three Python warnings are upstream deprecation warnings.

## API surfaces

- `/api/v1/*` — frozen official V1 demo, benchmark, and explanations.
- `/api/v2/status` — development gates, artifact-backed status, and held-out seal state.
- `/api/v2/scenarios` — runnable development scenarios and non-runnable sealed scenarios.
- `/api/v2/simulations/run` — deterministic development-only V2 replay with policy XAI, shield XAI, forecasts, risk, reward audit, heat flow, and energy ledger.

The V2 API rejects held-out scenario execution with HTTP 422 while no candidate is eligible.

## Repository map

- `src/envs` — V1 simulator and `src/envs/v2` physics/forecast-aware environment.
- `src/agents`, `src/baselines`, `src/shields` — learned, traditional, and guarded control.
- `src/forecasting`, `src/risk`, `src/xai` — proactive context and explanation layers.
- `src/evaluation`, `training`, `scripts/v2` — benchmarks, ablations, and training workflows.
- `api`, `src/services` — typed HTTP/application layer.
- `frontend` — orange–pink–purple themed control room and V2 development lab.
- `configs/reward_profiles`, `configs/v2` — versioned experiment and protocol configuration.
- `models/v2`, `outputs/v2` — isolated V2 checkpoints and evidence; V1 is preserved.

## Scope and limitations

This is a transparent single-zone engineering simulator, not an EnergyPlus replacement or a production building-management controller. V1 explanations and V2 local Q-margin ablations explain model sensitivity, not physical causality. V2 still exposes the limitations of coupled discrete control, and no claim is made about V2 held-out generalization because the final split has never been opened.

More detail: [architecture](docs/architecture.md), [setup](docs/setup.md), [demo guide](docs/demo.md), and [V2 results](docs/v2-results.md).
