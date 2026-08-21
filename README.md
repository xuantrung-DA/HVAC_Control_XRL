# XRL-HVAC

**Explainable Reinforcement Learning for Smart Building HVAC Control**

XRL-HVAC is a portfolio-grade AI engineering system in which a reinforcement-learning controller operates a lightweight smart-building digital twin. The frozen demo DQN balances energy cost, occupant comfort, indoor air quality, and control stability—and explains every decision with local feature attribution and verified counterfactuals.

> Traditional controllers optimize energy reasonably well, but DQN learns a more balanced HVAC policy that generalizes to unseen stress conditions while preserving occupant comfort and air quality.

## Why this project is interesting

This is not a UI wrapped around an untested model. The repository includes a seeded Gymnasium environment, traditional and learned controllers, curriculum training, multi-seed held-out evaluation, checkpoint integrity checks, practical DQN explainability, a typed FastAPI contract, and an interactive Next.js dashboard.

On the held-out `combined_stress` scenario, the selected DQN uses approximately 15% more energy than Rule-Based control but reduces comfort violation by about 99% and CO₂ violation by 100%.

| Controller | Reward ↓ penalty | Energy | Comfort violation | CO₂ violation | Interpretation |
|---|---:|---:|---:|---:|---|
| Rule-Based | -112.39 | 64.05 kWh | 33.33% | 6.25% | Energy-efficient, constraint violations |
| DQN | **-25.18** | 73.73 kWh | **0.35%** | **0.00%** | Best balanced feasible trade-off |
| Double DQN | -25.21 | 77.49 kWh | **0.00%** | **0.00%** | Strong, but higher energy |
| PPO | -109.30 | **59.49 kWh** | 35.07% | **0.00%** | Energy-oriented policy |

Values are mean results across training/evaluation seeds on the held-out test split. Full evidence is in `outputs/metrics/step5/benchmark_report.json`.

## System overview

```text
Weather · Occupancy · Price
            │
            ▼
  Gymnasium building twin ──────► Temperature · CO₂ · Energy
            │                                  │
            ▼                                  ▼
 OFF · LOW · MEDIUM · HIGH ◄──── Frozen DQN + reward
            │
            ├── Integrated Gradients attribution
            ├── Bounded counterfactual search
            └── Step-level trajectory explanations
                         │
                         ▼
                FastAPI → Next.js dashboard
```

Controllers progress from `Random → Thermostat → Rule-Based → Q-Learning → DQN → Double DQN → PPO`. One episode is 24 hours with 96 15-minute decisions. Training uses Normal, Hot Day, and High Occupancy; Expensive Electricity is validation; Combined Stress is held out.

## Quick start

Requirements: Python 3.12+, Node.js 24+, and 16 GB RAM. CUDA is optional; the demo checkpoint is intentionally small (18,308 parameters) and runs on CPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\start_demo.py
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. API documentation is available at `http://localhost:8000/docs`.

Or run both services with Docker:

```bash
docker compose up --build
```

## Explainability

- **Feature attribution:** Integrated Gradients over the selected-action Q-value margin versus its runner-up. Both signed contribution and normalized absolute importance are returned.
- **Counterfactual:** sparse bounded grid search over indoor temperature, occupancy, CO₂, and electricity price. A result is returned only when a real DQN forward pass confirms the action changed.
- **Trajectory:** state, action, reward components, energy, status, attribution, and counterfactual for every simulation step.
- **Faithfulness:** completeness, reference ablation, deterministic replay, bounds, and action-flip checks. Outputs explicitly avoid causal language.

Across 384 Step 6 decisions, action-flip validity and counterfactual bounds validity were 100%; mean absolute attribution–ablation correlation was 0.723.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm run typecheck
npm run build
npm audit
```

The current implementation passes 75 Python tests, strict TypeScript checks, a production Next.js build, and an npm audit with zero known vulnerabilities.

## Repository map

- `src/envs` — deterministic-by-seed building physics, scenarios, reward
- `src/agents` / `src/baselines` — learned and traditional controllers
- `src/xai` — attribution, counterfactual, trajectory explanations
- `src/evaluation` / `training` — multi-objective benchmark pipeline
- `src/services` / `api` — application layer and HTTP contract
- `frontend` — themed interactive control room
- `configs` — environment, algorithm, evaluation, and XAI settings
- `tests` — unit, contract, reproducibility, and integration coverage

See [architecture](docs/architecture.md), [setup](docs/setup.md), and [demo guide](docs/demo.md) for details.

## Scope and limitations

The simulator is a transparent single-zone approximation, not EnergyPlus. XAI explains the learned function locally and is not proof of physical causality. Counterfactual states respect feature bounds but are not guaranteed to be reachable from every prior trajectory. These constraints are intentional: the project prioritizes reproducible engineering and an understandable end-to-end AI product on a consumer laptop.
