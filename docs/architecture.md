# Architecture

XRL-HVAC separates physical simulation, controller inference, explanation, evaluation, and delivery. No API route owns domain logic; routes validate input and delegate to application services.

```text
Next.js control room
        │ typed JSON / HTTP
        ▼
FastAPI schemas + routes
        │
        ├── SimulationService ──► Gymnasium HVACEnv
        │                              ├── BuildingSimulator
        │                              ├── ThermalModel / CO2Model
        │                              └── decomposed reward
        ├── AgentService ───────► SHA-verified frozen DQN
        └── ExplanationService ─► attribution + counterfactual
                                       │
                                       └── trajectory artifacts
```

## Stable model interfaces

All controllers implement `BaseAgent.predict(observation, deterministic=True) -> int`. The observation vector has nine stable positions: indoor temperature, outdoor temperature, relative humidity, occupancy, CO₂, price, time sine, time cosine, and prior HVAC action. The discrete action contract is OFF, LOW, MEDIUM, HIGH.

The DQN consumes observations scaled to `[-1, 1]`. Its 9→128→128→4 MLP contains 18,308 trainable parameters. Checkpoint loading checks the algorithm identifier; `AgentService` additionally verifies the frozen SHA-256 from `models/demo_manifest.json`.

## Evaluation boundary

The training curriculum uses Normal, Hot Day, and High Occupancy. Expensive Electricity is a validation-only scenario and Combined Stress is held out for final generalization testing. Model selection uses a balanced Energy/Cost–Comfort–CO₂ score with feasibility and reproducibility checks rather than cumulative reward alone.

## Explainability boundary

Integrated Gradients targets `Q(selected) - Q(runner-up)` using a configured reference observation. Per-feature reference ablation supplies an independent local sensitivity check. Counterfactual search prioritizes one-feature edits, then a bounded two-feature fallback. Neither method claims physical causality.

## Runtime boundary

FastAPI exposes only inference, deterministic simulation, explanations, and read-only benchmark evidence. Training is intentionally excluded from the runtime API. The frontend replays a completed episode locally, so animation never changes environment semantics or model results.
