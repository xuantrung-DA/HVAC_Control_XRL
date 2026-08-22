# Architecture

XRL-HVAC keeps simulation, control, explanation, evaluation, and delivery independent. HTTP routes validate inputs and delegate to services; training is never exposed through the runtime API.

```text
Next.js dashboard
        | typed JSON
        v
FastAPI routes
        +-- V1 services ---> frozen DQN + V1 Gymnasium environment
        +-- V2 services ---> SHA-verified experimental DQN
                                  | proposed action
                                  v
                          predictive safety shield
                                  | executed action
                                  v
       scenario --> forecast --> risk --> V2 Gymnasium environment
                                         |
                                         v
                         2R1C physics + reward audit
```

## V1 boundary

All V1 controllers implement `BaseAgent.predict(observation, deterministic=True) -> int`. The stable nine-feature observation contains indoor/outdoor temperature, relative humidity, occupancy, CO₂, price, time sine/cosine, and prior action. Actions are `OFF`, `LOW`, `MEDIUM`, and `HIGH`.

The frozen 9→128→128→4 DQN has 18,308 parameters. `AgentService` verifies its SHA-256 against `models/demo_manifest.json` before inference.

## V2 physical boundary

`V2HVACEnv` exposes a 35-feature observation schema (`xrl_hvac_v2_obs_002`) with current state, 1h/4h forecasts, uncertainty, online trends, risk, and reliability. The underlying 2R1C simulator accounts separately for:

- opaque envelope and window transfer;
- infiltration and controlled ventilation;
- solar, occupant, electronics, lighting, base, and cleaning gains;
- delayed cooling delivery and nonlinear efficiency;
- indoor moisture and occupancy/airflow-coupled CO₂;
- HVAC, fan, lights, electronics, base-load energy, peak power, and TOU cost.

V1 is not overwritten: V2 lives under versioned configs, models, outputs, tests, and environment modules.

## Control and safety boundary

The original V2 learned DQN proposed a coupled discrete action. `PredictiveSafetyShield` independently evaluated constraints and recorded one of `ALLOW`, `CLAMP`, `REJECT`, or `FALLBACK`; both proposed and executed actions remain available in the development XAI artifacts.

The final hybrid iteration reuses the frozen DQN as a **sensible-cooling proposer**. `HybridControlGuard` clamps cooling at thermal bounds, chooses ventilation independently from CO₂/risk, and operates an independently metered dehumidifier from relative humidity. The architecture is learning-augmented: deterministic control retains physical constraint authority.

Continuous SAC was implemented only after development evidence showed that discrete actions coupled cooling and ventilation in conflicting ways. It did not pass the go/no-go gate and is not exposed as a demo controller.

## Explainability boundary

V1 uses Integrated Gradients, ablation checks, and bounded counterfactual search. V2 uses local selected-vs-runner-up Q-margin ablation across all 35 features and a bounded one-feature counterfactual search over interpretable physical inputs.

Policy and shield explanations are separate. The policy explanation is associational and explicitly sets `causal_claim=false`; the shield explanation is grounded in deterministic constraint evaluation but is also not labeled causal.

## Evaluation boundary

V2 train, validation, and held-out scenarios are disjoint. Model selection used development gates in this order: critical safety, comfort/CO₂, energy/cost, resilience, then Pareto/reward. The hybrid seed-42 candidate passed development, was frozen with 28 component hashes, and failed the one-shot Combined Stress comfort gate. That scenario is now observed and cannot be reused for tuning; four resilience/safety scenarios remain sealed.

## Runtime boundary

The frontend requests a deterministic 96-step trajectory, then replays it locally. Animation cannot change simulator state or model outputs. V1 remains the official demo. V2 is retained as an auditable closed iteration rather than promoted as a replacement controller.
