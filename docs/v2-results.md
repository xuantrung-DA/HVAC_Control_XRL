# V2 development results

## Locked objective

Before training, V2 required whole-building energy and electricity cost at or below the frozen Rule-Based V2 baseline, comfort violation below 5%, CO₂ violation below 1%, zero critical safety violations, reproducible checkpoints, and no action collapse. Constraint gates precede cumulative reward in controller selection.

Training scenarios: Normal, Hot Day, High Occupancy, High Humidity. Validation scenarios: Expensive Electricity, Meeting Surge, High Electronics Load, Cleaning Event. Combined Stress and four resilience/safety scenarios remain sealed.

## Multi-seed DQN validation

Across training seeds 42, 123, and 2026:

| Variant | Energy, kWh | Cost | Comfort violation | CO₂ violation | Shield intervention |
|---|---:|---:|---:|---:|---:|
| Rule-Based V2 | 423.32 ± 25.99 | 94.38 ± 23.88 | 39.44 ± 5.53% | 0.00% | 0.00% |
| DQN without shield | **401.20 ± 4.85** | **88.70 ± 1.64** | **15.93 ± 2.64%** | 3.67 ± 2.20% | 0.00% |
| DQN with shield | 409.36 ± 2.47 | 91.70 ± 1.01 | 32.84 ± 4.55% | **0.00%** | 14.44 ± 1.47% |

The selected development checkpoint is seed 2026 at 15,000 steps. Its development metrics were 403.03 kWh, 12.22% comfort violation, and 1.04% CO₂ violation. It is reproducible and avoids single-action collapse, but it is not acceptance-eligible.

## SAC go/no-go

SAC decoupled cooling and ventilation after DQN evidence justified continuous action research. The first locked seed was stopped after 20,000 steps: comfort violation was 73.15% and CO₂ violation 2.17%. Remaining seeds were not run, consistent with the predeclared go/no-go rule.

## Equal-budget ablations

At 10,000 steps on seed 2026 / validation seed 901:

| Variant | Energy, kWh | Comfort violation | CO₂ violation |
|---|---:|---:|---:|
| Full dynamic V2 | 397.93 | 22.22% | 9.64% |
| Fixed reward | 417.66 | 22.78% | 7.55% |
| No forecast | 435.74 | 68.33% | 9.64% |
| No trend | 402.59 | 21.11% | 8.59% |
| No risk | 437.25 | 20.00% | 12.50% |

Forecast and risk features were directionally valuable, but no ablation passed all gates. These runs support diagnosis, not a final controller claim.

## Protocol decision

`outputs/v2/protocol/held_out_status.json` records `SEALED_NOT_RUN`, `final_test_opened=false`, and `candidate_checkpoint=null`. This is the final V2 MVP outcome: the system and evaluation protocol are complete, while controller performance remains an explicit development failure.

Representative policy/shield explanations from three development scenarios are stored as JSON and CSV under `outputs/v2/xai`. They were generated without accessing any held-out scenario.
