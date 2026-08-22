# V2 results — closed iteration

V2 is complete as an engineering iteration. Its hybrid candidate passed development validation, was frozen with a SHA-256 manifest, and then failed the one-shot Combined Stress comfort gate. V1 remains the official demo controller.

## Locked objective

The final gate required energy and cost at or below an actuator-matched Rule-Based controller, comfort violation below 5%, CO₂ violation below 1%, and zero critical safety violations. Constraints precede reward in model selection.

Development scenarios were Expensive Electricity, Meeting Surge, High Electronics Load, and Cleaning Event with seeds 901–903. The final Combined Stress run used the predeclared seeds 1701–1705. Unexpected Surge, Forecast Failure, Heatwave, and Door Left Open remain unopened.

## Root-cause investigation

The original V2 action coupled sensible cooling and ventilation. Controlled ablations found that occupant latent moisture and infiltration created a physical conflict: ventilation protected CO₂ but imported heat/moisture, while extra cooling could overcool the zone without sufficient latent removal.

The resulting learning-augmented architecture separates:

- DQN sensible-cooling proposal;
- deterministic thermal guard;
- deterministic IAQ ventilation;
- independent, separately metered dehumidification.

The dehumidifier is configured at 8 kg/h and 2.5 kW. Its energy is included in whole-building energy, controllable energy, peak power, and Time-of-Use cost.

## Clean development benchmark

| Controller | Energy | Cost | Comfort | CO₂ | Critical safety |
|---|---:|---:|---:|---:|---:|
| Matched Rule-Based | 401.62 kWh | 90.86 | 5.00% | 0.00% | 0 |
| **DQN seed 42 + hybrid guard** | **396.98 kWh** | **90.73** | **1.11%** | **0.00%** | **0** |
| DQN seed 123 + hybrid guard | 398.21 kWh | 93.27 | 4.26% | 0.00% | 0 |
| DQN seed 2026 + hybrid guard | 405.43 kWh | 91.71 | 0.74% | 0.00% | 0 |

Only seed 42 passed every gate against the matched baseline and was selected by the locked order: constraints, safety, energy, cost, then comfort. The clean benchmark was executed twice; the timestamp-independent result hash was identical.

## Freeze

- Candidate: `seed_42_full_best.pt`
- Checkpoint SHA-256: `06b0aede0b4e5ca91f4c5976fe493f3be6f4a6a62e24cf136291a560ac2429e0`
- Frozen components: 28
- Frozen bundle SHA-256: `1fa47a571bd99a15168b2c45ea6183156da20753253835e6397eed15f3bd9e73`
- Verification: 149 Python tests passed

The simulator, actuator configuration, guard thresholds, checkpoint, evaluator, reward, forecast model, scenario protocol, and one-shot runner were hashed before opening Combined Stress.

## One-shot Combined Stress — FINAL FAIL

| Controller | Energy | Cost | Comfort | Temperature | Humidity | CO₂ | Critical safety |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched Rule-Based | 586.02 kWh | 221.10 | 88.84% | 83.93% | 83.04% | 0.00% | 0 |
| **Frozen DQN + hybrid guard** | **573.08 kWh** | **218.47** | **98.67%** | **97.78%** | **82.15%** | **0.00%** | **0** |

Gate result:

- Energy ≤ Rule-Based: **PASS**
- Cost ≤ Rule-Based: **PASS**
- Comfort < 5%: **FAIL**
- CO₂ < 1%: **PASS**
- Critical safety = 0: **PASS**

The frozen DQN saved 2.21% energy and 1.19% cost but did not preserve comfort. The matched baseline also failed comfort severely, and both controllers spent roughly 45–47% of the episode at HIGH cooling while the dehumidifier ran about 40–41%. This supports an actuator-capacity/design-envelope limitation. Because DQN comfort was worse than the matched baseline, policy generalization remains a secondary failure rather than being excused by hardware limits.

Combined Stress is now observed and cannot be reused for tuning or V3 model selection. Its one-shot receipt is immutable evidence. The remaining four held-out scenarios were not opened.

## Evidence

- [`development_benchmark.json`](../outputs/v2/hybrid/development_benchmark.json)
- [`frozen_candidate_manifest.json`](../outputs/v2/hybrid/frozen_candidate_manifest.json)
- [`combined_stress_one_shot.json`](../outputs/v2/hybrid/combined_stress_one_shot.json)
- [`held_out_status.json`](../outputs/v2/protocol/held_out_status.json)

The local failure-diagnosis search space is intentionally excluded from Git. The production hybrid implementation and final evidence are retained.
