# Demo guide

## Recommended walkthrough

1. Begin with the **V1 frozen DQN** and Combined Stress. Run the 24-hour replay and explain that V1 is the evidence-backed official demo controller.
2. At a HIGH action, show signed attribution and normalized importance. State that these are local model sensitivities, not physical causes.
3. Show the verified bounded counterfactual and compare the same V1 scenario with Rule-Based control.
4. Move to the **V2 closed iteration**. Point out the visible `FINAL GATE FAIL` and `OPENED ONCE` labels.
5. Run a development scenario. Show actual vs 1-hour forecast, risk, dynamic reward priority, and the separate `DQN proposed → shield → executed` path.
6. Use the heat-flow and energy ledgers to explain why indoor state changes and where electricity is consumed.
7. Close with the protocol story: the hybrid candidate passed development and reduced final energy/cost, but failed Combined Stress comfort. The test was opened once, the failure was retained, and the benchmark was not bent after seeing results.

## Recruiter-friendly summary

> I built two generations of an explainable HVAC-control system. V1 proves the end-to-end RL/XAI product. V2 adds richer building physics, forecasts, online risk, dynamic reward auditing, and learning-augmented actuator control. Its frozen hybrid candidate passed development but failed the one-shot Combined Stress comfort gate, so I preserved the failure instead of tuning on the test.

## Honest caveats

- The simulator is lightweight, single-zone, and not a substitute for EnergyPlus or commissioning data.
- V1 and V2 optimize configured objectives; stakeholder priorities can change the preferred controller.
- Local feature attribution and counterfactuals do not establish physical causality.
- The final hybrid design decouples cooling, ventilation, and dehumidification, but its current actuator capacity is insufficient under Combined Stress.
- Combined Stress is now observed and may be used only as a regression case, never for V3 model selection.
