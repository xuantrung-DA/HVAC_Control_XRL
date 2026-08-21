# Demo guide

## Recommended walkthrough

1. Begin with the **V1 frozen DQN** and Combined Stress. Run the 24-hour replay and explain that V1 is the evidence-backed official demo controller.
2. At a HIGH action, show signed attribution and normalized importance. State that these are local model sensitivities, not physical causes.
3. Show the verified bounded counterfactual and compare the same V1 scenario with Rule-Based control.
4. Move to the **V2 Development Lab**. Point out the visible `DEVELOPMENT FAIL` and `HELD-OUT SEALED` labels.
5. Run a development scenario. Show actual vs 1-hour forecast, risk, dynamic reward priority, and the separate `DQN proposed → shield → executed` path.
6. Use the heat-flow and energy ledgers to explain why indoor state changes and where electricity is consumed.
7. Close with the protocol story: V2 energy/cost improved, but comfort/IAQ gates failed; therefore the final test was not opened and the benchmark was not bent after seeing results.

## Recruiter-friendly summary

> I built two generations of an explainable HVAC-control system. V1 proves the end-to-end RL/XAI product. V2 adds richer building physics, forecasts, online risk, dynamic reward auditing, and a predictive shield. When no V2 policy passed the locked constraints, I preserved the failed evidence and kept held-out data sealed instead of overstating performance.

## Honest caveats

- The simulator is lightweight, single-zone, and not a substitute for EnergyPlus or commissioning data.
- V1 and V2 optimize configured objectives; stakeholder priorities can change the preferred controller.
- Local feature attribution and counterfactuals do not establish physical causality.
- The V2 discrete action map couples cooling and ventilation; this is a documented source of conflicting comfort/IAQ behavior.
- No V2 held-out generalization claim is valid until an eligible checkpoint is frozen and the one-shot final protocol is executed.
