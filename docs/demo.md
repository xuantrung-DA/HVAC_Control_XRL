# Demo guide

## Recommended walkthrough

1. Start on Combined Stress with the frozen DQN. Explain that weather, occupancy, and peak pricing were combined only in the held-out test.
2. Run the simulation and play the 96-step timeline. Point out that building airflow and dashboard metrics follow the selected timestep rather than a decorative animation.
3. At a HIGH action, show signed feature contributions and normalized importance. State clearly that these are local model attributions, not physical causes.
4. Show the counterfactual card. The proposed feature edit is bounded, and the target action was verified with another DQN forward pass.
5. Move to the evidence panel: DQN spends roughly 15% more energy than Rule-Based but removes almost all comfort violation and all measured CO₂ violation on held-out stress.
6. Switch to Rule-Based to demonstrate that the same simulator and metrics contract supports traditional control.

## Recruiter-friendly summary

“I built a reproducible building digital twin, trained and benchmarked four RL families against traditional control, selected a DQN from multi-seed held-out evidence, then made its decisions inspectable with attribution and validated counterfactuals. The result is delivered through a typed API and an interactive dashboard, with Docker and automated tests.”

## Honest caveats

- The simulator is lightweight and single-zone; it does not replace EnergyPlus validation.
- The selected policy optimizes a configured multi-objective reward, so different stakeholder weights can change the preferred controller.
- Integrated Gradients is reference-dependent and local.
- Counterfactual feature vectors can be valid yet dynamically unreachable from a particular history.
