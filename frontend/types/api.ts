export type Controller = "dqn" | "rule_based" | "fixed_thermostat" | "random";
export type Scenario =
  | "normal"
  | "hot_day"
  | "high_occupancy"
  | "expensive_electricity"
  | "combined_stress";

export interface BuildingState {
  indoor_temperature_c: number;
  outdoor_temperature_c: number;
  relative_humidity_pct: number;
  occupancy: number;
  co2_ppm: number;
  electricity_price_per_kwh: number;
  time_sin: number;
  time_cos: number;
  hvac_action: number;
}

export interface Contribution {
  feature: keyof BuildingState;
  value: number;
  reference_value: number;
  signed_contribution: number;
  signed_percentage: number;
  absolute_importance: number;
  absolute_importance_pct: number;
  direction: "supports_selected_action" | "opposes_selected_action" | "neutral";
  ablation_margin_change: number;
}

export interface Attribution {
  action: number;
  action_name: string;
  contrast_action: number;
  contrast_action_name: string;
  q_values: number[];
  decision_margin: number;
  contributions: Contribution[];
  human_readable: string;
  causal_claim: false;
}

export interface Counterfactual {
  found: boolean;
  counterfactual_action: number | null;
  counterfactual_action_name: string | null;
  normalized_l1_distance: number | null;
  changes: Array<{
    feature: keyof BuildingState;
    original_value: number;
    counterfactual_value: number;
    delta: number;
  }>;
  human_readable: string;
  action_changed: boolean;
  within_bounds: boolean;
}

export interface TrajectoryStep {
  step: number;
  timestamp: string;
  hour: number;
  state: BuildingState;
  action: number;
  action_name: string;
  reward: number;
  energy_kwh: number;
  electricity_cost: number;
  comfort_status: "comfortable" | "violation";
  co2_status: "acceptable" | "violation";
  feature_attribution?: Attribution;
  counterfactual?: Counterfactual;
}

export interface SimulationResult {
  controller: Controller;
  scenario: Scenario;
  seed: number;
  summary: Record<string, number | string | Record<string, number>>;
  trajectory: TrajectoryStep[];
}

export interface BenchmarkReport {
  recommended_demo_controller: {
    controller: string;
    selection_score: number;
    evidence: Record<string, Record<string, number | boolean>>;
  };
  split_summary: Array<Record<string, number | string>>;
}
