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

export type V2Scenario =
  | "normal_v2"
  | "hot_day_v2"
  | "high_occupancy_v2"
  | "high_humidity_v2"
  | "expensive_electricity_v2"
  | "meeting_surge_v2"
  | "high_electronics_load_v2"
  | "cleaning_event_v2";

export interface V2Status {
  simulator_version: string;
  lifecycle: "closed";
  development_status: "PASS_HYBRID_CANDIDATE";
  final_status: "FAIL";
  official_demo_controller: string;
  v2_controller: {
    parameters: number;
    role: string;
    eligible_for_demo_replacement: false;
    checkpoint_sha256: string;
    development_gates: Record<string, boolean>;
  };
  held_out: { status: "PARTIALLY_OPENED_HYBRID_COMBINED_STRESS"; final_test_opened: true; reason: string };
  hybrid_final: { status: "COMPLETED_FAIL"; acceptance_pass: false; acceptance_gates: Record<string, boolean> };
  scenario_access: { development: V2Scenario[]; held_out_opened: string[]; held_out_sealed: string[] };
}

export interface V2TrajectoryStep {
  step: number;
  timestamp: string;
  hour: number;
  state: Record<string, number>;
  proposed_action: number;
  proposed_action_name: string;
  executed_action: number;
  executed_action_name: string;
  reward: number;
  reward_audit: {
    priority_percent: Record<"energy" | "comfort" | "co2", number>;
    effective_weights: Record<string, number>;
    weighted_penalties: Record<string, number>;
    comfort_margin_c: number;
    humidity_margin_pct: number;
    co2_margin_ppm: number;
  };
  energy: Record<string, number>;
  heat_flows: Record<string, number>;
  risk: Record<string, number>;
  forecast: {
    forecasts: Array<{
      horizon_hours: number;
      values: Record<string, { point: number; lower: number; upper: number }>;
    }>;
  };
  policy_explanation?: {
    human_readable: string;
    causal_claim: false;
    contributions: Array<{
      feature: string;
      signed_contribution: number;
      absolute_importance_pct: number;
    }>;
    counterfactual?: {
      found: boolean;
      human_readable: string;
      counterfactual_action_name?: string;
    };
  };
  shield_explanation: {
    decision: string;
    intervention: boolean;
    reason: string;
    human_readable: string;
  };
  comfort_status: "comfortable" | "violation";
  co2_status: "acceptable" | "violation";
}

export interface V2SimulationResult {
  controller: "v2_dqn_experimental";
  scenario: V2Scenario;
  seed: number;
  status: "DEVELOPMENT_FAIL";
  disclaimer: string;
  summary: Record<string, unknown>;
  trajectory: V2TrajectoryStep[];
}
