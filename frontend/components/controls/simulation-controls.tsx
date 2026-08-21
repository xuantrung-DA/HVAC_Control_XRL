import { Play, RotateCcw, Sparkles } from "lucide-react";
import type { Controller, Scenario } from "@/types/api";

interface SimulationControlsProps {
  controller: Controller;
  scenario: Scenario;
  loading: boolean;
  onControllerChange: (value: Controller) => void;
  onScenarioChange: (value: Scenario) => void;
  onRun: () => void;
  onReset: () => void;
}

const controllers: Array<{ value: Controller; label: string }> = [
  { value: "dqn", label: "DQN · Frozen demo" },
  { value: "rule_based", label: "Rule-based" },
  { value: "fixed_thermostat", label: "Thermostat" },
  { value: "random", label: "Random" },
];

const scenarios: Array<{ value: Scenario; label: string }> = [
  { value: "normal", label: "Normal day" },
  { value: "hot_day", label: "Hot day" },
  { value: "high_occupancy", label: "High occupancy" },
  { value: "expensive_electricity", label: "Expensive electricity" },
  { value: "combined_stress", label: "Combined stress" },
];

export function SimulationControls(props: SimulationControlsProps) {
  return (
    <section className="control-bar" aria-label="Simulation controls">
      <div className="control-bar__intro">
        <span className="status-dot" />
        <div>
          <p className="eyebrow">Live policy sandbox</p>
          <p>24 hours · 15-minute control intervals</p>
        </div>
      </div>
      <label>
        <span>Controller</span>
        <select value={props.controller} onChange={(event) => props.onControllerChange(event.target.value as Controller)}>
          {controllers.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
        </select>
      </label>
      <label>
        <span>Scenario</span>
        <select value={props.scenario} onChange={(event) => props.onScenarioChange(event.target.value as Scenario)}>
          {scenarios.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
        </select>
      </label>
      <div className="control-bar__actions">
        <button className="button button--ghost" onClick={props.onReset} aria-label="Reset playback">
          <RotateCcw size={17} /> Reset
        </button>
        <button className="button button--primary" onClick={props.onRun} disabled={props.loading}>
          {props.loading ? <Sparkles className="spin" size={17} /> : <Play size={17} fill="currentColor" />}
          {props.loading ? "Explaining…" : "Run simulation"}
        </button>
      </div>
    </section>
  );
}
