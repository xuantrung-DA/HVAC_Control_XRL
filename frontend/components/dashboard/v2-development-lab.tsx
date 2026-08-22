"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ArrowRight, BrainCircuit, FlaskConical, LockKeyhole, Play, ShieldAlert } from "lucide-react";
import { getV2Status, runV2Simulation } from "@/services/api";
import type { V2Scenario, V2SimulationResult, V2Status } from "@/types/api";

const SCENARIOS: Array<{ value: V2Scenario; label: string }> = [
  { value: "normal_v2", label: "Normal day" },
  { value: "hot_day_v2", label: "Hot day" },
  { value: "high_occupancy_v2", label: "High occupancy" },
  { value: "high_humidity_v2", label: "High humidity" },
  { value: "expensive_electricity_v2", label: "Expensive electricity" },
  { value: "meeting_surge_v2", label: "Scheduled meeting" },
  { value: "high_electronics_load_v2", label: "High electronics load" },
  { value: "cleaning_event_v2", label: "Cleaning event" },
];

const LABELS: Record<string, string> = {
  indoor_temperature_c: "Indoor temperature",
  indoor_relative_humidity_pct: "Indoor humidity",
  occupancy: "Occupancy",
  co2_ppm: "CO₂",
  electricity_price_per_kwh: "Electricity price",
  thermal_risk: "Thermal risk",
  humidity_risk: "Humidity risk",
  co2_risk: "CO₂ risk",
  forecast_uncertainty: "Forecast uncertainty",
};

export function V2DevelopmentLab() {
  const [status, setStatus] = useState<V2Status>();
  const [scenario, setScenario] = useState<V2Scenario>("normal_v2");
  const [result, setResult] = useState<V2SimulationResult>();
  const [index, setIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  useEffect(() => {
    const controller = new AbortController();
    void getV2Status(controller.signal).then(setStatus).catch((reason) => setError((reason as Error).message));
    return () => controller.abort();
  }, []);

  async function run() {
    setLoading(true); setError(undefined);
    try {
      const data = await runV2Simulation(scenario);
      setResult(data); setIndex(0);
    } catch (reason) { setError((reason as Error).message); }
    finally { setLoading(false); }
  }

  const step = result?.trajectory[index];
  const topFeatures = useMemo(() => [...(step?.policy_explanation?.contributions ?? [])]
    .sort((a, b) => b.absolute_importance_pct - a.absolute_importance_pct).slice(0, 5), [step]);
  const forecast = step?.forecast.forecasts.find((item) => item.horizon_hours === 1);
  const priorities = step?.reward_audit.priority_percent;
  const energyEntries = ["hvac_cooling_kwh", "ventilation_fan_kwh", "lighting_kwh", "electronics_kwh", "base_building_kwh"]
    .map((key) => [key, step?.energy[key] ?? 0] as const);
  const heatEntries = ["opaque_envelope_kw", "windows_kw", "infiltration_kw", "ventilation_kw", "solar_kw", "occupants_kw", "electronics_kw", "hvac_cooling_kw"]
    .map((key) => [key, step?.heat_flows[key] ?? 0] as const);

  return (
    <section className="v2-lab" id="v2-lab">
      <div className="v2-heading">
        <div><p className="eyebrow">V2 closed engineering iteration</p><h2>Physics, foresight and guarded control.</h2><p>The hybrid candidate passed development, then failed the one-shot Combined Stress comfort gate. V1 remains the official demo.</p></div>
        <div className="v2-badges"><span className="fail-badge"><AlertTriangle /> FINAL GATE FAIL</span><span className="sealed-badge"><LockKeyhole /> OPENED ONCE</span></div>
      </div>

      <div className="protocol-strip">
        <span><strong>Official demo</strong> V1 frozen DQN</span>
        <span><strong>Experimental model</strong> {status?.v2_controller.parameters?.toLocaleString() ?? "21,636"} params</span>
        <span><strong>Final test</strong> Combined Stress · complete</span>
        <span><strong>Replacement eligible</strong> No</span>
      </div>

      <div className="v2-controls panel">
        <div><FlaskConical /><span><strong>Development scenarios only</strong><small>Seed 901 · deterministic inference · shield enabled</small></span></div>
        <label>Scenario<select value={scenario} onChange={(event) => setScenario(event.target.value as V2Scenario)}>{SCENARIOS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
        <button className="button button--primary" onClick={() => void run()} disabled={loading}><Play /> {loading ? "Generating 96 explanations…" : "Run V2 development"}</button>
      </div>
      {error && <div className="error-banner">{error}</div>}

      <div className="v2-grid">
        <article className="panel v2-twin">
          <div className="panel-heading"><div><p className="eyebrow">2R1C digital twin</p><h3>Physical state + 1h forecast</h3></div><span className="mode-pill">{step?.timestamp ?? "Not run"}</span></div>
          <div className="state-pairs">
            <div><span>Indoor</span><strong>{(step?.state.indoor_temperature_c ?? 24).toFixed(1)}°C</strong><small>{(step?.state.indoor_relative_humidity_pct ?? 50).toFixed(1)}% RH</small></div>
            <div><span>Outdoor now</span><strong>{(step?.state.outdoor_temperature_c ?? 0).toFixed(1)}°C</strong><small>{Math.round(step?.state.occupancy ?? 0)} occupants</small></div>
            <div className="forecast-orb"><span>Forecast +1h</span><strong>{(forecast?.values.outdoor_temperature_c?.point ?? 0).toFixed(1)}°C</strong><small>{forecast ? `${forecast.values.outdoor_temperature_c.lower.toFixed(1)}–${forecast.values.outdoor_temperature_c.upper.toFixed(1)}°C confidence band` : "Run to inspect"}</small></div>
          </div>
          <div className="risk-list">{["thermal_risk", "humidity_risk", "co2_risk", "forecast_uncertainty"].map((key) => <div key={key}><span>{LABELS[key]}</span><i><b style={{ width: `${100 * (step?.risk[key] ?? 0)}%` }} /></i><strong>{Math.round(100 * (step?.risk[key] ?? 0))}%</strong></div>)}</div>
        </article>

        <article className="panel control-path">
          <div className="panel-heading"><div><p className="eyebrow">Two-layer decision</p><h3>Policy proposal → safety execution</h3></div><ShieldAlert /></div>
          <div className="action-flow"><span><small>DQN proposed</small><strong>{step?.proposed_action_name ?? "—"}</strong></span><ArrowRight /><span className={step?.shield_explanation.intervention ? "intervened" : "allowed"}><small>Shield · {step?.shield_explanation.decision ?? "—"}</small><strong>{step?.executed_action_name ?? "—"}</strong></span></div>
          <p className="shield-reason">{step?.shield_explanation.human_readable ?? "Run a development scenario to inspect policy and shield decisions separately."}</p>
          <div className="priority-title"><span>Dynamic reward priority</span><small>contextual weights, not outcome share</small></div>
          <div className="priority-stack">{(["energy", "comfort", "co2"] as const).map((key) => <div key={key} style={{ width: `${priorities?.[key] ?? 33.33}%` }} className={`priority-${key}`}><span>{key}</span><strong>{(priorities?.[key] ?? 0).toFixed(0)}%</strong></div>)}</div>
        </article>

        <article className="panel v2-xai">
          <div className="panel-heading"><div><p className="eyebrow">Policy explanation</p><h3>Local Q-margin sensitivity</h3></div><BrainCircuit /></div>
          <p>{step?.policy_explanation?.human_readable ?? "No policy explanation loaded yet."}</p>
          <div className="importance-list">{topFeatures.map((item) => <div className="importance-row" key={item.feature}><div><span>{LABELS[item.feature] ?? item.feature.replaceAll("_", " ")}</span><strong>{item.absolute_importance_pct.toFixed(1)}%</strong></div><div className="importance-track"><i className={item.signed_contribution >= 0 ? "positive" : "negative"} style={{ width: `${Math.max(2, item.absolute_importance_pct)}%` }} /></div></div>)}</div>
          <small className="xai-caveat">Associational, local explanation—not a causal claim.</small>
        </article>

        <article className="panel flow-breakdown">
          <div className="panel-heading"><div><p className="eyebrow">Energy and heat ledger</p><h3>Where energy goes · why temperature moves</h3></div></div>
          <div className="ledger-columns"><div><h4>Energy / interval</h4>{energyEntries.map(([key, value]) => <p key={key}><span>{key.replaceAll("_", " ")}</span><strong>{value.toFixed(3)} kWh</strong></p>)}</div><div><h4>Heat flow</h4>{heatEntries.map(([key, value]) => <p key={key}><span>{key.replaceAll("_", " ")}</span><strong>{value > 0 ? "+" : ""}{value.toFixed(1)} kW</strong></p>)}</div></div>
        </article>
      </div>

      <div className="v2-timeline panel"><span>{step?.timestamp ?? "00:00"}</span><input aria-label="V2 simulation timeline" type="range" min="0" max={Math.max(0, (result?.trajectory.length ?? 1) - 1)} value={index} onChange={(event) => setIndex(Number(event.target.value))} /><span>{index + 1}/{result?.trajectory.length ?? 96}</span></div>
      <div className="protocol-note"><LockKeyhole /><p><strong>Protocol integrity:</strong> Combined Stress was opened once for the frozen hybrid candidate and failed comfort; reruns are prohibited. Unexpected Surge, Forecast Failure, Heatwave and Door Left Open remain sealed.</p></div>
    </section>
  );
}
