import { ArrowRight, BrainCircuit, ShieldCheck } from "lucide-react";
import type { TrajectoryStep } from "@/types/api";

const labels: Record<string, string> = {
  indoor_temperature_c: "Indoor temperature",
  outdoor_temperature_c: "Outdoor temperature",
  relative_humidity_pct: "Humidity",
  occupancy: "Occupancy",
  co2_ppm: "CO₂",
  electricity_price_per_kwh: "Electricity price",
  time_sin: "Time · sine",
  time_cos: "Time · cosine",
  hvac_action: "Previous HVAC state",
};

export function ExplanationPanel({ step }: { step?: TrajectoryStep }) {
  const attribution = step?.feature_attribution;
  const contributions = [...(attribution?.contributions ?? [])].sort((a, b) => b.absolute_importance_pct - a.absolute_importance_pct).slice(0, 5);
  return (
    <section className="xai-panel panel">
      <div className="panel-heading">
        <div><p className="eyebrow">Decision intelligence</p><h2>Why this action?</h2></div>
        <span className="faithful-badge"><ShieldCheck size={14} /> Locally faithful</span>
      </div>
      {attribution ? <>
        <div className="decision-summary">
          <div className="decision-icon"><BrainCircuit size={23} /></div>
          <p>{attribution.human_readable}</p>
        </div>
        <div className="importance-list">
          {contributions.map((item) => <div className="importance-row" key={item.feature}>
            <div><span>{labels[item.feature] ?? item.feature}</span><strong>{item.absolute_importance_pct.toFixed(1)}%</strong></div>
            <div className="importance-track"><i className={item.signed_contribution >= 0 ? "positive" : "negative"} style={{ width: `${Math.max(2, item.absolute_importance_pct)}%` }} /></div>
          </div>)}
        </div>
        {step?.counterfactual?.found && <div className="counterfactual-card">
          <span>Smallest action-changing state edit</span>
          <div><strong>{step.counterfactual.changes[0]?.original_value.toFixed(1)}</strong><ArrowRight size={16} /><strong>{step.counterfactual.changes[0]?.counterfactual_value.toFixed(1)}</strong><em>{step.counterfactual.counterfactual_action_name}</em></div>
          <p>{step.counterfactual.human_readable}</p>
        </div>}
      </> : <div className="empty-xai"><BrainCircuit size={28} /><p>Run the frozen DQN to inspect local feature attribution and a verified counterfactual at every timestep.</p></div>}
    </section>
  );
}
