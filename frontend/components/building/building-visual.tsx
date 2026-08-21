import { Fan, Users } from "lucide-react";
import type { TrajectoryStep } from "@/types/api";

interface BuildingVisualProps { step?: TrajectoryStep }

export function BuildingVisual({ step }: BuildingVisualProps) {
  const action = step?.action ?? 0;
  const airflow = ["off", "low", "medium", "high"][action];
  const people = Math.min(8, Math.ceil((step?.state.occupancy ?? 0) / 10));
  return (
    <section className="building-panel panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Digital twin</p>
          <h2>Zone A · Operations floor</h2>
        </div>
        <span className={`mode-pill mode-pill--${airflow}`}><Fan size={14} /> HVAC {airflow.toUpperCase()}</span>
      </div>
      <div className={`building building--${airflow}`}>
        <div className="building__glow" />
        <div className="building__shell">
          <div className="building__roof"><span /><span /><span /></div>
          <div className="building__floor building__floor--top">
            <div className="building__sensor">CO₂ <strong>{Math.round(step?.state.co2_ppm ?? 500)}</strong><small>ppm</small></div>
            <div className="air-streams" aria-hidden="true"><i /><i /><i /></div>
          </div>
          <div className="building__floor building__floor--main">
            <div className="occupants" aria-label={`${step?.state.occupancy ?? 0} occupants`}>
              {Array.from({ length: people }, (_, index) => <Users key={index} size={16} />)}
            </div>
            <div className="temperature-orb">
              <span>Indoor</span>
              <strong>{(step?.state.indoor_temperature_c ?? 24).toFixed(1)}°</strong>
              <small>Comfort target 22–25°C</small>
            </div>
          </div>
          <div className="building__base">
            <span>Outdoor {(step?.state.outdoor_temperature_c ?? 0).toFixed(1)}°C</span>
            <span>{Math.round(step?.state.relative_humidity_pct ?? 0)}% RH</span>
          </div>
        </div>
      </div>
    </section>
  );
}
