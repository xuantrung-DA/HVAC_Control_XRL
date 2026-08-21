import type { TrajectoryStep } from "@/types/api";

interface TelemetryChartProps { data: TrajectoryStep[]; activeIndex: number }

function points(values: number[], width: number, height: number, min: number, max: number) {
  return values.map((value, index) => {
    const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * width;
    const y = height - ((value - min) / Math.max(max - min, 0.001)) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

export function TelemetryChart({ data, activeIndex }: TelemetryChartProps) {
  const temperatures = data.map((item) => item.state.indoor_temperature_c);
  const co2 = data.map((item) => item.state.co2_ppm);
  const activeX = data.length <= 1 ? 0 : (activeIndex / (data.length - 1)) * 680;
  return (
    <section className="telemetry panel">
      <div className="panel-heading">
        <div><p className="eyebrow">24-hour telemetry</p><h2>Environment response</h2></div>
        <div className="chart-legend"><span><i className="legend-temp" /> Temperature</span><span><i className="legend-co2" /> CO₂</span></div>
      </div>
      <div className="chart-wrap">
        <svg viewBox="0 0 680 210" role="img" aria-label="Indoor temperature and CO2 over 24 hours">
          <defs>
            <linearGradient id="tempGradient" x1="0" x2="1"><stop stopColor="#8b5cf6" /><stop offset=".5" stopColor="#ef3f88" /><stop offset="1" stopColor="#ff9d42" /></linearGradient>
            <filter id="chartGlow"><feGaussianBlur stdDeviation="3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          </defs>
          {[0, 1, 2, 3, 4].map((line) => <line key={line} x1="0" x2="680" y1={10 + line * 45} y2={10 + line * 45} className="chart-grid" />)}
          {data.length > 1 && <>
            <polyline points={points(temperatures, 680, 180, 18, 32)} className="chart-line chart-line--temp" filter="url(#chartGlow)" />
            <polyline points={points(co2, 680, 180, 350, 1400)} className="chart-line chart-line--co2" />
            <line x1={activeX} x2={activeX} y1="0" y2="190" className="chart-cursor" />
          </>}
          <text x="0" y="207">00:00</text><text x="324" y="207">12:00</text><text x="642" y="207">24:00</text>
        </svg>
      </div>
    </section>
  );
}
