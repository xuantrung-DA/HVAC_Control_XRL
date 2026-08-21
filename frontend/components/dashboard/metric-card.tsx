import type { ReactNode } from "react";

interface MetricCardProps {
  eyebrow: string;
  value: string;
  detail: string;
  icon: ReactNode;
  tone?: "violet" | "coral" | "amber" | "cyan";
}

export function MetricCard({ eyebrow, value, detail, icon, tone = "violet" }: MetricCardProps) {
  return (
    <article className={`metric-card metric-card--${tone}`}>
      <div className="metric-card__icon" aria-hidden="true">{icon}</div>
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <strong className="metric-card__value">{value}</strong>
        <p className="metric-card__detail">{detail}</p>
      </div>
    </article>
  );
}
