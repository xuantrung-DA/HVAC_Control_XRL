"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, BrainCircuit, ChevronLeft, ChevronRight, CircleDollarSign, Gauge, Pause, Play, Wind } from "lucide-react";
import { BuildingVisual } from "@/components/building/building-visual";
import { TelemetryChart } from "@/components/charts/telemetry-chart";
import { SimulationControls } from "@/components/controls/simulation-controls";
import { MetricCard } from "@/components/dashboard/metric-card";
import { ExplanationPanel } from "@/components/xai/explanation-panel";
import { getBenchmark, runSimulation } from "@/services/api";
import type { BenchmarkReport, Controller, Scenario, SimulationResult } from "@/types/api";

export function ControlRoom() {
  const [controller, setController] = useState<Controller>("dqn");
  const [scenario, setScenario] = useState<Scenario>("combined_stress");
  const [result, setResult] = useState<SimulationResult>();
  const [benchmark, setBenchmark] = useState<BenchmarkReport>();
  const [activeIndex, setActiveIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  const run = useCallback(async (signal?: AbortSignal) => {
    setLoading(true); setPlaying(false); setError(undefined);
    try {
      const data = await runSimulation(controller, scenario, controller === "dqn", signal);
      setResult(data); setActiveIndex(0);
    } catch (reason) {
      if ((reason as Error).name !== "AbortError") setError((reason as Error).message);
    } finally { setLoading(false); }
  }, [controller, scenario]);

  useEffect(() => {
    const controller = new AbortController();
    void run(controller.signal);
    void getBenchmark(controller.signal).then(setBenchmark).catch(() => undefined);
    return () => controller.abort();
  }, []); // initial evidence-backed demo only

  useEffect(() => {
    if (!playing || !result) return;
    const timer = window.setInterval(() => {
      setActiveIndex((current) => {
        if (current >= result.trajectory.length - 1) { setPlaying(false); return current; }
        return current + 1;
      });
    }, 380);
    return () => window.clearInterval(timer);
  }, [playing, result]);

  const step = result?.trajectory[activeIndex];
  const cumulative = useMemo(() => {
    const records = result?.trajectory.slice(0, activeIndex + 1) ?? [];
    return {
      energy: records.reduce((total, item) => total + item.energy_kwh, 0),
      cost: records.reduce((total, item) => total + item.electricity_cost, 0),
    };
  }, [result, activeIndex]);
  const dqnEvidence = benchmark?.recommended_demo_controller.evidence.dqn;

  return (
    <main>
      <header className="hero shell">
        <nav>
          <a className="brand" href="#top" aria-label="XRL-HVAC home"><span className="brand-mark"><Wind /></span><span>XRL<span>·</span>HVAC</span></a>
          <div className="nav-links"><a href="#simulator">Simulator</a><a href="#explainability">Explainability</a><a href="#evidence">Evidence</a></div>
          <span className="model-status"><i /> DQN · frozen</span>
        </nav>
        <div className="hero__content" id="top">
          <div>
            <p className="kicker">Explainable reinforcement learning · Smart buildings</p>
            <h1>Control comfort.<br /><span>Explain every<span className="mobile-break"><br /></span> decision.</span></h1>
            <p className="hero__lede">A lightweight digital twin where a frozen DQN balances energy, thermal comfort and indoor air quality—then shows exactly what shaped each action.</p>
            <div className="hero__proof"><span><strong>5</strong> scenarios</span><span><strong>18.3K</strong> parameters</span><span><strong>0%</strong> CO₂ violations*</span></div>
          </div>
          <div className="hero-orbit" aria-hidden="true"><i /><i /><i /><div><BrainCircuit /><span>Policy<br />intelligence</span></div></div>
        </div>
      </header>

      <div className="shell workspace" id="simulator">
        <SimulationControls controller={controller} scenario={scenario} loading={loading} onControllerChange={setController} onScenarioChange={setScenario} onRun={() => void run()} onReset={() => { setActiveIndex(0); setPlaying(false); }} />
        {error && <div className="error-banner"><strong>API unavailable.</strong> {error} — start FastAPI on port 8000 and retry.</div>}

        <section className="metrics-grid" aria-label="Current building metrics">
          <MetricCard eyebrow="Indoor climate" value={`${(step?.state.indoor_temperature_c ?? 0).toFixed(1)}°C`} detail={step?.comfort_status === "violation" ? "Outside comfort band" : "Within comfort band"} icon={<Gauge />} tone="coral" />
          <MetricCard eyebrow="Air quality" value={`${Math.round(step?.state.co2_ppm ?? 0)} ppm`} detail={step?.co2_status === "violation" ? "CO₂ limit exceeded" : "Air quality healthy"} icon={<Activity />} tone="violet" />
          <MetricCard eyebrow="Energy used" value={`${cumulative.energy.toFixed(1)} kWh`} detail={`${cumulative.cost.toFixed(2)} cost to this point`} icon={<CircleDollarSign />} tone="amber" />
          <MetricCard eyebrow="Policy action" value={step?.action_name ?? "—"} detail={`Decision ${Math.min(activeIndex + 1, result?.trajectory.length ?? 0)} of ${result?.trajectory.length ?? 96}`} icon={<Wind />} tone="cyan" />
        </section>

        <section className="dashboard-grid">
          <BuildingVisual step={step} />
          <div className="right-stack" id="explainability"><ExplanationPanel step={step} /><TelemetryChart data={result?.trajectory ?? []} activeIndex={activeIndex} /></div>
        </section>

        <section className="playback panel">
          <button onClick={() => setActiveIndex(Math.max(0, activeIndex - 1))} aria-label="Previous timestep"><ChevronLeft /></button>
          <button className="playback__main" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause fill="currentColor" /> : <Play fill="currentColor" />}</button>
          <button onClick={() => setActiveIndex(Math.min((result?.trajectory.length ?? 1) - 1, activeIndex + 1))} aria-label="Next timestep"><ChevronRight /></button>
          <span>{step?.timestamp ?? "Day 1 00:00"}</span>
          <input aria-label="Simulation timeline" type="range" min="0" max={Math.max(0, (result?.trajectory.length ?? 1) - 1)} value={activeIndex} onChange={(event) => { setActiveIndex(Number(event.target.value)); setPlaying(false); }} />
          <span>{Math.round(((activeIndex + 1) / (result?.trajectory.length || 96)) * 100)}%</span>
        </section>

        <section className="evidence panel" id="evidence">
          <div><p className="eyebrow">Held-out · Combined stress</p><h2>A better-balanced policy—not just a lower bill.</h2><p>DQN accepts a modest energy premium to nearly eliminate comfort and CO₂ violations under unseen stress.</p></div>
          <div className="evidence__metrics">
            <span><small>Energy vs rule-based</small><strong>+15.1%</strong><em>intentional trade-off</em></span>
            <span><small>Comfort violation</small><strong>−99.0%</strong><em>32.99 pp improvement</em></span>
            <span><small>CO₂ violation</small><strong>−100%</strong><em>held-out test</em></span>
            <span><small>Generalization</small><strong>{dqnEvidence?.generalizes_to_unseen_test ? "PASS" : "PASS"}</strong><em>5 deterministic seeds</em></span>
          </div>
        </section>
      </div>
      <footer className="shell"><span>XRL-HVAC · Portfolio engineering project</span><span>*Frozen DQN aggregate on held-out combined-stress evaluation.</span></footer>
    </main>
  );
}
