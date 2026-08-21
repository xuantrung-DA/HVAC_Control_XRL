import type { BenchmarkReport, Controller, Scenario, SimulationResult } from "@/types/api";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `API request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function runSimulation(
  controller: Controller,
  scenario: Scenario,
  includeExplanations: boolean,
  signal?: AbortSignal,
): Promise<SimulationResult> {
  return request<SimulationResult>("/simulations/run", {
    method: "POST",
    signal,
    body: JSON.stringify({
      controller,
      scenario,
      seed: 707,
      include_explanations: includeExplanations,
    }),
  });
}

export function getBenchmark(signal?: AbortSignal): Promise<BenchmarkReport> {
  return request<BenchmarkReport>("/metrics/benchmark", { signal });
}
