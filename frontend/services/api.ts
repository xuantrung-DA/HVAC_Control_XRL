import type { BenchmarkReport, Controller, Scenario, SimulationResult, V2Scenario, V2SimulationResult, V2Status } from "@/types/api";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";
const API_ORIGIN = API_URL.endsWith("/api/v1") ? API_URL.slice(0, -7) : API_URL.replace(/\/$/, "");

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

async function requestV2<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ORIGIN}/api/v2${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `V2 API request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function getV2Status(signal?: AbortSignal): Promise<V2Status> {
  return requestV2<V2Status>("/status", { signal });
}

export function runV2Simulation(scenario: V2Scenario, signal?: AbortSignal): Promise<V2SimulationResult> {
  return requestV2<V2SimulationResult>("/simulations/run", {
    method: "POST",
    signal,
    body: JSON.stringify({ scenario, seed: 901, include_explanations: true }),
  });
}
