// Single data-access boundary (plan's design goal): every view fetches
// through these functions, never `fetch("/data/...")` directly. Swapping
// generated JSON files for a live API later is a one-file change here --
// no component needs to change.

import type { Backtest, Calibration, SeasonMeta, SeasonPlan, Simulation, WeekBoard } from "./types";

const BASE = "/data";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}/${path}`);
  if (!res.ok) {
    throw new Error(`Failed to load ${path}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export function getSeasonMeta(): Promise<SeasonMeta> {
  return getJSON<SeasonMeta>("season_meta.json");
}

export function getWeekBoard(week: number): Promise<WeekBoard> {
  return getJSON<WeekBoard>(`week_${String(week).padStart(2, "0")}_board.json`);
}

export function getSeasonPlan(): Promise<SeasonPlan> {
  return getJSON<SeasonPlan>("season_plan.json");
}

export function getCalibration(): Promise<Calibration> {
  return getJSON<Calibration>("calibration.json");
}

export function getBacktest(): Promise<Backtest> {
  return getJSON<Backtest>("backtest.json");
}

export function getSimulation(): Promise<Simulation> {
  return getJSON<Simulation>("simulation.json");
}
