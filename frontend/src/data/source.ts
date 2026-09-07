// Single data-access boundary (plan's design goal): every view fetches
// through these functions, never `fetch("/data/...")` directly. Swapping
// generated JSON files for a live API later is a one-file change here --
// no component needs to change.

import type { Backtest, Calibration, SeasonMeta, SeasonPlan, Simulation, WeekBoard } from "./types";

const BASE = "/data";

async function getJSON<T>(path: string): Promise<T> {
  // no-store: after a pipeline export regenerates these static files, a
  // refetch should see the new bytes rather than a cached copy.
  const res = await fetch(`${BASE}/${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load ${path}: ${res.status} ${res.statusText}`);
  }
  // A missing file under /data does NOT 404: the dev server answers unknown
  // paths with index.html (the SPA fallback) at status 200, so res.ok is
  // true and the JSON parse then fails on "<!doctype" with a message that
  // says nothing about what actually went wrong. Check the content type and
  // report the real problem instead.
  const contentType = res.headers.get("content-type") ?? "";
  if (!contentType.includes("json")) {
    throw new Error(
      `No data file for ${path}. It hasn't been generated yet — run \`make export\` (or use the Dashboard's pipeline controls) to create it.`,
    );
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
