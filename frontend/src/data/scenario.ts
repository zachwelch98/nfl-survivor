// What-if locks, persisted in localStorage. Same three conventions as
// storage.ts: zod-validated on read, reset rather than crash on a bad shape,
// try/catch on write.
//
// Deliberately a *separate* key from survivor:user-state:v1. Two reasons:
//
//  1. Safety. storage.ts's schema pins `version: z.literal(1)` and silently
//     resets to empty on any parse failure, so widening that schema would
//     wipe every recorded pick on first load.
//  2. Meaning. A what-if is exploratory. Trying five wild scenarios must not
//     rewrite the picks you have actually committed to -- `importFromPicks`
//     copies in one direction only, and nothing here ever writes back.

import { z } from "zod";
import type { ScenarioLock, ScenarioState, UserState } from "./types";

const STORAGE_KEY = "survivor:scenario:v1";

const scenarioLockSchema = z.object({
  season: z.number(),
  week: z.number(),
  team: z.string(),
});

const scenarioStateSchema = z.object({
  version: z.literal(1),
  locks: z.array(scenarioLockSchema),
});

function emptyState(): ScenarioState {
  return { version: 1, locks: [] };
}

export function loadScenarioState(): ScenarioState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptyState();
    const parsed = scenarioStateSchema.safeParse(JSON.parse(raw));
    return parsed.success ? parsed.data : emptyState();
  } catch {
    return emptyState();
  }
}

export function saveScenarioState(state: ScenarioState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // private browsing / storage disabled -- locks just don't survive a reload
  }
}

/** One lock per (season, week); setting a week replaces whatever was there. */
export function setLock(state: ScenarioState, lock: ScenarioLock): ScenarioState {
  const others = state.locks.filter((l) => !(l.season === lock.season && l.week === lock.week));
  return { ...state, locks: [...others, lock].sort((a, b) => a.season - b.season || a.week - b.week) };
}

export function clearLock(state: ScenarioState, season: number, week: number): ScenarioState {
  return { ...state, locks: state.locks.filter((l) => !(l.season === season && l.week === week)) };
}

export function clearAllLocks(state: ScenarioState, season: number): ScenarioState {
  return { ...state, locks: state.locks.filter((l) => l.season !== season) };
}

export function locksForSeason(state: ScenarioState, season: number): ScenarioLock[] {
  return state.locks.filter((l) => l.season === season).sort((a, b) => a.week - b.week);
}

/**
 * Seed locks from My Picks. Only *pending* picks in weeks the optimizer still
 * decides become locks — a decided pick (or anything before the current week)
 * isn't a constraint on a future decision, it's a team removed from the pool,
 * which `burnedTeams` handles instead.
 *
 * Replaces this season's locks wholesale rather than merging, so the button
 * does what it says: the scenario now matches My Picks.
 */
export function importFromPicks(
  state: ScenarioState,
  userState: UserState,
  season: number,
  currentWeek: number,
): ScenarioState {
  const imported = userState.picks
    .filter((p) => p.season === season && p.week >= currentWeek && p.outcome === "pending")
    .map((p) => ({ season, week: p.week, team: p.team }));
  const preserved = state.locks.filter((l) => l.season !== season);
  return {
    ...state,
    locks: [...preserved, ...imported].sort((a, b) => a.season - b.season || a.week - b.week),
  };
}

/**
 * Teams no longer in the pool: anything already played, plus any decided pick.
 * Distinct from a lock — burned is removal, locked is a constraint — and
 * burned always wins, since you cannot commit a team you have already spent.
 */
export function burnedTeams(userState: UserState, season: number, currentWeek: number): Set<string> {
  return new Set(
    userState.picks
      .filter((p) => p.season === season && (p.week < currentWeek || p.outcome !== "pending"))
      .map((p) => p.team),
  );
}
