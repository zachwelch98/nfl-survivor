// Bridges the exported survival matrix to the assignment solver: what the
// locks mean, which of them can actually be honored, and what the scenario
// costs. Kept out of the view so the rules are unit-testable — this is where
// every user-facing edge case in the what-if planner is decided.
//
// assignment.ts stays free of app types on purpose; this is the only module
// that knows about SurvivalEntry, ScenarioLock and My Picks.

import { probKey, solveOptimalAssignment, InfeasibleAssignmentError } from "./assignment";
import type { AssignmentResult } from "./assignment";
import type { ScenarioLock, SurvivalEntry } from "../data/types";

/**
 * Restrict the probability table to what the locks allow: a locked week may
 * only use its locked team, and a locked team may only be used in its locked
 * week. That is the entire locking mechanism — the solver itself never learns
 * what a lock is, which is exactly the contract optimize/assignment.py
 * documents ("the caller expresses unavailability by omitting the pair").
 */
export function applyLocks(entries: SurvivalEntry[], locks: ScenarioLock[]): Map<string, number> {
  const weekToTeam = new Map(locks.map((l) => [l.week, l.team]));
  const teamToWeek = new Map(locks.map((l) => [l.team, l.week]));
  const out = new Map<string, number>();
  for (const e of entries) {
    const lockedTeamThisWeek = weekToTeam.get(e.week);
    if (lockedTeamThisWeek !== undefined && lockedTeamThisWeek !== e.team) continue;
    const lockedWeekThisTeam = teamToWeek.get(e.team);
    if (lockedWeekThisTeam !== undefined && lockedWeekThisTeam !== e.week) continue;
    out.set(probKey(e.week, e.team), e.survival_prob);
  }
  return out;
}

export interface LockValidation {
  /** Locks that can actually be applied, in week order. */
  valid: ScenarioLock[];
  /** Human-readable reasons, each naming the lock that caused it. */
  problems: string[];
  /** Locks for weeks the current export no longer covers. */
  stale: ScenarioLock[];
}

/**
 * Separate the honorable locks from the rest, with a specific reason for each
 * rejection. Infeasibility here is user-caused and fixable, so "no plan found"
 * is never an acceptable answer — the user has to be told which lock to change.
 */
export function validateLocks(
  locks: ScenarioLock[],
  weeks: number[],
  entries: SurvivalEntry[],
  burned: Set<string>,
): LockValidation {
  const playable = new Set(entries.map((e) => probKey(e.week, e.team)));
  const weekSet = new Set(weeks);
  const problems: string[] = [];
  const stale: ScenarioLock[] = [];
  const valid: ScenarioLock[] = [];
  const seenTeam = new Map<string, number>();

  for (const l of [...locks].sort((a, b) => a.week - b.week)) {
    if (!weekSet.has(l.week)) {
      stale.push(l);
      continue;
    }
    if (burned.has(l.team)) {
      problems.push(
        `Week ${l.week}: ${l.team} is already used — clear that lock, or remove the pick in My Picks.`,
      );
      continue;
    }
    if (!playable.has(probKey(l.week, l.team))) {
      problems.push(
        `Week ${l.week}: ${l.team} has no game that week (bye). Pick a different team, or a different week.`,
      );
      continue;
    }
    const other = seenTeam.get(l.team);
    if (other !== undefined) {
      problems.push(
        `${l.team} is locked into both week ${other} and week ${l.week} — a team can only be used once.`,
      );
      continue;
    }
    seenTeam.set(l.team, l.week);
    valid.push(l);
  }
  return { valid, problems, stale };
}

export interface ScenarioSolution {
  /** Optimum given only burned teams — the upper bound on any scenario. */
  baseline: AssignmentResult | null;
  /** Optimum given the locks as well. */
  scenario: AssignmentResult | null;
  /** baseline − scenario, in probability. Always >= 0. Null if either failed. */
  costOfLocks: number | null;
  error: string | null;
}

/**
 * Solve twice: once with only burned teams removed (the upper bound), once
 * with the locks applied. The difference is what the user's commitments cost.
 *
 * `locks` must already have been through `validateLocks` — this deliberately
 * does not re-validate, so an unhonorable lock surfaces as a loud infeasible
 * solve rather than being silently dropped.
 */
export function solveScenario(
  entries: SurvivalEntry[],
  weeks: number[],
  availableTeams: string[],
  locks: ScenarioLock[],
): ScenarioSolution {
  let baseline: AssignmentResult | null = null;
  let scenario: AssignmentResult | null = null;
  let error: string | null = null;

  try {
    baseline = solveOptimalAssignment(weeks, availableTeams, applyLocks(entries, []));
  } catch (e) {
    return {
      baseline: null,
      scenario: null,
      costOfLocks: null,
      error:
        e instanceof InfeasibleAssignmentError
          ? `No complete plan exists even before your locks: ${e.message}`
          : String(e),
    };
  }

  try {
    scenario = solveOptimalAssignment(weeks, availableTeams, applyLocks(entries, locks));
  } catch (e) {
    error =
      e instanceof InfeasibleAssignmentError
        ? `Your locks leave no complete plan — ${e.message}. Clear a lock to continue.`
        : String(e);
  }

  // The baseline is by construction an upper bound (the locked problem is the
  // same problem with cells removed), so clamp float noise rather than ever
  // reporting a lock that improved on the optimum.
  const costOfLocks =
    baseline && scenario ? Math.max(0, baseline.totalSurvivalProb - scenario.totalSurvivalProb) : null;

  return { baseline, scenario, costOfLocks, error };
}
