// Picks + outcomes persisted in localStorage (plan's design), validated
// with zod on read so a shape change degrades to a reset instead of a
// crash. This is the only piece of user state not served by the pipeline.

import { z } from "zod";
import type { UserPick, UserState } from "./types";

const STORAGE_KEY = "survivor:user-state:v1";

const userPickSchema = z.object({
  season: z.number(),
  week: z.number(),
  team: z.string(),
  outcome: z.enum(["pending", "win", "loss", "tie"]),
});

const userStateSchema = z.object({
  version: z.literal(1),
  picks: z.array(userPickSchema),
});

function emptyState(): UserState {
  return { version: 1, picks: [] };
}

export function loadUserState(): UserState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptyState();
    const parsed = userStateSchema.safeParse(JSON.parse(raw));
    return parsed.success ? parsed.data : emptyState();
  } catch {
    return emptyState();
  }
}

export function saveUserState(state: UserState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // private browsing / storage disabled -- state simply doesn't persist this session
  }
}

export function upsertPick(state: UserState, pick: UserPick): UserState {
  const withoutThisWeek = state.picks.filter((p) => !(p.season === pick.season && p.week === pick.week));
  return { ...state, picks: [...withoutThisWeek, pick].sort((a, b) => a.season - b.season || a.week - b.week) };
}

export function removePick(state: UserState, season: number, week: number): UserState {
  return { ...state, picks: state.picks.filter((p) => !(p.season === season && p.week === week)) };
}

export function usedTeams(state: UserState, season: number, throughWeek?: number): Set<string> {
  return new Set(
    state.picks
      .filter((p) => p.season === season && (throughWeek === undefined || p.week <= throughWeek))
      .map((p) => p.team),
  );
}

export function survivalRecord(state: UserState, season: number): { wins: number; losses: number; ties: number; currentStreak: number; eliminatedWeek: number | null } {
  const picks = state.picks.filter((p) => p.season === season).sort((a, b) => a.week - b.week);
  let wins = 0;
  let losses = 0;
  let ties = 0;
  let eliminatedWeek: number | null = null;
  let streak = 0;
  for (const p of picks) {
    if (p.outcome === "win") {
      wins++;
      streak++;
    } else if (p.outcome === "loss") {
      losses++;
      streak = 0;
      if (eliminatedWeek === null) eliminatedWeek = p.week;
    } else if (p.outcome === "tie") {
      ties++;
      streak = 0;
    }
  }
  return { wins, losses, ties, currentStreak: streak, eliminatedWeek };
}
