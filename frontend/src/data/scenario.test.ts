import { describe, expect, it } from "vitest";
import {
  burnedTeams,
  clearAllLocks,
  clearLock,
  importFromPicks,
  locksForSeason,
  setLock,
} from "./scenario";
import type { ScenarioState, UserState } from "./types";

const empty: ScenarioState = { version: 1, locks: [] };

describe("lock reducers", () => {
  it("keeps one lock per week, replacing on re-set", () => {
    let s = setLock(empty, { season: 2026, week: 3, team: "KC" });
    s = setLock(s, { season: 2026, week: 3, team: "BUF" });
    expect(s.locks).toEqual([{ season: 2026, week: 3, team: "BUF" }]);
  });

  it("keeps locks sorted by week", () => {
    let s = setLock(empty, { season: 2026, week: 9, team: "KC" });
    s = setLock(s, { season: 2026, week: 2, team: "BUF" });
    expect(s.locks.map((l) => l.week)).toEqual([2, 9]);
  });

  it("clears one week without touching the others", () => {
    let s = setLock(empty, { season: 2026, week: 3, team: "KC" });
    s = setLock(s, { season: 2026, week: 5, team: "BUF" });
    expect(clearLock(s, 2026, 3).locks.map((l) => l.week)).toEqual([5]);
  });

  it("clears a season without touching another season", () => {
    let s = setLock(empty, { season: 2026, week: 3, team: "KC" });
    s = setLock(s, { season: 2025, week: 3, team: "BUF" });
    expect(clearAllLocks(s, 2026).locks).toEqual([{ season: 2025, week: 3, team: "BUF" }]);
    expect(locksForSeason(s, 2026).map((l) => l.team)).toEqual(["KC"]);
  });
});

describe("importFromPicks", () => {
  const picks: UserState = {
    version: 1,
    picks: [
      { season: 2026, week: 1, team: "LA", outcome: "win" }, // decided -> burned, not locked
      { season: 2026, week: 2, team: "PHI", outcome: "loss" }, // decided -> burned, not locked
      { season: 2026, week: 5, team: "KC", outcome: "pending" }, // future intent -> lock
      { season: 2026, week: 9, team: "BUF", outcome: "pending" }, // future intent -> lock
      { season: 2025, week: 5, team: "DAL", outcome: "pending" }, // other season -> ignored
    ],
  };

  it("imports only pending picks at or after the current week", () => {
    const s = importFromPicks(empty, picks, 2026, 3);
    expect(locksForSeason(s, 2026)).toEqual([
      { season: 2026, week: 5, team: "KC" },
      { season: 2026, week: 9, team: "BUF" },
    ]);
  });

  it("treats a pending pick in an already-past week as burned, not locked", () => {
    const past: UserState = {
      version: 1,
      picks: [{ season: 2026, week: 2, team: "KC", outcome: "pending" }],
    };
    expect(locksForSeason(importFromPicks(empty, past, 2026, 5), 2026)).toEqual([]);
    expect(burnedTeams(past, 2026, 5).has("KC")).toBe(true);
  });

  it("replaces this season's locks rather than merging with them", () => {
    const before = setLock(empty, { season: 2026, week: 12, team: "SF" });
    const after = importFromPicks(before, picks, 2026, 3);
    expect(locksForSeason(after, 2026).map((l) => l.week)).toEqual([5, 9]);
  });

  it("leaves other seasons' locks alone", () => {
    const before = setLock(empty, { season: 2025, week: 12, team: "SF" });
    const after = importFromPicks(before, picks, 2026, 3);
    expect(locksForSeason(after, 2025)).toEqual([{ season: 2025, week: 12, team: "SF" }]);
  });
});

describe("burnedTeams", () => {
  const state: UserState = {
    version: 1,
    picks: [
      { season: 2026, week: 1, team: "LA", outcome: "win" },
      { season: 2026, week: 2, team: "PHI", outcome: "loss" },
      { season: 2026, week: 3, team: "SF", outcome: "tie" },
      { season: 2026, week: 7, team: "KC", outcome: "pending" },
      { season: 2025, week: 1, team: "DAL", outcome: "win" },
    ],
  };

  it("burns decided picks and anything before the current week", () => {
    expect(burnedTeams(state, 2026, 4)).toEqual(new Set(["LA", "PHI", "SF"]));
  });

  it("does not burn a pending pick in a week still to be decided", () => {
    expect(burnedTeams(state, 2026, 4).has("KC")).toBe(false);
  });

  it("is scoped to one season", () => {
    expect(burnedTeams(state, 2026, 4).has("DAL")).toBe(false);
  });
});
