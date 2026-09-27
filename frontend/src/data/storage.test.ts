import { describe, expect, it } from "vitest";
import { nextUnpickedWeek } from "./storage";
import type { UserState } from "./types";

const picks = (...weeks: number[]): UserState => ({
  version: 1,
  picks: weeks.map((w) => ({ season: 2026, week: w, team: `T${w}`, outcome: "win" as const })),
});

describe("nextUnpickedWeek", () => {
  it("advances past the last logged pick even when the export is behind", () => {
    // The motivating case: picks logged for weeks 1 and 2, bundle still
    // exported for week 2. The next decision is week 3, not week 2.
    expect(nextUnpickedWeek(picks(1, 2), 2026, 2)).toBe(3);
  });

  it("falls back to the exported week when nothing has been picked", () => {
    expect(nextUnpickedWeek(picks(), 2026, 2)).toBe(2);
  });

  it("treats the export as the floor when picks lag behind it", () => {
    // Fell behind on logging: the bundle is the better authority.
    expect(nextUnpickedWeek(picks(1, 2), 2026, 8)).toBe(8);
  });

  it("ignores picks from another season", () => {
    const mixed: UserState = {
      version: 1,
      picks: [
        { season: 2025, week: 15, team: "KC", outcome: "win" },
        { season: 2026, week: 1, team: "JAX", outcome: "win" },
      ],
    };
    expect(nextUnpickedWeek(mixed, 2026, 1)).toBe(2);
  });

  it("does not run past the end of the season", () => {
    expect(nextUnpickedWeek(picks(18), 2026, 18)).toBe(18);
  });

  it("uses the highest pick, not the count, when weeks are skipped", () => {
    expect(nextUnpickedWeek(picks(1, 5), 2026, 2)).toBe(6);
  });
});
