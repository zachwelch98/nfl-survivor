import { describe, expect, it } from "vitest";
import matrixJson from "../../public/data/survival_matrix.json";
import type { ScenarioLock, SurvivalEntry, SurvivalMatrix } from "../data/types";
import { probKey, solveOptimalAssignment } from "./assignment";
import { applyLocks, solveScenario, validateLocks } from "./planner";

const matrix = matrixJson as unknown as SurvivalMatrix;

function entry(week: number, team: string, prob: number, opponent = "OPP"): SurvivalEntry {
  return {
    week,
    team,
    opponent,
    is_home: true,
    game_id: `g${week}_${team}`,
    survival_prob: prob,
    weeks_ahead: 0,
    source: "model",
  };
}

describe("applyLocks", () => {
  const entries = [
    entry(1, "A", 0.9),
    entry(1, "B", 0.8),
    entry(2, "A", 0.7),
    entry(2, "B", 0.6),
  ];

  it("is a no-op with no locks", () => {
    expect(applyLocks(entries, []).size).toBe(4);
  });

  it("removes every other team from the locked week", () => {
    const m = applyLocks(entries, [{ season: 2026, week: 1, team: "A" }]);
    expect(m.has(probKey(1, "A"))).toBe(true);
    expect(m.has(probKey(1, "B"))).toBe(false);
  });

  it("reserves the locked team out of every other week", () => {
    const m = applyLocks(entries, [{ season: 2026, week: 1, team: "A" }]);
    expect(m.has(probKey(2, "A"))).toBe(false);
    expect(m.has(probKey(2, "B"))).toBe(true);
  });
});

describe("validateLocks", () => {
  const entries = [entry(1, "A", 0.9), entry(1, "B", 0.8), entry(2, "A", 0.7), entry(2, "B", 0.6)];
  const weeks = [1, 2];

  it("accepts a lock on a team that plays that week", () => {
    const v = validateLocks([{ season: 2026, week: 1, team: "A" }], weeks, entries, new Set());
    expect(v.valid).toHaveLength(1);
    expect(v.problems).toEqual([]);
  });

  it("rejects a lock on a bye week by name", () => {
    const v = validateLocks([{ season: 2026, week: 1, team: "ZZ" }], weeks, entries, new Set());
    expect(v.valid).toEqual([]);
    expect(v.problems[0]).toMatch(/Week 1: ZZ has no game that week \(bye\)/);
  });

  it("rejects a lock on a burned team by name", () => {
    const v = validateLocks([{ season: 2026, week: 1, team: "A" }], weeks, entries, new Set(["A"]));
    expect(v.valid).toEqual([]);
    expect(v.problems[0]).toMatch(/already used/);
  });

  it("rejects the second of two locks on the same team", () => {
    const locks: ScenarioLock[] = [
      { season: 2026, week: 1, team: "A" },
      { season: 2026, week: 2, team: "A" },
    ];
    const v = validateLocks(locks, weeks, entries, new Set());
    expect(v.valid).toHaveLength(1);
    expect(v.valid[0].week).toBe(1); // the earlier week is kept
    expect(v.problems[0]).toMatch(/locked into both week 1 and week 2/);
  });

  it("treats a lock on a week outside the export as stale, not an error", () => {
    const v = validateLocks([{ season: 2026, week: 9, team: "A" }], weeks, entries, new Set());
    expect(v.stale).toHaveLength(1);
    expect(v.problems).toEqual([]);
  });
});

describe("solveScenario on the real exported season", () => {
  const entries = matrix.entries;
  const weeks = matrix.weeks;
  const teams = matrix.teams;

  it("with no locks, matches the unconstrained optimum", () => {
    const s = solveScenario(entries, weeks, teams, []);
    expect(s.error).toBeNull();
    expect(s.costOfLocks).toBe(0);
    expect(s.scenario?.totalSurvivalProb).toBeCloseTo(s.baseline?.totalSurvivalProb as number, 15);
  });

  it("honors a lock, and never reports it as free money", () => {
    const base = solveScenario(entries, weeks, teams, []);
    const optimalWeek7 = base.scenario?.picks.find((p) => p.week === 7)?.team;

    // Lock a different team into week 7 than the one the optimizer wanted.
    const alternative = entries.find(
      (e) => e.week === 7 && e.team !== optimalWeek7 && e.survival_prob > 0.5,
    ) as SurvivalEntry;
    const locks: ScenarioLock[] = [{ season: 2026, week: 7, team: alternative.team }];

    const s = solveScenario(entries, weeks, teams, locks);
    expect(s.error).toBeNull();
    expect(s.scenario?.picks.find((p) => p.week === 7)?.team).toBe(alternative.team);

    // A lock can only ever cost you probability: the locked problem is the
    // unlocked one with cells removed, so its optimum cannot be higher.
    expect(s.costOfLocks).toBeGreaterThan(0);
    expect(s.scenario?.totalSurvivalProb).toBeLessThan(base.scenario?.totalSurvivalProb as number);
  });

  it("never lets a lock beat the unconstrained optimum, across many locks", () => {
    const baseline = solveScenario(entries, weeks, teams, []).scenario
      ?.totalSurvivalProb as number;

    for (const w of weeks) {
      for (const e of entries.filter((x) => x.week === w).slice(0, 4)) {
        const s = solveScenario(entries, weeks, teams, [{ season: 2026, week: w, team: e.team }]);
        if (!s.scenario) continue;
        expect(s.scenario.totalSurvivalProb).toBeLessThanOrEqual(baseline + 1e-12);
        expect(s.costOfLocks).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it("locking the team the optimizer already chose is free", () => {
    const base = solveScenario(entries, weeks, teams, []);
    const pick = base.scenario?.picks.find((p) => p.week === 4);
    const s = solveScenario(entries, weeks, teams, [
      { season: 2026, week: 4, team: pick?.team as string },
    ]);
    expect(s.costOfLocks).toBeCloseTo(0, 15);
  });

  it("stacks multiple locks and still fills every other week", () => {
    const w2 = entries.find((e) => e.week === 2)?.team as string;
    const w11 = entries.find((e) => e.week === 11 && e.team !== w2)?.team as string;
    const locks: ScenarioLock[] = [
      { season: 2026, week: 2, team: w2 },
      { season: 2026, week: 11, team: w11 },
    ];
    const s = solveScenario(entries, weeks, teams, locks);
    expect(s.error).toBeNull();
    expect(s.scenario?.picks).toHaveLength(weeks.length);
    for (const l of locks) {
      expect(s.scenario?.picks.find((p) => p.week === l.week)?.team).toBe(l.team);
    }
    expect(new Set(s.scenario?.picks.map((p) => p.team)).size).toBe(weeks.length);
  });

  it("reports a specific error when burned teams leave too few to go around", () => {
    const almostAll = teams.slice(0, teams.length - 3); // leave 3 teams for 18 weeks
    const s = solveScenario(entries, weeks, teams.filter((t) => !almostAll.includes(t)), []);
    expect(s.baseline).toBeNull();
    expect(s.error).toMatch(/No complete plan exists/);
  });

  it("surfaces an error rather than misbehaving if handed unvalidated duplicate locks", () => {
    // solveScenario's contract is that locks have already been through
    // validateLocks. If a caller skips that, the same team pinned to two weeks
    // must fail loudly rather than quietly dropping one of them.
    const team = entries.find((e) => e.week === 2)?.team as string;
    const alsoLater = entries.find((e) => e.week > 2 && e.team === team) as SurvivalEntry;
    const s = solveScenario(entries, weeks, teams, [
      { season: 2026, week: 2, team },
      { season: 2026, week: alsoLater.week, team },
    ]);
    expect(s.error).toMatch(/no complete plan/i);
    expect(s.scenario).toBeNull();
    expect(s.costOfLocks).toBeNull();
    // And validateLocks would have caught it first.
    const v = validateLocks(
      [
        { season: 2026, week: 2, team },
        { season: 2026, week: alsoLater.week, team },
      ],
      weeks,
      entries,
      new Set(),
    );
    expect(v.valid).toHaveLength(1);
    expect(v.problems[0]).toMatch(/can only be used once/);
  });

  it("agrees with a direct solve of the same locked table", () => {
    const locks: ScenarioLock[] = [{ season: 2026, week: 3, team: entries.find((e) => e.week === 3)?.team as string }];
    const s = solveScenario(entries, weeks, teams, locks);
    const direct = solveOptimalAssignment(weeks, teams, applyLocks(entries, locks));
    expect(s.scenario?.totalLogSurvival).toBeCloseTo(direct.totalLogSurvival, 12);
  });
});
