import { describe, expect, it } from "vitest";
import matrixJson from "../../public/data/survival_matrix.json";
import type { ScenarioLock, SurvivalEntry, SurvivalMatrix } from "../data/types";
import { probKey, solveOptimalAssignment } from "./assignment";
import type { AssignmentResult } from "./assignment";
import { applyLocks, diffPlans, solveScenario, validateLocks } from "./planner";

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

  // Weeks are taken from the matrix, never named. The exported bundle only
  // covers weeks from the current one forward, so a literal week number is a
  // fixture with an expiry date -- these tests were pinned to weeks 2, 4, 7
  // and 11, and started failing the moment the season moved past week 2.
  const firstWeek = weeks[0];
  const secondWeek = weeks[1];
  const midWeek = weeks[Math.floor(weeks.length / 2)];
  const lateWeek = weeks[weeks.length - 2];

  it("with no locks, matches the unconstrained optimum", () => {
    const s = solveScenario(entries, weeks, teams, []);
    expect(s.error).toBeNull();
    expect(s.costOfLocks).toBe(0);
    expect(s.scenario?.totalSurvivalProb).toBeCloseTo(s.baseline?.totalSurvivalProb as number, 15);
  });

  it("honors a lock, and never reports it as free money", () => {
    const base = solveScenario(entries, weeks, teams, []);
    const optimalMid = base.scenario?.picks.find((p) => p.week === midWeek)?.team;

    // Lock a different team into week 7 than the one the optimizer wanted.
    const alternative = entries.find(
      (e) => e.week === midWeek && e.team !== optimalMid && e.survival_prob > 0.5,
    ) as SurvivalEntry;
    const locks: ScenarioLock[] = [{ season: 2026, week: midWeek, team: alternative.team }];

    const s = solveScenario(entries, weeks, teams, locks);
    expect(s.error).toBeNull();
    expect(s.scenario?.picks.find((p) => p.week === midWeek)?.team).toBe(alternative.team);

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
    const pick = base.scenario?.picks.find((p) => p.week === secondWeek);
    const s = solveScenario(entries, weeks, teams, [
      { season: 2026, week: secondWeek, team: pick?.team as string },
    ]);
    expect(s.costOfLocks).toBeCloseTo(0, 15);
  });

  it("stacks multiple locks and still fills every other week", () => {
    const a = entries.find((e) => e.week === firstWeek)?.team as string;
    const b = entries.find((e) => e.week === lateWeek && e.team !== a)?.team as string;
    const locks: ScenarioLock[] = [
      { season: 2026, week: firstWeek, team: a },
      { season: 2026, week: lateWeek, team: b },
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
    const team = entries.find((e) => e.week === firstWeek)?.team as string;
    const alsoLater = entries.find((e) => e.week > firstWeek && e.team === team) as SurvivalEntry;
    const s = solveScenario(entries, weeks, teams, [
      { season: 2026, week: firstWeek, team },
      { season: 2026, week: alsoLater.week, team },
    ]);
    expect(s.error).toMatch(/no complete plan/i);
    expect(s.scenario).toBeNull();
    expect(s.costOfLocks).toBeNull();
    // And validateLocks would have caught it first.
    const v = validateLocks(
      [
        { season: 2026, week: firstWeek, team },
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
    const locks: ScenarioLock[] = [
      { season: 2026, week: secondWeek, team: entries.find((e) => e.week === secondWeek)?.team as string },
    ];
    const s = solveScenario(entries, weeks, teams, locks);
    const direct = solveOptimalAssignment(weeks, teams, applyLocks(entries, locks));
    expect(s.scenario?.totalLogSurvival).toBeCloseTo(direct.totalLogSurvival, 12);
  });
});

describe("diffPlans", () => {
  const entries = matrix.entries;
  const weeks = matrix.weeks;
  const teams = matrix.teams;

  // Weeks are taken from the matrix, never named. The exported bundle only
  // covers weeks from the current one forward, so a literal week number is a
  // fixture with an expiry date -- these tests were pinned to weeks 2, 4, 7
  // and 11, and started failing the moment the season moved past week 2.
  const firstWeek = weeks[0];
  const midWeek = weeks[Math.floor(weeks.length / 2)];

  it("reports no changes when there are no locks", () => {
    const s = solveScenario(entries, weeks, teams, []);
    const d = diffPlans(s.baseline as AssignmentResult, s.scenario as AssignmentResult);
    expect(d.changes).toEqual([]);
    expect(d.dropped).toEqual([]);
    expect(d.added).toEqual([]);
  });

  it("names the locked week and the knock-on week, not just the locked one", () => {
    // The motivating case: locking a team into the current week when the plan was
    // already using in a later week frees that later week too.
    const base = solveScenario(entries, weeks, teams, []);
    const basePicks = base.scenario as AssignmentResult;
    const firstWeekTeam = basePicks.picks.find((p) => p.week === firstWeek)?.team as string;

    // Find a team used later in the plan that also plays in week 1.
    const laterPick = basePicks.picks.find(
      (p) => p.week > firstWeek && entries.some((e) => e.week === firstWeek && e.team === p.team),
    );
    if (!laterPick) return; // schedule-dependent; nothing to assert

    const s = solveScenario(entries, weeks, teams, [
      { season: 2026, week: firstWeek, team: laterPick.team },
    ]);
    const d = diffPlans(base.scenario as AssignmentResult, s.scenario as AssignmentResult);

    const changedWeeks = d.changes.map((c) => c.week);
    expect(changedWeeks).toContain(firstWeek);
    expect(changedWeeks).toContain(laterPick.week); // the knock-on
    expect(d.changes.find((c) => c.week === firstWeek)?.from).toBe(firstWeekTeam);
    expect(d.changes.find((c) => c.week === firstWeek)?.to).toBe(laterPick.team);
  });

  it("reports the displaced team as dropped when it isn't redeployed", () => {
    const base = solveScenario(entries, weeks, teams, []);
    const basePicks = base.scenario as AssignmentResult;
    const laterPick = basePicks.picks.find(
      (p) => p.week > firstWeek && entries.some((e) => e.week === firstWeek && e.team === p.team),
    );
    if (!laterPick) return;

    const s = solveScenario(entries, weeks, teams, [
      { season: 2026, week: firstWeek, team: laterPick.team },
    ]);
    const d = diffPlans(base.scenario as AssignmentResult, s.scenario as AssignmentResult);

    // Whatever leaves the plan must actually be gone from it, and whatever
    // arrives must actually be in it -- the counts have to balance.
    expect(d.dropped.length).toBe(d.added.length);
    for (const t of d.dropped) {
      expect(s.scenario?.picks.some((p) => p.team === t)).toBe(false);
    }
    for (const t of d.added) {
      expect(basePicks.picks.some((p) => p.team === t)).toBe(false);
    }
  });

  it("every changed week is a real swap, and unchanged weeks are omitted", () => {
    const base = solveScenario(entries, weeks, teams, []);
    const pickMid = (base.scenario as AssignmentResult).picks.find((p) => p.week === midWeek);
    const other = entries.find(
      (e) => e.week === midWeek && e.team !== pickMid?.team && e.survival_prob > 0.5,
    ) as SurvivalEntry;
    const s = solveScenario(entries, weeks, teams, [{ season: 2026, week: midWeek, team: other.team }]);
    const d = diffPlans(base.scenario as AssignmentResult, s.scenario as AssignmentResult);

    expect(d.changes.length).toBeGreaterThan(0);
    for (const c of d.changes) {
      expect(c.from).not.toBe(c.to);
      expect(s.scenario?.picks.find((p) => p.week === c.week)?.team).toBe(c.to);
      expect(base.scenario?.picks.find((p) => p.week === c.week)?.team).toBe(c.from);
    }
    const changedWeeks = new Set(d.changes.map((c) => c.week));
    for (const w of weeks) {
      if (changedWeeks.has(w)) continue;
      expect(s.scenario?.picks.find((p) => p.week === w)?.team).toBe(
        base.scenario?.picks.find((p) => p.week === w)?.team,
      );
    }
  });
});
