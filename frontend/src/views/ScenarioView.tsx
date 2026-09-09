// What-if planner: lock teams into specific weeks, and the optimizer fills in
// every other week at maximum survival probability.
//
// The whole solve runs in the browser. Locks don't change any win probability
// -- only which (week, team) cells the optimizer may use -- so the expensive
// half (Monte Carlo over every remaining game, plus the Method 3 fit) happens
// once at `make export` time and ships as survival_matrix.json. Everything
// here is arithmetic over ~18x32 numbers, which is why a lock toggle is
// instant instead of a ~2s Python round-trip. See src/lib/assignment.ts.

import { useEffect, useMemo, useState } from "react";
import { AsyncView } from "../components/AsyncView";
import { Card, StatTile } from "../components/Card";
import { ProbPill } from "../components/ProbBar";
import { getSeasonMeta, getSeasonPlan, getSurvivalMatrix } from "../data/source";
import { burnedTeams, locksForSeason } from "../data/scenario";
import type { SeasonMeta, SeasonPlan, SurvivalEntry, SurvivalMatrix } from "../data/types";
import { costOfLocking, probKey, solveOptimalAssignment } from "../lib/assignment";
import { applyLocks, diffPlans, solveScenario, validateLocks } from "../lib/planner";
import { teamName } from "../lib/teams";
import { useAsync } from "../lib/useAsync";
import { useScenarioState } from "../lib/useScenarioState";
import { useUserState } from "../lib/useUserState";

const BUTTON =
  "rounded-md border border-slate-200 px-3 py-1.5 text-sm font-medium text-ink hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors";

interface Bundle {
  matrix: SurvivalMatrix;
  plan: SeasonPlan;
  meta: SeasonMeta;
}

function pct(p: number, digits = 3): string {
  return `${(p * 100).toFixed(digits)}%`;
}

/**
 * Turn a log-probability cost into the share of your season odds it gives up.
 *
 * costOfLocking works in log space because that is what the optimizer sums,
 * but "0.41 log-points" means nothing to a person. A cost of `c` multiplies
 * your total season survival by exp(-c), so `1 - exp(-c)` is the fraction of
 * your odds the choice throws away — a real percentage, and the thing anyone
 * actually wants to know.
 */
function costShare(logCost: number): number {
  return 1 - Math.exp(-logCost);
}

export default function ScenarioView() {
  const bundle = useAsync<Bundle>(
    () =>
      Promise.all([getSurvivalMatrix(), getSeasonPlan(), getSeasonMeta()]).then(
        ([matrix, plan, meta]) => ({ matrix, plan, meta }),
      ),
    [],
  );

  return (
    <AsyncView state={bundle}>{(data) => <Planner bundle={data} />}</AsyncView>
  );
}

function Planner({ bundle }: { bundle: Bundle }) {
  const { matrix, plan, meta } = bundle;
  const season = matrix.season;
  const currentWeek = matrix.current_week;

  // Both hooks are owned here and passed down: each useUserState/useScenarioState
  // call is independent local state with no Context behind it, so calling them
  // in sibling components would silently desynchronize them.
  const { state: userState } = useUserState();
  const { state: scenario, lock, unlock, clearAll, importPicks } = useScenarioState();
  const [showGrid, setShowGrid] = useState(false);

  const burned = useMemo(
    () => burnedTeams(userState, season, currentWeek),
    [userState, season, currentWeek],
  );

  const entries = matrix.entries;
  const weeks = matrix.weeks;
  const available = useMemo(
    () => matrix.teams.filter((t) => !burned.has(t)),
    [matrix.teams, burned],
  );

  const entryByKey = useMemo(() => {
    const m = new Map<string, SurvivalEntry>();
    for (const e of entries) m.set(probKey(e.week, e.team), e);
    return m;
  }, [entries]);

  const teamsByWeek = useMemo(() => {
    const m = new Map<number, SurvivalEntry[]>();
    for (const e of entries) {
      if (burned.has(e.team)) continue;
      const list = m.get(e.week);
      if (list) list.push(e);
      else m.set(e.week, [e]);
    }
    return m;
  }, [entries, burned]);

  // Infeasibility here is user-caused and fixable, so each rejected lock has
  // to be named rather than collapsing into a blank plan. See lib/planner.ts.
  const {
    valid: validLocks,
    problems: lockProblems,
    stale: staleLocks,
  } = useMemo(
    () => validateLocks(locksForSeason(scenario, season), weeks, entries, burned),
    [scenario, season, weeks, entries, burned],
  );

  // --- Solve: unconstrained baseline, then the locked scenario ---------------
  const solved = useMemo(
    () => solveScenario(entries, weeks, available, validLocks),
    [entries, weeks, available, validLocks],
  );

  // Per-week cost of every alternative, each priced against the *other* locks
  // held fixed — the question the dropdown actually asks.
  const costs = useMemo(() => {
    const out = new Map<string, number>();
    if (!solved.scenario) return out;
    for (const w of weeks) {
      const otherLocks = validLocks.filter((l) => l.week !== w);
      const probs = applyLocks(entries, otherLocks);
      try {
        const base = solveOptimalAssignment(weeks, available, probs).totalLogSurvival;
        for (const [k, v] of costOfLocking(weeks, available, probs, base, [w])) out.set(k, v);
      } catch {
        // This week can't be priced under the other locks; leave it blank
        // rather than showing a number that isn't measured against anything.
      }
    }
    return out;
  }, [entries, validLocks, weeks, available, solved.scenario]);

  // Cross-language canary. The browser solver and the Python solver must agree
  // on the same instance; if they ever diverge, this page is quietly lying.
  useEffect(() => {
    if (!import.meta.env.DEV || plan.total_survival_prob === null) return;
    try {
      const exportAvailable = matrix.teams.filter((t) => !meta.used_teams.includes(t));
      const check = solveOptimalAssignment(weeks, exportAvailable, applyLocks(entries, []));
      const diff = Math.abs(check.totalSurvivalProb - plan.total_survival_prob);
      if (diff > 1e-9) {
        console.warn(
          `[what-if] browser solver disagrees with season_plan.json: ${check.totalSurvivalProb} vs ${plan.total_survival_prob} (diff ${diff}). ` +
            `src/lib/assignment.ts and backend/survivor/optimize/assignment.py have diverged.`,
        );
      }
    } catch (e) {
      console.warn("[what-if] cross-check against season_plan.json could not run:", e);
    }
  }, [matrix.teams, meta.used_teams, weeks, entries, plan.total_survival_prob]);

  const lockedByWeek = useMemo(
    () => new Map(validLocks.map((l) => [l.week, l.team])),
    [validLocks],
  );

  const diff = useMemo(
    () => (solved.baseline && solved.scenario ? diffPlans(solved.baseline, solved.scenario) : null),
    [solved.baseline, solved.scenario],
  );
  const changedByWeek = useMemo(
    () => new Map((diff?.changes ?? []).map((c) => [c.week, c])),
    [diff],
  );

  const { baseline, scenario: scenarioPlan, costOfLocks, error } = solved;
  const pendingImportable = userState.picks.some(
    (p) => p.season === season && p.week >= currentWeek && p.outcome === "pending",
  );

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">What-If Planner</h1>
        <p className="text-sm text-slate-500 mt-1">
          Commit a team to a week and the optimizer re-solves every other week around it. Probabilities
          come from the last <code className="text-xs">make export</code> — week {currentWeek} from posted
          lines, later weeks from the Monte Carlo + ML blend with horizon decay applied.
        </p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatTile
          label="Best possible"
          value={baseline ? pct(baseline.totalSurvivalProb) : "—"}
        />
        <StatTile
          label="With your locks"
          value={scenarioPlan ? pct(scenarioPlan.totalSurvivalProb) : "—"}
          tone={costOfLocks !== null && costOfLocks > 0 ? "bad" : "good"}
        />
        <StatTile
          label="Cost of your locks"
          value={
            costOfLocks === null || baseline === null
              ? "—"
              : costOfLocks <= 1e-12
                ? "none"
                : `−${((costOfLocks / baseline.totalSurvivalProb) * 100).toFixed(1)}%`
          }
          tone={costOfLocks !== null && costOfLocks > 1e-12 ? "bad" : "default"}
        />
        <StatTile label="Locks active" value={`${validLocks.length} / ${weeks.length}`} />
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}
      {lockProblems.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
          <div className="font-medium mb-1">Some locks can't be applied:</div>
          <ul className="list-disc pl-5 space-y-0.5">
            {lockProblems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}
      {staleLocks.length > 0 && (
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
          Ignoring {staleLocks.length} lock{staleLocks.length === 1 ? "" : "s"} for week
          {staleLocks.length === 1 ? "" : "s"} {staleLocks.map((l) => l.week).join(", ")} — those weeks are
          no longer in the exported plan.
        </div>
      )}
      {burned.size > 0 && (
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
          Out of the pool from My Picks: {[...burned].sort().join(", ")}
        </div>
      )}

      {diff && diff.changes.length > 0 && baseline && scenarioPlan && (
        <Card
          title="What your locks changed"
          subtitle="Every week not listed here is unaffected."
        >
          <ul className="space-y-1.5 text-sm">
            {diff.changes.map((c) => (
              <li key={c.week} className="flex flex-wrap items-baseline gap-x-2">
                <span className="text-slate-500 w-16 shrink-0">Week {c.week}</span>
                <span className="text-slate-400 line-through">
                  {c.from} {Math.round(c.fromProb * 100)}%
                </span>
                <span className="text-slate-400">→</span>
                <span className="font-medium">
                  {c.to} {Math.round(c.toProb * 100)}%
                </span>
                <span className="text-slate-400 text-xs">{teamName(c.to)}</span>
                {lockedByWeek.get(c.week) === c.to ? (
                  <span className="text-xs text-amber-700">your lock</span>
                ) : (
                  <span className="text-xs text-slate-400">knock-on effect</span>
                )}
              </li>
            ))}
          </ul>

          {(diff.dropped.length > 0 || diff.added.length > 0) && (
            <div className="mt-3 pt-3 border-t border-slate-100 text-sm text-slate-600 space-y-1">
              {diff.dropped.length > 0 && (
                <div>
                  <span className="font-medium">Leaves the plan:</span> {diff.dropped.join(", ")}
                </div>
              )}
              {diff.added.length > 0 && (
                <div>
                  <span className="font-medium">Enters the plan:</span> {diff.added.join(", ")}
                </div>
              )}
              <p className="text-xs text-slate-400 pt-1">
                A team leaving the plan isn't wasted — there are {matrix.teams.length} teams and only{" "}
                {weeks.length} weeks, so {matrix.teams.length - weeks.length} go unused in any plan.
                It simply wasn't better than what already had each remaining slot.
              </p>
            </div>
          )}

          <div className="mt-3 pt-3 border-t border-slate-100 text-sm">
            Season survival: <span className="tabular-nums">{pct(baseline.totalSurvivalProb)}</span>{" "}
            <span className="text-slate-400">→</span>{" "}
            <span className="tabular-nums font-medium">{pct(scenarioPlan.totalSurvivalProb)}</span>
            {costOfLocks !== null && costOfLocks > 1e-12 && (
              <span className="text-loss">
                {" "}— giving up {((costOfLocks / baseline.totalSurvivalProb) * 100).toFixed(1)}% of
                your odds
              </span>
            )}
          </div>
        </Card>
      )}

      <Card
        title="Your season"
        subtitle="Locked weeks are yours; the rest is the optimal fill given those commitments."
      >
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <button
            className={BUTTON}
            onClick={() => importPicks(userState, season, currentWeek)}
            disabled={!pendingImportable}
            title={
              pendingImportable
                ? "Copy pending future picks from My Picks into these locks"
                : "No pending future picks in My Picks to import"
            }
          >
            Import from My Picks
          </button>
          <button className={BUTTON} onClick={() => clearAll(season)} disabled={validLocks.length === 0}>
            Clear all locks
          </button>
          <button className={BUTTON} onClick={() => setShowGrid((v) => !v)}>
            {showGrid ? "Hide" : "Show"} cost-to-lock grid
          </button>
          <span className="text-xs text-slate-400">
            Locks are a scratchpad — they never change My Picks.
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-slate-400 border-b border-slate-200">
                <th className="py-2 pr-3">Week</th>
                <th className="py-2 px-3">Team</th>
                <th className="py-2 px-3">Change</th>
                <th className="py-2 px-3">Matchup</th>
                <th className="py-2 px-3">Survival</th>
                <th className="py-2 pl-3">Lock</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {weeks.map((w) => {
                const lockedTeam = lockedByWeek.get(w);
                const changed = changedByWeek.get(w);
                const pick = scenarioPlan?.picks.find((p) => p.week === w);
                const team = lockedTeam ?? pick?.team ?? null;
                const entry = team ? entryByKey.get(probKey(w, team)) : undefined;
                const options = (teamsByWeek.get(w) ?? [])
                  .filter((e) => {
                    const heldElsewhere = validLocks.find((l) => l.team === e.team && l.week !== w);
                    return !heldElsewhere;
                  })
                  .map((e) => ({ e, cost: costs.get(probKey(w, e.team)) }))
                  .sort((a, b) => (a.cost ?? Infinity) - (b.cost ?? Infinity) || b.e.survival_prob - a.e.survival_prob);

                return (
                  <tr
                    key={w}
                    className={
                      lockedTeam ? "bg-amber-50/60" : changed ? "bg-blue-50/40" : undefined
                    }
                  >
                    <td className="py-1.5 pr-3 text-slate-500 whitespace-nowrap">Week {w}</td>
                    <td className="py-1.5 px-3">
                      <select
                        className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm min-w-56"
                        value={lockedTeam ?? ""}
                        onChange={(ev) => {
                          const v = ev.target.value;
                          if (v) lock({ season, week: w, team: v });
                          else unlock(season, w);
                        }}
                      >
                        <option value="">
                          {team ? `(optimizer picks ${team})` : "(no team available)"}
                        </option>
                        {options.map(({ e, cost }) => (
                          <option key={e.team} value={e.team}>
                            {e.team} · {Math.round(e.survival_prob * 100)}%
                            {cost !== undefined && cost > 1e-9
                              ? ` — costs ${(costShare(cost) * 100).toFixed(1)}% of your season odds`
                              : " — best available"}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="py-1.5 px-3 whitespace-nowrap">
                      {changed && (
                        <div className="text-xs text-slate-400">
                          was {changed.from} ({Math.round(changed.fromProb * 100)}%)
                        </div>
                      )}
                    </td>
                    <td className="py-1.5 px-3 text-slate-500 whitespace-nowrap">
                      {entry ? (
                        <>
                          {entry.is_home ? "vs" : "@"} {entry.opponent}{" "}
                          <span className="text-slate-400">{teamName(entry.opponent)}</span>
                        </>
                      ) : (
                        <span className="text-slate-300">—</span>
                      )}
                    </td>
                    <td className="py-1.5 px-3">
                      <ProbPill value={entry ? entry.survival_prob : null} />
                    </td>
                    <td className="py-1.5 pl-3">
                      {lockedTeam ? (
                        <button
                          className="text-xs font-medium text-amber-700 hover:underline"
                          onClick={() => unlock(season, w)}
                        >
                          🔒 locked · clear
                        </button>
                      ) : (
                        <button
                          className="text-xs text-slate-400 hover:text-ink hover:underline disabled:opacity-40"
                          disabled={!pick}
                          onClick={() => pick && lock({ season, week: w, team: pick.team })}
                        >
                          lock this
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {showGrid && (
        <CostGrid weeks={weeks} teams={available} costs={costs} lockedByWeek={lockedByWeek} />
      )}
    </div>
  );
}

/**
 * The whole decision surface at once: for every week and every team, what
 * committing to that cell would cost. Only affordable because the solve is
 * local — this is ~600 assignment solves, recomputed on every lock change.
 */
function CostGrid({
  weeks,
  teams,
  costs,
  lockedByWeek,
}: {
  weeks: number[];
  teams: string[];
  costs: Map<string, number>;
  lockedByWeek: Map<number, string>;
}) {
  // Scale colour by the displayed share, not the raw log cost -- otherwise a
  // handful of hopeless cells saturate the scale and every real distinction
  // between the affordable choices washes out to the same pale pink.
  const max = useMemo(() => {
    let m = 0;
    for (const v of costs.values()) if (Number.isFinite(v)) m = Math.max(m, 1 - Math.exp(-v));
    return m || 1;
  }, [costs]);

  return (
    <Card
      title="Cost-to-lock grid"
      subtitle="The share of your total season odds you give up by committing each team to each week, holding your other locks fixed. Darker is worse; blank means no game."
    >
      <div className="overflow-x-auto">
        <table className="text-[10px] tabular-nums border-separate border-spacing-0.5">
          <thead>
            <tr>
              <th className="sticky left-0 bg-white pr-2 text-left font-medium text-slate-400">Wk</th>
              {teams.map((t) => (
                <th key={t} className="px-0.5 font-medium text-slate-400 w-8">
                  {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {weeks.map((w) => (
              <tr key={w}>
                <td className="sticky left-0 bg-white pr-2 text-slate-500 whitespace-nowrap">{w}</td>
                {teams.map((t) => {
                  const c = costs.get(probKey(w, t));
                  const isLocked = lockedByWeek.get(w) === t;
                  if (c === undefined) {
                    return <td key={t} className="text-center text-slate-200">·</td>;
                  }
                  const intensity = Math.min(1, costShare(c) / max);
                  return (
                    <td
                      key={t}
                      title={`Week ${w} · ${t}: ${c === 0 ? "optimal choice" : `gives up ${(costShare(c) * 100).toFixed(1)}% of your season odds`}`}
                      className={`text-center rounded-sm ${isLocked ? "ring-1 ring-amber-500" : ""}`}
                      style={{
                        backgroundColor: `rgba(220, 38, 38, ${(intensity * 0.85).toFixed(3)})`,
                        color: intensity > 0.5 ? "white" : "#475569",
                      }}
                    >
                      {c < 1e-9 ? "·" : (costShare(c) * 100).toFixed(0)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
