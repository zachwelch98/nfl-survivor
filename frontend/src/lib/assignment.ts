// A TypeScript port of backend/survivor/optimize/assignment.py.
//
// Why this exists in two languages: the what-if planner lets you lock a team
// into a week and immediately see the optimal plan for every other week. Locks
// don't change the win probabilities at all -- only which cells the optimizer
// may use -- so the expensive half (Monte Carlo + the ML fit) runs once at
// export time and ships as survival_matrix.json. Re-solving is then pure
// combinatorics over ~18x32 numbers, which is microseconds here and a ~2s
// Python round-trip otherwise.
//
// The Python module is the reference implementation. This must agree with it
// exactly, and tests/test_optimizer.py generates the fixtures that prove it
// does (see assignment.test.ts).
//
// Same contract as the original: this is deliberately agnostic about *why* a
// (week, team) pair is unavailable -- already used, on bye, locked out by
// another week's lock, no game scheduled. The caller expresses all of it by
// omitting the pair from `survivalProb`. Anything absent costs +infinity.

export interface WeekPick {
  week: number;
  team: string;
  survivalProb: number;
}

export interface AssignmentResult {
  picks: WeekPick[]; // sorted by week
  totalSurvivalProb: number; // product of every pick's survivalProb
  totalLogSurvival: number; // sum of logs -- what is actually maximized
}

export class InfeasibleAssignmentError extends Error {
  readonly week?: number;
  constructor(message: string, week?: number) {
    super(message);
    this.name = "InfeasibleAssignmentError";
    this.week = week;
  }
}

/** Key for the `survivalProb` map. Mirrors Python's `(week, team)` tuple key. */
export function probKey(week: number, team: string): string {
  return `${week}:${team}`;
}

// Infeasible cells carry a large *finite* cost rather than Infinity. The
// algorithm below maintains dual potentials by subtraction, and Infinity minus
// Infinity is NaN, which would silently poison the whole solve. Real costs are
// -log(p) for p in (0, 1], so a genuine cell is never worse than ~40 even for
// absurdly small probabilities; BIG is 7+ orders of magnitude clear of that,
// and 18 rows of it (1.8e10) is far inside the range where doubles still count
// exactly. Any chosen cell at or above BIG_THRESHOLD means no feasible
// assignment existed.
const BIG = 1e9;
const BIG_THRESHOLD = BIG / 2;

/**
 * cost[i][j] = -log(survivalProb[weeks[i], teams[j]]), or BIG if that pair is
 * unavailable. A probability of exactly 0 is a legitimate value -- a team the
 * model gives literally no chance -- and maps to BIG rather than throwing;
 * only a value outside [0, 1] indicates a caller bug.
 */
export function buildCostMatrix(
  weeks: number[],
  teams: string[],
  survivalProb: Map<string, number>,
): number[][] {
  const cost: number[][] = [];
  for (const w of weeks) {
    const row = new Array<number>(teams.length).fill(BIG);
    for (let j = 0; j < teams.length; j++) {
      const p = survivalProb.get(probKey(w, teams[j]));
      if (p === undefined) continue;
      if (!(p >= 0 && p <= 1)) {
        throw new Error(`survivalProb for (week=${w}, team=${teams[j]}) must be in [0, 1], got ${p}`);
      }
      if (p > 0) row[j] = -Math.log(p);
    }
    cost.push(row);
  }
  return cost;
}

/**
 * Rectangular linear sum assignment (Jonker-Volgenant shortest augmenting
 * path with dual potentials), rows <= cols. Returns, for each row, the column
 * assigned to it. O(n^2 * m): with 18 weeks and 32 teams that is ~10k
 * operations, so the whole what-if grid re-solves faster than a repaint.
 *
 * This is the same family of algorithm scipy.optimize.linear_sum_assignment
 * uses, which is why the two agree on total cost.
 */
export function linearSumAssignment(cost: number[][]): number[] {
  const n = cost.length;
  if (n === 0) return [];
  const m = cost[0].length;
  if (m < n) throw new Error(`linearSumAssignment requires cols >= rows, got ${n} rows and ${m} cols`);

  // 1-indexed working arrays, with column 0 as the sentinel "unassigned" slot.
  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const colToRow = new Int32Array(m + 1); // colToRow[j] = row matched to column j
  const way = new Int32Array(m + 1); // predecessor column on the augmenting path

  for (let i = 1; i <= n; i++) {
    colToRow[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(Infinity);
    const used = new Uint8Array(m + 1);

    do {
      used[j0] = 1;
      const i0 = colToRow[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      // rows <= cols guarantees an unused column remains, so j1 is always set.
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[colToRow[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (colToRow[j0] !== 0);

    // Walk the augmenting path back, flipping matches as we go.
    do {
      const j1 = way[j0];
      colToRow[j0] = colToRow[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const rowToCol = new Array<number>(n).fill(-1);
  for (let j = 1; j <= m; j++) {
    if (colToRow[j] !== 0) rowToCol[colToRow[j] - 1] = j - 1;
  }
  return rowToCol;
}

/**
 * The exact, provably-optimal pick sequence given fixed win probabilities.
 * Every week gets a distinct team, maximizing the product of survival
 * probabilities across the whole set.
 *
 * Maximizing the product is the same as maximizing the sum of logs, which is
 * exactly a bipartite assignment problem -- no search, no greedy per-week
 * maximization (which is provably worse: it spends strong teams early and
 * strands you with coin flips in December).
 */
export function solveOptimalAssignment(
  weeks: number[],
  teams: string[],
  survivalProb: Map<string, number>,
): AssignmentResult {
  if (weeks.length === 0) {
    return { picks: [], totalSurvivalProb: 1, totalLogSurvival: 0 };
  }
  if (teams.length < weeks.length) {
    throw new InfeasibleAssignmentError(
      `${weeks.length} weeks to fill but only ${teams.length} teams available — cannot assign a distinct team to each week`,
    );
  }

  const cost = buildCostMatrix(weeks, teams, survivalProb);
  const rowToCol = linearSumAssignment(cost);

  const picks: WeekPick[] = [];
  let totalLog = 0;
  for (let i = 0; i < weeks.length; i++) {
    const j = rowToCol[i];
    const c = j >= 0 ? cost[i][j] : BIG;
    if (c >= BIG_THRESHOLD) {
      throw new InfeasibleAssignmentError(`no available team for week ${weeks[i]}`, weeks[i]);
    }
    totalLog += -c;
    picks.push({ week: weeks[i], team: teams[j], survivalProb: Math.exp(-c) });
  }

  picks.sort((a, b) => a.week - b.week);
  return { picks, totalSurvivalProb: Math.exp(totalLog), totalLogSurvival: totalLog };
}

/**
 * For every playable (week, team) cell, how much total survival probability
 * you give up by pinning that team into that week — in log terms, so it is
 * additive and always >= 0.
 *
 * This generalizes the backend's `cost_of_deviation_by_team` (optimize/kbest.py)
 * from a single week to the whole grid: pin the cell, solve the residual
 * problem on the remaining weeks and teams, compare against the baseline.
 *
 * `baselineLogSurvival` MUST be the optimum of the same problem being passed
 * in — i.e. already constrained by whatever locks are active. Measuring
 * deviations against a plan the user's own locks have made unreachable
 * produces numbers that look precise and mean nothing.
 *
 * `restrictToWeeks` prices only those weeks. The planner uses it to ask the
 * question the dropdown actually poses — "holding my *other* locks, what does
 * each choice for this week cost?" — which needs a different baseline per
 * week, and so cannot be answered by one pass over the whole grid.
 */
export function costOfLocking(
  weeks: number[],
  teams: string[],
  survivalProb: Map<string, number>,
  baselineLogSurvival: number,
  restrictToWeeks?: number[],
): Map<string, number> {
  const out = new Map<string, number>();
  if (weeks.length === 0) return out;
  const priced = restrictToWeeks ? weeks.filter((w) => restrictToWeeks.includes(w)) : weeks;

  for (const w of priced) {
    const otherWeeks = weeks.filter((x) => x !== w);
    for (const t of teams) {
      const p = survivalProb.get(probKey(w, t));
      if (p === undefined || p <= 0) continue; // not playable here; nothing to price

      const otherTeams = teams.filter((x) => x !== t);
      let residualLog: number;
      try {
        residualLog = solveOptimalAssignment(otherWeeks, otherTeams, survivalProb).totalLogSurvival;
      } catch (e) {
        if (e instanceof InfeasibleAssignmentError) continue; // pinning here breaks the season
        throw e;
      }
      // Clamp at 0: the baseline is by construction an upper bound, so any
      // negative here is float noise, never a genuinely better plan.
      out.set(probKey(w, t), Math.max(0, baselineLogSurvival - (Math.log(p) + residualLog)));
    }
  }
  return out;
}
