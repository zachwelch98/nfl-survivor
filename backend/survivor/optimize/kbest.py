"""Alternative season paths and cost-of-deviation (plan §4).

Two complementary views on "how confident should I be in the optimal pick":

  - `cost_of_deviation_by_team`: for every team playable *this specific
    week*, how much total-season survival probability is given up by
    picking that team now instead of the optimizer's choice — computed by
    fixing that (week, team) pair and re-solving the assignment over
    everything else. This is what actually makes a recommendation
    trustworthy: a tiny cost-of-deviation means the "right" pick barely
    matters, a large one means it matters a lot.
  - `k_best_assignments`: the k best *complete* alternative season plans,
    via Murty's algorithm (1968) — repeatedly partition the solution space
    by forbidding one edge of the previous best solution at a time and
    re-solving, which is the standard method for ranking assignment
    problem solutions exactly (no heuristic, no risk of missing or
    duplicating a solution).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from survivor.optimize.assignment import InfeasibleSeasonError, SeasonPlan, WeekPick, build_cost_matrix, solve_optimal_assignment


def cost_of_deviation_by_team(weeks: list[int], teams: list[str], survival_prob: dict[tuple[int, str], float], current_week: int) -> dict[str, float]:
    """Extra season-long survival probability given up (in log-odds terms,
    i.e. additional -log(survival_prob) cost) by picking each candidate
    team at `current_week` instead of optimally. 0.0 for whichever team the
    unconstrained optimum actually picks that week.
    """
    baseline = solve_optimal_assignment(weeks, teams, survival_prob)
    baseline_cost = -baseline.total_log_survival

    other_weeks = [w for w in weeks if w != current_week]
    deviations: dict[str, float] = {}

    for team in teams:
        p = survival_prob.get((current_week, team))
        if p is None:
            continue  # not playable this week at all
        fixed_cost = -np.log(p)
        other_teams = [t for t in teams if t != team]

        if not other_weeks:
            total_cost = fixed_cost
        else:
            residual_matrix = build_cost_matrix(other_weeks, other_teams, survival_prob)
            try:
                row_ind, col_ind = linear_sum_assignment(residual_matrix)
                residual_cost = residual_matrix[row_ind, col_ind].sum()
                if not np.isfinite(residual_cost):
                    deviations[team] = float("inf")
                    continue
            except ValueError:
                deviations[team] = float("inf")
                continue
            total_cost = fixed_cost + residual_cost

        deviations[team] = max(0.0, float(total_cost - baseline_cost))  # clamp tiny negative fp noise

    return deviations


@dataclass(frozen=True)
class _MurtyNode:
    required: tuple[tuple[int, int], ...]  # (row_idx, col_idx) pairs fixed by earlier partitioning
    forbidden: frozenset[tuple[int, int]]


def _solve_node(cost_matrix: np.ndarray, node: _MurtyNode) -> tuple[float, dict[int, int]] | None:
    n_rows, n_cols = cost_matrix.shape
    working = cost_matrix.copy()
    for r, c in node.forbidden:
        working[r, c] = np.inf
    for r, c in node.required:
        row_mask = np.ones(n_cols, dtype=bool)
        row_mask[c] = False
        working[r, row_mask] = np.inf
        col_mask = np.ones(n_rows, dtype=bool)
        col_mask[r] = False
        working[col_mask, c] = np.inf

    try:
        row_ind, col_ind = linear_sum_assignment(working)
    except ValueError:
        return None
    total_cost = working[row_ind, col_ind].sum()
    if not np.isfinite(total_cost):
        return None
    return float(total_cost), dict(zip(row_ind.tolist(), col_ind.tolist()))


def k_best_assignments(cost_matrix: np.ndarray, k: int) -> list[tuple[float, dict[int, int]]]:
    """The k best distinct complete assignments (by total cost, ascending)
    for a raw cost matrix, via Murty's algorithm. Returns (total_cost,
    {row_idx: col_idx}) pairs — index-space, not team/week labels; see
    `k_best_season_plans` for the labeled wrapper used by callers.
    """
    root = _MurtyNode(required=(), forbidden=frozenset())
    root_result = _solve_node(cost_matrix, root)
    if root_result is None:
        return []

    counter = 0
    heap: list[tuple[float, int, _MurtyNode, dict[int, int]]] = [(root_result[0], counter, root, root_result[1])]
    results: list[tuple[float, dict[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    while heap and len(results) < k:
        cost, _, node, assignment = heapq.heappop(heap)
        key = tuple(sorted(assignment.items()))
        if key in seen:
            continue
        seen.add(key)
        results.append((cost, assignment))

        required_so_far = dict(node.required)
        free_rows = [r for r in range(cost_matrix.shape[0]) if r not in required_so_far]
        cumulative_required = dict(required_so_far)
        for r in free_rows:
            c = assignment[r]
            child = _MurtyNode(required=tuple(cumulative_required.items()), forbidden=node.forbidden | {(r, c)})
            child_result = _solve_node(cost_matrix, child)
            if child_result is not None:
                counter += 1
                heapq.heappush(heap, (child_result[0], counter, child, child_result[1]))
            cumulative_required[r] = c  # next sibling in this loop requires (r, c) to stay fixed

    return results


def k_best_season_plans(weeks: list[int], teams: list[str], survival_prob: dict[tuple[int, str], float], k: int) -> list[SeasonPlan]:
    """Labeled wrapper around k_best_assignments — the k best distinct
    complete season plans, best first."""
    if len(teams) < len(weeks):
        raise InfeasibleSeasonError(f"{len(weeks)} remaining weeks but only {len(teams)} teams available")
    cost_matrix = build_cost_matrix(weeks, teams, survival_prob)
    raw = k_best_assignments(cost_matrix, k)

    plans = []
    for total_cost, assignment in raw:
        picks = []
        for r, c in sorted(assignment.items()):
            log_p = -cost_matrix[r, c]
            picks.append(WeekPick(week=weeks[r], team=teams[c], survival_prob=float(np.exp(log_p))))
        plans.append(SeasonPlan(picks=tuple(picks), total_survival_prob=float(np.exp(-total_cost)), total_log_survival=float(-total_cost)))
    return plans
