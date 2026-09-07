"""Exact season-sequencing optimizer (plan §4).

Key insight: maximizing survival probability across the whole season,
`Π_w p(pick_w)`, is equivalent to maximizing `Σ_w log p(pick_w)` — which is
exactly a bipartite assignment problem (weeks x teams, one team per week,
each team at most once). That has an exact polynomial-time solution via
`scipy.optimize.linear_sum_assignment` — no DP over 2^32 subsets, no beam
search heuristic, no greedy per-week maximization (which is provably
suboptimal: it spends strong teams early and can strand the picker with
only coin-flips left in December).

This module is deliberately agnostic about *why* a (week, team) pair is
unavailable (team already used, on bye, no game scheduled that week, or a
week already played and no longer a decision) — the caller expresses all
of that by simply omitting infeasible pairs from `survival_prob`. Anything
not present in that dict is treated as cost = +inf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


class InfeasibleSeasonError(Exception):
    """Raised when no complete assignment exists — e.g. more remaining
    weeks than teams still available (should be effectively impossible in
    a real 32-team survivor pool, but guarded against rather than left to
    surface as an opaque scipy ValueError)."""


@dataclass(frozen=True)
class WeekPick:
    week: int
    team: str
    survival_prob: float  # P(this pick survives), already reflecting the pool's tie rule


@dataclass(frozen=True)
class SeasonPlan:
    picks: tuple[WeekPick, ...]  # sorted by week
    total_survival_prob: float  # product of every pick's survival_prob — the actual objective in probability terms
    total_log_survival: float  # sum of log(survival_prob) — what the optimizer actually maximizes

    def team_for_week(self, week: int) -> str | None:
        for p in self.picks:
            if p.week == week:
                return p.team
        return None


def build_cost_matrix(weeks: list[int], teams: list[str], survival_prob: dict[tuple[int, str], float]) -> np.ndarray:
    """cost[i,j] = -log(survival_prob[weeks[i], teams[j]]), or +inf if that
    pair is absent from `survival_prob` (infeasible: team unavailable that
    week for any reason) or the probability is <= 0."""
    cost = np.full((len(weeks), len(teams)), np.inf)
    for i, w in enumerate(weeks):
        for j, t in enumerate(teams):
            p = survival_prob.get((w, t))
            if p is None:
                continue
            if not (0.0 < p <= 1.0):
                raise ValueError(f"survival_prob for (week={w}, team={t}) must be in (0, 1], got {p}")
            cost[i, j] = -np.log(p)
    return cost


def solve_optimal_assignment(weeks: list[int], teams: list[str], survival_prob: dict[tuple[int, str], float]) -> SeasonPlan:
    """The exact, provably-optimal season pick sequence given fixed win
    probabilities. Every remaining week gets a distinct team maximizing the
    product of survival probabilities across the whole set."""
    if not weeks:
        return SeasonPlan(picks=(), total_survival_prob=1.0, total_log_survival=0.0)
    if len(teams) < len(weeks):
        raise InfeasibleSeasonError(f"{len(weeks)} remaining weeks but only {len(teams)} teams available — cannot assign a distinct team to each week")

    cost = build_cost_matrix(weeks, teams, survival_prob)
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ValueError as e:
        raise InfeasibleSeasonError("no complete assignment exists for the given weeks/teams/survival_prob") from e

    picks = []
    total_log = 0.0
    for r, c in zip(row_ind, col_ind):
        c_val = cost[r, c]
        if not np.isfinite(c_val):
            raise InfeasibleSeasonError(f"no feasible team for week {weeks[r]}")
        log_p = -c_val
        total_log += log_p
        picks.append(WeekPick(week=weeks[r], team=teams[c], survival_prob=float(np.exp(log_p))))

    picks.sort(key=lambda wp: wp.week)
    return SeasonPlan(picks=tuple(picks), total_survival_prob=float(np.exp(total_log)), total_log_survival=float(total_log))


def greedy_assignment(weeks: list[int], teams: list[str], survival_prob: dict[tuple[int, str], float]) -> SeasonPlan:
    """The naive alternative this project argues against: each week, pick
    whichever available team has the highest survival probability *for
    that week alone*. Included for comparison/testing, not for actual use —
    see plan's rationale for why this is provably suboptimal across a
    season (it spends strong teams early with no regard for future weeks).
    """
    used: set[str] = set()
    picks = []
    total_log = 0.0
    for w in sorted(weeks):
        candidates = [(t, survival_prob[(w, t)]) for t in teams if t not in used and (w, t) in survival_prob]
        if not candidates:
            raise InfeasibleSeasonError(f"no feasible team for week {w}")
        team, p = max(candidates, key=lambda tp: tp[1])
        used.add(team)
        total_log += np.log(p)
        picks.append(WeekPick(week=w, team=team, survival_prob=p))
    return SeasonPlan(picks=tuple(picks), total_survival_prob=float(np.exp(total_log)), total_log_survival=float(total_log))
