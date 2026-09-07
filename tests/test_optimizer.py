"""Correctness tests for the season-sequencing optimizer (plan's explicit
"Correctness tests" list): exact assignment verified against brute force,
greedy shown strictly worse on a constructed case, plus k-best and
cost-of-deviation coverage.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from survivor.optimize.assignment import (
    InfeasibleSeasonError,
    build_cost_matrix,
    greedy_assignment,
    solve_optimal_assignment,
)
from survivor.optimize.kbest import cost_of_deviation_by_team, k_best_assignments, k_best_season_plans


def _brute_force_best(weeks: list[int], teams: list[str], survival_prob: dict) -> tuple[float, tuple]:
    """Exhaustive search over every way to assign distinct teams to weeks —
    the ground truth the optimizer is checked against."""
    best_log = -np.inf
    best_assignment = None
    for perm in itertools.permutations(teams, len(weeks)):
        log_total = 0.0
        feasible = True
        for w, t in zip(weeks, perm):
            p = survival_prob.get((w, t))
            if p is None:
                feasible = False
                break
            log_total += np.log(p)
        if feasible and log_total > best_log:
            best_log = log_total
            best_assignment = perm
    return best_log, best_assignment


def _hand_case():
    """4 weeks, 5 teams — small enough to brute force (5*4*3*2 = 120 perms).
    Team E's week-1 number (0.95) is the single highest value anywhere in
    the matrix, tempting a greedy picker — but E is *also* the best option
    in week 4 (0.80), while team A (week 1's second-best option, 0.70) has
    no good alternative week (its next-best week is only 0.30). Grabbing E
    in week 1 therefore strands A with a bad week-4 number, and the
    verified-by-brute-force optimum is A/B/C/E across weeks 1-4, not E/B/C/D.
    """
    weeks = [1, 2, 3, 4]
    teams = ["A", "B", "C", "D", "E"]
    survival_prob = {
        (1, "A"): 0.70, (1, "B"): 0.50, (1, "C"): 0.50, (1, "D"): 0.50, (1, "E"): 0.95,
        (2, "A"): 0.50, (2, "B"): 0.85, (2, "C"): 0.50, (2, "D"): 0.50, (2, "E"): 0.50,
        (3, "A"): 0.50, (3, "B"): 0.50, (3, "C"): 0.85, (3, "D"): 0.50, (3, "E"): 0.50,
        (4, "A"): 0.30, (4, "B"): 0.50, (4, "C"): 0.50, (4, "D"): 0.50, (4, "E"): 0.80,
    }
    return weeks, teams, survival_prob


# ---------------------------------------------------------------------------
# Exact assignment vs brute force
# ---------------------------------------------------------------------------


def test_optimal_assignment_matches_brute_force():
    weeks, teams, survival_prob = _hand_case()
    plan = solve_optimal_assignment(weeks, teams, survival_prob)
    brute_log, brute_assignment = _brute_force_best(weeks, teams, survival_prob)

    assert plan.total_log_survival == pytest.approx(brute_log, abs=1e-9)
    assert tuple(p.team for p in plan.picks) == brute_assignment


def test_optimal_assignment_beats_taking_the_single_best_edge():
    """Team E's week-1 edge (0.95) is the highest single number in the whole
    matrix, but committing to it forces E to be unused in weeks 2-4 where
    it's terrible, and forces some other team to cover week 1 worse than
    optimal — the optimizer must not be fooled by the single biggest number."""
    weeks, teams, survival_prob = _hand_case()
    plan = solve_optimal_assignment(weeks, teams, survival_prob)
    assert plan.team_for_week(1) != "E"


def test_greedy_is_strictly_worse_than_optimal_on_constructed_case():
    weeks, teams, survival_prob = _hand_case()
    optimal = solve_optimal_assignment(weeks, teams, survival_prob)
    greedy = greedy_assignment(weeks, teams, survival_prob)
    assert greedy.total_log_survival < optimal.total_log_survival
    # greedy takes E in week 1 (locally best that week), which is exactly the trap
    assert greedy.team_for_week(1) == "E"


# ---------------------------------------------------------------------------
# build_cost_matrix / basic solve behavior
# ---------------------------------------------------------------------------


def test_build_cost_matrix_missing_pair_is_infinite():
    cost = build_cost_matrix([1, 2], ["A", "B"], {(1, "A"): 0.8, (2, "B"): 0.6})
    assert cost[0, 0] == pytest.approx(-np.log(0.8))
    assert np.isinf(cost[0, 1])
    assert np.isinf(cost[1, 0])
    assert cost[1, 1] == pytest.approx(-np.log(0.6))


def test_build_cost_matrix_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        build_cost_matrix([1], ["A"], {(1, "A"): 1.5})
    with pytest.raises(ValueError):
        build_cost_matrix([1], ["A"], {(1, "A"): -0.1})


def test_build_cost_matrix_accepts_zero_as_a_legitimate_guaranteed_loss():
    """A team the model gives literally no chance is a valid (if rare)
    input — it must map to +inf cost, not raise. This is a real edge case
    a Monte Carlo simulation with finitely many sims can actually produce."""
    cost = build_cost_matrix([1], ["A", "B"], {(1, "A"): 0.0, (1, "B"): 0.5})
    assert np.isinf(cost[0, 0])
    assert cost[0, 1] == pytest.approx(-np.log(0.5))


def test_solve_optimal_assignment_avoids_a_zero_probability_team_when_an_alternative_exists():
    plan = solve_optimal_assignment([1], ["A", "B"], {(1, "A"): 0.0, (1, "B"): 0.5})
    assert plan.team_for_week(1) == "B"


def test_solve_optimal_assignment_empty_weeks():
    plan = solve_optimal_assignment([], ["A", "B"], {})
    assert plan.picks == ()
    assert plan.total_survival_prob == 1.0


def test_solve_optimal_assignment_infeasible_more_weeks_than_teams():
    with pytest.raises(InfeasibleSeasonError):
        solve_optimal_assignment([1, 2, 3], ["A", "B"], {(1, "A"): 0.9, (2, "B"): 0.8, (3, "A"): 0.7})


def test_solve_optimal_assignment_infeasible_no_valid_team_for_a_week():
    with pytest.raises(InfeasibleSeasonError):
        solve_optimal_assignment([1, 2], ["A", "B"], {(1, "A"): 0.9})  # week 2 has no options at all


def test_solve_optimal_assignment_rectangular_uses_best_subset_of_teams():
    """10 teams available, only 3 weeks remaining -> exactly 3 teams used,
    the 3 that maximize total log-survival, not necessarily the 3
    individually-highest single numbers."""
    weeks = [1, 2, 3]
    teams = [f"T{i}" for i in range(10)]
    survival_prob = {(w, t): 0.5 + 0.01 * i for w in weeks for i, t in enumerate(teams)}
    plan = solve_optimal_assignment(weeks, teams, survival_prob)
    assert len(plan.picks) == 3
    assert len({p.team for p in plan.picks}) == 3


# ---------------------------------------------------------------------------
# k_best_assignments (Murty's algorithm)
# ---------------------------------------------------------------------------


def test_k_best_first_result_matches_optimal_solve():
    weeks, teams, survival_prob = _hand_case()
    optimal = solve_optimal_assignment(weeks, teams, survival_prob)
    plans = k_best_season_plans(weeks, teams, survival_prob, k=3)
    assert plans[0].total_log_survival == pytest.approx(optimal.total_log_survival, abs=1e-9)
    assert {p.team for p in plans[0].picks} == {p.team for p in optimal.picks}


def test_k_best_matches_brute_force_top_k():
    """Enumerate every assignment by brute force, rank by total log-survival,
    and check Murty's k-best matches exactly (values and team sets) for the
    top 5 — the real correctness bar for a k-best implementation."""
    weeks, teams, survival_prob = _hand_case()
    all_results = []
    for perm in itertools.permutations(teams, len(weeks)):
        log_total = 0.0
        feasible = True
        for w, t in zip(weeks, perm):
            p = survival_prob.get((w, t))
            if p is None:
                feasible = False
                break
            log_total += np.log(p)
        if feasible:
            all_results.append((-log_total, frozenset(zip(weeks, perm))))  # cost = -log
    all_results.sort(key=lambda x: x[0])

    plans = k_best_season_plans(weeks, teams, survival_prob, k=5)
    for i, plan in enumerate(plans):
        expected_cost = all_results[i][0]
        assert -plan.total_log_survival == pytest.approx(expected_cost, abs=1e-9), f"mismatch at rank {i}"
        got_pairs = frozenset((p.week, p.team) for p in plan.picks)
        assert got_pairs == all_results[i][1]


def test_k_best_returns_distinct_assignments():
    weeks, teams, survival_prob = _hand_case()
    plans = k_best_season_plans(weeks, teams, survival_prob, k=6)
    seen = set()
    for plan in plans:
        key = frozenset((p.week, p.team) for p in plan.picks)
        assert key not in seen
        seen.add(key)


def test_k_best_costs_are_non_decreasing():
    weeks, teams, survival_prob = _hand_case()
    plans = k_best_season_plans(weeks, teams, survival_prob, k=6)
    costs = [-p.total_log_survival for p in plans]
    assert costs == sorted(costs)


def test_k_best_handles_more_requested_than_exist():
    """Only 5 teams, 4 weeks -> at most P(5,4)=120 distinct assignments;
    asking for way more than exist should just return what's there, not crash."""
    weeks, teams, survival_prob = _hand_case()
    plans = k_best_season_plans(weeks, teams, survival_prob, k=1000)
    assert 0 < len(plans) <= 120


# ---------------------------------------------------------------------------
# cost_of_deviation_by_team
# ---------------------------------------------------------------------------


def test_cost_of_deviation_zero_for_the_optimal_pick_itself():
    weeks, teams, survival_prob = _hand_case()
    optimal = solve_optimal_assignment(weeks, teams, survival_prob)
    deviations = cost_of_deviation_by_team(weeks, teams, survival_prob, current_week=1)
    optimal_team_week1 = optimal.team_for_week(1)
    assert deviations[optimal_team_week1] == pytest.approx(0.0, abs=1e-9)


def test_cost_of_deviation_positive_for_the_tempting_trap_team():
    """Team E looks best in week 1 (0.95) but is a trap — deviating to it
    must show a strictly positive cost, quantifying exactly why it's wrong."""
    weeks, teams, survival_prob = _hand_case()
    deviations = cost_of_deviation_by_team(weeks, teams, survival_prob, current_week=1)
    assert deviations["E"] > 0.05  # meaningfully worse, not just floating point noise


def test_cost_of_deviation_matches_brute_force():
    """For each candidate team at week 1, the true best total (with that
    team fixed) minus the true global best, via brute force, must match."""
    weeks, teams, survival_prob = _hand_case()
    global_best_log, _ = _brute_force_best(weeks, teams, survival_prob)

    deviations = cost_of_deviation_by_team(weeks, teams, survival_prob, current_week=1)

    for fixed_team in teams:
        best_with_fixed = -np.inf
        for perm in itertools.permutations([t for t in teams if t != fixed_team], len(weeks) - 1):
            full_perm = (fixed_team,) + perm
            log_total = np.log(survival_prob[(1, fixed_team)])
            for w, t in zip(weeks[1:], perm):
                log_total += np.log(survival_prob[(w, t)])
            best_with_fixed = max(best_with_fixed, log_total)
        expected_deviation = global_best_log - best_with_fixed
        assert deviations[fixed_team] == pytest.approx(expected_deviation, abs=1e-9), f"team={fixed_team}"


def test_cost_of_deviation_only_includes_playable_teams():
    weeks, teams, survival_prob = _hand_case()
    del_prob = dict(survival_prob)
    del del_prob[(1, "D")]  # D not playable in week 1 (e.g. on bye)
    deviations = cost_of_deviation_by_team(weeks, teams, del_prob, current_week=1)
    assert "D" not in deviations
