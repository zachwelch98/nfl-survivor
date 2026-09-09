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


# --- Cross-language fixtures -------------------------------------------------
#
# The what-if planner re-solves the assignment in the browser so that locking a
# team into a week is instant instead of a Python round-trip. That means the
# exact optimizer now exists in two languages, and the README's claim that it
# is "exact, not heuristic" has to hold in both. This test is the bridge: it
# generates instances here, solves them with scipy, and writes the *optimal
# total cost* for the TypeScript suite to check itself against.
#
# Cost, not the assignment. Ties are common (and one case below is built
# entirely of them), so several distinct assignments can be equally optimal.
# Demanding cell-for-cell agreement with scipy would be a flaky test wearing a
# strict test's clothes; the TS side asserts validity *and* equal total cost,
# which is the property that actually matters.

FIXTURE_PATH = "frontend/src/lib/__fixtures__/assignment_cases.json"


def _fixture_cases() -> list[dict]:
    rng = np.random.default_rng(20260908)
    cases: list[dict] = []

    def add(name: str, n_weeks: int, n_teams: int, drop_frac: float, quantize: int | None = None):
        weeks = list(range(1, n_weeks + 1))
        teams = [f"T{j:02d}" for j in range(n_teams)]
        probs: dict[tuple[int, str], float] = {}
        for w in weeks:
            for t in teams:
                if rng.random() < drop_frac:
                    continue  # unavailable: bye, no game, no model output
                p = float(rng.uniform(0.35, 0.99))
                if quantize:
                    p = round(p, quantize)
                probs[(w, t)] = p
        cases.append({"name": name, "weeks": weeks, "teams": teams, "probs": probs})

    add("square_dense", 6, 6, 0.0)
    add("rectangular_like_a_real_season", 18, 32, 0.0)
    add("sparse_with_byes", 10, 14, 0.25)
    add("very_sparse", 8, 12, 0.5)
    add("single_week", 1, 32, 0.0)
    # Deliberate ties: two decimal places over 12x16 guarantees repeats, so the
    # optimum is reached by many different assignments.
    add("ties_everywhere", 12, 16, 0.0, quantize=2)

    # Exactly-square-and-forced: one feasible team per week, so there is a
    # unique answer and any bug shows immediately.
    forced_probs = {(w, f"T{w - 1:02d}"): 0.5 + 0.01 * w for w in range(1, 6)}
    cases.append({
        "name": "forced_unique",
        "weeks": list(range(1, 6)),
        "teams": [f"T{j:02d}" for j in range(5)],
        "probs": forced_probs,
    })
    return cases


def test_write_assignment_fixtures_for_the_typescript_solver(tmp_path):
    """Regenerates frontend/src/lib/__fixtures__/assignment_cases.json.

    Runs as part of the normal suite so the fixtures can never drift from the
    Python solver: if scipy's answer changes, this file changes, and the
    TypeScript suite starts failing until it is reconciled.
    """
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    out: list[dict] = []

    for case in _fixture_cases():
        weeks, teams, probs = case["weeks"], case["teams"], case["probs"]
        try:
            plan = solve_optimal_assignment(weeks, teams, probs)
            expected = {
                "feasible": True,
                "total_log_survival": plan.total_log_survival,
                "total_survival_prob": plan.total_survival_prob,
            }
        except InfeasibleSeasonError:
            expected = {"feasible": False, "total_log_survival": None, "total_survival_prob": None}

        out.append({
            "name": case["name"],
            "weeks": weeks,
            "teams": teams,
            # JSON has no tuple keys; the TS side rebuilds a Map from "week:team".
            "probs": {f"{w}:{t}": p for (w, t), p in sorted(probs.items())},
            "expected": expected,
        })

    # An all-infeasible row: available in principle, but zero probability, which
    # build_cost_matrix maps to +inf. No complete assignment exists.
    hopeless = {(1, "A"): 0.9, (1, "B"): 0.8, (2, "A"): 0.0, (2, "B"): 0.0}
    out.append({
        "name": "infeasible_zero_row",
        "weeks": [1, 2],
        "teams": ["A", "B"],
        "probs": {f"{w}:{t}": p for (w, t), p in sorted(hopeless.items())},
        "expected": {"feasible": False, "total_log_survival": None, "total_survival_prob": None},
    })
    # Fewer teams than weeks.
    out.append({
        "name": "infeasible_too_few_teams",
        "weeks": [1, 2, 3],
        "teams": ["A", "B"],
        "probs": {"1:A": 0.9, "1:B": 0.8, "2:A": 0.7, "2:B": 0.6, "3:A": 0.5, "3:B": 0.4},
        "expected": {"feasible": False, "total_log_survival": None, "total_survival_prob": None},
    })

    path = repo_root / FIXTURE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")

    # Sanity-check what we just wrote rather than only that we wrote it.
    written = json.loads(path.read_text())
    assert len(written) == 9
    assert {c["name"] for c in written} >= {"rectangular_like_a_real_season", "ties_everywhere", "forced_unique"}
    feasible = [c for c in written if c["expected"]["feasible"]]
    assert len(feasible) == 7
    assert all(0.0 < c["expected"]["total_survival_prob"] <= 1.0 for c in feasible)

    forced = next(c for c in written if c["name"] == "forced_unique")
    expected_log = sum(np.log(0.5 + 0.01 * w) for w in range(1, 6))
    assert forced["expected"]["total_log_survival"] == pytest.approx(expected_log, abs=1e-12)
