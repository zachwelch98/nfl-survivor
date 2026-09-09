"""Tests for export/json_bundle.py: structural correctness and JSON
serializability, validated against real live data."""

from __future__ import annotations

import json

import polars as pl
import pytest

from survivor.config import CONFIG
from survivor.data.nflverse import load_pbp, load_schedules
from survivor.export.json_bundle import (
    build_season_meta,
    build_season_plan,
    build_survival_matrix,
    build_week_board,
)
from survivor.features.drive_rates import build_drive_table
from survivor.models.montecarlo import DriveSimConfig
from survivor.optimize.assignment import solve_optimal_assignment
from survivor.optimize.projection import DEFAULT_DECAY_RATE

FAST_MC = DriveSimConfig(n_simulations=1000, seed=0)


@pytest.fixture(scope="module")
def real_data():
    pbp = load_pbp([2024, 2025])
    schedules = load_schedules()
    return pbp, schedules, build_drive_table(pbp)


def test_season_meta_is_json_serializable():
    meta = build_season_meta(2026, 1, used_teams=["KC"])
    serialized = json.dumps(meta)
    assert json.loads(serialized)["used_teams"] == ["KC"]


def test_week_board_2026_week1_structure(real_data):
    pbp, schedules, drive_table = real_data
    board = build_week_board(schedules, drive_table, 2026, 1, used_teams=[], mc_config=FAST_MC)
    assert board["season"] == 2026 and board["week"] == 1
    assert len(board["games"]) > 0
    for g in board["games"]:
        assert 0 <= g["vegas_home_win_prob"] <= 1
        assert 0 <= g["vegas_away_win_prob"] <= 1
        if g["mc_home_win_prob"] is not None:
            assert 0 <= g["mc_home_win_prob"] <= 1
    assert board["recommended_pick"] is not None
    json.dumps(board)  # must be fully JSON-serializable


def test_week_board_excludes_used_teams_from_recommendation(real_data):
    pbp, schedules, drive_table = real_data
    board = build_week_board(schedules, drive_table, 2026, 1, used_teams=[], mc_config=FAST_MC)
    top_pick = board["recommended_pick"]
    board_excluded = build_week_board(schedules, drive_table, 2026, 1, used_teams=[top_pick], mc_config=FAST_MC)
    assert board_excluded["recommended_pick"] != top_pick
    assert top_pick not in board_excluded["cost_of_deviation"]


def test_season_plan_2026_full_structure(real_data):
    pbp, schedules, drive_table = real_data
    plan = build_season_plan(schedules, drive_table, 2026, 1, used_teams=[], mc_config=FAST_MC, k_alternates=2)
    assert plan["season"] == 2026
    assert len(plan["recommended_sequence"]) == 18  # full remaining season, week 1 to 18
    teams_used = [p["team"] for p in plan["recommended_sequence"]]
    assert len(teams_used) == len(set(teams_used))  # never reuses a team
    assert plan["total_survival_prob"] is not None
    assert 0 < plan["total_survival_prob"] <= 1
    assert len(plan["alternates"]) <= 2
    json.dumps(plan)


def test_season_plan_respects_used_teams(real_data):
    pbp, schedules, drive_table = real_data
    plan = build_season_plan(schedules, drive_table, 2026, 5, used_teams=["KC", "BUF"], mc_config=FAST_MC, k_alternates=1)
    teams_used = [p["team"] for p in plan["recommended_sequence"]]
    assert "KC" not in teams_used
    assert "BUF" not in teams_used
    assert len(plan["recommended_sequence"]) == 14  # weeks 5-18


def test_week_board_carries_ml_fields(real_data):
    """Method 3 appears on the board for comparison. It is optional by
    design -- without a team_game_table the fields are present but null,
    which is exactly what a bundle exported from a narrow pbp load looks
    like, and the frontend renders that as an em dash rather than breaking.
    """
    from survivor.features.team_ratings import build_team_game_table

    pbp, schedules, drive_table = real_data

    without = build_week_board(schedules, drive_table, 2026, 1, [])
    assert all("ml_home_win_prob" in g for g in without["games"])
    assert all(g["ml_home_win_prob"] is None for g in without["games"])

    with_ml = build_week_board(
        schedules, drive_table, 2026, 1, [], team_game_table=build_team_game_table(pbp)
    )
    scored = [g for g in with_ml["games"] if g["ml_home_win_prob"] is not None]
    assert len(scored) == len(with_ml["games"])
    for g in scored:
        total = g["ml_home_win_prob"] + g["ml_away_win_prob"] + g["ml_tie_prob"]
        assert total == pytest.approx(1.0, abs=1e-6)

    # The recommendation still comes from Vegas -- Method 3 is shown, not obeyed.
    assert with_ml["recommended_pick"] == without["recommended_pick"]


def test_survival_matrix_structure(real_data):
    """survival_matrix.json is the what-if planner's only input: the whole
    (week, team) -> survival_prob table, unfiltered, annotated with the
    matchup each probability refers to."""
    pbp, schedules, drive_table = real_data
    table, payload = build_survival_matrix(schedules, drive_table, 2026, 1, mc_config=FAST_MC)

    assert payload["season"] == 2026 and payload["current_week"] == 1
    assert payload["weeks"] == list(range(1, 19))
    assert len(payload["teams"]) == 32
    assert payload["tie_counts_as_loss"] == CONFIG.pool.tie_counts_as_loss
    assert payload["decay_rate"] == DEFAULT_DECAY_RATE

    # Every table key round-trips into an entry, and nothing is invented.
    assert {(e["week"], e["team"]) for e in payload["entries"]} == set(table.keys())

    for e in payload["entries"]:
        assert 0.0 <= e["survival_prob"] <= 1.0
        assert e["weeks_ahead"] == e["week"] - 1
        assert e["source"] == ("vegas" if e["week"] == 1 else "model")
        assert e["opponent"] != e["team"]
    json.dumps(payload)


def test_survival_matrix_entries_match_the_real_schedule(real_data):
    """Each entry claims a specific game and a specific side of it. If those
    were wrong the frontend would show a pick against the wrong opponent, so
    check them against the schedule rather than trusting the join."""
    pbp, schedules, drive_table = real_data
    _, payload = build_survival_matrix(schedules, drive_table, 2026, 1, mc_config=FAST_MC)

    games = {
        row["game_id"]: (row["home_team"], row["away_team"])
        for row in schedules.filter(
            (pl.col("season") == 2026) & (pl.col("game_type") == "REG")
        ).iter_rows(named=True)
    }
    for e in payload["entries"]:
        home, away = games[e["game_id"]]
        if e["is_home"]:
            assert (e["team"], e["opponent"]) == (home, away)
        else:
            assert (e["team"], e["opponent"]) == (away, home)


def test_survival_matrix_is_unfiltered_and_reproduces_the_season_plan(real_data):
    """The contract the TypeScript solver must also satisfy: filtering the
    exported matrix by used teams and solving it reproduces season_plan.json
    exactly. This is what lets the browser re-solve locally instead of
    round-tripping to Python for every what-if."""
    pbp, schedules, drive_table = real_data
    used = ["KC", "BUF"]

    table, payload = build_survival_matrix(schedules, drive_table, 2026, 1, mc_config=FAST_MC)
    # Unfiltered: used teams are still present, because availability is the
    # consumer's decision, not the matrix's.
    assert any(t in used for (_, t) in table.keys())

    plan = build_season_plan(
        schedules, drive_table, 2026, 1, used_teams=used,
        mc_config=FAST_MC, k_alternates=1, survival_prob=table,
    )

    available = [t for t in payload["teams"] if t not in used]
    restricted = {(w, t): p for (w, t), p in table.items() if t in available}
    solved = solve_optimal_assignment(payload["weeks"], available, restricted)

    assert [(p.week, p.team) for p in solved.picks] == [
        (p["week"], p["team"]) for p in plan["recommended_sequence"]
    ]
    assert solved.total_survival_prob == pytest.approx(plan["total_survival_prob"], rel=1e-12)
