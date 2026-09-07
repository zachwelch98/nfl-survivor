"""Tests for export/json_bundle.py: structural correctness and JSON
serializability, validated against real live data."""

from __future__ import annotations

import json

import pytest

from survivor.data.nflverse import load_pbp, load_schedules
from survivor.export.json_bundle import build_season_meta, build_season_plan, build_week_board
from survivor.features.drive_rates import build_drive_table
from survivor.models.montecarlo import DriveSimConfig

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
