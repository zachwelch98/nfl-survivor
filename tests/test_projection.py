"""Tests for optimize/projection.py: horizon decay and survival-prob assembly."""

from __future__ import annotations

import polars as pl
import pytest

from survivor.models.montecarlo import SimResult
from survivor.optimize.projection import build_survival_prob_table, shrink_future_prob_to_half


def test_shrink_zero_weeks_ahead_is_unchanged():
    assert shrink_future_prob_to_half(0.9, weeks_ahead=0) == 0.9
    assert shrink_future_prob_to_half(0.2, weeks_ahead=0) == 0.2


def test_shrink_moves_toward_half_as_horizon_grows():
    p = 0.9
    shrunk_near = shrink_future_prob_to_half(p, weeks_ahead=2, decay_rate=0.1)
    shrunk_far = shrink_future_prob_to_half(p, weeks_ahead=15, decay_rate=0.1)
    assert 0.5 < shrunk_far < shrunk_near < p


def test_shrink_symmetric_around_half():
    high = shrink_future_prob_to_half(0.8, weeks_ahead=5, decay_rate=0.1)
    low = shrink_future_prob_to_half(0.2, weeks_ahead=5, decay_rate=0.1)
    assert high - 0.5 == pytest.approx(0.5 - low)


def test_shrink_never_crosses_half():
    for wa in [1, 5, 10, 20, 50]:
        assert shrink_future_prob_to_half(0.95, weeks_ahead=wa, decay_rate=0.05) > 0.5
        assert shrink_future_prob_to_half(0.05, weeks_ahead=wa, decay_rate=0.05) < 0.5


def _sim_result(home_win, away_win, tie):
    return SimResult(
        home_win_prob=home_win, away_win_prob=away_win, tie_prob=tie,
        mean_home_score=24, mean_away_score=20, simulated_spread=4, simulated_total=44,
        home_scores=None, away_scores=None,
    )


def test_build_survival_prob_table_current_week_no_decay():
    current = pl.DataFrame(
        {"week": [5, 5], "home_team": ["KC", "SF"], "away_team": ["DEN", "SEA"], "home_win_prob": [0.7, 0.4], "away_win_prob": [0.28, 0.58], "tie_prob": [0.02, 0.02]}
    )
    future = pl.DataFrame(schema={"game_id": pl.Utf8, "week": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8})
    table = build_survival_prob_table(current, future, {}, tie_counts_as_loss=True)
    assert table[(5, "KC")] == pytest.approx(0.7)
    assert table[(5, "DEN")] == pytest.approx(0.28)


def test_build_survival_prob_table_tie_rule_changes_survival():
    current = pl.DataFrame({"week": [5], "home_team": ["KC"], "away_team": ["DEN"], "home_win_prob": [0.7], "away_win_prob": [0.28], "tie_prob": [0.02]})
    future = pl.DataFrame(schema={"game_id": pl.Utf8, "week": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8})
    table_loss = build_survival_prob_table(current, future, {}, tie_counts_as_loss=True)
    table_not_loss = build_survival_prob_table(current, future, {}, tie_counts_as_loss=False)
    assert table_loss[(5, "KC")] == pytest.approx(0.7)
    assert table_not_loss[(5, "KC")] == pytest.approx(0.72)  # 0.7 + 0.02 tie credited as a non-loss


def test_build_survival_prob_table_future_weeks_use_mc_with_decay():
    current = pl.DataFrame({"week": [5], "home_team": ["KC"], "away_team": ["DEN"], "home_win_prob": [0.7], "away_win_prob": [0.28], "tie_prob": [0.02]})
    future = pl.DataFrame({"game_id": ["g1"], "week": [8], "home_team": ["KC"], "away_team": ["LAC"]})
    mc_results = {"g1": _sim_result(0.9, 0.08, 0.02)}
    table = build_survival_prob_table(current, future, mc_results, tie_counts_as_loss=True, decay_rate=0.1)
    # week 8 is 3 weeks ahead of current week 5 -> decayed toward 0.5, must be less than raw 0.9
    assert 0.5 < table[(8, "KC")] < 0.9


def test_build_survival_prob_table_skips_games_without_mc_result():
    current = pl.DataFrame({"week": [5], "home_team": ["KC"], "away_team": ["DEN"], "home_win_prob": [0.7], "away_win_prob": [0.28], "tie_prob": [0.02]})
    future = pl.DataFrame({"game_id": ["missing"], "week": [8], "home_team": ["KC"], "away_team": ["LAC"]})
    table = build_survival_prob_table(current, future, {}, tie_counts_as_loss=True)
    assert (8, "KC") not in table
    assert (8, "LAC") not in table


def test_simulate_all_remaining_games_real_data_smoke():
    from survivor.clock import AsOf
    from survivor.data.nflverse import load_pbp, load_schedules
    from survivor.features.drive_rates import build_drive_table
    from survivor.models.montecarlo import DriveSimConfig
    from survivor.optimize.projection import simulate_all_remaining_games

    pbp = load_pbp([2023, 2024])
    schedules = load_schedules()
    drive_table = build_drive_table(pbp)
    as_of = AsOf(season=2024, week=10, game_type="REG")

    future = schedules.filter((pl.col("season") == 2024) & (pl.col("week") == 12) & (pl.col("game_type") == "REG")).select(["game_id", "home_team", "away_team"])
    results = simulate_all_remaining_games(drive_table, schedules, as_of, future, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4, mc_config=DriveSimConfig(n_simulations=2000, seed=0))
    assert len(results) > 0
    for sim in results.values():
        assert 0 <= sim.home_win_prob <= 1
        assert sim.home_win_prob + sim.away_win_prob + sim.tie_prob == pytest.approx(1.0, abs=1e-9)
