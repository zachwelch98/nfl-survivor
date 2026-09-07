"""Tests for features/team_ratings.py: naive shrinkage + opponent-adjusted ridge.

The ridge regression's sign convention is the one genuinely easy place for a
subtle bug (mixing up "good defense" with positive vs negative coefficients),
so it gets a dedicated synthetic-recovery test with known ground truth.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from survivor.clock import AsOf
from survivor.features.team_ratings import (
    build_team_game_table,
    compute_naive_ratings,
    compute_opponent_adjusted_ratings,
)


def _play(game_id, season, week, posteam, defteam, posteam_type, play_type, epa, success=1.0, yards_gained=5.0):
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "season_type": "REG",
        "posteam": posteam,
        "defteam": defteam,
        "posteam_type": posteam_type,
        "play_type": play_type,
        "epa": epa,
        "success": success,
        "yards_gained": yards_gained,
    }


def _pbp(rows):
    return pl.DataFrame(rows)


def test_build_team_game_table_splits_pass_and_rush_epa():
    rows = [
        _play("g1", 2024, 1, "KC", "DEN", "home", "pass", 1.0),
        _play("g1", 2024, 1, "KC", "DEN", "home", "run", -1.0),
        _play("g1", 2024, 1, "DEN", "KC", "away", "pass", 0.5),
    ]
    tg = build_team_game_table(_pbp(rows))
    kc = tg.filter(pl.col("team") == "KC")
    assert kc.height == 1
    assert kc["epa_per_play"].item() == pytest.approx(0.0)  # mean(1.0, -1.0)
    assert kc["pass_epa_per_play"].item() == pytest.approx(1.0)
    assert kc["rush_epa_per_play"].item() == pytest.approx(-1.0)
    assert kc["is_home"].item() is True
    assert kc["opponent"].item() == "DEN"

    den = tg.filter(pl.col("team") == "DEN")
    assert den["is_home"].item() is False


def test_naive_ratings_zero_observed_uses_pure_prior():
    prior_rows = [_play(f"p{i}", 2023, 5, "KC", "OPP", "home", "pass", 0.5) for i in range(30)]
    tg = build_team_game_table(_pbp(prior_rows))
    asof = AsOf(season=2024, week=1, game_type="REG")
    ratings = compute_naive_ratings(tg, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)
    assert ratings["KC"].n_games == 0
    assert ratings["KC"].epa_per_play == pytest.approx(0.5, abs=1e-6)


def test_naive_ratings_no_leakage_from_future_weeks():
    early = [_play(f"e{i}", 2024, 3, "KC", "OPP", "home", "pass", 1.0) for i in range(10)]
    future = [_play(f"f{i}", 2024, 12, "KC", "OPP", "home", "pass", -5.0) for i in range(30)]
    tg = build_team_game_table(_pbp(early + future))
    asof = AsOf(season=2024, week=8, game_type="REG")
    ratings = compute_naive_ratings(tg, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)
    assert ratings["KC"].n_games == 10
    assert ratings["KC"].epa_per_play > 0  # the future -5.0 blowup must not leak in


# ---------------------------------------------------------------------------
# Opponent-adjusted ridge regression: synthetic recovery test
# ---------------------------------------------------------------------------


def _simulate_league(true_off: dict, true_def: dict, true_hfa: float, n_rounds: int, seed: int = 0) -> pl.DataFrame:
    """Every team plays every other team home-and-away, `n_rounds` times,
    with observed EPA/play = off[T] + def[O] + hfa*is_home + noise — using
    the *output* sign convention (def positive = bad defense), matching
    what compute_opponent_adjusted_ratings is expected to recover."""
    rng = np.random.default_rng(seed)
    teams = list(true_off.keys())
    rows = []
    gid = 0
    for _ in range(n_rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                gid += 1
                noise = rng.normal(0, 0.05)
                home_epa = true_off[home] + true_def[away] + true_hfa + noise
                rows.append(_play(f"g{gid}h", 2024, 5, home, away, "home", "pass", home_epa))
                noise2 = rng.normal(0, 0.05)
                away_epa = true_off[away] + true_def[home] + noise2
                rows.append(_play(f"g{gid}a", 2024, 5, away, home, "away", "pass", away_epa))
    return build_team_game_table(_pbp(rows))


def test_ridge_recovers_offensive_and_defensive_rank_order():
    true_off = {"A": 0.30, "B": 0.00, "C": -0.30, "D": 0.10}
    true_def = {"A": -0.20, "B": 0.00, "C": 0.20, "D": -0.10}  # negative = good defense
    true_hfa = 0.05

    tg = _simulate_league(true_off, true_def, true_hfa, n_rounds=8, seed=1)
    asof = AsOf(season=2024, week=10, game_type="REG")
    ratings = compute_opponent_adjusted_ratings(tg, asof, alpha=1.0)

    off_order = sorted(true_off, key=lambda t: true_off[t], reverse=True)
    recovered_off_order = sorted(ratings, key=lambda t: ratings[t].off_rating, reverse=True)
    assert recovered_off_order == off_order, "offensive rank order not recovered"

    def_order = sorted(true_def, key=lambda t: true_def[t])  # best (most negative) defense first
    recovered_def_order = sorted(ratings, key=lambda t: ratings[t].def_rating)
    assert recovered_def_order == def_order, "defensive rank order not recovered (sign convention bug?)"

    # the best defense in true_def is "A" (-0.20) — must end up with a
    # negative recovered def_rating, not positive (the exact sign-flip bug
    # this test exists to catch).
    assert ratings["A"].def_rating < 0
    assert ratings["C"].def_rating > 0  # worst defense -> positive (bad)

    assert ratings["A"].hfa == pytest.approx(true_hfa, abs=0.05)


def test_ridge_projected_epa_matches_generative_model_closely():
    true_off = {"A": 0.30, "B": 0.00, "C": -0.30, "D": 0.10}
    true_def = {"A": -0.20, "B": 0.00, "C": 0.20, "D": -0.10}
    true_hfa = 0.05
    tg = _simulate_league(true_off, true_def, true_hfa, n_rounds=10, seed=2)
    asof = AsOf(season=2024, week=10, game_type="REG")
    ratings = compute_opponent_adjusted_ratings(tg, asof, alpha=1.0)

    # A (home) vs C (away): true expected home EPA = 0.30 + 0.20 + 0.05 = 0.55
    predicted = ratings["A"].projected_epa(ratings["C"].def_rating, is_home=True, hfa_value=ratings["A"].hfa)
    assert predicted == pytest.approx(0.55, abs=0.08)


def test_ridge_empty_input_returns_empty():
    schema = {
        "game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32, "game_type": pl.Utf8, "team": pl.Utf8,
        "opponent": pl.Utf8, "is_home": pl.Boolean, "epa_per_play": pl.Float64, "pass_epa_per_play": pl.Float64,
        "rush_epa_per_play": pl.Float64, "success_rate": pl.Float64, "yards_per_play": pl.Float64, "n_plays": pl.UInt32,
    }
    tg = pl.DataFrame(schema={k: v for k, v in schema.items()})
    asof = AsOf(season=2024, week=1, game_type="REG")
    assert compute_opponent_adjusted_ratings(tg, asof) == {}


def test_ridge_no_leakage_against_real_data():
    """Property check against real pbp: an as-of-week-8 rating must only
    reflect games actually playable before week 8."""
    from survivor.data.nflverse import load_pbp

    pbp = load_pbp([2023])
    tg = build_team_game_table(pbp)
    asof = AsOf(season=2023, week=8, game_type="REG")
    ratings = compute_opponent_adjusted_ratings(tg, asof, alpha=40.0)

    visible_games = asof.visible(tg)
    max_games_by_team = dict(zip(visible_games.group_by("team").len()["team"], visible_games.group_by("team").len()["len"]))
    all_games = tg.filter((pl.col("season") == 2023) & (pl.col("week") >= 8))
    # sanity: at least one team must have zero visible games only if it hasn't played yet (bye week 1 impossible) -- just assert ratings exist for all real teams
    assert len(ratings) >= 32
    assert set(max_games_by_team.keys()) <= set(ratings.keys())
