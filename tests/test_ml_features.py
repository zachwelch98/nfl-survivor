"""Tests for features/ml_features.py — Method 3's feature assembly.

The load-bearing test in this file is `test_no_leakage_from_future_results`:
everything else checks that individual features are computed correctly, but
that one checks the property the whole model's validity rests on. It follows
the same full-vs-truncated construction as
test_model_comparison.py::test_walkforward_blend_no_leakage_from_future_weeks
— if any feature peeked at a future result, deleting the future would change
the answer.
"""

from __future__ import annotations

import polars as pl
import pytest

from survivor.clock import AsOf
from survivor.features.drive_rates import build_drive_table
from survivor.features.ml_features import (
    QBContinuityFeatures,
    build_feature_table,
    build_starter_history,
    compute_rolling_form,
    feature_columns,
    qb_continuity_features,
    resolve_starter,
)
from survivor.features.team_ratings import build_team_game_table


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------


def _team_game(season, week, team, opponent, is_home, epa, success=0.45):
    return {
        "game_id": f"{season}_{week:02d}_{team}_{opponent}",
        "season": season,
        "week": week,
        "game_type": "REG",
        "team": team,
        "opponent": opponent,
        "is_home": is_home,
        "epa_per_play": epa,
        "pass_epa_per_play": epa,
        "rush_epa_per_play": epa,
        "success_rate": success,
        "yards_per_play": 5.0,
        "n_plays": 60,
    }


def _schedule_row(season, week, home, away, home_score=None, away_score=None, home_qb=None, away_qb=None):
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "season": season,
        "week": week,
        "game_type": "REG",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_qb_id": home_qb,
        "away_qb_id": away_qb,
        "home_rest": 7,
        "away_rest": 7,
        "roof": "outdoors",
        "temp": 60,
        "wind": 5,
        "stadium_id": None,
        "gameday": f"{season}-09-{10 + week:02d}",
        "weekday": "Sunday",
        "div_game": 0,
    }


# ---------------------------------------------------------------------------
# Rolling form
# ---------------------------------------------------------------------------


def test_rolling_form_falls_back_to_season_rating_below_floor():
    """Below the floor the rolling stat must equal the season-level rating
    exactly, which is what makes trajectory structurally zero rather than an
    imputed guess."""
    games = pl.DataFrame([_team_game(2024, 1, "KC", "DEN", True, 0.30)])
    form = compute_rolling_form(
        games, "KC", window=3, min_games_floor=3, season_level_epa=0.111, season_level_success_rate=0.456
    )
    assert form.is_small_sample_fallback is True
    assert form.n_games_used == 1
    assert form.epa_per_play == pytest.approx(0.111)
    assert form.success_rate == pytest.approx(0.456)


def test_rolling_form_uses_plain_mean_of_last_window_games():
    games = pl.DataFrame(
        [
            _team_game(2024, 1, "KC", "DEN", True, 0.10),
            _team_game(2024, 2, "KC", "LV", False, 0.20),
            _team_game(2024, 3, "KC", "LAC", True, 0.30),
            _team_game(2024, 4, "KC", "NYJ", True, 0.90),  # most recent
        ]
    )
    form = compute_rolling_form(games, "KC", window=3, min_games_floor=3, season_level_epa=0.0, season_level_success_rate=0.0)
    assert form.is_small_sample_fallback is False
    assert form.n_games_used == 3
    # last 3 only: 0.20, 0.30, 0.90 -- the 0.10 game is outside the window
    assert form.epa_per_play == pytest.approx((0.20 + 0.30 + 0.90) / 3)


def test_rolling_form_uses_all_available_when_between_floor_and_window():
    """The floor is on games *available*, not on filling the window."""
    games = pl.DataFrame(
        [
            _team_game(2024, 1, "KC", "DEN", True, 0.10),
            _team_game(2024, 2, "KC", "LV", False, 0.20),
            _team_game(2024, 3, "KC", "LAC", True, 0.30),
        ]
    )
    form = compute_rolling_form(games, "KC", window=5, min_games_floor=3, season_level_epa=0.0, season_level_success_rate=0.0)
    assert form.is_small_sample_fallback is False
    assert form.n_games_used == 3
    assert form.epa_per_play == pytest.approx(0.20)


def test_rolling_form_crosses_season_boundary():
    games = pl.DataFrame(
        [
            _team_game(2023, 17, "KC", "DEN", True, 1.0),
            _team_game(2023, 18, "KC", "LV", False, 1.0),
            _team_game(2024, 1, "KC", "LAC", True, 1.0),
        ]
    )
    form = compute_rolling_form(games, "KC", window=3, min_games_floor=3, season_level_epa=0.0, season_level_success_rate=0.0)
    assert form.n_games_used == 3
    assert form.epa_per_play == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# QB resolution: the training/inference asymmetry
# ---------------------------------------------------------------------------


def _depth_charts(rows):
    return pl.DataFrame(rows, schema={"season": pl.Int64, "week": pl.Int64, "club_code": pl.Utf8, "position": pl.Utf8, "depth_team": pl.Utf8, "gsis_id": pl.Utf8})


def test_resolve_starter_prefers_the_games_own_row_over_depth_chart():
    """A played game's own qb_id is a pre-game fact about that game; it wins
    over the depth chart even when they disagree."""
    charts = _depth_charts([{"season": 2024, "week": 5, "club_code": "KC", "position": "QB", "depth_team": "1", "gsis_id": "DEPTH-QB"}])
    starter, projected = resolve_starter("ROW-QB", charts, "KC", AsOf(2024, 5))
    assert starter == "ROW-QB"
    assert projected is False


def test_resolve_starter_falls_back_to_depth_chart_when_row_is_null():
    """The genuinely-future game: no qb_id on the row, so use the most
    recent depth chart at or before the as-of point."""
    charts = _depth_charts(
        [
            {"season": 2024, "week": 3, "club_code": "KC", "position": "QB", "depth_team": "1", "gsis_id": "OLD-QB"},
            {"season": 2024, "week": 5, "club_code": "KC", "position": "QB", "depth_team": "1", "gsis_id": "NEW-QB"},
            {"season": 2024, "week": 9, "club_code": "KC", "position": "QB", "depth_team": "1", "gsis_id": "FUTURE-QB"},
        ]
    )
    starter, projected = resolve_starter(None, charts, "KC", AsOf(2024, 5))
    assert starter == "NEW-QB"  # week 9's chart is after the as-of point
    assert projected is True


def test_resolve_starter_degrades_instead_of_raising():
    assert resolve_starter(None, None, "KC", AsOf(2024, 5)) == (None, True)
    junk = pl.DataFrame({"unexpected": ["schema"]})
    assert resolve_starter(None, junk, "KC", AsOf(2024, 5)) == (None, True)


def test_qb_continuity_counts_run_and_detects_change():
    schedules = pl.DataFrame(
        [
            _schedule_row(2024, 1, "KC", "DEN", 20, 10, home_qb="QB-A"),
            _schedule_row(2024, 2, "KC", "LV", 20, 10, home_qb="QB-A"),
            _schedule_row(2024, 3, "KC", "LAC", 20, 10, home_qb="QB-A"),
        ]
    )
    history = build_starter_history(schedules)
    as_of = AsOf(2024, 4)
    visible = as_of.visible(history)

    same = qb_continuity_features(visible, as_of, "KC", "QB-A", starter_is_projected=False)
    assert same.starter_changed is False
    assert same.games_started_this_season == 3

    changed = qb_continuity_features(visible, as_of, "KC", "QB-B", starter_is_projected=False)
    assert changed.starter_changed is True
    assert changed.games_started_this_season == 0


def test_qb_continuity_run_stops_at_the_season_boundary():
    """A run that straddles last January is not a run 'this season'."""
    schedules = pl.DataFrame(
        [
            _schedule_row(2023, 17, "KC", "DEN", 20, 10, home_qb="QB-A"),
            _schedule_row(2023, 18, "KC", "LV", 20, 10, home_qb="QB-A"),
            _schedule_row(2024, 1, "KC", "LAC", 20, 10, home_qb="QB-A"),
        ]
    )
    as_of = AsOf(2024, 2)
    feats = qb_continuity_features(as_of.visible(build_starter_history(schedules)), as_of, "KC", "QB-A", False)
    assert feats.starter_changed is False
    assert feats.games_started_this_season == 1  # only the 2024 game counts


def test_qb_continuity_unresolvable_starter_is_neutral_not_an_error():
    empty = pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64, "game_type": pl.Utf8, "team": pl.Utf8, "starter_id": pl.Utf8})
    feats = qb_continuity_features(empty, AsOf(2024, 1), "KC", None, True)
    assert feats == QBContinuityFeatures(None, True, False, 0)


# ---------------------------------------------------------------------------
# Feature table: schema + leakage
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_data():
    from survivor.data.nflverse import load_pbp, load_schedules

    schedules = load_schedules()
    pbp = load_pbp([2024, 2025])
    return schedules, build_team_game_table(pbp), build_drive_table(pbp)


def test_feature_table_has_every_declared_column(real_data):
    schedules, team_game_table, drive_table = real_data
    games = schedules.filter(
        (pl.col("season") == 2025) & (pl.col("game_type") == "REG") & (pl.col("week") <= 3)
    )
    table = build_feature_table(games, team_game_table, drive_table, schedules)

    declared = feature_columns((3, 5))
    assert set(declared) <= set(table.columns)
    assert table.height > 0
    # A column that is null for every single row is a silently-broken
    # feature, not a feature -- catch it here rather than in the model.
    all_null = [c for c in declared if table[c].null_count() == table.height]
    assert all_null == []


def test_feature_table_drops_ties_and_labels_decisive_games(real_data):
    schedules, team_game_table, drive_table = real_data
    games = schedules.filter(
        (pl.col("season") == 2024) & (pl.col("game_type") == "REG") & (pl.col("week") <= 6)
    )
    table = build_feature_table(games, team_game_table, drive_table, schedules, drop_ties=True)
    assert set(table["home_win"].unique().to_list()) <= {0, 1}
    assert table["home_win"].null_count() == 0


def test_feature_table_labels_unplayed_games_null(real_data):
    """Inference on a future season: features are built, label is null."""
    schedules, team_game_table, drive_table = real_data
    games = schedules.filter(
        (pl.col("season") == 2026) & (pl.col("game_type") == "REG") & (pl.col("week") == 1)
    )
    table = build_feature_table(games, team_game_table, drive_table, schedules, drop_ties=False)
    assert table.height > 0
    assert table["home_win"].null_count() == table.height
    assert table["proj_epa_diff"].null_count() == 0  # features still computed


def test_no_leakage_from_future_results(real_data):
    """THE test this model's validity rests on.

    Build week 1-3 features twice: once with the full season available, once
    with every later game's *result* erased (scores nulled, results-derived
    tables truncated). Every feature value must be identical. If anything
    reached forward in time, deleting the future would move the numbers.

    Note what is deliberately NOT erased: the schedule rows themselves.
    Who plays whom and where is published in May, so it is legitimately
    known at week 1 -- see the module docstring's three data categories.
    """
    schedules, team_game_table, drive_table = real_data
    cutoff_week = 3
    target = schedules.filter(
        (pl.col("season") == 2025) & (pl.col("game_type") == "REG") & (pl.col("week") <= cutoff_week)
    )

    full = build_feature_table(target, team_game_table, drive_table, schedules)

    future = (pl.col("season") > 2025) | ((pl.col("season") == 2025) & (pl.col("week") > cutoff_week))
    blinded_schedules = schedules.with_columns(
        pl.when(future).then(None).otherwise(pl.col("home_score")).alias("home_score"),
        pl.when(future).then(None).otherwise(pl.col("away_score")).alias("away_score"),
        pl.when(future).then(None).otherwise(pl.col("home_qb_id")).alias("home_qb_id"),
        pl.when(future).then(None).otherwise(pl.col("away_qb_id")).alias("away_qb_id"),
    )
    blinded_target = blinded_schedules.filter(
        (pl.col("season") == 2025) & (pl.col("game_type") == "REG") & (pl.col("week") <= cutoff_week)
    )
    truncate = ~((pl.col("season") == 2025) & (pl.col("week") > cutoff_week))
    blinded = build_feature_table(
        blinded_target,
        team_game_table.filter(truncate),
        drive_table.filter(truncate),
        blinded_schedules,
    )

    assert full.height == blinded.height
    assert full.columns == blinded.columns
    for column in full.columns:
        assert full[column].to_list() == blinded[column].to_list(), f"{column} changed when the future was erased"
