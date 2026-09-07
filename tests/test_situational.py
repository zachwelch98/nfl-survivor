"""Tests for features/situational.py: rest, travel, weather, motivation."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from survivor.clock import AsOf
from survivor.features.situational import (
    MOTIVATION_START_WEEK,
    _rank_playoff_teams,
    motivation_efficiency_haircut,
    motivation_status,
    rest_features,
    simulate_playoff_probabilities,
    standings_as_of,
    team_home_base,
    travel_features,
    weather_features,
)


# ---------------------------------------------------------------------------
# Rest
# ---------------------------------------------------------------------------


def test_rest_features_none_on_missing_data():
    assert rest_features(None, 7) is None
    assert rest_features(7, None) is None


def test_rest_features_short_week_and_bye_flags():
    f = rest_features(home_rest=4, away_rest=13)
    assert f.rest_advantage_home == 4 - 13
    assert f.home_short_week is True
    assert f.away_short_week is False
    assert f.home_off_bye is False
    assert f.away_off_bye is True


def test_rest_features_normal_week_no_flags():
    f = rest_features(home_rest=7, away_rest=7)
    assert f.rest_advantage_home == 0
    assert not f.home_short_week and not f.away_short_week
    assert not f.home_off_bye and not f.away_off_bye


# ---------------------------------------------------------------------------
# Travel
# ---------------------------------------------------------------------------


def test_team_home_base_relocated_franchise():
    from survivor.data.nflverse import load_schedules

    games = load_schedules()
    assert team_home_base(games, "KC", 2024) is not None
    assert team_home_base(games, "LV", 2024) is not None  # relocated OAK->LV


def test_team_home_base_unknown_team_returns_none():
    schedules = pl.DataFrame({"season": [2024], "home_team": ["KC"], "game_type": ["REG"], "stadium_id": ["KAN00"]})
    assert team_home_base(schedules, "ZZZ", 2024) is None


def test_travel_features_neutral_site_detected():
    schedules = pl.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "home_team": ["KC", "KC", "KC"],
            "game_type": ["REG", "REG", "REG"],
            "stadium_id": ["KAN00", "KAN00", "KAN00"],
        }
    )
    # KC's home base is KAN00; playing a "home" game at LON00 (Wembley) should flag neutral site.
    f = travel_features(schedules, "KC", 2024, "LON00", date(2024, 10, 1), is_designated_home=True)
    assert f is not None
    assert f.is_neutral_site is True
    assert f.travel_miles > 4000  # KC to London is a long way

    g = travel_features(schedules, "KC", 2024, "KAN00", date(2024, 10, 1), is_designated_home=True)
    assert g.is_neutral_site is False
    assert g.travel_miles == 0.0


def test_travel_features_unknown_team_returns_none():
    schedules = pl.DataFrame({"season": [2024], "home_team": ["KC"], "game_type": ["REG"], "stadium_id": ["KAN00"]})
    assert travel_features(schedules, "ZZZ", 2024, "KAN00", date(2024, 10, 1), is_designated_home=False) is None


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


def test_weather_dome_is_neutral():
    f = weather_features(roof="dome", temp=None, wind=None)
    assert f.is_dome is True
    assert f.pass_efficiency_multiplier == 1.0
    assert f.fg_success_multiplier == 1.0


def test_weather_cold_and_windy_outdoors_suppresses_efficiency():
    f = weather_features(roof="outdoors", temp=20, wind=20)
    assert f.is_dome is False
    assert f.cold_game is True
    assert f.high_wind is True
    assert f.pass_efficiency_multiplier < 1.0
    assert f.fg_success_multiplier < 1.0


def test_weather_mild_outdoors_no_penalty():
    f = weather_features(roof="outdoors", temp=65, wind=5)
    assert f.pass_efficiency_multiplier == 1.0
    assert f.fg_success_multiplier == 1.0


# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------


def _schedule_row(season, week, home, away, home_score, away_score, game_type="REG"):
    return {
        "season": season, "week": week, "game_type": game_type,
        "home_team": home, "away_team": away, "home_score": home_score, "away_score": away_score,
    }


def test_standings_as_of_tallies_wins_losses_ties():
    rows = [
        _schedule_row(2024, 1, "KC", "DEN", 20, 10),  # KC win, DEN loss
        _schedule_row(2024, 2, "KC", "LAC", 14, 14),  # tie
        _schedule_row(2024, 3, "DEN", "KC", 30, 20),  # KC loss, DEN win
    ]
    schedules = pl.DataFrame(rows)
    asof = AsOf(season=2024, week=4, game_type="REG")
    standings = standings_as_of(schedules, asof)
    assert standings["KC"].wins == 1
    assert standings["KC"].losses == 1
    assert standings["KC"].ties == 1
    assert standings["KC"].games_played == 3
    assert standings["KC"].win_pct == pytest.approx((1 + 0.5) / 3)


def test_standings_as_of_excludes_future_games():
    rows = [
        _schedule_row(2024, 1, "KC", "DEN", 20, 10),
        _schedule_row(2024, 10, "KC", "DEN", 40, 0),  # future relative to as_of week 4
    ]
    schedules = pl.DataFrame(rows)
    asof = AsOf(season=2024, week=4, game_type="REG")
    standings = standings_as_of(schedules, asof)
    assert standings["KC"].games_played == 1


# ---------------------------------------------------------------------------
# Playoff ranking / motivation
# ---------------------------------------------------------------------------


def test_rank_playoff_teams_selects_division_winners_and_wildcards():
    import numpy as np

    from survivor.data.nflverse import load_teams

    teams_df = load_teams()
    team_division = dict(zip(teams_df["team_abbr"].to_list(), teams_df["team_division"].to_list()))
    team_conf = dict(zip(teams_df["team_abbr"].to_list(), teams_df["team_conf"].to_list()))

    # give every team a distinct win_pct so there are no ties to worry about
    all_teams = sorted(team_division.keys())
    win_pct = {t: 0.5 + 0.001 * i for i, t in enumerate(all_teams)}

    rng = np.random.default_rng(0)
    made_it = _rank_playoff_teams(win_pct, team_division, team_conf, rng)

    assert len(made_it) == 14  # 7 per conference x 2 conferences

    # the single best team in each division must always make it (division winner)
    for div in set(team_division.values()):
        div_teams = [t for t in all_teams if team_division[t] == div]
        best = max(div_teams, key=lambda t: win_pct[t])
        assert best in made_it


def test_simulate_playoff_probabilities_extremes():
    """A team guaranteed to win every remaining game should end up with a
    high playoff probability; a team guaranteed to lose every remaining
    game, a low one."""
    from survivor.data.nflverse import load_schedules, load_teams

    teams_df = load_teams()
    team_division = dict(zip(teams_df["team_abbr"].to_list(), teams_df["team_division"].to_list()))
    team_conf = dict(zip(teams_df["team_abbr"].to_list(), teams_df["team_conf"].to_list()))

    schedules = load_schedules()
    as_of = AsOf(season=2023, week=15, game_type="REG")

    remaining = schedules.filter(
        (pl.col("season") == 2023) & (pl.col("game_type") == "REG") & (pl.col("week") >= 15)
    ).select(["home_team", "away_team"])

    # KC guaranteed to win every remaining game it's in; CAR guaranteed to lose every one.
    tie = 0.002

    def _prob_row(home, away):
        if home == "KC" or away == "KC":
            hw = 1 - tie if home == "KC" else tie
        elif home == "CAR" or away == "CAR":
            hw = tie if home == "CAR" else 1 - tie
        else:
            hw = 0.5
        return hw, 1 - hw - tie, tie

    rows = []
    for row in remaining.iter_rows(named=True):
        hw, aw, tw = _prob_row(row["home_team"], row["away_team"])
        rows.append({"home_team": row["home_team"], "away_team": row["away_team"], "home_win_prob": hw, "away_win_prob": aw, "tie_prob": tw})
    remaining_probs = pl.DataFrame(rows)

    probs = simulate_playoff_probabilities(schedules, as_of, remaining_probs, team_division, team_conf, n_sims=300, seed=1)

    assert probs["KC"] > 0.7  # KC was already very strong in 2023 and wins out
    assert probs["CAR"] < 0.05  # CAR was the league's worst team in 2023 and loses out


def test_motivation_status_before_week_15_is_always_fighting():
    asof = AsOf(season=2024, week=10, game_type="REG")
    status = motivation_status({"KC": 0.99, "CAR": 0.01}, asof)
    assert status == {"KC": "fighting", "CAR": "fighting"}


def test_motivation_status_after_week_15_classifies_correctly():
    asof = AsOf(season=2024, week=MOTIVATION_START_WEEK, game_type="REG")
    status = motivation_status({"KC": 0.99, "CAR": 0.01, "GB": 0.5}, asof)
    assert status["KC"] == "clinched"
    assert status["CAR"] == "eliminated"
    assert status["GB"] == "fighting"


def test_motivation_efficiency_haircut():
    assert motivation_efficiency_haircut("fighting") == 1.0
    assert motivation_efficiency_haircut("clinched", magnitude=0.03) == pytest.approx(0.97)
    assert motivation_efficiency_haircut("eliminated", magnitude=0.03) == pytest.approx(0.97)
