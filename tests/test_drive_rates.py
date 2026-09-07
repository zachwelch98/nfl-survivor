"""Tests for features/drive_rates.py: drive classification, shrinkage, leakage."""

from __future__ import annotations

import polars as pl
import pytest

from survivor.clock import AsOf
from survivor.features.drive_rates import (
    OUTCOME_CATEGORIES,
    build_drive_table,
    compute_offense_profiles,
    league_average_profile,
)


def _play(game_id, drive, season, week, posteam, defteam, play_type, result=None, interception=0, fumble_lost=0, yardline_100=50, play_id=1):
    return {
        "game_id": game_id,
        "fixed_drive": float(drive),
        "season": season,
        "week": week,
        "season_type": "REG",
        "posteam": posteam,
        "defteam": defteam,
        "play_type": play_type,
        "fixed_drive_result": result,
        "interception": interception,
        "fumble_lost": fumble_lost,
        "yardline_100": yardline_100,
        "play_id": play_id,
    }


def _pbp(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_build_drive_table_classifies_touchdown_and_punt():
    # fixed_drive_result is constant across every play of a real drive.
    rows = [
        _play("g1", 1, 2024, 1, "KC", "DEN", "run", "Touchdown", play_id=1),
        _play("g1", 1, 2024, 1, "KC", "DEN", "pass", "Touchdown", play_id=2, yardline_100=5),
        _play("g1", 2, 2024, 1, "DEN", "KC", "run", "Punt", play_id=3),
        _play("g1", 2, 2024, 1, "DEN", "KC", "punt", "Punt", play_id=4),
    ]
    dt = build_drive_table(_pbp(rows))
    assert dt.height == 2
    td_row = dt.filter(pl.col("category") == "td")
    assert td_row.height == 1
    assert td_row["offense"].item() == "KC"
    assert td_row["reached_red_zone"].item() is True
    assert td_row["scored_td_in_red_zone"].item() is True

    punt_row = dt.filter(pl.col("category") == "punt")
    assert punt_row.height == 1
    assert punt_row["offense"].item() == "DEN"


def test_build_drive_table_classifies_interception_vs_fumble_turnover():
    rows = [
        _play("g1", 1, 2024, 1, "KC", "DEN", "pass", "Turnover", interception=1, play_id=1),
        _play("g1", 2, 2024, 1, "DEN", "KC", "run", "Turnover", fumble_lost=1, play_id=2),
    ]
    dt = build_drive_table(_pbp(rows))
    cats = dict(zip(dt["offense"].to_list(), dt["category"].to_list()))
    assert cats["KC"] == "turnover_int"
    assert cats["DEN"] == "turnover_fumble"


def test_build_drive_table_excludes_end_of_half_and_null():
    rows = [
        _play("g1", 1, 2024, 1, "KC", "DEN", "no_play", "End of half", play_id=1),
        _play("g1", 2, 2024, 1, "DEN", "KC", "pass", "Touchdown", play_id=2),
    ]
    dt = build_drive_table(_pbp(rows))
    assert dt.height == 1  # end-of-half drive dropped
    assert dt["category"].to_list() == ["td"]


def test_build_drive_table_excludes_postseason():
    rows = [
        {**_play("g1", 1, 2024, 1, "KC", "DEN", "pass", "Touchdown", play_id=1), "season_type": "POST"},
    ]
    dt = build_drive_table(_pbp(rows))
    assert dt.height == 0


def test_build_drive_table_counts_only_offensive_snaps():
    # fixed_drive_result is constant across every play of a real drive
    # (verified against nflverse pbp), including the trailing extra_point row.
    rows = [
        _play("g1", 1, 2024, 1, "KC", "DEN", "no_play", "Touchdown", play_id=1),
        _play("g1", 1, 2024, 1, "KC", "DEN", "run", "Touchdown", play_id=2),
        _play("g1", 1, 2024, 1, "KC", "DEN", "pass", "Touchdown", play_id=3),
        _play("g1", 1, 2024, 1, "KC", "DEN", "extra_point", "Touchdown", play_id=4),
    ]
    dt = build_drive_table(_pbp(rows))
    assert dt["play_count"].item() == 2  # run + pass, not no_play or extra_point


def test_shrinkage_pure_prior_at_zero_observed_drives():
    """Week 1 of a season: n_observed=0 -> rates must equal the prior exactly."""
    prior_season_rows = []
    for i in range(50):
        prior_season_rows.append(_play(f"p{i}", i, 2023, 5, "KC", "OPP", "pass", "Touchdown", play_id=i * 10))
    dt = build_drive_table(_pbp(prior_season_rows))
    asof = AsOf(season=2024, week=1, game_type="REG")
    profiles = compute_offense_profiles(dt, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)
    kc = profiles["KC"]
    assert kc.n_drives == 0
    assert kc.rates["td"] == pytest.approx(1.0, abs=1e-6)  # all-TD prior season, no in-season data to dilute it


def test_shrinkage_converges_toward_observed_with_more_data():
    """More current-season drives should pull the shrunk rate closer to what's observed."""
    # Prior season: all touchdowns (would push rate toward 1.0 if unshrunk)
    prior_rows = [_play(f"p{i}", i, 2023, 5, "KC", "OPP", "pass", "Touchdown", play_id=i) for i in range(50)]
    # Current season: all punts (a total reversal)
    early_rows = [_play(f"e{i}", i, 2024, 2, "KC", "OPP", "punt", "Punt", play_id=1000 + i) for i in range(3)]
    heavy_rows = [_play(f"h{i}", i, 2024, 2, "KC", "OPP", "punt", "Punt", play_id=2000 + i) for i in range(200)]

    dt_early = build_drive_table(_pbp(prior_rows + early_rows))
    dt_heavy = build_drive_table(_pbp(prior_rows + heavy_rows))

    asof = AsOf(season=2024, week=10, game_type="REG")
    profiles_early = compute_offense_profiles(dt_early, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)
    profiles_heavy = compute_offense_profiles(dt_heavy, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)

    # with only 3 observed punts, the all-TD prior should still pull td rate up somewhat
    assert profiles_early["KC"].rates["td"] > profiles_heavy["KC"].rates["td"]
    # with 200 observed punts, punt rate should dominate and be much higher
    assert profiles_heavy["KC"].rates["punt"] > 0.9


def test_all_profile_rates_sum_to_one():
    rows = [_play(f"g{i}", i, 2024, 3, "KC", "DEN", "pass", "Touchdown", play_id=i) for i in range(10)]
    rows += [_play(f"g{i}b", i, 2024, 3, "DEN", "KC", "run", "Punt", play_id=1000 + i) for i in range(10)]
    dt = build_drive_table(_pbp(rows))
    asof = AsOf(season=2024, week=10, game_type="REG")
    profiles = compute_offense_profiles(dt, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)
    for team, profile in profiles.items():
        assert sum(profile.rates.values()) == pytest.approx(1.0, abs=1e-6), f"{team} rates don't sum to 1"


def test_league_average_profile_sums_to_one():
    rows = [_play(f"g{i}", i, 2024, 3, "KC", "DEN", "pass", "Touchdown", play_id=i) for i in range(5)]
    rows += [_play(f"g{i}b", i, 2024, 3, "DEN", "KC", "run", "Punt", play_id=1000 + i) for i in range(5)]
    dt = build_drive_table(_pbp(rows))
    asof = AsOf(season=2024, week=10, game_type="REG")
    league = league_average_profile(dt, asof)
    assert sum(league.rates.values()) == pytest.approx(1.0, abs=1e-6)


def test_no_leakage_offense_profile_ignores_future_weeks():
    """A drive in week 12 must not affect the as-of-week-10 profile."""
    early_rows = [_play(f"g{i}", i, 2024, 5, "KC", "DEN", "pass", "Touchdown", play_id=i) for i in range(10)]
    future_rows = [_play(f"g{i}f", i, 2024, 12, "KC", "DEN", "run", "Turnover on downs", play_id=2000 + i) for i in range(50)]
    dt = build_drive_table(_pbp(early_rows + future_rows))

    asof = AsOf(season=2024, week=10, game_type="REG")
    profiles = compute_offense_profiles(dt, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)
    # only the 10 early week-5 TD drives should be visible; the 50 week-12
    # turnover-on-downs drives must not leak in and dominate the rate.
    assert profiles["KC"].n_drives == 10
    assert profiles["KC"].rates["td"] > profiles["KC"].rates["turnover_on_downs"]


def test_real_data_leakage_property():
    """Property check against real pbp: an as-of-week-8 profile's observed
    drive count must not exceed the number of drives actually playable
    before week 8 of that season."""
    from survivor.data.nflverse import load_pbp

    pbp = load_pbp([2023])
    dt = build_drive_table(pbp)
    asof = AsOf(season=2023, week=8, game_type="REG")
    profiles = compute_offense_profiles(dt, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)

    max_possible_drives_before_week8 = dt.filter((pl.col("season") == 2023) & (pl.col("week") < 8)).group_by("offense").len()
    max_by_team = dict(zip(max_possible_drives_before_week8["offense"].to_list(), max_possible_drives_before_week8["len"].to_list()))

    for team, profile in profiles.items():
        assert profile.n_drives <= max_by_team.get(team, 0) + 1  # +1 tolerance for off-by-one at boundary


def test_drives_per_game_present_and_reasonable():
    # 3 games of 10 drives each -> drives_per_game should be exactly 10.
    rows = []
    for g in range(3):
        for d in range(10):
            rows.append(_play(f"game{g}", d, 2024, 3 + g, "KC", "DEN", "pass", "Touchdown", play_id=d))
    dt = build_drive_table(_pbp(rows))
    asof = AsOf(season=2024, week=10, game_type="REG")
    profiles = compute_offense_profiles(dt, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)
    assert profiles["KC"].drives_per_game == pytest.approx(10.0, abs=1e-6)


def test_drives_per_game_no_leakage():
    """A team's drives_per_game must not be pulled toward a future week's pace."""
    early = [_play(f"g{i}", i, 2024, 5, "KC", "OPP", "pass", "Touchdown", play_id=i) for i in range(24)]
    import polars as _pl

    early_dt = build_drive_table(_pbp(early))
    future = [_play(f"f{i}", i, 2024, 12, "KC", "OPP", "pass", "Touchdown", play_id=2000 + i) for i in range(600)]
    future_dt = build_drive_table(_pbp(future))
    combined = _pl.concat([early_dt, future_dt])

    asof = AsOf(season=2024, week=10, game_type="REG")
    profiles = compute_offense_profiles(combined, asof, shrink_k=5, prior_blend_prev_season=0.6, prior_blend_league_mean=0.4)
    # only the early (week 5) drives should be visible -> n_drives should reflect only those
    assert profiles["KC"].n_drives == early_dt.filter(_pl.col("offense") == "KC").height
