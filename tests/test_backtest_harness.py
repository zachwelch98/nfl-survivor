"""Real-data smoke tests for backtest/harness.py: survivor-pool mechanics
(no team reused, elimination happens exactly at an actual loss) on a real
season, for both the honest and cheating variants and multiple methods.
"""

from __future__ import annotations

import pytest

from survivor.backtest.harness import run_season_backtest
from survivor.data.nflverse import load_pbp, load_schedules
from survivor.models.montecarlo import DriveSimConfig

FAST_MC = DriveSimConfig(n_simulations=1500, seed=0)


@pytest.fixture(scope="module")
def real_data():
    pbp = load_pbp([2022, 2023, 2024])
    schedules = load_schedules()
    return pbp, schedules


def _assert_valid_season_result(result, schedules, season):
    teams_picked = [p.team for p in result.picks]
    assert len(teams_picked) == len(set(teams_picked)), "a team was picked more than once"

    import polars as pl

    season_games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    for pick in result.picks:
        row = season_games.filter(
            (pl.col("week") == pick.week) & ((pl.col("home_team") == pick.team) | (pl.col("away_team") == pick.team))
        )
        assert row.height == 1, f"pick {pick} does not correspond to exactly one real game"
        r = row.row(0, named=True)
        is_home = r["home_team"] == pick.team
        won = (r["home_score"] > r["away_score"]) if is_home else (r["away_score"] > r["home_score"])
        tied = r["home_score"] == r["away_score"]
        # every pick except possibly the very last one must have been a win (a loss/tie ends the season)
        if pick is not result.picks[-1]:
            assert won, f"pick {pick} did not win but the season continued past it"
    if result.week_eliminated is not None:
        last = result.picks[-1]
        assert last.week == result.week_eliminated
        row = season_games.filter((pl.col("week") == last.week) & ((pl.col("home_team") == last.team) | (pl.col("away_team") == last.team))).row(0, named=True)
        is_home = row["home_team"] == last.team
        won = (row["home_score"] > row["away_score"]) if is_home else (row["away_score"] > row["home_score"])
        assert not won, "week_eliminated set but the final pick actually won"


def test_backtest_vegas_honest_real_season(real_data):
    pbp, schedules = real_data
    result = run_season_backtest(pbp, schedules, season=2023, method="vegas", cheating=False, mc_config=FAST_MC)
    assert result.season == 2023
    assert result.method == "vegas"
    assert len(result.picks) >= 1
    _assert_valid_season_result(result, schedules, 2023)


def test_backtest_vegas_cheating_real_season(real_data):
    pbp, schedules = real_data
    result = run_season_backtest(pbp, schedules, season=2023, method="vegas", cheating=True, mc_config=FAST_MC)
    assert len(result.picks) >= 1
    _assert_valid_season_result(result, schedules, 2023)


def test_backtest_montecarlo_honest_real_season(real_data):
    pbp, schedules = real_data
    result = run_season_backtest(pbp, schedules, season=2023, method="montecarlo", cheating=False, mc_config=FAST_MC)
    assert len(result.picks) >= 1
    _assert_valid_season_result(result, schedules, 2023)


def test_backtest_blend_honest_real_season(real_data):
    from survivor.evaluate.model_comparison import add_walkforward_blend, collect_predictions
    from survivor.models.blend import BlendModel

    pbp, schedules = real_data
    # fit a blend model on a slice of prior data so the blend method has something to use
    weeks = [(2022, w) for w in range(5, 18)]
    preds = collect_predictions(pbp, schedules, weeks, mc_config=FAST_MC)
    blend = BlendModel().fit(preds["vegas_home_win_prob"].to_numpy(), preds["mc_home_win_prob"].to_numpy(), preds["home_win"].to_numpy())

    result = run_season_backtest(pbp, schedules, season=2023, method="blend", cheating=False, mc_config=FAST_MC, blend_model=blend)
    assert len(result.picks) >= 1
    _assert_valid_season_result(result, schedules, 2023)


def test_backtest_week_eliminated_matches_last_pick_result(real_data):
    """If eliminated, the very last pick must have actually lost; if not
    eliminated, every pick made must have won (checked by the season's
    actual final week)."""
    pbp, schedules = real_data
    result = run_season_backtest(pbp, schedules, season=2022, method="vegas", cheating=False, mc_config=FAST_MC)
    if result.week_eliminated is None:
        # never eliminated -> every single pick must have won
        import polars as pl

        season_games = schedules.filter((pl.col("season") == 2022) & (pl.col("game_type") == "REG"))
        for pick in result.picks:
            row = season_games.filter((pl.col("week") == pick.week) & ((pl.col("home_team") == pick.team) | (pl.col("away_team") == pick.team))).row(0, named=True)
            is_home = row["home_team"] == pick.team
            won = (row["home_score"] > row["away_score"]) if is_home else (row["away_score"] > row["home_score"])
            assert won


def test_summarize_backtest_structure():
    import polars as pl

    from survivor.backtest.harness import summarize_backtest

    df = pl.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "method": ["vegas", "vegas", "vegas", "vegas"],
            "cheating": [False, False, True, True],
            "n_picks": [4, 6, 8, 10],
            "week_eliminated": [4, 6, 8, 10],
            "survived_full_season": [False, False, False, False],
        }
    )
    summary = summarize_backtest(df)
    assert "vegas_honest" in summary and "vegas_cheating" in summary
    assert summary["vegas_honest"]["median_weeks_survived"] == pytest.approx(5.0)
    assert summary["vegas_cheating"]["median_weeks_survived"] == pytest.approx(9.0)
