"""Tests for the AsOf leakage gate (backend/survivor/clock.py).

This is the single most important correctness property in the project
(plan §1): if this gate is wrong, every downstream backtest number is a lie
no matter how careful the modeling code is. Covered here:
  - basic within-season week ordering
  - postseason stage ordering (week numbers reset for WC/DIV/CON/SB)
  - season boundaries
  - a property check against real nflverse data with random AsOf points
"""

from __future__ import annotations

import polars as pl
import pytest

from survivor.clock import AsOf, stage_rank


def _games(rows: list[tuple[int, str, int, str]]) -> pl.DataFrame:
    """rows: (season, game_type, week, game_id) -> minimal synthetic schedule."""
    return pl.DataFrame(
        {
            "season": [r[0] for r in rows],
            "game_type": [r[1] for r in rows],
            "week": [r[2] for r in rows],
            "game_id": [r[3] for r in rows],
        }
    )


def test_stage_rank_orders_regular_before_playoffs():
    assert stage_rank("REG") < stage_rank("WC") < stage_rank("DIV") < stage_rank("CON") < stage_rank("SB")


def test_stage_rank_rejects_unknown_type():
    with pytest.raises(ValueError):
        stage_rank("PRESEASON")


def test_asof_rejects_invalid_week():
    with pytest.raises(ValueError):
        AsOf(season=2024, week=0)


def test_within_season_week_ordering():
    games = _games(
        [
            (2024, "REG", 1, "g1"),
            (2024, "REG", 2, "g2"),
            (2024, "REG", 3, "g3"),
        ]
    )
    vis = AsOf(season=2024, week=3).visible(games)
    assert set(vis["game_id"].to_list()) == {"g1", "g2"}


def test_week_one_sees_nothing_from_same_season():
    games = _games([(2024, "REG", 1, "g1"), (2024, "REG", 2, "g2")])
    vis = AsOf(season=2024, week=1).visible(games)
    assert vis.height == 0


def test_prior_season_fully_visible_at_week_one():
    games = _games([(2023, "REG", 18, "g_prev_last"), (2023, "SB", 1, "g_prev_sb"), (2024, "REG", 1, "g_this")])
    vis = AsOf(season=2024, week=1).visible(games)
    assert set(vis["game_id"].to_list()) == {"g_prev_last", "g_prev_sb"}


def test_postseason_week_numbers_do_not_leak_into_regular_season():
    """WC week 1 must not be confused with REG week 1 of the same season —
    naive (season, week) comparison would wrongly treat them as equal."""
    games = _games(
        [
            (2024, "REG", 18, "g_reg18"),
            (2024, "WC", 1, "g_wc1"),
            (2024, "DIV", 1, "g_div1"),
        ]
    )
    # Standing at DIV week 1: REG 18 and WC 1 must be visible, DIV 1 (itself) must not.
    vis = AsOf(season=2024, week=1, game_type="DIV").visible(games)
    assert set(vis["game_id"].to_list()) == {"g_reg18", "g_wc1"}

    # Standing at REG week 1 of the *same* season number must not see WC/DIV
    # from... well REG week1 has nothing before it same season regardless,
    # but this asserts stage comparison isn't accidentally inverted.
    vis2 = AsOf(season=2024, week=1, game_type="REG").visible(games)
    assert vis2.height == 0


def test_super_bowl_visible_to_next_season_week_one():
    games = _games([(2023, "SB", 1, "g_sb")])
    vis = AsOf(season=2024, week=1).visible(games)
    assert vis["game_id"].to_list() == ["g_sb"]


@pytest.mark.parametrize("week", [1, 5, 10, 15, 18])
def test_no_leakage_against_real_nflverse_schedule(week):
    """Property check (plan §1): for real AsOf points, no visible game's
    gameday is on/after the earliest gameday of games at-or-after that point.
    """
    from survivor.data.nflverse import load_schedules

    games = load_schedules()
    seasons = games["season"].unique().to_list()

    checked = 0
    for season in seasons:
        asof = AsOf(season=season, week=week, game_type="REG")
        vis = asof.visible(games)

        target_and_later = games.filter(
            (pl.col("season") > season)
            | ((pl.col("season") == season) & (pl.col("game_type") != "REG"))
            | ((pl.col("season") == season) & (pl.col("game_type") == "REG") & (pl.col("week") >= week))
        )
        if target_and_later.height == 0 or vis.height == 0:
            continue
        checked += 1
        max_visible_day = vis["gameday"].max()
        min_target_day = target_and_later["gameday"].min()
        assert max_visible_day < min_target_day, (
            f"leakage at season={season} week={week}: "
            f"a visible game on {max_visible_day} is not before target games starting {min_target_day}"
        )

    assert checked > 0, "property test did not actually check anything — fixture is broken"
