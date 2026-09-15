"""Smoke tests for cli.py: the click commands are wired correctly and
produce output against real live data."""

from __future__ import annotations

from click.testing import CliRunner

from survivor.cli import cli


def test_recommend_command_runs_against_real_data():
    runner = CliRunner()
    result = runner.invoke(cli, ["recommend", "--season", "2026", "--week", "1", "--pbp-seasons", "2024,2025"])
    assert result.exit_code == 0, result.output
    assert "Recommended pick" in result.output
    assert "Cost of deviation" in result.output


def test_recommend_respects_used_teams():
    runner = CliRunner()
    result = runner.invoke(cli, ["recommend", "--season", "2026", "--week", "1", "--used", "LAC", "--pbp-seasons", "2024,2025"])
    assert result.exit_code == 0, result.output
    assert "Recommended pick: LAC" not in result.output


def test_export_command_writes_files(tmp_path, monkeypatch):
    from survivor import config as config_module

    monkeypatch.setattr(config_module.CONFIG.paths, "frontend_data", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "--season", "2026", "--week", "1", "--pbp-seasons", "2024,2025"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "season_meta.json").exists()
    assert (tmp_path / "week_01_board.json").exists()
    assert (tmp_path / "season_plan.json").exists()


def test_refresh_default_pbp_seasons_follow_the_schedule():
    """The seasons a routine refresh keeps warm used to be hardcoded in the
    Makefile, which meant they went stale every September -- the moment the
    new season's play-by-play began to exist was the moment `make refresh`
    stopped fetching it. They are now derived from the schedule."""
    import polars as pl

    from survivor.data.nflverse import current_season, default_pbp_seasons, has_played_games

    schedules = pl.DataFrame(
        {
            "season": [2025, 2025, 2026, 2026],
            "week": [1, 2, 1, 2],
            "home_score": [20, 17, 13, None],
        },
        schema={"season": pl.Int64, "week": pl.Int64, "home_score": pl.Int64},
    )

    assert current_season(schedules) == 2026
    assert has_played_games(schedules, 2026) is True
    # Season in progress plus the one before it.
    assert default_pbp_seasons(schedules) == [2025, 2026]


def test_refresh_default_drops_a_season_that_has_not_kicked_off():
    """nflreadpy rejects a play-by-play request for a season with no games
    rather than returning an empty frame, so the season must be dropped until
    it actually starts -- the normal state all spring and summer."""
    import polars as pl

    from survivor.data.nflverse import default_pbp_seasons, has_played_games

    schedules = pl.DataFrame(
        {
            "season": [2025, 2026],
            "week": [1, 1],
            "home_score": [20, None],
        },
        schema={"season": pl.Int64, "week": pl.Int64, "home_score": pl.Int64},
    )

    assert has_played_games(schedules, 2026) is False
    assert default_pbp_seasons(schedules) == [2025]
    # Explicitly asking about a started season still works.
    assert default_pbp_seasons(schedules, season=2025) == [2024, 2025]


def test_refresh_no_pbp_flag_skips_play_by_play(monkeypatch):
    """--no-pbp keeps the old "metadata only" behaviour available now that a
    bare `refresh` fetches play-by-play by default."""
    from survivor.data import nflverse

    calls: dict[str, object] = {}
    monkeypatch.setattr(nflverse, "load_schedules", lambda refresh=False: calls.setdefault("schedules", True))
    monkeypatch.setattr(nflverse, "load_teams", lambda refresh=False: calls.setdefault("teams", True))
    monkeypatch.setattr(nflverse, "load_injuries", lambda refresh=False: calls.setdefault("injuries", True))
    monkeypatch.setattr(nflverse, "load_depth_charts", lambda refresh=False: calls.setdefault("depth", True))
    monkeypatch.setattr(nflverse, "load_pbp", lambda seasons, refresh=False: calls.setdefault("pbp", seasons))

    assert nflverse.refresh_all(skip_pbp=True) == []
    assert "pbp" not in calls
    assert {"schedules", "teams", "injuries", "depth"} <= set(calls)
