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
