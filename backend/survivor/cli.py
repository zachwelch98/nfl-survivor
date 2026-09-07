"""Weekly operation (plan §7): refresh data, recommend a pick, export the
JSON bundle the frontend reads. Thin orchestration over everything else in
this package — no new modeling logic lives here.

    survivor refresh              # re-pull nflverse data
    survivor recommend --season 2026 --week 1 --used KC,BUF
    survivor export --season 2026 --week 1 --used KC,BUF
    survivor backtest --start 2010 --end 2025
"""

from __future__ import annotations

import click
import polars as pl

from survivor.config import CONFIG
from survivor.features.drive_rates import build_drive_table
from survivor.features.team_ratings import build_team_game_table


def _default_pbp_seasons(season: int, schedules) -> list[int]:
    """[season-1, season] normally, but nflverse has no play-by-play for a
    season with zero games played yet (e.g. recommending for next week's
    season before it kicks off), and nflreadpy rejects such a season
    outright — drop it from the request in that case, since drive_rates.py
    would find nothing there to use anyway."""
    import polars as pl

    played = schedules.filter((pl.col("season") == season) & pl.col("home_score").is_not_null())
    if played.height == 0:
        return [season - 1]
    return [season - 1, season]


def _ml_pbp_seasons(season: int) -> list[int]:
    """Seasons of play-by-play Method 3 needs loaded.

    Unlike Methods 1 and 2 — which only ever look back one season from each
    as-of point, so `[season-1, season]` suffices — Method 3 trains across
    every season from `train_floor_season` forward, and models/ml.py drops
    (rather than imputes) any training row with no play-by-play history. Load
    less than this and the model quietly has almost nothing to learn from.
    One extra season below the floor gives the first training season its
    prior-season baseline.
    """
    floor = CONFIG.model.ml.train_floor_season
    return list(range(floor - 1, season + 1))


def _parse_used(used: str | None) -> list[str]:
    if not used:
        return []
    return [t.strip().upper() for t in used.split(",") if t.strip()]


def _fit_backtest_ml_model(team_game_table, drive_table, schedules, season: int):
    """Method 3 fit once per backtest season on strictly-prior seasons —
    the season-cadence pattern, and the reason run_season_backtest takes a
    pre-fit model rather than fitting one itself."""
    from survivor.clock import AsOf
    from survivor.export.json_bundle import _fit_ml_model

    train = AsOf(season=season, week=1, game_type="REG").visible(schedules)
    return _fit_ml_model(team_game_table, drive_table, schedules, train)


@click.group()
def cli() -> None:
    """NFL Survivor Pool ML Advisor."""


@cli.command()
@click.option("--pbp-seasons", default=None, help="Comma-separated seasons to refresh play-by-play for (e.g. 2024,2025). Defaults to none (schedules/teams/injuries/depth charts only).")
def refresh(pbp_seasons: str | None) -> None:
    """Force-refetch all cached nflverse data."""
    from survivor.data.nflverse import refresh_all

    seasons = [int(s) for s in pbp_seasons.split(",")] if pbp_seasons else None
    refresh_all(pbp_seasons=seasons)
    click.echo("Refreshed schedules, teams, injuries, depth charts" + (f", and pbp for {seasons}" if seasons else "."))


@cli.command()
@click.option("--season", type=int, required=True)
@click.option("--week", type=int, required=True)
@click.option("--used", default="", help="Comma-separated teams already used this season, e.g. KC,BUF")
@click.option("--pbp-seasons", default=None, help="Comma-separated seasons of play-by-play to load (defaults to season and season-1).")
def recommend(season: int, week: int, used: str, pbp_seasons: str | None) -> None:
    """Print this week's recommended pick and the full remaining-season plan."""
    from survivor.data.nflverse import load_pbp, load_schedules
    from survivor.export.json_bundle import build_season_plan, build_week_board

    used_teams = _parse_used(used)
    schedules = load_schedules()
    seasons = [int(s) for s in pbp_seasons.split(",")] if pbp_seasons else _ml_pbp_seasons(season)
    pbp = load_pbp(seasons)
    drive_table = build_drive_table(pbp)
    team_game_table = build_team_game_table(pbp)

    board = build_week_board(
        schedules, drive_table, season, week, used_teams, team_game_table=team_game_table
    )
    click.echo(f"\n=== Week {week}, {season} — Recommended pick: {board['recommended_pick']} ===\n")
    for g in sorted(board["games"], key=lambda g: g["vegas_home_win_prob"], reverse=True):
        mc = f"{g['mc_home_win_prob']:.3f}" if g["mc_home_win_prob"] is not None else "  n/a"
        ml = f"{g['ml_home_win_prob']:.3f}" if g["ml_home_win_prob"] is not None else "  n/a"
        click.echo(
            f"  {g['away_team']:>4s} @ {g['home_team']:<4s}  spread={g['spread_line']:+.1f}  "
            f"vegas_home={g['vegas_home_win_prob']:.3f}  mc_home={mc}  ml_home={ml}"
        )

    click.echo("\nCost of deviation (extra season-long risk vs. the optimal pick):")
    for team, cost in sorted(board["cost_of_deviation"].items(), key=lambda kv: kv[1]):
        click.echo(f"  {team:4s}  {cost:.4f}")

    plan = build_season_plan(
        schedules, drive_table, season, week, used_teams, team_game_table=team_game_table
    )
    click.echo(f"\n=== Full remaining-season plan (total survival prob: {plan['total_survival_prob']:.4%}) ===" if plan["total_survival_prob"] else "\nNo feasible full-season plan.")
    for p in plan["recommended_sequence"]:
        click.echo(f"  week {p['week']:2d}: {p['team']:4s}  p={p['survival_prob']:.3f}")


@cli.command()
@click.option("--season", type=int, required=True)
@click.option("--week", type=int, required=True)
@click.option("--used", default="")
@click.option("--pbp-seasons", default=None)
@click.option("--with-analysis/--no-analysis", default=False, help="Also (re)compute calibration/backtest/simulation JSON — slower.")
def export(season: int, week: int, used: str, pbp_seasons: str | None, with_analysis: bool) -> None:
    """Write the full JSON bundle the frontend reads."""
    from survivor.data.nflverse import load_pbp, load_schedules
    from survivor.export.json_bundle import export_all

    used_teams = _parse_used(used)
    schedules = load_schedules()
    seasons = [int(s) for s in pbp_seasons.split(",")] if pbp_seasons else _ml_pbp_seasons(season)
    pbp = load_pbp(seasons)

    comparison_predictions = None
    backtest_results = None
    strategy_summary = None

    if with_analysis:
        from survivor.evaluate.model_comparison import add_walkforward_blend, collect_predictions, summarize_comparison
        from survivor.backtest.harness import run_season_backtest, summarize_backtest
        from survivor.simulate.season import precompute_weekly_survival_tables, simulate_strategies, summarize_strategy_outcomes

        # comparison_weeks spans multiple past seasons and each of those
        # needs its own prior-season pbp too (drive_rates.py's "this_season
        # + prior_season" lookback) — the narrow `pbp` loaded above (scoped
        # to just the recommendation) isn't wide enough for this.
        # Widened for Method 3: it trains from train_floor_season forward,
        # while the model comparison itself only needs the last few seasons.
        analysis_pbp = load_pbp(_ml_pbp_seasons(season - 1))

        click.echo("Computing model comparison (this takes a while)...")
        comparison_weeks = [(s, w) for s in range(season - 4, season) for w in range(1, 19)]
        comparison_predictions = collect_predictions(
            analysis_pbp,
            schedules,
            comparison_weeks,
            team_game_table=build_team_game_table(analysis_pbp),
        )
        comparison_predictions = add_walkforward_blend(comparison_predictions)

        click.echo("Running backtest across historical seasons...")
        rows = []
        for bt_season in range(CONFIG.season.backtest_seasons[0], CONFIG.season.backtest_seasons[1] + 1):
            bt_pbp = load_pbp([bt_season - 1, bt_season])
            for method in ["vegas", "montecarlo"]:
                for cheating in [False, True]:
                    result = run_season_backtest(bt_pbp, schedules, bt_season, method, cheating)
                    rows.append({
                        "season": bt_season, "method": method, "cheating": cheating,
                        "n_picks": len(result.picks),
                        "week_eliminated": result.week_eliminated,
                        "survived_full_season": result.week_eliminated is None,
                    })
        backtest_results = pl.DataFrame(rows)

        click.echo("Running strategy simulation...")
        tables = precompute_weekly_survival_tables(analysis_pbp, schedules, season - 1)
        weeks = sorted(tables.keys())
        all_teams = sorted({t for w in weeks for (tw, t) in tables[w].keys()})
        outcomes = simulate_strategies(tables, weeks, all_teams)
        strategy_summary = summarize_strategy_outcomes(outcomes, weeks)

    export_all(
        schedules, pbp, season, week, used_teams,
        comparison_predictions=comparison_predictions,
        backtest_results=backtest_results,
        strategy_summary=strategy_summary,
    )
    click.echo(f"Exported JSON bundle to {CONFIG.paths.resolve('frontend_data')}")


@cli.command()
@click.option("--start", type=int, default=None)
@click.option("--end", type=int, default=None)
def backtest(start: int | None, end: int | None) -> None:
    """Run the backtest harness across a season range and print the summary."""
    from survivor.data.nflverse import load_pbp, load_schedules
    from survivor.backtest.harness import run_season_backtest, summarize_backtest

    start = start or CONFIG.season.backtest_seasons[0]
    end = end or CONFIG.season.backtest_seasons[1]
    schedules = load_schedules()

    # Method 3 trains across every prior season, so its feature tables are
    # built once from wide play-by-play outside the loop. Extra future rows
    # sitting in those tables are not a leak: every as-of point filters them
    # out via AsOf.visible() before anything reads them.
    from survivor.models.ml import MLWinProbModel

    wide_pbp = load_pbp(_ml_pbp_seasons(end))
    wide_drive_table = build_drive_table(wide_pbp)
    wide_team_game_table = build_team_game_table(wide_pbp)

    rows = []
    for season in range(start, end + 1):
        pbp = load_pbp([season - 1, season])
        ml_model = _fit_backtest_ml_model(
            wide_team_game_table, wide_drive_table, schedules, season
        )
        methods = ["vegas", "montecarlo"] + (["ml"] if ml_model is not None else [])
        for method in methods:
            for cheating in [False, True]:
                result = run_season_backtest(
                    pbp, schedules, season, method, cheating, ml_model=ml_model
                )
                click.echo(f"{season} {method:11s} cheating={cheating!s:5s} weeks_survived={len(result.picks):2d}")
                rows.append({
                    "season": season, "method": method, "cheating": cheating,
                    "n_picks": len(result.picks), "week_eliminated": result.week_eliminated,
                    "survived_full_season": result.week_eliminated is None,
                })

    summary = summarize_backtest(pl.DataFrame(rows))
    click.echo("\n=== SUMMARY ===")
    for key, s in summary.items():
        click.echo(f"{key:20s} median={s['median_weeks_survived']:.1f}  mean={s['mean_weeks_survived']:.2f}  P(reach wk10)={s['p_survive_to_week10']:.3f}")


if __name__ == "__main__":
    cli()
