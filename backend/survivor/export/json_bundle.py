"""JSON export layer (plan §9): the contract between the Python pipeline
and the React frontend. Every function returns a plain, JSON-serializable
dict shaped exactly like the frontend's TypeScript types
(frontend/src/data/types.ts) — the two are hand-kept in sync rather than
generated from a shared schema, a deliberate scope cut given this
project's size; `make export` regenerates the JSON files so drift is at
least caught at review time rather than silently.

Files written into frontend/public/data/:
  season_meta.json, week_{NN}_board.json, season_plan.json,
  calibration.json, backtest.json, simulation.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from survivor.clock import AsOf
from survivor.config import CONFIG
from survivor.features.drive_rates import build_drive_table
from survivor.models.montecarlo import DriveSimConfig, combine_matchup_rates, combined_n_drives, simulate_game
from survivor.features.drive_rates import compute_defense_profiles, compute_offense_profiles, league_average_profile
from survivor.models.base import pick_survival_prob
from survivor.models.vegas import VegasModel
from survivor.optimize.assignment import InfeasibleSeasonError, solve_optimal_assignment
from survivor.optimize.kbest import cost_of_deviation_by_team, k_best_season_plans
from survivor.optimize.projection import build_survival_prob_table, simulate_all_remaining_games


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def build_season_meta(season: int, current_week: int, used_teams: list[str]) -> dict:
    return {
        "season": season,
        "current_week": current_week,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_method": CONFIG.model.primary_method,
        "tie_counts_as_loss": CONFIG.pool.tie_counts_as_loss,
        "used_teams": used_teams,
    }


def build_week_board(
    schedules: pl.DataFrame,
    drive_table: pl.DataFrame,
    season: int,
    week: int,
    used_teams: list[str],
    mc_config: DriveSimConfig | None = None,
) -> dict:
    """Per-game Vegas vs. Monte Carlo win probability, plus the optimizer's
    recommended pick (and every available team's cost-of-deviation) among
    teams not yet used. Both methods are always computed and shown side by
    side regardless of primary_method — the frontend's whole point is
    letting the user see the disagreement, per plan's "Weekly odds board."
    """
    mc_config = mc_config or DriveSimConfig(**{k: v for k, v in CONFIG.model.montecarlo.model_dump().items() if k in ("n_simulations", "seed")})
    as_of = AsOf(season=season, week=week, game_type="REG")
    train = as_of.visible(schedules)
    target = schedules.filter((pl.col("season") == season) & (pl.col("week") == week) & (pl.col("game_type") == "REG"))

    vegas = VegasModel().fit(train)
    vegas_preds = vegas.predict_proba(target)

    off = compute_offense_profiles(drive_table, as_of, CONFIG.model.montecarlo.rating_shrink_k, CONFIG.model.montecarlo.prior_blend_prev_season, CONFIG.model.montecarlo.prior_blend_league_mean)
    deff = compute_defense_profiles(drive_table, as_of, CONFIG.model.montecarlo.rating_shrink_k, CONFIG.model.montecarlo.prior_blend_prev_season, CONFIG.model.montecarlo.prior_blend_league_mean)
    league = league_average_profile(drive_table, as_of)

    games = []
    survival_prob: dict[tuple[int, str], float] = {}
    for row, vrow in zip(target.iter_rows(named=True), vegas_preds.iter_rows(named=True)):
        home, away = row["home_team"], row["away_team"]
        mc_result = None
        if home in off and away in off and home in deff and away in deff:
            home_rates = combine_matchup_rates(off[home].rates, deff[away].rates, league.rates)
            away_rates = combine_matchup_rates(off[away].rates, deff[home].rates, league.rates)
            n_drives = combined_n_drives(off[home].drives_per_game, off[away].drives_per_game, league.drives_per_game)
            mc_result = simulate_game(home_rates, away_rates, n_drives, mc_config)

        games.append({
            "game_id": row["game_id"],
            "home_team": home,
            "away_team": away,
            "gameday": row["gameday"],
            "spread_line": row["spread_line"],
            "vegas_home_win_prob": vrow["home_win_prob"],
            "vegas_away_win_prob": vrow["away_win_prob"],
            "vegas_tie_prob": vrow["tie_prob"],
            "mc_home_win_prob": mc_result.home_win_prob if mc_result else None,
            "mc_away_win_prob": mc_result.away_win_prob if mc_result else None,
            "mc_tie_prob": mc_result.tie_prob if mc_result else None,
            "mc_simulated_spread": mc_result.simulated_spread if mc_result else None,
        })

        if home not in used_teams:
            survival_prob[(week, home)] = pick_survival_prob(vrow["home_win_prob"], vrow["away_win_prob"], vrow["tie_prob"], True, CONFIG.pool.tie_counts_as_loss)
        if away not in used_teams:
            survival_prob[(week, away)] = pick_survival_prob(vrow["home_win_prob"], vrow["away_win_prob"], vrow["tie_prob"], False, CONFIG.pool.tie_counts_as_loss)

    available_teams = list(survival_prob_teams(survival_prob, week))
    recommended = None
    deviations = {}
    if available_teams:
        try:
            plan = solve_optimal_assignment([week], available_teams, survival_prob)
            recommended = plan.team_for_week(week)
            deviations = cost_of_deviation_by_team([week], available_teams, survival_prob, current_week=week)
        except InfeasibleSeasonError:
            pass

    return {
        "season": season,
        "week": week,
        "games": games,
        "recommended_pick": recommended,
        "cost_of_deviation": deviations,
    }


def survival_prob_teams(survival_prob: dict[tuple[int, str], float], week: int) -> list[str]:
    return sorted({t for (w, t) in survival_prob.keys() if w == week})


def build_season_plan(
    schedules: pl.DataFrame,
    drive_table: pl.DataFrame,
    season: int,
    current_week: int,
    used_teams: list[str],
    mc_config: DriveSimConfig | None = None,
    k_alternates: int = 3,
) -> dict:
    """The full remaining-season recommended sequence, plus the k best
    alternate season paths — plan §4's "season plan" view."""
    mc_config = mc_config or DriveSimConfig(n_simulations=CONFIG.model.montecarlo.n_simulations, seed=CONFIG.model.montecarlo.seed)
    season_games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    all_weeks = sorted(season_games["week"].unique().to_list())
    remaining_weeks = [w for w in all_weeks if w >= current_week]
    all_teams = sorted(set(season_games["home_team"].to_list()) | set(season_games["away_team"].to_list()))
    available_teams = [t for t in all_teams if t not in used_teams]

    as_of = AsOf(season=season, week=current_week, game_type="REG")
    train = as_of.visible(schedules)
    vegas = VegasModel().fit(train)

    current_target = season_games.filter(pl.col("week") == current_week)
    current_probs = vegas.predict_proba(current_target)

    future_games = season_games.filter(pl.col("week").is_in([w for w in remaining_weeks if w != current_week])).select(["game_id", "week", "home_team", "away_team"])
    mc_results = simulate_all_remaining_games(
        drive_table, schedules, as_of, future_games,
        CONFIG.model.montecarlo.rating_shrink_k, CONFIG.model.montecarlo.prior_blend_prev_season, CONFIG.model.montecarlo.prior_blend_league_mean, mc_config,
    )
    table = build_survival_prob_table(current_probs, future_games, mc_results, CONFIG.pool.tie_counts_as_loss)
    table = {(w, t): p for (w, t), p in table.items() if t in available_teams}

    try:
        plan = solve_optimal_assignment(remaining_weeks, available_teams, table)
        alternates = k_best_season_plans(remaining_weeks, available_teams, table, k=k_alternates)
    except InfeasibleSeasonError:
        return {"season": season, "current_week": current_week, "recommended_sequence": [], "alternates": [], "total_survival_prob": None}

    return {
        "season": season,
        "current_week": current_week,
        "recommended_sequence": [{"week": p.week, "team": p.team, "survival_prob": p.survival_prob} for p in plan.picks],
        "total_survival_prob": plan.total_survival_prob,
        "alternates": [
            {
                "total_survival_prob": alt.total_survival_prob,
                "picks": [{"week": p.week, "team": p.team, "survival_prob": p.survival_prob} for p in alt.picks],
            }
            for alt in alternates
        ],
    }


def export_all(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    current_week: int,
    used_teams: list[str],
    out_dir: Path | None = None,
    comparison_predictions: pl.DataFrame | None = None,
    backtest_results: pl.DataFrame | None = None,
    strategy_summary: dict | None = None,
) -> None:
    """Writes every JSON file the frontend consumes. `comparison_predictions`,
    `backtest_results`, and `strategy_summary` are optional pre-computed
    inputs (each is potentially expensive to (re)compute) — pass them in
    when already available rather than recomputing here.
    """
    from survivor.evaluate.model_comparison import summarize_comparison
    from survivor.backtest.harness import summarize_backtest

    out_dir = out_dir or CONFIG.paths.resolve("frontend_data")
    drive_table = build_drive_table(pbp)

    _write_json(out_dir / "season_meta.json", build_season_meta(season, current_week, used_teams))
    _write_json(out_dir / f"week_{current_week:02d}_board.json", build_week_board(schedules, drive_table, season, current_week, used_teams))
    _write_json(out_dir / "season_plan.json", build_season_plan(schedules, drive_table, season, current_week, used_teams))

    if comparison_predictions is not None and comparison_predictions.height > 0:
        _write_json(out_dir / "calibration.json", summarize_comparison(comparison_predictions))
    if backtest_results is not None and backtest_results.height > 0:
        _write_json(out_dir / "backtest.json", {
            "summary": summarize_backtest(backtest_results),
            "seasons": backtest_results.to_dicts(),
        })
    if strategy_summary is not None:
        _write_json(out_dir / "simulation.json", strategy_summary)
