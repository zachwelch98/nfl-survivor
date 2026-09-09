"""JSON export layer (plan §9): the contract between the Python pipeline
and the React frontend. Every function returns a plain, JSON-serializable
dict shaped exactly like the frontend's TypeScript types
(frontend/src/data/types.ts) — the two are hand-kept in sync rather than
generated from a shared schema, a deliberate scope cut given this
project's size; `make export` regenerates the JSON files so drift is at
least caught at review time rather than silently.

Files written into frontend/public/data/:
  season_meta.json, week_{NN}_board.json, season_plan.json,
  survival_matrix.json, calibration.json, backtest.json, simulation.json
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
from survivor.models.ml import MLWinProbModel
from survivor.models.vegas import VegasModel
from survivor.optimize.assignment import InfeasibleSeasonError, solve_optimal_assignment
from survivor.optimize.kbest import cost_of_deviation_by_team, k_best_season_plans
from survivor.optimize.projection import (
    DEFAULT_DECAY_RATE,
    build_survival_prob_table,
    predict_all_remaining_games_ml,
    simulate_all_remaining_games,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _fit_ml_model(
    team_game_table: pl.DataFrame | None,
    drive_table: pl.DataFrame,
    schedules: pl.DataFrame,
    train: pl.DataFrame,
    depth_charts: pl.DataFrame | None = None,
) -> MLWinProbModel | None:
    """Method 3, fit on `train`, or None if it can't be.

    Returns None rather than raising on any of the ordinary reasons a fit is
    impossible (no team_game_table passed, play-by-play too narrow to cover
    the training seasons, too few usable rows). Method 3 is an addition to
    the board, not a prerequisite for producing one — the same posture
    build_week_board already takes toward a missing Monte Carlo profile.
    """
    if team_game_table is None:
        return None
    cfg = CONFIG.model.ml
    try:
        return MLWinProbModel(
            team_game_table,
            drive_table,
            schedules,
            depth_charts,
            classifier=cfg.classifier,
            calibration_method=cfg.calibration_method,
            shrink_k=cfg.rating_shrink_k,
            prior_blend_prev_season=cfg.prior_blend_prev_season,
            prior_blend_league_mean=cfg.prior_blend_league_mean,
            opp_adj_alpha=cfg.opp_adj_alpha,
            rolling_windows=tuple(cfg.rolling_windows),
            rolling_min_games_floor=cfg.rolling_min_games_floor,
            min_train_games=cfg.min_train_games,
        ).fit(train)
    except ValueError:
        return None


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
    team_game_table: pl.DataFrame | None = None,
    depth_charts: pl.DataFrame | None = None,
    ml_model: MLWinProbModel | None = None,
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

    # Method 3 is shown alongside for comparison; the recommended pick still
    # comes from Vegas, which beats it on the current week by 0.0133 Brier
    # with the CI excluding zero (see models/ml.py and config.yaml).
    # Method 3 is fit per season, not per week, so a multi-week export fits
    # once and passes the same model in rather than paying for it 18 times.
    if ml_model is None:
        ml_model = _fit_ml_model(team_game_table, drive_table, schedules, train, depth_charts)
    ml_by_game: dict[str, dict] = {}
    if ml_model is not None:
        ml_by_game = {row["game_id"]: row for row in ml_model.predict_proba(target).iter_rows(named=True)}

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

        ml_row = ml_by_game.get(row["game_id"])

        # No posted line means no market opinion -- VegasModel falls back to
        # an even split internally, and reporting that 50% as though it were
        # a prediction would be a lie of presentation. Emit null so the board
        # shows an em dash. Lines for late-season games genuinely don't exist
        # yet: as of 2026 week 1, week 12 has 5 of 16 posted and week 18 has
        # none.
        has_line = row["spread_line"] is not None or (
            row.get("home_moneyline") is not None and row.get("away_moneyline") is not None
        )
        games.append({
            "game_id": row["game_id"],
            "home_team": home,
            "away_team": away,
            "gameday": row["gameday"],
            "spread_line": row["spread_line"],
            "vegas_home_win_prob": vrow["home_win_prob"] if has_line else None,
            "vegas_away_win_prob": vrow["away_win_prob"] if has_line else None,
            "vegas_tie_prob": vrow["tie_prob"] if has_line else None,
            "mc_home_win_prob": mc_result.home_win_prob if mc_result else None,
            "mc_away_win_prob": mc_result.away_win_prob if mc_result else None,
            "mc_tie_prob": mc_result.tie_prob if mc_result else None,
            "mc_simulated_spread": mc_result.simulated_spread if mc_result else None,
            "ml_home_win_prob": ml_row["home_win_prob"] if ml_row else None,
            "ml_away_win_prob": ml_row["away_win_prob"] if ml_row else None,
            "ml_tie_prob": ml_row["tie_prob"] if ml_row else None,
        })

        # Which method drives this board's recommendation: Vegas where a line
        # exists (it beats both others outright), otherwise Method 3, then
        # Method 2. Without this, a future week with no lines would rank every
        # team off VegasModel's internal coin flip and then star an
        # essentially arbitrary "recommended" pick.
        if has_line:
            probs = (vrow["home_win_prob"], vrow["away_win_prob"], vrow["tie_prob"])
        elif ml_row is not None:
            probs = (ml_row["home_win_prob"], ml_row["away_win_prob"], ml_row["tie_prob"])
        elif mc_result is not None:
            probs = (mc_result.home_win_prob, mc_result.away_win_prob, mc_result.tie_prob)
        else:
            probs = None

        if probs is not None:
            if home not in used_teams:
                survival_prob[(week, home)] = pick_survival_prob(*probs, True, CONFIG.pool.tie_counts_as_loss)
            if away not in used_teams:
                survival_prob[(week, away)] = pick_survival_prob(*probs, False, CONFIG.pool.tie_counts_as_loss)

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

    n_with_line = sum(1 for g in games if g["vegas_home_win_prob"] is not None)
    return {
        "season": season,
        "week": week,
        "games": games,
        "recommended_pick": recommended,
        "cost_of_deviation": deviations,
        # How many games actually have a posted line, so the frontend can say
        # plainly when a future week's board is model-driven rather than
        # market-driven.
        "n_games_with_line": n_with_line,
        "recommendation_source": "vegas" if n_with_line == len(games) else ("mixed" if n_with_line else "model"),
    }


def survival_prob_teams(survival_prob: dict[tuple[int, str], float], week: int) -> list[str]:
    return sorted({t for (w, t) in survival_prob.keys() if w == week})


def build_survival_matrix(
    schedules: pl.DataFrame,
    drive_table: pl.DataFrame,
    season: int,
    current_week: int,
    mc_config: DriveSimConfig | None = None,
    team_game_table: pl.DataFrame | None = None,
    depth_charts: pl.DataFrame | None = None,
    future_blend_model=None,
    ml_model: MLWinProbModel | None = None,
    decay_rate: float = DEFAULT_DECAY_RATE,
) -> tuple[dict[tuple[int, str], float], dict]:
    """The (week, team) -> survival_prob table for every remaining week, plus
    a JSON payload of the same numbers annotated with their matchup context.

    Returns the table **unfiltered** — no used-teams exclusion. Two callers
    want it for different reasons: `build_season_plan` filters it to the
    available teams and solves, while `survival_matrix.json` ships the whole
    thing so the frontend's what-if planner can apply its own availability
    (which lives in localStorage and can legitimately differ from whatever
    `USED=` was passed to `make export`).

    This is the expensive half of the season plan — one Monte Carlo pass over
    every remaining game plus one Method 3 fit — and it does not depend on
    which teams are used or locked. So it is computed once and shared, which
    is also why `ml_model` arrives pre-fit rather than being fit here.
    """
    mc_config = mc_config or DriveSimConfig(n_simulations=CONFIG.model.montecarlo.n_simulations, seed=CONFIG.model.montecarlo.seed)
    season_games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    all_weeks = sorted(season_games["week"].unique().to_list())
    remaining_weeks = [w for w in all_weeks if w >= current_week]
    all_teams = sorted(set(season_games["home_team"].to_list()) | set(season_games["away_team"].to_list()))

    as_of = AsOf(season=season, week=current_week, game_type="REG")
    vegas = VegasModel().fit(as_of.visible(schedules))

    current_target = season_games.filter(pl.col("week") == current_week)
    current_probs = vegas.predict_proba(current_target)

    # Method 2 only needs the four key columns; Method 3's feature builder
    # needs the full schedule row (rest, roof, weather, stadium, qb ids), so
    # keep both rather than re-selecting.
    future_games_full = season_games.filter(
        pl.col("week").is_in([w for w in remaining_weeks if w != current_week])
    )
    future_games = future_games_full.select(["game_id", "week", "home_team", "away_team"])
    mc_results = simulate_all_remaining_games(
        drive_table, schedules, as_of, future_games,
        CONFIG.model.montecarlo.rating_shrink_k, CONFIG.model.montecarlo.prior_blend_prev_season, CONFIG.model.montecarlo.prior_blend_league_mean, mc_config,
    )
    # Future weeks: Method 2 + Method 3 (see optimize/projection.py for the
    # walk-forward evidence that ml beats mc there, and that a *fitted* blend
    # beats a fixed average). Current week stays Vegas.
    ml_results = predict_all_remaining_games_ml(ml_model, future_games_full) if ml_model else {}

    table = build_survival_prob_table(
        current_probs,
        future_games,
        mc_results,
        CONFIG.pool.tie_counts_as_loss,
        decay_rate=decay_rate,
        ml_results_by_game=ml_results,
        future_blend_model=future_blend_model,
    )

    # Matchup context for each (week, team), so the frontend can show who the
    # pick would be playing without re-deriving it from 18 week boards.
    context: dict[tuple[int, str], dict] = {}
    for row in season_games.filter(pl.col("week").is_in(remaining_weeks)).iter_rows(named=True):
        w, home, away = row["week"], row["home_team"], row["away_team"]
        context[(w, home)] = {"game_id": row["game_id"], "opponent": away, "is_home": True}
        context[(w, away)] = {"game_id": row["game_id"], "opponent": home, "is_home": False}

    entries = []
    for (w, team), prob in sorted(table.items()):
        ctx = context.get((w, team))
        if ctx is None:
            # A probability with no scheduled game would mean the table and the
            # schedule disagree; skip rather than emit an entry the frontend
            # cannot explain.
            continue
        entries.append({
            "week": w,
            "team": team,
            "opponent": ctx["opponent"],
            "is_home": ctx["is_home"],
            "game_id": ctx["game_id"],
            "survival_prob": prob,
            "weeks_ahead": w - current_week,
            "source": "vegas" if w == current_week else "model",
        })

    payload = {
        "season": season,
        "current_week": current_week,
        "decay_rate": decay_rate,
        "tie_counts_as_loss": CONFIG.pool.tie_counts_as_loss,
        "weeks": remaining_weeks,
        "teams": all_teams,
        "entries": entries,
    }
    return table, payload


def build_season_plan(
    schedules: pl.DataFrame,
    drive_table: pl.DataFrame,
    season: int,
    current_week: int,
    used_teams: list[str],
    mc_config: DriveSimConfig | None = None,
    k_alternates: int = 3,
    team_game_table: pl.DataFrame | None = None,
    depth_charts: pl.DataFrame | None = None,
    future_blend_model=None,
    survival_prob: dict[tuple[int, str], float] | None = None,
) -> dict:
    """The full remaining-season recommended sequence, plus the k best
    alternate season paths — plan §4's "season plan" view.

    `survival_prob` is the unfiltered table from `build_survival_matrix`. Pass
    it when it has already been computed (as `export_all` does) to skip the
    Monte Carlo pass and the Method 3 fit; omit it and this computes its own.
    """
    season_games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    all_weeks = sorted(season_games["week"].unique().to_list())
    remaining_weeks = [w for w in all_weeks if w >= current_week]
    all_teams = sorted(set(season_games["home_team"].to_list()) | set(season_games["away_team"].to_list()))
    available_teams = [t for t in all_teams if t not in used_teams]

    if survival_prob is None:
        train = AsOf(season=season, week=current_week, game_type="REG").visible(schedules)
        survival_prob, _ = build_survival_matrix(
            schedules, drive_table, season, current_week,
            mc_config=mc_config, team_game_table=team_game_table, depth_charts=depth_charts,
            future_blend_model=future_blend_model,
            ml_model=_fit_ml_model(team_game_table, drive_table, schedules, train, depth_charts),
        )
    table = {(w, t): p for (w, t), p in survival_prob.items() if t in available_teams}

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
    team_game_table: pl.DataFrame | None = None,
    depth_charts: pl.DataFrame | None = None,
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
    # A board per remaining week, not just the current one: the frontend's
    # week selector offers every week of the season, and a missing file there
    # surfaces as an error rather than an empty state.
    #
    # Showing a future week's posted line is legitimate here in a way it is
    # NOT in the backtest. The distinction is when the number existed: a line
    # posted today for week 5 is real, currently-available market information,
    # whereas games.csv's value for a *historical* future week is that week's
    # eventual closing line, which absorbed injury news and sharp money that
    # hadn't happened at the simulated decision point. Live board, real line;
    # backtest, no line. See optimize/projection.py.
    season_weeks = sorted(
        schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))["week"]
        .unique()
        .to_list()
    )
    board_weeks = [w for w in season_weeks if w >= current_week]

    # Fit Method 3 once for the whole season (season cadence) and reuse it for
    # every week's board, rather than refitting per week.
    season_start_train = AsOf(season=season, week=current_week, game_type="REG").visible(schedules)
    shared_ml_model = _fit_ml_model(team_game_table, drive_table, schedules, season_start_train, depth_charts)

    for board_week in board_weeks:
        _write_json(
            out_dir / f"week_{board_week:02d}_board.json",
            build_week_board(
                schedules, drive_table, season, board_week, used_teams,
                team_game_table=team_game_table, depth_charts=depth_charts,
                ml_model=shared_ml_model,
            ),
        )
    # The expensive half of the season plan — one Monte Carlo pass over every
    # remaining game — is computed once here and shared by season_plan.json and
    # survival_matrix.json, reusing the Method 3 fit made above for the boards.
    # (Before this, build_season_plan fit Method 3 a second time on its own.)
    survival_prob, matrix_payload = build_survival_matrix(
        schedules, drive_table, season, current_week,
        team_game_table=team_game_table, depth_charts=depth_charts,
        ml_model=shared_ml_model,
    )
    _write_json(out_dir / "survival_matrix.json", matrix_payload)
    _write_json(
        out_dir / "season_plan.json",
        build_season_plan(
            schedules, drive_table, season, current_week, used_teams,
            team_game_table=team_game_table, depth_charts=depth_charts,
            survival_prob=survival_prob,
        ),
    )

    if comparison_predictions is not None and comparison_predictions.height > 0:
        _write_json(out_dir / "calibration.json", summarize_comparison(comparison_predictions))
    if backtest_results is not None and backtest_results.height > 0:
        _write_json(out_dir / "backtest.json", {
            "summary": summarize_backtest(backtest_results),
            "seasons": backtest_results.to_dicts(),
        })
    if strategy_summary is not None:
        _write_json(out_dir / "simulation.json", strategy_summary)
