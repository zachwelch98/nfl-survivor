"""Backtest harness (plan §6): replay a season week-by-week, running the
exact same pipeline weekly operation would (as-of features -> win
probabilities -> optimizer -> pick), resolving against what actually
happened, until eliminated or the season ends.

Two variants for how *future* weeks (beyond the one being decided) get
their probabilities, both run and reported side by side:

  - honest (cheating=False): Method 2's as-of simulation with horizon
    decay (optimize/projection.py) — never touches a future week's actual
    posted line, which would not have been known at decision time.
  - cheating (cheating=True): the future week's *actual* closing
    spread_line/moneyline run through a walk-forward-fit VegasModel — real
    lookahead, kept only to quantify how much that assumption is worth by
    comparison. The model's own fit is still walk-forward-legitimate; only
    the input line itself is illegally known early.

The *current* week's decision probability always comes from whichever
`method` is under test ("vegas", "montecarlo", or "blend") — that choice is
orthogonal to the honest/cheating axis above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from survivor.clock import AsOf
from survivor.features.drive_rates import build_drive_table
from survivor.models.base import pick_survival_prob
from survivor.models.blend import BlendModel
from survivor.models.montecarlo import DriveSimConfig
from survivor.models.vegas import VegasModel
from survivor.optimize.assignment import InfeasibleSeasonError, WeekPick, solve_optimal_assignment
from survivor.optimize.projection import build_survival_prob_table, simulate_all_remaining_games

REGULAR_SEASON_WEEKS = tuple(range(1, 19))


@dataclass(frozen=True)
class SeasonResult:
    season: int
    method: str  # "vegas" | "montecarlo" | "ml" | "blend"
    cheating: bool
    picks: tuple[WeekPick, ...]  # actual picks made, in order, up to elimination or season end
    week_eliminated: int | None  # None = survived every played week


def _current_week_probs(
    method: str,
    season: int,
    week: int,
    schedules: pl.DataFrame,
    drive_table: pl.DataFrame,
    target_games: pl.DataFrame,
    shrink_k: float,
    prior_blend_prev_season: float,
    prior_blend_league_mean: float,
    mc_config: DriveSimConfig,
    blend_model: BlendModel | None,
    ml_model=None,
) -> pl.DataFrame:
    """Returns target_games with home_win_prob/away_win_prob/tie_prob added,
    using only data visible before this week — the method actually being
    evaluated for the *current* week's decision."""
    as_of = AsOf(season=season, week=week, game_type="REG")
    train = as_of.visible(schedules)

    if method == "vegas":
        model = VegasModel().fit(train)
        return model.predict_proba(target_games)

    if method == "ml":
        # Arrives pre-fit from the caller, exactly like blend_model: Method 3
        # trains across many prior seasons, while this function's own
        # drive_table is deliberately scoped to one season for Method 2.
        if ml_model is None:
            raise ValueError("ml method requires a fitted ml_model")
        return ml_model.predict_proba(target_games)

    mc_results = simulate_all_remaining_games(
        drive_table, schedules, as_of, target_games.select(["game_id", "home_team", "away_team"]),
        shrink_k, prior_blend_prev_season, prior_blend_league_mean, mc_config,
    )
    if method == "montecarlo":
        rows = []
        for row in target_games.iter_rows(named=True):
            sim = mc_results.get(row["game_id"])
            rows.append({**row, "home_win_prob": sim.home_win_prob if sim else 0.5, "away_win_prob": sim.away_win_prob if sim else 0.5, "tie_prob": sim.tie_prob if sim else 0.0})
        return pl.DataFrame(rows)

    if method == "blend":
        if blend_model is None:
            raise ValueError("blend method requires a fitted blend_model")
        vegas_model = VegasModel().fit(train)
        vegas_preds = vegas_model.predict_proba(target_games)
        rows = []
        for row, vrow in zip(target_games.iter_rows(named=True), vegas_preds.iter_rows(named=True)):
            sim = mc_results.get(row["game_id"])
            vegas_share = vrow["home_win_prob"] / (vrow["home_win_prob"] + vrow["away_win_prob"])
            mc_share = sim.home_win_prob / (sim.home_win_prob + sim.away_win_prob) if sim else 0.5
            blended_share = float(
                blend_model.predict({"vegas": np.array([vegas_share]), "mc": np.array([mc_share])})[0]
            )
            tie_prob = sim.tie_prob if sim else vrow["tie_prob"]
            rows.append({**row, "home_win_prob": blended_share * (1 - tie_prob), "away_win_prob": (1 - blended_share) * (1 - tie_prob), "tie_prob": tie_prob})
        return pl.DataFrame(rows)

    raise ValueError(f"unknown method {method!r}")


def _cheating_future_probs(schedules: pl.DataFrame, season: int, decision_week: int, future_games: pl.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Real-lookahead variant: fit Vegas walk-forward, but feed it the
    future games' *actual* closing lines (real lookahead by construction —
    see module docstring). Keyed by game_id -> (home_wp, away_wp, tie_p)."""
    as_of = AsOf(season=season, week=decision_week, game_type="REG")
    train = as_of.visible(schedules)
    model = VegasModel().fit(train)
    preds = model.predict_proba(future_games)
    return {row["game_id"]: (row["home_win_prob"], row["away_win_prob"], row["tie_prob"]) for row in preds.iter_rows(named=True)}


def run_season_backtest(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    season: int,
    method: str,
    cheating: bool,
    tie_counts_as_loss: bool = True,
    shrink_k: float = 5.0,
    prior_blend_prev_season: float = 0.6,
    prior_blend_league_mean: float = 0.4,
    decay_rate: float = 0.05,
    mc_config: DriveSimConfig | None = None,
    blend_model: BlendModel | None = None,
    ml_model=None,
) -> SeasonResult:
    """Replay one season week by week, picking the optimizer's choice each
    week from among not-yet-used teams, resolving against the real result."""
    mc_config = mc_config or DriveSimConfig(n_simulations=3000)
    drive_table = build_drive_table(pbp)
    season_games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    all_teams = sorted(set(season_games["home_team"].to_list()) | set(season_games["away_team"].to_list()))

    used_teams: set[str] = set()
    picks: list[WeekPick] = []
    week_eliminated: int | None = None

    weeks_played = sorted(season_games.filter(pl.col("home_score").is_not_null())["week"].unique().to_list())

    for week in weeks_played:
        remaining_weeks = [w for w in weeks_played if w >= week]
        target_this_week = season_games.filter((pl.col("week") == week) & pl.col("home_score").is_not_null())
        if target_this_week.height == 0:
            continue

        current_probs = _current_week_probs(
            method, season, week, schedules, drive_table, target_this_week,
            shrink_k, prior_blend_prev_season, prior_blend_league_mean, mc_config, blend_model, ml_model,
        )

        future_weeks = [w for w in remaining_weeks if w != week]
        future_games_full = season_games.filter(pl.col("week").is_in(future_weeks))
        future_games = future_games_full.select(["game_id", "week", "home_team", "away_team"])

        if cheating:
            # _cheating_future_probs needs the real spread_line/moneyline columns,
            # which the narrowed `future_games` selection above doesn't carry.
            future_probs_raw = _cheating_future_probs(schedules, season, week, future_games_full)
            table: dict[tuple[int, str], float] = {}
            for r in current_probs.iter_rows(named=True):
                table[(week, r["home_team"])] = pick_survival_prob(r["home_win_prob"], r["away_win_prob"], r["tie_prob"], True, tie_counts_as_loss)
                table[(week, r["away_team"])] = pick_survival_prob(r["home_win_prob"], r["away_win_prob"], r["tie_prob"], False, tie_counts_as_loss)
            for row in future_games.iter_rows(named=True):
                probs = future_probs_raw.get(row["game_id"])
                if probs is None:
                    continue
                hwp, awp, tp = probs
                table[(row["week"], row["home_team"])] = pick_survival_prob(hwp, awp, tp, True, tie_counts_as_loss)
                table[(row["week"], row["away_team"])] = pick_survival_prob(hwp, awp, tp, False, tie_counts_as_loss)
        else:
            as_of = AsOf(season=season, week=week, game_type="REG")
            mc_results = simulate_all_remaining_games(
                drive_table, schedules, as_of, future_games.select(["game_id", "home_team", "away_team"]),
                shrink_k, prior_blend_prev_season, prior_blend_league_mean, mc_config,
            )
            table = build_survival_prob_table(current_probs, future_games, mc_results, tie_counts_as_loss, decay_rate)

        available_teams = [t for t in all_teams if t not in used_teams]
        # restrict the table to teams still available (already-used teams simply excluded from the candidate list)
        restricted_table = {(w, t): p for (w, t), p in table.items() if t in available_teams}

        try:
            plan = solve_optimal_assignment(remaining_weeks, available_teams, restricted_table)
        except InfeasibleSeasonError:
            break

        pick_team = plan.team_for_week(week)
        if pick_team is None:
            break

        actual_row = target_this_week.filter((pl.col("home_team") == pick_team) | (pl.col("away_team") == pick_team))
        if actual_row.height == 0:
            break  # picked team had a bye this week (shouldn't happen if table built correctly) -- stop rather than guess
        r = actual_row.row(0, named=True)
        is_home = r["home_team"] == pick_team
        home_score, away_score = r["home_score"], r["away_score"]
        won = (home_score > away_score) if is_home else (away_score > home_score)
        tied = home_score == away_score
        survived = won or (tied and not tie_counts_as_loss)

        actual_survival_prob = 1.0 if survived else 0.0  # realized outcome, not the pre-game estimate
        picks.append(WeekPick(week=week, team=pick_team, survival_prob=actual_survival_prob))
        used_teams.add(pick_team)

        if not survived:
            week_eliminated = week
            break

    return SeasonResult(season=season, method=method, cheating=cheating, picks=tuple(picks), week_eliminated=week_eliminated)


def summarize_backtest(results: pl.DataFrame) -> dict:
    """Package per-season backtest rows (as produced by running
    run_season_backtest across many seasons/methods/variants) into the
    plan §6 deliverable: survival distribution per method, and how the
    honest vs. cheating variants compare (quantifying what real future-line
    lookahead would be worth).
    """
    summary = (
        results.group_by(["method", "cheating"])
        .agg(
            pl.col("n_picks").median().alias("median_weeks_survived"),
            pl.col("n_picks").mean().alias("mean_weeks_survived"),
            (pl.col("n_picks") >= 10).mean().alias("p_survive_to_week10"),
            pl.col("survived_full_season").mean().alias("p_survive_full_season"),
            pl.len().alias("n_seasons"),
        )
        .sort(["method", "cheating"])
    )
    return {row["method"] + ("_cheating" if row["cheating"] else "_honest"): row for row in summary.iter_rows(named=True)}
