"""Season-long strategy comparison via Monte Carlo (plan §5).

Given one real season's *evolving* as-of information (a genuinely different
probability snapshot each week, reflecting real new information becoming
available as the season unfolds — the same walk-forward machinery as
backtest/harness.py), this samples many alternate realizations of how the
games could have bounced and compares decision strategies against the same
underlying uncertainty:

  - greedy: each week, the single highest win-probability available team,
    ignoring the rest of the season entirely.
  - static: the full-season optimal assignment solved once, at week 1,
    using week 1's (necessarily more uncertain, more heavily decayed)
    projections for every future week — then followed rigidly regardless
    of what actually happens.
  - adaptive: re-solved every week using that week's own as-of table (this
    project's actual weekly-operation design) — the comparison against
    static is exactly what quantifies the value of re-optimizing.
  - save_best: a simple heuristic that will not spend a team ranked in the
    league's top `reserve_count` (by the static plan's implied quality)
    before `reserve_until_week`, picking greedy among the rest until then.

Probabilities are fixed per (decision week, target week) — sampling varies
only the *realized outcome*, not the team-strength estimates themselves;
re-deriving team strength from simulated (fictional) results would be a
different, much larger undertaking than what this comparison calls for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from survivor.backtest.harness import _current_week_probs
from survivor.clock import AsOf
from survivor.features.drive_rates import build_drive_table
from survivor.models.base import pick_survival_prob
from survivor.models.blend import BlendModel
from survivor.models.montecarlo import DriveSimConfig
from survivor.optimize.assignment import InfeasibleSeasonError, solve_optimal_assignment
from survivor.optimize.projection import build_survival_prob_table, simulate_all_remaining_games

STRATEGIES = ("greedy", "static", "adaptive", "save_best")


def precompute_weekly_survival_tables(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    season: int,
    method: str = "vegas",
    tie_counts_as_loss: bool = True,
    shrink_k: float = 5.0,
    prior_blend_prev_season: float = 0.6,
    prior_blend_league_mean: float = 0.4,
    decay_rate: float = 0.05,
    mc_config: DriveSimConfig | None = None,
    blend_model: BlendModel | None = None,
    ml_model=None,
) -> dict[int, dict[tuple[int, str], float]]:
    """weekly_tables[W] is the full (target_week, team) -> survival_prob
    table as known at decision week W — current week from `method`, every
    later week from Method 2 + horizon decay. Computed once per real
    decision week (18 calls), independent of how many simulations are run
    on top of it afterward.
    """
    mc_config = mc_config or DriveSimConfig(n_simulations=3000)
    drive_table = build_drive_table(pbp)
    season_games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    weeks_played = sorted(season_games.filter(pl.col("home_score").is_not_null())["week"].unique().to_list())

    tables = {}
    for week in weeks_played:
        target_this_week = season_games.filter(pl.col("week") == week)
        current_probs = _current_week_probs(
            method, season, week, schedules, drive_table, target_this_week,
            shrink_k, prior_blend_prev_season, prior_blend_league_mean, mc_config, blend_model, ml_model,
        )
        future_weeks = [w for w in weeks_played if w > week]
        future_games = season_games.filter(pl.col("week").is_in(future_weeks)).select(["game_id", "week", "home_team", "away_team"])

        as_of = AsOf(season=season, week=week, game_type="REG")
        mc_results = simulate_all_remaining_games(
            drive_table, schedules, as_of, future_games, shrink_k, prior_blend_prev_season, prior_blend_league_mean, mc_config,
        )
        table = build_survival_prob_table(current_probs, future_games, mc_results, tie_counts_as_loss, decay_rate)
        tables[week] = table

    return tables


@dataclass(frozen=True)
class StrategyOutcome:
    strategy: str
    week_eliminated: np.ndarray  # per simulation; np.nan = survived every played week


def _static_plan(weekly_tables: dict[int, dict[tuple[int, str], float]], weeks: list[int], teams: list[str]) -> dict[int, str]:
    week1 = weeks[0]
    table = {k: v for k, v in weekly_tables[week1].items() if k[0] in weeks}
    try:
        plan = solve_optimal_assignment(weeks, teams, table)
    except InfeasibleSeasonError:
        return {}
    return {p.week: p.team for p in plan.picks}


def _team_quality_rank(weekly_tables: dict[int, dict[tuple[int, str], float]], weeks: list[int], teams: list[str]) -> list[str]:
    """Teams ranked best-to-worst by their best available survival_prob
    across the whole season, as a simple proxy for "how good is this team"
    — used by the save_best heuristic to decide which teams are worth
    holding back."""
    best_per_team = {t: 0.0 for t in teams}
    for w in weeks:
        for (tw, t), p in weekly_tables[w].items():
            if t in best_per_team:
                best_per_team[t] = max(best_per_team[t], p)
    return sorted(teams, key=lambda t: best_per_team[t], reverse=True)


def simulate_strategies(
    weekly_tables: dict[int, dict[tuple[int, str], float]],
    weeks: list[int],
    teams: list[str],
    n_sims: int = 5000,
    seed: int = 0,
    reserve_count: int = 6,
    reserve_until_week: int = 8,
    strategies: tuple[str, ...] = STRATEGIES,
) -> dict[str, StrategyOutcome]:
    """Run n_sims simulated seasons per strategy. Each simulation samples,
    for the currently-picked team in each week, whether it survives that
    week from weekly_tables[week][(week, team)] — i.e. always that week's
    *own* (least-decayed, most current) estimate for the realized outcome,
    regardless of which table informed the pick itself.
    """
    rng = np.random.default_rng(seed)
    static_plan = _static_plan(weekly_tables, weeks, teams) if "static" in strategies else {}
    quality_rank = _team_quality_rank(weekly_tables, weeks, teams) if "save_best" in strategies else []
    reserved_teams = set(quality_rank[:reserve_count])

    results = {s: np.full(n_sims, np.nan) for s in strategies}

    def _greedy_fallback(available: list[str], current_week_prob) -> str | None:
        """Used whenever a strategy's primary logic can't produce a valid
        pick (e.g. the assignment is infeasible because every remaining
        option is a certain loss) — a real picker still has to choose
        *something* that week rather than the simulation silently treating
        "no valid plan" as survival."""
        candidates = [(t, current_week_prob(t)) for t in available if current_week_prob(t) is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda tp: tp[1])[0]

    for sim in range(n_sims):
        for strategy in strategies:
            used: set[str] = set()
            eliminated_week = None
            for week in weeks:
                available = [t for t in teams if t not in used]
                if not available:
                    break
                current_week_prob = lambda t: weekly_tables[week].get((week, t))  # noqa: E731

                if strategy == "greedy":
                    pick = _greedy_fallback(available, current_week_prob)
                elif strategy == "static":
                    pick = static_plan.get(week)
                    if pick is None or pick in used:
                        pick = _greedy_fallback(available, current_week_prob)
                elif strategy == "adaptive":
                    remaining_weeks = [w for w in weeks if w >= week]
                    table = {k: v for k, v in weekly_tables[week].items() if k[0] in remaining_weeks and k[1] in available}
                    try:
                        plan = solve_optimal_assignment(remaining_weeks, available, table)
                        pick = plan.team_for_week(week)
                    except InfeasibleSeasonError:
                        pick = None
                    if pick is None:
                        pick = _greedy_fallback(available, current_week_prob)
                elif strategy == "save_best":
                    eligible = [t for t in available if t not in reserved_teams or week >= reserve_until_week]
                    pick = _greedy_fallback(eligible, current_week_prob) or _greedy_fallback(available, current_week_prob)
                else:
                    raise ValueError(f"unknown strategy {strategy!r}")

                p = current_week_prob(pick)
                if p is None:
                    break
                used.add(pick)
                survives = rng.random() < p
                if not survives:
                    eliminated_week = week
                    break

            results[strategy][sim] = eliminated_week if eliminated_week is not None else np.nan

    return {s: StrategyOutcome(strategy=s, week_eliminated=results[s]) for s in strategies}


def summarize_strategy_outcomes(outcomes: dict[str, StrategyOutcome], weeks: list[int]) -> dict:
    """Median/mean week eliminated, P(survive to week N) curve, and
    P(survive the whole run) per strategy — plan §5's deliverable."""
    summary = {}
    for name, outcome in outcomes.items():
        elim = outcome.week_eliminated
        survived_mask = np.isnan(elim)
        weeks_survived = np.where(survived_mask, len(weeks), elim - weeks[0])
        summary[name] = {
            "median_weeks_survived": float(np.median(weeks_survived)),
            "mean_weeks_survived": float(np.mean(weeks_survived)),
            "p_survive_full_run": float(np.mean(survived_mask)),
            "survive_to_week_curve": {
                int(w): float(np.mean(survived_mask | (elim > w))) for w in weeks
            },
        }
    return summary
