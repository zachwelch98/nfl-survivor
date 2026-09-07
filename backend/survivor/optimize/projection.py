"""Builds the (week, team) -> survival_prob table the optimizer consumes
(plan §4), for both real weekly operation and the backtest harness.

The current week's probability comes from whichever method is being
evaluated (Vegas's posted line, Monte Carlo's simulation, or the blend).
Every week *beyond* the current one uses Method 2's simulation
unconditionally — never the schedule's spread_line/moneyline — because
those future games' posted lines don't exist yet at decision time in real
operation, and in a historical backtest, `games.csv`'s value for a future
week is that week's actual *closing* line, which bakes in information
(injury news, sharp money, etc.) that would not have been available at the
as-of point. Method 2 has no such problem: it's a generative simulation
from current, as-of-legal team strength estimates, so it can score any
future matchup without lookahead by construction (see plan §4).

`shrink_future_prob_to_half` then widens that projection toward a coin
flip as the horizon grows, reflecting the genuine extra uncertainty about
a matchup many weeks out (injuries, form changes, unknowable events)
beyond what today's team-strength snapshot can capture.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from survivor.clock import AsOf
from survivor.features.drive_rates import compute_defense_profiles, compute_offense_profiles, league_average_profile
from survivor.models.base import pick_survival_prob
from survivor.models.montecarlo import DriveSimConfig, SimResult, combine_matchup_rates, combined_n_drives, simulate_game


def shrink_future_prob_to_half(p: float, weeks_ahead: int, decay_rate: float = 0.05) -> float:
    """weeks_ahead=0 (the current week) returns p unchanged; larger
    horizons blend p toward 0.5 with weight exp(-decay_rate * weeks_ahead).
    """
    if weeks_ahead <= 0:
        return p
    weight = float(np.exp(-decay_rate * weeks_ahead))
    return 0.5 + weight * (p - 0.5)


def simulate_all_remaining_games(
    drive_table: pl.DataFrame,
    schedules: pl.DataFrame,
    as_of: AsOf,
    remaining_games: pl.DataFrame,
    shrink_k: float,
    prior_blend_prev_season: float,
    prior_blend_league_mean: float,
    mc_config: DriveSimConfig | None = None,
) -> dict[str, SimResult]:
    """Method 2's simulated result for every game in `remaining_games`
    (columns: game_id, home_team, away_team), keyed by game_id. Profiles are
    computed once at `as_of` and reused for every game — the whole point of
    "no lookahead for the future" is that these do not change per target
    week, only the matchup being scored does.
    """
    mc_config = mc_config or DriveSimConfig()
    off = compute_offense_profiles(drive_table, as_of, shrink_k, prior_blend_prev_season, prior_blend_league_mean)
    deff = compute_defense_profiles(drive_table, as_of, shrink_k, prior_blend_prev_season, prior_blend_league_mean)
    league = league_average_profile(drive_table, as_of)

    results = {}
    for row in remaining_games.iter_rows(named=True):
        home, away, game_id = row["home_team"], row["away_team"], row["game_id"]
        if home not in off or away not in off or home not in deff or away not in deff:
            continue
        home_rates = combine_matchup_rates(off[home].rates, deff[away].rates, league.rates)
        away_rates = combine_matchup_rates(off[away].rates, deff[home].rates, league.rates)
        n_drives = combined_n_drives(off[home].drives_per_game, off[away].drives_per_game, league.drives_per_game)
        results[game_id] = simulate_game(home_rates, away_rates, n_drives, mc_config)
    return results


def build_survival_prob_table(
    current_week_probs: pl.DataFrame,
    future_games: pl.DataFrame,
    mc_results_by_game: dict[str, SimResult],
    tie_counts_as_loss: bool,
    decay_rate: float = 0.05,
) -> dict[tuple[int, str], float]:
    """Assemble the full (week, team) -> survival_prob table for the
    optimizer: current week from `current_week_probs` (must have week,
    home_team, away_team, home_win_prob, away_win_prob, tie_prob — from
    whichever method is being evaluated), every later week from Method 2's
    simulation with horizon-decay applied.

    `future_games` must have columns game_id, week, home_team, away_team
    for every not-yet-played game *strictly after* the current week.
    """
    table: dict[tuple[int, str], float] = {}

    current_week = int(current_week_probs["week"][0]) if current_week_probs.height > 0 else None
    for row in current_week_probs.iter_rows(named=True):
        w = row["week"]
        home_survival = pick_survival_prob(row["home_win_prob"], row["away_win_prob"], row["tie_prob"], picked_is_home=True, tie_counts_as_loss=tie_counts_as_loss)
        away_survival = pick_survival_prob(row["home_win_prob"], row["away_win_prob"], row["tie_prob"], picked_is_home=False, tie_counts_as_loss=tie_counts_as_loss)
        table[(w, row["home_team"])] = home_survival
        table[(w, row["away_team"])] = away_survival

    for row in future_games.iter_rows(named=True):
        game_id, w, home, away = row["game_id"], row["week"], row["home_team"], row["away_team"]
        sim = mc_results_by_game.get(game_id)
        if sim is None:
            continue
        weeks_ahead = w - current_week if current_week is not None else w
        home_survival = pick_survival_prob(sim.home_win_prob, sim.away_win_prob, sim.tie_prob, picked_is_home=True, tie_counts_as_loss=tie_counts_as_loss)
        away_survival = pick_survival_prob(sim.home_win_prob, sim.away_win_prob, sim.tie_prob, picked_is_home=False, tie_counts_as_loss=tie_counts_as_loss)
        table[(w, home)] = shrink_future_prob_to_half(home_survival, weeks_ahead, decay_rate)
        table[(w, away)] = shrink_future_prob_to_half(away_survival, weeks_ahead, decay_rate)

    return table
