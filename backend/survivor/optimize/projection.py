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

## Method 3 in future weeks

Method 3 (models/ml.py) is as-of-legal for future weeks for exactly the
same reason Method 2 is: it is fit on strictly-prior seasons and predicts
from as-of features, never from the future game's own line or result. And
it is measurably better at the job. Walk-forward on 2022-2025 (n=1084,
identical games):

    vegas  Brier 0.2101   (unusable for future weeks -- no line exists yet)
    ml     Brier 0.2234   high-confidence band: predicted 90.9%, actual 87.8%
    mc     Brier 0.2465   high-confidence band: predicted 90.7%, actual 77.1%

ml beats mc by 0.0231 with a bootstrap CI excluding zero, and is far less
overconfident in the 0.85-0.99 band that survivor picks live in.

So future weeks now use a *fitted* blend of the two rather than Monte
Carlo alone. Fitted, not a fixed average: a 50/50 logit blend scores
0.2304, significantly *worse* than ml alone, because it re-admits the
weaker model at a weight it hasn't earned. The walk-forward fitted blend
scores 0.2242 -- statistically indistinguishable from ml alone, with the
best high-confidence calibration measured (predicted 91.4%, actual 90.2%).
That is what "complement rather than replace" has to mean once there are
numbers: the weights are learned, so mc is down-rated where it is bad and
the blend adapts if ml ever degrades.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from survivor.clock import AsOf
from survivor.features.drive_rates import compute_defense_profiles, compute_offense_profiles, league_average_profile
from survivor.models._prob_utils import EPS as _EPS
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


def predict_all_remaining_games_ml(
    ml_model,
    remaining_games: pl.DataFrame,
) -> dict[str, tuple[float, float, float]]:
    """Method 3's (home_win_prob, away_win_prob, tie_prob) per game_id.

    Mirrors `simulate_all_remaining_games`'s shape for Method 2. `ml_model`
    must already be fit by the caller — the season-cadence pattern, same as
    `blend_model` in backtest/harness.py.
    """
    if ml_model is None or remaining_games.height == 0:
        return {}
    preds = ml_model.predict_proba(remaining_games)
    return {
        row["game_id"]: (row["home_win_prob"], row["away_win_prob"], row["tie_prob"])
        for row in preds.iter_rows(named=True)
    }


def _combine_future_probs(
    mc: SimResult | None,
    ml: tuple[float, float, float] | None,
    blend_model=None,
) -> tuple[float, float, float] | None:
    """Resolve one future game's probabilities from whichever methods are
    available, preferring the option with the best measured evidence:

      1. both + a fitted blend  -> the blend (best high-confidence calibration)
      2. both, no blend model   -> ml alone (a fixed average measured *worse*
                                   than ml alone, so it is deliberately not
                                   the fallback)
      3. one of them            -> that one
      4. neither                -> None, and the caller skips the game

    See the module docstring for the numbers behind that ordering.
    """
    if mc is None and ml is None:
        return None
    if ml is None:
        return (mc.home_win_prob, mc.away_win_prob, mc.tie_prob)
    if mc is None:
        return ml

    ml_home, ml_away, ml_tie = ml
    tie_prob = (mc.tie_prob + ml_tie) / 2.0
    if blend_model is None:
        return ml

    mc_share = mc.home_win_prob / max(mc.home_win_prob + mc.away_win_prob, _EPS)
    ml_share = ml_home / max(ml_home + ml_away, _EPS)
    try:
        blended = float(
            blend_model.predict({"mc": np.array([mc_share]), "ml": np.array([ml_share])})[0]
        )
    except (ValueError, RuntimeError):
        # Blend fit on a different input set, or never fit -- fall back to the
        # better single model rather than failing the whole season plan.
        return ml
    win_mass = 1.0 - tie_prob
    return (blended * win_mass, (1.0 - blended) * win_mass, tie_prob)


def build_survival_prob_table(
    current_week_probs: pl.DataFrame,
    future_games: pl.DataFrame,
    mc_results_by_game: dict[str, SimResult],
    tie_counts_as_loss: bool,
    decay_rate: float = 0.05,
    ml_results_by_game: dict[str, tuple[float, float, float]] | None = None,
    future_blend_model=None,
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

    ml_results_by_game = ml_results_by_game or {}
    for row in future_games.iter_rows(named=True):
        game_id, w, home, away = row["game_id"], row["week"], row["home_team"], row["away_team"]
        combined = _combine_future_probs(
            mc_results_by_game.get(game_id), ml_results_by_game.get(game_id), future_blend_model
        )
        if combined is None:
            continue
        home_prob, away_prob, tie_prob = combined
        weeks_ahead = w - current_week if current_week is not None else w
        home_survival = pick_survival_prob(home_prob, away_prob, tie_prob, picked_is_home=True, tie_counts_as_loss=tie_counts_as_loss)
        away_survival = pick_survival_prob(home_prob, away_prob, tie_prob, picked_is_home=False, tie_counts_as_loss=tie_counts_as_loss)
        table[(w, home)] = shrink_future_prob_to_half(home_survival, weeks_ahead, decay_rate)
        table[(w, away)] = shrink_future_prob_to_half(away_survival, weeks_ahead, decay_rate)

    return table
