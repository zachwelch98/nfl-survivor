"""As-of team efficiency ratings (plan §1), opponent-adjusted via ridge
regression.

Two layers per as-of point, both built from real offensive snaps
(play_type in pass/run) with a non-null EPA:

  1. Naive shrinkage: the same `rating = w*observed + (1-w)*prior` formula
     used throughout this project (see drive_rates.py), applied to EPA/play
     (overall, and split pass/rush), success rate, and yards/play.
  2. Opponent-adjusted power rating: a ridge regression solved once per
     as-of point across every team-game observation,

        observed_epa_per_play = off_rating[offense] - def_rating[defense]
                                 + hfa * is_home + error

     for all teams' off/def ratings simultaneously — the standard
     schedule-adjustment approach (the same idea underlying Massey/SRS
     point-margin ratings, applied here to a per-play efficiency stat
     instead of game margin). This is what lets "beat a bad defense" and
     "beat a good defense" count differently, which naive shrinkage cannot
     do on its own.

Consumers: situational adjustments (features/situational.py) and the
optimizer's future-week spread projection beyond the posted-line horizon
(plan §4) — Method 2's own matchup simulation (models/montecarlo.py) uses
drive_rates.py's category-rate profiles directly and does not depend on
this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from survivor.clock import AsOf

_OFFENSIVE_SNAP = (pl.col("season_type") == "REG") & pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null()


def build_team_game_table(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team): season, week, game_type='REG', team,
    opponent, is_home, epa_per_play, pass_epa_per_play, rush_epa_per_play,
    success_rate, yards_per_play, n_plays.
    """
    snaps = pbp.filter(_OFFENSIVE_SNAP)

    per_team_game = snaps.group_by(["game_id", "posteam"]).agg(
        pl.col("season").first(),
        pl.col("week").first(),
        pl.col("defteam").first().alias("opponent"),
        (pl.col("posteam_type") == "home").first().alias("is_home"),
        pl.col("epa").mean().alias("epa_per_play"),
        pl.col("epa").filter(pl.col("play_type") == "pass").mean().alias("pass_epa_per_play"),
        pl.col("epa").filter(pl.col("play_type") == "run").mean().alias("rush_epa_per_play"),
        pl.col("success").mean().alias("success_rate"),
        pl.col("yards_gained").mean().alias("yards_per_play"),
        pl.len().alias("n_plays"),
    )
    per_team_game = per_team_game.rename({"posteam": "team"}).with_columns(pl.lit("REG").alias("game_type"))
    return per_team_game.select(
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "team",
            "opponent",
            "is_home",
            "epa_per_play",
            "pass_epa_per_play",
            "rush_epa_per_play",
            "success_rate",
            "yards_per_play",
            "n_plays",
        ]
    )


def _shrink_scalar(observed: float, n_observed: int, prior: float, k: float) -> float:
    w = n_observed / (n_observed + k) if (n_observed + k) > 0 else 0.0
    return w * observed + (1 - w) * prior


@dataclass(frozen=True)
class TeamRating:
    team: str
    n_games: int
    epa_per_play: float  # shrunk, naive (not opponent-adjusted)
    pass_epa_per_play: float
    rush_epa_per_play: float
    success_rate: float
    yards_per_play: float


def compute_naive_ratings(
    team_game_table: pl.DataFrame,
    as_of: AsOf,
    shrink_k: float,
    prior_blend_prev_season: float,
    prior_blend_league_mean: float,
) -> dict[str, TeamRating]:
    """Shrinkage-only ratings, no opponent adjustment — the "v1" per plan §1."""
    visible = as_of.visible(team_game_table)
    this_season = visible.filter(pl.col("season") == as_of.season)
    prior_season = visible.filter(pl.col("season") == as_of.season - 1)

    metric_cols = ["epa_per_play", "pass_epa_per_play", "rush_epa_per_play", "success_rate", "yards_per_play"]

    def _means(df: pl.DataFrame) -> dict[str, float]:
        if df.height == 0:
            return dict.fromkeys(metric_cols, 0.0)
        return {c: float(df[c].mean()) if df[c].mean() is not None else 0.0 for c in metric_cols}

    league_mean = _means(this_season) if this_season.height > 0 else _means(visible)

    blend_total = prior_blend_prev_season + prior_blend_league_mean
    teams = sorted(set(visible["team"].drop_nulls().unique().to_list()))
    ratings: dict[str, TeamRating] = {}

    for team in teams:
        team_this = this_season.filter(pl.col("team") == team)
        team_prior = prior_season.filter(pl.col("team") == team)
        prior_means = _means(team_prior) if team_prior.height > 0 else league_mean

        team_prior_blend = {
            c: (prior_blend_prev_season * prior_means[c] + prior_blend_league_mean * league_mean[c]) / blend_total
            if blend_total > 0
            else league_mean[c]
            for c in metric_cols
        }

        n_observed = team_this.height
        observed_means = _means(team_this)
        shrunk = {c: _shrink_scalar(observed_means[c], n_observed, team_prior_blend[c], shrink_k) for c in metric_cols}

        ratings[team] = TeamRating(
            team=team,
            n_games=n_observed,
            epa_per_play=shrunk["epa_per_play"],
            pass_epa_per_play=shrunk["pass_epa_per_play"],
            rush_epa_per_play=shrunk["rush_epa_per_play"],
            success_rate=shrunk["success_rate"],
            yards_per_play=shrunk["yards_per_play"],
        )

    return ratings


@dataclass(frozen=True)
class OpponentAdjustedRating:
    team: str
    off_rating: float  # EPA/play produced, above (positive)/below (negative) league average, adjusted for defenses faced
    def_rating: float  # EPA/play allowed, above (positive = worse D)/below (negative = better D) league average
    hfa: float  # shared across all teams — the league home-field advantage estimate

    def projected_epa(self, opponent_def_rating: float, is_home: bool, hfa_value: float) -> float:
        """Expected EPA/play for this team's offense against a given opponent
        defense: off_rating + opponent's def_rating (+ hfa if this team is
        home). Both ratings share the same sign convention (positive =
        favors the offense), so combining them is a plain sum."""
        return self.off_rating + opponent_def_rating + (hfa_value if is_home else 0.0)


def compute_opponent_adjusted_ratings(
    team_game_table: pl.DataFrame,
    as_of: AsOf,
    alpha: float = 40.0,
) -> dict[str, OpponentAdjustedRating]:
    """Ridge regression: observed_epa[team,opp,game] = off[team] - def[opp] + hfa*is_home + error.

    `alpha` is the ridge L2 penalty in the same units as (EPA/play)^2 per
    coefficient; it exists mainly to keep the system well-posed (raw
    off_i - def_j is only identified up to an additive constant without
    regularization, exactly as in Massey/SRS-style ratings) and to prevent
    teams with very few games this season from getting an extreme
    "unregularized" fit. A team that has never appeared as of this point
    (no visible games yet) is not included in the output.
    """
    visible = as_of.visible(team_game_table)
    if visible.height == 0:
        return {}

    teams = sorted(set(visible["team"].to_list()) | set(visible["opponent"].to_list()))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    n_obs = visible.height
    # columns: [0:n_teams) = offense dummies, [n_teams:2*n_teams) = defense dummies (negated), last col = hfa
    x = np.zeros((n_obs, 2 * n_teams + 1))
    y = np.zeros(n_obs)

    team_col = visible["team"].to_list()
    opp_col = visible["opponent"].to_list()
    is_home_col = visible["is_home"].to_list()
    epa_col = visible["epa_per_play"].to_list()

    for i in range(n_obs):
        x[i, team_idx[team_col[i]]] = 1.0
        x[i, n_teams + team_idx[opp_col[i]]] = -1.0
        x[i, -1] = 1.0 if is_home_col[i] else 0.0
        y[i] = epa_col[i]

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x, y)

    off_coefs = model.coef_[:n_teams]
    # The ridge's raw internal def_coefs[i] entered the fitted equation as
    # `y = off[T] - def_coefs[O] + hfa*is_home` (see the -1 in x[i, n_teams+idx]
    # above), so a *good* defense — one that suppresses opposing offenses —
    # gets a large *positive* raw def_coefs[i]. That is the opposite of the
    # public convention for "defensive EPA/play allowed" (negative = good
    # defense, matching the sign of offensive EPA where positive = good
    # offense), so the sign is flipped here before returning: a consumer can
    # then compute a projected matchup EPA as a plain sum,
    # off_rating[offense] + def_rating[defense] (+ hfa if home) — see
    # OpponentAdjustedRating.projected_epa().
    def_coefs_raw = model.coef_[n_teams : 2 * n_teams]
    hfa = float(model.coef_[-1])

    ratings: dict[str, OpponentAdjustedRating] = {}
    for team in teams:
        i = team_idx[team]
        ratings[team] = OpponentAdjustedRating(team=team, off_rating=float(off_coefs[i]), def_rating=float(-def_coefs_raw[i]), hfa=hfa)

    return ratings
