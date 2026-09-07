"""Drive-level outcome rates, as-of, for the Monte Carlo game simulator
(plan §2, Method 2).

Builds one row per drive from play-by-play data, classifies each drive into
one of 8 outcome categories, and computes each team's offensive (what
happens when they have the ball) and defensive (what happens to opponents
against them) rate in each category — shrunk toward a prior per plan §1's
formula: `rating = w * observed + (1-w) * prior`, `w = n / (n + k)`, where
`prior = 0.6 * prior_season_final_rating + 0.4 * current_season_league_mean`.

Scope note on leakage: this module only uses REG-season play-by-play.
nflverse's pbp numbers postseason weeks with a different (non-resetting)
convention than the schedules dataset's game_type stages (clock.py), and
since survivor pools only run in the regular season, restricting to REG and
tagging every row `game_type="REG"` lets this reuse the *same* tested
`AsOf.visible()` gate as everything else, rather than writing a second,
subtly-different leakage filter for playoff weeks.

Known simplification: a relocated franchise's first season under a new
abbreviation (e.g. OAK -> LV) has no prior-season row to look up, so its
prior falls back to the pure league-average (0% weight on team history)
rather than following the old abbreviation across the rename.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from survivor.clock import AsOf

OUTCOME_CATEGORIES: tuple[str, ...] = (
    "td",
    "fg",
    "punt",
    "turnover_int",
    "turnover_fumble",
    "turnover_on_downs",
    "missed_fg",
    "safety",
)

# fixed_drive_result values that are meaningful "how did this drive end" events.
# "End of half" and null are clock artifacts (the clock ran out), not football
# decisions, and are excluded from rate denominators.
_MEANINGFUL_RESULTS = {
    "Touchdown",
    "Field goal",
    "Punt",
    "Turnover",
    "Turnover on downs",
    "Missed field goal",
    "Opp touchdown",
    "Safety",
}


def build_drive_table(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, fixed_drive): season, week, game_type='REG',
    offense (posteam), defense (defteam), outcome category, play_count,
    reached_red_zone, scored_td_in_red_zone.
    """
    reg = pbp.filter((pl.col("season_type") == "REG") & pl.col("fixed_drive").is_not_null())
    reg = reg.sort(["game_id", "fixed_drive", "play_id"])

    per_drive = reg.group_by(["game_id", "fixed_drive"]).agg(
        pl.col("season").first(),
        pl.col("week").first(),
        pl.col("posteam").first().alias("offense"),
        pl.col("defteam").first().alias("defense"),
        # fixed_drive_result is constant across every play of a drive (verified
        # against real pbp data, including trailing extra_point/kickoff rows),
        # so .first() is used here — same as season/week above — while
        # interception/fumble_lost below use .last() because those genuinely
        # are per-play flags that are only set (1.0) on the actual turnover play.
        pl.col("fixed_drive_result").first().alias("result"),
        pl.col("interception").last().alias("_interception"),
        pl.col("fumble_lost").last().alias("_fumble_lost"),
        # count only actual offensive snaps (pass/run/kneel/spike) — excludes
        # no_play (declined/pre-snap penalties), kickoff, punt, FG/PAT snaps,
        # which otherwise inflate "plays per drive" well past the commonly
        # reported ~5.5-6.5 (verified: unfiltered pl.len() gives ~8.0-8.2).
        pl.col("play_type").is_in(["pass", "run", "qb_kneel", "qb_spike"]).sum().alias("play_count"),
        (pl.col("yardline_100") <= 20).any().alias("reached_red_zone"),
    )

    per_drive = per_drive.filter(pl.col("offense").is_not_null() & pl.col("result").is_in(list(_MEANINGFUL_RESULTS)))

    category = (
        pl.when(pl.col("result") == "Touchdown")
        .then(pl.lit("td"))
        .when(pl.col("result") == "Field goal")
        .then(pl.lit("fg"))
        .when(pl.col("result") == "Punt")
        .then(pl.lit("punt"))
        .when(pl.col("result") == "Turnover on downs")
        .then(pl.lit("turnover_on_downs"))
        .when(pl.col("result") == "Missed field goal")
        .then(pl.lit("missed_fg"))
        .when(pl.col("result") == "Safety")
        .then(pl.lit("safety"))
        .when(pl.col("result").is_in(["Turnover", "Opp touchdown"]) & (pl.col("_interception") == 1))
        .then(pl.lit("turnover_int"))
        .otherwise(pl.lit("turnover_fumble"))  # fumble_lost==1, plus the small unclassified remainder
        .alias("category")
    )
    scored_td_in_rz = (pl.col("reached_red_zone") & (pl.col("result") == "Touchdown")).alias("scored_td_in_red_zone")

    out = per_drive.with_columns([category, scored_td_in_rz, pl.lit("REG").alias("game_type")])
    return out.select(
        [
            "game_id",
            "fixed_drive",
            "season",
            "week",
            "game_type",
            "offense",
            "defense",
            "category",
            "play_count",
            "reached_red_zone",
            "scored_td_in_red_zone",
        ]
    )


def _rate_distribution(drives: pl.DataFrame) -> dict[str, float]:
    """Raw (unshrunk) category rate distribution over a set of drives."""
    n = drives.height
    rates = dict.fromkeys(OUTCOME_CATEGORIES, 0.0)
    if n == 0:
        return rates
    counts = drives["category"].value_counts()
    for row in counts.iter_rows(named=True):
        if row["category"] in rates:
            rates[row["category"]] = row["count"] / n
    return rates


def _shrink_rates(observed: dict[str, float], n_observed: int, prior: dict[str, float], k: float) -> dict[str, float]:
    w = n_observed / (n_observed + k) if (n_observed + k) > 0 else 0.0
    return {cat: w * observed.get(cat, 0.0) + (1 - w) * prior.get(cat, 0.0) for cat in OUTCOME_CATEGORIES}


def _shrink_scalar(observed: float, n_observed: int, prior: float, k: float) -> float:
    w = n_observed / (n_observed + k) if (n_observed + k) > 0 else 0.0
    return w * observed + (1 - w) * prior


@dataclass(frozen=True)
class TeamDriveProfile:
    team: str
    n_drives: int  # current-season observed drive count (for diagnostics)
    rates: dict[str, float]  # shrunk category distribution, sums to ~1.0
    plays_per_drive: float
    red_zone_td_rate: float  # P(TD | reached red zone), shrunk
    drives_per_game: float  # shrunk pace metric — possessions/game (league avg ~9.7)


def _team_profiles(
    drive_table: pl.DataFrame,
    as_of: AsOf,
    team_col: str,
    shrink_k: float,
    prior_blend_prev_season: float,
    prior_blend_league_mean: float,
) -> dict[str, TeamDriveProfile]:
    visible = as_of.visible(drive_table)
    this_season = visible.filter(pl.col("season") == as_of.season)
    prior_season = visible.filter(pl.col("season") == as_of.season - 1)

    league_mean_rates = _rate_distribution(this_season) if this_season.height > 0 else _rate_distribution(visible)
    league_mean_ppd = float(this_season["play_count"].mean()) if this_season.height > 0 else (
        float(visible["play_count"].mean()) if visible.height > 0 else 5.5
    )
    rz_league = this_season.filter(pl.col("reached_red_zone"))
    league_mean_rz_td = float(rz_league["scored_td_in_red_zone"].mean()) if rz_league.height > 0 else 0.55

    def _drives_per_game_by_team(df: pl.DataFrame) -> dict[str, float]:
        if df.height == 0:
            return {}
        per_game = df.group_by(["game_id", team_col]).agg(pl.len().alias("n"))
        avg = per_game.group_by(team_col).agg(pl.col("n").mean().alias("avg"))
        return dict(zip(avg[team_col].to_list(), avg["avg"].to_list()))

    dpg_this_season = _drives_per_game_by_team(this_season)
    dpg_prior_season = _drives_per_game_by_team(prior_season)
    league_mean_dpg = float(sum(dpg_this_season.values()) / len(dpg_this_season)) if dpg_this_season else (
        float(sum(dpg_prior_season.values()) / len(dpg_prior_season)) if dpg_prior_season else 9.7
    )

    teams = sorted(set(visible[team_col].drop_nulls().unique().to_list()))
    profiles: dict[str, TeamDriveProfile] = {}

    for team in teams:
        team_this_season = this_season.filter(pl.col(team_col) == team)
        team_prior_season = prior_season.filter(pl.col(team_col) == team)

        prior_season_rates = _rate_distribution(team_prior_season) if team_prior_season.height > 0 else league_mean_rates
        team_prior = {
            cat: prior_blend_prev_season * prior_season_rates.get(cat, 0.0) + prior_blend_league_mean * league_mean_rates.get(cat, 0.0)
            for cat in OUTCOME_CATEGORIES
        }
        # re-normalize the blend weights in case they don't sum to 1 in config
        blend_total = prior_blend_prev_season + prior_blend_league_mean
        if blend_total > 0:
            team_prior = {cat: v / blend_total for cat, v in team_prior.items()}

        n_observed = team_this_season.height
        observed_rates = _rate_distribution(team_this_season)
        shrunk_rates = _shrink_rates(observed_rates, n_observed, team_prior, shrink_k)

        observed_ppd = float(team_this_season["play_count"].mean()) if n_observed > 0 else league_mean_ppd
        prior_ppd = float(team_prior_season["play_count"].mean()) if team_prior_season.height > 0 else league_mean_ppd
        team_prior_ppd = prior_blend_prev_season * prior_ppd + prior_blend_league_mean * league_mean_ppd
        if blend_total > 0:
            team_prior_ppd /= blend_total
        shrunk_ppd = _shrink_scalar(observed_ppd, n_observed, team_prior_ppd, shrink_k)

        team_rz_this = team_this_season.filter(pl.col("reached_red_zone"))
        n_rz_observed = team_rz_this.height
        observed_rz_td = float(team_rz_this["scored_td_in_red_zone"].mean()) if n_rz_observed > 0 else league_mean_rz_td
        team_rz_prior = team_prior_season.filter(pl.col("reached_red_zone"))
        prior_rz_td = float(team_rz_prior["scored_td_in_red_zone"].mean()) if team_rz_prior.height > 0 else league_mean_rz_td
        team_prior_rz = prior_blend_prev_season * prior_rz_td + prior_blend_league_mean * league_mean_rz_td
        if blend_total > 0:
            team_prior_rz /= blend_total
        shrunk_rz_td = _shrink_scalar(observed_rz_td, n_rz_observed, team_prior_rz, shrink_k)

        # drives_per_game is a per-game (not per-drive) average, so it's
        # weighted by games played rather than shrink_k (which is calibrated
        # in drive-count units) — pace stabilizes quickly, so a small fixed
        # k of 2 games is used instead.
        n_games_observed = team_this_season["game_id"].n_unique() if n_observed > 0 else 0
        observed_dpg = dpg_this_season.get(team, league_mean_dpg)
        prior_dpg = dpg_prior_season.get(team, league_mean_dpg)
        team_prior_dpg = prior_blend_prev_season * prior_dpg + prior_blend_league_mean * league_mean_dpg
        if blend_total > 0:
            team_prior_dpg /= blend_total
        shrunk_dpg = _shrink_scalar(observed_dpg, n_games_observed, team_prior_dpg, k=2.0)

        profiles[team] = TeamDriveProfile(
            team=team,
            n_drives=n_observed,
            rates=shrunk_rates,
            plays_per_drive=shrunk_ppd,
            red_zone_td_rate=shrunk_rz_td,
            drives_per_game=shrunk_dpg,
        )

    return profiles


def compute_offense_profiles(
    drive_table: pl.DataFrame, as_of: AsOf, shrink_k: float, prior_blend_prev_season: float, prior_blend_league_mean: float
) -> dict[str, TeamDriveProfile]:
    return _team_profiles(drive_table, as_of, "offense", shrink_k, prior_blend_prev_season, prior_blend_league_mean)


def compute_defense_profiles(
    drive_table: pl.DataFrame, as_of: AsOf, shrink_k: float, prior_blend_prev_season: float, prior_blend_league_mean: float
) -> dict[str, TeamDriveProfile]:
    return _team_profiles(drive_table, as_of, "defense", shrink_k, prior_blend_prev_season, prior_blend_league_mean)


def league_average_profile(drive_table: pl.DataFrame, as_of: AsOf) -> TeamDriveProfile:
    """League-wide rates as of this point — the neutral baseline the
    matchup log-odds combination (models/montecarlo.py) adjusts away from."""
    visible = as_of.visible(drive_table)
    this_season = visible.filter(pl.col("season") == as_of.season)
    base = this_season if this_season.height > 0 else visible
    rates = _rate_distribution(base)
    ppd = float(base["play_count"].mean()) if base.height > 0 else 5.5
    rz = base.filter(pl.col("reached_red_zone"))
    rz_td = float(rz["scored_td_in_red_zone"].mean()) if rz.height > 0 else 0.55
    if base.height > 0:
        per_game = base.group_by(["game_id", "offense"]).agg(pl.len().alias("n"))
        dpg = float(per_game["n"].mean())
    else:
        dpg = 9.7
    return TeamDriveProfile(team="LEAGUE", n_drives=base.height, rates=rates, plays_per_drive=ppd, red_zone_td_rate=rz_td, drives_per_game=dpg)
