"""Situational adjustments (plan §1/§2): rest, travel, weather, motivation.

Rest and weather are near-direct reads of columns nflverse already provides
(home_rest/away_rest, roof/temp/wind) — this module's job there is just to
turn raw values into adjustment-ready signals. Travel is computed from
stadiums.py against each team's empirically-determined "home base" for the
season (teams occasionally play a game at a neutral/international site, so
"home team" and "home base" are not the same thing).

Motivation (plan: "playoff-elimination/motivation status late in the
season") is the one genuinely complex piece: whether a team is clinched or
eliminated depends on the whole conference's remaining results, so this
runs a Monte Carlo simulation of the rest of the regular season using
caller-supplied win probabilities (any WinProbModel's output — this module
takes a probability table rather than importing a model, keeping the
dependency direction features -> models instead of the reverse).

Known simplification: playoff seeding here uses win percentage only, with
ties broken uniformly at random — real NFL tiebreakers (head-to-head,
division/conference record, strength of victory, ...) are not implemented.
This is a deliberate scope cut: the motivation signal only needs to roughly
separate "clearly in", "clearly out", and "still fighting" teams, not
adjudicate exact seeding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from survivor.clock import AsOf
from survivor.data.stadiums import timezone_offset_hours, travel_distance_miles

# ---------------------------------------------------------------------------
# Rest
# ---------------------------------------------------------------------------

SHORT_WEEK_MAX_REST_DAYS = 5  # Thursday games (4 days) and similar short turnarounds
BYE_WEEK_MIN_REST_DAYS = 12  # coming off a bye (typically 13-14 days rest)


@dataclass(frozen=True)
class RestFeatures:
    home_rest: int
    away_rest: int
    rest_advantage_home: int  # positive favors home team
    home_short_week: bool
    away_short_week: bool
    home_off_bye: bool
    away_off_bye: bool


def rest_features(home_rest: int | None, away_rest: int | None) -> RestFeatures | None:
    if home_rest is None or away_rest is None:
        return None
    return RestFeatures(
        home_rest=home_rest,
        away_rest=away_rest,
        rest_advantage_home=home_rest - away_rest,
        home_short_week=home_rest <= SHORT_WEEK_MAX_REST_DAYS,
        away_short_week=away_rest <= SHORT_WEEK_MAX_REST_DAYS,
        home_off_bye=home_rest >= BYE_WEEK_MIN_REST_DAYS,
        away_off_bye=away_rest >= BYE_WEEK_MIN_REST_DAYS,
    )


# ---------------------------------------------------------------------------
# Travel
# ---------------------------------------------------------------------------


def team_home_base(schedules: pl.DataFrame, team: str, season: int) -> str | None:
    """The stadium_id a team calls home in a given season — the most common
    stadium_id among that team's home games that season. Falls back to the
    most recent prior season with data (handles week 1, before any home
    games this season have been played)."""
    this_season = schedules.filter((pl.col("season") == season) & (pl.col("home_team") == team) & (pl.col("game_type") == "REG"))
    if this_season.height > 0:
        mode = this_season["stadium_id"].mode()
        if mode.len() > 0:
            return mode[0]

    prior = schedules.filter((pl.col("season") < season) & (pl.col("home_team") == team) & (pl.col("game_type") == "REG"))
    if prior.height > 0:
        latest_season = prior["season"].max()
        mode = prior.filter(pl.col("season") == latest_season)["stadium_id"].mode()
        if mode.len() > 0:
            return mode[0]
    return None


@dataclass(frozen=True)
class TravelFeatures:
    travel_miles: float
    timezone_offset_hours: float  # positive = traveling east (the harder direction for an early kickoff)
    is_neutral_site: bool  # the "home" team is also away from its own home base


def travel_features(schedules: pl.DataFrame, team: str, season: int, game_stadium_id: str, game_date, is_designated_home: bool) -> TravelFeatures | None:
    home_base = team_home_base(schedules, team, season)
    if home_base is None:
        return None
    miles = travel_distance_miles(home_base, game_stadium_id)
    tz = timezone_offset_hours(home_base, game_stadium_id, game_date)
    if miles is None or tz is None:
        return None
    return TravelFeatures(
        travel_miles=miles,
        timezone_offset_hours=tz,
        is_neutral_site=is_designated_home and home_base != game_stadium_id,
    )


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeatherFeatures:
    is_dome: bool
    cold_game: bool  # temp < 32F, outdoors
    high_wind: bool  # wind >= 15mph — meaningfully suppresses passing/kicking
    pass_efficiency_multiplier: float  # applied to passing EPA/success in montecarlo.py
    fg_success_multiplier: float


def weather_features(roof: str | None, temp: int | None, wind: int | None) -> WeatherFeatures:
    is_dome = roof in ("dome", "closed")
    if is_dome:
        return WeatherFeatures(is_dome=True, cold_game=False, high_wind=False, pass_efficiency_multiplier=1.0, fg_success_multiplier=1.0)

    cold = temp is not None and temp < 32
    windy = wind is not None and wind >= 15

    pass_mult = 1.0
    fg_mult = 1.0
    if windy:
        pass_mult *= 0.93
        fg_mult *= 0.90
    if cold:
        pass_mult *= 0.97
        fg_mult *= 0.97

    return WeatherFeatures(is_dome=False, cold_game=cold, high_wind=windy, pass_efficiency_multiplier=pass_mult, fg_success_multiplier=fg_mult)


# ---------------------------------------------------------------------------
# Motivation (weeks 15-18 only)
# ---------------------------------------------------------------------------

MOTIVATION_START_WEEK = 15
_PLAYOFF_TEAMS_PER_CONFERENCE = 7
_DIVISIONS_PER_CONFERENCE = 4


@dataclass(frozen=True)
class TeamRecord:
    team: str
    wins: int
    losses: int
    ties: int
    games_played: int

    @property
    def win_pct(self) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.games_played


def standings_as_of(schedules: pl.DataFrame, as_of: AsOf) -> dict[str, TeamRecord]:
    """Actual W-L-T records from REG games completed strictly before as_of."""
    played = as_of.visible(schedules).filter(
        (pl.col("game_type") == "REG") & (pl.col("season") == as_of.season) & pl.col("home_score").is_not_null()
    )
    records: dict[str, dict[str, int]] = {}

    def _bump(team: str, w: int, l: int, t: int) -> None:
        r = records.setdefault(team, {"w": 0, "l": 0, "t": 0})
        r["w"] += w
        r["l"] += l
        r["t"] += t

    for row in played.iter_rows(named=True):
        home, away = row["home_team"], row["away_team"]
        if row["home_score"] > row["away_score"]:
            _bump(home, 1, 0, 0)
            _bump(away, 0, 1, 0)
        elif row["home_score"] < row["away_score"]:
            _bump(home, 0, 1, 0)
            _bump(away, 1, 0, 0)
        else:
            _bump(home, 0, 0, 1)
            _bump(away, 0, 0, 1)

    return {
        team: TeamRecord(team=team, wins=r["w"], losses=r["l"], ties=r["t"], games_played=r["w"] + r["l"] + r["t"])
        for team, r in records.items()
    }


def _rank_playoff_teams(final_win_pct: dict[str, float], team_division: dict[str, str], team_conf: dict[str, str], rng: np.random.Generator) -> set[str]:
    """Simplified seeding: division winner = best win% in division (ties
    broken randomly); remaining conference spots by win% (ties broken
    randomly) up to _PLAYOFF_TEAMS_PER_CONFERENCE total per conference."""
    made_it: set[str] = set()
    conferences = set(team_conf.values())
    for conf in conferences:
        conf_teams = [t for t, c in team_conf.items() if c == conf]
        divisions = {team_division[t] for t in conf_teams}
        division_winners = []
        for div in divisions:
            div_teams = [t for t in conf_teams if team_division[t] == div]
            # random tiny jitter breaks exact ties uniformly at random without bias
            best = max(div_teams, key=lambda t: (final_win_pct[t], rng.random()))
            division_winners.append(best)
        made_it.update(division_winners)

        remaining = [t for t in conf_teams if t not in division_winners]
        remaining_ranked = sorted(remaining, key=lambda t: (final_win_pct[t], rng.random()), reverse=True)
        n_wildcards = _PLAYOFF_TEAMS_PER_CONFERENCE - len(division_winners)
        made_it.update(remaining_ranked[:n_wildcards])

    return made_it


def simulate_playoff_probabilities(
    schedules: pl.DataFrame,
    as_of: AsOf,
    remaining_game_probs: pl.DataFrame,
    team_division: dict[str, str],
    team_conf: dict[str, str],
    n_sims: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """P(make playoffs) per team via Monte Carlo over the rest of the
    regular season.

    `remaining_game_probs` must have one row per not-yet-played REG game in
    as_of.season with columns home_team, away_team, home_win_prob,
    away_win_prob, tie_prob — typically this season's remaining schedule
    scored by a WinProbModel using only as-of-visible data (no leakage: the
    probabilities are predictions, never the actual future results).
    """
    current = standings_as_of(schedules, as_of)
    all_teams = set(team_division.keys())
    base_wins = {t: current[t].wins if t in current else 0 for t in all_teams}
    base_losses = {t: current[t].losses if t in current else 0 for t in all_teams}
    base_ties = {t: current[t].ties if t in current else 0 for t in all_teams}
    base_games = {t: current[t].games_played if t in current else 0 for t in all_teams}

    rng = np.random.default_rng(seed)
    teams = sorted(all_teams)
    team_pos = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    sim_wins = np.tile(np.array([base_wins[t] for t in teams], dtype=float), (n_sims, 1))
    sim_ties = np.tile(np.array([base_ties[t] for t in teams], dtype=float), (n_sims, 1))
    sim_games = np.tile(np.array([base_games[t] for t in teams], dtype=float), (n_sims, 1))

    for row in remaining_game_probs.iter_rows(named=True):
        home, away = row["home_team"], row["away_team"]
        p = np.array([row["home_win_prob"], row["away_win_prob"], row["tie_prob"]])
        p = p / p.sum()
        outcomes = rng.choice(3, size=n_sims, p=p)  # 0=home win, 1=away win, 2=tie
        hi, ai = team_pos[home], team_pos[away]
        sim_wins[outcomes == 0, hi] += 1
        sim_wins[outcomes == 1, ai] += 1
        sim_ties[outcomes == 2, hi] += 1
        sim_ties[outcomes == 2, ai] += 1
        sim_games[:, hi] += 1
        sim_games[:, ai] += 1

    made_playoffs_count = dict.fromkeys(teams, 0)
    for s in range(n_sims):
        final_win_pct = {t: (sim_wins[s, team_pos[t]] + 0.5 * sim_ties[s, team_pos[t]]) / max(sim_games[s, team_pos[t]], 1) for t in teams}
        made_it = _rank_playoff_teams(final_win_pct, team_division, team_conf, rng)
        for t in made_it:
            made_playoffs_count[t] += 1

    return {t: made_playoffs_count[t] / n_sims for t in teams}


def motivation_status(playoff_probs: dict[str, float], as_of: AsOf, clinched_threshold: float = 0.97, eliminated_threshold: float = 0.03) -> dict[str, str]:
    """"clinched" / "eliminated" / "fighting" per team — only meaningful for
    week >= MOTIVATION_START_WEEK (plan: "late in the season"); returns all
    teams as "fighting" before that, since the signal isn't applicable yet.
    """
    if as_of.week < MOTIVATION_START_WEEK or as_of.game_type != "REG":
        return {t: "fighting" for t in playoff_probs}
    status = {}
    for team, p in playoff_probs.items():
        if p >= clinched_threshold:
            status[team] = "clinched"
        elif p <= eliminated_threshold:
            status[team] = "eliminated"
        else:
            status[team] = "fighting"
    return status


def motivation_efficiency_haircut(status: str, magnitude: float = 0.03) -> float:
    """A small efficiency multiplier haircut applied to a team's offense/
    defense profile when they have nothing left to play for (clinched
    teams resting starters; eliminated teams playing out the string).
    Returned as a multiplicative factor <= 1.0; "fighting" teams get 1.0
    (no adjustment)."""
    if status in ("clinched", "eliminated"):
        return 1.0 - magnitude
    return 1.0
