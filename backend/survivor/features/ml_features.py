"""Per-matchup feature assembly for Method 3, the supervised ML model
(models/ml.py).

This module owns exactly one job: turn "a game, and an as-of point" into a
flat row of numbers. It computes no ratings of its own — every input comes
from a module that already exists and already enforces the leakage gate:

  features/team_ratings.py   naive shrunk + opponent-adjusted ridge EPA
  features/drive_rates.py    per-drive outcome category profiles
  features/situational.py    rest, travel, weather, standings
  models/montecarlo.py       log5 matchup combination of those profiles

What is genuinely new here is the *assembly*, plus three feature families
the existing modules don't cover: rolling recent form, trajectory (recent
minus season-long), and starting-QB continuity.

## Leakage

Every historical input is read through `AsOf.visible()` — never a
hand-rolled season/week filter. Three categories of data are touched, and
they are not equally dangerous:

  1. **Prior results** (team_game_table, drive_table, completed scores) —
     strictly gated by `AsOf.visible()`. This is the dangerous category and
     it is the one the gate exists for.
  2. **The target game's own pre-game facts** (rest days, roof, temp, wind,
     stadium, kickoff weekday, and the starting QB when known) — used
     directly. These are known before kickoff, exactly like `spread_line`
     is for models/vegas.py. Using a game's own pre-game metadata to
     predict that game is not lookahead.
  3. **The published schedule** (who plays whom, where, in which week) —
     used un-gated, deliberately. The full-season schedule is released in
     May, months before week 1, so "which stadium is this team's home base
     this season" (situational.team_home_base) is genuinely known in
     advance. Only *results* attached to those rows are gated.

`tests/test_ml_features.py::test_no_leakage_from_future_results` pins this
down: null out every future score and truncate the results-derived tables,
and every early-week feature row must come out bit-for-bit identical.

## Cost

`compute_opponent_adjusted_ratings` runs a league-wide ridge fit per call,
and `compute_offense_profiles`/`compute_defense_profiles` scan the whole
visible drive table. A training set of ~5,000 games spans only ~270
distinct (season, week) as-of points, so `build_feature_table` groups by
as-of point and computes each group's team dictionaries exactly once,
rather than once per game. Same reason `collect_predictions` groups its
per-week work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from survivor.clock import AsOf
from survivor.features.drive_rates import (
    OUTCOME_CATEGORIES,
    TeamDriveProfile,
    compute_defense_profiles,
    compute_offense_profiles,
    league_average_profile,
)
from survivor.features.situational import (
    RestFeatures,
    TravelFeatures,
    WeatherFeatures,
    rest_features,
    standings_as_of,
    travel_features,
    weather_features,
)
from survivor.features.team_ratings import (
    OpponentAdjustedRating,
    TeamRating,
    compute_naive_ratings,
    compute_opponent_adjusted_ratings,
)
from survivor.models.montecarlo import combine_matchup_rates

_NAIVE_METRICS = ("epa_per_play", "pass_epa_per_play", "rush_epa_per_play", "success_rate", "yards_per_play")

# Weekday one-hots. Thursday/Monday games carry real short-week and
# prime-time effects; everything that isn't Thursday/Monday/Sunday (Friday
# and Saturday games, mostly late-December scheduling) collapses into
# "other" rather than getting its own sparse column.
_WEEKDAYS = ("Thursday", "Sunday", "Monday")


def _as_date(value) -> date | None:
    """Coerce a schedule row's `gameday` to a `datetime.date`.

    `stadiums.timezone_offset_hours` takes a real `date` (it needs DST
    resolution), but nflverse's cached schedules store `gameday` as an ISO
    string. Converting at this boundary keeps the geography helpers'
    contract honest instead of loosening them to accept anything.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Rolling recent form
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollingForm:
    team: str
    window: int
    n_games_used: int
    is_small_sample_fallback: bool
    epa_per_play: float
    success_rate: float


def compute_rolling_form(
    visible_team_games: pl.DataFrame,
    team: str,
    window: int,
    min_games_floor: int,
    season_level_epa: float,
    season_level_success_rate: float,
) -> RollingForm:
    """Unweighted mean over this team's last `window` visible games.

    `visible_team_games` must ALREADY be `as_of.visible(team_game_table)` —
    this function does no gating of its own, so that the caller can gate
    once per as-of point instead of once per team.

    The window crosses season boundaries freely: a team's roster and system
    don't reset on January 1, and refusing to look back past week 1 would
    make early-season rolling features pure noise exactly when they're
    least reliable already.

    Small-sample safeguard: if fewer than `min_games_floor` games are
    visible at all, this returns the team's season-level *shrunk* rating
    verbatim and flags the fallback. That makes the trajectory feature
    (recent minus season) structurally exactly 0.0 rather than an imputed
    guess — "no signal yet" never gets to masquerade as "confirmed flat
    form." Note the floor is on games *available*, not on `window`: a team
    with 4 visible games and window=5 uses its real 4-game mean.
    """
    team_games = visible_team_games.filter(pl.col("team") == team).sort(["season", "week"])
    n_available = team_games.height

    if n_available < min_games_floor:
        return RollingForm(
            team=team,
            window=window,
            n_games_used=n_available,
            is_small_sample_fallback=True,
            epa_per_play=season_level_epa,
            success_rate=season_level_success_rate,
        )

    recent = team_games.tail(window)
    epa = recent["epa_per_play"].mean()
    success = recent["success_rate"].mean()
    return RollingForm(
        team=team,
        window=window,
        n_games_used=recent.height,
        is_small_sample_fallback=False,
        epa_per_play=float(epa) if epa is not None else season_level_epa,
        success_rate=float(success) if success is not None else season_level_success_rate,
    )


# ---------------------------------------------------------------------------
# Starting QB continuity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QBContinuityFeatures:
    starter_id: str | None
    starter_is_projected: bool  # True = resolved from a depth chart, not from the game's own row
    starter_changed: bool  # vs. this team's most recent prior visible game
    games_started_this_season: int  # current starter's consecutive run, backward from most recent


def build_starter_history(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, game) with that game's starting QB: season, week,
    game_type, team, starter_id.

    Built once and reused across every as-of point, since the underlying
    schedules table doesn't change between them. Rows with a null starter
    (an unplayed future game) are kept — `as_of.visible()` filters by time,
    and a null starter in the *past* is genuinely "unknown," not "future."
    """
    home = schedules.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("game_type"),
        pl.col("home_team").alias("team"),
        pl.col("home_qb_id").alias("starter_id"),
    )
    away = schedules.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("game_type"),
        pl.col("away_team").alias("team"),
        pl.col("away_qb_id").alias("starter_id"),
    )
    return pl.concat([home, away])


def resolve_starter(
    row_qb_id: str | None,
    depth_charts: pl.DataFrame | None,
    team: str,
    as_of: AsOf,
) -> tuple[str | None, bool]:
    """The entire training/inference asymmetry, in one place.

    If the game's own row carries a starter (`home_qb_id`/`away_qb_id`),
    use it. That is correct for a training row AND for a historical
    backtest target row: a game's own starting QB is a pre-game fact about
    that game, the same class of information `spread_line` is for
    models/vegas.py — not a peek at the future.

    Only when that field is null — which in practice means a genuinely
    unplayed future game (verified: 240 of 272 2026 games have a null
    `home_qb_id`, versus zero nulls in completed seasons) — does this fall
    back to the most recent depth chart at or before the as-of point.

    There is deliberately no `mode="train"|"predict"` flag: the row's own
    data decides, so no caller can get it wrong by forgetting to set one.

    Returns `(starter_id, is_projected)`. Missing/unusable depth charts
    return `(None, True)` rather than raising — a team whose starter can't
    be resolved still gets every other feature.
    """
    if row_qb_id is not None:
        return row_qb_id, False

    if depth_charts is None:
        return None, True

    required = {"season", "week", "club_code", "position", "depth_team", "gsis_id"}
    if not required.issubset(set(depth_charts.columns)):
        return None, True

    candidates = depth_charts.filter(
        (pl.col("club_code") == team)
        & (pl.col("position") == "QB")
        & (pl.col("depth_team").cast(pl.Utf8) == "1")
        & (
            (pl.col("season") < as_of.season)
            | ((pl.col("season") == as_of.season) & (pl.col("week") <= as_of.week))
        )
    ).sort(["season", "week"])

    if candidates.height == 0:
        return None, True
    return candidates["gsis_id"][-1], True


def qb_continuity_features(
    visible_starters: pl.DataFrame,
    as_of: AsOf,
    team: str,
    current_starter_id: str | None,
    starter_is_projected: bool,
) -> QBContinuityFeatures:
    """Compare `current_starter_id` against this team's visible starting-QB
    history to derive continuity signals.

    `visible_starters` must ALREADY be `as_of.visible(starter_history)`.
    `games_started_this_season` counts the current starter's consecutive
    run backward from the most recent game, stopping at the season boundary
    — a QB's 3-game run within this season is a different signal from a
    3-game run that straddles last January.
    """
    if current_starter_id is None:
        return QBContinuityFeatures(None, True, False, 0)

    history = (
        visible_starters.filter((pl.col("team") == team) & pl.col("starter_id").is_not_null())
        .sort(["season", "week"])
    )
    if history.height == 0:
        return QBContinuityFeatures(current_starter_id, starter_is_projected, False, 0)

    starters = history["starter_id"].to_list()
    seasons = history["season"].to_list()

    changed = starters[-1] != current_starter_id

    run = 0
    for starter, season in zip(reversed(starters), reversed(seasons)):
        if starter != current_starter_id or season != as_of.season:
            break
        run += 1

    return QBContinuityFeatures(current_starter_id, starter_is_projected, changed, run)


# ---------------------------------------------------------------------------
# The pure per-matchup combiner
# ---------------------------------------------------------------------------


def _rating_fields(prefix: str, rating: TeamRating | None) -> dict[str, float | None]:
    if rating is None:
        return {f"{prefix}_{m}": None for m in _NAIVE_METRICS} | {f"{prefix}_n_games": None}
    return {f"{prefix}_{m}": getattr(rating, m) for m in _NAIVE_METRICS} | {f"{prefix}_n_games": float(rating.n_games)}


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _matchup_rate_fields(
    prefix: str,
    offense: TeamDriveProfile | None,
    defense: TeamDriveProfile | None,
    league: TeamDriveProfile,
) -> dict[str, float | None]:
    """This side's per-drive outcome distribution once its offense is
    matched against the opponent's defense — reusing Method 2's own log5
    combination rather than a second, subtly-different implementation. The
    ML model therefore sees the same matchup-resolved quantities Method 2
    simulates from, and the two methods differ in how they *use* them
    (fitted classifier vs. stochastic simulation), not in what they see.
    """
    if offense is None or defense is None:
        return {f"{prefix}_matchup_{cat}": None for cat in OUTCOME_CATEGORIES}
    rates = combine_matchup_rates(offense.rates, defense.rates, league.rates)
    return {f"{prefix}_matchup_{cat}": rates[cat] for cat in OUTCOME_CATEGORIES}


def _turnover_rate(profile: TeamDriveProfile | None) -> float | None:
    if profile is None:
        return None
    return profile.rates["turnover_int"] + profile.rates["turnover_fumble"]


def _build_matchup_row(
    home_team: str,
    away_team: str,
    home_naive: TeamRating | None,
    away_naive: TeamRating | None,
    home_adj: OpponentAdjustedRating | None,
    away_adj: OpponentAdjustedRating | None,
    home_off: TeamDriveProfile | None,
    away_off: TeamDriveProfile | None,
    home_def: TeamDriveProfile | None,
    away_def: TeamDriveProfile | None,
    league: TeamDriveProfile,
    home_rolling: dict[int, RollingForm],
    away_rolling: dict[int, RollingForm],
    home_qb: QBContinuityFeatures,
    away_qb: QBContinuityFeatures,
    rest: RestFeatures | None,
    home_travel: TravelFeatures | None,
    away_travel: TravelFeatures | None,
    weather: WeatherFeatures,
    home_win_pct: float | None,
    away_win_pct: float | None,
    week: int,
    div_game: int | None,
    weekday: str | None,
) -> dict[str, float | None]:
    """Pure: resolved inputs in, one flat feature dict out. No I/O, no
    `AsOf`, no dataframe scans — which is what makes it unit-testable on
    hand-built synthetic inputs."""
    row: dict[str, float | None] = {}

    # --- naive shrunk ratings: raw both sides, plus the differential ---
    row |= _rating_fields("home", home_naive)
    row |= _rating_fields("away", away_naive)
    for metric in _NAIVE_METRICS:
        row[f"{metric}_diff"] = _diff(row[f"home_{metric}"], row[f"away_{metric}"])

    # --- opponent-adjusted ratings ---
    # `proj_epa_diff` is the single most direct "who is better, accounting
    # for who they played" number available: each offense's projected
    # EPA/play against this specific opponent's defense, differenced.
    if home_adj is not None and away_adj is not None:
        hfa = home_adj.hfa
        home_proj = home_adj.projected_epa(away_adj.def_rating, is_home=True, hfa_value=hfa)
        away_proj = away_adj.projected_epa(home_adj.def_rating, is_home=False, hfa_value=hfa)
        row |= {
            "home_off_rating": home_adj.off_rating,
            "home_def_rating": home_adj.def_rating,
            "away_off_rating": away_adj.off_rating,
            "away_def_rating": away_adj.def_rating,
            "hfa": hfa,
            "home_proj_epa": home_proj,
            "away_proj_epa": away_proj,
            "proj_epa_diff": home_proj - away_proj,
            "off_rating_diff": home_adj.off_rating - away_adj.off_rating,
            "def_rating_diff": home_adj.def_rating - away_adj.def_rating,
        }
    else:
        row |= dict.fromkeys(
            (
                "home_off_rating",
                "home_def_rating",
                "away_off_rating",
                "away_def_rating",
                "hfa",
                "home_proj_epa",
                "away_proj_epa",
                "proj_epa_diff",
                "off_rating_diff",
                "def_rating_diff",
            ),
            None,
        )

    # --- rolling form + trajectory ---
    for window in sorted(home_rolling):
        h, a = home_rolling[window], away_rolling[window]
        row[f"home_rolling{window}_epa"] = h.epa_per_play
        row[f"away_rolling{window}_epa"] = a.epa_per_play
        row[f"rolling{window}_epa_diff"] = _diff(h.epa_per_play, a.epa_per_play)
        row[f"home_rolling{window}_success"] = h.success_rate
        row[f"away_rolling{window}_success"] = a.success_rate
        row[f"rolling{window}_success_diff"] = _diff(h.success_rate, a.success_rate)
        row[f"home_rolling{window}_is_fallback"] = float(h.is_small_sample_fallback)
        row[f"away_rolling{window}_is_fallback"] = float(a.is_small_sample_fallback)
        # Trajectory: recent form relative to the team's own season-level
        # rating. Positive = playing better lately than their season says.
        home_traj = _diff(h.epa_per_play, row["home_epa_per_play"])
        away_traj = _diff(a.epa_per_play, row["away_epa_per_play"])
        row[f"home_trajectory{window}_epa"] = home_traj
        row[f"away_trajectory{window}_epa"] = away_traj
        row[f"trajectory{window}_epa_diff"] = _diff(home_traj, away_traj)

    # --- drive-rate matchup distributions ---
    row |= _matchup_rate_fields("home", home_off, away_def, league)
    row |= _matchup_rate_fields("away", away_off, home_def, league)
    row["td_rate_diff"] = _diff(row.get("home_matchup_td"), row.get("away_matchup_td"))
    row["turnover_rate_diff"] = _diff(
        (row["home_matchup_turnover_int"] + row["home_matchup_turnover_fumble"])
        if row.get("home_matchup_turnover_int") is not None
        else None,
        (row["away_matchup_turnover_int"] + row["away_matchup_turnover_fumble"])
        if row.get("away_matchup_turnover_int") is not None
        else None,
    )
    row["plays_per_drive_diff"] = _diff(
        home_off.plays_per_drive if home_off else None, away_off.plays_per_drive if away_off else None
    )
    row["red_zone_td_rate_diff"] = _diff(
        home_off.red_zone_td_rate if home_off else None, away_off.red_zone_td_rate if away_off else None
    )
    row["drives_per_game_diff"] = _diff(
        home_off.drives_per_game if home_off else None, away_off.drives_per_game if away_off else None
    )
    row["home_off_turnover_rate"] = _turnover_rate(home_off)
    row["away_off_turnover_rate"] = _turnover_rate(away_off)
    row["home_def_turnover_rate"] = _turnover_rate(home_def)
    row["away_def_turnover_rate"] = _turnover_rate(away_def)

    # --- situational ---
    row |= {
        "rest_advantage_home": float(rest.rest_advantage_home) if rest else None,
        "home_rest": float(rest.home_rest) if rest else None,
        "away_rest": float(rest.away_rest) if rest else None,
        "home_short_week": float(rest.home_short_week) if rest else None,
        "away_short_week": float(rest.away_short_week) if rest else None,
        "home_off_bye": float(rest.home_off_bye) if rest else None,
        "away_off_bye": float(rest.away_off_bye) if rest else None,
        "home_travel_miles": home_travel.travel_miles if home_travel else None,
        "away_travel_miles": away_travel.travel_miles if away_travel else None,
        "travel_miles_diff": _diff(
            home_travel.travel_miles if home_travel else None, away_travel.travel_miles if away_travel else None
        ),
        "home_tz_offset": home_travel.timezone_offset_hours if home_travel else None,
        "away_tz_offset": away_travel.timezone_offset_hours if away_travel else None,
        "tz_offset_diff": _diff(
            home_travel.timezone_offset_hours if home_travel else None,
            away_travel.timezone_offset_hours if away_travel else None,
        ),
        "is_neutral_site": float(home_travel.is_neutral_site) if home_travel else None,
        "is_dome": float(weather.is_dome),
        "cold_game": float(weather.cold_game),
        "high_wind": float(weather.high_wind),
        "pass_efficiency_multiplier": weather.pass_efficiency_multiplier,
        "fg_success_multiplier": weather.fg_success_multiplier,
        "div_game": float(div_game) if div_game is not None else None,
        "week": float(week),
    }
    for day in _WEEKDAYS:
        row[f"is_{day.lower()}"] = float(weekday == day) if weekday is not None else None

    # --- standings (cheap, un-simulated: see module note on motivation) ---
    row |= {
        "home_win_pct": home_win_pct,
        "away_win_pct": away_win_pct,
        "win_pct_diff": _diff(home_win_pct, away_win_pct),
    }

    # --- QB continuity ---
    row |= {
        "home_starter_changed": float(home_qb.starter_changed),
        "away_starter_changed": float(away_qb.starter_changed),
        "home_games_started_this_season": float(home_qb.games_started_this_season),
        "away_games_started_this_season": float(away_qb.games_started_this_season),
        "home_starter_is_projected": float(home_qb.starter_is_projected),
        "away_starter_is_projected": float(away_qb.starter_is_projected),
    }

    return row


def feature_columns(rolling_windows: tuple[int, ...]) -> tuple[str, ...]:
    """The exact feature column names produced for a given set of rolling
    windows, in a stable order. Derived from one synthetic `_build_matchup_row`
    call rather than a hand-maintained literal list, so the two can't drift
    apart — a duplicated list is exactly the kind of thing that goes stale
    the first time a feature is added.
    """
    league = TeamDriveProfile(
        team="LEAGUE",
        n_drives=0,
        rates=dict.fromkeys(OUTCOME_CATEGORIES, 1.0 / len(OUTCOME_CATEGORIES)),
        plays_per_drive=5.5,
        red_zone_td_rate=0.55,
        drives_per_game=9.7,
    )
    empty_rolling = {
        w: RollingForm(team="X", window=w, n_games_used=0, is_small_sample_fallback=True, epa_per_play=0.0, success_rate=0.0)
        for w in rolling_windows
    }
    empty_qb = QBContinuityFeatures(None, True, False, 0)
    row = _build_matchup_row(
        home_team="H",
        away_team="A",
        home_naive=None,
        away_naive=None,
        home_adj=None,
        away_adj=None,
        home_off=None,
        away_off=None,
        home_def=None,
        away_def=None,
        league=league,
        home_rolling=empty_rolling,
        away_rolling=empty_rolling,
        home_qb=empty_qb,
        away_qb=empty_qb,
        rest=None,
        home_travel=None,
        away_travel=None,
        weather=weather_features(None, None, None),
        home_win_pct=None,
        away_win_pct=None,
        week=1,
        div_game=None,
        weekday=None,
    )
    return tuple(row.keys())


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

PASSTHROUGH_COLUMNS = ("game_id", "season", "week", "home_team", "away_team", "home_win")


def build_feature_table(
    games: pl.DataFrame,
    team_game_table: pl.DataFrame,
    drive_table: pl.DataFrame,
    schedules: pl.DataFrame,
    depth_charts: pl.DataFrame | None = None,
    shrink_k: float = 5.0,
    prior_blend_prev_season: float = 0.6,
    prior_blend_league_mean: float = 0.4,
    opp_adj_alpha: float = 40.0,
    rolling_windows: tuple[int, ...] = (3, 5),
    rolling_min_games_floor: int = 3,
    drop_ties: bool = True,
) -> pl.DataFrame:
    """One feature row per game in `games`.

    `games` needs game_id/season/week/game_type/home_team/away_team plus the
    pre-game metadata columns (home_rest, away_rest, roof, temp, wind,
    stadium_id, gameday, weekday, div_game, home_qb_id, away_qb_id) and,
    for training rows, home_score/away_score.

    `home_win` is 1/0 for completed decisive games and null for unplayed
    ones, so the same function serves training and inference. With
    `drop_ties=True` (the training default) tied games are dropped, matching
    `collect_predictions`'s convention; inference callers pass False since
    an unplayed game has no outcome to tie.

    Games are grouped by (season, week, game_type) so each as-of point's
    ratings/profiles/standings are computed once and shared — see the
    module docstring on cost.
    """
    if games.height == 0:
        return pl.DataFrame()

    starter_history = build_starter_history(schedules)
    travel_cache: dict[tuple, TravelFeatures | None] = {}

    rows: list[dict] = []
    group_keys = ["season", "week", "game_type"]

    for (season, week, game_type), group in games.sort(group_keys).group_by(group_keys, maintain_order=True):
        as_of = AsOf(season=int(season), week=int(week), game_type=game_type)

        # --- computed once per as-of point, shared by every game in it ---
        naive = compute_naive_ratings(
            team_game_table, as_of, shrink_k, prior_blend_prev_season, prior_blend_league_mean
        )
        adjusted = compute_opponent_adjusted_ratings(team_game_table, as_of, alpha=opp_adj_alpha)
        offense = compute_offense_profiles(
            drive_table, as_of, shrink_k, prior_blend_prev_season, prior_blend_league_mean
        )
        defense = compute_defense_profiles(
            drive_table, as_of, shrink_k, prior_blend_prev_season, prior_blend_league_mean
        )
        league = league_average_profile(drive_table, as_of)
        standings = standings_as_of(schedules, as_of)
        visible_team_games = as_of.visible(team_game_table)
        visible_starters = as_of.visible(starter_history)

        rolling_cache: dict[tuple[str, int], RollingForm] = {}

        def _rolling(team: str) -> dict[int, RollingForm]:
            out = {}
            for window in rolling_windows:
                key = (team, window)
                if key not in rolling_cache:
                    rating = naive.get(team)
                    rolling_cache[key] = compute_rolling_form(
                        visible_team_games,
                        team,
                        window,
                        rolling_min_games_floor,
                        season_level_epa=rating.epa_per_play if rating else 0.0,
                        season_level_success_rate=rating.success_rate if rating else 0.0,
                    )
                out[window] = rolling_cache[key]
            return out

        def _travel(team: str, stadium_id, gameday, is_designated_home: bool) -> TravelFeatures | None:
            # travel_features re-derives the team's home base (a mode over
            # that season's home games) on every call, so memoize the whole
            # result: a team visits the same stadium on the same date at
            # most once, and across a multi-season build this collapses
            # thousands of repeated mode computations.
            game_date = _as_date(gameday)
            if stadium_id is None or game_date is None:
                return None
            cache_key = (team, int(season), stadium_id, game_date, is_designated_home)
            if cache_key not in travel_cache:
                travel_cache[cache_key] = travel_features(
                    schedules, team, int(season), stadium_id, game_date, is_designated_home
                )
            return travel_cache[cache_key]

        for game in group.iter_rows(named=True):
            home, away = game["home_team"], game["away_team"]

            home_starter, home_projected = resolve_starter(game.get("home_qb_id"), depth_charts, home, as_of)
            away_starter, away_projected = resolve_starter(game.get("away_qb_id"), depth_charts, away, as_of)

            feature_row = _build_matchup_row(
                home_team=home,
                away_team=away,
                home_naive=naive.get(home),
                away_naive=naive.get(away),
                home_adj=adjusted.get(home),
                away_adj=adjusted.get(away),
                home_off=offense.get(home),
                away_off=offense.get(away),
                home_def=defense.get(home),
                away_def=defense.get(away),
                league=league,
                home_rolling=_rolling(home),
                away_rolling=_rolling(away),
                home_qb=qb_continuity_features(visible_starters, as_of, home, home_starter, home_projected),
                away_qb=qb_continuity_features(visible_starters, as_of, away, away_starter, away_projected),
                rest=rest_features(game.get("home_rest"), game.get("away_rest")),
                home_travel=_travel(home, game.get("stadium_id"), game.get("gameday"), True),
                away_travel=_travel(away, game.get("stadium_id"), game.get("gameday"), False),
                weather=weather_features(game.get("roof"), game.get("temp"), game.get("wind")),
                home_win_pct=standings[home].win_pct if home in standings else None,
                away_win_pct=standings[away].win_pct if away in standings else None,
                week=int(week),
                div_game=game.get("div_game"),
                weekday=game.get("weekday"),
            )

            # An unplayed game and a tie both yield a null label: one has no
            # outcome yet, the other has no *decisive* outcome, and the
            # classifier trains on decisive games only (same convention as
            # collect_predictions and SpreadLogisticModel).
            home_score, away_score = game.get("home_score"), game.get("away_score")
            is_played = home_score is not None and away_score is not None
            is_tie = is_played and home_score == away_score
            if drop_ties and is_tie:
                continue
            home_win = None if (not is_played or is_tie) else int(home_score > away_score)

            rows.append(
                {
                    "game_id": game["game_id"],
                    "season": int(season),
                    "week": int(week),
                    "home_team": home,
                    "away_team": away,
                    "home_win": home_win,
                    **feature_row,
                }
            )

    if not rows:
        return pl.DataFrame()

    # Explicit schema rather than inference. Every feature is a float by
    # construction, but polars infers dtypes from the first 100 rows only,
    # and a feature that is legitimately null for the first 100 games (an
    # early-season travel or opponent-adjusted value, say) would be inferred
    # as Null and then fail to accept a float thousands of rows later.
    schema = {
        "game_id": pl.Utf8,
        "season": pl.Int64,
        "week": pl.Int64,
        "home_team": pl.Utf8,
        "away_team": pl.Utf8,
        "home_win": pl.Int64,
    } | {name: pl.Float64 for name in feature_columns(tuple(rolling_windows))}
    return pl.DataFrame(rows, schema=schema, strict=False)
