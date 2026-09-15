"""Thin caching wrapper around nflreadpy.

nflreadpy already caches its own HTTP responses, but we keep a local parquet
snapshot per dataset so that:
  - backtests are reproducible even if upstream nflverse data is amended later
    (call `refresh=True` deliberately, not implicitly, to pull in changes)
  - offline reruns (`make recommend` without network) are fast and possible
  - play-by-play, which is large, is cached per-season so a weekly refresh
    only re-fetches the current season instead of the entire league history

Every loader here returns a polars.DataFrame with the schema nflreadpy hands
back — no column renaming happens here, only caching (with one deliberate
exception: relocated-franchise abbreviations, see RELOCATED_FRANCHISES).
Column selection and derivation belongs in features/, not this module.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfr
import polars as pl

from survivor.config import CONFIG

RAW_DIR = CONFIG.paths.resolve("raw")

# nflverse's two core datasets disagree about relocated franchises: the
# play-by-play uses each team's *current* abbreviation for every season of
# its history, while the schedules table uses whatever the abbreviation was
# at the time. So a 2011 Rams game is `STL` in schedules and `LA` in pbp.
#
# That mismatch silently broke every join between the two: any feature keyed
# by pbp (drive_rates.py profiles, team_ratings.py ratings) simply had no
# entry under the schedule's historical name, and the consuming code —
# optimize/projection.py, export/json_bundle.py — treats a missing profile as
# "skip this game." The effect was that all three franchises' games vanished
# from every backtest season before their move: 48 of 256 games in 2011,
# for instance.
#
# Normalizing schedules to the pbp convention here fixes it in one place, for
# every consumer at once. pbp is the one that's already self-consistent
# across history, so it wins.
RELOCATED_FRANCHISES = {
    "STL": "LA",  # Rams, St. Louis -> Los Angeles (2016)
    "SD": "LAC",  # Chargers, San Diego -> Los Angeles (2017)
    "OAK": "LV",  # Raiders, Oakland -> Las Vegas (2020)
}


def normalize_team_abbreviations(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Map historical franchise abbreviations to their current form."""
    present = [c for c in columns if c in df.columns]
    if not present:
        return df
    return df.with_columns([pl.col(c).replace(RELOCATED_FRANCHISES) for c in present])


def _ensure_dir() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_schedules(refresh: bool = False) -> pl.DataFrame:
    """Full schedule history: results, spread/moneyline, rest, weather, etc.

    This is the backbone dataset — nearly every feature and the Vegas model
    are built from its columns. See plan §1 for the column inventory.

    Team abbreviations are normalized to the play-by-play convention on the
    way out (see RELOCATED_FRANCHISES); the cached parquet keeps nflverse's
    raw form so a refresh stays a faithful snapshot.
    """
    _ensure_dir()
    path = RAW_DIR / "schedules.parquet"
    if refresh or not path.exists():
        df = nfr.load_schedules(seasons=True)
        df.write_parquet(path)
    else:
        df = pl.read_parquet(path)
    return normalize_team_abbreviations(df, ("home_team", "away_team"))


def load_teams(refresh: bool = False) -> pl.DataFrame:
    """Team identity table (handles franchise relocations/renames)."""
    _ensure_dir()
    path = RAW_DIR / "teams.parquet"
    if refresh or not path.exists():
        df = nfr.load_teams()
        df.write_parquet(path)
        return df
    return pl.read_parquet(path)


def load_injuries(refresh: bool = False) -> pl.DataFrame:
    """Weekly injury reports, 2009+."""
    _ensure_dir()
    path = RAW_DIR / "injuries.parquet"
    if refresh or not path.exists():
        df = nfr.load_injuries(seasons=True)
        df.write_parquet(path)
        return df
    return pl.read_parquet(path)


def load_depth_charts(refresh: bool = False) -> pl.DataFrame:
    """Weekly depth charts — used to detect a projected starting-QB change."""
    _ensure_dir()
    path = RAW_DIR / "depth_charts.parquet"
    if refresh or not path.exists():
        df = nfr.load_depth_charts(seasons=True)
        df.write_parquet(path)
        return df
    return pl.read_parquet(path)


def load_pbp(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Play-by-play, cached one parquet file per season.

    Only fetches seasons whose cache file is missing or `refresh=True`. Each
    season is ~100MB, so callers should pass only the seasons they actually
    need (feature builders filter by AsOf anyway — there is never a reason
    to load a season that couldn't possibly be visible).
    """
    _ensure_dir()
    pbp_dir = RAW_DIR / "pbp"
    pbp_dir.mkdir(parents=True, exist_ok=True)

    to_fetch = []
    cached_frames = []
    for season in seasons:
        path = pbp_dir / f"{season}.parquet"
        if refresh or not path.exists():
            to_fetch.append(season)
        else:
            cached_frames.append(pl.read_parquet(path))

    fetched_frames = []
    if to_fetch:
        fetched = nfr.load_pbp(seasons=to_fetch)
        for season in to_fetch:
            season_df = fetched.filter(pl.col("season") == season)
            season_df.write_parquet(pbp_dir / f"{season}.parquet")
            fetched_frames.append(season_df)

    frames = cached_frames + fetched_frames
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort(["season", "week", "game_id"])


def current_season(schedules: pl.DataFrame) -> int:
    """The season the schedule covers — the latest one with games on it.

    nflverse publishes a season's schedule months before kickoff, so this
    rolls over in the spring. That is fine for every caller here because they
    all guard separately on whether games have actually been *played*.
    """
    return int(schedules["season"].max())


def has_played_games(schedules: pl.DataFrame, season: int) -> bool:
    """Whether `season` has any final score yet.

    Worth naming because nflreadpy *rejects* a play-by-play request for a
    season with no games rather than returning an empty frame, so every
    caller assembling a season list has to check this first.
    """
    return (
        schedules.filter((pl.col("season") == season) & pl.col("home_score").is_not_null()).height > 0
    )


def default_pbp_seasons(schedules: pl.DataFrame, season: int | None = None) -> list[int]:
    """The season in progress and the one before it — what Methods 1 and 2
    need, and what a routine refresh should keep warm. Drops the current
    season until it has actually started (see `has_played_games`)."""
    season = season if season is not None else current_season(schedules)
    return [season - 1, season] if has_played_games(schedules, season) else [season - 1]


def refresh_all(pbp_seasons: list[int] | None = None, skip_pbp: bool = False) -> list[int]:
    """Force-refetch every cached dataset. Backs `make refresh`. Returns the
    play-by-play seasons actually refreshed.

    `pbp_seasons=None` means "work it out from the schedule" rather than
    "skip play-by-play". The distinction matters: the seasons used to be
    hardcoded in the Makefile, which went stale every September — the moment
    the new season's play-by-play started existing was the moment the refresh
    stopped fetching it. Schedules are refreshed first so the decision is
    made against today's results, not yesterday's cache.
    """
    schedules = load_schedules(refresh=True)
    load_teams(refresh=True)
    load_injuries(refresh=True)
    load_depth_charts(refresh=True)
    if skip_pbp:
        return []
    if pbp_seasons is None:
        pbp_seasons = default_pbp_seasons(schedules)
    if pbp_seasons:
        load_pbp(pbp_seasons, refresh=True)
    return pbp_seasons
