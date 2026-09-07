"""Thin caching wrapper around nflreadpy.

nflreadpy already caches its own HTTP responses, but we keep a local parquet
snapshot per dataset so that:
  - backtests are reproducible even if upstream nflverse data is amended later
    (call `refresh=True` deliberately, not implicitly, to pull in changes)
  - offline reruns (`make recommend` without network) are fast and possible
  - play-by-play, which is large, is cached per-season so a weekly refresh
    only re-fetches the current season instead of the entire league history

Every loader here returns a polars.DataFrame with the schema nflreadpy hands
back — no column renaming happens here, only caching. Column selection and
derivation belongs in features/, not this module.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfr
import polars as pl

from survivor.config import CONFIG

RAW_DIR = CONFIG.paths.resolve("raw")


def _ensure_dir() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_schedules(refresh: bool = False) -> pl.DataFrame:
    """Full schedule history: results, spread/moneyline, rest, weather, etc.

    This is the backbone dataset — nearly every feature and the Vegas model
    are built from its columns. See plan §1 for the column inventory.
    """
    _ensure_dir()
    path = RAW_DIR / "schedules.parquet"
    if refresh or not path.exists():
        df = nfr.load_schedules(seasons=True)
        df.write_parquet(path)
        return df
    return pl.read_parquet(path)


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


def refresh_all(pbp_seasons: list[int] | None = None) -> None:
    """Force-refetch every cached dataset. Backs `make refresh`."""
    load_schedules(refresh=True)
    load_teams(refresh=True)
    load_injuries(refresh=True)
    load_depth_charts(refresh=True)
    if pbp_seasons:
        load_pbp(pbp_seasons, refresh=True)
