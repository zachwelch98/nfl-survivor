"""The leakage gate.

Every feature builder in this project takes an `AsOf` and must go through
`AsOf.visible(df)` to decide what data it is allowed to see. This is the one
enforcement point for "no lookahead" — see plan §1. Get this file right and
correctness elsewhere follows; get it wrong and every backtest number is a
lie no matter how careful the modeling code is.

Games are ordered by (season, stage, week), where `stage` accounts for the
postseason: regular season is stage 0, and playoff rounds (WC/DIV/CON/SB)
are later stages whose week numbers restart from 1 and would otherwise sort
*before* week 18 of the regular season if compared naively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

GameType = Literal["REG", "WC", "DIV", "CON", "SB"]

# Chronological stage ordering within a season. Higher = later in the season.
_STAGE_RANK: dict[str, int] = {
    "REG": 0,
    "WC": 1,
    "DIV": 2,
    "CON": 3,
    "SB": 4,
}


def stage_rank(game_type: str) -> int:
    try:
        return _STAGE_RANK[game_type]
    except KeyError as e:
        raise ValueError(f"Unknown game_type {game_type!r}; expected one of {list(_STAGE_RANK)}") from e


@dataclass(frozen=True)
class AsOf:
    """A point in time defined as 'immediately before this week's games'.

    Feature/model code asking "what happened before week W of season S"
    should construct AsOf(season=S, week=W, game_type="REG") and call
    `.visible(games_df)` — never filter season/week by hand elsewhere.
    """

    season: int
    week: int
    game_type: GameType = "REG"

    def __post_init__(self) -> None:
        if self.game_type not in _STAGE_RANK:
            raise ValueError(f"Unknown game_type {self.game_type!r}")
        if self.week < 1:
            raise ValueError(f"week must be >= 1, got {self.week}")

    @property
    def sort_key(self) -> tuple[int, int, int]:
        return (self.season, stage_rank(self.game_type), self.week)

    def visible(self, games: pl.DataFrame) -> pl.DataFrame:
        """Return only rows strictly before this point in the season.

        `games` must have columns: season (int), week (int), game_type (str).
        A row is visible iff its (season, stage_rank(game_type), week) sorts
        strictly before this AsOf's own key — i.e. games that have *already
        finished* (or at minimum, already kicked off) by the time we are
        standing at week `self.week`.
        """
        s, stage, w = self.sort_key
        rank_expr = pl.col("game_type").replace_strict(_STAGE_RANK, return_dtype=pl.Int64)
        key_season = pl.col("season")
        key_stage = rank_expr
        key_week = pl.col("week")

        before_season = key_season < s
        same_season_before_stage = (key_season == s) & (key_stage < stage)
        same_season_same_stage_before_week = (key_season == s) & (key_stage == stage) & (key_week < w)

        return games.filter(before_season | same_season_before_stage | same_season_same_stage_before_week)

    def __str__(self) -> str:
        return f"{self.season}-{self.game_type}-W{self.week:02d}"
