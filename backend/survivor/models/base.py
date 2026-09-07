"""Common interface both win-probability methods implement.

A model is fit on a set of *already-completed* games and predicts on any
set of games (completed or upcoming). Nothing in this interface enforces
the no-lookahead rule — that's `AsOf.visible()`'s job (see clock.py). Every
caller is responsible for only ever passing `asof.visible(games)` to `fit`.

Both models output three probabilities per game — home win, away win, tie —
because whether a tie counts as a survivor-pool loss is a pool rule, not a
modeling assumption (config.pool.tie_counts_as_loss). Keeping tie_prob as
its own column lets that rule be applied downstream instead of baked in.
"""

from __future__ import annotations

from typing import Protocol

import polars as pl

REQUIRED_OUTPUT_COLUMNS = ("home_win_prob", "away_win_prob", "tie_prob")


class WinProbModel(Protocol):
    name: str

    def fit(self, games: pl.DataFrame) -> "WinProbModel":
        """Fit on completed games (must have home_score/away_score, non-null)."""
        ...

    def predict_proba(self, games: pl.DataFrame) -> pl.DataFrame:
        """Return `games` with home_win_prob/away_win_prob/tie_prob columns added.

        The three columns must sum to 1.0 (within floating point tolerance)
        on every row.
        """
        ...


def assert_valid_output(df: pl.DataFrame, atol: float = 1e-6) -> None:
    """Sanity check every model's predict_proba must satisfy. Use in tests."""
    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in df.columns:
            raise AssertionError(f"missing required output column: {col}")
    total = df["home_win_prob"] + df["away_win_prob"] + df["tie_prob"]
    bad = df.filter((total - 1.0).abs() > atol)
    if bad.height > 0:
        raise AssertionError(f"{bad.height} rows have probabilities not summing to 1.0")


def pick_survival_prob(
    home_win_prob: float,
    away_win_prob: float,
    tie_prob: float,
    picked_is_home: bool,
    tie_counts_as_loss: bool,
) -> float:
    """P(the picked side does not lose), given a pool's tie rule.

    This is the quantity survivor-pool decisions actually care about — not
    P(win) in isolation. Most pools treat a tie as elimination (the plan's
    and config's default), but the rule is a pool-config decision, so it is
    applied here rather than folded into the model output.
    """
    win_prob = home_win_prob if picked_is_home else away_win_prob
    if tie_counts_as_loss:
        return win_prob
    return win_prob + tie_prob
