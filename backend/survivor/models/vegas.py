"""Method 1 (baseline): convert Vegas spread/moneyline into win probability.

Two independent conversions, blended where both exist:
  - spread_line -> P(home win), via logistic regression fit on completed
    games (walk-forward: callers pass only `AsOf.visible()` games to `fit`).
    Complete back to 1999, so this is the model that can run every week.
  - moneyline -> P(home win), via de-vigging. Complete 2010+.

A closed-form normal-distribution sanity check (`spread_to_prob_normal`) is
included so the fitted logistic isn't trusted blindly — see `sigma_from_games`.

Ties are modeled separately as a small constant rate rather than fit against
spread, because they are extremely rare (15 of 6,967 REG games from
1999-2025, a 0.22% rate) with no significant relationship to |spread| in
that sample — fitting a curve to 15 events would be fitting noise.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

from survivor.config import CONFIG
from survivor.models.base import REQUIRED_OUTPUT_COLUMNS

_COMPLETED = (pl.col("home_score").is_not_null()) & (pl.col("away_score").is_not_null())


# ---------------------------------------------------------------------------
# Tie modeling
# ---------------------------------------------------------------------------


class ConstantTieModel:
    """Tie probability = empirical rate in training data, applied to all games."""

    def __init__(self) -> None:
        self.tie_rate: float = 0.0

    def fit(self, games: pl.DataFrame) -> "ConstantTieModel":
        completed = games.filter(_COMPLETED)
        if completed.height == 0:
            self.tie_rate = 0.0
            return self
        is_tie = (completed["home_score"] == completed["away_score"]).cast(pl.Float64)
        self.tie_rate = float(is_tie.mean())
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.tie_rate)


# ---------------------------------------------------------------------------
# Spread -> win probability
# ---------------------------------------------------------------------------


def sigma_from_games(games: pl.DataFrame) -> float:
    """Residual std of (home_margin - spread_line) over decisive games.

    This is the classic "spread is an unbiased point estimate of margin,
    residual ~ Normal(0, sigma)" assumption — used only for the closed-form
    sanity check, not the primary model.
    """
    completed = games.filter(_COMPLETED & pl.col("spread_line").is_not_null() & (pl.col("home_score") != pl.col("away_score")))
    if completed.height < 10:
        return 13.5  # fallback close to the commonly-cited historical NFL value
    margin = (completed["home_score"] - completed["away_score"]).to_numpy()
    spread = completed["spread_line"].to_numpy()
    resid = margin - spread
    return float(np.std(resid, ddof=1))


def spread_to_prob_normal(spread_line: np.ndarray, sigma: float) -> np.ndarray:
    """Closed-form sanity check: P(home win) = Phi(spread_line / sigma).

    Positive spread_line means the home team is favored (verified empirically
    against nflverse data: corr(spread_line, home_margin) ~= 0.43, regression
    slope ~= 1.0) — a fair coin-toss-continuous approximation, ties ignored.
    """
    return stats.norm.cdf(spread_line / sigma)


class SpreadLogisticModel:
    """Logistic regression: P(home win | not tie) ~ spread_line.

    Fit on decisive (non-tie) completed games only; refit walk-forward by
    callers passing progressively larger AsOf.visible() windows.
    """

    def __init__(self) -> None:
        self.clf: LogisticRegression | None = None
        self.sigma_reference: float = 13.5  # for spread_to_prob_normal() sanity check

    def fit(self, games: pl.DataFrame) -> "SpreadLogisticModel":
        decisive = games.filter(_COMPLETED & pl.col("spread_line").is_not_null() & (pl.col("home_score") != pl.col("away_score")))
        if decisive.height < 20:
            raise ValueError(f"need at least 20 decisive games with spread_line to fit SpreadLogisticModel, got {decisive.height}")
        x = decisive["spread_line"].to_numpy().reshape(-1, 1)
        y = (decisive["home_score"] > decisive["away_score"]).to_numpy().astype(int)
        self.clf = LogisticRegression()
        self.clf.fit(x, y)
        self.sigma_reference = sigma_from_games(games)
        return self

    def predict_win_share(self, spread_line: np.ndarray) -> np.ndarray:
        """P(home win | not tie), i.e. win share of the two decisive outcomes."""
        if self.clf is None:
            raise RuntimeError("call fit() before predict_win_share()")
        return self.clf.predict_proba(spread_line.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Moneyline -> win probability (de-vigging)
# ---------------------------------------------------------------------------


def moneyline_to_implied_prob(ml: np.ndarray) -> np.ndarray:
    """Raw (vig-inflated) implied probability from American odds.

    Computed via masked assignment rather than np.where(pos_branch,
    neg_branch) because np.where evaluates both branches unconditionally —
    at ml=+100 the discarded negative-odds branch would divide by zero
    (-100/(-100+100)) and raise a spurious RuntimeWarning even though its
    result is never used.
    """
    ml = np.asarray(ml, dtype=float)
    out = np.empty_like(ml)
    pos = ml > 0
    out[pos] = 100.0 / (ml[pos] + 100.0)
    out[~pos] = (-ml[~pos]) / (-ml[~pos] + 100.0)
    return out


def devig_proportional(home_ml: np.ndarray, away_ml: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize raw implied probabilities to sum to 1 (the standard, simple de-vig)."""
    pi_home = moneyline_to_implied_prob(home_ml)
    pi_away = moneyline_to_implied_prob(away_ml)
    total = pi_home + pi_away
    return pi_home / total, pi_away / total


def devig_shin(home_ml: np.ndarray, away_ml: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shin's (1992/1993) method: models a fraction `z` of insider/informed
    money in the raw quoted probabilities and solves for the true
    probabilities that remove it. For two outcomes, given raw implied
    probabilities pi_i summing to Pi (the overround), z solves:

        sum_i [ sqrt(z^2 + 4(1-z) pi_i^2 / Pi) - z ] / (2(1-z)) = 1

    As z -> 0 this reduces to proportional normalization, so the two methods
    agree exactly on a vig-free market and diverge as the overround grows.
    """
    pi_home = moneyline_to_implied_prob(home_ml)
    pi_away = moneyline_to_implied_prob(away_ml)
    out_home = np.empty_like(pi_home)
    out_away = np.empty_like(pi_away)
    for i in range(len(pi_home)):
        ph, pa = pi_home[i], pi_away[i]
        big_pi = ph + pa

        def _shin_prob(pi_i: float, z: float) -> float:
            return (np.sqrt(z**2 + 4 * (1 - z) * pi_i**2 / big_pi) - z) / (2 * (1 - z))

        def _sum_minus_one(z: float) -> float:
            if z <= 1e-12:
                return (ph + pa) / big_pi - 1.0  # proportional at z=0, always 0
            return _shin_prob(ph, z) + _shin_prob(pa, z) - 1.0

        # z must be in [0, 1); overround is typically 1-8%, so z is small.
        # Bracket search: at z=0 the residual is 0 (proportional sums to 1 trivially
        # is NOT true in general for this formula's z->0 limit approached from above,
        # so use a tiny epsilon lower bound instead of exactly 0).
        lo, hi = 1e-9, 0.5
        try:
            z = brentq(_sum_minus_one, lo, hi, xtol=1e-10)
        except ValueError:
            # no root in range (near-fair market) -> proportional is the right answer
            z = 0.0
        if z <= 1e-9:
            out_home[i], out_away[i] = ph / big_pi, pa / big_pi
        else:
            out_home[i] = _shin_prob(ph, z)
            out_away[i] = _shin_prob(pa, z)
    return out_home, out_away


# ---------------------------------------------------------------------------
# Combined Vegas model
# ---------------------------------------------------------------------------


class VegasModel:
    """Blends spread-derived and moneyline-derived win probabilities.

    Falls back to spread-only when moneyline is missing (pre-2010, or any
    row where a moneyline wasn't posted). Ties get the constant-rate model's
    probability; the home/away split of the remaining probability mass comes
    from whichever of spread/moneyline is available.
    """

    name = "vegas"

    def __init__(self, devig_method: str | None = None) -> None:
        self.devig_method = devig_method or CONFIG.model.vegas.devig_method
        self.spread_model = SpreadLogisticModel()
        self.tie_model = ConstantTieModel()

    def fit(self, games: pl.DataFrame) -> "VegasModel":
        self.spread_model.fit(games)
        self.tie_model.fit(games)
        return self

    def predict_proba(self, games: pl.DataFrame) -> pl.DataFrame:
        n = games.height
        tie_prob = self.tie_model.predict(n)
        win_mass = 1.0 - tie_prob  # probability mass split between home/away win

        has_spread = games["spread_line"].is_not_null().to_numpy()
        has_ml = (
            games["home_moneyline"].is_not_null().to_numpy() & games["away_moneyline"].is_not_null().to_numpy()
            if "home_moneyline" in games.columns
            else np.zeros(n, dtype=bool)
        )

        home_share = np.full(n, np.nan)

        spread_arr = games["spread_line"].fill_null(0.0).to_numpy()
        spread_share = self.spread_model.predict_win_share(spread_arr)

        if has_ml.any():
            home_ml = games["home_moneyline"].fill_null(100).to_numpy()
            away_ml = games["away_moneyline"].fill_null(100).to_numpy()
            if self.devig_method == "shin":
                ml_home_share, _ = devig_shin(home_ml, away_ml)
            else:
                ml_home_share, _ = devig_proportional(home_ml, away_ml)
        else:
            ml_home_share = np.full(n, np.nan)

        both = has_spread & has_ml
        spread_only = has_spread & ~has_ml
        ml_only = (~has_spread) & has_ml

        home_share[both] = 0.5 * spread_share[both] + 0.5 * ml_home_share[both]
        home_share[spread_only] = spread_share[spread_only]
        home_share[ml_only] = ml_home_share[ml_only]
        # neither spread nor moneyline available: fall back to a coin flip
        # among decisive outcomes (should be rare/never in practice).
        neither = ~has_spread & ~has_ml
        home_share[neither] = 0.5

        home_win_prob = win_mass * home_share
        away_win_prob = win_mass * (1 - home_share)

        return games.with_columns(
            [
                pl.Series("home_win_prob", home_win_prob),
                pl.Series("away_win_prob", away_win_prob),
                pl.Series("tie_prob", tie_prob),
            ]
        )
