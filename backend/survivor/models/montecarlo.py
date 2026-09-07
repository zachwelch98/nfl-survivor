"""Method 2 (challenger): drive-level Monte Carlo game simulator (plan §2).

Core idea: a game is an alternating sequence of possessions. Each team's
per-drive outcome distribution (features/drive_rates.py's shrunk category
rates) is combined with its opponent's defensive rate allowed using a
multinomial generalization of the "log5" formula, situational adjustments
(HFA, weather, rest, motivation) are layered in as logit shifts, and the
whole game is simulated n times in vectorized numpy — no Python loop over
individual simulations or drives.

Combination formula (plan §2), per category k:
    logit(p_matchup[k]) = logit(p_off[k]) + logit(p_def[k]) - logit(p_league[k])
followed by renormalizing across categories so they sum to 1 (the
"multinomial generalization" — ordinary log5 is exactly this for a single
binary category; there is no single canonical way to extend it past two
outcomes, so independent-per-category-then-renormalize is the documented
choice here).

This module's `simulate_game` takes already-combined per-team category
distributions and a drive count — it does not know about teams, seasons,
or AsOf. The orchestration in `MonteCarloModel` wires drive_rates.py
profiles and situational.py adjustments into those distributions. Method 2
deliberately does not share an interface with VegasModel (models/vegas.py)
yet: its inputs (play-by-play-derived profiles keyed by AsOf, plus a
schedule row's rest/weather/travel) are structurally different from
Vegas's (a schedule row with a posted line), and forcing a premature common
`fit(games)/predict_proba(games)` shape now, before the backtest harness
that actually needs to swap between them exists, would obscure that
difference rather than clarify it. A thin adapter will be added when the
backtest harness is built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from survivor.features.drive_rates import OUTCOME_CATEGORIES

_EPS = 1e-9
_CATEGORY_INDEX = {c: i for i, c in enumerate(OUTCOME_CATEGORIES)}
_TD_IDX = _CATEGORY_INDEX["td"]
_FG_IDX = _CATEGORY_INDEX["fg"]
_SAFETY_IDX = _CATEGORY_INDEX["safety"]
_TURNOVER_IDXS = (_CATEGORY_INDEX["turnover_int"], _CATEGORY_INDEX["turnover_fumble"])


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def combine_matchup_rates(off_rates: dict[str, float], def_rates: dict[str, float], league_rates: dict[str, float]) -> dict[str, float]:
    """Log5-style combination of an offense's category rates with the
    opposing defense's allowed rates, relative to the league baseline —
    see module docstring for the formula. Renormalized to sum to 1.

    If off_rates == def_rates == league_rates for every category (a
    perfectly average matchup), this returns league_rates unchanged.
    """
    adjusted = {}
    for k in OUTCOME_CATEGORIES:
        logit_adj = _logit(off_rates[k]) + _logit(def_rates[k]) - _logit(league_rates[k])
        adjusted[k] = _sigmoid(logit_adj)
    total = sum(adjusted.values())
    if total <= 0:
        return dict(league_rates)
    return {k: v / total for k, v in adjusted.items()}


def apply_logit_shifts(rates: dict[str, float], shifts: dict[str, float]) -> dict[str, float]:
    """Add a per-category logit shift (e.g. from situational adjustments)
    and renormalize. A shift of 0.0 on every category is a no-op."""
    adjusted = {}
    for k in OUTCOME_CATEGORIES:
        adjusted[k] = _sigmoid(_logit(rates[k]) + shifts.get(k, 0.0))
    total = sum(adjusted.values())
    if total <= 0:
        return dict(rates)
    return {k: v / total for k, v in adjusted.items()}


def multiplier_to_category_shifts(multiplier: float, scale: float = 3.0) -> dict[str, float]:
    """Convert a single scoring-efficiency multiplier (1.0 = neutral, <1.0 =
    worse offensive conditions e.g. bad weather/short rest/no motivation)
    into per-category logit shifts: td/fg shift down, turnovers shift up,
    by the same log-multiplier magnitude (scaled by `scale` since a raw
    EPA-style multiplier of e.g. 0.93 is a small perturbation in EPA terms
    but should move scoring-category odds by a more noticeable amount to
    have any real effect once passed through combine_matchup_rates's
    renormalization). Punt/missed_fg/turnover_on_downs/safety are left
    unshifted and simply absorb whatever renormalization requires.
    """
    log_m = float(np.log(max(multiplier, 1e-6)))
    shifts = dict.fromkeys(OUTCOME_CATEGORIES, 0.0)
    shifts["td"] = scale * log_m
    shifts["fg"] = scale * log_m
    shifts["turnover_int"] = -scale * log_m
    shifts["turnover_fumble"] = -scale * log_m
    return shifts


@dataclass(frozen=True)
class DriveSimConfig:
    n_simulations: int = 10000
    seed: int = 1337
    xp_success_rate: float = 0.94
    return_td_given_turnover_rate: float = 0.088  # empirical: 56/635 "Opp touchdown" drives, 2024 season
    # Regulation-time scores tie roughly 4% of the time in this simulator (a
    # discrete drive-scoring process genuinely produces that many exact
    # ties — validated against real per-game data), but real NFL games end
    # tied only ~0.2% of the time because overtime resolves nearly all of
    # them. ot_still_tied_rate is P(remains tied | reached overtime); 0.05
    # is chosen so that 0.04 (simulated regulation-tie rate) * 0.05 ~= 0.002,
    # matching the real empirical rate (see models/vegas.py's
    # ConstantTieModel, fit at ~0.0020-0.0022 on real data).
    ot_still_tied_rate: float = 0.05


def _drive_points(draws: np.ndarray, rng: np.random.Generator, config: DriveSimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Convert a (n_sims, n_drives) array of category-index draws into
    (offense_points, defense_bonus_points), both same shape. defense_bonus
    covers pick-six/fumble-six returns and safeties — points that go to the
    team *without* the ball on that drive."""
    xp_made = rng.random(draws.shape) < config.xp_success_rate
    offense_points = np.where(draws == _TD_IDX, 6 + xp_made.astype(int), np.where(draws == _FG_IDX, 3, 0))

    is_turnover_live = np.isin(draws, _TURNOVER_IDXS)
    xp_made_defense = rng.random(draws.shape) < config.xp_success_rate
    return_td = is_turnover_live & (rng.random(draws.shape) < config.return_td_given_turnover_rate)
    is_safety = draws == _SAFETY_IDX

    defense_bonus = np.where(return_td, 6 + xp_made_defense.astype(int), np.where(is_safety, 2, 0))
    return offense_points, defense_bonus


def _resolve_win_probabilities(home_scores: np.ndarray, away_scores: np.ndarray, ot_still_tied_rate: float) -> tuple[float, float, float]:
    """Turn raw regulation-equivalent scores into final win/loss/tie
    probabilities, modeling the fact that real NFL overtime resolves almost
    all regulation ties into a winner. The tied mass is split between home
    and away in proportion to each side's *decisive* win share (a proxy for
    who would tend to prevail if the drive-exchange continued), and only a
    small residual stays tied — see DriveSimConfig.ot_still_tied_rate.
    """
    decisive_home = float(np.mean(home_scores > away_scores))
    decisive_away = float(np.mean(home_scores < away_scores))
    tied_frac = float(np.mean(home_scores == away_scores))

    decisive_total = decisive_home + decisive_away
    if decisive_total <= 0:
        # degenerate case (e.g. n_drives so small every sim ties) — split evenly
        home_share, away_share = 0.5, 0.5
    else:
        home_share = decisive_home / decisive_total
        away_share = decisive_away / decisive_total

    resolved_mass = tied_frac * (1 - ot_still_tied_rate)
    final_home = decisive_home + resolved_mass * home_share
    final_away = decisive_away + resolved_mass * away_share
    final_tie = tied_frac * ot_still_tied_rate
    return final_home, final_away, final_tie


@dataclass(frozen=True)
class SimResult:
    home_win_prob: float
    away_win_prob: float
    tie_prob: float
    mean_home_score: float
    mean_away_score: float
    simulated_spread: float  # mean(home_score - away_score); positive favors home
    simulated_total: float
    home_scores: np.ndarray
    away_scores: np.ndarray


def simulate_game(
    home_matchup_rates: dict[str, float],
    away_matchup_rates: dict[str, float],
    n_drives: int,
    config: DriveSimConfig | None = None,
) -> SimResult:
    """Simulate `config.n_simulations` full games, each team getting
    `n_drives` possessions (a documented simplification: real teams' drive
    counts differ by at most 1 and depend on clock/pace interaction this
    simulator does not model — see features/drive_rates.py's
    drives_per_game docstring for how a single shared n_drives is derived).
    """
    config = config or DriveSimConfig()
    if n_drives < 1:
        raise ValueError(f"n_drives must be >= 1, got {n_drives}")

    rng = np.random.default_rng(config.seed)
    categories = OUTCOME_CATEGORIES
    home_p = np.array([home_matchup_rates[c] for c in categories])
    away_p = np.array([away_matchup_rates[c] for c in categories])
    home_p = home_p / home_p.sum()
    away_p = away_p / away_p.sum()

    home_draws = rng.choice(len(categories), size=(config.n_simulations, n_drives), p=home_p)
    away_draws = rng.choice(len(categories), size=(config.n_simulations, n_drives), p=away_p)

    home_off_pts, away_def_bonus = _drive_points(home_draws, rng, config)  # away is on defense during home's drives
    away_off_pts, home_def_bonus = _drive_points(away_draws, rng, config)  # home is on defense during away's drives

    home_scores = home_off_pts.sum(axis=1) + home_def_bonus.sum(axis=1)
    away_scores = away_off_pts.sum(axis=1) + away_def_bonus.sum(axis=1)

    home_win_prob, away_win_prob, tie_prob = _resolve_win_probabilities(home_scores, away_scores, config.ot_still_tied_rate)

    return SimResult(
        home_win_prob=home_win_prob,
        away_win_prob=away_win_prob,
        tie_prob=tie_prob,
        mean_home_score=float(home_scores.mean()),
        mean_away_score=float(away_scores.mean()),
        simulated_spread=float((home_scores - away_scores).mean()),
        simulated_total=float((home_scores + away_scores).mean()),
        home_scores=home_scores,
        away_scores=away_scores,
    )


def combined_n_drives(home_drives_per_game: float, away_drives_per_game: float, league_avg_drives_per_game: float) -> int:
    """Both teams get the same number of drives in this simulator (strict
    alternation — see simulate_game's docstring), so their individual pace
    tendencies are combined into one shared number: each team's
    drives-per-game relative to league average contributes a symmetric
    factor, since both share the same game clock.
    """
    if league_avg_drives_per_game <= 0:
        return round(home_drives_per_game)
    home_factor = home_drives_per_game / league_avg_drives_per_game
    away_factor = away_drives_per_game / league_avg_drives_per_game
    combined = league_avg_drives_per_game * (home_factor + away_factor) / 2.0
    return max(1, round(combined))
