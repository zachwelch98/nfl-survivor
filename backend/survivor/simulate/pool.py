"""Pool-field simulation (plan §5): P(sole survivor) / P(win the pool),
not just P(you personally survive) — the objective that actually matters
in a real pool, since maximizing your own survival probability is not the
same as maximizing your chance of *winning*, which depends on the field.

No public API publishes pick-popularity data (SurvivorGrid/PoolGenius show
it on-page only), so it enters as a hand-maintained CSV at
data/popularity/{season}_wk{NN}.csv — one row per team with a `pick_pct`
column (need not sum to 100 across teams if only partial data is
available; it's renormalized). Every function here degrades gracefully to
returning None / skipping the pool-relative comparison when no popularity
data exists for a given week, per the plan's explicit design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from survivor.config import CONFIG


def popularity_csv_path(season: int, week: int) -> Path:
    return CONFIG.paths.resolve("popularity") / f"{season}_wk{week:02d}.csv"


def load_popularity(season: int, week: int) -> dict[str, float] | None:
    """{team: pick_pct in [0,1]}, renormalized to sum to 1 across teams
    with data. Returns None if the file doesn't exist (graceful
    degradation — see module docstring)."""
    path = popularity_csv_path(season, week)
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if "team" not in df.columns or "pick_pct" not in df.columns:
        raise ValueError(f"{path} must have 'team' and 'pick_pct' columns")
    total = df["pick_pct"].sum()
    if total <= 0:
        return None
    return dict(zip(df["team"].to_list(), (df["pick_pct"] / total).to_list()))


@dataclass(frozen=True)
class PoolSimResult:
    n_field_entries: int
    p_sole_survivor: float  # you survive AND every field entry is eliminated by the same week (or earlier)
    p_share_win: float  # you survive AND at least one field entry also survives to the same final week
    p_you_survive: float  # your own survival probability (same as the non-pool metric, for comparison)
    field_median_week_eliminated: float


def simulate_pool(
    your_pick_survival_probs: dict[int, float],  # week -> P(your pick that week survives)
    field_popularity_by_week: dict[int, dict[str, float]],  # week -> {team: pick_pct}
    game_survival_probs: dict[int, dict[str, float]],  # week -> {team: P(that team wins/survives that week)}
    weeks: list[int],
    n_field_entries: int = 10000,
    seed: int = 0,
) -> PoolSimResult:
    """Simulate the field alongside your own picks. Each field entry, each
    week it's still alive, picks a team by sampling from that week's
    popularity distribution (a simplification: real entries don't repick
    teams they've already used, but with no visibility into any individual
    entry's history, sampling i.i.d. from the aggregate weekly distribution
    is the standard tractable approximation used by public pick-percentage
    tools) — then survives or not per that team's real survival probability.
    Your own outcome is sampled independently from `your_pick_survival_probs`.
    """
    rng = np.random.default_rng(seed)
    field_alive = np.ones(n_field_entries, dtype=bool)
    field_eliminated_week = np.full(n_field_entries, -1)

    you_alive_scalar = True
    for week in weeks:
        if you_alive_scalar:
            p_you = your_pick_survival_probs.get(week)
            if p_you is not None and rng.random() >= p_you:
                you_alive_scalar = False

        pop = field_popularity_by_week.get(week)
        survive = game_survival_probs.get(week)
        if pop and survive:
            teams = list(pop.keys())
            probs = np.array([pop[t] for t in teams])
            probs = probs / probs.sum()
            still_alive_idx = np.where(field_alive)[0]
            if len(still_alive_idx) > 0:
                picks = rng.choice(len(teams), size=len(still_alive_idx), p=probs)
                survival_by_team = np.array([survive.get(teams[i], 0.5) for i in picks])
                outcomes = rng.random(len(still_alive_idx)) < survival_by_team
                newly_eliminated = still_alive_idx[~outcomes]
                field_alive[newly_eliminated] = False
                field_eliminated_week[newly_eliminated] = week

    n_still_alive_at_end = int(field_alive.sum())
    p_you_survive = 1.0 if you_alive_scalar else 0.0

    result_sole = you_alive_scalar and n_still_alive_at_end == 0
    result_share = you_alive_scalar and n_still_alive_at_end > 0

    return PoolSimResult(
        n_field_entries=n_field_entries,
        p_sole_survivor=float(result_sole),
        p_share_win=float(result_share),
        p_you_survive=p_you_survive,
        field_median_week_eliminated=float(np.median(np.where(field_eliminated_week < 0, weeks[-1], field_eliminated_week))),
    )


def simulate_pool_many_trials(
    your_pick_survival_probs: dict[int, float],
    field_popularity_by_week: dict[int, dict[str, float]],
    game_survival_probs: dict[int, dict[str, float]],
    weeks: list[int],
    n_trials: int = 2000,
    n_field_entries: int = 10000,
    seed: int = 0,
) -> dict:
    """Repeats simulate_pool across many independent trials (each trial
    resampling both your own outcome and the whole field) to get stable
    P(sole survivor) / P(share win) estimates rather than a single 0/1 draw.
    """
    sole = np.zeros(n_trials)
    share = np.zeros(n_trials)
    you_survive = np.zeros(n_trials)
    for t in range(n_trials):
        r = simulate_pool(your_pick_survival_probs, field_popularity_by_week, game_survival_probs, weeks, n_field_entries, seed=seed + t)
        sole[t] = r.p_sole_survivor
        share[t] = r.p_share_win
        you_survive[t] = r.p_you_survive
    return {
        "p_sole_survivor": float(sole.mean()),
        "p_share_win": float(share.mean()),
        "p_you_survive": float(you_survive.mean()),
        "n_trials": n_trials,
        "n_field_entries": n_field_entries,
    }
