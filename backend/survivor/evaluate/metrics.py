"""Scoring rules for comparing win-probability models (plan §3).

The whole point of this module is to let the data decide whether the
Monte Carlo challenger actually beats the Vegas baseline — Vegas closing
lines are a strong, efficient-market baseline, so "beats it" must mean the
bootstrap CI on the metric difference excludes zero, not merely that the
point estimate looks better on one sample.

All functions here operate on a single decisive-outcome probability (e.g.
P(home win), among home/away, ties excluded or handled by the caller) and a
matching binary outcome array — they are deliberately generic rather than
tied to any one model's output shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12  # clip to avoid log(0) for a probability of exactly 0 or 1


def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error between predicted probability and {0,1} outcome.
    Lower is better; 0 is perfect, 0.25 is what an uninformative p=0.5 gets
    on a 50/50 sample, 1.0 is maximally wrong."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    return float(np.mean((prob - outcome) ** 2))


def log_loss(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean negative log-likelihood. Lower is better; punishes confident
    wrong calls much more severely than Brier does."""
    prob = np.clip(np.asarray(prob, dtype=float), _EPS, 1 - _EPS)
    outcome = np.asarray(outcome, dtype=float)
    return float(-np.mean(outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)))


def brier_skill_score(prob: np.ndarray, outcome: np.ndarray, reference_prob: np.ndarray) -> float:
    """Brier Skill Score of `prob` relative to `reference_prob` (typically Vegas).

    BSS = 1 - BS(model) / BS(reference). Positive means `prob` beats the
    reference; negative means the reference is better; 0 means a tie. This
    is the metric that actually answers "is the challenger better than
    Vegas", not the raw Brier scores in isolation (which don't have an
    intrinsic scale for "how much better").
    """
    bs_model = brier_score(prob, outcome)
    bs_ref = brier_score(reference_prob, outcome)
    if bs_ref == 0:
        return 0.0 if bs_model == 0 else -np.inf
    return 1.0 - bs_model / bs_ref


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_boot: int

    @property
    def excludes_zero(self) -> bool:
        """True iff the CI does not straddle 0 — the bar for "genuinely different",
        per plan §3 ("treat the challenger as better only if the CI on the
        difference excludes zero — not on the point estimate")."""
        return self.ci_low > 0 or self.ci_high < 0


def bootstrap_metric_diff(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    outcome: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap CI for metric_fn(prob_a, outcome) - metric_fn(prob_b, outcome).

    Resamples games (rows) with replacement so the CI reflects sampling
    uncertainty over *games*, which is the right resampling unit here (not
    resampling probabilities and outcomes independently, which would break
    the pairing between a model's prediction and what actually happened).

    Positive point_estimate means A scores *lower* (better, for Brier/log
    loss) than B when metric_fn is one of those — callers comparing "is the
    challenger better than Vegas" should pass (challenger, vegas) as (a, b)
    and interpret a positive, zero-excluding result as "challenger wins".
    """
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    n = len(outcome)
    if not (len(prob_a) == len(prob_b) == n):
        raise ValueError("prob_a, prob_b, outcome must be the same length")

    point = metric_fn(prob_b, outcome) - metric_fn(prob_a, outcome)  # b - a: positive = a better (lower)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric_fn(prob_b[idx], outcome[idx]) - metric_fn(prob_a[idx], outcome[idx])

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    return BootstrapResult(point_estimate=float(point), ci_low=float(lo), ci_high=float(hi), n_boot=n_boot)


def metrics_by_group(prob: np.ndarray, outcome: np.ndarray, groups: np.ndarray) -> dict:
    """Brier/log loss/n broken out by an arbitrary group label array (e.g.
    week buckets 'wk1-4', 'wk5-9', ...). Used to check whether a model's
    edge (or Vegas's) is concentrated in a particular part of the season —
    plan §3 expects Vegas to dominate early weeks and the gap to narrow
    later, which argues for a week-dependent blend rather than one global
    winner."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        out[g] = {
            "n": int(mask.sum()),
            "brier": brier_score(prob[mask], outcome[mask]),
            "log_loss": log_loss(prob[mask], outcome[mask]),
        }
    return out
