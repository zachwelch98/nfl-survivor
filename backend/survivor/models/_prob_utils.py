"""Logit/sigmoid helpers shared by the models.

These existed twice already — scalar versions in montecarlo.py (for the
log5 category combination) and an array version in blend.py (for the
logistic stack) — and Method 3's future-week blend in optimize/projection.py
would have made three. One definition, one clipping epsilon, one place to
change it.

The clip matters: an un-clipped `logit(0.0)` or `logit(1.0)` is infinite,
which silently turns a downstream mean or regression coefficient into NaN.
Every entry point here clips first.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def logit(p: float) -> float:
    """Scalar log-odds, clipped away from 0 and 1."""
    p = min(max(p, EPS), 1 - EPS)
    return float(np.log(p / (1 - p)))


def sigmoid(x: float) -> float:
    """Scalar inverse of `logit`."""
    return float(1.0 / (1.0 + np.exp(-x)))


def logit_array(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Vectorized log-odds.

    Defaults to a wider epsilon (1e-6) than the scalar path because its
    consumer is a fitted `LogisticRegression`: a 1e-9 clip puts an extreme
    input around ±20.7 in logit space, which is enough to dominate a
    stacking coefficient. This is the value blend.py has always used.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid_array(x: np.ndarray) -> np.ndarray:
    """Vectorized inverse of `logit_array`."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def blend_logit_mean(probs: list[np.ndarray] | np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    """Weighted average of probabilities in log-odds space.

    Averaging in logit space rather than probability space is the right
    default for combining two models that disagree: the arithmetic mean of
    0.95 and 0.55 is 0.75, which understates a genuine consensus that the
    home team is favored, while the logit mean (~0.86) preserves the
    multiplicative structure odds actually have. Used by
    optimize/projection.py for the fixed Monte-Carlo/ML future-week blend,
    where no fitted weighting has earned its place yet.
    """
    stacked = np.column_stack([logit_array(p) for p in probs])
    if weights is None:
        combined = stacked.mean(axis=1)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != stacked.shape[1]:
            raise ValueError(f"got {len(w)} weights for {stacked.shape[1]} probability arrays")
        combined = (stacked * w).sum(axis=1) / w.sum()
    return sigmoid_array(combined)
