"""Reliability diagrams and confidence-band slicing (plan §3).

A model can have a perfectly respectable overall Brier score while being
badly miscalibrated exactly where survivor picks live — the 0.85-0.95
"safe favorite" band — because that band is a small fraction of all games.
`high_confidence_slice` exists specifically to catch that failure mode
rather than letting it hide inside an aggregate number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from survivor.evaluate.metrics import brier_score, log_loss


@dataclass(frozen=True)
class ReliabilityBin:
    bin_low: float
    bin_high: float
    n: int
    mean_predicted: float
    empirical_rate: float

    @property
    def gap(self) -> float:
        """Positive means the model is overconfident in this bin (predicted
        higher than what actually happened); negative means underconfident."""
        return self.mean_predicted - self.empirical_rate


def reliability_curve(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> list[ReliabilityBin]:
    """Bin predictions into `n_bins` equal-width bins over [0, 1] and compare
    mean predicted probability to empirical event rate within each bin.
    Empty bins are omitted rather than reported as zero/NaN."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin includes the right edge (prob == 1.0)
        mask = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bins.append(
            ReliabilityBin(
                bin_low=float(lo),
                bin_high=float(hi),
                n=n,
                mean_predicted=float(prob[mask].mean()),
                empirical_rate=float(outcome[mask].mean()),
            )
        )
    return bins


@dataclass(frozen=True)
class ConfidenceSliceResult:
    threshold_low: float
    threshold_high: float
    n: int
    brier: float | None
    log_loss: float | None
    empirical_win_rate: float | None
    mean_predicted: float | None


def high_confidence_slice(
    prob: np.ndarray, outcome: np.ndarray, low: float = 0.85, high: float = 0.99
) -> ConfidenceSliceResult:
    """Metrics restricted to predictions in [low, high] — the confidence
    band survivor-pool favorites actually live in. A model that is well
    calibrated overall but overconfident here would recommend picks that
    survive less often than the number on the board claims."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    mask = (prob >= low) & (prob <= high)
    n = int(mask.sum())
    if n == 0:
        return ConfidenceSliceResult(low, high, 0, None, None, None, None)
    return ConfidenceSliceResult(
        threshold_low=low,
        threshold_high=high,
        n=n,
        brier=brier_score(prob[mask], outcome[mask]),
        log_loss=log_loss(prob[mask], outcome[mask]),
        empirical_win_rate=float(outcome[mask].mean()),
        mean_predicted=float(prob[mask].mean()),
    )
