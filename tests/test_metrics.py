"""Tests for evaluate/metrics.py and evaluate/calibration.py."""

from __future__ import annotations

import numpy as np
import pytest

from survivor.evaluate.calibration import high_confidence_slice, reliability_curve
from survivor.evaluate.metrics import (
    BootstrapResult,
    bootstrap_metric_diff,
    brier_score,
    brier_skill_score,
    log_loss,
    metrics_by_group,
)


def test_brier_score_perfect_and_worst_case():
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert brier_score(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_brier_score_uninformative_half_on_fifty_fifty_sample():
    prob = np.full(100, 0.5)
    outcome = np.array([1, 0] * 50, dtype=float)
    assert brier_score(prob, outcome) == pytest.approx(0.25)


def test_log_loss_perfect_prediction_near_zero():
    prob = np.array([0.999999, 0.000001])
    outcome = np.array([1.0, 0.0])
    assert log_loss(prob, outcome) < 1e-4


def test_log_loss_confident_wrong_is_punished_hard():
    """As confidence in a wrong call approaches 1, log loss grows unboundedly
    while Brier score stays capped at 1.0 — this asymmetry is the entire
    reason to report both metrics side by side."""
    outcome_wrong = np.array([0.0])
    ll_moderate = log_loss(np.array([0.9]), outcome_wrong)
    ll_extreme = log_loss(np.array([0.999999]), outcome_wrong)
    bs_moderate = brier_score(np.array([0.9]), outcome_wrong)
    bs_extreme = brier_score(np.array([0.999999]), outcome_wrong)

    assert ll_extreme > 5 * ll_moderate  # log loss blows up as prob -> 1
    assert bs_extreme <= 1.0 and bs_moderate <= 1.0  # Brier is always bounded by 1
    assert bs_extreme / bs_moderate < 2  # Brier's growth is comparatively mild


def test_brier_skill_score_positive_when_model_beats_reference():
    outcome = np.array([1.0, 0.0, 1.0, 0.0])
    good = np.array([0.9, 0.1, 0.9, 0.1])
    bad = np.array([0.5, 0.5, 0.5, 0.5])
    bss = brier_skill_score(good, outcome, reference_prob=bad)
    assert bss > 0


def test_brier_skill_score_zero_for_identical_models():
    outcome = np.array([1.0, 0.0, 1.0, 0.0])
    prob = np.array([0.7, 0.3, 0.6, 0.4])
    assert brier_skill_score(prob, outcome, reference_prob=prob) == pytest.approx(0.0)


def test_bootstrap_metric_diff_detects_clear_winner():
    """Challenger is unambiguously better-calibrated than a coin-flip
    reference on a large decisive sample — CI should exclude zero and favor
    the challenger."""
    rng = np.random.default_rng(42)
    n = 2000
    outcome = rng.integers(0, 2, n).astype(float)
    challenger = np.clip(outcome * 0.8 + (1 - outcome) * 0.2 + rng.normal(0, 0.05, n), 0.01, 0.99)
    reference = np.full(n, 0.5)

    result = bootstrap_metric_diff(challenger, reference, outcome, metric_fn=brier_score, n_boot=500, seed=1)
    assert isinstance(result, BootstrapResult)
    assert result.point_estimate > 0  # challenger has lower (better) Brier than reference
    assert result.excludes_zero


def test_bootstrap_metric_diff_no_difference_for_identical_inputs():
    rng = np.random.default_rng(0)
    n = 500
    outcome = rng.integers(0, 2, n).astype(float)
    prob = np.clip(outcome * 0.7 + (1 - outcome) * 0.3, 0.01, 0.99)
    result = bootstrap_metric_diff(prob, prob, outcome, metric_fn=brier_score, n_boot=300, seed=2)
    assert result.point_estimate == pytest.approx(0.0)
    assert not result.excludes_zero


def test_bootstrap_metric_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bootstrap_metric_diff(np.array([0.5, 0.5]), np.array([0.5]), np.array([1.0, 0.0]), metric_fn=brier_score)


def test_metrics_by_group_splits_correctly():
    prob = np.array([0.9, 0.9, 0.5, 0.5])
    outcome = np.array([1.0, 1.0, 1.0, 0.0])
    groups = np.array(["early", "early", "late", "late"])
    out = metrics_by_group(prob, outcome, groups)
    assert set(out.keys()) == {"early", "late"}
    assert out["early"]["n"] == 2
    assert out["early"]["brier"] == pytest.approx(brier_score(np.array([0.9, 0.9]), np.array([1.0, 1.0])))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_reliability_curve_perfect_calibration():
    rng = np.random.default_rng(0)
    n = 10000
    prob = rng.uniform(0, 1, n)
    outcome = (rng.uniform(0, 1, n) < prob).astype(float)
    bins = reliability_curve(prob, outcome, n_bins=10)
    assert len(bins) == 10
    for b in bins:
        assert abs(b.gap) < 0.05  # well-calibrated within sampling noise


def test_reliability_curve_skips_empty_bins():
    prob = np.array([0.05, 0.06, 0.95, 0.96])  # nothing in the middle bins
    outcome = np.array([0.0, 0.0, 1.0, 1.0])
    bins = reliability_curve(prob, outcome, n_bins=10)
    assert len(bins) == 2  # only the two populated bins reported


def test_reliability_curve_top_bin_includes_probability_one():
    prob = np.array([1.0])
    outcome = np.array([1.0])
    bins = reliability_curve(prob, outcome, n_bins=10)
    assert len(bins) == 1
    assert bins[0].n == 1


def test_high_confidence_slice_empty_when_no_predictions_in_range():
    prob = np.array([0.1, 0.2, 0.3])
    outcome = np.array([0.0, 1.0, 0.0])
    result = high_confidence_slice(prob, outcome, low=0.85, high=0.99)
    assert result.n == 0
    assert result.brier is None


def test_high_confidence_slice_computes_metrics_on_subset_only():
    prob = np.array([0.9, 0.92, 0.5, 0.5])
    outcome = np.array([1.0, 0.0, 1.0, 0.0])
    result = high_confidence_slice(prob, outcome, low=0.85, high=0.99)
    assert result.n == 2
    assert result.empirical_win_rate == pytest.approx(0.5)  # one of the two high-conf picks lost
