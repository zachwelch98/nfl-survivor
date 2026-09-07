"""Tests for models/blend.py: the fitted ensemble over N named methods."""

from __future__ import annotations

import numpy as np
import pytest

from survivor.models._prob_utils import logit_array as _logit_array
from survivor.models.blend import BlendModel


def test_logit_array_round_trips_through_sigmoid():
    p = np.array([0.1, 0.5, 0.9])
    logit = _logit_array(p)
    sigmoid_back = 1 / (1 + np.exp(-logit))
    np.testing.assert_allclose(sigmoid_back, p, atol=1e-5)


def test_blend_requires_minimum_sample_size():
    with pytest.raises(ValueError):
        BlendModel().fit({"vegas": np.array([0.6, 0.4]), "mc": np.array([0.55, 0.45])}, np.array([1, 0]))


def test_blend_recovers_better_input_when_one_source_is_pure_noise():
    """If Vegas is genuinely predictive (with realistic irreducible outcome
    randomness) and Monte Carlo is pure noise, the fitted blend should end
    up close to Vegas alone, not average them toward the noise source."""
    rng = np.random.default_rng(0)
    n = 3000
    true_prob = rng.uniform(0.2, 0.8, n)
    outcome = (rng.uniform(0, 1, n) < true_prob).astype(int)  # genuine irreducible randomness
    vegas = np.clip(true_prob + rng.normal(0, 0.03, n), 0.02, 0.98)  # well-calibrated, small noise
    mc_noise = np.clip(rng.uniform(0.3, 0.7, n), 0.02, 0.98)  # uncorrelated with outcome

    model = BlendModel().fit({"vegas": vegas, "mc": mc_noise}, outcome)
    blended = model.predict({"vegas": vegas, "mc": mc_noise})

    from survivor.evaluate.metrics import brier_score

    bs_blend = brier_score(blended, outcome.astype(float))
    bs_vegas = brier_score(vegas, outcome.astype(float))
    bs_noise = brier_score(mc_noise, outcome.astype(float))
    assert bs_blend < bs_noise
    assert bs_blend == pytest.approx(bs_vegas, abs=0.01)


def test_blend_output_is_valid_probability():
    rng = np.random.default_rng(1)
    n = 100
    outcome = rng.integers(0, 2, n)
    vegas = np.clip(outcome * 0.7 + (1 - outcome) * 0.3 + rng.normal(0, 0.05, n), 0.05, 0.95)
    mc = np.clip(outcome * 0.65 + (1 - outcome) * 0.35 + rng.normal(0, 0.08, n), 0.05, 0.95)
    model = BlendModel().fit({"vegas": vegas, "mc": mc}, outcome)
    preds = model.predict({"vegas": vegas, "mc": mc})
    assert np.all((preds >= 0) & (preds <= 1))


def test_blend_generalizes_to_three_named_inputs():
    """The three-way case Method 3 needs: two informative inputs plus one
    noise source, and the blend should still track the informative ones."""
    rng = np.random.default_rng(7)
    n = 3000
    true_prob = rng.uniform(0.2, 0.8, n)
    outcome = (rng.uniform(0, 1, n) < true_prob).astype(int)
    vegas = np.clip(true_prob + rng.normal(0, 0.03, n), 0.02, 0.98)
    ml = np.clip(true_prob + rng.normal(0, 0.06, n), 0.02, 0.98)
    mc_noise = np.clip(rng.uniform(0.3, 0.7, n), 0.02, 0.98)

    inputs = {"vegas": vegas, "mc": mc_noise, "ml": ml}
    model = BlendModel().fit(inputs, outcome)
    blended = model.predict(inputs)

    from survivor.evaluate.metrics import brier_score

    assert model.input_keys == ("vegas", "mc", "ml")
    assert np.all((blended >= 0) & (blended <= 1))
    assert brier_score(blended, outcome.astype(float)) < brier_score(mc_noise, outcome.astype(float))


def test_blend_predict_rejects_a_different_input_set():
    """With N unlabeled stacked columns, a swapped or renamed input would
    otherwise produce plausible-looking garbage instead of an error."""
    rng = np.random.default_rng(2)
    n = 200
    outcome = rng.integers(0, 2, n)
    a, b = rng.uniform(0.2, 0.8, n), rng.uniform(0.2, 0.8, n)
    model = BlendModel().fit({"vegas": a, "mc": b}, outcome)

    with pytest.raises(ValueError, match="fit on inputs"):
        model.predict({"vegas": a, "ml": b})
    with pytest.raises(ValueError, match="fit on inputs"):
        model.predict({"vegas": a, "mc": b, "ml": b})


def test_blend_minimum_sample_size_scales_with_input_count():
    rng = np.random.default_rng(3)
    n = 50  # enough for 2 inputs (40), not for 3 (60)
    outcome = rng.integers(0, 2, n)
    arrays = {k: rng.uniform(0.2, 0.8, n) for k in ("vegas", "mc", "ml")}

    BlendModel().fit({"vegas": arrays["vegas"], "mc": arrays["mc"]}, outcome)  # fine
    with pytest.raises(ValueError, match="at least 60"):
        BlendModel().fit(arrays, outcome)
