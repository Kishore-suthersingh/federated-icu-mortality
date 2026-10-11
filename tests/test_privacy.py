"""Tests for the DP machinery.

A privacy bug is the worst kind: it fails silently and the paper ships a claim
that is simply false. These test the properties that would be violated.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import privacy  # noqa: E402


# --- accountant ------------------------------------------------------

def test_epsilon_decreases_with_more_noise():
    """More noise must buy more privacy. If not, calibration is meaningless."""
    def eps(sigma):
        acc = privacy.RDPAccountant()
        acc.step(q=0.1, noise_multiplier=sigma, count=10)
        return acc.get_epsilon(1e-5)

    assert eps(0.5) > eps(1.0) > eps(2.0) > eps(4.0)


def test_epsilon_increases_with_composition():
    """Repeated release must cost more budget."""
    def eps(steps):
        acc = privacy.RDPAccountant()
        acc.step(q=0.1, noise_multiplier=1.0, count=steps)
        return acc.get_epsilon(1e-5)

    assert eps(1) < eps(10) < eps(100)


def test_zero_noise_gives_no_guarantee():
    acc = privacy.RDPAccountant()
    acc.step(q=1.0, noise_multiplier=0.0, count=1)
    assert np.isinf(acc.get_epsilon(1e-5))


def test_no_sampling_costs_nothing():
    assert privacy.rdp_sampled_gaussian(q=0.0, noise_multiplier=1.0, alpha=10) == 0.0


def test_subsampling_amplifies_privacy():
    """Lower participation must yield a smaller epsilon at equal noise."""
    def eps(q):
        acc = privacy.RDPAccountant()
        acc.step(q=q, noise_multiplier=1.0, count=10)
        return acc.get_epsilon(1e-5)

    assert eps(0.01) < eps(0.1) < eps(1.0)


def test_full_participation_matches_plain_gaussian():
    """At q=1 the sampled mechanism reduces to alpha / (2 sigma^2)."""
    sigma, alpha = 2.0, 8.0
    assert privacy.rdp_sampled_gaussian(1.0, sigma, alpha) == pytest.approx(
        alpha / (2 * sigma**2)
    )


def test_calibration_achieves_target_epsilon():
    """The calibrated sigma must actually deliver the requested budget."""
    for target in (0.5, 1.0, 3.0, 10.0):
        sigma = privacy.calibrate_noise_multiplier(
            target_epsilon=target, delta=1e-5, q=0.5, steps=30
        )
        acc = privacy.RDPAccountant()
        acc.step(q=0.5, noise_multiplier=sigma, count=30)
        achieved = acc.get_epsilon(1e-5)
        assert achieved <= target + 1e-2, f"target {target}, achieved {achieved}"


def test_tighter_budget_needs_more_noise():
    def sigma_for(eps):
        return privacy.calibrate_noise_multiplier(eps, 1e-5, q=0.5, steps=30)

    assert sigma_for(0.5) > sigma_for(1.0) > sigma_for(10.0)


def test_calibration_raises_on_impossible_target():
    with pytest.raises(ValueError):
        privacy.calibrate_noise_multiplier(
            target_epsilon=1e-9, delta=1e-10, q=1.0, steps=10_000, max_sigma=100.0
        )


# --- clipping --------------------------------------------------------

def test_clip_enforces_norm_bound():
    """The whole DP guarantee rests on this bound holding."""
    update = OrderedDict({"a": torch.tensor([3.0, 4.0]),      # norm 5
                          "b": torch.tensor([[12.0, 0.0]])})   # -> total 13
    clipped, norm = privacy.clip_update(update, max_norm=1.0)

    assert norm == pytest.approx(13.0)
    total = sum(float(v.pow(2).sum()) for v in clipped.values()) ** 0.5
    assert total <= 1.0 + 1e-6


def test_clip_leaves_small_updates_untouched():
    update = OrderedDict({"a": torch.tensor([0.3, 0.4])})  # norm 0.5
    clipped, norm = privacy.clip_update(update, max_norm=1.0)
    assert norm == pytest.approx(0.5)
    assert torch.allclose(clipped["a"], update["a"])


def test_clip_preserves_direction():
    """Clipping must rescale, not rotate."""
    update = OrderedDict({"a": torch.tensor([3.0, 4.0])})
    clipped, _ = privacy.clip_update(update, max_norm=1.0)
    assert torch.allclose(clipped["a"], torch.tensor([0.6, 0.8]), atol=1e-6)


# --- aggregation -----------------------------------------------------

def test_dp_aggregation_without_noise_is_mean_update():
    global_state = OrderedDict({"w": torch.zeros(3)})
    updates = [
        OrderedDict({"w": torch.tensor([3.0, 3.0, 3.0])}),
        OrderedDict({"w": torch.tensor([1.0, 1.0, 1.0])}),
    ]
    result = privacy.aggregate_dp(global_state, updates,
                                  clip_norm=100.0, noise_multiplier=0.0)
    assert torch.allclose(result["w"], torch.tensor([2.0, 2.0, 2.0]), atol=1e-6)


def test_dp_aggregation_actually_adds_noise():
    """Two runs with different RNG state must differ — else no noise is applied."""
    global_state = OrderedDict({"w": torch.zeros(100)})
    updates = [OrderedDict({"w": torch.ones(100)})]

    a = privacy.aggregate_dp(global_state, updates, clip_norm=1.0,
                             noise_multiplier=1.0,
                             generator=torch.Generator().manual_seed(1))
    b = privacy.aggregate_dp(global_state, updates, clip_norm=1.0,
                             noise_multiplier=1.0,
                             generator=torch.Generator().manual_seed(2))
    assert not torch.allclose(a["w"], b["w"])


def test_dp_aggregation_noise_scales_with_multiplier():
    global_state = OrderedDict({"w": torch.zeros(5000)})
    updates = [OrderedDict({"w": torch.zeros(5000)})]

    def spread(sigma):
        out = privacy.aggregate_dp(
            global_state, updates, clip_norm=1.0, noise_multiplier=sigma,
            generator=torch.Generator().manual_seed(0))
        return float(out["w"].std())

    assert spread(4.0) > spread(1.0) > spread(0.25)


def test_dp_aggregation_rejects_empty():
    with pytest.raises(ValueError):
        privacy.aggregate_dp(OrderedDict({"w": torch.zeros(3)}), [],
                             clip_norm=1.0, noise_multiplier=1.0)


def test_compute_update_is_difference():
    global_state = OrderedDict({"w": torch.tensor([1.0, 2.0])})
    local_state = OrderedDict({"w": torch.tensor([4.0, 6.0])})
    delta = privacy.compute_update(global_state, local_state)
    assert torch.allclose(delta["w"], torch.tensor([3.0, 4.0]))


def test_integer_buffers_survive_dp_aggregation():
    global_state = OrderedDict({"n": torch.tensor(3, dtype=torch.long)})
    updates = [OrderedDict({"n": torch.tensor(0, dtype=torch.long)})]
    result = privacy.aggregate_dp(global_state, updates,
                                  clip_norm=1.0, noise_multiplier=1.0)
    assert result["n"].dtype == torch.long
    assert int(result["n"]) == 3


# --- embedding mechanism ---------------------------------------------

def test_embeddings_are_clipped_to_the_ball():
    rng = np.random.default_rng(0)
    Z = rng.normal(0, 10, size=(200, 16))
    out = privacy.privatize_embeddings(Z, clip_norm=1.0,
                                       noise_multiplier=0.0, rng=rng)
    assert np.linalg.norm(out, axis=1).max() <= 1.0 + 1e-5


def test_embedding_noise_increases_distortion():
    rng = np.random.default_rng(0)
    Z = rng.normal(0, 0.1, size=(500, 16))

    def distortion(sigma):
        out = privacy.privatize_embeddings(
            Z, clip_norm=1.0, noise_multiplier=sigma,
            rng=np.random.default_rng(1))
        return float(np.mean((out - Z) ** 2))

    assert distortion(1.0) > distortion(0.1) > distortion(0.0)


# --- adaptive clipping -----------------------------------------------

def test_warm_start_matches_the_target_quantile():
    """A fixed clip of 1.0 against norms of ~1000 is what destroyed training."""
    clipper = privacy.AdaptiveClipper(initial=1.0, target_quantile=0.5, noise=0.0)
    norms = [100.0, 200.0, 300.0, 400.0, 500.0]
    assert clipper.warm_start(norms) == pytest.approx(300.0)


def test_clip_grows_when_most_clients_are_clipped():
    """Too much clipping must push the bound up, not leave it stuck."""
    clipper = privacy.AdaptiveClipper(initial=1.0, target_quantile=0.5, noise=0.0)
    before = clipper.clip_norm
    clipper.update([50.0] * 10)  # every norm far above the bound
    assert clipper.clip_norm > before


def test_clip_shrinks_when_nothing_is_clipped():
    clipper = privacy.AdaptiveClipper(initial=100.0, target_quantile=0.5, noise=0.0)
    before = clipper.clip_norm
    clipper.update([0.01] * 10)  # every norm far below the bound
    assert clipper.clip_norm < before


def test_clip_converges_toward_the_median():
    """Repeated updates should track the true median of the norms."""
    rng = np.random.default_rng(0)
    norms = list(rng.normal(10.0, 1.0, size=200))
    clipper = privacy.AdaptiveClipper(initial=0.5, target_quantile=0.5,
                                      lr=0.5, noise=0.0)
    for _ in range(200):
        clipper.update(norms)
    assert clipper.clip_norm == pytest.approx(np.median(norms), rel=0.25)


def test_clipper_handles_empty_round():
    clipper = privacy.AdaptiveClipper(initial=2.0, noise=0.0)
    assert clipper.update([]) == 2.0


def test_clipper_stays_positive_under_extremes():
    clipper = privacy.AdaptiveClipper(initial=1.0, target_quantile=0.5, noise=0.0)
    for _ in range(500):
        clipper.update([1e-12] * 5)
    assert clipper.clip_norm > 0


# --- ledger ----------------------------------------------------------

def test_ledger_reports_disabled_honestly():
    summary = privacy.PrivacyLedger(privacy.DPConfig(enabled=False)).summary()
    assert summary["enabled"] is False
    assert "locality" in summary["guarantee"].lower()


def test_ledger_reports_both_channels_separately():
    cfg = privacy.DPConfig(enabled=True, noise_multiplier=1.0,
                           embedding_noise_multiplier=1.0)
    ledger = privacy.PrivacyLedger(cfg)
    ledger.record_round(q=0.5, noise_multiplier=1.0)
    ledger.record_embedding_release(q=1.0, noise_multiplier=1.0)

    summary = ledger.summary()
    assert "channel_1_model_updates" in summary
    assert "channel_2_embeddings" in summary
    # The two channels protect different units and must say so.
    assert summary["channel_1_model_updates"]["unit_of_privacy"] != \
           summary["channel_2_embeddings"]["unit_of_privacy"]
    assert np.isfinite(summary["channel_1_model_updates"]["epsilon"])


def test_ledger_warns_when_embeddings_unprotected():
    cfg = privacy.DPConfig(enabled=True, noise_multiplier=1.0,
                           protect_embeddings=False)
    summary = privacy.PrivacyLedger(cfg).summary()
    assert summary["channel_2_embeddings"]["protected"] is False
    assert "warning" in summary["channel_2_embeddings"]
