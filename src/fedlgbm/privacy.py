"""Differential privacy for the federated hybrid.

This addresses the reviewers' central objection: the manuscript asserted
"privacy-preserving" but implemented nothing beyond data locality.

THE TWO-CHANNEL PROBLEM
-----------------------
The hybrid architecture leaks through two distinct channels, and protecting one
does not protect the other. This is the point the paper should make, because it
is specific to hybrid neural/tree federated designs rather than generic FL:

  Channel 1 — model updates. Each round every client uploads MLP parameters.
              Protected by user-level DP-FedAvg (McMahan et al., 2018): clip
              each client's update to L2 norm C, add Gaussian noise at the
              server, so the aggregate is insensitive to any single client.

  Channel 2 — embeddings. To train the central LightGBM, clients upload an
              embedding *per record*. This is a per-example channel, not a
              per-client one, and DP-FedAvg says nothing about it. Standard FL
              papers have no analogue because the whole model normally stays
              distributed. Protected here by a separate Gaussian mechanism on
              bounded embeddings.

Reporting a single epsilon for the system would be wrong. The two mechanisms
protect different units (client vs. record), so they are accounted separately
and reported as a pair. Composing them into one number would require a common
unit of privacy that does not exist here.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# RDP accountant for the sampled Gaussian mechanism
# --------------------------------------------------------------------

# Standard order grid; dense at low orders where tight bounds usually sit.
DEFAULT_ORDERS: tuple[float, ...] = (
    tuple(1.0 + x / 10.0 for x in range(1, 100))
    + tuple(range(11, 64))
    + (128.0, 256.0, 512.0)
)


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    lo, hi = min(a, b), max(a, b)
    return hi + math.log1p(math.exp(lo - hi))


def rdp_sampled_gaussian(q: float, noise_multiplier: float, alpha: float) -> float:
    """RDP of the sampled Gaussian mechanism at order alpha.

    For integer alpha, uses the exact binomial expansion of Mironov et al.
    (2019), "Renyi Differential Privacy of the Sampled Gaussian Mechanism":

        eps(alpha) = 1/(alpha-1) * log( sum_k C(alpha,k) (1-q)^(alpha-k) q^k
                                        exp(k(k-1) / (2 sigma^2)) )

    Non-integer orders are conservatively bounded by the next integer up, which
    over-reports epsilon slightly. Erring toward a larger reported epsilon is
    the safe direction for a privacy claim.
    """
    if q == 0.0:
        return 0.0
    if noise_multiplier <= 0:
        return np.inf
    if q == 1.0:
        # Full participation: plain Gaussian mechanism.
        return alpha / (2.0 * noise_multiplier**2)

    int_alpha = int(math.ceil(alpha))
    if int_alpha <= 1:
        return 0.0

    log_sum = -np.inf
    for k in range(int_alpha + 1):
        term = (
            _log_comb(int_alpha, k)
            + (int_alpha - k) * math.log1p(-q)
            + k * math.log(q)
            + (k * k - k) / (2.0 * noise_multiplier**2)
        )
        log_sum = _log_add(log_sum, term)

    return max(log_sum / (int_alpha - 1), 0.0)


def rdp_to_dp(
    rdp: np.ndarray, orders: np.ndarray, delta: float
) -> tuple[float, float]:
    """Convert RDP guarantees to (epsilon, delta)-DP.

    Uses the improved conversion of Balle et al. (2020), which is tighter than
    the classical eps = rdp + log(1/delta)/(alpha-1), especially at small delta.
    Returns the best epsilon and the order that achieved it.
    """
    orders = np.atleast_1d(orders)
    rdp = np.atleast_1d(rdp)

    valid = orders > 1
    orders, rdp = orders[valid], rdp[valid]
    if len(orders) == 0:
        return np.inf, np.nan

    eps = (
        rdp
        + np.log1p(-1.0 / orders)
        - (np.log(delta) + np.log(orders)) / (orders - 1.0)
    )
    # Guard against numerical negatives at extreme orders.
    eps = np.where(np.isfinite(eps), eps, np.inf)
    best = int(np.nanargmin(eps))
    return float(max(eps[best], 0.0)), float(orders[best])


class RDPAccountant:
    """Accumulates RDP across composed mechanism invocations."""

    def __init__(self, orders: tuple[float, ...] = DEFAULT_ORDERS) -> None:
        self.orders = np.array(orders, dtype=np.float64)
        self.rdp = np.zeros_like(self.orders)
        self.steps = 0

    def step(self, q: float, noise_multiplier: float, count: int = 1) -> None:
        """Compose `count` invocations at sampling rate q."""
        added = np.array(
            [rdp_sampled_gaussian(q, noise_multiplier, a) for a in self.orders]
        )
        self.rdp += added * count
        self.steps += count

    def get_epsilon(self, delta: float) -> float:
        return rdp_to_dp(self.rdp, self.orders, delta)[0]

    def summary(self, delta: float) -> dict:
        eps, order = rdp_to_dp(self.rdp, self.orders, delta)
        return {
            "epsilon": eps,
            "delta": delta,
            "optimal_rdp_order": order,
            "composed_steps": self.steps,
        }


def calibrate_noise_multiplier(
    target_epsilon: float,
    delta: float,
    q: float,
    steps: int,
    tolerance: float = 1e-3,
    max_sigma: float = 1e4,
) -> float:
    """Binary-search the smallest sigma meeting a target epsilon.

    Lets experiments be specified as "epsilon = 1" rather than as an opaque
    noise multiplier, which is how privacy budgets are reported in papers.
    """
    def epsilon_at(sigma: float) -> float:
        acc = RDPAccountant()
        acc.step(q, sigma, steps)
        return acc.get_epsilon(delta)

    lo, hi = 1e-3, 1.0
    while epsilon_at(hi) > target_epsilon:
        hi *= 2.0
        if hi > max_sigma:
            raise ValueError(
                f"Cannot reach epsilon={target_epsilon} with sigma<={max_sigma}. "
                "Reduce steps/rounds, lower the sampling rate, or relax epsilon."
            )

    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if epsilon_at(mid) > target_epsilon:
            lo = mid
        else:
            hi = mid

    logger.info(
        "Calibrated sigma=%.4f for (eps=%.2f, delta=%.1e, q=%.3f, steps=%d)",
        hi, target_epsilon, delta, q, steps,
    )
    return hi


# --------------------------------------------------------------------
# Channel 1: user-level DP on model updates
# --------------------------------------------------------------------

@dataclass
class DPConfig:
    """Differential privacy settings for both channels."""

    enabled: bool = False

    # --- Channel 1: model updates (user-level DP-FedAvg) ---
    clip_norm: float = 1.0

    # Adaptive clipping (Andrew et al., 2021). A fixed clip norm is extremely
    # brittle: this model's raw update norms are O(10^3), so clipping to 1.0
    # scales the signal down ~1000x and leaves the aggregate dominated by
    # noise, which makes training diverge. Adaptive clipping tracks a quantile
    # of the observed norms instead, keeping roughly `clip_quantile` of updates
    # unclipped regardless of the model's scale.
    adaptive_clipping: bool = True
    clip_quantile: float = 0.5
    clip_lr: float = 0.2
    # Noise on the quantile statistic. This is a private estimate, so it draws
    # from the same budget; it is small relative to the update noise but must
    # be disclosed rather than treated as free.
    clip_quantile_noise: float = 0.01
    # Set either noise_multiplier directly, or target_epsilon to calibrate it.
    noise_multiplier: float | None = None
    target_epsilon: float | None = 3.0
    delta: float = 1e-5

    # --- Channel 2: embeddings (record-level DP) ---
    protect_embeddings: bool = True
    embedding_clip_norm: float = 1.0
    embedding_noise_multiplier: float | None = None
    embedding_target_epsilon: float | None = 3.0
    embedding_delta: float = 1e-5


def clip_update(
    update: OrderedDict, max_norm: float
) -> tuple[OrderedDict, float]:
    """Clip a client's parameter delta to L2 norm `max_norm`.

    Clipping bounds each client's sensitivity, which is what makes the noise
    scale calibratable. Returns the clipped update and its original norm (worth
    logging: if most norms sit far below the clip, the noise is larger than
    necessary and utility is being wasted).
    """
    squared = 0.0
    for value in update.values():
        if torch.is_floating_point(value):
            squared += float(value.pow(2).sum())
    norm = math.sqrt(squared)

    factor = min(1.0, max_norm / (norm + 1e-12))
    clipped = OrderedDict(
        (k, v * factor if torch.is_floating_point(v) else v.clone())
        for k, v in update.items()
    )
    return clipped, norm


class AdaptiveClipper:
    """Tracks a target quantile of client update norms (Andrew et al., 2021).

    Multiplicative update:

        C_{t+1} = C_t * exp(-eta * (frac_below_t - target_quantile))

    where frac_below_t is the noised fraction of clients whose update norm fell
    below the current clip. If too many clients are being clipped the bound
    grows; if too few, it shrinks. This keeps clipping proportionate to the
    model's actual update scale instead of to an arbitrary constant.

    The fraction is a data-dependent statistic, so it is released with noise.
    Its privacy cost is small but real and is disclosed in the ledger.
    """

    def __init__(
        self,
        initial: float,
        target_quantile: float = 0.5,
        lr: float = 0.2,
        noise: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.clip_norm = float(initial)
        self.target_quantile = target_quantile
        self.lr = lr
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    def update(self, observed_norms: list[float]) -> float:
        """Consume one round's norms and return the next clip bound."""
        if not observed_norms:
            return self.clip_norm

        norms = np.asarray(observed_norms, dtype=np.float64)
        fraction_below = float((norms <= self.clip_norm).mean())

        if self.noise > 0:
            fraction_below += self.rng.normal(0.0, self.noise)
            fraction_below = float(np.clip(fraction_below, 0.0, 1.0))

        previous = self.clip_norm
        self.clip_norm *= math.exp(
            -self.lr * (fraction_below - self.target_quantile)
        )
        # Keep the bound in a sane range; runaway growth or collapse is worse
        # than a slightly mis-set constant.
        self.clip_norm = float(np.clip(self.clip_norm, 1e-6, 1e9))

        self.history.append({
            "clip_norm_before": previous,
            "clip_norm_after": self.clip_norm,
            "noised_fraction_below": fraction_below,
            "median_observed_norm": float(np.median(norms)),
        })
        return self.clip_norm

    def warm_start(self, observed_norms: list[float]) -> float:
        """Initialise the bound directly from the first round's norms.

        Starting at the target quantile avoids wasting many rounds while the
        multiplicative update crawls from an arbitrary initial value up to the
        right order of magnitude.
        """
        if observed_norms:
            self.clip_norm = float(
                max(np.quantile(observed_norms, self.target_quantile), 1e-6)
            )
            logger.info(
                "Adaptive clipping warm-started at C=%.4g (median observed norm %.4g)",
                self.clip_norm, float(np.median(observed_norms)),
            )
        return self.clip_norm


def compute_update(
    global_state: OrderedDict, local_state: OrderedDict
) -> OrderedDict:
    """Client delta: local parameters minus the global starting point.

    DP-FedAvg noises the *delta*, not the raw weights, because the delta is
    what has bounded sensitivity to one client's data.
    """
    return OrderedDict(
        (k, local_state[k] - global_state[k]
            if torch.is_floating_point(local_state[k])
            else local_state[k].clone())
        for k in global_state
    )


def aggregate_dp(
    global_state: OrderedDict,
    client_updates: list[OrderedDict],
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> OrderedDict:
    """DP-FedAvg aggregation (McMahan et al., 2018).

        w_{t+1} = w_t + (1/m) * ( sum_k clip(delta_k, C) + N(0, sigma^2 C^2 I) )

    Note this is a *uniform* average, not sample-count-weighted as in plain
    FedAvg (Eq. 6). Weighting by n_k would make sensitivity depend on client
    sizes and break the noise calibration. The paper must state this: enabling
    DP changes the aggregation rule, it does not merely add noise to it.
    """
    m = len(client_updates)
    if m == 0:
        raise ValueError("No client updates to aggregate")

    std = noise_multiplier * clip_norm

    new_state = OrderedDict()
    for key in global_state:
        base = global_state[key]
        if not torch.is_floating_point(base):
            new_state[key] = base.clone()
            continue

        summed = torch.zeros_like(base, dtype=torch.float32)
        for update in client_updates:
            summed += update[key].to(torch.float32)

        if std > 0:
            noise = torch.normal(
                mean=0.0, std=std, size=summed.shape,
                generator=generator, device=summed.device,
            )
            summed += noise

        new_state[key] = (base.to(torch.float32) + summed / m).to(base.dtype)

    return new_state


# --------------------------------------------------------------------
# Channel 2: record-level DP on embeddings
# --------------------------------------------------------------------

def privatize_embeddings(
    Z: np.ndarray,
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the Gaussian mechanism to per-record embeddings.

    Each embedding is first projected onto the L2 ball of radius `clip_norm`,
    bounding one record's contribution, then perturbed with
    N(0, (noise_multiplier * clip_norm)^2).

    The unit of privacy here is one *record*, not one client. That is a weaker
    guarantee than the user-level DP on channel 1 and must not be conflated
    with it in the write-up.
    """
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / (norms + 1e-12))
    clipped = Z * scale

    std = noise_multiplier * clip_norm
    if std <= 0:
        return clipped.astype(np.float32)

    noise = rng.normal(0.0, std, size=clipped.shape)
    return (clipped + noise).astype(np.float32)


class PrivacyLedger:
    """Tracks both channels and reports them separately.

    Deliberately refuses to emit a single combined epsilon — see module
    docstring for why that number would be meaningless.
    """

    def __init__(self, cfg: DPConfig) -> None:
        self.cfg = cfg
        self.update_accountant = RDPAccountant()
        self.embedding_accountant = RDPAccountant()
        self.clip_norms_observed: list[float] = []
        self.clipper: "AdaptiveClipper | None" = None

    def record_round(self, q: float, noise_multiplier: float) -> None:
        self.update_accountant.step(q, noise_multiplier)

    def record_embedding_release(self, q: float, noise_multiplier: float) -> None:
        self.embedding_accountant.step(q, noise_multiplier)

    def summary(self) -> dict:
        if not self.cfg.enabled:
            return {
                "enabled": False,
                "guarantee": (
                    "Data locality only. No formal DP guarantee: raw records "
                    "stay on-device, but model updates and per-record "
                    "embeddings are released without calibrated noise."
                ),
            }

        out: dict = {
            "enabled": True,
            "channel_1_model_updates": {
                "unit_of_privacy": "client (user-level DP)",
                "mechanism": "DP-FedAvg: L2 clipping + Gaussian noise",
                "clip_norm": self.cfg.clip_norm,
                "noise_multiplier": self.cfg.noise_multiplier,
                **self.update_accountant.summary(self.cfg.delta),
            },
        }

        if self.clipper is not None:
            final_clip = self.clipper.clip_norm
            out["channel_1_model_updates"]["adaptive_clipping"] = {
                "enabled": True,
                "target_quantile": self.cfg.clip_quantile,
                "final_clip_norm": final_clip,
                "quantile_noise": self.cfg.clip_quantile_noise,
                "note": (
                    "The clip bound is estimated from a noised quantile of "
                    "observed update norms. This statistic is data-dependent "
                    "and carries a small additional privacy cost beyond the "
                    "reported epsilon."
                ),
            }
        else:
            final_clip = self.cfg.clip_norm
            out["channel_1_model_updates"]["adaptive_clipping"] = {"enabled": False}

        if self.clip_norms_observed:
            norms = np.array(self.clip_norms_observed)
            fraction_clipped = float((norms > final_clip).mean())
            out["channel_1_model_updates"]["observed_update_norms"] = {
                "mean": float(norms.mean()),
                "median": float(np.median(norms)),
                "p95": float(np.percentile(norms, 95)),
                "final_clip_norm": final_clip,
                "fraction_clipped": fraction_clipped,
            }
            # Near-total clipping means the signal is being crushed relative to
            # the noise; the run is likely to be useless and should say so.
            if fraction_clipped > 0.95 and np.median(norms) > 10 * final_clip:
                out["channel_1_model_updates"]["observed_update_norms"]["warning"] = (
                    "Almost every update was clipped by a large factor. The "
                    "aggregate is dominated by noise and utility is likely "
                    "destroyed. Enable adaptive clipping or raise the clip norm."
                )

        if self.cfg.protect_embeddings:
            out["channel_2_embeddings"] = {
                "unit_of_privacy": "record (per-example DP)",
                "mechanism": "L2 clipping + Gaussian noise on embeddings",
                "clip_norm": self.cfg.embedding_clip_norm,
                "noise_multiplier": self.cfg.embedding_noise_multiplier,
                **self.embedding_accountant.summary(self.cfg.embedding_delta),
            }
        else:
            out["channel_2_embeddings"] = {
                "protected": False,
                "warning": (
                    "Embeddings released without noise. User-level DP on model "
                    "updates does NOT protect this channel."
                ),
            }

        out["note"] = (
            "Epsilons are reported per channel and are NOT composable into a "
            "single value: channel 1 protects a client, channel 2 protects a "
            "record. They quantify different guarantees."
        )
        return out
