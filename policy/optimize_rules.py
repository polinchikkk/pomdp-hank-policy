from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairwiseComparison:
    left: str
    right: str
    num_trajectories: int
    mean_delta: float
    ci_low: float
    ci_high: float
    win_rate: float
    tie_rate: float
    loss_rate: float


def bootstrap_interval(values: np.ndarray, *, seed: int = 2027, draws: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def compare_paired_losses(
    *,
    left_name: str,
    right_name: str,
    left_losses: np.ndarray,
    right_losses: np.ndarray,
    tie_eps: float = 1e-12,
) -> PairwiseComparison:
    left_losses = np.asarray(left_losses, dtype=float)
    right_losses = np.asarray(right_losses, dtype=float)
    if left_losses.shape != right_losses.shape:
        raise ValueError("Paired losses must have the same shape.")
    delta = left_losses - right_losses
    ci_low, ci_high = bootstrap_interval(delta)
    return PairwiseComparison(
        left=left_name,
        right=right_name,
        num_trajectories=int(delta.size),
        mean_delta=float(np.mean(delta)),
        ci_low=ci_low,
        ci_high=ci_high,
        win_rate=float(np.mean(delta < -tie_eps)),
        tie_rate=float(np.mean(np.abs(delta) <= tie_eps)),
        loss_rate=float(np.mean(delta > tie_eps)),
    )
