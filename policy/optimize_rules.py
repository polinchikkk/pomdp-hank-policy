from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy.inference import paired_bootstrap_ci, summarize_paired_inference


@dataclass(frozen=True)
class PairwiseComparison:
    left: str
    right: str
    num_trajectories: int
    mean_delta: float
    median_delta: float
    ci_low: float
    ci_high: float
    permutation_p_value: float
    sign_flip_p_value: float
    win_rate: float
    tie_rate: float
    loss_rate: float


def bootstrap_interval(values: np.ndarray, *, seed: int = 2027, draws: int = 2000) -> tuple[float, float]:
    return paired_bootstrap_ci(values, seed=seed, n_boot=draws)


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
    inference = summarize_paired_inference(delta, n_boot=2_000, n_perm=4_000, tie_eps=tie_eps)
    return PairwiseComparison(
        left=left_name,
        right=right_name,
        num_trajectories=inference.num_observations,
        mean_delta=inference.mean_delta,
        median_delta=inference.median_delta,
        ci_low=inference.bootstrap_ci_low,
        ci_high=inference.bootstrap_ci_high,
        permutation_p_value=inference.permutation_p_value,
        sign_flip_p_value=inference.sign_flip_p_value,
        win_rate=inference.win_rate,
        tie_rate=inference.tie_rate,
        loss_rate=inference.loss_rate,
    )
