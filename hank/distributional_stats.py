from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistributionalStatistics:
    mean_mpc: float
    low_liquidity_share: float


def summarize_distributional_statistics(
    *,
    mpc: np.ndarray,
    liquid_assets: np.ndarray,
    weights: np.ndarray,
    low_liquidity_cutoff: float = 0.0,
) -> DistributionalStatistics:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    mpc = np.asarray(mpc, dtype=float)
    liquid_assets = np.asarray(liquid_assets, dtype=float)
    return DistributionalStatistics(
        mean_mpc=float(np.sum(weights * mpc)),
        low_liquidity_share=float(np.sum(weights[liquid_assets <= low_liquidity_cutoff])),
    )
