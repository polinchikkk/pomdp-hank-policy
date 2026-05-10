from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class InferenceSummary:
    num_observations: int
    num_clusters: int
    mean_delta: float
    median_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    clustered_se: float
    clustered_ci_low: float
    clustered_ci_high: float
    wild_ci_low: float
    wild_ci_high: float
    permutation_p_value: float
    sign_flip_p_value: float
    win_rate: float
    tie_rate: float
    loss_rate: float


def paired_bootstrap_ci(
    delta: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 2027,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = _clean_1d(delta)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[draws].mean(axis=1)
    return _quantile_interval(means, alpha=alpha)


def cluster_bootstrap_ci(
    delta: np.ndarray,
    cluster_id: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 2027,
    alpha: float = 0.05,
) -> tuple[float, float]:
    cluster_means = _cluster_means(delta, cluster_id)
    if cluster_means.size == 0:
        return float("nan"), float("nan")
    if cluster_means.size == 1:
        value = float(cluster_means[0])
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, cluster_means.size, size=(int(n_boot), cluster_means.size))
    means = cluster_means[draws].mean(axis=1)
    return _quantile_interval(means, alpha=alpha)


def sign_flip_test(
    delta: np.ndarray,
    *,
    n_perm: int = 10_000,
    seed: int = 2027,
    alternative: str = "two-sided",
) -> float:
    values = _clean_1d(delta)
    if values.size == 0:
        return float("nan")
    observed = float(values.mean())
    if np.allclose(values, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), values.size), replace=True)
    simulated = (signs * values).mean(axis=1)
    return _tail_probability(simulated, observed, alternative=alternative)


def wild_bootstrap_ci(
    delta: np.ndarray,
    cluster_id: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 2027,
    alpha: float = 0.05,
) -> tuple[float, float]:
    cluster_means = _cluster_means(delta, cluster_id)
    if cluster_means.size == 0:
        return float("nan"), float("nan")
    if cluster_means.size == 1:
        value = float(cluster_means[0])
        return value, value
    observed = float(cluster_means.mean())
    centered = cluster_means - observed
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_boot), cluster_means.size), replace=True)
    boot = observed + (signs * centered).mean(axis=1)
    return _quantile_interval(boot, alpha=alpha)


def bh_adjust_pvalues(pvalues: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if finite.size == 0:
        return adjusted
    order = np.argsort(finite)
    ranked = finite[order]
    m = finite.size
    raw = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    clipped = np.clip(monotone, 0.0, 1.0)
    restored = np.empty_like(clipped)
    restored[order] = clipped
    adjusted[finite_mask] = restored
    return adjusted


def paired_permutation_test(
    delta: np.ndarray,
    *,
    n_perm: int = 10_000,
    seed: int = 2027,
    alternative: str = "two-sided",
) -> float:
    return sign_flip_test(delta, n_perm=n_perm, seed=seed, alternative=alternative)


def summarize_paired_inference(
    delta: np.ndarray,
    *,
    cluster_id: np.ndarray | None = None,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 2027,
    tie_eps: float = 1e-12,
) -> InferenceSummary:
    values = _clean_1d(delta)
    if cluster_id is None:
        clusters = np.arange(values.size)
    else:
        clusters = np.asarray(cluster_id)
        if clusters.shape[0] != np.asarray(delta).shape[0]:
            raise ValueError("cluster_id must have the same length as delta.")
        mask = np.isfinite(np.asarray(delta, dtype=float))
        clusters = clusters[mask]
    boot_low, boot_high = paired_bootstrap_ci(values, n_boot=n_boot, seed=seed)
    cluster_low, cluster_high = cluster_bootstrap_ci(values, clusters, n_boot=n_boot, seed=seed + 1)
    wild_low, wild_high = wild_bootstrap_ci(values, clusters, n_boot=n_boot, seed=seed + 2)
    cluster_means = _cluster_means(values, clusters)
    clustered_se = _clustered_standard_error(cluster_means)
    permutation_p = paired_permutation_test(values, n_perm=n_perm, seed=seed + 3)
    sign_flip_p = sign_flip_test(cluster_means, n_perm=n_perm, seed=seed + 4)
    return InferenceSummary(
        num_observations=int(values.size),
        num_clusters=int(cluster_means.size),
        mean_delta=float(values.mean()) if values.size else float("nan"),
        median_delta=float(np.median(values)) if values.size else float("nan"),
        bootstrap_ci_low=boot_low,
        bootstrap_ci_high=boot_high,
        clustered_se=clustered_se,
        clustered_ci_low=cluster_low,
        clustered_ci_high=cluster_high,
        wild_ci_low=wild_low,
        wild_ci_high=wild_high,
        permutation_p_value=permutation_p,
        sign_flip_p_value=sign_flip_p,
        win_rate=float(np.mean(values < -tie_eps)) if values.size else float("nan"),
        tie_rate=float(np.mean(np.abs(values) <= tie_eps)) if values.size else float("nan"),
        loss_rate=float(np.mean(values > tie_eps)) if values.size else float("nan"),
    )


def _clean_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _cluster_means(delta: np.ndarray, cluster_id: np.ndarray) -> np.ndarray:
    values = np.asarray(delta, dtype=float).reshape(-1)
    clusters = np.asarray(cluster_id).reshape(-1)
    if values.shape[0] != clusters.shape[0]:
        raise ValueError("delta and cluster_id must have the same length.")
    mask = np.isfinite(values)
    values = values[mask]
    clusters = clusters[mask]
    if values.size == 0:
        return np.asarray([], dtype=float)
    unique = np.unique(clusters)
    return np.asarray([values[clusters == item].mean() for item in unique], dtype=float)


def _clustered_standard_error(cluster_means: np.ndarray) -> float:
    values = np.asarray(cluster_means, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))


def _quantile_interval(values: np.ndarray, *, alpha: float) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _tail_probability(simulated: np.ndarray, observed: float, *, alternative: str) -> float:
    if alternative == "two-sided":
        return float((np.sum(np.abs(simulated) >= abs(observed)) + 1.0) / (simulated.size + 1.0))
    if alternative == "less":
        return float((np.sum(simulated <= observed) + 1.0) / (simulated.size + 1.0))
    if alternative == "greater":
        return float((np.sum(simulated >= observed) + 1.0) / (simulated.size + 1.0))
    raise ValueError("alternative must be 'two-sided', 'less', or 'greater'.")
