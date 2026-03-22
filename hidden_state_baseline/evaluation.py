from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .kalman_filter import KalmanFilterResults


def _safe_correlation(actual: np.ndarray, estimated: np.ndarray) -> float:
    if actual.std() <= 0.0 or estimated.std() <= 0.0:
        return 1.0
    return float(np.corrcoef(actual, estimated)[0, 1])


def evaluate_filter_performance(
    true_states: np.ndarray,
    filter_results: KalmanFilterResults,
    label: str,
    state_names: Sequence[str],
) -> tuple[dict[str, float | str | dict], pd.DataFrame]:
    actual_states = np.asarray(true_states, dtype=float)
    if actual_states.ndim != 2:
        raise ValueError("true_states must be a two-dimensional array of shape (T, state_dim).")

    filtered_means = np.asarray(filter_results.filtered_means, dtype=float)
    filtered_variances = np.maximum(
        np.diagonal(filter_results.filtered_covariances, axis1=1, axis2=2),
        0.0,
    )
    filtered_stds = np.sqrt(filtered_variances)
    lower_95 = filtered_means - 1.96 * filtered_stds
    upper_95 = filtered_means + 1.96 * filtered_stds
    errors = filtered_means - actual_states

    state_metrics: dict[str, dict[str, float]] = {}
    frame_dict: dict[str, np.ndarray] = {}
    for index, name in enumerate(state_names):
        actual = actual_states[:, index]
        filtered = filtered_means[:, index]
        filtered_std = filtered_stds[:, index]
        lower = lower_95[:, index]
        upper = upper_95[:, index]
        error = errors[:, index]

        state_metrics[name] = {
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "correlation": _safe_correlation(actual, filtered),
            "mae": float(np.mean(np.abs(error))),
            "mean_confidence_band_width": float(np.mean(upper - lower)),
            "coverage_95": float(np.mean((actual >= lower) & (actual <= upper))),
        }
        frame_dict[f"filtered_{name}"] = filtered
        frame_dict[f"filtered_std_{name}"] = filtered_std
        frame_dict[f"lower_95_{name}"] = lower
        frame_dict[f"upper_95_{name}"] = upper
        frame_dict[f"filter_error_{name}"] = error

    diagnostics: dict[str, float | str | dict] = {
        "scenario": label,
        "aggregate_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "aggregate_mae": float(np.mean(np.abs(errors))),
        "mean_confidence_band_width": float(np.mean(upper_95 - lower_95)),
        "mean_coverage_95": float(np.mean((actual_states >= lower_95) & (actual_states <= upper_95))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "log_likelihood": float(filter_results.log_likelihood),
        "state_metrics": state_metrics,
    }

    frame = pd.DataFrame(frame_dict)
    return diagnostics, frame
