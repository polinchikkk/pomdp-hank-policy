from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearFilterState:
    mean: np.ndarray
    covariance: np.ndarray


def kalman_update(
    *,
    predicted_mean: np.ndarray,
    predicted_covariance: np.ndarray,
    observation: np.ndarray,
    observation_matrix: np.ndarray,
    observation_covariance: np.ndarray,
) -> LinearFilterState:
    predicted_mean = np.asarray(predicted_mean, dtype=float)
    predicted_covariance = np.asarray(predicted_covariance, dtype=float)
    observation = np.asarray(observation, dtype=float)
    observation_matrix = np.asarray(observation_matrix, dtype=float)
    observation_covariance = np.asarray(observation_covariance, dtype=float)

    innovation = observation - observation_matrix @ predicted_mean
    innovation_covariance = observation_matrix @ predicted_covariance @ observation_matrix.T + observation_covariance
    gain = predicted_covariance @ observation_matrix.T @ np.linalg.pinv(innovation_covariance)
    mean = predicted_mean + gain @ innovation
    covariance = predicted_covariance - gain @ observation_matrix @ predicted_covariance
    covariance = 0.5 * (covariance + covariance.T)
    return LinearFilterState(mean=mean, covariance=covariance)
