from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_space import LinearGaussianStateSpaceModel


@dataclass(frozen=True)
class KalmanFilterResults:
    predicted_means: np.ndarray
    predicted_covariances: np.ndarray
    filtered_means: np.ndarray
    filtered_covariances: np.ndarray
    innovations: np.ndarray
    innovation_covariances: np.ndarray
    kalman_gains: np.ndarray
    log_likelihood: float


def run_kalman_filter(
    model: LinearGaussianStateSpaceModel,
    observations: np.ndarray,
    initial_mean: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
) -> KalmanFilterResults:
    observations = np.asarray(observations, dtype=float)
    num_periods = observations.shape[0]
    state_dim = model.transition_matrix.shape[0]
    obs_dim = model.observation_matrix.shape[0]

    if initial_mean is None:
        initial_mean = model.initial_state_mean()
    if initial_covariance is None:
        initial_covariance = model.stationary_state_covariance()

    predicted_means = np.zeros((num_periods, state_dim), dtype=float)
    predicted_covariances = np.zeros((num_periods, state_dim, state_dim), dtype=float)
    filtered_means = np.zeros((num_periods, state_dim), dtype=float)
    filtered_covariances = np.zeros((num_periods, state_dim, state_dim), dtype=float)
    innovations = np.zeros((num_periods, obs_dim), dtype=float)
    innovation_covariances = np.zeros((num_periods, obs_dim, obs_dim), dtype=float)
    kalman_gains = np.zeros((num_periods, state_dim, obs_dim), dtype=float)

    predicted_mean = np.asarray(initial_mean, dtype=float).reshape(state_dim)
    predicted_covariance = np.asarray(initial_covariance, dtype=float).reshape(state_dim, state_dim)
    identity = np.eye(state_dim, dtype=float)
    log_likelihood = 0.0

    for period in range(num_periods):
        predicted_means[period] = predicted_mean
        predicted_covariances[period] = predicted_covariance

        innovation = observations[period] - model.observation_matrix @ predicted_mean
        innovation_covariance = (
            model.observation_matrix @ predicted_covariance @ model.observation_matrix.T
            + model.measurement_noise_cov
        )
        innovation_covariance_inv = np.linalg.inv(innovation_covariance)
        kalman_gain = predicted_covariance @ model.observation_matrix.T @ innovation_covariance_inv

        filtered_mean = predicted_mean + kalman_gain @ innovation
        filtered_covariance = (
            (identity - kalman_gain @ model.observation_matrix)
            @ predicted_covariance
            @ (identity - kalman_gain @ model.observation_matrix).T
            + kalman_gain @ model.measurement_noise_cov @ kalman_gain.T
        )

        innovations[period] = innovation
        innovation_covariances[period] = innovation_covariance
        kalman_gains[period] = kalman_gain
        filtered_means[period] = filtered_mean
        filtered_covariances[period] = filtered_covariance

        sign, logdet = np.linalg.slogdet(innovation_covariance)
        if sign <= 0:
            raise RuntimeError("Innovation covariance is not positive definite.")
        log_likelihood += -0.5 * (
            obs_dim * np.log(2.0 * np.pi)
            + logdet
            + innovation.T @ innovation_covariance_inv @ innovation
        )

        predicted_mean = model.transition_matrix @ filtered_mean
        predicted_covariance = (
            model.transition_matrix @ filtered_covariance @ model.transition_matrix.T
            + model.process_noise_cov
        )

    return KalmanFilterResults(
        predicted_means=predicted_means,
        predicted_covariances=predicted_covariances,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        innovations=innovations,
        innovation_covariances=innovation_covariances,
        kalman_gains=kalman_gains,
        log_likelihood=float(log_likelihood),
    )
