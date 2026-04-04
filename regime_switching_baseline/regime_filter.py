from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regime_model import RegimeSwitchingModel


@dataclass(frozen=True)
class SwitchingKalmanFilterResults:
    predicted_mode_probabilities: np.ndarray
    filtered_mode_probabilities: np.ndarray
    predicted_means: np.ndarray
    predicted_covariances: np.ndarray
    filtered_means: np.ndarray
    filtered_covariances: np.ndarray
    regime_conditioned_means: np.ndarray
    regime_conditioned_covariances: np.ndarray
    innovations: np.ndarray
    innovation_covariances: np.ndarray
    log_likelihood: float


def _stable_innovation_terms(
    innovation_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    candidate = 0.5 * (innovation_covariance + innovation_covariance.T)
    jitter = 1.0e-12
    sign, logdet = np.linalg.slogdet(candidate)
    while sign <= 0:
        candidate = candidate + jitter * np.eye(candidate.shape[0], dtype=float)
        jitter *= 10.0
        sign, logdet = np.linalg.slogdet(candidate)
        if jitter > 1.0:
            raise RuntimeError("Innovation covariance is not positive definite.")
    inverse = np.linalg.inv(candidate)
    return candidate, inverse, float(logdet)


def _imm_update_step(
    *,
    model: RegimeSwitchingModel,
    previous_mode_probabilities: np.ndarray,
    previous_mode_means: np.ndarray,
    previous_mode_covariances: np.ndarray,
    observation: np.ndarray,
    observation_matrix: np.ndarray,
    measurement_covariance: np.ndarray,
    previous_control: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    num_regimes = model.num_regimes()
    state_dim = len(model.state_names)
    identity = np.eye(state_dim, dtype=float)

    predicted_mode_probabilities = previous_mode_probabilities @ model.regime_transition_matrix
    predicted_mode_probabilities = np.clip(predicted_mode_probabilities, 1.0e-12, None)
    predicted_mode_probabilities = predicted_mode_probabilities / np.sum(predicted_mode_probabilities)

    regime_conditioned_means = np.zeros((num_regimes, state_dim), dtype=float)
    regime_conditioned_covariances = np.zeros((num_regimes, state_dim, state_dim), dtype=float)
    predicted_means = np.zeros_like(regime_conditioned_means)
    predicted_covariances = np.zeros_like(regime_conditioned_covariances)
    innovations = np.zeros((num_regimes, observation.shape[0]), dtype=float)
    innovation_covariances = np.zeros((num_regimes, observation.shape[0], observation.shape[0]), dtype=float)
    mode_log_likelihoods = np.zeros((num_regimes,), dtype=float)

    for regime_index in range(num_regimes):
        mixing_denominator = predicted_mode_probabilities[regime_index]
        if mixing_denominator <= 0.0:
            mixing_weights = np.full((num_regimes,), 1.0 / num_regimes, dtype=float)
        else:
            mixing_weights = (
                model.regime_transition_matrix[:, regime_index] * previous_mode_probabilities / mixing_denominator
            )
            mixing_weights = np.clip(mixing_weights, 1.0e-12, None)
            mixing_weights = mixing_weights / np.sum(mixing_weights)

        mixed_mean = np.sum(mixing_weights[:, None] * previous_mode_means, axis=0)
        mixed_covariance = np.zeros((state_dim, state_dim), dtype=float)
        for source_regime in range(num_regimes):
            gap = previous_mode_means[source_regime] - mixed_mean
            mixed_covariance += mixing_weights[source_regime] * (
                previous_mode_covariances[source_regime] + np.outer(gap, gap)
            )

        transition = model.transition_matrices[regime_index]
        control = model.control_loadings[regime_index]
        process_noise = model.process_noise_covariances[regime_index]

        predicted_mean = transition @ mixed_mean + control * previous_control
        predicted_covariance = transition @ mixed_covariance @ transition.T + process_noise
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)

        innovation = observation - observation_matrix @ predicted_mean
        innovation_covariance, innovation_precision, logdet = _stable_innovation_terms(
            observation_matrix @ predicted_covariance @ observation_matrix.T + measurement_covariance
        )
        kalman_gain = predicted_covariance @ observation_matrix.T @ innovation_precision
        filtered_mean = predicted_mean + kalman_gain @ innovation
        filtered_covariance = (
            (identity - kalman_gain @ observation_matrix)
            @ predicted_covariance
            @ (identity - kalman_gain @ observation_matrix).T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        filtered_covariance = 0.5 * (filtered_covariance + filtered_covariance.T)

        regime_conditioned_means[regime_index] = filtered_mean
        regime_conditioned_covariances[regime_index] = filtered_covariance
        predicted_means[regime_index] = predicted_mean
        predicted_covariances[regime_index] = predicted_covariance
        innovations[regime_index] = innovation
        innovation_covariances[regime_index] = innovation_covariance
        mode_log_likelihoods[regime_index] = -0.5 * (
            observation.shape[0] * np.log(2.0 * np.pi)
            + logdet
            + innovation.T @ innovation_precision @ innovation
        )

    joint_log_weights = np.log(predicted_mode_probabilities) + mode_log_likelihoods
    max_log_weight = float(np.max(joint_log_weights))
    normalized_weights = np.exp(joint_log_weights - max_log_weight)
    filtered_mode_probabilities = normalized_weights / np.sum(normalized_weights)

    combined_mean = np.sum(filtered_mode_probabilities[:, None] * regime_conditioned_means, axis=0)
    combined_covariance = np.zeros((state_dim, state_dim), dtype=float)
    for regime_index in range(num_regimes):
        gap = regime_conditioned_means[regime_index] - combined_mean
        combined_covariance += filtered_mode_probabilities[regime_index] * (
            regime_conditioned_covariances[regime_index] + np.outer(gap, gap)
        )

    log_likelihood_increment = max_log_weight + np.log(np.sum(normalized_weights))
    return (
        predicted_mode_probabilities,
        filtered_mode_probabilities,
        predicted_means,
        predicted_covariances,
        regime_conditioned_means,
        regime_conditioned_covariances,
        combined_mean,
        combined_covariance,
        innovations,
        innovation_covariances,
        float(log_likelihood_increment),
    )


def run_switching_kalman_filter(
    *,
    model: RegimeSwitchingModel,
    observations: np.ndarray,
    observation_names: tuple[str, ...],
    measurement_covariance: np.ndarray,
    control_path: np.ndarray,
) -> SwitchingKalmanFilterResults:
    num_periods = observations.shape[0]
    num_regimes = model.num_regimes()
    state_dim = len(model.state_names)
    observation_matrix = model.observation_matrix[[model.observation_index(name) for name in observation_names]]

    mode_probabilities = model.stationary_regime_distribution()
    mode_means = np.repeat(model.initial_state_mean()[None, :], num_regimes, axis=0)
    mode_covariances = model.stationary_state_covariances()

    predicted_mode_probabilities = np.zeros((num_periods, num_regimes), dtype=float)
    filtered_mode_probabilities = np.zeros_like(predicted_mode_probabilities)
    predicted_means = np.zeros((num_periods, num_regimes, state_dim), dtype=float)
    predicted_covariances = np.zeros((num_periods, num_regimes, state_dim, state_dim), dtype=float)
    filtered_means = np.zeros((num_periods, state_dim), dtype=float)
    filtered_covariances = np.zeros((num_periods, state_dim, state_dim), dtype=float)
    regime_conditioned_means = np.zeros((num_periods, num_regimes, state_dim), dtype=float)
    regime_conditioned_covariances = np.zeros((num_periods, num_regimes, state_dim, state_dim), dtype=float)
    innovations = np.zeros((num_periods, num_regimes, observations.shape[1]), dtype=float)
    innovation_covariances = np.zeros((num_periods, num_regimes, observations.shape[1], observations.shape[1]), dtype=float)
    log_likelihood = 0.0

    previous_control = 0.0
    for period in range(num_periods):
        (
            predicted_mode_probabilities[period],
            filtered_mode_probabilities[period],
            predicted_means[period],
            predicted_covariances[period],
            regime_conditioned_means[period],
            regime_conditioned_covariances[period],
            filtered_means[period],
            filtered_covariances[period],
            innovations[period],
            innovation_covariances[period],
            increment,
        ) = _imm_update_step(
            model=model,
            previous_mode_probabilities=mode_probabilities,
            previous_mode_means=mode_means,
            previous_mode_covariances=mode_covariances,
            observation=observations[period],
            observation_matrix=observation_matrix,
            measurement_covariance=measurement_covariance,
            previous_control=previous_control,
        )
        log_likelihood += increment
        mode_probabilities = filtered_mode_probabilities[period]
        mode_means = regime_conditioned_means[period]
        mode_covariances = regime_conditioned_covariances[period]
        previous_control = control_path[period]

    return SwitchingKalmanFilterResults(
        predicted_mode_probabilities=predicted_mode_probabilities,
        filtered_mode_probabilities=filtered_mode_probabilities,
        predicted_means=predicted_means,
        predicted_covariances=predicted_covariances,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        regime_conditioned_means=regime_conditioned_means,
        regime_conditioned_covariances=regime_conditioned_covariances,
        innovations=innovations,
        innovation_covariances=innovation_covariances,
        log_likelihood=float(log_likelihood),
    )
