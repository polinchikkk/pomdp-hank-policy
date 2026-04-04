from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hank_partial_info_baseline.config import NOISY_OBSERVATION_NAMES

from .regime_filter import SwitchingKalmanFilterResults, _imm_update_step
from .regime_model import RegimeSwitchingConfig, RegimeSwitchingModel


@dataclass(frozen=True)
class RegimePolicyRun:
    scenario_name: str
    scenario_label: str
    info_scenario_name: str
    info_scenario_label: str
    gap_name: str
    gap_label: str
    policy_name: str
    policy_label: str
    noisy_observation_names: tuple[str, ...]
    hidden_regimes: np.ndarray
    true_states: np.ndarray
    observations: np.ndarray
    all_observables: np.ndarray
    policy_rate: np.ndarray
    total_policy_shock: np.ndarray
    base_policy_shock: np.ndarray
    filtered_states: np.ndarray | None
    filtered_covariances: np.ndarray | None
    filtered_mode_probabilities: np.ndarray | None
    filter_results: SwitchingKalmanFilterResults | None


def simulate_hidden_regimes(
    model: RegimeSwitchingModel,
    horizon: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    regimes = np.zeros((horizon,), dtype=int)
    stationary = model.stationary_regime_distribution()
    regimes[0] = int(rng.choice(model.num_regimes(), p=stationary))
    for period in range(1, horizon):
        regimes[period] = int(
            rng.choice(
                model.num_regimes(),
                p=model.regime_transition_matrix[regimes[period - 1]],
            )
        )
    return regimes


def generate_regime_observations(
    model: RegimeSwitchingModel,
    states: np.ndarray,
) -> pd.DataFrame:
    rows = {"period": np.arange(states.shape[0], dtype=int)}
    for index, name in enumerate(model.state_names):
        rows[f"state_{name}"] = states[:, index]
    for obs_index, name in enumerate(model.observation_names):
        rows[f"observable_{name}"] = states @ model.observation_matrix[obs_index]
    return pd.DataFrame(rows)


def _base_policy_shock_path(config: RegimeSwitchingConfig) -> np.ndarray:
    partial = config.partial_config
    path = np.zeros(partial.horizon, dtype=float)
    path[partial.base_policy_shock_period] = partial.base_policy_shock_size
    for period in range(partial.base_policy_shock_period + 1, partial.horizon):
        path[period] = partial.base_policy_shock_persistence * path[period - 1]
    return path


def _measurement_covariance(config: RegimeSwitchingConfig, scenario: dict) -> tuple[dict[str, float], np.ndarray]:
    std_map = {
        name: config.partial_config.base_measurement_noise()[name] * scenario["noise_scale"]
        for name in scenario["noisy_observations"]
    }
    covariance = np.diag([std_map[name] ** 2 for name in scenario["noisy_observations"]]).astype(float)
    return std_map, covariance


def _simulate_state_path(
    *,
    model: RegimeSwitchingModel,
    hidden_regimes: np.ndarray,
    innovations: np.ndarray,
    total_policy_shock: np.ndarray,
) -> np.ndarray:
    horizon = len(hidden_regimes)
    state_dim = len(model.state_names)
    states = np.zeros((horizon, state_dim), dtype=float)
    for period in range(horizon):
        regime = hidden_regimes[period]
        transition = model.transition_matrices[regime]
        control = model.control_loadings[regime]
        if period == 0:
            states[period] = innovations[period]
        else:
            states[period] = (
                transition @ states[period - 1]
                + control * total_policy_shock[period - 1]
                + innovations[period]
            )
    return states


def _all_observables(model: RegimeSwitchingModel, states: np.ndarray) -> np.ndarray:
    return states @ model.observation_matrix.T


def _rule_term(states: np.ndarray, model: RegimeSwitchingModel, phi_pi: float, phi_y: float) -> np.ndarray:
    idx_rstar = model.state_index("rstar_gap")
    idx_pi = model.state_index("inflation_gap")
    idx_output = model.state_index("output_gap")
    return states[:, idx_rstar] + phi_pi * states[:, idx_pi] + phi_y * states[:, idx_output]


def _policy_from_rule_term(rule_term: np.ndarray, base_policy_shock: np.ndarray, rho_i: float) -> tuple[np.ndarray, np.ndarray]:
    policy_rate = np.zeros_like(rule_term)
    previous_rate = 0.0
    for period in range(len(rule_term)):
        policy_rate[period] = rho_i * previous_rate + (1.0 - rho_i) * rule_term[period] + base_policy_shock[period]
        previous_rate = policy_rate[period]
    return policy_rate, base_policy_shock.copy()


def simulate_full_information_policy(
    *,
    model: RegimeSwitchingModel,
    config: RegimeSwitchingConfig,
    scenario: dict,
    hidden_regimes: np.ndarray,
    innovations: np.ndarray,
    measurement_noise: np.ndarray,
    phi_pi: float,
    phi_y: float,
    rho_i: float,
) -> RegimePolicyRun:
    base_policy_shock = _base_policy_shock_path(config)
    total_policy_shock = base_policy_shock.copy()
    true_states = _simulate_state_path(
        model=model,
        hidden_regimes=hidden_regimes,
        innovations=innovations,
        total_policy_shock=total_policy_shock,
    )
    full_rule = _rule_term(true_states, model, phi_pi, phi_y)
    policy_rate, _ = _policy_from_rule_term(full_rule, base_policy_shock, rho_i)
    all_observables = _all_observables(model, true_states)
    noisy_indices = [model.observation_index(name) for name in scenario["noisy_observations"]]
    observations = all_observables[:, noisy_indices] + measurement_noise
    return RegimePolicyRun(
        scenario_name=scenario["name"],
        scenario_label=scenario["label"],
        info_scenario_name=scenario["info_scenario_name"],
        info_scenario_label=scenario["info_scenario_label"],
        gap_name=scenario["gap_name"],
        gap_label=scenario["gap_label"],
        policy_name="full_information_rule",
        policy_label="Полная информация",
        noisy_observation_names=tuple(scenario["noisy_observations"]),
        hidden_regimes=hidden_regimes,
        true_states=true_states,
        observations=observations,
        all_observables=all_observables,
        policy_rate=policy_rate,
        total_policy_shock=total_policy_shock,
        base_policy_shock=base_policy_shock,
        filtered_states=None,
        filtered_covariances=None,
        filtered_mode_probabilities=None,
        filter_results=None,
    )


def simulate_filtered_policy(
    *,
    model: RegimeSwitchingModel,
    config: RegimeSwitchingConfig,
    scenario: dict,
    hidden_regimes: np.ndarray,
    innovations: np.ndarray,
    measurement_noise: np.ndarray,
    phi_pi: float,
    phi_y: float,
    rho_i: float,
) -> RegimePolicyRun:
    horizon = config.partial_config.horizon
    state_dim = len(model.state_names)
    num_regimes = model.num_regimes()
    base_policy_shock = _base_policy_shock_path(config)
    _, measurement_covariance = _measurement_covariance(config, scenario)
    observation_matrix = model.observation_matrix[
        [model.observation_index(name) for name in scenario["noisy_observations"]]
    ]

    hidden_states = np.zeros((horizon, state_dim), dtype=float)
    observations = np.zeros((horizon, len(scenario["noisy_observations"])), dtype=float)
    all_observables = np.zeros((horizon, len(NOIY_OBSERVATION_NAMES := NOISY_OBSERVATION_NAMES)), dtype=float)
    policy_rate = np.zeros((horizon,), dtype=float)
    total_policy_shock = np.zeros((horizon,), dtype=float)
    filtered_states = np.zeros((horizon, state_dim), dtype=float)
    filtered_covariances = np.zeros((horizon, state_dim, state_dim), dtype=float)
    filtered_mode_probabilities = np.zeros((horizon, num_regimes), dtype=float)
    predicted_mode_probabilities = np.zeros_like(filtered_mode_probabilities)
    predicted_means = np.zeros((horizon, num_regimes, state_dim), dtype=float)
    predicted_covariances = np.zeros((horizon, num_regimes, state_dim, state_dim), dtype=float)
    regime_conditioned_means = np.zeros((horizon, num_regimes, state_dim), dtype=float)
    regime_conditioned_covariances = np.zeros((horizon, num_regimes, state_dim, state_dim), dtype=float)
    innovations_store = np.zeros((horizon, num_regimes, len(scenario["noisy_observations"])), dtype=float)
    innovation_covariances = np.zeros((horizon, num_regimes, len(scenario["noisy_observations"]), len(scenario["noisy_observations"])), dtype=float)

    mode_probabilities = model.stationary_regime_distribution()
    mode_means = np.repeat(model.initial_state_mean()[None, :], num_regimes, axis=0)
    mode_covariances = model.stationary_state_covariances()

    idx_rstar = model.state_index("rstar_gap")
    idx_pi = model.state_index("inflation_gap")
    idx_output = model.state_index("output_gap")
    previous_rate = 0.0
    previous_total_control = 0.0
    log_likelihood = 0.0

    for period in range(horizon):
        regime = hidden_regimes[period]
        transition = model.transition_matrices[regime]
        control = model.control_loadings[regime]
        if period == 0:
            hidden_states[period] = innovations[period]
        else:
            hidden_states[period] = (
                transition @ hidden_states[period - 1]
                + control * total_policy_shock[period - 1]
                + innovations[period]
            )

        all_observables[period] = hidden_states[period] @ model.observation_matrix.T
        observations[period] = all_observables[period][[model.observation_index(name) for name in scenario["noisy_observations"]]] + measurement_noise[period]

        (
            predicted_mode_probabilities[period],
            filtered_mode_probabilities[period],
            predicted_means[period],
            predicted_covariances[period],
            regime_conditioned_means[period],
            regime_conditioned_covariances[period],
            filtered_states[period],
            filtered_covariances[period],
            innovations_store[period],
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
            previous_control=previous_total_control,
        )
        log_likelihood += increment
        mode_probabilities = filtered_mode_probabilities[period]
        mode_means = regime_conditioned_means[period]
        mode_covariances = regime_conditioned_covariances[period]

        true_rule_term = hidden_states[period, idx_rstar] + phi_pi * hidden_states[period, idx_pi] + phi_y * hidden_states[period, idx_output]
        filtered_rule_term = filtered_states[period, idx_rstar] + phi_pi * filtered_states[period, idx_pi] + phi_y * filtered_states[period, idx_output]
        policy_rate[period] = rho_i * previous_rate + (1.0 - rho_i) * filtered_rule_term + base_policy_shock[period]
        total_policy_shock[period] = base_policy_shock[period] + (1.0 - rho_i) * (filtered_rule_term - true_rule_term)
        previous_rate = policy_rate[period]
        previous_total_control = total_policy_shock[period]

    filter_results = SwitchingKalmanFilterResults(
        predicted_mode_probabilities=predicted_mode_probabilities,
        filtered_mode_probabilities=filtered_mode_probabilities,
        predicted_means=predicted_means,
        predicted_covariances=predicted_covariances,
        filtered_means=filtered_states,
        filtered_covariances=filtered_covariances,
        regime_conditioned_means=regime_conditioned_means,
        regime_conditioned_covariances=regime_conditioned_covariances,
        innovations=innovations_store,
        innovation_covariances=innovation_covariances,
        log_likelihood=float(log_likelihood),
    )
    return RegimePolicyRun(
        scenario_name=scenario["name"],
        scenario_label=scenario["label"],
        info_scenario_name=scenario["info_scenario_name"],
        info_scenario_label=scenario["info_scenario_label"],
        gap_name=scenario["gap_name"],
        gap_label=scenario["gap_label"],
        policy_name="classical_filtered_rule",
        policy_label="Classical: switching filter + fixed rule",
        noisy_observation_names=tuple(scenario["noisy_observations"]),
        hidden_regimes=hidden_regimes,
        true_states=hidden_states,
        observations=observations,
        all_observables=all_observables,
        policy_rate=policy_rate,
        total_policy_shock=total_policy_shock,
        base_policy_shock=base_policy_shock,
        filtered_states=filtered_states,
        filtered_covariances=filtered_covariances,
        filtered_mode_probabilities=filtered_mode_probabilities,
        filter_results=filter_results,
    )
