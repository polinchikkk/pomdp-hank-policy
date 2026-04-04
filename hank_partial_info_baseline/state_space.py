from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from hank_full_baseline.calibration import HANKCalibration
from hank_full_baseline.distribution import build_group_masks, household_path_levels, path_distribution_statistics, stationary_distribution
from hank_full_baseline.household_solver import compute_mpc, compute_mpc_path
from hank_full_baseline.transition import solve_transition

from .config import HANKPartialInfoConfig, NOISY_OBSERVATION_NAMES, STATE_NAMES


AVAILABLE_INPUTS = ("rstar", "Z", "G", "monetary_policy_shock")


@dataclass(frozen=True)
class ReducedHANKStateSpaceModel:
    state_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    transition_matrix: np.ndarray
    control_loadings: np.ndarray
    process_noise_cov: np.ndarray
    observation_matrix: np.ndarray
    observation_fit_rmse: np.ndarray
    exogenous_state_names: tuple[str, ...]
    endogenous_state_names: tuple[str, ...]
    steady_state_statistics: dict[str, float]
    training_summary: dict[str, float]

    def state_index(self, name: str) -> int:
        return self.state_names.index(name)

    def observation_index(self, name: str) -> int:
        return self.observation_names.index(name)

    def initial_state_mean(self) -> np.ndarray:
        return np.zeros(len(self.state_names), dtype=float)

    def stationary_state_covariance(self) -> np.ndarray:
        try:
            covariance = solve_discrete_lyapunov(self.transition_matrix, self.process_noise_cov)
        except Exception:
            covariance = np.diag(np.maximum(np.diag(self.process_noise_cov), 1.0e-8))
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        clipped = np.clip(eigenvalues, 1.0e-10, None)
        return eigenvectors @ np.diag(clipped) @ eigenvectors.T


@dataclass(frozen=True)
class ControlledKalmanFilterResults:
    predicted_means: np.ndarray
    predicted_covariances: np.ndarray
    filtered_means: np.ndarray
    filtered_covariances: np.ndarray
    innovations: np.ndarray
    innovation_covariances: np.ndarray
    kalman_gains: np.ndarray
    log_likelihood: float


@dataclass(frozen=True)
class ScenarioSimulation:
    scenario_name: str
    scenario_label: str
    noisy_observation_names: tuple[str, ...]
    true_states: np.ndarray
    filtered_states: np.ndarray
    filtered_covariances: np.ndarray
    observations: np.ndarray
    base_policy_shock: np.ndarray
    additional_policy_wedge: np.ndarray
    total_policy_shock: np.ndarray
    full_information_rate: np.ndarray
    filtered_rate: np.ndarray
    exogenous_paths: dict[str, np.ndarray]
    filter_results: ControlledKalmanFilterResults


def _zero_shock_inputs(horizon: int) -> dict[str, np.ndarray]:
    return {name: np.zeros(horizon, dtype=float) for name in AVAILABLE_INPUTS}


def _impulse_path(horizon: int, scale: float, persistence: float = 0.0) -> np.ndarray:
    path = np.zeros(horizon, dtype=float)
    path[0] = scale
    for period in range(1, horizon):
        path[period] = persistence * path[period - 1]
    return path


def _simulate_linear_superposition(
    responses: dict[str, dict[str, np.ndarray]],
    shock_paths: dict[str, np.ndarray],
    variable_names: tuple[str, ...],
    horizon: int,
) -> dict[str, np.ndarray]:
    simulated: dict[str, np.ndarray] = {name: np.zeros(horizon, dtype=float) for name in variable_names}
    for input_name, path in shock_paths.items():
        for variable_name in variable_names:
            simulated[variable_name] += np.convolve(path, responses[input_name][variable_name])[:horizon]
    return simulated


def _sample_ar1(rng: np.random.Generator, horizon: int, rho: float, sigma: float) -> np.ndarray:
    series = np.zeros(horizon, dtype=float)
    innovations = rng.normal(scale=sigma, size=horizon)
    series[0] = innovations[0]
    for period in range(1, horizon):
        series[period] = rho * series[period - 1] + innovations[period]
    return series


def build_linear_responses(
    bundle,
    hank_config: HANKCalibration,
    partial_config: HANKPartialInfoConfig,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    ss = bundle["ss"]
    mpc = compute_mpc(ss)
    D_ss = stationary_distribution(ss)
    groups = build_group_masks(ss, hank_config, mpc)
    steady_stats = {
        "share_low_liquidity": float(D_ss[groups["groups"]["low_liquid"]].sum()),
        "mean_mpc": float(np.sum(D_ss * mpc)),
    }

    responses: dict[str, dict[str, np.ndarray]] = {}
    for input_name in AVAILABLE_INPUTS:
        shock_inputs = _zero_shock_inputs(partial_config.response_horizon)
        shock_inputs[input_name] = _impulse_path(
            partial_config.response_horizon,
            partial_config.impulse_scale,
            persistence=0.0,
        )
        transition = solve_transition(bundle, shock_inputs)
        path_levels = household_path_levels(ss, transition)
        mpc_path = compute_mpc_path(path_levels)
        distribution_stats = path_distribution_statistics(ss, path_levels, hank_config, mpc_path)
        scale = partial_config.impulse_scale

        responses[input_name] = {
            "rstar_gap": shock_inputs["rstar"] / scale,
            "productivity_gap": shock_inputs["Z"] / scale,
            "fiscal_gap": shock_inputs["G"] / scale,
            "inflation_gap": transition["pi"] / scale,
            "output_gap": transition["output_gap"] / scale,
            "low_liquidity_gap": (
                distribution_stats["share_low_liquidity"].to_numpy(dtype=float) - steady_stats["share_low_liquidity"]
            ) / scale,
            "mean_mpc_gap": (
                distribution_stats["mean_mpc"].to_numpy(dtype=float) - steady_stats["mean_mpc"]
            ) / scale,
            "pi": transition["pi"] / scale,
            "C": transition["C"] / scale,
            "w": transition["w"] / scale,
            "N": transition["N"] / scale,
            "share_low_liquidity": (
                distribution_stats["share_low_liquidity"].to_numpy(dtype=float) - steady_stats["share_low_liquidity"]
            ) / scale,
            "mean_mpc": (
                distribution_stats["mean_mpc"].to_numpy(dtype=float) - steady_stats["mean_mpc"]
            ) / scale,
        }
        responses[input_name]["output_gap"] = transition["output_gap"] / scale

    return responses, steady_stats


def fit_reduced_state_space(
    bundle,
    hank_config: HANKCalibration,
    partial_config: HANKPartialInfoConfig,
) -> ReducedHANKStateSpaceModel:
    responses, steady_stats = build_linear_responses(bundle, hank_config, partial_config)
    training_horizon = partial_config.training_periods + partial_config.training_burn_in
    rng = np.random.default_rng(partial_config.random_seed)
    training_shocks = {
        "rstar": _sample_ar1(rng, training_horizon, partial_config.rstar_rho, partial_config.rstar_std),
        "Z": _sample_ar1(rng, training_horizon, partial_config.productivity_rho, partial_config.productivity_std),
        "G": _sample_ar1(rng, training_horizon, partial_config.fiscal_rho, partial_config.fiscal_std),
        "monetary_policy_shock": _sample_ar1(
            rng,
            training_horizon,
            partial_config.training_policy_rho,
            partial_config.training_policy_std,
        ),
    }
    variable_names = STATE_NAMES + NOISY_OBSERVATION_NAMES
    simulated = _simulate_linear_superposition(responses, training_shocks, variable_names, training_horizon)

    states = np.column_stack([simulated[name] for name in STATE_NAMES])[partial_config.training_burn_in :]
    observations = np.column_stack([simulated[name] for name in NOISY_OBSERVATION_NAMES])[partial_config.training_burn_in :]
    controls = training_shocks["monetary_policy_shock"][partial_config.training_burn_in :]

    exogenous_dim = 3
    state_dim = len(STATE_NAMES)
    endogenous_slice = slice(exogenous_dim, state_dim)
    regressors = np.column_stack([states[:-1], controls[:-1]])
    targets = states[1:, endogenous_slice]
    coefficients, *_ = np.linalg.lstsq(regressors, targets, rcond=None)
    fitted = regressors @ coefficients
    residuals = targets - fitted

    transition_matrix = np.zeros((state_dim, state_dim), dtype=float)
    transition_matrix[0, 0] = partial_config.rstar_rho
    transition_matrix[1, 1] = partial_config.productivity_rho
    transition_matrix[2, 2] = partial_config.fiscal_rho
    transition_matrix[endogenous_slice, :] = coefficients[:-1, :].T
    control_loadings = np.zeros((state_dim,), dtype=float)
    control_loadings[endogenous_slice] = coefficients[-1, :]

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition_matrix))))
    shrink_factor = 1.0
    if spectral_radius >= 0.98:
        shrink_factor = 0.95 / spectral_radius
        transition_matrix[endogenous_slice, :] *= shrink_factor
        control_loadings[endogenous_slice] *= shrink_factor

    process_noise_cov = np.zeros((state_dim, state_dim), dtype=float)
    process_noise_cov[0, 0] = partial_config.rstar_std**2
    process_noise_cov[1, 1] = partial_config.productivity_std**2
    process_noise_cov[2, 2] = partial_config.fiscal_std**2
    if residuals.shape[0] > 1:
        process_noise_cov[endogenous_slice, endogenous_slice] = np.cov(residuals, rowvar=False)
    process_noise_cov += 1.0e-12 * np.eye(state_dim)

    observation_matrix = np.zeros((len(NOISY_OBSERVATION_NAMES), state_dim), dtype=float)
    observation_fit_rmse = np.zeros((len(NOISY_OBSERVATION_NAMES),), dtype=float)
    direct_state_map = {
        "pi": "inflation_gap",
        "output_gap": "output_gap",
        "share_low_liquidity": "low_liquidity_gap",
        "mean_mpc": "mean_mpc_gap",
    }
    state_lookup = {name: index for index, name in enumerate(STATE_NAMES)}
    for row_index, observation_name in enumerate(NOISY_OBSERVATION_NAMES):
        if observation_name in direct_state_map:
            observation_matrix[row_index, state_lookup[direct_state_map[observation_name]]] = 1.0
            fitted_observation = states @ observation_matrix[row_index]
        else:
            coefficients_obs, *_ = np.linalg.lstsq(states, observations[:, row_index], rcond=None)
            observation_matrix[row_index] = coefficients_obs
            fitted_observation = states @ coefficients_obs
        observation_fit_rmse[row_index] = float(
            np.sqrt(np.mean(np.square(observations[:, row_index] - fitted_observation)))
        )

    training_summary = {
        "training_periods": float(len(states)),
        "mean_abs_transition_residual": float(np.mean(np.abs(residuals))),
        "max_abs_transition_residual": float(np.max(np.abs(residuals))),
        "mean_observation_fit_rmse": float(np.mean(observation_fit_rmse)),
        "spectral_radius": spectral_radius,
        "stability_shrink_factor": shrink_factor,
    }

    return ReducedHANKStateSpaceModel(
        state_names=STATE_NAMES,
        observation_names=NOISY_OBSERVATION_NAMES,
        transition_matrix=transition_matrix,
        control_loadings=control_loadings,
        process_noise_cov=process_noise_cov,
        observation_matrix=observation_matrix,
        observation_fit_rmse=observation_fit_rmse,
        exogenous_state_names=STATE_NAMES[:3],
        endogenous_state_names=STATE_NAMES[3:],
        steady_state_statistics=steady_stats,
        training_summary=training_summary,
    )


def _base_policy_shock_path(config: HANKPartialInfoConfig) -> np.ndarray:
    path = np.zeros(config.horizon, dtype=float)
    path[config.base_policy_shock_period] = config.base_policy_shock_size
    for period in range(config.base_policy_shock_period + 1, config.horizon):
        path[period] = config.base_policy_shock_persistence * path[period - 1]
    return path


def simulate_information_scenario(
    model: ReducedHANKStateSpaceModel,
    partial_config: HANKPartialInfoConfig,
    scenario: dict,
    seed: int,
    phi_pi: float,
    phi_y: float,
    rho_i: float,
) -> ScenarioSimulation:
    rng = np.random.default_rng(seed)
    state_dim = len(model.state_names)
    observation_names = tuple(scenario["noisy_observations"])
    measurement_noise_std = {
        name: partial_config.base_measurement_noise()[name] * scenario["noise_scale"]
        for name in observation_names
    }

    innovations = np.zeros((partial_config.horizon, state_dim), dtype=float)
    innovations[:, 0] = rng.normal(scale=partial_config.rstar_std, size=partial_config.horizon)
    innovations[:, 1] = rng.normal(scale=partial_config.productivity_std, size=partial_config.horizon)
    innovations[:, 2] = rng.normal(scale=partial_config.fiscal_std, size=partial_config.horizon)
    measurement_noise = np.column_stack([
        rng.normal(scale=measurement_noise_std[name], size=partial_config.horizon)
        for name in observation_names
    ])

    base_policy_shock = _base_policy_shock_path(partial_config)
    additional_policy_wedge = np.zeros(partial_config.horizon, dtype=float)
    total_policy_shock = np.zeros(partial_config.horizon, dtype=float)
    total_policy_shock[:] = base_policy_shock

    true_states = np.zeros((partial_config.horizon, state_dim), dtype=float)
    filtered_states = np.zeros_like(true_states)
    filtered_covariances = np.zeros((partial_config.horizon, state_dim, state_dim), dtype=float)
    observations = np.zeros((partial_config.horizon, len(observation_names)), dtype=float)
    full_information_rate = np.zeros(partial_config.horizon, dtype=float)
    filtered_rate = np.zeros(partial_config.horizon, dtype=float)

    observation_matrix = model.observation_matrix[[model.observation_index(name) for name in observation_names]]
    filtered_mean = model.initial_state_mean()
    filtered_covariance = model.stationary_state_covariance()
    previous_total_control = 0.0
    previous_full_rate = 0.0
    previous_filtered_rate = 0.0
    predicted_means = np.zeros_like(true_states)
    predicted_covariances = np.zeros((partial_config.horizon, state_dim, state_dim), dtype=float)
    innovations_store = np.zeros((partial_config.horizon, len(observation_names)), dtype=float)
    innovation_covariances = np.zeros((partial_config.horizon, len(observation_names), len(observation_names)), dtype=float)
    kalman_gains = np.zeros((partial_config.horizon, state_dim, len(observation_names)), dtype=float)
    identity = np.eye(state_dim, dtype=float)
    measurement_cov = np.diag([measurement_noise_std[name] ** 2 for name in observation_names]).astype(float)
    log_likelihood = 0.0

    idx_rstar = model.state_index("rstar_gap")
    idx_pi = model.state_index("inflation_gap")
    idx_output = model.state_index("output_gap")

    for period in range(partial_config.horizon):
        if period == 0:
            true_state = innovations[period]
        else:
            true_state = (
                model.transition_matrix @ true_states[period - 1]
                + model.control_loadings * total_policy_shock[period - 1]
                + innovations[period]
            )
        true_states[period] = true_state

        predicted_mean = model.transition_matrix @ filtered_mean + model.control_loadings * previous_total_control
        predicted_covariance = (
            model.transition_matrix @ filtered_covariance @ model.transition_matrix.T
            + model.process_noise_cov
        )
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
        predicted_means[period] = predicted_mean
        predicted_covariances[period] = predicted_covariance

        observed_values = observation_matrix @ true_state + measurement_noise[period]
        observations[period] = observed_values

        innovation = observed_values - observation_matrix @ predicted_mean
        innovation_covariance = observation_matrix @ predicted_covariance @ observation_matrix.T + measurement_cov
        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
        jitter = 1.0e-12
        sign, logdet = np.linalg.slogdet(innovation_covariance)
        while sign <= 0:
            innovation_covariance = innovation_covariance + jitter * np.eye(len(observation_names), dtype=float)
            jitter *= 10.0
            sign, logdet = np.linalg.slogdet(innovation_covariance)
            if jitter > 1.0:
                raise RuntimeError("Innovation covariance is not positive definite.")
        innovation_covariance_inv = np.linalg.inv(innovation_covariance)
        kalman_gain = predicted_covariance @ observation_matrix.T @ innovation_covariance_inv
        filtered_mean = predicted_mean + kalman_gain @ innovation
        filtered_covariance = (
            (identity - kalman_gain @ observation_matrix)
            @ predicted_covariance
            @ (identity - kalman_gain @ observation_matrix).T
            + kalman_gain @ measurement_cov @ kalman_gain.T
        )
        filtered_covariance = 0.5 * (filtered_covariance + filtered_covariance.T)

        filtered_states[period] = filtered_mean
        filtered_covariances[period] = filtered_covariance
        innovations_store[period] = innovation
        innovation_covariances[period] = innovation_covariance
        kalman_gains[period] = kalman_gain

        log_likelihood += -0.5 * (
            len(observation_names) * np.log(2.0 * np.pi)
            + logdet
            + innovation.T @ innovation_covariance_inv @ innovation
        )

        true_rule_term = true_state[idx_rstar] + phi_pi * true_state[idx_pi] + phi_y * true_state[idx_output]
        filtered_rule_term = filtered_mean[idx_rstar] + phi_pi * filtered_mean[idx_pi] + phi_y * filtered_mean[idx_output]

        full_information_rate[period] = rho_i * previous_full_rate + (1.0 - rho_i) * true_rule_term + base_policy_shock[period]
        filtered_rate[period] = rho_i * previous_filtered_rate + (1.0 - rho_i) * filtered_rule_term + base_policy_shock[period]
        additional_policy_wedge[period] = (1.0 - rho_i) * (filtered_rule_term - true_rule_term)
        total_policy_shock[period] = base_policy_shock[period] + additional_policy_wedge[period]

        previous_total_control = total_policy_shock[period]
        previous_full_rate = full_information_rate[period]
        previous_filtered_rate = filtered_rate[period]

    filter_results = ControlledKalmanFilterResults(
        predicted_means=predicted_means,
        predicted_covariances=predicted_covariances,
        filtered_means=filtered_states,
        filtered_covariances=filtered_covariances,
        innovations=innovations_store,
        innovation_covariances=innovation_covariances,
        kalman_gains=kalman_gains,
        log_likelihood=float(log_likelihood),
    )

    return ScenarioSimulation(
        scenario_name=scenario["name"],
        scenario_label=scenario["label"],
        noisy_observation_names=observation_names,
        true_states=true_states,
        filtered_states=filtered_states,
        filtered_covariances=filtered_covariances,
        observations=observations,
        base_policy_shock=base_policy_shock,
        additional_policy_wedge=additional_policy_wedge,
        total_policy_shock=total_policy_shock,
        full_information_rate=full_information_rate,
        filtered_rate=filtered_rate,
        exogenous_paths={
            "rstar": true_states[:, model.state_index("rstar_gap")],
            "Z": true_states[:, model.state_index("productivity_gap")],
            "G": true_states[:, model.state_index("fiscal_gap")],
        },
        filter_results=filter_results,
    )


def scenario_state_frames(
    simulation: ScenarioSimulation,
    model: ReducedHANKStateSpaceModel,
    confidence_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = np.arange(simulation.true_states.shape[0], dtype=int)
    true_frame = pd.DataFrame({
        "scenario": simulation.scenario_name,
        "scenario_label": simulation.scenario_label,
        "period": periods,
    })
    filtered_frame = true_frame.copy()

    filtered_variances = np.maximum(
        np.diagonal(simulation.filtered_covariances, axis1=1, axis2=2),
        0.0,
    )
    filtered_stds = np.sqrt(filtered_variances)

    for index, state_name in enumerate(model.state_names):
        true_frame[f"true_{state_name}"] = simulation.true_states[:, index]
        filtered_frame[f"true_{state_name}"] = simulation.true_states[:, index]
        filtered_frame[f"filtered_{state_name}"] = simulation.filtered_states[:, index]
        filtered_frame[f"filtered_std_{state_name}"] = filtered_stds[:, index]
        filtered_frame[f"lower_{state_name}"] = simulation.filtered_states[:, index] - confidence_scale * filtered_stds[:, index]
        filtered_frame[f"upper_{state_name}"] = simulation.filtered_states[:, index] + confidence_scale * filtered_stds[:, index]
        filtered_frame[f"error_{state_name}"] = simulation.filtered_states[:, index] - simulation.true_states[:, index]

    filtered_frame["full_information_rate"] = simulation.full_information_rate
    filtered_frame["filtered_rate"] = simulation.filtered_rate
    filtered_frame["additional_policy_wedge"] = simulation.additional_policy_wedge
    return true_frame, filtered_frame


def scenario_observation_frame(simulation: ScenarioSimulation) -> pd.DataFrame:
    frame = pd.DataFrame({
        "scenario": simulation.scenario_name,
        "scenario_label": simulation.scenario_label,
        "period": np.arange(simulation.observations.shape[0], dtype=int),
        "known_policy_rate": simulation.filtered_rate,
        "full_information_rate": simulation.full_information_rate,
    })
    for index, name in enumerate(simulation.noisy_observation_names):
        frame[f"observed_{name}"] = simulation.observations[:, index]
    return frame
