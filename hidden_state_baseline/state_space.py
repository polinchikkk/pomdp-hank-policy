from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from nk_baseline.model import CONTROL_NAMES, NKParameters, SHOCK_NAMES, STATE_NAMES
from nk_baseline.solver import solve_linear_nk_model


@dataclass(frozen=True)
class LinearGaussianStateSpaceModel:
    transition_matrix: np.ndarray
    process_loading: np.ndarray
    observation_matrix: np.ndarray
    measurement_loading: np.ndarray
    process_noise_cov: np.ndarray
    measurement_noise_cov: np.ndarray
    state_names: tuple[str, ...]
    innovation_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    policy_name: str

    def initial_state_mean(self) -> np.ndarray:
        return np.zeros((self.transition_matrix.shape[0],), dtype=float)

    def stationary_state_covariance(self) -> np.ndarray:
        return solve_discrete_lyapunov(self.transition_matrix, self.process_noise_cov)


def _observation_indices(observation_names: tuple[str, ...] | None) -> tuple[tuple[str, ...], list[int]]:
    selected_names = tuple(CONTROL_NAMES if observation_names is None else observation_names)
    invalid_names = sorted(set(selected_names) - set(CONTROL_NAMES))
    if invalid_names:
        raise ValueError(f"Unknown observation names: {', '.join(invalid_names)}")

    lookup = {name: index for index, name in enumerate(CONTROL_NAMES)}
    return selected_names, [lookup[name] for name in selected_names]


def build_multishock_state_space_model(
    params: NKParameters,
    measurement_noise_std: float | np.ndarray,
    observation_names: tuple[str, ...] | None = None,
) -> LinearGaussianStateSpaceModel:
    nk_solution = solve_linear_nk_model(params=params)
    selected_observations, observation_indices = _observation_indices(observation_names)
    measurement_std_vector = np.broadcast_to(
        np.asarray(measurement_noise_std, dtype=float),
        (len(selected_observations),),
    ).astype(float)

    transition_matrix = nk_solution.transition_matrix.astype(float)
    process_loading = nk_solution.shock_matrix.astype(float)
    observation_matrix = nk_solution.policy_matrix[observation_indices, :].astype(float)
    measurement_loading = np.diag(measurement_std_vector).astype(float)

    return LinearGaussianStateSpaceModel(
        transition_matrix=transition_matrix,
        process_loading=process_loading,
        observation_matrix=observation_matrix,
        measurement_loading=measurement_loading,
        process_noise_cov=process_loading @ process_loading.T,
        measurement_noise_cov=measurement_loading @ measurement_loading.T,
        state_names=tuple(STATE_NAMES),
        innovation_names=tuple(SHOCK_NAMES),
        observation_names=selected_observations,
        policy_name="i",
    )


def simulate_hidden_states(
    model: LinearGaussianStateSpaceModel,
    periods: int,
    burn_in: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    total_periods = periods + burn_in
    state_dim = model.transition_matrix.shape[0]
    process_dim = model.process_loading.shape[1]
    states = np.zeros((total_periods, state_dim), dtype=float)
    standardized_innovations = rng.standard_normal((total_periods, process_dim))
    realized_innovations = np.zeros((total_periods, process_dim), dtype=float)

    for period in range(1, total_periods):
        realized_innovations[period] = standardized_innovations[period]
        states[period] = (
            model.transition_matrix @ states[period - 1]
            + model.process_loading @ standardized_innovations[period]
        )

    frame_dict: dict[str, np.ndarray] = {"t": np.arange(-burn_in, periods, dtype=int)}
    for index, name in enumerate(model.state_names):
        frame_dict[f"true_{name}"] = states[:, index]
    for index, name in enumerate(model.innovation_names):
        frame_dict[f"process_eps_{name}"] = realized_innovations[:, index]

    frame = pd.DataFrame(frame_dict)
    frame = frame.loc[frame["t"] >= 0].reset_index(drop=True)
    return frame, standardized_innovations


def generate_observations(
    model: LinearGaussianStateSpaceModel,
    true_states: pd.DataFrame,
    standardized_measurement_noise: np.ndarray,
) -> pd.DataFrame:
    state_values = true_states[[f"true_{name}" for name in model.state_names]].to_numpy(dtype=float)
    measurement_noise = standardized_measurement_noise @ model.measurement_loading.T
    noiseless_observations = state_values @ model.observation_matrix.T
    observations = noiseless_observations + measurement_noise

    frame = pd.DataFrame({"t": true_states["t"].to_numpy(dtype=int)})
    for index, name in enumerate(model.observation_names):
        frame[f"obs_{name}"] = observations[:, index]
        frame[f"signal_{name}"] = noiseless_observations[:, index]
        frame[f"measurement_noise_{name}"] = measurement_noise[:, index]
    return frame


def state_space_spec_payload(
    params: NKParameters,
    model: LinearGaussianStateSpaceModel,
    baseline_noise_label: str,
    measurement_noise_scenarios: dict[str, float],
    observation_designs: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    return {
        "stage": "stage3_hidden_state_baseline",
        "source_model": "stage2_small_linear_nk",
        "description": (
            "Linear-Gaussian hidden-state baseline with latent natural-rate, cost-push, "
            "and monetary-policy shocks inferred from noisy New Keynesian observables."
        ),
        "linearization_point": "zero steady state in gap variables",
        "hidden_state": list(model.state_names),
        "structural_innovations": list(model.innovation_names),
        "observations": list(model.observation_names),
        "available_observation_designs": {
            label: list(names) for label, names in (observation_designs or {}).items()
        },
        "policy_instrument": model.policy_name,
        "structural_parameters": params.to_dict(),
        "state_transition": {
            "A": model.transition_matrix.tolist(),
            "B": model.process_loading.tolist(),
            "Q": model.process_noise_cov.tolist(),
        },
        "measurement_equation": {
            "C": model.observation_matrix.tolist(),
            "D": model.measurement_loading.tolist(),
            "R": model.measurement_noise_cov.tolist(),
        },
        "baseline_measurement_noise_scenario": baseline_noise_label,
        "measurement_noise_scenarios": measurement_noise_scenarios,
    }
