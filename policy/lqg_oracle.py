from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hank_ssj.state_space import StateSpaceSpec, spectral_radius
from policy.linear_rules import LinearRule


@dataclass(frozen=True)
class LQGLossWeights:
    inflation: float = 1.0
    output_gap: float = 1.0
    consumption: float = 0.25
    rate_smoothing: float = 0.1


@dataclass(frozen=True)
class LinearControlSystem:
    state_names: tuple[str, ...]
    A: np.ndarray
    Q: np.ndarray
    B: np.ndarray
    initial_mean: np.ndarray
    initial_cov: np.ndarray
    policy_input_source: str


@dataclass(frozen=True)
class LQRSolution:
    K: np.ndarray
    P: np.ndarray
    iterations: int
    converged: bool
    closed_loop_spectral_radius: float
    K_path: np.ndarray | None = None


@dataclass(frozen=True)
class LinearPolicyLoss:
    total_loss: float
    inflation_loss: float
    output_gap_loss: float
    consumption_loss: float
    rate_smoothing_loss: float


STATE_TO_OBSERVATION = {
    "pi": "pi_obs",
    "Y": "Y_obs",
    "C": "C_obs",
    "mean_mpc_centered": "mean_mpc_centered_obs",
    "share_low_liquidity_centered": "share_low_liquidity_centered_obs",
    "interest_exposure_centered": "interest_exposure_centered_obs",
}

FILTERED_FEATURE_BY_STATE = {
    "pi": "E_pi",
    "Y": "E_Y",
    "C": "E_C",
    "mean_mpc_centered": "E_mean_mpc",
    "share_low_liquidity_centered": "E_low_liquidity_share",
    "interest_exposure_centered": "E_interest_exposure",
}

JACOBIAN_OUTPUT_BY_STATE = {
    "pi": "pi",
    "Y": "Y",
    "C": "C",
    "mean_mpc_centered": "mean_mpc_centered",
    "share_low_liquidity_centered": "share_low_liquidity_centered",
    "interest_exposure_centered": "interest_exposure_centered",
}


def load_state_space_filter_spec(path: Path, information_state: str) -> StateSpaceSpec:
    """Load one observation-specific Kalman specification from the joint state-space JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    filters = payload.get("filters", {})
    if information_state not in filters:
        raise ValueError(
            f"State-space file {path} does not contain filter {information_state!r}. "
            f"Available filters: {sorted(filters)}"
        )
    item = filters[information_state]
    return StateSpaceSpec(
        state_names=tuple(item["state_names"]),
        observation_names=tuple(item["observation_names"]),
        A=np.asarray(item["A"], dtype=float),
        Q=np.asarray(item["Q"], dtype=float),
        M=np.asarray(item["M"], dtype=float),
        R=np.asarray(item["R"], dtype=float),
        initial_mean=np.asarray(item["initial_mean"], dtype=float),
        initial_cov=np.asarray(item["initial_cov"], dtype=float),
    )


def load_linear_control_system(
    *,
    state_space_spec_json: Path,
    jacobians_npz: Path,
    observables: pd.DataFrame,
    policy_col: str = "i",
    policy_input_source: str = "ssj_one_step",
) -> LinearControlSystem:
    """Build the controlled linear system used by the Riccati oracle.

    The common transition matrix A and process covariance Q are taken from the joint Kalman
    specification.  The policy-input matrix B is not invented independently: by default it is
    read from the local SSJ response of state variables to the rate path.  If a state is not
    present in the Jacobian bundle, the function falls back to a regression estimate from the
    same HANK/SSJ trajectories.
    """

    payload = json.loads(state_space_spec_json.read_text(encoding="utf-8"))
    state_names = tuple(payload["state_names"])
    A = np.asarray(payload["A"], dtype=float)
    Q = np.asarray(payload["Q"], dtype=float)
    initial_mean = np.asarray(payload["initial_mean"], dtype=float)
    initial_cov = np.asarray(payload["initial_cov"], dtype=float)

    if policy_input_source == "ssj_one_step":
        B = policy_input_matrix_from_jacobians(
            jacobians_npz=jacobians_npz,
            state_names=state_names,
            fallback_observables=observables,
            A=A,
            policy_col=policy_col,
        )
    elif policy_input_source == "transition_regression":
        B = estimate_policy_input_matrix(
            observables=observables,
            state_names=list(state_names),
            A=A,
            policy_col=policy_col,
        )
    else:
        raise ValueError(f"Unknown policy input source: {policy_input_source}")

    return LinearControlSystem(
        state_names=state_names,
        A=A,
        Q=Q,
        B=B,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
        policy_input_source=policy_input_source,
    )


def policy_input_matrix_from_jacobians(
    *,
    jacobians_npz: Path,
    state_names: tuple[str, ...],
    fallback_observables: pd.DataFrame,
    A: np.ndarray,
    policy_col: str = "i",
    ridge: float = 1e-10,
) -> np.ndarray:
    """Approximate B from the one-period SSJ effect of the rate path on each state."""

    fallback = estimate_policy_input_matrix(
        observables=fallback_observables,
        state_names=list(state_names),
        A=A,
        policy_col=policy_col,
    )
    with np.load(jacobians_npz, allow_pickle=True) as bundle:
        if "J_monetary_policy_shock_i" not in bundle.files:
            return fallback
        rate_response = np.asarray(bundle["J_monetary_policy_shock_i"], dtype=float)
        horizon = rate_response.shape[0]
        normal = rate_response.T @ rate_response + float(ridge) * np.eye(horizon)
        shock_from_rate = np.linalg.solve(normal, rate_response.T)
        values: list[float] = []
        for state_name, fallback_value in zip(state_names, fallback[:, 0]):
            output = JACOBIAN_OUTPUT_BY_STATE.get(state_name, state_name)
            key = f"J_monetary_policy_shock_{output}"
            if key not in bundle.files:
                values.append(float(fallback_value))
                continue
            effect = np.asarray(bundle[key], dtype=float)[:horizon, :horizon] @ shock_from_rate
            one_step = np.diag(effect[1:, :-1])
            if one_step.size == 0 or not np.all(np.isfinite(one_step)):
                values.append(float(fallback_value))
            else:
                values.append(float(np.median(one_step)))
    return np.asarray(values, dtype=float).reshape(len(state_names), 1)


def estimate_policy_input_matrix(
    *,
    observables: pd.DataFrame,
    state_names: list[str],
    A: np.ndarray,
    policy_col: str = "i",
    ridge: float = 1e-12,
) -> np.ndarray:
    """Estimate B in x_{t+1}=Ax_t+B i_t+eps from HANK/SSJ trajectories."""

    required = {"scenario", "period", policy_col, *state_names}
    missing = required.difference(observables.columns)
    if missing:
        raise ValueError(f"Cannot estimate B: missing columns {sorted(missing)}")
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    u_blocks: list[np.ndarray] = []
    for _, group in observables.sort_values(["scenario", "period"]).groupby("scenario", sort=False):
        values = group[state_names].to_numpy(dtype=float)
        controls = group[policy_col].to_numpy(dtype=float)
        if len(group) < 2:
            continue
        x_blocks.append(values[:-1])
        y_blocks.append(values[1:])
        u_blocks.append(controls[:-1, None])
    if not x_blocks:
        raise ValueError("Cannot estimate B: no scenario has at least two periods.")
    x = np.vstack(x_blocks)
    y = np.vstack(y_blocks)
    u = np.vstack(u_blocks)
    residual = y - x @ A.T
    denom = float(u.T @ u + ridge)
    return ((u.T @ residual) / denom).reshape(len(state_names), 1)


def solve_lqr_with_rate_smoothing(
    *,
    system: LinearControlSystem,
    weights: LQGLossWeights | None = None,
    tolerance: float = 1e-11,
    max_iterations: int = 10_000,
) -> LQRSolution:
    """Solve the infinite-horizon LQR with lagged rate in the state.

    The augmented state is z_t=(x_t, i_{t-1}).  The stage loss includes
    pi^2 + y^2 + 0.25 C^2 + lambda_i (i_t-i_{t-1})^2.
    """

    weights = LQGLossWeights() if weights is None else weights
    A_aug, B_aug, Q_aug, R, N = _augmented_lqr_matrices(system, weights)
    P = Q_aug.copy()
    converged = False
    K = np.zeros((1, A_aug.shape[0]), dtype=float)
    for iteration in range(1, max_iterations + 1):
        G = _regularize_square(R + B_aug.T @ P @ B_aug, floor=1e-14)
        K = np.linalg.solve(G, B_aug.T @ P @ A_aug + N.T)
        P_next = Q_aug + A_aug.T @ P @ A_aug - (A_aug.T @ P @ B_aug + N) @ K
        P_next = 0.5 * (P_next + P_next.T)
        if float(np.max(np.abs(P_next - P))) < tolerance:
            P = P_next
            converged = True
            break
        P = P_next
    radius = spectral_radius(A_aug - B_aug @ K)
    return LQRSolution(
        K=K,
        P=P,
        iterations=iteration,
        converged=converged,
        closed_loop_spectral_radius=radius,
        K_path=None,
    )


def solve_finite_horizon_lqr_with_rate_smoothing(
    *,
    system: LinearControlSystem,
    horizon: int,
    weights: LQGLossWeights | None = None,
) -> LQRSolution:
    """Solve the finite-horizon Riccati recursion for the experiment horizon."""

    weights = LQGLossWeights() if weights is None else weights
    A_aug, B_aug, Q_aug, R, N = _augmented_lqr_matrices(system, weights)
    n_aug = A_aug.shape[0]
    P = np.zeros((n_aug, n_aug), dtype=float)
    gains = np.zeros((int(horizon), 1, n_aug), dtype=float)
    for period in reversed(range(int(horizon))):
        G = _regularize_square(R + B_aug.T @ P @ B_aug, floor=1e-14)
        K = np.linalg.solve(G, B_aug.T @ P @ A_aug + N.T)
        gains[period] = K
        P = Q_aug + A_aug.T @ P @ A_aug - (A_aug.T @ P @ B_aug + N) @ K
        P = 0.5 * (P + P.T)
    radius = max(spectral_radius(A_aug - B_aug @ gains[period]) for period in range(int(horizon)))
    return LQRSolution(
        K=gains[0],
        P=P,
        iterations=int(horizon),
        converged=True,
        closed_loop_spectral_radius=float(radius),
        K_path=gains,
    )


def simulate_lqg_path(
    *,
    system: LinearControlSystem,
    observation_spec: StateSpaceSpec,
    lqr_solution: LQRSolution,
    base_path: pd.DataFrame,
    observations: pd.DataFrame,
    weights: LQGLossWeights | None = None,
    max_abs_rate: float | None = None,
    max_abs_rate_change: float | None = None,
) -> tuple[LinearPolicyLoss, np.ndarray]:
    """Simulate an optimal LQG controller on one linearized HANK/SSJ path."""

    weights = LQGLossWeights() if weights is None else weights
    state = _state_matrix(base_path, system.state_names)
    innovations = _state_innovations(base_path=base_path, system=system)
    observation_noise = _observation_noise(base_path=base_path, observations=observations, spec=observation_spec)

    periods = state.shape[0]
    counterfactual = np.zeros_like(state)
    rates = np.zeros(periods, dtype=float)
    counterfactual[0] = state[0]
    mean_prior = np.asarray(system.initial_mean, dtype=float).copy()
    cov_prior = np.asarray(system.initial_cov, dtype=float).copy()
    lagged_rate = 0.0
    for period in range(periods):
        if period > 0:
            counterfactual[period] = (
                system.A @ counterfactual[period - 1]
                + system.B[:, 0] * rates[period - 1]
                + innovations[period - 1]
            )
        observed = observation_spec.M @ counterfactual[period] + observation_noise[period]
        mean_post, cov_post = kalman_measurement_update(
            mean_prior=mean_prior,
            cov_prior=cov_prior,
            observation=observed,
            M=observation_spec.M,
            R=observation_spec.R,
        )
        rate = _rate_from_gain(_gain_for_period(lqr_solution, period), mean_post, lagged_rate)
        rate = _clip_rate(rate, lagged_rate, max_abs_rate=max_abs_rate, max_abs_rate_change=max_abs_rate_change)
        rates[period] = rate
        lagged_rate = rate
        mean_prior = system.A @ mean_post + system.B[:, 0] * rate
        cov_prior = _regularize_square(system.A @ cov_post @ system.A.T + system.Q, floor=1e-14)
    return linear_policy_loss(counterfactual, rates, system.state_names, weights), rates


def simulate_lqr_full_state_path(
    *,
    system: LinearControlSystem,
    lqr_solution: LQRSolution,
    base_path: pd.DataFrame,
    weights: LQGLossWeights | None = None,
    max_abs_rate: float | None = None,
    max_abs_rate_change: float | None = None,
) -> tuple[LinearPolicyLoss, np.ndarray]:
    """Simulate the full-state LQR benchmark on one linearized HANK/SSJ path."""

    weights = LQGLossWeights() if weights is None else weights
    state = _state_matrix(base_path, system.state_names)
    innovations = _state_innovations(base_path=base_path, system=system)
    periods = state.shape[0]
    counterfactual = np.zeros_like(state)
    rates = np.zeros(periods, dtype=float)
    counterfactual[0] = state[0]
    lagged_rate = 0.0
    for period in range(periods):
        if period > 0:
            counterfactual[period] = (
                system.A @ counterfactual[period - 1]
                + system.B[:, 0] * rates[period - 1]
                + innovations[period - 1]
            )
        rate = _rate_from_gain(_gain_for_period(lqr_solution, period), counterfactual[period], lagged_rate)
        rate = _clip_rate(rate, lagged_rate, max_abs_rate=max_abs_rate, max_abs_rate_change=max_abs_rate_change)
        rates[period] = rate
        lagged_rate = rate
    return linear_policy_loss(counterfactual, rates, system.state_names, weights), rates


def simulate_simple_filtered_rule_path(
    *,
    system: LinearControlSystem,
    observation_spec: StateSpaceSpec,
    rule: LinearRule,
    base_path: pd.DataFrame,
    observations: pd.DataFrame,
    weights: LQGLossWeights | None = None,
    max_abs_rate: float | None = None,
    max_abs_rate_change: float | None = None,
) -> tuple[LinearPolicyLoss, np.ndarray]:
    """Evaluate an existing restricted linear rule in the same linear state-space DGP."""

    weights = LQGLossWeights() if weights is None else weights
    state = _state_matrix(base_path, system.state_names)
    innovations = _state_innovations(base_path=base_path, system=system)
    observation_noise = _observation_noise(base_path=base_path, observations=observations, spec=observation_spec)

    periods = state.shape[0]
    counterfactual = np.zeros_like(state)
    rates = np.zeros(periods, dtype=float)
    counterfactual[0] = state[0]
    mean_prior = np.asarray(system.initial_mean, dtype=float).copy()
    cov_prior = np.asarray(system.initial_cov, dtype=float).copy()
    lagged_rate = 0.0
    for period in range(periods):
        if period > 0:
            counterfactual[period] = (
                system.A @ counterfactual[period - 1]
                + system.B[:, 0] * rates[period - 1]
                + innovations[period - 1]
            )
        observed = observation_spec.M @ counterfactual[period] + observation_noise[period]
        mean_post, cov_post = kalman_measurement_update(
            mean_prior=mean_prior,
            cov_prior=cov_prior,
            observation=observed,
            M=observation_spec.M,
            R=observation_spec.R,
        )
        features = _features_from_estimate(mean_post, system.state_names)
        rate = rule.rate(features, lagged_rate)
        rate = _clip_rate(rate, lagged_rate, max_abs_rate=max_abs_rate, max_abs_rate_change=max_abs_rate_change)
        rates[period] = rate
        lagged_rate = rate
        mean_prior = system.A @ mean_post + system.B[:, 0] * rate
        cov_prior = _regularize_square(system.A @ cov_post @ system.A.T + system.Q, floor=1e-14)
    return linear_policy_loss(counterfactual, rates, system.state_names, weights), rates


def kalman_measurement_update(
    *,
    mean_prior: np.ndarray,
    cov_prior: np.ndarray,
    observation: np.ndarray,
    M: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = M @ mean_prior
    innovation = observation - predicted
    S = _regularize_square(M @ cov_prior @ M.T + R, floor=1e-14)
    gain = np.linalg.solve(S, M @ cov_prior).T
    mean_post = mean_prior + gain @ innovation
    identity = np.eye(cov_prior.shape[0])
    cov_post = (identity - gain @ M) @ cov_prior @ (identity - gain @ M).T + gain @ R @ gain.T
    return mean_post, _regularize_square(cov_post, floor=1e-14)


def linear_policy_loss(
    state_path: np.ndarray,
    rates: np.ndarray,
    state_names: tuple[str, ...],
    weights: LQGLossWeights,
) -> LinearPolicyLoss:
    state_index = {name: index for index, name in enumerate(state_names)}
    pi = state_path[:, state_index["pi"]]
    y = state_path[:, state_index["Y"]]
    c = state_path[:, state_index["C"]]
    rate_change = rates - np.r_[0.0, rates[:-1]]
    inflation_loss = float(np.sum(weights.inflation * pi**2))
    output_gap_loss = float(np.sum(weights.output_gap * y**2))
    consumption_loss = float(np.sum(weights.consumption * c**2))
    rate_smoothing_loss = float(np.sum(weights.rate_smoothing * rate_change**2))
    return LinearPolicyLoss(
        total_loss=inflation_loss + output_gap_loss + consumption_loss + rate_smoothing_loss,
        inflation_loss=inflation_loss,
        output_gap_loss=output_gap_loss,
        consumption_loss=consumption_loss,
        rate_smoothing_loss=rate_smoothing_loss,
    )


def _augmented_lqr_matrices(
    system: LinearControlSystem,
    weights: LQGLossWeights,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_state = len(system.state_names)
    A_aug = np.zeros((n_state + 1, n_state + 1), dtype=float)
    A_aug[:n_state, :n_state] = system.A
    B_aug = np.zeros((n_state + 1, 1), dtype=float)
    B_aug[:n_state, :] = system.B
    B_aug[n_state, 0] = 1.0

    Q_aug = np.zeros((n_state + 1, n_state + 1), dtype=float)
    state_index = {name: index for index, name in enumerate(system.state_names)}
    Q_aug[state_index["pi"], state_index["pi"]] = float(weights.inflation)
    Q_aug[state_index["Y"], state_index["Y"]] = float(weights.output_gap)
    Q_aug[state_index["C"], state_index["C"]] = float(weights.consumption)
    Q_aug[n_state, n_state] = float(weights.rate_smoothing)

    R = np.asarray([[float(weights.rate_smoothing)]], dtype=float)
    N = np.zeros((n_state + 1, 1), dtype=float)
    N[n_state, 0] = -float(weights.rate_smoothing)
    return A_aug, B_aug, Q_aug, R, N


def _state_matrix(frame: pd.DataFrame, state_names: tuple[str, ...]) -> np.ndarray:
    missing = set(state_names).difference(frame.columns)
    if missing:
        raise ValueError(f"Base path is missing state columns: {sorted(missing)}")
    return frame.loc[:, list(state_names)].to_numpy(dtype=float)


def _state_innovations(*, base_path: pd.DataFrame, system: LinearControlSystem) -> np.ndarray:
    state = _state_matrix(base_path, system.state_names)
    controls = base_path["i"].to_numpy(dtype=float)
    if state.shape[0] < 2:
        return np.zeros((0, state.shape[1]), dtype=float)
    return state[1:] - state[:-1] @ system.A.T - controls[:-1, None] @ system.B.T


def _observation_noise(
    *,
    base_path: pd.DataFrame,
    observations: pd.DataFrame,
    spec: StateSpaceSpec,
) -> np.ndarray:
    state = _state_matrix(base_path, spec.state_names)
    missing = set(spec.observation_names).difference(observations.columns)
    if missing:
        raise ValueError(f"Observation path is missing columns: {sorted(missing)}")
    observed = observations.loc[:, list(spec.observation_names)].to_numpy(dtype=float)
    return observed - state @ spec.M.T


def _features_from_estimate(estimate: np.ndarray, state_names: tuple[str, ...]) -> dict[str, float]:
    return {
        FILTERED_FEATURE_BY_STATE[state_name]: float(estimate[index])
        for index, state_name in enumerate(state_names)
        if state_name in FILTERED_FEATURE_BY_STATE
    }


def _rate_from_gain(K: np.ndarray, state_estimate: np.ndarray, lagged_rate: float) -> float:
    z = np.concatenate([state_estimate, np.asarray([lagged_rate], dtype=float)])
    return -float(K @ z)


def _gain_for_period(solution: LQRSolution, period: int) -> np.ndarray:
    if solution.K_path is None:
        return solution.K
    index = min(int(period), solution.K_path.shape[0] - 1)
    return solution.K_path[index]


def _clip_rate(
    rate: float,
    lagged_rate: float,
    *,
    max_abs_rate: float | None,
    max_abs_rate_change: float | None,
) -> float:
    value = float(rate)
    if max_abs_rate_change is not None:
        value = float(np.clip(value, lagged_rate - max_abs_rate_change, lagged_rate + max_abs_rate_change))
    if max_abs_rate is not None:
        value = float(np.clip(value, -max_abs_rate, max_abs_rate))
    return value


def _regularize_square(matrix: np.ndarray, *, floor: float) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, floor)
    return (eigvecs * eigvals) @ eigvecs.T
