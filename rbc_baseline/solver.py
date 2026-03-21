from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import ordqz

from .model import RBCParameters, RBCSteadyState, equilibrium_residuals

JACOBIAN_NAMES = ("f_state_t", "f_control_t", "f_state_tp1", "f_control_tp1")


@dataclass(frozen=True)
class LinearRBCSolution:
    policy_matrix: np.ndarray
    transition_matrix: np.ndarray
    shock_vector: np.ndarray
    jacobians: dict[str, np.ndarray]
    residual_norm: float
    spectral_radius: float
    solver_name: str
    generalized_eigenvalues: np.ndarray
    stable_root_count: int
    roots_outside_unit_circle_count: int
    infinite_root_count: int
    bk_condition_satisfied: bool
    invariant_block_condition_number: float

    def controls(self, state: np.ndarray) -> np.ndarray:
        return self.policy_matrix @ state

    def next_state(self, state: np.ndarray, innovation: float = 0.0) -> np.ndarray:
        return self.transition_matrix @ state + self.shock_vector * innovation


def numerical_jacobian(
    function,
    point: np.ndarray,
    step_size: float = 1e-6,
) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    base_value = function(point)
    jacobian = np.zeros((base_value.size, point.size), dtype=float)

    for index in range(point.size):
        step = np.zeros_like(point)
        step[index] = step_size
        jacobian[:, index] = (function(point + step) - function(point - step)) / (2.0 * step_size)

    return jacobian


def linearize_equilibrium_system(
    params: RBCParameters,
    steady_state: RBCSteadyState,
) -> dict[str, np.ndarray]:
    zero_state = np.zeros(2, dtype=float)
    zero_control = np.zeros(2, dtype=float)

    f_state_t = numerical_jacobian(
        lambda vector: equilibrium_residuals(
            params=params,
            steady_state=steady_state,
            state_t=vector,
            control_t=zero_control,
            state_tp1=zero_state,
            control_tp1=zero_control,
        ),
        point=zero_state,
    )
    f_control_t = numerical_jacobian(
        lambda vector: equilibrium_residuals(
            params=params,
            steady_state=steady_state,
            state_t=zero_state,
            control_t=vector,
            state_tp1=zero_state,
            control_tp1=zero_control,
        ),
        point=zero_control,
    )
    f_state_tp1 = numerical_jacobian(
        lambda vector: equilibrium_residuals(
            params=params,
            steady_state=steady_state,
            state_t=zero_state,
            control_t=zero_control,
            state_tp1=vector,
            control_tp1=zero_control,
        ),
        point=zero_state,
    )
    f_control_tp1 = numerical_jacobian(
        lambda vector: equilibrium_residuals(
            params=params,
            steady_state=steady_state,
            state_t=zero_state,
            control_t=zero_control,
            state_tp1=zero_state,
            control_tp1=vector,
        ),
        point=zero_control,
    )
    return {
        "f_state_t": f_state_t,
        "f_control_t": f_control_t,
        "f_state_tp1": f_state_tp1,
        "f_control_tp1": f_control_tp1,
    }


def _solve_via_generalized_schur(
    f_state_t: np.ndarray,
    f_control_t: np.ndarray,
    f_state_tp1: np.ndarray,
    f_control_tp1: np.ndarray,
    stable_tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, float]:
    # Klein/Sims-style QZ setup for a one-lead linear rational expectations system:
    # [f_state_t f_control_t] w_t = -[f_state_tp1 f_control_tp1] w_{t+1}.
    state_dim = f_state_t.shape[1]

    lhs_matrix = np.hstack([f_state_t, f_control_t])
    rhs_matrix = -np.hstack([f_state_tp1, f_control_tp1])

    schur_s, schur_t, alpha, beta, _, right_vectors = ordqz(
        lhs_matrix,
        rhs_matrix,
        sort="iuc",
        output="complex",
    )

    generalized_eigenvalues = np.divide(
        alpha,
        beta,
        out=np.full(alpha.shape, np.inf + 0.0j, dtype=complex),
        where=np.abs(beta) > 1e-12,
    )
    finite_mask = np.isfinite(generalized_eigenvalues)
    stable_mask = finite_mask & (np.abs(generalized_eigenvalues) < 1.0 - stable_tolerance)
    outside_unit_circle_mask = finite_mask & (np.abs(generalized_eigenvalues) > 1.0 + stable_tolerance)
    infinite_root_mask = ~finite_mask
    stable_root_count = int(np.sum(stable_mask))
    roots_outside_unit_circle_count = int(np.sum(outside_unit_circle_mask))
    infinite_root_count = int(np.sum(infinite_root_mask))

    if stable_root_count != state_dim:
        raise RuntimeError(
            "Blanchard-Kahn condition failed: "
            f"expected {state_dim} stable roots, found {stable_root_count}."
        )

    z11 = right_vectors[:state_dim, :state_dim]
    z21 = right_vectors[state_dim:, :state_dim]
    invariant_block_condition_number = float(np.linalg.cond(z11))
    if not np.isfinite(invariant_block_condition_number) or invariant_block_condition_number > 1e10:
        raise RuntimeError(
            "The stable invariant subspace is ill-conditioned; "
            "cannot recover a reliable policy matrix."
        )

    stable_s = schur_s[:state_dim, :state_dim]
    stable_t = schur_t[:state_dim, :state_dim]

    policy_matrix = np.linalg.solve(z11.T, z21.T).T
    transition_core = np.linalg.solve(stable_t, stable_s)
    transition_matrix = np.linalg.solve(z11.T, (z11 @ transition_core).T).T

    return (
        np.real_if_close(policy_matrix, tol=1000).astype(float),
        np.real_if_close(transition_matrix, tol=1000).astype(float),
        generalized_eigenvalues,
        stable_root_count,
        roots_outside_unit_circle_count,
        infinite_root_count,
        invariant_block_condition_number,
    )


def solve_linear_policy(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    tolerance: float = 1e-10,
) -> LinearRBCSolution:
    jacobians = linearize_equilibrium_system(params=params, steady_state=steady_state)

    (
        policy_matrix,
        transition_matrix,
        generalized_eigenvalues,
        stable_root_count,
        roots_outside_unit_circle_count,
        infinite_root_count,
        invariant_block_condition_number,
    ) = _solve_via_generalized_schur(
        f_state_t=jacobians["f_state_t"],
        f_control_t=jacobians["f_control_t"],
        f_state_tp1=jacobians["f_state_tp1"],
        f_control_tp1=jacobians["f_control_tp1"],
    )

    residual_matrix = (
        jacobians["f_state_t"]
        + jacobians["f_control_t"] @ policy_matrix
        + jacobians["f_state_tp1"] @ transition_matrix
        + jacobians["f_control_tp1"] @ (policy_matrix @ transition_matrix)
    )
    residual_norm = float(np.max(np.abs(residual_matrix)))
    if residual_norm >= tolerance:
        raise RuntimeError(
            "Generalized Schur solver returned a policy system with residuals above tolerance."
        )

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition_matrix))))

    return LinearRBCSolution(
        policy_matrix=policy_matrix,
        transition_matrix=transition_matrix,
        shock_vector=np.array([0.0, params.sigma], dtype=float),
        jacobians=jacobians,
        residual_norm=residual_norm,
        spectral_radius=spectral_radius,
        solver_name="generalized_schur_qz",
        generalized_eigenvalues=generalized_eigenvalues,
        stable_root_count=stable_root_count,
        roots_outside_unit_circle_count=roots_outside_unit_circle_count,
        infinite_root_count=infinite_root_count,
        bk_condition_satisfied=True,
        invariant_block_condition_number=invariant_block_condition_number,
    )
