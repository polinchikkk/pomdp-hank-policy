from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import ordqz

from .model import NKParameters, nk_system_matrices


@dataclass(frozen=True)
class LinearNKSolution:
    policy_matrix: np.ndarray
    transition_matrix: np.ndarray
    shock_matrix: np.ndarray
    system_matrices: dict[str, np.ndarray]
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

    def next_state(self, state: np.ndarray, innovation: np.ndarray | None = None) -> np.ndarray:
        if innovation is None:
            innovation = np.zeros(self.shock_matrix.shape[1], dtype=float)
        return self.transition_matrix @ state + self.shock_matrix @ innovation


def _qz_decomposition(
    system_matrices: dict[str, np.ndarray],
    stable_tolerance: float = 1e-8,
) -> dict:
    f_state_t = system_matrices["f_state_t"]
    f_control_t = system_matrices["f_control_t"]
    f_state_tp1 = system_matrices["f_state_tp1"]
    f_control_tp1 = system_matrices["f_control_tp1"]

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
    bk_condition_satisfied = stable_root_count == state_dim

    result = {
        "generalized_eigenvalues": generalized_eigenvalues,
        "stable_root_count": stable_root_count,
        "roots_outside_unit_circle_count": roots_outside_unit_circle_count,
        "infinite_root_count": infinite_root_count,
        "bk_condition_satisfied": bk_condition_satisfied,
        "policy_matrix": None,
        "transition_matrix": None,
        "invariant_block_condition_number": np.inf,
    }
    if not bk_condition_satisfied:
        return result

    z11 = right_vectors[:state_dim, :state_dim]
    z21 = right_vectors[state_dim:, :state_dim]
    invariant_block_condition_number = float(np.linalg.cond(z11))
    result["invariant_block_condition_number"] = invariant_block_condition_number
    if not np.isfinite(invariant_block_condition_number) or invariant_block_condition_number > 1e10:
        return result

    stable_s = schur_s[:state_dim, :state_dim]
    stable_t = schur_t[:state_dim, :state_dim]
    policy_matrix = np.linalg.solve(z11.T, z21.T).T
    transition_core = np.linalg.solve(stable_t, stable_s)
    transition_matrix = np.linalg.solve(z11.T, (z11 @ transition_core).T).T

    result["policy_matrix"] = np.real_if_close(policy_matrix, tol=1000).astype(float)
    result["transition_matrix"] = np.real_if_close(transition_matrix, tol=1000).astype(float)
    return result


def determinacy_diagnostics(params: NKParameters) -> dict[str, float | int | bool]:
    system_matrices = nk_system_matrices(params)
    qz_result = _qz_decomposition(system_matrices=system_matrices)
    transition_matrix = qz_result["transition_matrix"]
    spectral_radius = None
    if transition_matrix is not None:
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition_matrix))))

    return {
        "stable_root_count": qz_result["stable_root_count"],
        "roots_outside_unit_circle_count": qz_result["roots_outside_unit_circle_count"],
        "infinite_root_count": qz_result["infinite_root_count"],
        "bk_condition_satisfied": bool(qz_result["bk_condition_satisfied"]),
        "invariant_block_condition_number": float(qz_result["invariant_block_condition_number"]),
        "spectral_radius": spectral_radius,
        "determinate": bool(
            qz_result["bk_condition_satisfied"] and qz_result["transition_matrix"] is not None
        ),
    }


def solve_linear_nk_model(
    params: NKParameters,
    tolerance: float = 1e-10,
) -> LinearNKSolution:
    system_matrices = nk_system_matrices(params=params)
    qz_result = _qz_decomposition(system_matrices=system_matrices)
    if qz_result["transition_matrix"] is None or qz_result["policy_matrix"] is None:
        raise RuntimeError("The NK model is indeterminate under the current Taylor-rule parameters.")

    residual_matrix = (
        system_matrices["f_state_t"]
        + system_matrices["f_control_t"] @ qz_result["policy_matrix"]
        + system_matrices["f_state_tp1"] @ qz_result["transition_matrix"]
        + system_matrices["f_control_tp1"] @ (qz_result["policy_matrix"] @ qz_result["transition_matrix"])
    )
    residual_norm = float(np.max(np.abs(residual_matrix)))
    if residual_norm >= tolerance:
        raise RuntimeError("The NK QZ solver returned residuals above tolerance.")

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(qz_result["transition_matrix"]))))
    return LinearNKSolution(
        policy_matrix=qz_result["policy_matrix"],
        transition_matrix=qz_result["transition_matrix"],
        shock_matrix=system_matrices["shock_impact_matrix"],
        system_matrices=system_matrices,
        residual_norm=residual_norm,
        spectral_radius=spectral_radius,
        solver_name="generalized_schur_qz",
        generalized_eigenvalues=qz_result["generalized_eigenvalues"],
        stable_root_count=qz_result["stable_root_count"],
        roots_outside_unit_circle_count=qz_result["roots_outside_unit_circle_count"],
        infinite_root_count=qz_result["infinite_root_count"],
        bk_condition_satisfied=bool(qz_result["bk_condition_satisfied"]),
        invariant_block_condition_number=float(qz_result["invariant_block_condition_number"]),
    )
