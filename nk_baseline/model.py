from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

STATE_NAMES = ("r_n", "u", "nu")
CONTROL_NAMES = ("x", "pi", "i")
SHOCK_NAMES = ("demand", "costpush", "monetary")


@dataclass(frozen=True)
class NKParameters:
    beta: float = 0.99
    sigma: float = 1.0
    kappa: float = 0.1
    phi_pi: float = 1.5
    phi_x: float = 0.5
    rho_r: float = 0.8
    rho_u: float = 0.5
    rho_nu: float = 0.5
    sigma_r: float = 0.01
    sigma_u: float = 0.01
    sigma_nu: float = 0.0025

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def model_equations() -> list[str]:
    return [
        "x_t = E_t[x_{t+1}] - (1/sigma) * (i_t - E_t[pi_{t+1}] - r_t^n)",
        "pi_t = beta * E_t[pi_{t+1}] + kappa * x_t + u_t",
        "i_t = phi_pi * pi_t + phi_x * x_t + nu_t",
        "r^n_{t+1} = rho_r * r^n_t + sigma_r * eps^r_{t+1}",
        "u_{t+1} = rho_u * u_t + sigma_u * eps^u_{t+1}",
        "nu_{t+1} = rho_nu * nu_t + sigma_nu * eps^nu_{t+1}",
    ]


def nk_system_matrices(params: NKParameters) -> dict[str, np.ndarray]:
    f_state_t = np.array(
        [
            [-1.0 / params.sigma, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-params.rho_r, 0.0, 0.0],
            [0.0, -params.rho_u, 0.0],
            [0.0, 0.0, -params.rho_nu],
        ],
        dtype=float,
    )
    f_control_t = np.array(
        [
            [1.0, 0.0, 1.0 / params.sigma],
            [-params.kappa, 1.0, 0.0],
            [-params.phi_x, -params.phi_pi, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    f_state_tp1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    f_control_tp1 = np.array(
        [
            [-1.0, -1.0 / params.sigma, 0.0],
            [0.0, -params.beta, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    f_shock_tp1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-params.sigma_r, 0.0, 0.0],
            [0.0, -params.sigma_u, 0.0],
            [0.0, 0.0, -params.sigma_nu],
        ],
        dtype=float,
    )
    shock_impact_matrix = np.diag([params.sigma_r, params.sigma_u, params.sigma_nu]).astype(float)

    return {
        "f_state_t": f_state_t,
        "f_control_t": f_control_t,
        "f_state_tp1": f_state_tp1,
        "f_control_tp1": f_control_tp1,
        "f_shock_tp1": f_shock_tp1,
        "shock_impact_matrix": shock_impact_matrix,
    }


def model_spec_payload(params: NKParameters) -> dict:
    return {
        "model_name": "small_linear_new_keynesian_policy_baseline",
        "linearization_point": {
            "x": 0.0,
            "pi": 0.0,
            "i": 0.0,
            "r_n": 0.0,
            "u": 0.0,
            "nu": 0.0,
        },
        "state_variables": list(STATE_NAMES),
        "control_variables": list(CONTROL_NAMES),
        "shock_names": list(SHOCK_NAMES),
        "policy_rule": "i_t = phi_pi * pi_t + phi_x * x_t + nu_t",
        "equations": model_equations(),
        "parameters": params.to_dict(),
    }
