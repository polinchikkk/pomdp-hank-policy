from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

STATE_NAMES = ("log_k_dev", "z")
CONTROL_NAMES = ("log_c_dev", "log_n_dev")
OBSERVABLE_NAMES = ("y", "c", "i", "k", "n")


@dataclass(frozen=True)
class RBCParameters:
    beta: float = 0.99
    alpha: float = 0.36
    delta: float = 0.025
    rho: float = 0.95
    sigma: float = 0.01
    frisch_inverse: float = 1.0
    labor_ss: float = 0.33

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RBCSteadyState:
    k: float
    c: float
    n: float
    y: float
    i: float
    z: float
    psi: float
    mpk: float
    wage: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_steady_state(params: RBCParameters) -> RBCSteadyState:
    mpk = 1.0 / params.beta - (1.0 - params.delta)
    capital_to_labor = (params.alpha / mpk) ** (1.0 / (1.0 - params.alpha))
    n = params.labor_ss
    k = capital_to_labor * n
    y = k**params.alpha * n ** (1.0 - params.alpha)
    i = params.delta * k
    c = y - i
    wage = (1.0 - params.alpha) * y / n
    psi = wage / (c * n**params.frisch_inverse)
    return RBCSteadyState(k=k, c=c, n=n, y=y, i=i, z=0.0, psi=psi, mpk=mpk, wage=wage)


def unpack_transformed_variables(
    steady_state: RBCSteadyState,
    state: np.ndarray,
    control: np.ndarray,
) -> tuple[float, float, float, float]:
    k = steady_state.k * np.exp(state[0])
    z = state[1]
    c = steady_state.c * np.exp(control[0])
    n = steady_state.n * np.exp(control[1])
    return k, z, c, n


def production(
    params: RBCParameters,
    k: float,
    n: float,
    z: float,
) -> float:
    return np.exp(z) * k**params.alpha * n ** (1.0 - params.alpha)


def observables_from_state(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    state: np.ndarray,
    control: np.ndarray,
) -> dict[str, float]:
    k, z, c, n = unpack_transformed_variables(steady_state, state, control)
    y = production(params=params, k=k, n=n, z=z)
    i = y - c
    return {
        "k": k,
        "z": z,
        "c": c,
        "n": n,
        "y": y,
        "i": i,
        "log_k_dev": state[0],
        "log_c_dev": control[0],
        "log_n_dev": control[1],
        "log_y_dev": np.log(y / steady_state.y),
        "log_i_dev": np.log(i / steady_state.i),
    }


def equilibrium_residuals(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    state_t: np.ndarray,
    control_t: np.ndarray,
    state_tp1: np.ndarray,
    control_tp1: np.ndarray,
    shock_tp1: float = 0.0,
) -> np.ndarray:
    k_t, z_t, c_t, n_t = unpack_transformed_variables(
        steady_state=steady_state,
        state=state_t,
        control=control_t,
    )
    k_tp1, z_tp1, c_tp1, n_tp1 = unpack_transformed_variables(
        steady_state=steady_state,
        state=state_tp1,
        control=control_tp1,
    )

    y_t = production(params=params, k=k_t, n=n_t, z=z_t)
    mpk_tp1 = params.alpha * np.exp(z_tp1) * k_tp1 ** (params.alpha - 1.0) * n_tp1 ** (
        1.0 - params.alpha
    )
    wage_t = (1.0 - params.alpha) * y_t / n_t

    return np.array(
        [
            1.0 / c_t - params.beta * (1.0 / c_tp1) * (mpk_tp1 + 1.0 - params.delta),
            steady_state.psi * n_t**params.frisch_inverse - wage_t / c_t,
            k_tp1 - ((1.0 - params.delta) * k_t + y_t - c_t),
            z_tp1 - params.rho * z_t - params.sigma * shock_tp1,
        ],
        dtype=float,
    )
