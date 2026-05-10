from __future__ import annotations

import contextlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hank.calibration import default_calibration
from hank.distribution import household_path_levels, path_distribution_statistics, stationary_distribution
from hank.grids import state_mesh
from hank.household_solver import compute_mpc, compute_mpc_path
from hank.steady_state import solve_steady_state
from hank.transition import solve_transition


DISTRIBUTIONAL_JACOBIAN_OUTPUTS = {
    "mean_mpc_centered": "mean_mpc",
    "share_low_liquidity_centered": "share_low_liquidity",
    "interest_exposure_centered": "interest_exposure",
}


@dataclass(frozen=True)
class DistributionalJacobianSpec:
    source_jacobians: str
    output_jacobians: str
    output_long_csv: str
    horizon: int
    shock_name: str
    shock_size: float
    variables: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class DistributionalJacobianBuildResult:
    matrices: dict[str, np.ndarray]
    failed_shifted_transition_periods: tuple[int, ...]
    fallback_method: str


def augment_jacobians_with_distributional_policy_responses(
    *,
    base_jacobians_npz: Path,
    output_npz: Path,
    output_long_csv: Path,
    output_spec_json: Path,
    horizon: int | None = None,
    shock_name: str = "monetary_policy_shock",
    shock_size: float = 0.001,
    suppress_solver_output: bool = True,
) -> DistributionalJacobianBuildResult:
    """Add direct HANK transition responses of distributional statistics to an SSJ bundle.

    The existing aggregate SSJ bundle contains full matrices for aggregate variables.  Distributional
    statistics are not outputs of the sequence-space Jacobian solver, so here we compute their local
    response by shifting a one-period policy shock through the HANK transition solver and recomputing
    distributional statistics from household paths.  This is slower than exporting the Jacobian
    directly from the model, but it avoids the weaker aggregate-regression fallback.
    """

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_long_csv.parent.mkdir(parents=True, exist_ok=True)
    output_spec_json.parent.mkdir(parents=True, exist_ok=True)

    with np.load(base_jacobians_npz, allow_pickle=True) as bundle:
        arrays = {key: np.asarray(bundle[key]) for key in bundle.files}

    inferred_horizon = _infer_horizon(arrays)
    horizon = inferred_horizon if horizon is None else int(horizon)
    if horizon > inferred_horizon:
        raise ValueError(f"Requested horizon {horizon} exceeds base Jacobian horizon {inferred_horizon}.")

    result = build_distributional_policy_jacobians(
        horizon=horizon,
        shock_name=shock_name,
        shock_size=shock_size,
        suppress_solver_output=suppress_solver_output,
    )
    matrices = result.matrices
    for variable, matrix in matrices.items():
        padded = np.zeros((inferred_horizon, inferred_horizon), dtype=float)
        padded[:horizon, :horizon] = matrix
        arrays[f"J_{shock_name}_{variable}"] = padded

    arrays["metadata_distributional_jacobians"] = np.array(
        "Direct distributional responses computed by shifted HANK transition solves."
    )
    np.savez_compressed(output_npz, **arrays)

    long = _long_form(matrices, shock_name=shock_name)
    long.to_csv(output_long_csv, index=False)
    spec = DistributionalJacobianSpec(
        source_jacobians=str(base_jacobians_npz),
        output_jacobians=str(output_npz),
        output_long_csv=str(output_long_csv),
        horizon=int(horizon),
        shock_name=shock_name,
        shock_size=float(shock_size),
        variables=tuple(matrices.keys()),
        note=(
            "Распределительные отклики построены прямым пересчётом HANK transition solver для "
            "сдвинутого шока ставки в каждом периоде. Это сильнее aggregate-regression fallback, "
            "но всё ещё является локальной HANK/SSJ-проекцией, а не глобальной HANK-оптимизацией. "
            f"Периоды, где сдвинутый transition solver не сошёлся, заменены Toeplitz-сдвигом "
            f"нулевого HANK-отклика: {list(result.failed_shifted_transition_periods)}."
        ),
    )
    payload = asdict(spec)
    payload["failed_shifted_transition_periods"] = list(result.failed_shifted_transition_periods)
    payload["fallback_method"] = result.fallback_method
    output_spec_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_distributional_policy_jacobians(
    *,
    horizon: int,
    shock_name: str = "monetary_policy_shock",
    shock_size: float = 0.001,
    suppress_solver_output: bool = True,
) -> DistributionalJacobianBuildResult:
    config = default_calibration()
    if horizon > int(config.shock_T):
        raise ValueError(f"Requested horizon {horizon} exceeds HANK transition horizon {config.shock_T}.")
    bundle = solve_steady_state(config)
    ss = bundle["ss"]
    steady = _steady_distributional_values(ss, config)
    matrices = {
        variable: np.zeros((horizon, horizon), dtype=float)
        for variable in DISTRIBUTIONAL_JACOBIAN_OUTPUTS
    }
    impulse_zero = _distributional_response_for_shock_period(
        bundle=bundle,
        ss=ss,
        config=config,
        steady=steady,
        shock_name=shock_name,
        shock_size=shock_size,
        shock_period=0,
        horizon=horizon,
        suppress_solver_output=suppress_solver_output,
    )
    failed_periods: list[int] = []
    for shock_period in range(horizon):
        try:
            response = _distributional_response_for_shock_period(
                bundle=bundle,
                ss=ss,
                config=config,
                steady=steady,
                shock_name=shock_name,
                shock_size=shock_size,
                shock_period=shock_period,
                horizon=horizon,
                suppress_solver_output=suppress_solver_output,
            )
        except Exception:
            response = _shift_impulse_response(impulse_zero, shock_period=shock_period, horizon=horizon)
            failed_periods.append(int(shock_period))
        for variable, values in response.items():
            matrices[variable][:, shock_period] = values
    return DistributionalJacobianBuildResult(
        matrices=matrices,
        failed_shifted_transition_periods=tuple(failed_periods),
        fallback_method="toeplitz_shift_of_period_0_hank_transition_response",
    )


def _distributional_response_for_shock_period(
    *,
    bundle,
    ss,
    config,
    steady: dict[str, float],
    shock_name: str,
    shock_size: float,
    shock_period: int,
    horizon: int,
    suppress_solver_output: bool,
) -> dict[str, np.ndarray]:
    shock_path = np.zeros(int(config.shock_T), dtype=float)
    shock_path[int(shock_period)] = float(shock_size)
    if suppress_solver_output:
        with contextlib.redirect_stdout(io.StringIO()):
            transition = solve_transition(bundle, {shock_name: shock_path})
    else:
        transition = solve_transition(bundle, {shock_name: shock_path})
    full_path_levels = household_path_levels(ss, transition)
    mpc_path = compute_mpc_path(full_path_levels)
    distribution = path_distribution_statistics(
        ss,
        full_path_levels,
        config,
        mpc_path=mpc_path,
    ).sort_values("period")
    return {
        variable: (distribution[source_column].to_numpy(dtype=float)[:horizon] - steady[source_column]) / float(shock_size)
        for variable, source_column in DISTRIBUTIONAL_JACOBIAN_OUTPUTS.items()
    }


def _shift_impulse_response(
    impulse_zero: dict[str, np.ndarray],
    *,
    shock_period: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    shifted: dict[str, np.ndarray] = {}
    for variable, response in impulse_zero.items():
        values = np.zeros(horizon, dtype=float)
        available = horizon - int(shock_period)
        if available > 0:
            values[int(shock_period):] = response[:available]
        shifted[variable] = values
    return shifted


def required_distributional_jacobian_keys(shock_name: str = "monetary_policy_shock") -> tuple[str, ...]:
    return tuple(f"J_{shock_name}_{variable}" for variable in DISTRIBUTIONAL_JACOBIAN_OUTPUTS)


def has_direct_distributional_jacobians(jacobians_npz: Path, *, shock_name: str = "monetary_policy_shock") -> bool:
    with np.load(jacobians_npz, allow_pickle=True) as bundle:
        return all(key in bundle.files for key in required_distributional_jacobian_keys(shock_name))


def _steady_distributional_values(ss, config) -> dict[str, float]:
    distribution = stationary_distribution(ss)
    mpc = compute_mpc(ss)
    mesh = state_mesh(ss)
    return {
        "mean_mpc": float(np.sum(distribution * mpc)),
        "share_low_liquidity": float(np.sum(distribution * (mesh["b"] <= config.low_liquidity_threshold))),
        "interest_exposure": float(np.sum(distribution * mesh["b"] * mpc)),
    }


def _long_form(matrices: dict[str, np.ndarray], *, shock_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable, matrix in matrices.items():
        for response_period in range(matrix.shape[0]):
            for shock_period in range(matrix.shape[1]):
                rows.append(
                    {
                        "output": variable,
                        "input": shock_name,
                        "response_period": response_period,
                        "shock_period": shock_period,
                        "value": float(matrix[response_period, shock_period]),
                    }
                )
    return pd.DataFrame(rows)


def _infer_horizon(arrays: dict[str, np.ndarray]) -> int:
    matrix_shapes = [array.shape for key, array in arrays.items() if key.startswith("J_") and array.ndim == 2]
    if not matrix_shapes:
        raise ValueError("Base Jacobian bundle does not contain any matrix keys.")
    return int(min(shape[0] for shape in matrix_shapes))
