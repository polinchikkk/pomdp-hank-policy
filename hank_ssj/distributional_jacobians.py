from __future__ import annotations

import contextlib
import io
import json
import re
from dataclasses import asdict, dataclass, replace
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
    difference_method: str
    strict_mode: bool
    variables: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class TransitionSolveDiagnostics:
    shock_period: int
    shock_size: float
    signed_shock_size: float
    direction: str
    converged: bool
    iterations: int
    max_residual: float
    max_residual_any_iteration: float
    error_message: str


@dataclass(frozen=True)
class DistributionalJacobianBuildResult:
    matrices: dict[str, np.ndarray]
    failed_shifted_transition_periods: tuple[int, ...]
    fallback_method: str
    transition_diagnostics: tuple[TransitionSolveDiagnostics, ...]
    difference_method: str
    signed_derivative_matrices: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class _DistributionalShockSolve:
    deviations: dict[str, np.ndarray] | None
    diagnostics: TransitionSolveDiagnostics


def augment_jacobians_with_distributional_policy_responses(
    *,
    base_jacobians_npz: Path,
    output_npz: Path,
    output_long_csv: Path,
    output_spec_json: Path,
    output_diagnostics_csv: Path | None = None,
    horizon: int | None = None,
    shock_name: str = "monetary_policy_shock",
    shock_size: float = 0.001,
    difference_method: str = "central",
    strict_mode: bool = False,
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
        difference_method=difference_method,
        strict_mode=strict_mode,
        suppress_solver_output=suppress_solver_output,
    )
    matrices = result.matrices
    for variable, matrix in matrices.items():
        padded = np.zeros((inferred_horizon, inferred_horizon), dtype=float)
        padded[:horizon, :horizon] = matrix
        arrays[f"J_{shock_name}_{variable}"] = padded

    arrays["metadata_distributional_jacobians"] = np.array(
        f"Direct distributional responses computed by shifted HANK transition solves ({difference_method} difference)."
    )
    np.savez_compressed(output_npz, **arrays)

    long = _long_form(matrices, shock_name=shock_name)
    long.to_csv(output_long_csv, index=False)
    if output_diagnostics_csv is not None:
        output_diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        transition_diagnostics_to_frame(result).to_csv(output_diagnostics_csv, index=False)
    if result.failed_shifted_transition_periods:
        fallback_note = (
            f"Периоды, где сдвинутый transition solver не сошёлся, заменены Toeplitz-сдвигом "
            f"нулевого HANK-отклика: {list(result.failed_shifted_transition_periods)}."
        )
    else:
        fallback_note = "Все сдвинутые transition solves сошлись; Toeplitz fallback не использовался."
    spec = DistributionalJacobianSpec(
        source_jacobians=str(base_jacobians_npz),
        output_jacobians=str(output_npz),
        output_long_csv=str(output_long_csv),
        horizon=int(horizon),
        shock_name=shock_name,
        shock_size=float(shock_size),
        difference_method=difference_method,
        strict_mode=bool(strict_mode),
        variables=tuple(matrices.keys()),
        note=(
            "Распределительные отклики построены прямым пересчётом HANK transition solver для "
            "сдвинутого шока ставки в каждом периоде. Производная берётся центральной разностью, "
            "если не указан другой difference_method. Это сильнее aggregate-regression fallback, "
            "но всё ещё является локальной HANK/SSJ-проекцией, а не глобальной HANK-оптимизацией. "
            f"{fallback_note}"
        ),
    )
    payload = asdict(spec)
    payload["failed_shifted_transition_periods"] = list(result.failed_shifted_transition_periods)
    payload["fallback_period_count"] = len(result.failed_shifted_transition_periods)
    payload["fallback_method"] = result.fallback_method
    payload["transition_diagnostics"] = [asdict(row) for row in result.transition_diagnostics]
    output_spec_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_distributional_policy_jacobians(
    *,
    horizon: int,
    shock_name: str = "monetary_policy_shock",
    shock_size: float = 0.001,
    difference_method: str = "central",
    strict_mode: bool = False,
    suppress_solver_output: bool = True,
) -> DistributionalJacobianBuildResult:
    if shock_size <= 0.0:
        raise ValueError("shock_size must be positive.")
    if difference_method not in {"central", "forward"}:
        raise ValueError("difference_method must be either 'central' or 'forward'.")
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
    signed_derivative_matrices = {
        direction: {
            variable: np.full((horizon, horizon), np.nan, dtype=float)
            for variable in DISTRIBUTIONAL_JACOBIAN_OUTPUTS
        }
        for direction in ("plus", "minus")
    }
    transition_diagnostics: list[TransitionSolveDiagnostics] = []
    impulse_zero, diagnostics, signed_derivatives = _distributional_derivative_for_shock_period(
        bundle=bundle,
        ss=ss,
        config=config,
        steady=steady,
        shock_name=shock_name,
        shock_size=shock_size,
        shock_period=0,
        horizon=horizon,
        difference_method=difference_method,
        suppress_solver_output=suppress_solver_output,
    )
    transition_diagnostics.extend(diagnostics)
    _fill_signed_derivative_matrices(signed_derivative_matrices, signed_derivatives, shock_period=0)
    if impulse_zero is None:
        raise RuntimeError("Period-0 HANK transition failed; cannot construct Toeplitz fallback response.")
    failed_periods: list[int] = []
    for shock_period in range(horizon):
        if shock_period == 0:
            response = impulse_zero
        else:
            response, diagnostics, signed_derivatives = _distributional_derivative_for_shock_period(
                bundle=bundle,
                ss=ss,
                config=config,
                steady=steady,
                shock_name=shock_name,
                shock_size=shock_size,
                shock_period=shock_period,
                horizon=horizon,
                difference_method=difference_method,
                suppress_solver_output=suppress_solver_output,
            )
            transition_diagnostics.extend(diagnostics)
            _fill_signed_derivative_matrices(signed_derivative_matrices, signed_derivatives, shock_period=shock_period)
        if response is None:
            response = _shift_impulse_response(impulse_zero, shock_period=shock_period, horizon=horizon)
            failed_periods.append(int(shock_period))
        for variable, values in response.items():
            matrices[variable][:, shock_period] = values
    if strict_mode and failed_periods:
        raise RuntimeError(
            "Distributional Jacobian strict mode forbids shifted-transition fallback periods: "
            + ", ".join(str(period) for period in failed_periods)
        )
    return DistributionalJacobianBuildResult(
        matrices=matrices,
        failed_shifted_transition_periods=tuple(failed_periods),
        fallback_method="toeplitz_shift_of_period_0_hank_transition_response",
        transition_diagnostics=tuple(transition_diagnostics),
        difference_method=difference_method,
        signed_derivative_matrices=signed_derivative_matrices,
    )


def _distributional_derivative_for_shock_period(
    *,
    bundle,
    ss,
    config,
    steady: dict[str, float],
    shock_name: str,
    shock_size: float,
    shock_period: int,
    horizon: int,
    difference_method: str,
    suppress_solver_output: bool,
) -> tuple[
    dict[str, np.ndarray] | None,
    tuple[TransitionSolveDiagnostics, ...],
    dict[str, dict[str, np.ndarray]],
]:
    plus = _distributional_deviation_for_shock_period(
        bundle=bundle,
        ss=ss,
        config=config,
        steady=steady,
        shock_name=shock_name,
        shock_size=shock_size,
        signed_shock_size=shock_size,
        direction="plus",
        shock_period=shock_period,
        horizon=horizon,
        suppress_solver_output=suppress_solver_output,
    )
    if difference_method == "forward":
        if plus.deviations is None:
            return None, (plus.diagnostics,), {}
        plus_derivative = {
            variable: values / float(shock_size)
            for variable, values in plus.deviations.items()
        }
        return plus_derivative, (plus.diagnostics,), {"plus": plus_derivative}

    minus = _distributional_deviation_for_shock_period(
        bundle=bundle,
        ss=ss,
        config=config,
        steady=steady,
        shock_name=shock_name,
        shock_size=shock_size,
        signed_shock_size=-float(shock_size),
        direction="minus",
        shock_period=shock_period,
        horizon=horizon,
        suppress_solver_output=suppress_solver_output,
    )
    diagnostics = (plus.diagnostics, minus.diagnostics)
    if plus.deviations is None or minus.deviations is None:
        signed_derivatives: dict[str, dict[str, np.ndarray]] = {}
        if plus.deviations is not None:
            signed_derivatives["plus"] = {
                variable: values / float(shock_size)
                for variable, values in plus.deviations.items()
            }
        if minus.deviations is not None:
            signed_derivatives["minus"] = {
                variable: values / -float(shock_size)
                for variable, values in minus.deviations.items()
            }
        return None, diagnostics, signed_derivatives
    plus_derivative = {
        variable: values / float(shock_size)
        for variable, values in plus.deviations.items()
    }
    minus_derivative = {
        variable: values / -float(shock_size)
        for variable, values in minus.deviations.items()
    }
    return {
        variable: (plus.deviations[variable] - minus.deviations[variable]) / (2.0 * float(shock_size))
        for variable in DISTRIBUTIONAL_JACOBIAN_OUTPUTS
    }, diagnostics, {"plus": plus_derivative, "minus": minus_derivative}


def _distributional_deviation_for_shock_period(
    *,
    bundle,
    ss,
    config,
    steady: dict[str, float],
    shock_name: str,
    shock_size: float,
    signed_shock_size: float,
    direction: str,
    shock_period: int,
    horizon: int,
    suppress_solver_output: bool,
) -> _DistributionalShockSolve:
    shock_path = np.zeros(int(config.shock_T), dtype=float)
    shock_path[int(shock_period)] = float(signed_shock_size)
    transition, diagnostics = _solve_transition_with_diagnostics(
        bundle=bundle,
        shock_inputs={shock_name: shock_path},
        shock_period=shock_period,
        shock_size=shock_size,
        signed_shock_size=signed_shock_size,
        direction=direction,
        suppress_solver_output=suppress_solver_output,
    )
    if transition is None:
        return _DistributionalShockSolve(deviations=None, diagnostics=diagnostics)
    try:
        full_path_levels = household_path_levels(ss, transition)
        mpc_path = compute_mpc_path(full_path_levels)
        distribution = path_distribution_statistics(
            ss,
            full_path_levels,
            config,
            mpc_path=mpc_path,
        ).sort_values("period")
    except Exception as exc:
        return _DistributionalShockSolve(
            deviations=None,
            diagnostics=replace(
                diagnostics,
                converged=False,
                error_message=f"{type(exc).__name__}: {exc}",
            ),
        )
    return _DistributionalShockSolve(
        deviations={
            variable: distribution[source_column].to_numpy(dtype=float)[:horizon] - steady[source_column]
            for variable, source_column in DISTRIBUTIONAL_JACOBIAN_OUTPUTS.items()
        },
        diagnostics=diagnostics,
    )


def _solve_transition_with_diagnostics(
    *,
    bundle,
    shock_inputs: dict[str, np.ndarray],
    shock_period: int,
    shock_size: float,
    signed_shock_size: float,
    direction: str,
    suppress_solver_output: bool,
):
    buffer = io.StringIO()
    transition = None
    error_message = ""
    try:
        with contextlib.redirect_stdout(buffer):
            transition = solve_transition(bundle, shock_inputs)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
    captured = buffer.getvalue()
    if not suppress_solver_output and captured:
        print(captured, end="")
    parsed = _parse_transition_solver_output(captured)
    diagnostics = TransitionSolveDiagnostics(
        shock_period=int(shock_period),
        shock_size=float(shock_size),
        signed_shock_size=float(signed_shock_size),
        direction=direction,
        converged=transition is not None and not error_message,
        iterations=int(parsed["iterations"]),
        max_residual=float(parsed["max_residual"]),
        max_residual_any_iteration=float(parsed["max_residual_any_iteration"]),
        error_message=error_message,
    )
    return transition, diagnostics


def _parse_transition_solver_output(text: str) -> dict[str, float | int]:
    iteration_errors: list[list[float]] = []
    current_errors: list[float] | None = None
    for line in text.splitlines():
        if re.search(r"\bOn iteration\s+\d+", line):
            if current_errors is not None:
                iteration_errors.append(current_errors)
            current_errors = []
            continue
        match = re.search(r"max error for\s+.+?\s+is\s+([0-9.+\-Ee]+)", line)
        if match and current_errors is not None:
            current_errors.append(float(match.group(1)))
    if current_errors is not None:
        iteration_errors.append(current_errors)
    nonempty = [values for values in iteration_errors if values]
    if not nonempty:
        return {
            "iterations": 0,
            "max_residual": np.nan,
            "max_residual_any_iteration": np.nan,
        }
    all_errors = [value for values in nonempty for value in values]
    return {
        "iterations": len(nonempty),
        "max_residual": max(nonempty[-1]),
        "max_residual_any_iteration": max(all_errors),
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


def _fill_signed_derivative_matrices(
    matrices: dict[str, dict[str, np.ndarray]],
    signed_derivatives: dict[str, dict[str, np.ndarray]],
    *,
    shock_period: int,
) -> None:
    for direction, derivative_by_variable in signed_derivatives.items():
        if direction not in matrices:
            continue
        for variable, values in derivative_by_variable.items():
            matrices[direction][variable][:, int(shock_period)] = values


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


def transition_diagnostics_to_frame(
    result_or_diagnostics: DistributionalJacobianBuildResult | tuple[TransitionSolveDiagnostics, ...] | list[TransitionSolveDiagnostics],
) -> pd.DataFrame:
    diagnostics = (
        result_or_diagnostics.transition_diagnostics
        if isinstance(result_or_diagnostics, DistributionalJacobianBuildResult)
        else tuple(result_or_diagnostics)
    )
    return pd.DataFrame([asdict(row) for row in diagnostics])


def _infer_horizon(arrays: dict[str, np.ndarray]) -> int:
    matrix_shapes = [array.shape for key, array in arrays.items() if key.startswith("J_") and array.ndim == 2]
    if not matrix_shapes:
        raise ValueError("Base Jacobian bundle does not contain any matrix keys.")
    return int(min(shape[0] for shape in matrix_shapes))
