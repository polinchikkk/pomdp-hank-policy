from __future__ import annotations

import subprocess
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import RBCParameters, RBCSteadyState, equilibrium_residuals, observables_from_state
from .solver import LinearRBCSolution, numerical_jacobian

EXTERNAL_GENSYS_PACKAGE = "dsge"
EXTERNAL_GENSYS_VERSION = "0.1.3"
EXTERNAL_GENSYS_SOURCE = "dsge/gensys.py"


@dataclass(frozen=True)
class ExternalIRFBenchmark:
    source_package: str
    source_version: str
    source_file: str
    rc: tuple[int, int]
    existence: bool
    uniqueness: bool
    transition_matrix: np.ndarray
    impact_matrix: np.ndarray
    qz_irf: pd.DataFrame
    gensys_irf: pd.DataFrame
    irf_comparison: pd.DataFrame
    max_abs_diff: float
    rms_diff: float


def _benchmark_cache_dir() -> Path:
    return Path.home() / ".cache" / "pomdp-hank-policy" / "external_benchmarks"


def _ensure_external_gensys_wheel() -> Path:
    cache_dir = _benchmark_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = f"{EXTERNAL_GENSYS_PACKAGE}-{EXTERNAL_GENSYS_VERSION}-py3-none-any.whl"
    wheel_path = cache_dir / wheel_name
    if wheel_path.exists():
        return wheel_path

    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--no-deps",
        "--dest",
        str(cache_dir),
        f"{EXTERNAL_GENSYS_PACKAGE}=={EXTERNAL_GENSYS_VERSION}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not wheel_path.exists():
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Failed to download external gensys benchmark package: {stderr}")
    return wheel_path


def _load_external_gensys():
    wheel_path = _ensure_external_gensys_wheel()
    with zipfile.ZipFile(wheel_path) as archive:
        source = archive.read(EXTERNAL_GENSYS_SOURCE).decode("utf-8")
    module = types.ModuleType("external_dsge_gensys")
    exec(compile(source, EXTERNAL_GENSYS_SOURCE, "exec"), module.__dict__)
    return module.gensys


def _shock_jacobian(
    params: RBCParameters,
    steady_state: RBCSteadyState,
) -> np.ndarray:
    zero_state = np.zeros(2, dtype=float)
    zero_control = np.zeros(2, dtype=float)
    return numerical_jacobian(
        lambda vector: equilibrium_residuals(
            params=params,
            steady_state=steady_state,
            state_t=zero_state,
            control_t=zero_control,
            state_tp1=zero_state,
            control_tp1=zero_control,
            shock_tp1=vector[0],
        ),
        point=np.zeros(1, dtype=float),
    )


def _qz_stacked_irf(
    solution: LinearRBCSolution,
    horizon: int,
    shock_size: float,
) -> np.ndarray:
    stacked = np.zeros((horizon, 4), dtype=float)
    state = solution.shock_vector * shock_size
    for period in range(horizon):
        control = solution.controls(state)
        stacked[period, :2] = state
        stacked[period, 2:] = control
        state = solution.transition_matrix @ state
    return stacked


def _gensys_stacked_irf(
    transition_matrix: np.ndarray,
    impact_matrix: np.ndarray,
    horizon: int,
    shock_size: float,
) -> np.ndarray:
    stacked = np.zeros((horizon, transition_matrix.shape[0]), dtype=float)
    stacked[0] = impact_matrix[:, 0] * shock_size
    for period in range(horizon - 1):
        stacked[period + 1] = transition_matrix @ stacked[period]
    return stacked


def _stacked_irf_to_frame(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    stacked: np.ndarray,
    shock_size: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for period in range(stacked.shape[0]):
        state = stacked[period, :2]
        control = stacked[period, 2:]
        row: dict[str, float | int] = {
            "t": period,
            "epsilon_t": shock_size if period == 0 else 0.0,
            "shock_impact_t": float(params.sigma * shock_size if period == 0 else 0.0),
        }
        row.update(
            observables_from_state(
                params=params,
                steady_state=steady_state,
                state=state,
                control=control,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison_frame(qz_irf: pd.DataFrame, gensys_irf: pd.DataFrame) -> pd.DataFrame:
    comparison_columns = [
        "z",
        "log_k_dev",
        "log_c_dev",
        "log_n_dev",
        "log_y_dev",
        "log_i_dev",
    ]
    rows: list[dict[str, float | int]] = []
    for period in range(len(qz_irf)):
        row: dict[str, float | int] = {"t": int(qz_irf.iloc[period]["t"])}
        for column in comparison_columns:
            qz_value = float(qz_irf.iloc[period][column])
            gensys_value = float(gensys_irf.iloc[period][column])
            row[f"qz_{column}"] = qz_value
            row[f"gensys_{column}"] = gensys_value
            row[f"diff_{column}"] = qz_value - gensys_value
        rows.append(row)
    return pd.DataFrame(rows)


def run_external_gensys_irf_benchmark(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    solution: LinearRBCSolution,
    horizon: int,
    shock_size: float = 1.0,
) -> ExternalIRFBenchmark:
    gensys = _load_external_gensys()
    jacobians = solution.jacobians
    future_block = np.hstack([jacobians["f_state_tp1"], jacobians["f_control_tp1"]])
    current_block = np.hstack([jacobians["f_state_t"], jacobians["f_control_t"]])
    shock_block = _shock_jacobian(params=params, steady_state=steady_state)

    # In gensys form, expectational errors attach only to the forward-looking controls.
    transition_matrix, impact_matrix, rc = gensys(
        future_block,
        -current_block,
        -shock_block,
        jacobians["f_control_tp1"],
    )

    transition_matrix = np.real_if_close(transition_matrix, tol=1000).astype(float)
    impact_matrix = np.real_if_close(impact_matrix, tol=1000).astype(float)
    rc_tuple = (int(rc[0]), int(rc[1]))

    qz_stacked = _qz_stacked_irf(solution=solution, horizon=horizon, shock_size=shock_size)
    gensys_stacked = _gensys_stacked_irf(
        transition_matrix=transition_matrix,
        impact_matrix=impact_matrix,
        horizon=horizon,
        shock_size=shock_size,
    )

    qz_irf = _stacked_irf_to_frame(
        params=params,
        steady_state=steady_state,
        stacked=qz_stacked,
        shock_size=shock_size,
    )
    gensys_irf = _stacked_irf_to_frame(
        params=params,
        steady_state=steady_state,
        stacked=gensys_stacked,
        shock_size=shock_size,
    )
    irf_comparison = _comparison_frame(qz_irf=qz_irf, gensys_irf=gensys_irf)
    diff_matrix = irf_comparison.filter(like="diff_").to_numpy(dtype=float)

    return ExternalIRFBenchmark(
        source_package=EXTERNAL_GENSYS_PACKAGE,
        source_version=EXTERNAL_GENSYS_VERSION,
        source_file=EXTERNAL_GENSYS_SOURCE,
        rc=rc_tuple,
        existence=bool(rc_tuple[0] == 1),
        uniqueness=bool(rc_tuple[1] == 1),
        transition_matrix=transition_matrix,
        impact_matrix=impact_matrix,
        qz_irf=qz_irf,
        gensys_irf=gensys_irf,
        irf_comparison=irf_comparison,
        max_abs_diff=float(np.max(np.abs(diff_matrix))),
        rms_diff=float(np.sqrt(np.mean(np.square(diff_matrix)))),
    )
