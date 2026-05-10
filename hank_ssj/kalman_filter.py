from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .state_space import StateSpaceSpec, spectral_radius, state_space_spec_to_jsonable


DEFAULT_STATE_NAMES: tuple[str, ...] = (
    "pi",
    "Y",
    "C",
    "mean_mpc_centered",
    "share_low_liquidity_centered",
    "interest_exposure_centered",
)

OBSERVATION_BY_STATE: dict[str, str] = {
    "pi": "pi_obs",
    "Y": "Y_obs",
    "C": "C_obs",
    "mean_mpc_centered": "mean_mpc_centered_obs",
    "share_low_liquidity_centered": "share_low_liquidity_centered_obs",
    "interest_exposure_centered": "interest_exposure_centered_obs",
}

FILTERED_OUTPUT_BY_STATE: dict[str, str] = {
    "pi": "E_pi",
    "Y": "E_Y",
    "C": "E_C",
    "mean_mpc_centered": "E_mean_mpc",
    "share_low_liquidity_centered": "E_low_liquidity_share",
    "interest_exposure_centered": "E_interest_exposure",
}

OBSERVATION_STATE_BY_NAME: dict[str, str] = {value: key for key, value in OBSERVATION_BY_STATE.items()}

INFORMATION_STATE_OBSERVATIONS: dict[str, tuple[str, ...]] = {
    "aggregate_only": ("pi_obs", "Y_obs"),
    "aggregate_history": ("pi_obs", "Y_obs"),
    "filtered_aggregates": ("pi_obs", "Y_obs", "C_obs"),
    "observed_distribution": tuple(OBSERVATION_BY_STATE[state] for state in DEFAULT_STATE_NAMES),
    "filtered_distribution": tuple(OBSERVATION_BY_STATE[state] for state in DEFAULT_STATE_NAMES),
    "full_information": tuple(OBSERVATION_BY_STATE[state] for state in DEFAULT_STATE_NAMES),
}


@dataclass(frozen=True)
class JointKalmanBuildSpec:
    observables: str
    transition_observables: str | None
    observations: str
    observations_spec: str
    output_dir: str
    scalar_filtered_states: str | None
    state_names: tuple[str, ...]
    information_states: tuple[str, ...]
    transition_shrinkage: float
    covariance_floor: float
    max_spectral_radius: float
    note: str


@dataclass(frozen=True)
class KalmanFilterResult:
    means: np.ndarray
    covariances: np.ndarray
    traces: np.ndarray
    log_predictive_density: np.ndarray


def fit_state_transition(
    observables: pd.DataFrame,
    state_names: list[str],
    *,
    shrinkage: float = 0.15,
    covariance_floor: float = 1e-12,
    max_spectral_radius: float = 0.98,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a joint VAR(1) transition for the local HANK/SSJ state vector."""

    _require_columns(observables, {"scenario", "period", *state_names}, Path("<observables>"))
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    for _, group in observables.sort_values(["scenario", "period"]).groupby("scenario", sort=False):
        values = group[state_names].to_numpy(dtype=float)
        if values.shape[0] < 2:
            continue
        x_blocks.append(values[:-1])
        y_blocks.append(values[1:])
    if not x_blocks:
        raise ValueError("Cannot estimate state transition: no scenario has at least two periods.")

    x = np.vstack(x_blocks)
    y = np.vstack(y_blocks)
    ridge = covariance_floor * max(float(x.shape[0]), 1.0)
    xtx = x.T @ x + ridge * np.eye(len(state_names))
    a_ols = np.linalg.solve(xtx, x.T @ y).T
    diagonal_target = np.diag(np.diag(a_ols))
    A = (1.0 - shrinkage) * a_ols + shrinkage * diagonal_target
    A = _stabilize_transition(A, max_spectral_radius=max_spectral_radius)

    residuals = y - x @ A.T
    Q = _sample_covariance(residuals, covariance_floor=covariance_floor)
    return A, Q


def build_observation_matrix(
    state_names: Iterable[str],
    information_state: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the observation matrix for one information structure."""

    state_names = tuple(state_names)
    observation_names = INFORMATION_STATE_OBSERVATIONS.get(information_state)
    if observation_names is None:
        raise ValueError(
            f"Unknown information state for joint Kalman filter: {information_state}. "
            f"Known states: {sorted(INFORMATION_STATE_OBSERVATIONS)}"
        )
    M = np.zeros((len(observation_names), len(state_names)), dtype=float)
    state_index = {name: index for index, name in enumerate(state_names)}
    for row, observation_name in enumerate(observation_names):
        state_name = OBSERVATION_STATE_BY_NAME[observation_name]
        if state_name not in state_index:
            raise ValueError(f"Observation {observation_name} refers to missing state {state_name}.")
        M[row, state_index[state_name]] = 1.0
    return M, tuple(observation_names)


def build_joint_kalman_filtered_states(
    *,
    observables_csv: Path,
    observations_csv: Path,
    observations_spec_json: Path,
    output_dir: Path,
    transition_observables_csv: Path | None = None,
    scalar_filtered_states_csv: Path | None = None,
    state_names: tuple[str, ...] = DEFAULT_STATE_NAMES,
    information_states: tuple[str, ...] = ("filtered_aggregates", "filtered_distribution"),
    transition_shrinkage: float = 0.15,
    covariance_floor: float = 1e-12,
    max_spectral_radius: float = 0.98,
) -> pd.DataFrame:
    """Build joint Kalman-filtered states for aggregate and distributional information sets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    observables = pd.read_csv(observables_csv)
    transition_observables = (
        pd.read_csv(transition_observables_csv)
        if transition_observables_csv is not None
        else observables
    )
    observations = pd.read_csv(observations_csv)
    observation_spec = json.loads(observations_spec_json.read_text(encoding="utf-8"))
    keys = ["scenario", "scenario_label", "period", "observation_seed"]

    _require_columns(observables, {"scenario", "scenario_label", "period", *state_names}, observables_csv)
    _require_columns(
        transition_observables,
        {"scenario", "scenario_label", "period", *state_names},
        transition_observables_csv or observables_csv,
    )
    _require_columns(observations, {*keys, *OBSERVATION_BY_STATE.values()}, observations_csv)

    A, Q = fit_state_transition(
        transition_observables,
        list(state_names),
        shrinkage=transition_shrinkage,
        covariance_floor=covariance_floor,
        max_spectral_radius=max_spectral_radius,
    )
    initial_mean, initial_cov = _initial_state(transition_observables, state_names, covariance_floor)

    base = observations[keys].sort_values(keys).reset_index(drop=True)
    wide = base.copy()
    quality_rows: list[dict[str, object]] = []
    spec_payload: dict[str, object] = {
        "state_names": list(state_names),
        "A": A.tolist(),
        "Q": Q.tolist(),
        "initial_mean": initial_mean.tolist(),
        "initial_cov": initial_cov.tolist(),
        "spectral_radius": spectral_radius(A),
        "transition_shrinkage": transition_shrinkage,
        "covariance_floor": covariance_floor,
        "max_spectral_radius": max_spectral_radius,
        "filters": {},
    }
    cov_payload: dict[str, np.ndarray] = {}
    diagnostics_frames: list[pd.DataFrame] = []
    scalar = _load_scalar_filter(scalar_filtered_states_csv)

    truth = observations[keys + list(OBSERVATION_BY_STATE.values())].merge(
        observables[["scenario", "scenario_label", "period", *state_names]],
        on=["scenario", "scenario_label", "period"],
        how="inner",
        validate="many_to_one",
    )

    for information_state in information_states:
        M, observation_names = build_observation_matrix(state_names, information_state)
        R = _observation_covariance(
            observation_spec=observation_spec,
            observation_names=observation_names,
            covariance_floor=covariance_floor,
        )
        state_space_spec = StateSpaceSpec(
            state_names=tuple(state_names),
            observation_names=observation_names,
            A=A,
            Q=Q,
            M=M,
            R=R,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
        )
        spec_payload["filters"][information_state] = state_space_spec_to_jsonable(state_space_spec)

        state_frame, covariances, traces, log_density = _filter_all_paths(
            observations=observations,
            spec=state_space_spec,
            keys=keys,
        )
        state_frame = state_frame.sort_values(keys).reset_index(drop=True)
        _assert_same_keys(wide, state_frame, keys)

        suffix = _suffix_for_information_state(information_state)
        for state_name in state_names:
            output_name = FILTERED_OUTPUT_BY_STATE[state_name]
            wide[f"{output_name}_{suffix}"] = state_frame[output_name].to_numpy(dtype=float)
            if information_state == "filtered_distribution":
                # Backward-compatible columns for scripts that do not yet distinguish filters.
                wide[output_name] = state_frame[output_name].to_numpy(dtype=float)

        diagnostics = state_frame[keys].copy()
        diagnostics["information_state"] = information_state
        diagnostics["posterior_cov_trace"] = traces
        diagnostics["log_predictive_density"] = log_density
        diagnostics_frames.append(diagnostics)
        cov_payload[f"covariances_{information_state}"] = covariances
        cov_payload[f"posterior_trace_{information_state}"] = traces
        cov_payload[f"log_predictive_density_{information_state}"] = log_density

        quality_rows.extend(
            _filter_quality_rows(
                information_state=information_state,
                estimates=state_frame,
                truth=truth,
                scalar=scalar,
                state_names=state_names,
                observation_names=observation_names,
                traces=traces,
                log_density=log_density,
            )
        )

    wide.to_csv(output_dir / "kalman_filtered_states.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(output_dir / "filter_quality_joint.csv", index=False)
    pd.concat(diagnostics_frames, ignore_index=True).to_csv(output_dir / "kalman_filter_diagnostics.csv", index=False)
    (output_dir / "state_space_spec.json").write_text(
        json.dumps(spec_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "posterior_covariances.npz",
        scenario=wide["scenario"].to_numpy(),
        observation_seed=wide["observation_seed"].to_numpy(dtype=int),
        period=wide["period"].to_numpy(dtype=int),
        **cov_payload,
    )

    build_spec = JointKalmanBuildSpec(
        observables=str(observables_csv),
        transition_observables=str(transition_observables_csv) if transition_observables_csv is not None else None,
        observations=str(observations_csv),
        observations_spec=str(observations_spec_json),
        output_dir=str(output_dir),
        scalar_filtered_states=str(scalar_filtered_states_csv) if scalar_filtered_states_csv is not None else None,
        state_names=tuple(state_names),
        information_states=tuple(information_states),
        transition_shrinkage=float(transition_shrinkage),
        covariance_floor=float(covariance_floor),
        max_spectral_radius=float(max_spectral_radius),
        note=(
            "Совместный фильтр Калмана оценивает общий переход состояния q_{t+1}=Aq_t+eps_t. "
            "Матрица наблюдения M меняется по информационному состоянию, а A и Q остаются общими."
        ),
    )
    (output_dir / "joint_kalman_filter_spec.json").write_text(
        json.dumps(asdict(build_spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return wide


def run_kalman_filter(observations: np.ndarray, spec: StateSpaceSpec) -> KalmanFilterResult:
    """Run a joint Kalman filter for one observed path."""

    observations = np.asarray(observations, dtype=float)
    n_periods = observations.shape[0]
    n_state = len(spec.state_names)
    means = np.zeros((n_periods, n_state), dtype=float)
    covariances = np.zeros((n_periods, n_state, n_state), dtype=float)
    traces = np.zeros(n_periods, dtype=float)
    log_predictive_density = np.zeros(n_periods, dtype=float)

    mean = np.asarray(spec.initial_mean, dtype=float)
    cov = np.asarray(spec.initial_cov, dtype=float)
    for index in range(n_periods):
        if index > 0:
            mean = spec.A @ mean
            cov = _regularize_covariance(spec.A @ cov @ spec.A.T + spec.Q, floor=1e-14)
        predicted_obs = spec.M @ mean
        innovation = observations[index] - predicted_obs
        S = _regularize_covariance(spec.M @ cov @ spec.M.T + spec.R, floor=1e-14)
        log_predictive_density[index] = _log_gaussian_density(innovation, S)
        gain = np.linalg.solve(S, spec.M @ cov).T
        mean = mean + gain @ innovation
        identity = np.eye(n_state)
        cov = (identity - gain @ spec.M) @ cov @ (identity - gain @ spec.M).T + gain @ spec.R @ gain.T
        cov = _regularize_covariance(cov, floor=1e-14)
        means[index] = mean
        covariances[index] = cov
        traces[index] = float(np.trace(cov))
    return KalmanFilterResult(
        means=means,
        covariances=covariances,
        traces=traces,
        log_predictive_density=log_predictive_density,
    )


def _filter_all_paths(
    *,
    observations: pd.DataFrame,
    spec: StateSpaceSpec,
    keys: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    covariances: list[np.ndarray] = []
    traces: list[np.ndarray] = []
    log_density: list[np.ndarray] = []
    sort_cols = ["scenario", "observation_seed", "period"]
    for _, group in observations.sort_values(sort_cols).groupby(["scenario", "observation_seed"], sort=False):
        group = group.sort_values("period")
        result = run_kalman_filter(group[list(spec.observation_names)].to_numpy(dtype=float), spec)
        for row_index, (_, source_row) in enumerate(group.iterrows()):
            row = {key: source_row[key] for key in keys}
            for state_index, state_name in enumerate(spec.state_names):
                row[FILTERED_OUTPUT_BY_STATE[state_name]] = float(result.means[row_index, state_index])
            rows.append(row)
        covariances.append(result.covariances)
        traces.append(result.traces)
        log_density.append(result.log_predictive_density)
    return (
        pd.DataFrame(rows),
        np.concatenate(covariances, axis=0),
        np.concatenate(traces, axis=0),
        np.concatenate(log_density, axis=0),
    )


def _filter_quality_rows(
    *,
    information_state: str,
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    scalar: pd.DataFrame | None,
    state_names: tuple[str, ...],
    observation_names: tuple[str, ...],
    traces: np.ndarray,
    log_density: np.ndarray,
) -> list[dict[str, object]]:
    keys = ["scenario", "scenario_label", "period", "observation_seed"]
    observation_columns = [OBSERVATION_BY_STATE[state] for state in state_names]
    frame = estimates[keys + [FILTERED_OUTPUT_BY_STATE[state] for state in state_names]].merge(
        truth[keys + list(state_names) + observation_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_true"),
    )
    if scalar is not None:
        scalar_columns = [FILTERED_OUTPUT_BY_STATE[state] for state in state_names if FILTERED_OUTPUT_BY_STATE[state] in scalar.columns]
        frame = frame.merge(
            scalar[keys + scalar_columns],
            on=keys,
            how="left",
            validate="one_to_one",
            suffixes=("", "_scalar"),
        )

    rows: list[dict[str, object]] = []
    for state_name in state_names:
        output_name = FILTERED_OUTPUT_BY_STATE[state_name]
        true_values = frame[state_name].to_numpy(dtype=float)
        joint_values = frame[output_name].to_numpy(dtype=float)
        observation_name = OBSERVATION_BY_STATE[state_name]
        if observation_name in observation_names:
            rmse_observed = _rmse(frame[observation_name].to_numpy(dtype=float), true_values)
        else:
            rmse_observed = np.nan
        scalar_column = f"{output_name}_scalar"
        rmse_scalar = _rmse(frame[scalar_column].to_numpy(dtype=float), true_values) if scalar_column in frame.columns else np.nan
        rmse_joint = _rmse(joint_values, true_values)
        rows.append(
            {
                "information_state": information_state,
                "variable": state_name,
                "observation_name": observation_name if observation_name in observation_names else "",
                "num_observations": int(frame.shape[0]),
                "rmse_observed": rmse_observed,
                "rmse_scalar": rmse_scalar,
                "rmse_joint": rmse_joint,
                "rmse_joint_reduction_vs_observed": (rmse_observed - rmse_joint) / rmse_observed if rmse_observed > 0 else np.nan,
                "rmse_joint_reduction_vs_scalar": (rmse_scalar - rmse_joint) / rmse_scalar if rmse_scalar > 0 else np.nan,
                "mean_posterior_cov_trace": float(np.mean(traces)),
                "mean_log_predictive_density": float(np.mean(log_density)),
            }
        )
    return rows


def _observation_covariance(
    *,
    observation_spec: dict[str, object],
    observation_names: tuple[str, ...],
    covariance_floor: float,
) -> np.ndarray:
    base_scales = observation_spec.get("base_scales", {})
    noise_scale = float(observation_spec.get("noise_scale", 1.0))
    scale_floor = float(observation_spec.get("scale_floor", 0.0))
    variances: list[float] = []
    for observation_name in observation_names:
        state_name = OBSERVATION_STATE_BY_NAME[observation_name]
        scale = float(base_scales.get(state_name, scale_floor)) * noise_scale
        variances.append(max(scale * scale, covariance_floor))
    return np.diag(variances)


def _initial_state(
    observables: pd.DataFrame,
    state_names: tuple[str, ...],
    covariance_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    first_period = observables.sort_values(["scenario", "period"]).groupby("scenario", sort=False).head(1)
    values = first_period[list(state_names)].to_numpy(dtype=float)
    initial_mean = np.mean(values, axis=0)
    initial_cov = _sample_covariance(values - initial_mean, covariance_floor=covariance_floor)
    return initial_mean, initial_cov


def _sample_covariance(values: np.ndarray, *, covariance_floor: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Covariance input must be a two-dimensional array.")
    if values.shape[0] <= 1:
        scale = max(float(np.mean(values**2)) if values.size else covariance_floor, covariance_floor)
        return scale * np.eye(values.shape[1])
    cov = np.cov(values, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    return _regularize_covariance(cov, floor=covariance_floor)


def _regularize_covariance(covariance: np.ndarray, *, floor: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _stabilize_transition(A: np.ndarray, *, max_spectral_radius: float) -> np.ndarray:
    radius = spectral_radius(A)
    if radius <= max_spectral_radius or radius <= 0:
        return A
    return A * (max_spectral_radius / radius)


def _log_gaussian_density(innovation: np.ndarray, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance = _regularize_covariance(covariance, floor=1e-14)
        sign, logdet = np.linalg.slogdet(covariance)
    solved = np.linalg.solve(covariance, innovation)
    dimension = innovation.size
    return float(-0.5 * (dimension * np.log(2.0 * np.pi) + logdet + innovation @ solved))


def _rmse(values: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(values, dtype=float) - np.asarray(truth, dtype=float)) ** 2)))


def _suffix_for_information_state(information_state: str) -> str:
    if information_state == "filtered_aggregates":
        return "agg"
    if information_state == "filtered_distribution":
        return "dist"
    return information_state.replace("filtered_", "")


def _load_scalar_filter(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _assert_same_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> None:
    left_keys = left[keys].reset_index(drop=True)
    right_keys = right[keys].reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise ValueError("Filtered output keys do not match the observation panel.")


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
