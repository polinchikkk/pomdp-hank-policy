from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLE_MAP = {
    "pi": ("pi_obs", "E_pi"),
    "Y": ("Y_obs", "E_Y"),
    "C": ("C_obs", "E_C"),
    "mean_mpc_centered": ("mean_mpc_centered_obs", "E_mean_mpc"),
    "share_low_liquidity_centered": ("share_low_liquidity_centered_obs", "E_low_liquidity_share"),
    "interest_exposure_centered": ("interest_exposure_centered_obs", "E_interest_exposure"),
}


@dataclass(frozen=True)
class ScalarFilterParams:
    intercept: float
    persistence: float
    process_variance: float
    measurement_variance: float
    initial_mean: float
    initial_variance: float


@dataclass(frozen=True)
class FilterBuildSpec:
    source_observables: str
    source_observations: str
    variables: tuple[str, ...]
    variance_floor: float
    note: str


def build_filtered_states(
    *,
    observables_csv: Path,
    observations_csv: Path,
    observations_spec_json: Path,
    output_dir: Path,
    variance_floor: float = 1e-12,
) -> pd.DataFrame:
    """Build filtered states from noisy observations of HANK/SSJ variables."""

    observables = pd.read_csv(observables_csv)
    observations = pd.read_csv(observations_csv)
    observation_spec = json.loads(observations_spec_json.read_text(encoding="utf-8"))
    _require_columns(
        observables,
        {"scenario", "scenario_label", "period", *VARIABLE_MAP.keys()},
        observables_csv,
    )
    _require_columns(
        observations,
        {"scenario", "scenario_label", "period", "observation_seed", *(obs for obs, _ in VARIABLE_MAP.values())},
        observations_csv,
    )

    params = _estimate_filter_params(
        observables=observables,
        observation_spec=observation_spec,
        variance_floor=variance_floor,
    )

    filtered_frames: list[pd.DataFrame] = []
    for (scenario, observation_seed), group in observations.groupby(["scenario", "observation_seed"], sort=True):
        group = group.sort_values("period").reset_index(drop=True)
        filtered = group[["scenario", "scenario_label", "period", "observation_seed"]].copy()
        for variable, (observation_column, output_column) in VARIABLE_MAP.items():
            filtered[output_column] = _filter_series(
                observations=group[observation_column].to_numpy(dtype=float),
                params=params[variable],
            )
        filtered_frames.append(filtered)

    filtered_states = pd.concat(filtered_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_states.to_csv(output_dir / "filtered_states.csv", index=False)

    spec = FilterBuildSpec(
        source_observables=str(observables_csv),
        source_observations=str(observations_csv),
        variables=tuple(VARIABLE_MAP.keys()),
        variance_floor=variance_floor,
        note="Параметры фильтра оцениваются по HANK/SSJ-траекториям; самостоятельная экономическая динамика не задаётся.",
    )
    spec_payload = {
        **asdict(spec),
        "parameters": {name: asdict(value) for name, value in params.items()},
    }
    (output_dir / "filtered_states_spec.json").write_text(
        json.dumps(spec_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_filter_quality(
        filtered_states=filtered_states,
        observations=observations,
        observables=observables,
        output_dir=output_dir,
    )
    return filtered_states


def _estimate_filter_params(
    *,
    observables: pd.DataFrame,
    observation_spec: dict[str, object],
    variance_floor: float,
) -> dict[str, ScalarFilterParams]:
    observation_scales = observation_spec.get("observation_scales")
    if not isinstance(observation_scales, dict):
        base_scales = observation_spec.get("base_scales", {})
        if not isinstance(base_scales, dict):
            base_scales = {}
        noise_scale = float(observation_spec.get("noise_scale", 1.0))
        observation_scales = {
            variable: float(base_scales.get(variable, 0.0)) * noise_scale
            for variable in VARIABLE_MAP
        }
    params: dict[str, ScalarFilterParams] = {}
    for variable, (observation_column, _) in VARIABLE_MAP.items():
        intercept, persistence, process_variance = _estimate_scalar_transition(
            observables=observables,
            variable=variable,
            variance_floor=variance_floor,
        )
        state_variance = max(float(observables[variable].var(ddof=0)), variance_floor)
        measurement_scale = float(observation_scales.get(variable, np.sqrt(state_variance)))
        params[variable] = ScalarFilterParams(
            intercept=intercept,
            persistence=persistence,
            process_variance=max(process_variance, variance_floor),
            measurement_variance=max(measurement_scale**2, variance_floor),
            initial_mean=0.0,
            initial_variance=state_variance,
        )
    return params


def _estimate_scalar_transition(
    *,
    observables: pd.DataFrame,
    variable: str,
    variance_floor: float,
) -> tuple[float, float, float]:
    lagged_values: list[np.ndarray] = []
    current_values: list[np.ndarray] = []
    for _, group in observables.sort_values(["scenario", "period"]).groupby("scenario", sort=True):
        values = group[variable].to_numpy(dtype=float)
        if values.size >= 2:
            lagged_values.append(values[:-1])
            current_values.append(values[1:])
    lagged = np.concatenate(lagged_values)
    current = np.concatenate(current_values)
    design = np.column_stack([np.ones_like(lagged), lagged])
    intercept, persistence = np.linalg.lstsq(design, current, rcond=None)[0]
    residual = current - (intercept + persistence * lagged)
    process_variance = max(float(np.var(residual, ddof=0)), variance_floor)
    return float(intercept), float(persistence), process_variance


def _filter_series(*, observations: np.ndarray, params: ScalarFilterParams) -> np.ndarray:
    estimates = np.zeros_like(observations, dtype=float)
    mean = params.initial_mean
    variance = params.initial_variance
    for index, observation in enumerate(observations):
        if index > 0:
            mean = params.intercept + params.persistence * mean
            variance = params.persistence**2 * variance + params.process_variance
        gain = variance / (variance + params.measurement_variance)
        mean = mean + gain * (float(observation) - mean)
        variance = (1.0 - gain) * variance
        estimates[index] = mean
    return estimates


def _write_filter_quality(
    *,
    filtered_states: pd.DataFrame,
    observations: pd.DataFrame,
    observables: pd.DataFrame,
    output_dir: Path,
) -> None:
    keys = ["scenario", "scenario_label", "period"]
    merged = observations.merge(observables, on=keys, how="inner", validate="many_to_one").merge(
        filtered_states,
        on=[*keys, "observation_seed"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for variable, (observation_column, filtered_column) in VARIABLE_MAP.items():
        truth = merged[variable].to_numpy(dtype=float)
        observed = merged[observation_column].to_numpy(dtype=float)
        filtered = merged[filtered_column].to_numpy(dtype=float)
        rows.append(
            {
                "variable": variable,
                "rmse_observed": _rmse(observed, truth),
                "rmse_filtered": _rmse(filtered, truth),
                "rmse_reduction": _rmse(observed, truth) - _rmse(filtered, truth),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "filter_quality.csv", index=False)


def _rmse(values: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values - truth) ** 2)))


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
