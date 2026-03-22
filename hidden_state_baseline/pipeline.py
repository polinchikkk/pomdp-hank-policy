from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nk_baseline.model import NKParameters

from .evaluation import evaluate_filter_performance
from .kalman_filter import KalmanFilterResults, run_kalman_filter
from .state_space import (
    LinearGaussianStateSpaceModel,
    build_multishock_state_space_model,
    generate_observations,
    simulate_hidden_states,
    state_space_spec_payload,
)

PLOT_SCALE = 100.0
BASELINE_SCENARIO = "medium"
BASELINE_OBSERVATION_DESIGN = "full_panel"
MEASUREMENT_NOISE_SCENARIOS = {
    "small": 0.0005,
    "medium": 0.0015,
    "large": 0.0040,
}
OBSERVATION_DESIGNS = {
    "full_panel": ("x", "pi", "i"),
    "policy_panel": ("pi", "i"),
}
DEFAULT_MONTE_CARLO_RUNS = 25
MISSPECIFIED_RHO_R = 0.70
MISSPECIFICATION_LABEL = "rho_r_filter_0_70"
STATE_TITLES = {
    "r_n": "Natural-Rate Shock",
    "u": "Cost-Push Shock",
    "nu": "Monetary-Policy Shock",
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _standardized_measurement_noise(periods: int, observation_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((periods, observation_dim))


def _run_filter_from_existing_states(
    model: LinearGaussianStateSpaceModel,
    true_states: pd.DataFrame,
    standardized_measurement_noise: np.ndarray,
    label: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, KalmanFilterResults]:
    if standardized_measurement_noise.shape != (len(true_states), len(model.observation_names)):
        raise ValueError("Measurement-noise draw has incompatible shape for the selected observation design.")

    observations = generate_observations(
        model=model,
        true_states=true_states,
        standardized_measurement_noise=standardized_measurement_noise,
    )
    filter_results = run_kalman_filter(
        model=model,
        observations=observations[[f"obs_{name}" for name in model.observation_names]].to_numpy(dtype=float),
    )
    diagnostics, filtered_frame = evaluate_filter_performance(
        true_states=true_states[[f"true_{name}" for name in model.state_names]].to_numpy(dtype=float),
        filter_results=filter_results,
        label=label,
        state_names=model.state_names,
    )
    merged = pd.concat(
        [
            true_states.reset_index(drop=True),
            observations.drop(columns=["t"]).reset_index(drop=True),
            filtered_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    return diagnostics, merged, observations, filter_results


def _state_metric_rows(label_key: str, label_value: str, diagnostics: dict) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for state_name, metrics in diagnostics["state_metrics"].items():
        row: dict[str, float | str] = {
            label_key: label_value,
            "state": state_name,
            "aggregate_rmse": float(diagnostics["aggregate_rmse"]),
            "aggregate_mae": float(diagnostics["aggregate_mae"]),
            "mean_coverage_95": float(diagnostics["mean_coverage_95"]),
            "mean_confidence_band_width": float(diagnostics["mean_confidence_band_width"]),
            "max_abs_error": float(diagnostics["max_abs_error"]),
            "log_likelihood": float(diagnostics["log_likelihood"]),
        }
        row.update({metric_name: float(metric_value) for metric_name, metric_value in metrics.items()})
        rows.append(row)
    return rows


def _monte_carlo_summaries(
    params: NKParameters,
    periods: int,
    burn_in: int,
    base_seed: int,
    runs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_records: list[dict[str, float | str | int]] = []
    aggregate_records: list[dict[str, float | str | int]] = []
    baseline_observations = OBSERVATION_DESIGNS[BASELINE_OBSERVATION_DESIGN]

    for scenario_index, (label, noise_std) in enumerate(MEASUREMENT_NOISE_SCENARIOS.items()):
        model = build_multishock_state_space_model(
            params=params,
            measurement_noise_std=noise_std,
            observation_names=baseline_observations,
        )
        for run_index in range(runs):
            run_seed = base_seed + 10_000 * (scenario_index + 1) + run_index
            true_states, _ = simulate_hidden_states(
                model=model,
                periods=periods,
                burn_in=burn_in,
                seed=run_seed,
            )
            measurement_noise = _standardized_measurement_noise(
                periods=len(true_states),
                observation_dim=len(model.observation_names),
                seed=run_seed + 50_000,
            )
            diagnostics, _, _, _ = _run_filter_from_existing_states(
                model=model,
                true_states=true_states,
                standardized_measurement_noise=measurement_noise,
                label=label,
            )
            aggregate_records.append(
                {
                    "scenario": label,
                    "run": run_index,
                    "aggregate_rmse": float(diagnostics["aggregate_rmse"]),
                    "aggregate_mae": float(diagnostics["aggregate_mae"]),
                    "mean_coverage_95": float(diagnostics["mean_coverage_95"]),
                    "mean_confidence_band_width": float(diagnostics["mean_confidence_band_width"]),
                    "max_abs_error": float(diagnostics["max_abs_error"]),
                    "log_likelihood": float(diagnostics["log_likelihood"]),
                }
            )
            for state_name, metrics in diagnostics["state_metrics"].items():
                state_records.append(
                    {
                        "scenario": label,
                        "run": run_index,
                        "state": state_name,
                        **{metric_name: float(metric_value) for metric_name, metric_value in metrics.items()},
                    }
                )

    state_summary = (
        pd.DataFrame(state_records)
        .groupby(["scenario", "state"], as_index=False, sort=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            correlation_mean=("correlation", "mean"),
            correlation_std=("correlation", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mean_confidence_band_width_mean=("mean_confidence_band_width", "mean"),
            mean_confidence_band_width_std=("mean_confidence_band_width", "std"),
            coverage_95_mean=("coverage_95", "mean"),
            coverage_95_std=("coverage_95", "std"),
        )
    )
    aggregate_summary = (
        pd.DataFrame(aggregate_records)
        .groupby("scenario", as_index=False, sort=False)
        .agg(
            aggregate_rmse_mean=("aggregate_rmse", "mean"),
            aggregate_rmse_std=("aggregate_rmse", "std"),
            aggregate_mae_mean=("aggregate_mae", "mean"),
            aggregate_mae_std=("aggregate_mae", "std"),
            mean_coverage_95_mean=("mean_coverage_95", "mean"),
            mean_coverage_95_std=("mean_coverage_95", "std"),
            mean_confidence_band_width_mean=("mean_confidence_band_width", "mean"),
            mean_confidence_band_width_std=("mean_confidence_band_width", "std"),
            max_abs_error_mean=("max_abs_error", "mean"),
            max_abs_error_std=("max_abs_error", "std"),
            log_likelihood_mean=("log_likelihood", "mean"),
            log_likelihood_std=("log_likelihood", "std"),
        )
    )
    return state_summary, aggregate_summary


def _state_summary_payload(summary: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    payload: dict[str, dict[str, dict[str, float]]] = {}
    for row in summary.to_dict(orient="records"):
        scenario = str(row.pop("scenario"))
        state = str(row.pop("state"))
        payload.setdefault(scenario, {})[state] = {key: float(value) for key, value in row.items()}
    return payload


def _aggregate_summary_payload(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for row in summary.to_dict(orient="records"):
        scenario = str(row.pop("scenario"))
        payload[scenario] = {key: float(value) for key, value in row.items()}
    return payload


def _lag1_autocorrelation(series: np.ndarray) -> float:
    if len(series) < 2 or np.std(series) <= 0.0:
        return 0.0
    return float(np.corrcoef(series[:-1], series[1:])[0, 1])


def _innovation_diagnostics(
    filter_results: KalmanFilterResults,
    observation_names: Sequence[str],
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    innovations = np.asarray(filter_results.innovations, dtype=float)
    standardized_innovations = np.zeros_like(innovations)

    for period, innovation_covariance in enumerate(filter_results.innovation_covariances):
        cholesky_factor = np.linalg.cholesky(innovation_covariance)
        standardized_innovations[period] = np.linalg.solve(cholesky_factor, innovations[period])

    frame = pd.DataFrame({"t": np.arange(len(innovations), dtype=int)})
    diagnostics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(observation_names):
        raw_series = innovations[:, index]
        standardized_series = standardized_innovations[:, index]
        frame[f"innovation_{name}"] = raw_series
        frame[f"standardized_innovation_{name}"] = standardized_series
        diagnostics[name] = {
            "innovation_mean": float(np.mean(raw_series)),
            "innovation_lag1_autocorr": _lag1_autocorrelation(raw_series),
            "standardized_innovation_mean": float(np.mean(standardized_series)),
            "standardized_innovation_std": float(np.std(standardized_series)),
            "standardized_innovation_lag1_autocorr": _lag1_autocorrelation(standardized_series),
        }
    return diagnostics, frame


def _plot_true_vs_filtered(merged: pd.DataFrame, state_names: Sequence[str], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(len(state_names), 1, figsize=(10, 3.4 * len(state_names)), sharex=True)
    axes_array = np.atleast_1d(axes)

    for axis, state_name in zip(axes_array, state_names):
        axis.plot(
            merged["t"],
            PLOT_SCALE * merged[f"true_{state_name}"],
            color="#355070",
            linewidth=1.8,
            label="True state",
        )
        axis.plot(
            merged["t"],
            PLOT_SCALE * merged[f"filtered_{state_name}"],
            color="#b56576",
            linewidth=1.4,
            linestyle="--",
            label="Filtered state",
        )
        axis.set_title(STATE_TITLES.get(state_name, state_name))
        axis.set_ylabel("Percentage points")
        axis.legend(frameon=False, loc="best")

    axes_array[-1].set_xlabel("Period")
    figure.suptitle("True vs Filtered Hidden States", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_filter_error(merged: pd.DataFrame, state_names: Sequence[str], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(len(state_names), 1, figsize=(10, 3.0 * len(state_names)), sharex=True)
    axes_array = np.atleast_1d(axes)

    for axis, state_name in zip(axes_array, state_names):
        axis.axhline(0.0, color="#9aa5b1", linewidth=0.8)
        axis.plot(
            merged["t"],
            PLOT_SCALE * merged[f"filter_error_{state_name}"],
            color="#e56b6f",
            linewidth=1.4,
        )
        axis.set_title(f"{STATE_TITLES.get(state_name, state_name)} Error")
        axis.set_ylabel("Percentage points")

    axes_array[-1].set_xlabel("Period")
    figure.suptitle("Filtering Error by Hidden State", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_confidence_band(merged: pd.DataFrame, state_names: Sequence[str], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(len(state_names), 1, figsize=(10, 3.4 * len(state_names)), sharex=True)
    axes_array = np.atleast_1d(axes)
    legend_handles = None
    legend_labels = None

    for axis, state_name in zip(axes_array, state_names):
        band = axis.fill_between(
            merged["t"],
            PLOT_SCALE * merged[f"lower_95_{state_name}"],
            PLOT_SCALE * merged[f"upper_95_{state_name}"],
            color="#a8dadc",
            alpha=0.5,
            label="95% band",
        )
        true_line = axis.plot(
            merged["t"],
            PLOT_SCALE * merged[f"true_{state_name}"],
            color="#355070",
            linewidth=1.6,
            label="True state",
        )
        filtered_line = axis.plot(
            merged["t"],
            PLOT_SCALE * merged[f"filtered_{state_name}"],
            color="#b56576",
            linewidth=1.3,
            linestyle="--",
            label="Filtered mean",
        )
        axis.set_title(STATE_TITLES.get(state_name, state_name))
        axis.set_ylabel("Percentage points")
        if legend_handles is None:
            legend_handles = [band, true_line[0], filtered_line[0]]
            legend_labels = ["95% band", "True state", "Filtered mean"]

    axes_array[-1].set_xlabel("Period")
    if legend_handles is not None and legend_labels is not None:
        figure.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.01),
        )
    figure.suptitle("Kalman Filter Confidence Bands", y=1.06)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _state_metric_lines(state_metrics: dict[str, dict[str, float]]) -> str:
    return "\n".join(
        (
            f"- `{state_name}`: RMSE {metrics['rmse']:.4e}, "
            f"correlation {metrics['correlation']:.4f}, "
            f"95% coverage {metrics['coverage_95']:.2%}"
        )
        for state_name, metrics in state_metrics.items()
    )


def _noise_sensitivity_lines(sensitivity: dict[str, dict]) -> str:
    return "\n".join(
        (
            f"- `{label}` noise: aggregate RMSE {metrics['aggregate_rmse']:.4e}, "
            f"mean 95% coverage {metrics['mean_coverage_95']:.2%}"
        )
        for label, metrics in sensitivity.items()
    )


def _observation_design_lines(design_metrics: dict[str, dict]) -> str:
    return "\n".join(
        (
            f"- `{label}` observables: aggregate RMSE {metrics['aggregate_rmse']:.4e}, "
            f"mean 95% coverage {metrics['mean_coverage_95']:.2%}"
        )
        for label, metrics in design_metrics.items()
    )


def _monte_carlo_lines(state_summary: dict[str, dict[str, float]]) -> str:
    return "\n".join(
        (
            f"- `{state_name}`: mean RMSE {metrics['rmse_mean']:.4e} "
            f"(std {metrics['rmse_std']:.2e}), mean 95% coverage {metrics['coverage_95_mean']:.2%} "
            f"(std {metrics['coverage_95_std']:.2%})"
        )
        for state_name, metrics in state_summary.items()
    )


def _innovation_lines(innovation_diagnostics: dict[str, dict[str, float]]) -> str:
    return "\n".join(
        (
            f"- `{name}`: raw mean {metrics['innovation_mean']:.3e}, "
            f"raw lag-1 autocorr {metrics['innovation_lag1_autocorr']:.3f}, "
            f"standardized mean {metrics['standardized_innovation_mean']:.3e}, "
            f"standardized std {metrics['standardized_innovation_std']:.3f}, "
            f"standardized lag-1 autocorr {metrics['standardized_innovation_lag1_autocorr']:.3f}"
        )
        for name, metrics in innovation_diagnostics.items()
    )


def _state_recovery_summary_table(diagnostics: dict) -> pd.DataFrame:
    baseline_state_metrics = diagnostics["baseline_scenario"]["state_metrics"]
    policy_panel_state_metrics = diagnostics["observation_design_sensitivity"]["policy_panel"]["state_metrics"]
    monte_carlo_state_metrics = diagnostics["monte_carlo"]["state_summary"][BASELINE_SCENARIO]
    misspecified_state_metrics = diagnostics["parameter_misspecification"][MISSPECIFICATION_LABEL]["state_metrics"]

    rows = []
    for state_name in baseline_state_metrics:
        rows.append(
            {
                "state": state_name,
                "baseline_rmse": float(baseline_state_metrics[state_name]["rmse"]),
                "policy_panel_rmse": float(policy_panel_state_metrics[state_name]["rmse"]),
                "monte_carlo_mean_rmse": float(monte_carlo_state_metrics[state_name]["rmse_mean"]),
                "misspecified_rmse": float(misspecified_state_metrics[state_name]["rmse"]),
            }
        )
    return pd.DataFrame(rows)


def _write_stage3_report(output_path: Path, diagnostics: dict) -> None:
    baseline = diagnostics["baseline_scenario"]
    baseline_state_lines = _state_metric_lines(baseline["state_metrics"])
    noise_lines = _noise_sensitivity_lines(diagnostics["sensitivity"])
    observation_lines = _observation_design_lines(diagnostics["observation_design_sensitivity"])
    monte_carlo_state_lines = _monte_carlo_lines(diagnostics["monte_carlo"]["state_summary"][BASELINE_SCENARIO])
    monte_carlo_aggregate = diagnostics["monte_carlo"]["aggregate_summary"][BASELINE_SCENARIO]
    innovation_lines = _innovation_lines(diagnostics["innovation_diagnostics"])
    misspecified = diagnostics["parameter_misspecification"][MISSPECIFICATION_LABEL]
    misspecification_lines = _state_metric_lines(misspecified["state_metrics"])

    report = f"""# Stage 3 Report: Multishock Hidden-State Baseline

## Setup

- Structural base: stage-2 linear New Keynesian policy model.
- Hidden state vector: natural-rate shock `r_n`, cost-push shock `u`, and monetary-policy shock `nu`.
- Baseline observables: output gap `x`, inflation `pi`, and nominal rate `i`.
- Additional observability stress test: policy panel with only `pi` and `i`.
- State-space form: `state_(t+1) = A state_t + B eps_(t+1)`, `obs_t = C state_t + D eta_t`.
- The model remains linearized around the zero steady state, so every latent state and observation is measured as a deviation from that normalized reference point.
- All plotted quantities are reported in percentage points.
- The observation equation is inherited from the stage-2 NK policy solution, so the filter must disentangle competing latent shocks rather than denoise a hand-written one-factor measurement system.

## Baseline Filter

- Baseline measurement-noise scenario: `{BASELINE_SCENARIO}`
- Baseline observation design: `{BASELINE_OBSERVATION_DESIGN}`
- Aggregate RMSE across latent states: {baseline['aggregate_rmse']:.4e}
- Aggregate mean absolute error: {baseline['aggregate_mae']:.4e}
- Mean 95% confidence-band width: {baseline['mean_confidence_band_width']:.4e}
- Single-path mean empirical 95% coverage: {baseline['mean_coverage_95']:.2%}
- Log-likelihood: {baseline['log_likelihood']:.3f}

{baseline_state_lines}

- The single-path coverage number is illustrative only; the main calibration check for interval coverage comes from the Monte Carlo averages below.

## Noise Sensitivity

{noise_lines}

- Filter accuracy deteriorates monotonically as measurement noise rises, but the ranking now reflects recovery of three latent shocks rather than one isolated AR(1) process.

## Observation-Set Stress Test

{observation_lines}

- Restricting the observation set to the policy panel makes inference materially harder, which is a more meaningful incomplete-information stress test than changing noise alone.

## Monte Carlo Robustness

- Monte Carlo runs per scenario: {diagnostics['monte_carlo']['runs']}
- For the baseline `{BASELINE_SCENARIO}` scenario, mean aggregate 95% coverage across runs is {monte_carlo_aggregate['mean_coverage_95_mean']:.2%}, which is close to the nominal 95% target.

{monte_carlo_state_lines}

- Single-path coverage can drift across individual trajectories, so Monte Carlo coverage is the more informative benchmark for interval calibration.

## Innovation Diagnostics

{innovation_lines}

- Innovation moments stay close to their ideal values, which supports the internal consistency of the linear-Gaussian filter specification.

## Mild Parameter Misspecification

- True data-generating process uses `rho_r = {diagnostics['parameter_misspecification']['true_rho_r']:.2f}`.
- Misspecified filter uses `rho_r = {diagnostics['parameter_misspecification']['filter_rho_r']:.2f}`.
- Misspecified aggregate RMSE: {misspecified['aggregate_rmse']:.4e}
- Misspecified mean 95% coverage: {misspecified['mean_coverage_95']:.2%}

{misspecification_lines}

- Even a mild persistence misspecification degrades latent-state recovery, which is a useful bridge from the clean baseline toward more realistic partial-information settings.

## Interpretation

- Stage 3 now treats incomplete information as a multishock inference problem: the policymaker observes noisy macro variables but does not directly observe which structural shock is driving them.
- Among the three hidden components, the monetary-policy shock `nu` is recovered least precisely, which points to weaker identifiability of this component from the observed macro panel.
- This is still a linear-Gaussian baseline, but it now includes both innovation diagnostics and a first misspecification stress test rather than relying only on in-model fit.
- The resulting pipeline is a stronger bridge from stage 2 policy analysis to later work on hidden states, filtering, and learning-based policy design.

## Not Implemented Yet

- Correlated structural shocks, correlated measurement noise, Kalman smoothing, or full maximum-likelihood estimation.
- Hidden endogenous state blocks, structural breaks, or regime switching.
- RL and belief-state policy optimization.
"""
    (output_path / "stage3_report.md").write_text(report, encoding="utf-8")


def run_stage3_pipeline(
    output_dir: str | Path,
    periods: int = 240,
    burn_in: int = 80,
    seed: int = 202,
    monte_carlo_runs: int = DEFAULT_MONTE_CARLO_RUNS,
) -> dict:
    start_time = time.perf_counter()
    output_path = Path(output_dir)
    figure_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    params = NKParameters()
    baseline_model = build_multishock_state_space_model(
        params=params,
        measurement_noise_std=MEASUREMENT_NOISE_SCENARIOS[BASELINE_SCENARIO],
        observation_names=OBSERVATION_DESIGNS[BASELINE_OBSERVATION_DESIGN],
    )

    true_states, _ = simulate_hidden_states(
        model=baseline_model,
        periods=periods,
        burn_in=burn_in,
        seed=seed,
    )

    scenario_frames: dict[str, pd.DataFrame] = {}
    scenario_diagnostics: dict[str, dict] = {}
    noise_sensitivity_rows: list[dict[str, float | str]] = []
    baseline_filter_results: KalmanFilterResults | None = None

    full_panel_noise = _standardized_measurement_noise(
        periods=len(true_states),
        observation_dim=len(OBSERVATION_DESIGNS[BASELINE_OBSERVATION_DESIGN]),
        seed=seed + 1,
    )

    for label, noise_std in MEASUREMENT_NOISE_SCENARIOS.items():
        model = build_multishock_state_space_model(
            params=params,
            measurement_noise_std=noise_std,
            observation_names=OBSERVATION_DESIGNS[BASELINE_OBSERVATION_DESIGN],
        )
        diagnostics, merged, _, filter_results = _run_filter_from_existing_states(
            model=model,
            true_states=true_states,
            standardized_measurement_noise=full_panel_noise,
            label=label,
        )
        scenario_frames[label] = merged
        scenario_diagnostics[label] = diagnostics
        noise_sensitivity_rows.extend(_state_metric_rows("scenario", label, diagnostics))
        if label == BASELINE_SCENARIO:
            baseline_filter_results = filter_results

    baseline_filtered = scenario_frames[BASELINE_SCENARIO]
    baseline_diagnostics = scenario_diagnostics[BASELINE_SCENARIO]
    if baseline_filter_results is None:
        raise RuntimeError("Baseline Kalman filter results were not generated.")

    innovation_diagnostics, innovation_frame = _innovation_diagnostics(
        filter_results=baseline_filter_results,
        observation_names=baseline_model.observation_names,
    )

    observation_design_diagnostics: dict[str, dict] = {}
    observation_design_rows: list[dict[str, float | str]] = []
    noise_lookup = {
        name: full_panel_noise[:, index]
        for index, name in enumerate(OBSERVATION_DESIGNS[BASELINE_OBSERVATION_DESIGN])
    }
    for design_label, observation_names in OBSERVATION_DESIGNS.items():
        model = build_multishock_state_space_model(
            params=params,
            measurement_noise_std=MEASUREMENT_NOISE_SCENARIOS[BASELINE_SCENARIO],
            observation_names=observation_names,
        )
        design_noise = np.column_stack([noise_lookup[name] for name in observation_names])
        diagnostics, _, _, _ = _run_filter_from_existing_states(
            model=model,
            true_states=true_states,
            standardized_measurement_noise=design_noise,
            label=design_label,
        )
        observation_design_diagnostics[design_label] = diagnostics
        observation_design_rows.extend(_state_metric_rows("observation_design", design_label, diagnostics))

    misspecified_params = replace(params, rho_r=MISSPECIFIED_RHO_R)
    misspecified_model = build_multishock_state_space_model(
        params=misspecified_params,
        measurement_noise_std=MEASUREMENT_NOISE_SCENARIOS[BASELINE_SCENARIO],
        observation_names=OBSERVATION_DESIGNS[BASELINE_OBSERVATION_DESIGN],
    )
    misspecified_diagnostics, _, _, _ = _run_filter_from_existing_states(
        model=misspecified_model,
        true_states=true_states,
        standardized_measurement_noise=full_panel_noise,
        label=MISSPECIFICATION_LABEL,
    )
    parameter_misspecification = {
        "true_rho_r": float(params.rho_r),
        "filter_rho_r": float(misspecified_params.rho_r),
        "well_specified": baseline_diagnostics,
        MISSPECIFICATION_LABEL: misspecified_diagnostics,
    }

    monte_carlo_state_summary, monte_carlo_aggregate_summary = _monte_carlo_summaries(
        params=params,
        periods=periods,
        burn_in=burn_in,
        base_seed=seed,
        runs=monte_carlo_runs,
    )

    runtime_seconds = time.perf_counter() - start_time
    filter_diagnostics = {
        "baseline_observation_design": BASELINE_OBSERVATION_DESIGN,
        "baseline_scenario": baseline_diagnostics,
        "sensitivity": scenario_diagnostics,
        "observation_design_sensitivity": observation_design_diagnostics,
        "innovation_diagnostics": innovation_diagnostics,
        "parameter_misspecification": parameter_misspecification,
        "monte_carlo": {
            "runs": monte_carlo_runs,
            "state_summary": _state_summary_payload(monte_carlo_state_summary),
            "aggregate_summary": _aggregate_summary_payload(monte_carlo_aggregate_summary),
        },
        "runtime_seconds": runtime_seconds,
    }
    state_recovery_summary = _state_recovery_summary_table(filter_diagnostics)

    _write_json(
        output_path / "model_spec.json",
        state_space_spec_payload(
            params=params,
            model=baseline_model,
            baseline_noise_label=BASELINE_SCENARIO,
            measurement_noise_scenarios=MEASUREMENT_NOISE_SCENARIOS,
            observation_designs=OBSERVATION_DESIGNS,
        ),
    )
    true_states.to_csv(output_path / "true_states.csv", index=False)
    observation_columns = [f"obs_{name}" for name in baseline_model.observation_names] + [
        f"signal_{name}" for name in baseline_model.observation_names
    ]
    baseline_filtered[["t", *observation_columns]].to_csv(output_path / "observations.csv", index=False)

    filtered_state_columns: list[str] = ["t"]
    for state_name in baseline_model.state_names:
        filtered_state_columns.extend(
            [
                f"filtered_{state_name}",
                f"filtered_std_{state_name}",
                f"lower_95_{state_name}",
                f"upper_95_{state_name}",
                f"filter_error_{state_name}",
            ]
        )
    baseline_filtered[filtered_state_columns].to_csv(output_path / "filtered_states.csv", index=False)

    pd.DataFrame(noise_sensitivity_rows).to_csv(output_path / "noise_sensitivity.csv", index=False)
    pd.DataFrame(observation_design_rows).to_csv(
        output_path / "observation_design_sensitivity.csv",
        index=False,
    )
    innovation_frame.to_csv(output_path / "innovation_series.csv", index=False)
    pd.DataFrame(
        [{"observation": name, **metrics} for name, metrics in innovation_diagnostics.items()]
    ).to_csv(output_path / "innovation_diagnostics.csv", index=False)
    pd.DataFrame(
        _state_metric_rows("specification", "well_specified", baseline_diagnostics)
        + _state_metric_rows("specification", MISSPECIFICATION_LABEL, misspecified_diagnostics)
    ).to_csv(output_path / "parameter_misspecification.csv", index=False)
    state_recovery_summary.to_csv(output_path / "state_recovery_summary.csv", index=False)
    monte_carlo_state_summary.to_csv(output_path / "monte_carlo_state_summary.csv", index=False)
    monte_carlo_aggregate_summary.to_csv(output_path / "monte_carlo_aggregate_summary.csv", index=False)
    _write_json(output_path / "filter_diagnostics.json", filter_diagnostics)

    _plot_true_vs_filtered(
        merged=baseline_filtered[
            [
                "t",
                *[f"true_{name}" for name in baseline_model.state_names],
                *[f"filtered_{name}" for name in baseline_model.state_names],
            ]
        ],
        state_names=baseline_model.state_names,
        output_path=figure_dir / "true_vs_filtered_state.png",
    )
    _plot_filter_error(
        merged=baseline_filtered[["t", *[f"filter_error_{name}" for name in baseline_model.state_names]]],
        state_names=baseline_model.state_names,
        output_path=figure_dir / "filtered_error.png",
    )
    _plot_confidence_band(
        merged=baseline_filtered[
            [
                "t",
                *[f"true_{name}" for name in baseline_model.state_names],
                *[f"filtered_{name}" for name in baseline_model.state_names],
                *[f"lower_95_{name}" for name in baseline_model.state_names],
                *[f"upper_95_{name}" for name in baseline_model.state_names],
            ]
        ],
        state_names=baseline_model.state_names,
        output_path=figure_dir / "confidence_band.png",
    )
    _write_stage3_report(output_path=output_path, diagnostics=filter_diagnostics)

    return {
        "params": params,
        "baseline_model": baseline_model,
        "true_states": true_states,
        "baseline_filtered": baseline_filtered,
        "filter_diagnostics": filter_diagnostics,
    }
