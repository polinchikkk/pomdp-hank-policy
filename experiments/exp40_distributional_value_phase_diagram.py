from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp08_main_voi import (  # noqa: E402
    _fit_policy_final,
    _fit_policy_under_modes,
    _load_policy_optimization_config,
    _rule_rows,
    _supervised_candidates,
)
from hank_ssj import (  # noqa: E402
    HankSSJPolicyEnvironment,
    PolicyLossWeights,
    build_filtered_states,
    build_information_state_inputs,
    build_joint_kalman_filtered_states,
    build_noisy_observations,
)
from policy.inference import summarize_paired_inference  # noqa: E402
from policy.optimize_linear_rules import LinearRuleOptimizationBounds  # noqa: E402


DISTRIBUTIONAL_CENTERED_COLUMNS = (
    "mean_mpc_centered",
    "share_low_liquidity_centered",
    "interest_exposure_centered",
)

DISTRIBUTIONAL_LEVEL_COLUMNS = {
    "mean_mpc_centered": ("mean_mpc", "mean_mpc"),
    "share_low_liquidity_centered": ("share_low_liquidity", "share_low_liquidity"),
    "interest_exposure_centered": ("interest_exposure", "interest_exposure"),
}

CHANNEL_GROUPS = {
    "all": DISTRIBUTIONAL_CENTERED_COLUMNS,
    "mpc": ("mean_mpc_centered",),
    "liquidity": ("share_low_liquidity_centered",),
    "exposure": ("interest_exposure_centered",),
    "liquidity_exposure": ("share_low_liquidity_centered", "interest_exposure_centered"),
}

MAIN_STATES = ("filtered_aggregates", "filtered_distribution")


@dataclass(frozen=True)
class PhaseDiagramSpec:
    source_observables: str
    noise_reference_observables: str
    steady_distributional_values: str
    jacobians: str
    output_dir: str
    maps: tuple[str, ...]
    channel_feature_group: str
    channel_strength_grid: tuple[float, ...]
    aggregate_noise_grid: tuple[float, ...]
    distribution_noise_grid: tuple[float, ...]
    persistence_grid: tuple[float, ...]
    output_gap_weight_grid: tuple[float, ...]
    fixed_channel_strength: float
    fixed_aggregate_noise: float
    fixed_distribution_noise: float
    fixed_output_gap_weight: float
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    filter_type: str
    optimization_modes: tuple[str, ...]
    primary_optimization_mode: str
    num_candidates: int
    candidate_seed: int
    policy_optimization_config: str | None
    bootstrap_reps: int
    permutation_reps: int
    alpha: float
    note: str


@dataclass(frozen=True)
class RegimeCell:
    map_name: str
    x_name: str
    y_name: str
    x_value: float
    y_value: float
    channel_strength: float
    aggregate_noise_scale: float
    distribution_noise_scale: float
    distributional_persistence: float | None
    output_gap_weight: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phase diagrams for the value of distributional information.")
    parser.add_argument("--source-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--noise-reference-observables", default="")
    parser.add_argument("--steady-distributional-values", default="outputs/ssj/stochastic/steady_distributional_values.json")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/final_protocol/phase_diagrams")
    parser.add_argument("--maps", default="channel_noise,noise_noise,persistence_loss")
    parser.add_argument("--channel-feature-group", choices=sorted(CHANNEL_GROUPS), default="all")
    parser.add_argument("--channel-strength-grid", default="0,0.5,1,2")
    parser.add_argument("--aggregate-noise-grid", default="0.5,1,2")
    parser.add_argument("--distribution-noise-grid", default="0.5,1,2")
    parser.add_argument("--persistence-grid", default="0.2,0.5,0.8")
    parser.add_argument("--output-gap-weight-grid", default="0.5,1,2")
    parser.add_argument("--fixed-channel-strength", type=float, default=1.0)
    parser.add_argument("--fixed-aggregate-noise", type=float, default=1.0)
    parser.add_argument("--fixed-distribution-noise", type=float, default=1.0)
    parser.add_argument("--fixed-output-gap-weight", type=float, default=1.0)
    parser.add_argument("--validation-seeds", default="930:934")
    parser.add_argument("--test-seeds", default="960:969")
    parser.add_argument("--filter-type", choices=("joint", "scalar"), default="joint")
    parser.add_argument("--num-candidates", type=int, default=120)
    parser.add_argument("--candidate-seed", type=int, default=7041)
    parser.add_argument("--optimization-modes", default="random_candidates,grid_random,continuous")
    parser.add_argument("--primary-optimization-mode", default="continuous")
    parser.add_argument("--continuous-methods", default="L-BFGS-B")
    parser.add_argument("--num-starts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--intercept-bound", type=float, default=0.01)
    parser.add_argument("--standardized-coefficient-bound", type=float, default=0.05)
    parser.add_argument("--policy-optimization-config", default="")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--permutation-reps", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.maps = "channel_noise,noise_noise"
        args.channel_strength_grid = "0,1"
        args.aggregate_noise_grid = "0.75,1.5"
        args.distribution_noise_grid = "0.75,1.5"
        args.persistence_grid = "0.4,0.8"
        args.output_gap_weight_grid = "0.75,1.5"
        args.validation_seeds = "930:931"
        args.test_seeds = "960:962"
        args.num_candidates = min(int(args.num_candidates), 40)
        args.optimization_modes = "random_candidates,grid_random"
        args.primary_optimization_mode = "grid_random"
        args.policy_optimization_config = ""
        args.bootstrap_reps = min(int(args.bootstrap_reps), 500)
        args.permutation_reps = min(int(args.permutation_reps), 1000)
        if args.output_dir == "outputs/final_protocol/phase_diagrams":
            args.output_dir = "outputs/final_protocol/phase_diagrams_smoke"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    maps = tuple(part.strip() for part in args.maps.split(",") if part.strip())
    unknown_maps = sorted(set(maps).difference({"channel_noise", "noise_noise", "persistence_loss"}))
    if unknown_maps:
        raise ValueError(f"Unknown map(s): {unknown_maps}")

    validation_seeds = _parse_seed_range(args.validation_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)
    optimization_modes = tuple(part.strip() for part in args.optimization_modes.split(",") if part.strip())
    continuous_methods = tuple(part.strip() for part in args.continuous_methods.split(",") if part.strip())
    final_optimization_config = _load_policy_optimization_config(args.policy_optimization_config)
    if final_optimization_config is not None:
        primary_optimization_mode = "final_continuous"
        optimization_modes = ("final_continuous",)
        continuous_methods = tuple(final_optimization_config["methods"])
    else:
        primary_optimization_mode = args.primary_optimization_mode
    if primary_optimization_mode not in optimization_modes:
        raise ValueError("--primary-optimization-mode must be included in --optimization-modes.")

    cells = _build_cells(
        maps=maps,
        channel_strength_grid=_parse_float_grid(args.channel_strength_grid),
        aggregate_noise_grid=_parse_float_grid(args.aggregate_noise_grid),
        distribution_noise_grid=_parse_float_grid(args.distribution_noise_grid),
        persistence_grid=_parse_float_grid(args.persistence_grid),
        output_gap_weight_grid=_parse_float_grid(args.output_gap_weight_grid),
        fixed_channel_strength=float(args.fixed_channel_strength),
        fixed_aggregate_noise=float(args.fixed_aggregate_noise),
        fixed_distribution_noise=float(args.fixed_distribution_noise),
        fixed_output_gap_weight=float(args.fixed_output_gap_weight),
    )

    rows: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        print(f"Cell {index + 1}/{len(cells)}: {cell.map_name} x={cell.x_value:g} y={cell.y_value:g}", flush=True)
        rows.append(
            _run_or_load_cell(
                cell=cell,
                source_observables=Path(args.source_observables),
                noise_reference_observables=Path(args.noise_reference_observables)
                if args.noise_reference_observables
                else Path(args.source_observables),
                steady_distributional_values=Path(args.steady_distributional_values),
                jacobians=Path(args.jacobians),
                output_dir=output_dir,
                channel_feature_group=args.channel_feature_group,
                validation_seeds=validation_seeds,
                test_seeds=test_seeds,
                filter_type=args.filter_type,
                final_optimization_config=final_optimization_config,
                optimization_modes=optimization_modes,
                primary_optimization_mode=primary_optimization_mode,
                continuous_methods=continuous_methods,
                num_candidates=int(args.num_candidates),
                candidate_seed=int(args.candidate_seed) + index * 100,
                num_starts=int(args.num_starts),
                maxiter=int(args.maxiter),
                bounds=LinearRuleOptimizationBounds(
                    intercept_abs_bound=float(args.intercept_bound),
                    standardized_coefficient_abs_bound=float(args.standardized_coefficient_bound),
                ),
                bootstrap_reps=int(args.bootstrap_reps),
                permutation_reps=int(args.permutation_reps),
                alpha=float(args.alpha),
                skip_runs=bool(args.skip_runs),
                skip_existing=bool(args.skip_existing),
            )
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["map_name", "y_value", "x_value"]).reset_index(drop=True)
    summary.to_csv(output_dir / "distributional_value_phase_diagram.csv", index=False)
    for map_name, frame in summary.groupby("map_name", sort=False):
        path = output_dir / f"{map_name}.csv"
        frame.to_csv(path, index=False)
        _plot_map(frame, output_dir / f"fig_{map_name}")
    _write_report(summary, output_dir / "report_distributional_value_phase_diagram.md")

    spec = PhaseDiagramSpec(
        source_observables=args.source_observables,
        noise_reference_observables=args.noise_reference_observables or args.source_observables,
        steady_distributional_values=args.steady_distributional_values,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        maps=maps,
        channel_feature_group=args.channel_feature_group,
        channel_strength_grid=tuple(_parse_float_grid(args.channel_strength_grid)),
        aggregate_noise_grid=tuple(_parse_float_grid(args.aggregate_noise_grid)),
        distribution_noise_grid=tuple(_parse_float_grid(args.distribution_noise_grid)),
        persistence_grid=tuple(_parse_float_grid(args.persistence_grid)),
        output_gap_weight_grid=tuple(_parse_float_grid(args.output_gap_weight_grid)),
        fixed_channel_strength=float(args.fixed_channel_strength),
        fixed_aggregate_noise=float(args.fixed_aggregate_noise),
        fixed_distribution_noise=float(args.fixed_distribution_noise),
        fixed_output_gap_weight=float(args.fixed_output_gap_weight),
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        filter_type=args.filter_type,
        optimization_modes=optimization_modes,
        primary_optimization_mode=primary_optimization_mode,
        num_candidates=int(args.num_candidates),
        candidate_seed=int(args.candidate_seed),
        policy_optimization_config=args.policy_optimization_config or None,
        bootstrap_reps=int(args.bootstrap_reps),
        permutation_reps=int(args.permutation_reps),
        alpha=float(args.alpha),
        note=(
            "These maps are local comparative statics around the existing HANK/SSJ panel. "
            "The channel-strength axis scales centered distributional states; the persistence "
            "axis re-times those states while keeping their unconditional scale comparable."
        ),
    )
    (output_dir / "distributional_value_phase_diagram_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'distributional_value_phase_diagram.csv'}")
    print(f"Wrote {output_dir / 'report_distributional_value_phase_diagram.md'}")


def _build_cells(
    *,
    maps: tuple[str, ...],
    channel_strength_grid: list[float],
    aggregate_noise_grid: list[float],
    distribution_noise_grid: list[float],
    persistence_grid: list[float],
    output_gap_weight_grid: list[float],
    fixed_channel_strength: float,
    fixed_aggregate_noise: float,
    fixed_distribution_noise: float,
    fixed_output_gap_weight: float,
) -> list[RegimeCell]:
    cells: list[RegimeCell] = []
    if "channel_noise" in maps:
        for channel_strength in channel_strength_grid:
            for aggregate_noise in aggregate_noise_grid:
                cells.append(
                    RegimeCell(
                        map_name="channel_strength_x_aggregate_noise",
                        x_name="distributional_channel_strength",
                        y_name="aggregate_noise_scale",
                        x_value=float(channel_strength),
                        y_value=float(aggregate_noise),
                        channel_strength=float(channel_strength),
                        aggregate_noise_scale=float(aggregate_noise),
                        distribution_noise_scale=float(fixed_distribution_noise),
                        distributional_persistence=None,
                        output_gap_weight=float(fixed_output_gap_weight),
                    )
                )
    if "noise_noise" in maps:
        for aggregate_noise in aggregate_noise_grid:
            for distribution_noise in distribution_noise_grid:
                cells.append(
                    RegimeCell(
                        map_name="aggregate_noise_x_distribution_noise",
                        x_name="aggregate_noise_scale",
                        y_name="distribution_noise_scale",
                        x_value=float(aggregate_noise),
                        y_value=float(distribution_noise),
                        channel_strength=float(fixed_channel_strength),
                        aggregate_noise_scale=float(aggregate_noise),
                        distribution_noise_scale=float(distribution_noise),
                        distributional_persistence=None,
                        output_gap_weight=float(fixed_output_gap_weight),
                    )
                )
    if "persistence_loss" in maps:
        for persistence in persistence_grid:
            for output_gap_weight in output_gap_weight_grid:
                cells.append(
                    RegimeCell(
                        map_name="persistence_x_output_gap_weight",
                        x_name="distributional_persistence",
                        y_name="output_gap_loss_weight",
                        x_value=float(persistence),
                        y_value=float(output_gap_weight),
                        channel_strength=float(fixed_channel_strength),
                        aggregate_noise_scale=float(fixed_aggregate_noise),
                        distribution_noise_scale=float(fixed_distribution_noise),
                        distributional_persistence=float(persistence),
                        output_gap_weight=float(output_gap_weight),
                    )
                )
    return cells


def _run_or_load_cell(
    *,
    cell: RegimeCell,
    source_observables: Path,
    noise_reference_observables: Path,
    steady_distributional_values: Path,
    jacobians: Path,
    output_dir: Path,
    channel_feature_group: str,
    validation_seeds: list[int],
    test_seeds: list[int],
    filter_type: str,
    final_optimization_config: dict[str, object] | None,
    optimization_modes: tuple[str, ...],
    primary_optimization_mode: str,
    continuous_methods: tuple[str, ...],
    num_candidates: int,
    candidate_seed: int,
    num_starts: int,
    maxiter: int,
    bounds: LinearRuleOptimizationBounds,
    bootstrap_reps: int,
    permutation_reps: int,
    alpha: float,
    skip_runs: bool,
    skip_existing: bool,
) -> dict[str, object]:
    cell_dir = output_dir / "cells" / _cell_label(cell)
    metrics_path = cell_dir / "cell_metrics.json"
    if skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if skip_runs:
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing cached cell metrics: {metrics_path}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    cell_dir.mkdir(parents=True, exist_ok=True)
    observables_path = cell_dir / "hank_observables.csv"
    _write_regime_observables(
        source_observables=source_observables,
        steady_distributional_values=steady_distributional_values,
        output_path=observables_path,
        channel_strength=cell.channel_strength,
        channel_feature_group=channel_feature_group,
        distributional_persistence=cell.distributional_persistence,
    )

    all_observation_seeds = tuple(sorted(set(validation_seeds).union(test_seeds)))
    build_noisy_observations(
        observables_csv=observables_path,
        output_dir=cell_dir,
        seeds=all_observation_seeds,
        aggregate_noise_scale=cell.aggregate_noise_scale,
        distribution_noise_scale=cell.distribution_noise_scale,
        noise_reference_csv=noise_reference_observables,
    )
    filtered_states_path = _build_filter_outputs(
        filter_type=filter_type,
        observables_path=observables_path,
        observations_path=cell_dir / "hank_observations.csv",
        observations_spec_path=cell_dir / "hank_observations_spec.json",
        output_dir=cell_dir,
    )
    input_dir = cell_dir / "information_inputs"
    build_information_state_inputs(
        observables_csv=observables_path,
        observations_csv=cell_dir / "hank_observations.csv",
        filtered_states_csv=filtered_states_path,
        output_dir=input_dir,
    )

    loss_weights = PolicyLossWeights(output_gap=float(cell.output_gap_weight))
    environment = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=input_dir / "information_state_inputs_long.csv",
        hank_observables_csv=observables_path,
        jacobians_npz=jacobians,
        loss_weights=loss_weights,
    )

    fitted = {}
    rule_rows = []
    for state_index, information_state in enumerate(MAIN_STATES):
        extra_candidates = _supervised_candidates(
            environment=environment,
            information_state=information_state,
            validation_seeds=validation_seeds,
        )
        if final_optimization_config is None:
            fit_results = _fit_policy_under_modes(
                environment=environment,
                information_state=information_state,
                validation_seeds=validation_seeds,
                num_candidates=num_candidates,
                candidate_seed=candidate_seed + state_index,
                optimization_modes=optimization_modes,
                extra_candidates=extra_candidates,
                continuous_methods=continuous_methods,
                num_starts=num_starts,
                maxiter=maxiter,
                bounds=bounds,
            )
        else:
            fit_results = _fit_policy_final(
                environment=environment,
                information_state=information_state,
                validation_seeds=validation_seeds,
                candidate_seed=candidate_seed + state_index,
                extra_candidates=extra_candidates,
                config=final_optimization_config,
            )
        primary = fit_results[primary_optimization_mode]
        fitted[information_state] = primary.rule
        for mode, fit in fit_results.items():
            rule_rows.extend(_rule_rows(fit, optimization_mode=mode))

    pd.DataFrame(rule_rows).to_csv(cell_dir / "fitted_policy_rules.csv", index=False)
    losses = _evaluate_main_pair(environment, fitted, test_seeds)
    losses.to_csv(cell_dir / "trajectory_losses.csv", index=False)
    metrics = _cell_metrics(
        cell=cell,
        losses=losses,
        alpha=alpha,
        bootstrap_reps=bootstrap_reps,
        permutation_reps=permutation_reps,
        cell_dir=cell_dir,
        filter_type=filter_type,
        channel_feature_group=channel_feature_group,
        num_validation_seeds=len(validation_seeds),
        num_test_seeds=len(test_seeds),
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _write_regime_observables(
    *,
    source_observables: Path,
    steady_distributional_values: Path,
    output_path: Path,
    channel_strength: float,
    channel_feature_group: str,
    distributional_persistence: float | None,
) -> None:
    frame = pd.read_csv(source_observables)
    missing = sorted(set(DISTRIBUTIONAL_CENTERED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_observables} is missing columns: {missing}")
    if channel_feature_group not in CHANNEL_GROUPS:
        raise ValueError(f"Unknown channel feature group: {channel_feature_group}")
    steady = _load_steady_distributional_values(steady_distributional_values, frame)

    result = frame.copy()
    for column in CHANNEL_GROUPS[channel_feature_group]:
        result[column] = float(channel_strength) * result[column].to_numpy(dtype=float)
    if distributional_persistence is not None:
        result = _retime_distributional_persistence(
            result,
            columns=CHANNEL_GROUPS[channel_feature_group],
            target_persistence=float(distributional_persistence),
        )
    for centered, (level, steady_key) in DISTRIBUTIONAL_LEVEL_COLUMNS.items():
        if level in result.columns:
            result[level] = float(steady[steady_key]) + result[centered].to_numpy(dtype=float)
    result.to_csv(output_path, index=False)


def _retime_distributional_persistence(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    target_persistence: float,
) -> pd.DataFrame:
    result = frame.copy()
    target = float(np.clip(target_persistence, -0.98, 0.98))
    for column in columns:
        original = result[column].to_numpy(dtype=float)
        original_std = float(np.std(original, ddof=0))
        if original_std <= 0:
            continue
        base_rho = _estimate_ar1_persistence(result, column)
        transformed = np.zeros_like(original)
        for _, index in result.sort_values(["scenario", "period"]).groupby("scenario", sort=False).groups.items():
            positions = np.asarray(index, dtype=int)
            values = result.loc[positions, column].to_numpy(dtype=float)
            if values.size == 0:
                continue
            innovations = values.copy()
            if values.size > 1:
                innovations[1:] = values[1:] - base_rho * values[:-1]
                innovation_scale = np.sqrt(max(1.0 - target**2, 1e-6) / max(1.0 - base_rho**2, 1e-6))
                innovations[1:] = innovation_scale * innovations[1:]
            new_values = np.zeros_like(values)
            new_values[0] = values[0]
            for period in range(1, values.size):
                new_values[period] = target * new_values[period - 1] + innovations[period]
            transformed[positions] = new_values
        transformed = transformed - float(np.mean(transformed))
        transformed_std = float(np.std(transformed, ddof=0))
        if transformed_std > 0:
            transformed = transformed * (original_std / transformed_std)
        result[column] = transformed
    return result


def _estimate_ar1_persistence(frame: pd.DataFrame, column: str) -> float:
    lagged_blocks: list[np.ndarray] = []
    current_blocks: list[np.ndarray] = []
    for _, group in frame.sort_values(["scenario", "period"]).groupby("scenario", sort=False):
        values = group[column].to_numpy(dtype=float)
        if values.size < 2:
            continue
        lagged_blocks.append(values[:-1])
        current_blocks.append(values[1:])
    if not lagged_blocks:
        return 0.0
    lagged = np.concatenate(lagged_blocks)
    current = np.concatenate(current_blocks)
    denominator = float(lagged @ lagged)
    if denominator <= 1e-18:
        return 0.0
    return float(np.clip((lagged @ current) / denominator, -0.95, 0.95))


def _load_steady_distributional_values(path: Path, frame: pd.DataFrame) -> dict[str, float]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {}
    values: dict[str, float] = {}
    for _, (level, steady_key) in DISTRIBUTIONAL_LEVEL_COLUMNS.items():
        if steady_key in payload:
            values[steady_key] = float(payload[steady_key])
        elif level in frame.columns:
            values[steady_key] = float(frame[level].mean())
        else:
            values[steady_key] = 0.0
    return values


def _build_filter_outputs(
    *,
    filter_type: str,
    observables_path: Path,
    observations_path: Path,
    observations_spec_path: Path,
    output_dir: Path,
) -> Path:
    if filter_type == "joint":
        state_space_dir = output_dir / "state_space"
        build_joint_kalman_filtered_states(
            observables_csv=observables_path,
            observations_csv=observations_path,
            observations_spec_json=observations_spec_path,
            output_dir=state_space_dir,
        )
        return state_space_dir / "kalman_filtered_states.csv"
    if filter_type == "scalar":
        build_filtered_states(
            observables_csv=observables_path,
            observations_csv=observations_path,
            observations_spec_json=observations_spec_path,
            output_dir=output_dir,
        )
        return output_dir / "filtered_states.csv"
    raise ValueError(f"Unknown filter type: {filter_type}")


def _evaluate_main_pair(
    environment: HankSSJPolicyEnvironment,
    fitted: dict[str, object],
    test_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in environment.scenarios:
        for seed in test_seeds:
            for information_state in MAIN_STATES:
                loss = environment.simulate_scenario(
                    policy=fitted[information_state],
                    information_state=information_state,
                    scenario=scenario,
                    seed=seed,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "observation_seed": int(seed),
                        "information_state": information_state,
                        **asdict(loss),
                    }
                )
    return pd.DataFrame(rows)


def _cell_metrics(
    *,
    cell: RegimeCell,
    losses: pd.DataFrame,
    alpha: float,
    bootstrap_reps: int,
    permutation_reps: int,
    cell_dir: Path,
    filter_type: str,
    channel_feature_group: str,
    num_validation_seeds: int,
    num_test_seeds: int,
) -> dict[str, object]:
    pivot = losses.pivot_table(
        index=["scenario", "observation_seed"],
        columns="information_state",
        values="total_loss",
        aggfunc="first",
    ).reset_index()
    missing = sorted(set(MAIN_STATES).difference(pivot.columns))
    if missing:
        raise ValueError(f"Missing main-state losses: {missing}")
    delta = pivot["filtered_distribution"].to_numpy(dtype=float) - pivot["filtered_aggregates"].to_numpy(dtype=float)
    cluster_id = pivot["scenario"].to_numpy()
    inference = summarize_paired_inference(
        delta,
        cluster_id=cluster_id,
        n_boot=bootstrap_reps,
        n_perm=permutation_reps,
        seed=9021,
        tie_eps=1e-10,
    )
    scenario_delta = pivot.groupby("scenario", sort=False).apply(
        lambda group: float((group["filtered_distribution"] - group["filtered_aggregates"]).mean()),
        include_groups=False,
    )
    mvoi = -float(inference.mean_delta)
    clustered_mvoi_ci_low = -float(inference.clustered_ci_high)
    clustered_mvoi_ci_high = -float(inference.clustered_ci_low)
    significant = bool(inference.clustered_ci_high < 0.0 and inference.sign_flip_p_value <= alpha)
    return {
        "map_name": cell.map_name,
        "x_name": cell.x_name,
        "y_name": cell.y_name,
        "x_value": float(cell.x_value),
        "y_value": float(cell.y_value),
        "channel_strength": float(cell.channel_strength),
        "channel_feature_group": channel_feature_group,
        "aggregate_noise_scale": float(cell.aggregate_noise_scale),
        "distribution_noise_scale": float(cell.distribution_noise_scale),
        "distributional_persistence": None
        if cell.distributional_persistence is None
        else float(cell.distributional_persistence),
        "output_gap_weight": float(cell.output_gap_weight),
        "filter_type": filter_type,
        "num_validation_seeds": int(num_validation_seeds),
        "num_test_seeds": int(num_test_seeds),
        "num_trajectories": int(inference.num_observations),
        "num_clusters": int(inference.num_clusters),
        "loss_filtered_aggregates": float(pivot["filtered_aggregates"].mean()),
        "loss_filtered_distribution": float(pivot["filtered_distribution"].mean()),
        "mean_delta_distribution_minus_aggregates": float(inference.mean_delta),
        "mvoi_distribution": mvoi,
        "mvoi_clustered_ci_low": clustered_mvoi_ci_low,
        "mvoi_clustered_ci_high": clustered_mvoi_ci_high,
        "delta_clustered_ci_low": float(inference.clustered_ci_low),
        "delta_clustered_ci_high": float(inference.clustered_ci_high),
        "clustered_se": float(inference.clustered_se),
        "sign_flip_p": float(inference.sign_flip_p_value),
        "permutation_p": float(inference.permutation_p_value),
        "win_rate": float(inference.win_rate),
        "tie_rate": float(inference.tie_rate),
        "loss_rate": float(inference.loss_rate),
        "scenario_sign_stability": float(np.mean(scenario_delta.to_numpy(dtype=float) < 0.0)),
        "significant": significant,
        "effect_positive": bool(mvoi > 0.0),
        "cell_dir": str(cell_dir),
    }


def _plot_map(frame: pd.DataFrame, path_stem: Path) -> None:
    if frame.empty:
        return
    x_name = str(frame["x_name"].iloc[0])
    y_name = str(frame["y_name"].iloc[0])
    pivot = frame.pivot_table(index="y_value", columns="x_value", values="mvoi_distribution", aggfunc="first")
    stability = frame.pivot_table(index="y_value", columns="x_value", values="scenario_sign_stability", aggfunc="first")
    significance = frame.pivot_table(index="y_value", columns="x_value", values="significant", aggfunc="first").astype(float)
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    max_abs = max(max_abs, 1e-12)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    mesh = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=_extent_from_centers(x, y),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("MVOI: loss reduction from distributional information")
    legend_handles = []
    if x.size >= 2 and y.size >= 2:
        xx, yy = np.meshgrid(x, y)
        if _try_contour(ax, xx, yy, significance.to_numpy(dtype=float), level=0.5, color="black", linestyle="-"):
            legend_handles.append(Line2D([0], [0], color="black", linestyle="-", linewidth=1.4, label="significant"))
        if _try_contour(
            ax,
            xx,
            yy,
            stability.to_numpy(dtype=float),
            level=0.75,
            color="white",
            linestyle="--",
        ):
            legend_handles.append(Line2D([0], [0], color="white", linestyle="--", linewidth=1.4, label="sign-stable"))
    ax.axhline(y=0.0, color="0.5", linewidth=0.6, alpha=0.35)
    ax.axvline(x=0.0, color="0.5", linewidth=0.6, alpha=0.35)
    ax.set_xlabel(_axis_label(x_name))
    ax.set_ylabel(_axis_label(y_name))
    ax.set_title(_map_title(str(frame["map_name"].iloc[0])))
    ax.set_xticks(x)
    ax.set_yticks(y)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=220)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def _try_contour(
    ax,
    xx: np.ndarray,
    yy: np.ndarray,
    values: np.ndarray,
    *,
    level: float,
    color: str,
    linestyle: str,
) -> bool:
    if not np.isfinite(values).any():
        return False
    if float(np.nanmin(values)) > level or float(np.nanmax(values)) < level:
        return False
    ax.contour(xx, yy, values, levels=[level], colors=color, linestyles=linestyle, linewidths=1.4)
    return True


def _extent_from_centers(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    return (*_center_extent(x), *_center_extent(y))


def _center_extent(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        width = max(abs(float(values[0])) * 0.1, 0.5) if values.size else 0.5
        center = float(values[0]) if values.size else 0.0
        return center - width, center + width
    diffs = np.diff(np.sort(values))
    left = float(values[0] - diffs[0] / 2.0)
    right = float(values[-1] + diffs[-1] / 2.0)
    return left, right


def _axis_label(name: str) -> str:
    labels = {
        "distributional_channel_strength": "Distributional channel strength",
        "aggregate_noise_scale": "Noise in aggregate observations",
        "distribution_noise_scale": "Noise in distributional observations",
        "distributional_persistence": "Persistence of distributional state",
        "output_gap_loss_weight": "Policy loss weight on output gap",
    }
    return labels.get(name, name)


def _map_title(name: str) -> str:
    labels = {
        "channel_strength_x_aggregate_noise": "When distributional information is valuable",
        "aggregate_noise_x_distribution_noise": "Observation-noise tradeoff",
        "persistence_x_output_gap_weight": "Persistence and policy objective",
    }
    return labels.get(name, name)


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Distributional value phase diagrams",
        "",
        "Color is MVOI, defined as the loss reduction from filtered_distribution relative to filtered_aggregates.",
        "A black contour marks cells with a positive clustered/sign-flip result. A white dashed contour marks high scenario-level sign stability.",
        "",
    ]
    for map_name, frame in summary.groupby("map_name", sort=False):
        best = frame.sort_values("mvoi_distribution", ascending=False).head(1).iloc[0]
        worst = frame.sort_values("mvoi_distribution", ascending=True).head(1).iloc[0]
        passed = int(frame["significant"].sum())
        lines.extend(
            [
                f"## {map_name}",
                "",
                (
                    f"Best cell: {best['x_name']}={best['x_value']:.6g}, "
                    f"{best['y_name']}={best['y_value']:.6g}, MVOI={best['mvoi_distribution']:.6g}, "
                    f"sign-flip p={best['sign_flip_p']:.6g}."
                ),
                (
                    f"Worst cell: {worst['x_name']}={worst['x_value']:.6g}, "
                    f"{worst['y_name']}={worst['y_value']:.6g}, MVOI={worst['mvoi_distribution']:.6g}."
                ),
                f"Significant cells: {passed}/{len(frame)}.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def _parse_float_grid(value: str) -> list[float]:
    value = value.strip()
    if ":" in value:
        parts = [float(part) for part in value.split(":")]
        if len(parts) != 3:
            raise ValueError("Grid shorthand must be start:stop:num.")
        start, stop, num = parts
        return [float(item) for item in np.linspace(start, stop, int(num))]
    return [float(part) for part in value.split(",") if part.strip()]


def _cell_label(cell: RegimeCell) -> str:
    pieces = [
        cell.map_name,
        f"channel_{_float_label(cell.channel_strength)}",
        f"aggnoise_{_float_label(cell.aggregate_noise_scale)}",
        f"distnoise_{_float_label(cell.distribution_noise_scale)}",
        f"persistence_{_float_label(cell.distributional_persistence)}",
        f"yweight_{_float_label(cell.output_gap_weight)}",
    ]
    return "__".join(pieces)


def _float_label(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "base"
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
