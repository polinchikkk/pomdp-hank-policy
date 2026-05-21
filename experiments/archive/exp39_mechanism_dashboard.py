from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights  # noqa: E402


AGGREGATE_FEATURES = ("E_pi", "E_Y", "E_C")
DISTRIBUTIONAL_FEATURES = ("E_mean_mpc", "E_low_liquidity_share", "E_interest_exposure")
TARGETS = ("local_optimal_rate_t", "future_marginal_transmission_strength_t")

FEATURE_LABELS = {
    "E_pi": "expected inflation",
    "E_Y": "expected output",
    "E_C": "expected consumption",
    "E_mean_mpc": "mean MPC",
    "E_low_liquidity_share": "low-liquidity share",
    "E_interest_exposure": "interest-rate exposure",
    "E_mean_mpc_residual": "mean MPC residual",
    "E_low_liquidity_share_residual": "low-liquidity share residual",
    "E_interest_exposure_residual": "interest-rate exposure residual",
}


@dataclass(frozen=True)
class MechanismDashboardSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    num_folds: int
    ridge: float
    transmission_horizon: int
    transmission_decay: float
    bin_count: int
    note: str


@dataclass(frozen=True)
class RidgeProjection:
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="Final mechanism dashboard for distributional policy information.")
    parser.add_argument(
        "--information-inputs",
        default="outputs/ssj/stochastic/large_sample/test/information_inputs/information_state_inputs_long.csv",
    )
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/large_sample/test/hank_observables.csv")
    parser.add_argument(
        "--jacobians",
        default="outputs/ssj/stochastic/closed_loop_distributional_ssj/jacobians_distributional_augmented.npz",
    )
    parser.add_argument("--fallback-jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/final_protocol")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--transmission-horizon", type=int, default=12)
    parser.add_argument("--transmission-decay", type=float, default=0.95)
    parser.add_argument("--bin-count", type=int, default=5)
    parser.add_argument("--sign-flip-draws", type=int, default=4_000)
    parser.add_argument("--write-row-level-panels", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jacobians = Path(args.jacobians)
    if not jacobians.exists():
        jacobians = Path(args.fallback_jacobians)

    environment = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=Path(args.information_inputs),
        hank_observables_csv=Path(args.hank_observables),
        jacobians_npz=jacobians,
        loss_weights=PolicyLossWeights(),
    )
    dataset = _mechanism_dataset(
        environment,
        horizon=int(args.transmission_horizon),
        decay=float(args.transmission_decay),
    )
    dataset["fold"] = _assign_shock_seed_folds(dataset["shock_seed"], num_folds=int(args.num_folds))
    dataset = _add_distribution_signal_strength(dataset)

    predictions, coefficients = _crossfit_predictions(
        dataset=dataset,
        ridge=float(args.ridge),
    )
    dashboard = _dashboard_metrics(
        predictions,
        draws=int(args.sign_flip_draws),
    )
    seed_stability = _stability_by_seed(predictions)
    coefficient_stability = _coefficient_stability(coefficients)
    bins = _bin_gains(
        predictions=predictions,
        dataset=dataset,
        bin_count=int(args.bin_count),
    )

    if args.write_row_level_panels:
        dataset.to_csv(output_dir / "mechanism_dataset.csv", index=False)
        predictions.to_csv(output_dir / "mechanism_predictions.csv", index=False)
    dashboard.to_csv(output_dir / "mechanism_dashboard.csv", index=False)
    coefficients.to_csv(output_dir / "mechanism_coefficients.csv", index=False)
    coefficient_stability.to_csv(output_dir / "mechanism_coefficient_stability.csv", index=False)
    seed_stability.to_csv(output_dir / "mechanism_stability_by_seed.csv", index=False)
    bins.to_csv(output_dir / "mechanism_bins.csv", index=False)
    _write_report(
        dashboard=dashboard,
        coefficient_stability=coefficient_stability,
        bins=bins,
        output_path=output_dir / "report_mechanism_dashboard.md",
    )

    spec = MechanismDashboardSpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=str(jacobians),
        output_dir=args.output_dir,
        num_folds=int(args.num_folds),
        ridge=float(args.ridge),
        transmission_horizon=int(args.transmission_horizon),
        transmission_decay=float(args.transmission_decay),
        bin_count=int(args.bin_count),
        note=(
            "Final mechanism block. Targets are local_optimal_rate_t and a state-dependent "
            "future_marginal_transmission_strength_t. All projections are out-of-fold by shock_seed, "
            "so observation noise and periods from the same HANK/SSJ path do not leak across folds."
        ),
    )
    (output_dir / "mechanism_dashboard_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'mechanism_dashboard.csv'}")
    print(f"Wrote {output_dir / 'mechanism_coefficient_stability.csv'}")
    print(f"Wrote {output_dir / 'mechanism_bins.csv'}")


def _mechanism_dataset(
    environment: HankSSJPolicyEnvironment,
    *,
    horizon: int,
    decay: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    transmission = _future_marginal_transmission_strength(environment, horizon=horizon, decay=decay)
    for scenario in environment.scenarios:
        shock_seed = _shock_seed_from_scenario(scenario)
        optimal_rate = environment.optimal_rate_path(scenario=scenario)
        transmission_path = transmission[scenario]
        for observation_seed in environment.seeds:
            try:
                aggregate = environment.feature_matrix(
                    scenario=scenario,
                    information_state="filtered_aggregates",
                    seed=int(observation_seed),
                    feature_names=AGGREGATE_FEATURES,
                )
                distribution = environment.feature_matrix(
                    scenario=scenario,
                    information_state="filtered_distribution",
                    seed=int(observation_seed),
                    feature_names=(*AGGREGATE_FEATURES, *DISTRIBUTIONAL_FEATURES),
                )
            except KeyError:
                continue
            periods = min(len(optimal_rate), len(transmission_path), aggregate.shape[0], distribution.shape[0])
            frame = pd.DataFrame(aggregate[:periods], columns=AGGREGATE_FEATURES)
            for index, feature in enumerate(DISTRIBUTIONAL_FEATURES):
                frame[feature] = distribution[:periods, len(AGGREGATE_FEATURES) + index]
            frame.insert(0, "period", np.arange(periods, dtype=int))
            frame.insert(0, "observation_seed", int(observation_seed))
            frame.insert(0, "shock_seed", int(shock_seed))
            frame.insert(0, "scenario", scenario)
            frame["local_optimal_rate_t"] = optimal_rate[:periods]
            frame["future_marginal_transmission_strength_t"] = transmission_path[:periods]
            rows.append(frame)
    if not rows:
        raise ValueError("No mechanism rows were built. Check information inputs and observation seeds.")
    dataset = pd.concat(rows, ignore_index=True)
    dataset.insert(0, "row_id", np.arange(len(dataset), dtype=int))
    return dataset


def _future_marginal_transmission_strength(
    environment: HankSSJPolicyEnvironment,
    *,
    horizon: int,
    decay: float,
) -> dict[str, np.ndarray]:
    effect = environment._effects["output_gap"]
    result: dict[str, np.ndarray] = {}
    for (scenario,), base in environment._observables.items():
        frame = base.sort_values("period").reset_index(drop=True)
        output_gap = frame["output_gap"].to_numpy(dtype=float)
        periods = min(len(output_gap), effect.shape[0])
        values = np.zeros(periods, dtype=float)
        for period in range(periods):
            total = 0.0
            for step in range(int(horizon) + 1):
                response_period = period + step
                if response_period >= periods:
                    break
                response = float(effect[response_period, period])
                state_weight = abs(float(output_gap[response_period]))
                total += (float(decay) ** step) * abs(response) * state_weight
            values[period] = total
        result[str(scenario)] = values
    return result


def _assign_shock_seed_folds(shock_seed: pd.Series, *, num_folds: int) -> np.ndarray:
    seeds = sorted(int(value) for value in pd.Series(shock_seed).drop_duplicates())
    folds = max(2, min(int(num_folds), len(seeds)))
    mapping = {seed: index % folds for index, seed in enumerate(seeds)}
    return pd.Series(shock_seed).map(mapping).to_numpy(dtype=int)


def _add_distribution_signal_strength(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    values = result.loc[:, list(DISTRIBUTIONAL_FEATURES)].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0, ddof=0), 1e-12)
    z = (values - mean) / scale
    result["distribution_signal_strength"] = np.sqrt(np.mean(z**2, axis=1))
    return result


def _model_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "model": "filtered_aggregates_only",
            "model_family": "A",
            "model_label": "A. filtered aggregates only",
            "features": AGGREGATE_FEATURES,
            "residualized": False,
        }
    ]
    for feature in DISTRIBUTIONAL_FEATURES:
        specs.append(
            {
                "model": f"filtered_aggregates_plus_{feature.replace('E_', '')}",
                "model_family": "B",
                "model_label": f"B. filtered aggregates + {FEATURE_LABELS[feature]}",
                "features": (*AGGREGATE_FEATURES, feature),
                "residualized": False,
            }
        )
    specs.extend(
        [
            {
                "model": "filtered_aggregates_plus_all_distribution",
                "model_family": "C",
                "model_label": "C. filtered aggregates + all distributional statistics",
                "features": (*AGGREGATE_FEATURES, *DISTRIBUTIONAL_FEATURES),
                "residualized": False,
            },
            {
                "model": "residualized_distributional_statistics",
                "model_family": "D",
                "model_label": "D. residualized distributional statistics",
                "features": tuple(f"{feature}_residual" for feature in DISTRIBUTIONAL_FEATURES),
                "residualized": True,
            },
        ]
    )
    return specs


def _crossfit_predictions(
    *,
    dataset: pd.DataFrame,
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    model_specs = _model_specs()
    for target in TARGETS:
        for spec in model_specs:
            model = str(spec["model"])
            for fold in sorted(dataset["fold"].unique()):
                train = dataset[~dataset["fold"].eq(fold)].copy()
                test = dataset[dataset["fold"].eq(fold)].copy()
                if bool(spec["residualized"]):
                    predicted, coefficients = _fit_predict_residualized(
                        train=train,
                        test=test,
                        target=target,
                        ridge=ridge,
                    )
                else:
                    features = tuple(spec["features"])
                    projection = _fit_ridge(train, features, target, ridge=ridge)
                    predicted = _predict_ridge(test, projection)
                    coefficients = [
                        {
                            "feature": feature,
                            "coefficient": coefficient,
                        }
                        for feature, coefficient in zip(projection.feature_names, projection.coefficients)
                    ]
                out = test[["row_id", "scenario", "shock_seed", "observation_seed", "period", target]].copy()
                out = out.rename(columns={target: "target_value"})
                out["predicted_value"] = predicted
                out["target"] = target
                out["model"] = model
                out["model_family"] = str(spec["model_family"])
                out["model_label"] = str(spec["model_label"])
                out["fold"] = int(fold)
                prediction_rows.append(out)
                for item in coefficients:
                    feature = str(item["feature"])
                    coefficient = float(item["coefficient"])
                    coefficient_rows.append(
                        {
                            "target": target,
                            "model": model,
                            "model_family": str(spec["model_family"]),
                            "model_label": str(spec["model_label"]),
                            "fold": int(fold),
                            "feature": feature,
                            "feature_label": FEATURE_LABELS.get(feature, feature),
                            "coefficient": coefficient,
                            "coefficient_sign": _coefficient_sign(coefficient),
                        }
                    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions["error"] = predictions["predicted_value"] - predictions["target_value"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["squared_error"] = predictions["error"] ** 2
    return predictions, pd.DataFrame(coefficient_rows)


def _fit_predict_residualized(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    ridge: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    aggregate_projection = _fit_ridge(train, AGGREGATE_FEATURES, target, ridge=ridge)
    train_base = _predict_ridge(train, aggregate_projection)
    test_base = _predict_ridge(test, aggregate_projection)
    train_residual_target = train[target].to_numpy(dtype=float) - train_base

    residual_feature_names: list[str] = []
    train_residual_features: dict[str, np.ndarray] = {}
    test_residual_features: dict[str, np.ndarray] = {}
    for feature in DISTRIBUTIONAL_FEATURES:
        feature_projection = _fit_ridge_for_array(
            train.loc[:, list(AGGREGATE_FEATURES)].to_numpy(dtype=float),
            train[feature].to_numpy(dtype=float),
            AGGREGATE_FEATURES,
            ridge=ridge,
        )
        residual_name = f"{feature}_residual"
        residual_feature_names.append(residual_name)
        train_residual_features[residual_name] = train[feature].to_numpy(dtype=float) - _predict_ridge_array(
            train.loc[:, list(AGGREGATE_FEATURES)].to_numpy(dtype=float),
            feature_projection,
        )
        test_residual_features[residual_name] = test[feature].to_numpy(dtype=float) - _predict_ridge_array(
            test.loc[:, list(AGGREGATE_FEATURES)].to_numpy(dtype=float),
            feature_projection,
        )

    train_residual = pd.DataFrame(train_residual_features)
    train_residual[target] = train_residual_target
    residual_projection = _fit_ridge(train_residual, tuple(residual_feature_names), target, ridge=ridge)
    test_residual = pd.DataFrame(test_residual_features)
    predicted_residual = _predict_ridge(test_residual, residual_projection)
    coefficients = [
        {
            "feature": feature,
            "coefficient": coefficient,
        }
        for feature, coefficient in zip(residual_projection.feature_names, residual_projection.coefficients)
    ]
    return test_base + predicted_residual, coefficients


def _fit_ridge(frame: pd.DataFrame, features: tuple[str, ...], target: str, *, ridge: float) -> RidgeProjection:
    x = frame.loc[:, list(features)].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    return _fit_ridge_for_array(x, y, features, ridge=ridge)


def _fit_ridge_for_array(
    x: np.ndarray,
    y: np.ndarray,
    features: tuple[str, ...],
    *,
    ridge: float,
) -> RidgeProjection:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0, ddof=0), 1e-12)
    x_std = (x - mean) / scale
    design = np.column_stack([np.ones(x_std.shape[0]), x_std])
    penalty = float(ridge) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeProjection(
        feature_names=tuple(features),
        intercept=float(beta[0]),
        coefficients=tuple(float(value) for value in beta[1:]),
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
    )


def _predict_ridge(frame: pd.DataFrame, projection: RidgeProjection) -> np.ndarray:
    x = frame.loc[:, list(projection.feature_names)].to_numpy(dtype=float)
    return _predict_ridge_array(x, projection)


def _predict_ridge_array(x: np.ndarray, projection: RidgeProjection) -> np.ndarray:
    mean = np.asarray(projection.feature_mean, dtype=float)
    scale = np.asarray(projection.feature_scale, dtype=float)
    beta = np.asarray(projection.coefficients, dtype=float)
    return projection.intercept + ((np.asarray(x, dtype=float) - mean) / scale) @ beta


def _dashboard_metrics(predictions: pd.DataFrame, *, draws: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline = predictions[predictions["model"].eq("filtered_aggregates_only")][
        ["target", "row_id", "abs_error", "squared_error"]
    ].rename(columns={"abs_error": "baseline_abs_error", "squared_error": "baseline_squared_error"})
    for (target, model), group in predictions.groupby(["target", "model"], sort=False):
        merged = group.merge(baseline[baseline["target"].eq(target)], on=["target", "row_id"], how="left", validate="one_to_one")
        y = merged["target_value"].to_numpy(dtype=float)
        sse = float(merged["squared_error"].sum())
        sst = float(np.sum((y - y.mean()) ** 2))
        oof_r2 = 1.0 - sse / sst if sst > 0 else np.nan
        mae = float(merged["abs_error"].mean())
        rmse = float(np.sqrt(merged["squared_error"].mean()))
        baseline_mae = float(merged["baseline_abs_error"].mean())
        baseline_rmse = float(np.sqrt(merged["baseline_squared_error"].mean()))
        baseline_sse = float(merged["baseline_squared_error"].sum())
        baseline_r2 = 1.0 - baseline_sse / sst if sst > 0 else np.nan
        mae_gain = baseline_mae - mae
        r2_gain = oof_r2 - baseline_r2
        seed_gain = (
            merged.assign(abs_error_gain=merged["baseline_abs_error"] - merged["abs_error"])
            .groupby("shock_seed", sort=True)["abs_error_gain"]
            .mean()
            .to_numpy(dtype=float)
        )
        rows.append(
            {
                "target": target,
                "model": model,
                "model_family": str(group["model_family"].iloc[0]),
                "model_label": str(group["model_label"].iloc[0]),
                "num_observations": int(len(merged)),
                "num_shock_seeds": int(merged["shock_seed"].nunique()),
                "out_of_fold_r2": oof_r2,
                "out_of_fold_r2_gain_vs_aggregates": r2_gain,
                "mae": mae,
                "mae_gain_vs_aggregates": mae_gain,
                "rmse": rmse,
                "rmse_gain_vs_aggregates": baseline_rmse - rmse,
                "seed_positive_mae_gain_share": float(np.mean(seed_gain > 0.0)),
                "mae_gain_sign_flip_p": np.nan
                if model == "filtered_aggregates_only"
                else _cluster_sign_flip_greater(seed_gain, draws=draws),
            }
        )
    order = {str(spec["model"]): index for index, spec in enumerate(_model_specs())}
    return (
        pd.DataFrame(rows)
        .sort_values(["target", "model"], key=lambda col: col.map(order) if col.name == "model" else col)
        .reset_index(drop=True)
    )


def _stability_by_seed(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions[predictions["model"].eq("filtered_aggregates_only")][
        ["target", "row_id", "abs_error", "squared_error"]
    ].rename(columns={"abs_error": "baseline_abs_error", "squared_error": "baseline_squared_error"})
    rows: list[pd.DataFrame] = []
    for (target, model), group in predictions.groupby(["target", "model"], sort=False):
        merged = group.merge(baseline[baseline["target"].eq(target)], on=["target", "row_id"], how="left", validate="one_to_one")
        seed = (
            merged.groupby(["target", "model", "model_family", "model_label", "shock_seed"], sort=False)
            .agg(
                num_observations=("row_id", "count"),
                baseline_mae=("baseline_abs_error", "mean"),
                model_mae=("abs_error", "mean"),
                baseline_mse=("baseline_squared_error", "mean"),
                model_mse=("squared_error", "mean"),
            )
            .reset_index()
        )
        seed["mae_gain"] = seed["baseline_mae"] - seed["model_mae"]
        seed["mse_gain"] = seed["baseline_mse"] - seed["model_mse"]
        seed["helped"] = seed["mae_gain"] > 0.0
        rows.append(seed)
    return pd.concat(rows, ignore_index=True)


def _coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    return (
        coefficients.groupby(["target", "model", "model_family", "model_label", "feature", "feature_label"], sort=False)
        .agg(
            mean_coefficient=("coefficient", "mean"),
            std_coefficient=("coefficient", "std"),
            positive_share=("coefficient", lambda values: float(np.mean(np.asarray(values, dtype=float) > 0.0))),
            negative_share=("coefficient", lambda values: float(np.mean(np.asarray(values, dtype=float) < 0.0))),
            zero_share=("coefficient", lambda values: float(np.mean(np.asarray(values, dtype=float) == 0.0))),
            num_folds=("fold", "nunique"),
        )
        .reset_index()
        .assign(sign_stability=lambda frame: np.maximum(frame["positive_share"], frame["negative_share"]))
    )


def _bin_gains(
    *,
    predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    bin_count: int,
) -> pd.DataFrame:
    baseline = predictions[predictions["model"].eq("filtered_aggregates_only")][
        ["target", "row_id", "abs_error", "squared_error"]
    ].rename(columns={"abs_error": "baseline_abs_error", "squared_error": "baseline_squared_error"})
    signal = dataset[["row_id", "distribution_signal_strength"]].copy()
    rows: list[dict[str, object]] = []
    for (target, model), group in predictions[~predictions["model"].eq("filtered_aggregates_only")].groupby(
        ["target", "model"],
        sort=False,
    ):
        merged = (
            group.merge(baseline[baseline["target"].eq(target)], on=["target", "row_id"], how="left", validate="one_to_one")
            .merge(signal, on="row_id", how="left", validate="one_to_one")
            .copy()
        )
        merged["bin"] = pd.qcut(
            merged["distribution_signal_strength"].rank(method="first"),
            q=min(int(bin_count), merged["distribution_signal_strength"].nunique()),
            labels=False,
            duplicates="drop",
        )
        for bin_id, frame in merged.groupby("bin", sort=True):
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "model_family": str(frame["model_family"].iloc[0]),
                    "model_label": str(frame["model_label"].iloc[0]),
                    "bin_variable": "distribution_signal_strength",
                    "bin": int(bin_id),
                    "bin_low": float(frame["distribution_signal_strength"].min()),
                    "bin_high": float(frame["distribution_signal_strength"].max()),
                    "num_observations": int(len(frame)),
                    "num_shock_seeds": int(frame["shock_seed"].nunique()),
                    "mean_distribution_signal_strength": float(frame["distribution_signal_strength"].mean()),
                    "baseline_mae": float(frame["baseline_abs_error"].mean()),
                    "model_mae": float(frame["abs_error"].mean()),
                    "mae_gain": float(frame["baseline_abs_error"].mean() - frame["abs_error"].mean()),
                    "mse_gain": float(frame["baseline_squared_error"].mean() - frame["squared_error"].mean()),
                    "observation_help_share": float(np.mean(frame["baseline_abs_error"] > frame["abs_error"])),
                }
            )
    return pd.DataFrame(rows)


def _cluster_sign_flip_greater(values: np.ndarray, *, draws: int, seed: int = 3919) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(draws), values.size), replace=True)
    simulated = (signs * values).mean(axis=1)
    return float((np.sum(simulated >= observed) + 1.0) / (simulated.size + 1.0))


def _coefficient_sign(value: float, *, tol: float = 1e-14) -> str:
    if value > tol:
        return "positive"
    if value < -tol:
        return "negative"
    return "zero"


def _shock_seed_from_scenario(scenario: str) -> int:
    try:
        return int(str(scenario).split("_")[-1])
    except ValueError:
        return abs(hash(str(scenario))) % (2**31)


def _write_report(
    *,
    dashboard: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    bins: pd.DataFrame,
    output_path: Path,
) -> None:
    best_rows = (
        dashboard[~dashboard["model"].eq("filtered_aggregates_only")]
        .sort_values(["target", "mae_gain_vs_aggregates"], ascending=[True, False])
        .groupby("target", sort=False)
        .head(1)
    )
    lines = [
        "# Final Mechanism Dashboard",
        "",
        "This block asks why a central bank would want distributional features: do they improve",
        "out-of-fold recovery of the local optimal rate and the future marginal transmission state?",
        "Folds are assigned by shock_seed, not by individual rows.",
        "",
        "## Best Additions",
        "",
        best_rows[
            [
                "target",
                "model_label",
                "out_of_fold_r2_gain_vs_aggregates",
                "mae_gain_vs_aggregates",
                "seed_positive_mae_gain_share",
                "mae_gain_sign_flip_p",
            ]
        ].to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Coefficient Sign Stability",
        "",
        coefficient_stability[
            [
                "target",
                "model_label",
                "feature_label",
                "mean_coefficient",
                "positive_share",
                "negative_share",
                "sign_stability",
            ]
        ].head(30).to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Bins",
        "",
        "Bins are sorted by standardized distributional signal strength. Positive MAE gain means the",
        "distributional model improves over filtered aggregates in that bin.",
        "",
        bins.head(40).to_markdown(index=False, floatfmt=".6g"),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
