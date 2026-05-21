from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights
from policy.linear_rules import rule_spec_for_information_state


PROJECTION_STATES = (
    "filtered_aggregates",
    "filtered_distribution_mpc",
    "filtered_distribution_liquidity",
    "filtered_distribution_exposure",
    "filtered_distribution",
)

STATE_LABEL_RU = {
    "filtered_aggregates": "Фильтрованные агрегаты",
    "filtered_distribution_mpc": "Фильтрованные агрегаты + MPC",
    "filtered_distribution_liquidity": "Фильтрованные агрегаты + низкая ликвидность",
    "filtered_distribution_exposure": "Фильтрованные агрегаты + процентная экспозиция",
    "filtered_distribution": "Фильтрованные распределительные показатели",
}

DISTRIBUTIONAL_FEATURE_LABEL_RU = {
    "E_mean_mpc": "Средняя MPC",
    "E_low_liquidity_share": "Доля низколиквидных домохозяйств",
    "E_interest_exposure": "Процентная экспозиция",
}


@dataclass(frozen=True)
class MechanismProjectionSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    projection_states: tuple[str, ...]
    transmission_horizon: int
    note: str


@dataclass(frozen=True)
class LinearProjection:
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    target_mean_train: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanism test: predict local SSJ-optimal rate from information states.")
    parser.add_argument(
        "--information-inputs",
        default="outputs/ssj/stochastic/state_space/information_inputs/information_state_inputs_long.csv",
    )
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/mechanism_optimal_rate_projection")
    parser.add_argument("--figure-dir", default="article/figures")
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--transmission-horizon", type=int, default=12)
    parser.add_argument("--ridge", type=float, default=1e-8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    validation_seeds = _parse_seed_range(args.validation_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)

    environment = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=Path(args.information_inputs),
        hank_observables_csv=Path(args.hank_observables),
        jacobians_npz=Path(args.jacobians),
        loss_weights=PolicyLossWeights(),
    )

    datasets = {
        state: _projection_dataset(environment, state, validation_seeds, test_seeds)
        for state in PROJECTION_STATES
    }
    projections, prediction_frames, metric_rows = _run_optimal_rate_projections(
        datasets=datasets,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
        ridge=args.ridge,
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "mechanism_optimal_rate_projection.csv", index=False)
    predictions.to_csv(output_dir / "optimal_rate_projection_predictions.csv", index=False)
    _write_latex(metrics, output_dir / "table_mechanism_optimal_rate_projection.tex")

    residual_summary, residual_frame = _residualized_distribution_test(
        datasets=datasets,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
        ridge=args.ridge,
    )
    residual_summary.to_csv(output_dir / "mechanism_residual_projection.csv", index=False)
    residual_frame.to_csv(output_dir / "residualized_distribution_signal.csv", index=False)

    transmission_summary, transmission_frame = _transmission_projection_test(
        environment=environment,
        datasets=datasets,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
        horizon=args.transmission_horizon,
        ridge=args.ridge,
    )
    transmission_summary.to_csv(output_dir / "mechanism_transmission_projection.csv", index=False)
    transmission_frame.to_csv(output_dir / "transmission_projection_frame.csv", index=False)

    event_study = _event_study(
        predictions=predictions,
        observables_csv=Path(args.hank_observables),
        output_dir=output_dir,
    )
    _plot_mechanism_figure(
        residual_frame=residual_frame,
        transmission_frame=transmission_frame,
        figure_dir=figure_dir,
    )
    _plot_event_study(event_study, figure_dir)

    spec = MechanismProjectionSpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        projection_states=PROJECTION_STATES,
        transmission_horizon=int(args.transmission_horizon),
        note=(
            "Механизм проверяется через способность информационных состояний предсказывать "
            "локально SSJ-оптимальную ставку и будущую силу трансмиссии ставки."
        ),
    )
    (output_dir / "mechanism_optimal_rate_projection_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        metrics=metrics,
        residual_summary=residual_summary,
        transmission_summary=transmission_summary,
        output_path=output_dir / "report_mechanism_optimal_rate_projection.md",
    )
    print(f"Wrote {output_dir / 'mechanism_optimal_rate_projection.csv'}")
    print(f"Wrote {output_dir / 'table_mechanism_optimal_rate_projection.tex'}")
    print(f"Wrote {figure_dir / 'fig_mechanism_distribution_transmission.pdf'}")
    print(f"Wrote {figure_dir / 'fig_mechanism_event_study.pdf'}")


def _projection_dataset(
    environment: HankSSJPolicyEnvironment,
    information_state: str,
    validation_seeds: list[int],
    test_seeds: list[int],
) -> pd.DataFrame:
    spec = rule_spec_for_information_state(information_state)
    rows: list[pd.DataFrame] = []
    seed_split = {seed: "validation" for seed in validation_seeds}
    seed_split.update({seed: "test" for seed in test_seeds})
    for scenario in environment.scenarios:
        target = environment.optimal_rate_path(scenario=scenario)
        for seed, split in seed_split.items():
            features = environment.feature_matrix(
                scenario=scenario,
                information_state=information_state,
                seed=seed,
                feature_names=spec.feature_names,
            )
            periods = min(features.shape[0], target.size)
            frame = pd.DataFrame(features[:periods], columns=spec.feature_names)
            frame.insert(0, "period", np.arange(periods))
            frame.insert(0, "split", split)
            frame.insert(0, "observation_seed", int(seed))
            frame.insert(0, "scenario", scenario)
            frame["optimal_rate"] = target[:periods]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _run_optimal_rate_projections(
    *,
    datasets: dict[str, pd.DataFrame],
    validation_seeds: list[int],
    test_seeds: list[int],
    ridge: float,
) -> tuple[dict[str, LinearProjection], list[pd.DataFrame], list[dict[str, object]]]:
    del validation_seeds, test_seeds
    projections: dict[str, LinearProjection] = {}
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    aggregate_predictions: pd.DataFrame | None = None
    aggregate_rmse: float | None = None
    aggregate_errors: pd.DataFrame | None = None

    for state, dataset in datasets.items():
        feature_names = rule_spec_for_information_state(state).feature_names
        train = dataset[dataset["split"] == "validation"].copy()
        test = dataset[dataset["split"] == "test"].copy()
        projection = _fit_projection(train, feature_names, ridge=ridge)
        projections[state] = projection
        test_predictions = _predict_projection(test, projection)
        test_predictions["information_state"] = state
        test_predictions["information_state_ru"] = STATE_LABEL_RU[state]
        prediction_frames.append(test_predictions)
        metrics = _projection_metrics(test_predictions, train_target_mean=projection.target_mean_train)
        if state == "filtered_aggregates":
            aggregate_predictions = test_predictions
            aggregate_rmse = metrics["rmse"]
            aggregate_errors = test_predictions[["scenario", "observation_seed", "period", "squared_error"]].rename(
                columns={"squared_error": "squared_error_aggregate"}
            )
        p_value = np.nan
        delta_rmse = np.nan
        if aggregate_predictions is not None and aggregate_rmse is not None and state != "filtered_aggregates":
            paired = test_predictions.merge(
                aggregate_errors,
                on=["scenario", "observation_seed", "period"],
                how="inner",
                validate="one_to_one",
            )
            delta_rmse = metrics["rmse"] - aggregate_rmse
            p_value = _paired_sign_flip_p_value(
                paired["squared_error"].to_numpy(dtype=float)
                - paired["squared_error_aggregate"].to_numpy(dtype=float)
            )
        metric_rows.append(
            {
                "specification": state,
                "specification_ru": STATE_LABEL_RU[state],
                "num_observations": int(test_predictions.shape[0]),
                "rmse": metrics["rmse"],
                "oos_r2": metrics["oos_r2"],
                "delta_rmse_vs_filtered_aggregates": delta_rmse,
                "directional_accuracy": metrics["directional_accuracy"],
                "turning_point_accuracy": metrics["turning_point_accuracy"],
                "paired_p_value_vs_filtered_aggregates": p_value,
            }
        )
    return projections, prediction_frames, metric_rows


def _fit_projection(train: pd.DataFrame, feature_names: tuple[str, ...], *, ridge: float) -> LinearProjection:
    x = train.loc[:, list(feature_names)].to_numpy(dtype=float)
    y = train["optimal_rate"].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0, ddof=0), 1e-8)
    x_std = (x - mean) / scale
    design = np.column_stack([np.ones(x_std.shape[0]), x_std])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return LinearProjection(
        feature_names=tuple(feature_names),
        intercept=float(beta[0]),
        coefficients=tuple(float(value) for value in beta[1:]),
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        target_mean_train=float(np.mean(y)),
    )


def _predict_projection(frame: pd.DataFrame, projection: LinearProjection) -> pd.DataFrame:
    x = frame.loc[:, list(projection.feature_names)].to_numpy(dtype=float)
    mean = np.asarray(projection.feature_mean, dtype=float)
    scale = np.asarray(projection.feature_scale, dtype=float)
    beta = np.asarray(projection.coefficients, dtype=float)
    predicted = projection.intercept + ((x - mean) / scale) @ beta
    result = frame[["scenario", "observation_seed", "period", "optimal_rate"]].copy()
    result["predicted_rate"] = predicted
    result["error"] = result["predicted_rate"] - result["optimal_rate"]
    result["squared_error"] = result["error"] ** 2
    return result


def _projection_metrics(predictions: pd.DataFrame, *, train_target_mean: float) -> dict[str, float]:
    y = predictions["optimal_rate"].to_numpy(dtype=float)
    pred = predictions["predicted_rate"].to_numpy(dtype=float)
    err = pred - y
    sse = float(np.sum(err**2))
    sst = float(np.sum((y - train_target_mean) ** 2))
    directional = _directional_accuracy(y, pred)
    turning = _turning_point_accuracy(predictions)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "oos_r2": 1.0 - sse / sst if sst > 0 else np.nan,
        "directional_accuracy": directional,
        "turning_point_accuracy": turning,
    }


def _directional_accuracy(y: np.ndarray, pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.abs(y) > eps
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.sign(y[mask]) == np.sign(pred[mask])))


def _turning_point_accuracy(predictions: pd.DataFrame, *, eps: float = 1e-12) -> float:
    values: list[float] = []
    for _, group in predictions.sort_values(["scenario", "observation_seed", "period"]).groupby(
        ["scenario", "observation_seed"],
        sort=False,
    ):
        target_change = group["optimal_rate"].diff().to_numpy(dtype=float)[1:]
        predicted_change = group["predicted_rate"].diff().to_numpy(dtype=float)[1:]
        mask = np.abs(target_change) > eps
        if np.any(mask):
            values.append(float(np.mean(np.sign(target_change[mask]) == np.sign(predicted_change[mask]))))
    return float(np.mean(values)) if values else np.nan


def _paired_sign_flip_p_value(delta_squared_error: np.ndarray, *, draws: int = 4000, seed: int = 2027) -> float:
    delta = np.asarray(delta_squared_error, dtype=float)
    observed = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(draws, delta.size))
    simulated = signs @ delta / delta.size
    exceedances = int(np.sum(np.abs(simulated) >= abs(observed)))
    return float((exceedances + 1) / (draws + 1))


def _residualized_distribution_test(
    *,
    datasets: dict[str, pd.DataFrame],
    validation_seeds: list[int],
    test_seeds: list[int],
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del validation_seeds, test_seeds
    aggregate_features = rule_spec_for_information_state("filtered_aggregates").feature_names
    distribution_features = ("E_mean_mpc", "E_low_liquidity_share", "E_interest_exposure")
    aggregate = datasets["filtered_aggregates"]
    distribution = datasets["filtered_distribution"]
    merged = distribution.merge(
        aggregate[["scenario", "observation_seed", "period", "split", "E_pi", "E_Y", "E_C"]],
        on=["scenario", "observation_seed", "period", "split"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_agg"),
    )
    train = merged[merged["split"] == "validation"].copy()
    test = merged[merged["split"] == "test"].copy()
    aggregate_projection = _fit_projection(train, aggregate_features, ridge=ridge)
    train_agg_pred = _predict_projection(train, aggregate_projection)
    test_agg_pred = _predict_projection(test, aggregate_projection)
    train["optimal_rate_residual"] = train["optimal_rate"].to_numpy(dtype=float) - train_agg_pred["predicted_rate"].to_numpy(dtype=float)
    test["optimal_rate_residual"] = test["optimal_rate"].to_numpy(dtype=float) - test_agg_pred["predicted_rate"].to_numpy(dtype=float)

    residual_features: list[str] = []
    for feature in distribution_features:
        residual_name = f"{feature}_residual"
        residual_features.append(residual_name)
        feature_projection = _fit_feature_residual(train, aggregate_features, feature, ridge=ridge)
        train[residual_name] = train[feature].to_numpy(dtype=float) - _predict_feature(train, aggregate_features, feature_projection)
        test[residual_name] = test[feature].to_numpy(dtype=float) - _predict_feature(test, aggregate_features, feature_projection)

    residual_train = train[list(residual_features)].copy()
    residual_train["optimal_rate"] = train["optimal_rate_residual"].to_numpy(dtype=float)
    residual_projection = _fit_projection(
        residual_train,
        tuple(residual_features),
        ridge=ridge,
    )
    residual_test = test[["scenario", "observation_seed", "period", *residual_features]].copy()
    residual_test["optimal_rate"] = test["optimal_rate_residual"].to_numpy(dtype=float)
    predicted = _predict_projection(residual_test, residual_projection)
    zero_rmse = float(np.sqrt(np.mean(residual_test["optimal_rate"].to_numpy(dtype=float) ** 2)))
    rmse = float(np.sqrt(np.mean(predicted["squared_error"].to_numpy(dtype=float))))
    sst = float(np.sum(residual_test["optimal_rate"].to_numpy(dtype=float) ** 2))
    r2 = 1.0 - float(np.sum(predicted["squared_error"].to_numpy(dtype=float))) / sst if sst > 0 else np.nan
    summary_rows = [
        {
            "test": "distribution_residuals_after_aggregates",
            "rmse_zero_model": zero_rmse,
            "rmse_distribution_residual_model": rmse,
            "oos_r2_on_aggregate_residual": r2,
            "num_observations": int(predicted.shape[0]),
        }
    ]
    for feature in residual_features:
        corr = np.corrcoef(residual_test[feature].to_numpy(dtype=float), residual_test["optimal_rate"].to_numpy(dtype=float))[0, 1]
        summary_rows.append(
            {
                "test": f"correlation_{feature}",
                "rmse_zero_model": np.nan,
                "rmse_distribution_residual_model": np.nan,
                "oos_r2_on_aggregate_residual": corr,
                "num_observations": int(predicted.shape[0]),
            }
        )
    residual_frame = residual_test[
        ["scenario", "observation_seed", "period", "optimal_rate", *residual_features]
    ].copy()
    residual_frame["predicted_optimal_rate_residual"] = predicted["predicted_rate"].to_numpy(dtype=float)
    return pd.DataFrame(summary_rows), residual_frame


def _fit_feature_residual(
    train: pd.DataFrame,
    aggregate_features: tuple[str, ...],
    feature: str,
    *,
    ridge: float,
) -> LinearProjection:
    renamed = train[list(aggregate_features)].copy()
    renamed["optimal_rate"] = train[feature].to_numpy(dtype=float)
    return _fit_projection(renamed, aggregate_features, ridge=ridge)


def _predict_feature(frame: pd.DataFrame, feature_names: tuple[str, ...], projection: LinearProjection) -> np.ndarray:
    prediction_frame = frame[["scenario", "observation_seed", "period"] + list(feature_names)].copy()
    prediction_frame["optimal_rate"] = 0.0
    return _predict_projection(prediction_frame, projection)["predicted_rate"].to_numpy(dtype=float)


def _transmission_projection_test(
    *,
    environment: HankSSJPolicyEnvironment,
    datasets: dict[str, pd.DataFrame],
    validation_seeds: list[int],
    test_seeds: list[int],
    horizon: int,
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del validation_seeds, test_seeds
    proxy = _future_transmission_proxy(environment, horizon=horizon)
    aggregate = datasets["filtered_aggregates"].merge(proxy, on="period", how="left", validate="many_to_one")
    distribution = datasets["filtered_distribution"].merge(proxy, on="period", how="left", validate="many_to_one")
    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    for state, dataset in {
        "filtered_aggregates": aggregate,
        "filtered_distribution": distribution,
    }.items():
        feature_names = rule_spec_for_information_state(state).feature_names
        train_raw = dataset[dataset["split"] == "validation"].copy()
        test_raw = dataset[dataset["split"] == "test"].copy()
        train = train_raw[list(feature_names)].copy()
        train["optimal_rate"] = train_raw["future_transmission"].to_numpy(dtype=float)
        test = test_raw[["scenario", "observation_seed", "period", *feature_names]].copy()
        test["optimal_rate"] = test_raw["future_transmission"].to_numpy(dtype=float)
        projection = _fit_projection(train, feature_names, ridge=ridge)
        predicted = _predict_projection(test, projection)
        metrics = _projection_metrics(predicted, train_target_mean=projection.target_mean_train)
        rows.append(
            {
                "specification": state,
                "specification_ru": STATE_LABEL_RU[state],
                "target": "future_output_gap_transmission",
                "rmse": metrics["rmse"],
                "oos_r2": metrics["oos_r2"],
                "num_observations": int(predicted.shape[0]),
            }
        )
        predicted["information_state"] = state
        frames.append(
            test.rename(columns={"optimal_rate": "future_transmission"})[
                [
                    "scenario",
                    "observation_seed",
                    "period",
                    "future_transmission",
                    *[feature for feature in ("E_mean_mpc", "E_low_liquidity_share", "E_interest_exposure") if feature in test.columns],
                ]
            ].assign(predicted_transmission=predicted["predicted_rate"].to_numpy(dtype=float), information_state=state)
        )
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


def _future_transmission_proxy(environment: HankSSJPolicyEnvironment, *, horizon: int, decay: float = 0.95) -> pd.DataFrame:
    effect = environment._effects["output_gap"]
    periods = effect.shape[0]
    rows = []
    for period in range(periods):
        value = 0.0
        for step in range(horizon + 1):
            response_period = period + step
            if response_period >= periods:
                break
            value += (decay**step) * abs(float(effect[response_period, period]))
        rows.append({"period": period, "future_transmission": value})
    return pd.DataFrame(rows)


def _event_study(
    *,
    predictions: pd.DataFrame,
    observables_csv: Path,
    output_dir: Path,
) -> pd.DataFrame:
    aggregate = predictions[predictions["information_state"] == "filtered_aggregates"]
    candidates = predictions[predictions["information_state"].isin(["filtered_distribution_mpc", "filtered_distribution_exposure"])]
    mean_rmse = candidates.groupby("information_state")["squared_error"].mean().sort_values()
    best_state = str(mean_rmse.index[0])
    best = predictions[predictions["information_state"] == best_state]
    paired = aggregate.merge(
        best,
        on=["scenario", "observation_seed", "period"],
        how="inner",
        validate="one_to_one",
        suffixes=("_agg", "_dist"),
    )
    trajectory_gain = (
        paired.assign(error_reduction=paired["squared_error_agg"] - paired["squared_error_dist"])
        .groupby(["scenario", "observation_seed"], sort=False)["error_reduction"]
        .sum()
        .reset_index()
    )
    threshold = trajectory_gain["error_reduction"].quantile(0.90)
    winners = trajectory_gain[trajectory_gain["error_reduction"] >= threshold][["scenario", "observation_seed"]]
    selected = paired.merge(winners, on=["scenario", "observation_seed"], how="inner")
    observables = pd.read_csv(observables_csv)[["scenario", "period", "output_gap", "pi", "C"]]
    selected = selected.merge(observables, on=["scenario", "period"], how="left", validate="many_to_one")
    event = (
        selected.assign(
            abs_error_aggregate=lambda frame: frame["error_agg"].abs(),
            abs_error_distribution=lambda frame: frame["error_dist"].abs(),
            abs_output_gap=lambda frame: frame["output_gap"].abs(),
            best_distribution_state=best_state,
        )
        .groupby("period", sort=True)
        .agg(
            abs_error_aggregate=("abs_error_aggregate", "mean"),
            abs_error_distribution=("abs_error_distribution", "mean"),
            abs_output_gap=("abs_output_gap", "mean"),
            num_trajectories=("scenario", "count"),
            best_distribution_state=("best_distribution_state", "first"),
        )
        .reset_index()
    )
    event.to_csv(output_dir / "mechanism_event_study_top_winners.csv", index=False)
    return event


def _plot_mechanism_figure(
    *,
    residual_frame: pd.DataFrame,
    transmission_frame: pd.DataFrame,
    figure_dir: Path,
) -> None:
    dist = transmission_frame[transmission_frame["information_state"] == "filtered_distribution"].copy()
    residual = residual_frame.copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))

    axes[0].scatter(
        dist["E_mean_mpc"],
        dist["future_transmission"],
        s=8,
        alpha=0.20,
        color="#c06c2d",
        edgecolor="none",
    )
    axes[0].set_title("MPC и будущая трансмиссия")
    axes[0].set_xlabel("Оценённая средняя MPC")
    axes[0].set_ylabel("Будущая сила трансмиссии")

    axes[1].scatter(
        residual["E_mean_mpc_residual"],
        residual["optimal_rate"],
        s=8,
        alpha=0.20,
        color="#276c8f",
        edgecolor="none",
    )
    axes[1].set_title("Остаточный MPC-сигнал")
    axes[1].set_xlabel("MPC сверх агрегатов")
    axes[1].set_ylabel("Оптимальная ставка сверх агрегатов")

    binned = _binned_means(residual, "E_mean_mpc_residual", "optimal_rate", bins=12)
    axes[2].axhline(0.0, color="#222222", linewidth=0.8)
    axes[2].plot(binned["x"], binned["y"], marker="o", color="#4f6f3f", linewidth=1.8)
    axes[2].set_title("Binned means")
    axes[2].set_xlabel("MPC сверх агрегатов")
    axes[2].set_ylabel("Средний остаток оптимальной ставки")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Механизм: распределительный сигнал и трансмиссия ставки")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_mechanism_distribution_transmission.pdf")
    plt.close(fig)


def _plot_event_study(event: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axes[0].plot(event["period"], event["abs_error_aggregate"], label="Фильтрованные агрегаты", color="#276c8f")
    axes[0].plot(event["period"], event["abs_error_distribution"], label="Распределительный сигнал", color="#c06c2d")
    axes[0].set_title("Ошибка предсказания оптимальной ставки")
    axes[0].set_xlabel("Период")
    axes[0].set_ylabel("Средняя абсолютная ошибка")
    axes[0].legend(frameon=False)

    axes[1].plot(event["period"], event["abs_output_gap"], color="#4f6f3f")
    axes[1].set_title("Разрыв выпуска на выигрышных траекториях")
    axes[1].set_xlabel("Период")
    axes[1].set_ylabel("Средний абсолютный разрыв")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Event-study для траекторий с наибольшим выигрышем распределительного сигнала")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_mechanism_event_study.pdf")
    plt.close(fig)


def _binned_means(frame: pd.DataFrame, x_col: str, y_col: str, *, bins: int) -> pd.DataFrame:
    ordered = frame[[x_col, y_col]].dropna().sort_values(x_col).reset_index(drop=True)
    ordered["bin"] = pd.qcut(ordered.index, q=bins, labels=False, duplicates="drop")
    return (
        ordered.groupby("bin", sort=True)
        .agg(x=(x_col, "mean"), y=(y_col, "mean"))
        .reset_index(drop=True)
    )


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    display = frame[
        [
            "specification_ru",
            "rmse",
            "oos_r2",
            "delta_rmse_vs_filtered_aggregates",
            "paired_p_value_vs_filtered_aggregates",
            "directional_accuracy",
            "turning_point_accuracy",
        ]
    ].copy()
    display.columns = [
        "Спецификация",
        "RMSE",
        "OOS $R^2$",
        "$\\Delta$RMSE vs filt-agg",
        "paired p-value",
        "Directional accuracy",
        "Turning-point accuracy",
    ]
    for column in display.columns:
        if column != "Спецификация":
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(
    *,
    metrics: pd.DataFrame,
    residual_summary: pd.DataFrame,
    transmission_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = metrics.sort_values("rmse")
    best = ordered.iloc[0]
    aggregate = metrics[metrics["specification"] == "filtered_aggregates"].iloc[0]
    residual_main = residual_summary[residual_summary["test"] == "distribution_residuals_after_aggregates"].iloc[0]
    transmission = transmission_summary.set_index("specification")
    lines = [
        "# Механизм через локально SSJ-оптимальную ставку",
        "",
        "Проверка строит локально SSJ-оптимальную траекторию ставки для каждой HANK/SSJ-траектории "
        "и оценивает, какие информационные состояния лучше её предсказывают.",
        "",
        "## Основной результат",
        "",
        f"- Лучшее информационное состояние по RMSE: {best['specification_ru']} ({best['rmse']:.6g}).",
        f"- Фильтрованные агрегаты: RMSE {aggregate['rmse']:.6g}, OOS R2 {aggregate['oos_r2']:.3g}.",
        "",
        "## Остаточный распределительный сигнал",
        "",
        (
            "После удаления части оптимальной ставки, объясняемой фильтрованными агрегатами, "
            f"распределительные остатки дают OOS R2 {residual_main['oos_r2_on_aggregate_residual']:.3g} "
            f"и снижают RMSE остатка с {residual_main['rmse_zero_model']:.6g} до "
            f"{residual_main['rmse_distribution_residual_model']:.6g}."
        ),
        "",
        "## Будущая сила трансмиссии",
        "",
        (
            "Для proxy будущей силы трансмиссии OOS R2 у фильтрованных агрегатов равен "
            f"{transmission.loc['filtered_aggregates', 'oos_r2']:.3g}, а у распределительного состояния "
            f"{transmission.loc['filtered_distribution', 'oos_r2']:.3g}."
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
