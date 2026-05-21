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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights  # noqa: E402


AGGREGATE_FEATURES = ("E_pi", "E_Y", "E_C")
DISTRIBUTIONAL_FEATURES = ("E_mean_mpc", "E_low_liquidity_share", "E_interest_exposure")
TRANSMISSION_INDEX_FEATURE = "transmission_index"

FEATURE_LABELS = {
    "E_pi": "фильтрованная инфляция",
    "E_Y": "фильтрованный выпуск",
    "E_C": "фильтрованное потребление",
    "E_mean_mpc": "предельная склонность к потреблению",
    "E_low_liquidity_share": "доля низколиквидных",
    "E_interest_exposure": "чувствительность к ставке",
    TRANSMISSION_INDEX_FEATURE: "компактный индекс трансмиссии",
}

MODEL_SPECS = (
    {
        "model": "A_filtered_aggregates",
        "model_label": "A. агрегатная оценка состояния",
        "features": AGGREGATE_FEATURES,
        "first_stage_index": False,
    },
    {
        "model": "B_aggregates_plus_mpc",
        "model_label": "B. агрегатная оценка + MPC",
        "features": (*AGGREGATE_FEATURES, "E_mean_mpc"),
        "first_stage_index": False,
    },
    {
        "model": "C_aggregates_plus_low_liquidity",
        "model_label": "C. агрегатная оценка + доля низколиквидных",
        "features": (*AGGREGATE_FEATURES, "E_low_liquidity_share"),
        "first_stage_index": False,
    },
    {
        "model": "D_aggregates_plus_interest_exposure",
        "model_label": "D. агрегатная оценка + чувствительность к ставке",
        "features": (*AGGREGATE_FEATURES, "E_interest_exposure"),
        "first_stage_index": False,
    },
    {
        "model": "E_aggregates_plus_all_distributional",
        "model_label": "E. агрегатная оценка + все распределительные статистики",
        "features": (*AGGREGATE_FEATURES, *DISTRIBUTIONAL_FEATURES),
        "first_stage_index": False,
    },
    {
        "model": "F_aggregates_plus_transmission_index",
        "model_label": "F. агрегатная оценка + компактный индекс трансмиссии",
        "features": (*AGGREGATE_FEATURES, TRANSMISSION_INDEX_FEATURE),
        "first_stage_index": True,
    },
)


@dataclass(frozen=True)
class TransmissionStateValueSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_csv: str
    figure_path: str
    table_path: str
    horizon: int
    decay: float
    weight_y: float
    weight_c: float
    weight_pi: float
    num_folds: int
    ridge: float
    sign_flip_draws: int
    max_shock_seeds: int | None
    note: str


@dataclass(frozen=True)
class RidgeProjection:
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the hidden local monetary-transmission state from information sets.")
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
    parser.add_argument("--output-csv", default="outputs/final_protocol/transmission_state_value.csv")
    parser.add_argument("--figure-path", default="article/figures/fig_transmission_state_prediction.pdf")
    parser.add_argument("--table-path", default="article/tables/table_transmission_state_value.tex")
    parser.add_argument("--dataset-output", default="")
    parser.add_argument("--predictions-output", default="")
    parser.add_argument("--coefficients-output", default="")
    parser.add_argument("--spec-output", default="outputs/final_protocol/transmission_state_value_spec.json")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--weight-y", type=float, default=1.0)
    parser.add_argument("--weight-c", type=float, default=1.0)
    parser.add_argument("--weight-pi", type=float, default=1.0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--sign-flip-draws", type=int, default=4_000)
    parser.add_argument("--max-shock-seeds", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_shock_seeds = 12 if args.max_shock_seeds is None else min(int(args.max_shock_seeds), 12)
        args.num_folds = min(int(args.num_folds), 3)
        args.sign_flip_draws = min(int(args.sign_flip_draws), 1_000)
        if args.output_csv == "outputs/final_protocol/transmission_state_value.csv":
            args.output_csv = "outputs/final_protocol/transmission_state_value_smoke.csv"
        if args.figure_path == "article/figures/fig_transmission_state_prediction.pdf":
            args.figure_path = "article/figures/fig_transmission_state_prediction_smoke.pdf"
        if args.table_path == "article/tables/table_transmission_state_value.tex":
            args.table_path = "article/tables/table_transmission_state_value_smoke.tex"
        if args.spec_output == "outputs/final_protocol/transmission_state_value_spec.json":
            args.spec_output = "outputs/final_protocol/transmission_state_value_smoke_spec.json"

    jacobians = Path(args.jacobians)
    if not jacobians.exists():
        jacobians = Path(args.fallback_jacobians)

    environment = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=Path(args.information_inputs),
        hank_observables_csv=Path(args.hank_observables),
        jacobians_npz=jacobians,
        loss_weights=PolicyLossWeights(),
    )
    dataset = _transmission_dataset(
        environment=environment,
        horizon=int(args.horizon),
        decay=float(args.decay),
        weight_y=float(args.weight_y),
        weight_c=float(args.weight_c),
        weight_pi=float(args.weight_pi),
        max_shock_seeds=args.max_shock_seeds,
    )
    dataset["fold"] = _assign_shock_seed_folds(dataset["shock_seed"], num_folds=int(args.num_folds))
    predictions, coefficients = _crossfit_predictions(dataset=dataset, ridge=float(args.ridge))
    summary = _summary_table(
        dataset=dataset,
        predictions=predictions,
        coefficients=coefficients,
        draws=int(args.sign_flip_draws),
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    _write_latex(summary, Path(args.table_path))
    _plot(summary, Path(args.figure_path))

    if args.dataset_output:
        path = Path(args.dataset_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(path, index=False)
    if args.predictions_output:
        path = Path(args.predictions_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(path, index=False)
    if args.coefficients_output:
        path = Path(args.coefficients_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        coefficients.to_csv(path, index=False)

    spec = TransmissionStateValueSpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=str(jacobians),
        output_csv=str(output_csv),
        figure_path=args.figure_path,
        table_path=args.table_path,
        horizon=int(args.horizon),
        decay=float(args.decay),
        weight_y=float(args.weight_y),
        weight_c=float(args.weight_c),
        weight_pi=float(args.weight_pi),
        num_folds=int(args.num_folds),
        ridge=float(args.ridge),
        sign_flip_draws=int(args.sign_flip_draws),
        max_shock_seeds=args.max_shock_seeds,
        note=(
            "Transmission index tau_t(H) is the local SSJ norm of future output-gap, "
            "consumption, and inflation responses to a one-period rate movement. "
            "Model F uses a fold-generated compact index trained from all distributional "
            "statistics, so it is not an oracle copy of the target."
        ),
    )
    spec_path = Path(args.spec_output)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(asdict(spec), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {args.figure_path}")
    print(f"Wrote {args.table_path}")


def _transmission_dataset(
    *,
    environment: HankSSJPolicyEnvironment,
    horizon: int,
    decay: float,
    weight_y: float,
    weight_c: float,
    weight_pi: float,
    max_shock_seeds: int | None,
) -> pd.DataFrame:
    tau = _tau_path(
        environment,
        horizon=horizon,
        decay=decay,
        weight_y=weight_y,
        weight_c=weight_c,
        weight_pi=weight_pi,
    )
    scenarios = list(environment.scenarios)
    if max_shock_seeds is not None:
        scenarios = scenarios[: max(1, int(max_shock_seeds))]
    rows: list[pd.DataFrame] = []
    for scenario in scenarios:
        shock_seed = _shock_seed_from_scenario(scenario)
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
            except (KeyError, ValueError):
                continue
            periods = min(aggregate.shape[0], distribution.shape[0], tau.size)
            frame = pd.DataFrame(aggregate[:periods], columns=AGGREGATE_FEATURES)
            for index, feature in enumerate(DISTRIBUTIONAL_FEATURES):
                frame[feature] = distribution[:periods, len(AGGREGATE_FEATURES) + index]
            frame.insert(0, "period", np.arange(periods, dtype=int))
            frame.insert(0, "observation_seed", int(observation_seed))
            frame.insert(0, "shock_seed", int(shock_seed))
            frame.insert(0, "scenario", scenario)
            frame["tau_transmission"] = tau[:periods]
            rows.append(frame)
    if not rows:
        raise ValueError("No rows were built. Check information inputs for filtered_aggregates and filtered_distribution.")
    dataset = pd.concat(rows, ignore_index=True)
    dataset.insert(0, "row_id", np.arange(len(dataset), dtype=int))
    return dataset


def _tau_path(
    environment: HankSSJPolicyEnvironment,
    *,
    horizon: int,
    decay: float,
    weight_y: float,
    weight_c: float,
    weight_pi: float,
) -> np.ndarray:
    effects = environment._effects
    periods = min(effects["output_gap"].shape[0], effects["C"].shape[0], effects["pi"].shape[0])
    tau = np.zeros(periods, dtype=float)
    for period in range(periods):
        total = 0.0
        for step in range(1, int(horizon) + 1):
            response_period = period + step
            if response_period >= periods:
                break
            total += (float(decay) ** (step - 1)) * (
                float(weight_y) * abs(float(effects["output_gap"][response_period, period]))
                + float(weight_c) * abs(float(effects["C"][response_period, period]))
                + float(weight_pi) * abs(float(effects["pi"][response_period, period]))
            )
        tau[period] = total
    return tau


def _crossfit_predictions(*, dataset: pd.DataFrame, ridge: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    for fold in sorted(dataset["fold"].unique()):
        train = dataset[~dataset["fold"].eq(fold)].copy()
        test = dataset[dataset["fold"].eq(fold)].copy()
        train, test, index_projection = _add_fold_transmission_index(train=train, test=test, ridge=ridge)
        coefficient_rows.extend(
            _coefficient_rows(
                projection=index_projection,
                model="first_stage_distributional_index",
                model_label="Первый этап: распределительные статистики -> компактный индекс трансмиссии",
                fold=int(fold),
            )
        )
        for spec in MODEL_SPECS:
            projection = _fit_ridge(train, tuple(spec["features"]), "tau_transmission", ridge=ridge)
            predicted = _predict_ridge(test, projection)
            out = test[["row_id", "scenario", "shock_seed", "observation_seed", "period", "tau_transmission"]].copy()
            out["model"] = str(spec["model"])
            out["model_label"] = str(spec["model_label"])
            out["predicted_tau"] = predicted
            out["fold"] = int(fold)
            prediction_rows.append(out)
            coefficient_rows.extend(
                _coefficient_rows(
                    projection=projection,
                    model=str(spec["model"]),
                    model_label=str(spec["model_label"]),
                    fold=int(fold),
                )
            )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions["error"] = predictions["predicted_tau"] - predictions["tau_transmission"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["squared_error"] = predictions["error"] ** 2
    return predictions, pd.DataFrame(coefficient_rows)


def _add_fold_transmission_index(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame, RidgeProjection]:
    projection = _fit_ridge(train, DISTRIBUTIONAL_FEATURES, "tau_transmission", ridge=ridge)
    train = train.copy()
    test = test.copy()
    train[TRANSMISSION_INDEX_FEATURE] = _predict_ridge(train, projection)
    test[TRANSMISSION_INDEX_FEATURE] = _predict_ridge(test, projection)
    return train, test, projection


def _summary_table(
    *,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    draws: int,
) -> pd.DataFrame:
    baseline = predictions[predictions["model"].eq("A_filtered_aggregates")][
        ["row_id", "abs_error", "squared_error"]
    ].rename(columns={"abs_error": "baseline_abs_error", "squared_error": "baseline_squared_error"})
    rows: list[dict[str, object]] = []
    y_all = dataset.drop_duplicates("row_id")["tau_transmission"].to_numpy(dtype=float)
    target_mean = float(np.mean(y_all))
    target_std = float(np.std(y_all, ddof=0))
    target_cv = target_std / abs(target_mean) if abs(target_mean) > 1e-18 else np.nan
    for spec in MODEL_SPECS:
        model = str(spec["model"])
        group = predictions[predictions["model"].eq(model)].copy()
        merged = group.merge(baseline, on="row_id", how="left", validate="one_to_one")
        y = merged["tau_transmission"].to_numpy(dtype=float)
        sse = float(merged["squared_error"].sum())
        sst = float(np.sum((y - y.mean()) ** 2))
        oof_r2 = 1.0 - sse / sst if sst > 0 else np.nan
        baseline_sse = float(merged["baseline_squared_error"].sum())
        baseline_r2 = 1.0 - baseline_sse / sst if sst > 0 else np.nan
        mae = float(merged["abs_error"].mean())
        baseline_mae = float(merged["baseline_abs_error"].mean())
        seed_gain = (
            merged.assign(mae_gain=merged["baseline_abs_error"] - merged["abs_error"])
            .groupby("shock_seed", sort=True)["mae_gain"]
            .mean()
            .to_numpy(dtype=float)
        )
        rows.append(
            {
                "model": model,
                "model_label": str(spec["model_label"]),
                "features": ",".join(spec["features"]),
                "num_observations": int(len(merged)),
                "num_shock_seed_clusters": int(merged["shock_seed"].nunique()),
                "target_mean": target_mean,
                "target_std": target_std,
                "target_cv": target_cv,
                "unique_target_values": int(dataset["tau_transmission"].nunique()),
                "oof_R2": oof_r2,
                "delta_oof_R2_vs_filtered_aggregates": oof_r2 - baseline_r2,
                "MAE": mae,
                "MAE_gain": baseline_mae - mae,
                "coefficient_sign_stability": _model_coefficient_sign_stability(
                    coefficients=coefficients,
                    model=model,
                    features=tuple(spec["features"]),
                ),
                "shock_seed_cluster_p": np.nan
                if model == "A_filtered_aggregates"
                else _cluster_sign_flip_greater(seed_gain, draws=draws),
                "positive_cluster_gain_share": float(np.mean(seed_gain > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _model_coefficient_sign_stability(
    *,
    coefficients: pd.DataFrame,
    model: str,
    features: tuple[str, ...],
) -> float:
    target_features = [feature for feature in features if feature not in AGGREGATE_FEATURES]
    if not target_features:
        target_features = list(features)
    frame = coefficients[
        coefficients["model"].eq(model)
        & coefficients["feature"].isin(target_features)
    ].copy()
    if frame.empty:
        return np.nan
    stabilities = []
    for _, group in frame.groupby("feature", sort=False):
        values = group["coefficient"].to_numpy(dtype=float)
        signs = np.sign(values)
        stabilities.append(max(float(np.mean(signs > 0.0)), float(np.mean(signs < 0.0)), float(np.mean(signs == 0.0))))
    return float(np.mean(stabilities))


def _coefficient_rows(
    *,
    projection: RidgeProjection,
    model: str,
    model_label: str,
    fold: int,
) -> list[dict[str, object]]:
    rows = [
        {
            "model": model,
            "model_label": model_label,
            "fold": int(fold),
            "feature": "intercept",
            "feature_label": "intercept",
            "coefficient": float(projection.intercept),
            "coefficient_sign": _sign(float(projection.intercept)),
        }
    ]
    for feature, coefficient in zip(projection.feature_names, projection.coefficients):
        rows.append(
            {
                "model": model,
                "model_label": model_label,
                "fold": int(fold),
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "coefficient": float(coefficient),
                "coefficient_sign": _sign(float(coefficient)),
            }
        )
    return rows


def _fit_ridge(frame: pd.DataFrame, features: tuple[str, ...], target: str, *, ridge: float) -> RidgeProjection:
    x = frame.loc[:, list(features)].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
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
    mean = np.asarray(projection.feature_mean, dtype=float)
    scale = np.asarray(projection.feature_scale, dtype=float)
    beta = np.asarray(projection.coefficients, dtype=float)
    return float(projection.intercept) + ((x - mean) / scale) @ beta


def _assign_shock_seed_folds(shock_seed: pd.Series, *, num_folds: int) -> np.ndarray:
    seeds = sorted(int(value) for value in pd.Series(shock_seed).drop_duplicates())
    folds = max(2, min(int(num_folds), len(seeds)))
    mapping = {seed: index % folds for index, seed in enumerate(seeds)}
    return pd.Series(shock_seed).map(mapping).to_numpy(dtype=int)


def _cluster_sign_flip_greater(values: np.ndarray, *, draws: int, seed: int = 5741) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(draws), values.size), replace=True)
    simulated = (signs * values).mean(axis=1)
    return float((np.sum(simulated >= observed) + 1.0) / (simulated.size + 1.0))


def _write_latex(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_label",
        "oof_R2",
        "delta_oof_R2_vs_filtered_aggregates",
        "MAE_gain",
        "coefficient_sign_stability",
        "shock_seed_cluster_p",
    ]
    label_by_model = {str(spec["model"]): str(spec["model_label"]) for spec in MODEL_SPECS}
    summary = summary.copy()
    if "model" in summary.columns:
        summary["model_label"] = summary["model"].map(label_by_model).fillna(summary["model_label"])
    display = summary.loc[:, columns].copy()
    display = display.rename(
        columns={
            "model_label": "Модель",
            "oof_R2": "OOF R2",
            "delta_oof_R2_vs_filtered_aggregates": "Прирост OOF R2",
            "MAE_gain": "Снижение MAE",
            "coefficient_sign_stability": "Устойчивость знака",
            "shock_seed_cluster_p": "p-value по shock seed",
        }
    )
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _plot(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = summary[~summary["model"].eq("A_filtered_aggregates")].copy()
    labels = [
        "MPC",
        "Доля низколиквидных",
        "Чувствительность к ставке",
        "Все статистики",
        "Показатель отклика",
    ][: len(frame)]
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    r2_values = frame["delta_oof_R2_vs_filtered_aggregates"].to_numpy(dtype=float) * 1e3
    mae_growth = -frame["MAE_gain"].to_numpy(dtype=float) * 1e5
    axes[0].barh(y, r2_values, color=np.where(r2_values >= 0.0, "#3ba895", "#c45a70"))
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=12)
    axes[0].set_xlabel("Изменение R² × 10³", fontsize=12)
    axes[0].set_title("Изменение R²", fontsize=13)
    axes[1].barh(y, mae_growth, color=np.where(mae_growth <= 0.0, "#3ba895", "#c45a70"))
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Рост MAE × 10⁵", fontsize=12)
    axes[1].set_title("Рост MAE", fontsize=13)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _shock_seed_from_scenario(scenario: str) -> int:
    try:
        return int(str(scenario).split("_")[-1])
    except ValueError:
        return abs(hash(str(scenario))) % (2**31)


def _sign(value: float, *, tol: float = 1e-14) -> str:
    if value > tol:
        return "positive"
    if value < -tol:
        return "negative"
    return "zero"


if __name__ == "__main__":
    main()
