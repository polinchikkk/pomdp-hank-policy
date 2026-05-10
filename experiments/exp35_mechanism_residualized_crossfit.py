from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp22_mechanism_optimal_rate_projection import (  # noqa: E402
    _fit_projection,
    _parse_seed_range,
    _predict_projection,
)
from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights  # noqa: E402


AGGREGATE_FEATURES = ("E_pi", "E_Y", "E_C")
DISTRIBUTIONAL_FEATURES = ("E_mean_mpc", "E_low_liquidity_share", "E_interest_exposure")

FEATURE_LABEL_RU = {
    "E_mean_mpc": "MPC",
    "E_low_liquidity_share": "Доля низколиквидных",
    "E_interest_exposure": "Процентная экспозиция",
}


@dataclass(frozen=True)
class ResidualizedCrossfitSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    figure_path: str
    seeds: tuple[int, ...]
    num_folds: int
    ridge: float
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fit residual mechanism test for distributional information.")
    parser.add_argument("--information-inputs", default="outputs/ssj/stochastic/state_space/information_inputs/information_state_inputs_long.csv")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/mechanism_residualized_crossfit")
    parser.add_argument("--figure-path", default="article/figures/fig_mechanism_residualized_crossfit.pdf")
    parser.add_argument("--seeds", default="900:911")
    parser.add_argument("--num-folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--permutation-draws", type=int, default=5000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seed_range(args.seeds)

    environment = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=Path(args.information_inputs),
        hank_observables_csv=Path(args.hank_observables),
        jacobians_npz=Path(args.jacobians),
        loss_weights=PolicyLossWeights(),
    )
    dataset = _dataset(environment, seeds=seeds)
    dataset["fold"] = _assign_folds(dataset, num_folds=int(args.num_folds))

    residuals, fold_models = _crossfit_residuals(dataset, ridge=float(args.ridge))
    summary, coefficients = _residual_regression_summary(
        residuals,
        fold_models=fold_models,
        ridge=float(args.ridge),
        permutation_draws=int(args.permutation_draws),
    )
    by_fold = _fold_stability(residuals, ridge=float(args.ridge))
    feature_tests = _feature_level_tests(residuals, ridge=float(args.ridge), permutation_draws=int(args.permutation_draws))

    dataset.to_csv(output_dir / "crossfit_mechanism_dataset.csv", index=False)
    residuals.to_csv(output_dir / "crossfit_residuals.csv", index=False)
    summary.to_csv(output_dir / "residualized_crossfit_summary.csv", index=False)
    coefficients.to_csv(output_dir / "residualized_crossfit_coefficients.csv", index=False)
    by_fold.to_csv(output_dir / "coefficient_stability_by_fold.csv", index=False)
    feature_tests.to_csv(output_dir / "residualized_feature_tests.csv", index=False)
    _write_latex(summary, output_dir / "table_residualized_crossfit_summary.tex")
    _write_latex(feature_tests, output_dir / "table_residualized_feature_tests.tex")
    _write_report(summary, by_fold, feature_tests, output_dir / "report_mechanism_residualized_crossfit.md")
    _plot(residuals, summary, feature_tests, figure_path)

    spec = ResidualizedCrossfitSpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        figure_path=args.figure_path,
        seeds=tuple(seeds),
        num_folds=int(args.num_folds),
        ridge=float(args.ridge),
        note=(
            "Cross-fit mechanism test: optimal rate and distributional features are residualized "
            "against filtered aggregates out-of-fold. The final regression tests whether residual "
            "distributional information predicts the residual local SSJ-optimal rate."
        ),
    )
    (output_dir / "mechanism_residualized_crossfit_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'residualized_crossfit_summary.csv'}")
    print(f"Wrote {output_dir / 'coefficient_stability_by_fold.csv'}")
    print(f"Wrote {figure_path}")


def _dataset(environment: HankSSJPolicyEnvironment, *, seeds: list[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scenario in environment.scenarios:
        target = environment.optimal_rate_path(scenario=scenario)
        for seed in seeds:
            aggregate = environment.feature_matrix(
                scenario=scenario,
                information_state="filtered_aggregates",
                seed=seed,
                feature_names=AGGREGATE_FEATURES,
            )
            distribution = environment.feature_matrix(
                scenario=scenario,
                information_state="filtered_distribution",
                seed=seed,
                feature_names=(*AGGREGATE_FEATURES, *DISTRIBUTIONAL_FEATURES),
            )
            periods = min(target.size, aggregate.shape[0], distribution.shape[0])
            frame = pd.DataFrame(aggregate[:periods], columns=AGGREGATE_FEATURES)
            for index, feature in enumerate(DISTRIBUTIONAL_FEATURES):
                frame[feature] = distribution[:periods, len(AGGREGATE_FEATURES) + index]
            frame.insert(0, "period", np.arange(periods))
            frame.insert(0, "observation_seed", int(seed))
            frame.insert(0, "scenario", scenario)
            frame["cluster"] = frame["scenario"].astype(str) + "__" + str(seed)
            frame["optimal_rate"] = target[:periods]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _assign_folds(dataset: pd.DataFrame, *, num_folds: int) -> np.ndarray:
    clusters = sorted(dataset["cluster"].unique())
    mapping = {cluster: index % int(num_folds) for index, cluster in enumerate(clusters)}
    return dataset["cluster"].map(mapping).to_numpy(dtype=int)


def _crossfit_residuals(dataset: pd.DataFrame, *, ridge: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    for fold in sorted(dataset["fold"].unique()):
        train = dataset[~dataset["fold"].eq(fold)].copy()
        test = dataset[dataset["fold"].eq(fold)].copy()

        rate_projection = _fit_projection(train[[*AGGREGATE_FEATURES, "optimal_rate"]], AGGREGATE_FEATURES, ridge=ridge)
        rate_prediction = _predict_projection(test[["scenario", "observation_seed", "period", *AGGREGATE_FEATURES, "optimal_rate"]], rate_projection)
        out = test[["scenario", "observation_seed", "period", "cluster", "fold", "optimal_rate", *AGGREGATE_FEATURES, *DISTRIBUTIONAL_FEATURES]].copy()
        out["rate_residual"] = out["optimal_rate"].to_numpy(dtype=float) - rate_prediction["predicted_rate"].to_numpy(dtype=float)
        for index, feature in enumerate(AGGREGATE_FEATURES):
            model_rows.append(
                {
                    "fold": int(fold),
                    "model": "rate_on_aggregates",
                    "target": "optimal_rate",
                    "term": feature,
                    "coefficient": float(rate_projection.coefficients[index]),
                }
            )

        for feature in DISTRIBUTIONAL_FEATURES:
            feature_train = train[[*AGGREGATE_FEATURES]].copy()
            feature_train["optimal_rate"] = train[feature].to_numpy(dtype=float)
            feature_projection = _fit_projection(feature_train, AGGREGATE_FEATURES, ridge=ridge)
            feature_test = test[["scenario", "observation_seed", "period", *AGGREGATE_FEATURES]].copy()
            feature_test["optimal_rate"] = test[feature].to_numpy(dtype=float)
            feature_prediction = _predict_projection(feature_test, feature_projection)
            out[f"{feature}_residual"] = out[feature].to_numpy(dtype=float) - feature_prediction["predicted_rate"].to_numpy(dtype=float)
            for index, aggregate_feature in enumerate(AGGREGATE_FEATURES):
                model_rows.append(
                    {
                        "fold": int(fold),
                        "model": "distribution_on_aggregates",
                        "target": feature,
                        "term": aggregate_feature,
                        "coefficient": float(feature_projection.coefficients[index]),
                    }
                )
        rows.append(out)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(model_rows)


def _residual_regression_summary(
    residuals: pd.DataFrame,
    *,
    fold_models: pd.DataFrame,
    ridge: float,
    permutation_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del fold_models
    feature_names = tuple(f"{feature}_residual" for feature in DISTRIBUTIONAL_FEATURES)
    x = residuals.loc[:, list(feature_names)].to_numpy(dtype=float)
    y = residuals["rate_residual"].to_numpy(dtype=float)
    beta, predicted = _ridge_fit_predict(x, y, ridge=ridge)
    sse = float(np.sum((y - predicted) ** 2))
    sst = float(np.sum(y**2))
    residual_r2 = 1.0 - sse / sst if sst > 0 else np.nan
    partial_r2 = residual_r2
    p_value = _cluster_sign_flip_p_value(
        y=y,
        predicted=predicted,
        clusters=residuals["cluster"].to_numpy(dtype=object),
        draws=permutation_draws,
    )
    summary = pd.DataFrame(
        [
            {
                "test": "crossfit_residual_distribution_predicts_residual_optimal_rate",
                "num_observations": int(len(residuals)),
                "num_clusters": int(residuals["cluster"].nunique()),
                "num_folds": int(residuals["fold"].nunique()),
                "residual_rmse_zero_model": float(np.sqrt(np.mean(y**2))),
                "residual_rmse_distribution_model": float(np.sqrt(np.mean((y - predicted) ** 2))),
                "residual_R2": residual_r2,
                "partial_R2": partial_r2,
                "crossfit_p_value": p_value,
            }
        ]
    )
    coefficients = pd.DataFrame(
        [
            {
                "feature": feature.replace("_residual", ""),
                "feature_ru": FEATURE_LABEL_RU[feature.replace("_residual", "")],
                "coefficient": float(value),
            }
            for feature, value in zip(feature_names, beta)
        ]
    )
    return summary, coefficients


def _fold_stability(residuals: pd.DataFrame, *, ridge: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_names = tuple(f"{feature}_residual" for feature in DISTRIBUTIONAL_FEATURES)
    for fold, frame in residuals.groupby("fold", sort=True):
        x = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
        y = frame["rate_residual"].to_numpy(dtype=float)
        beta, predicted = _ridge_fit_predict(x, y, ridge=ridge)
        sst = float(np.sum(y**2))
        r2 = 1.0 - float(np.sum((y - predicted) ** 2)) / sst if sst > 0 else np.nan
        for feature, coefficient in zip(feature_names, beta):
            rows.append(
                {
                    "fold": int(fold),
                    "feature": feature.replace("_residual", ""),
                    "feature_ru": FEATURE_LABEL_RU[feature.replace("_residual", "")],
                    "coefficient": float(coefficient),
                    "fold_R2": r2,
                    "num_observations": int(len(frame)),
                }
            )
    result = pd.DataFrame(rows)
    stability = (
        result.groupby("feature", sort=False)
        .agg(
            feature_ru=("feature_ru", "first"),
            mean_coefficient=("coefficient", "mean"),
            std_coefficient=("coefficient", "std"),
            positive_share=("coefficient", lambda values: float(np.mean(np.asarray(values) > 0))),
            negative_share=("coefficient", lambda values: float(np.mean(np.asarray(values) < 0))),
            mean_fold_R2=("fold_R2", "mean"),
            min_fold_R2=("fold_R2", "min"),
            max_fold_R2=("fold_R2", "max"),
            num_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    return stability


def _feature_level_tests(residuals: pd.DataFrame, *, ridge: float, permutation_draws: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = residuals["rate_residual"].to_numpy(dtype=float)
    clusters = residuals["cluster"].to_numpy(dtype=object)
    for feature in DISTRIBUTIONAL_FEATURES:
        x = residuals[[f"{feature}_residual"]].to_numpy(dtype=float)
        beta, predicted = _ridge_fit_predict(x, y, ridge=ridge)
        sst = float(np.sum(y**2))
        r2 = 1.0 - float(np.sum((y - predicted) ** 2)) / sst if sst > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "feature_ru": FEATURE_LABEL_RU[feature],
                "coefficient": float(beta[0]),
                "residual_R2": r2,
                "crossfit_p_value": _cluster_sign_flip_p_value(
                    y=y,
                    predicted=predicted,
                    clusters=clusters,
                    draws=permutation_draws,
                ),
            }
        )
    return pd.DataFrame(rows)


def _ridge_fit_predict(x: np.ndarray, y: np.ndarray, *, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0, ddof=0), 1e-8)
    x_std = (x - mean) / scale
    beta = np.linalg.solve(x_std.T @ x_std + float(ridge) * np.eye(x_std.shape[1]), x_std.T @ y)
    return beta, x_std @ beta


def _cluster_sign_flip_p_value(
    *,
    y: np.ndarray,
    predicted: np.ndarray,
    clusters: np.ndarray,
    draws: int,
    seed: int = 2027,
) -> float:
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    clusters = np.asarray(clusters, dtype=object)
    observed = _r2(y, predicted)
    unique = np.asarray(sorted(set(clusters)), dtype=object)
    rng = np.random.default_rng(seed)
    simulated: list[float] = []
    for _ in range(int(draws)):
        signs_by_cluster = dict(zip(unique, rng.choice((-1.0, 1.0), size=len(unique))))
        signs = np.asarray([signs_by_cluster[cluster] for cluster in clusters], dtype=float)
        simulated.append(_r2(y, predicted * signs))
    simulated_array = np.asarray(simulated, dtype=float)
    return float((np.sum(simulated_array >= observed) + 1) / (len(simulated_array) + 1))


def _r2(y: np.ndarray, predicted: np.ndarray) -> float:
    sst = float(np.sum(y**2))
    if sst <= 0:
        return np.nan
    return 1.0 - float(np.sum((y - predicted) ** 2)) / sst


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    for column in numeric:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(summary: pd.DataFrame, by_fold: pd.DataFrame, feature_tests: pd.DataFrame, path: Path) -> None:
    row = summary.iloc[0]
    lines = [
        "# Cross-fit residual mechanism test",
        "",
        "Тест удаляет из локально оптимальной ставки и распределительных признаков ту часть, которая",
        "предсказывается фильтрованными агрегатами out-of-fold. Затем проверяется связь между остатками.",
        "",
        f"- residual R2: {row['residual_R2']:.6g};",
        f"- partial R2: {row['partial_R2']:.6g};",
        f"- crossfit p-value: {row['crossfit_p_value']:.6g};",
        f"- число кластеров: {int(row['num_clusters'])};",
        "",
        "## Устойчивость коэффициентов по фолдам",
        "",
        by_fold.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Однопризнаковые residual-тесты",
        "",
        feature_tests.to_markdown(index=False, floatfmt=".6g"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(residuals: pd.DataFrame, summary: pd.DataFrame, feature_tests: pd.DataFrame, figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].scatter(
        residuals["E_mean_mpc_residual"],
        residuals["rate_residual"],
        s=7,
        alpha=0.18,
        edgecolor="none",
        color="#276c8f",
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set_title("Residual optimal rate vs residual MPC")
    axes[0].set_xlabel("MPC residual")
    axes[0].set_ylabel("Optimal-rate residual")

    x = np.arange(len(feature_tests))
    axes[1].bar(x, feature_tests["residual_R2"], color="#c06c2d")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_tests["feature_ru"], rotation=15, ha="right")
    axes[1].set_title("Single-feature residual R2")
    axes[1].set_ylabel("Residual R2")
    fig.suptitle(f"Cross-fit residual mechanism, partial R2 = {summary.iloc[0]['partial_R2']:.3g}")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
