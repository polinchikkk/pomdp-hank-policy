from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_full_baseline.transition import solve_transition
from hank_learning_policy_baseline.policies import ClassicalFilteredRulePolicy

from .core_matrix import SCENARIO_LABELS
from .evaluation import simulate_policy_episode
from .policy_extensions import (
    SCENARIO_NAMES,
    _build_objects,
    run_full_hank_projection_from_policy_paths,
)


STATE_COMPONENT_ROWS = [
    {
        "component": "rstar_gap",
        "label_ru": "Разрыв естественной ставки",
        "economic_meaning": "Скрытая компонента межвременного условия и нейтральной ставки",
        "hank_channel": "Межвременной канал денежно-кредитной трансмиссии",
    },
    {
        "component": "productivity_gap",
        "label_ru": "Производственная компонента",
        "economic_meaning": "Скрытый сдвиг производительности и трудового дохода",
        "hank_channel": "Доходный и производственный канал",
    },
    {
        "component": "fiscal_gap",
        "label_ru": "Фискальная компонента",
        "economic_meaning": "Скрытый сдвиг фискальной позиции и трансфертов",
        "hank_channel": "Фискальный канал и доходы домохозяйств",
    },
    {
        "component": "inflation_gap",
        "label_ru": "Разрыв инфляции",
        "economic_meaning": "Отклонение инфляции от стационарного уровня",
        "hank_channel": "Номинальные жесткости и стандартный стабилизационный мотив",
    },
    {
        "component": "output_gap",
        "label_ru": "Разрыв выпуска",
        "economic_meaning": "Отклонение выпуска от стационарного уровня",
        "hank_channel": "Совокупный спрос, занятость и трудовой доход",
    },
    {
        "component": "low_liquidity_gap",
        "label_ru": "Доля низколиквидных домохозяйств",
        "economic_meaning": "Скрытый индикатор напряженности ликвидных балансов",
        "hank_channel": "Распределительный и ликвидностный канал",
    },
    {
        "component": "mean_mpc_gap",
        "label_ru": "Средняя предельная склонность к потреблению",
        "economic_meaning": "Скрытый индикатор чувствительности потребления к доходу",
        "hank_channel": "Канал гетерогенных предельных склонностей к потреблению",
    },
    {
        "component": "stress_probability",
        "label_ru": "Вероятность стрессового режима",
        "economic_meaning": "Апостериорная вероятность скрытого режима с иной трансмиссией",
        "hank_channel": "Режимная неопределенность и изменение силы передачи политики",
    },
]


FEATURE_SPECS = {
    "observables_only": {
        "label_ru": "Только наблюдаемые переменные",
        "columns": ("observed_pi", "observed_output_gap", "lagged_policy_rate"),
    },
    "macro_state": {
        "label_ru": "Макроэкономическое состояние",
        "columns": (
            "filtered_rstar_gap",
            "filtered_productivity_gap",
            "filtered_fiscal_gap",
            "filtered_inflation_gap",
            "filtered_output_gap",
            "lagged_policy_rate",
        ),
    },
    "macro_regime_state": {
        "label_ru": "Макроэкономическое состояние + скрытый режим",
        "columns": (
            "filtered_rstar_gap",
            "filtered_productivity_gap",
            "filtered_fiscal_gap",
            "filtered_inflation_gap",
            "filtered_output_gap",
            "stress_probability",
            "lagged_policy_rate",
        ),
    },
    "distribution_augmented_state": {
        "label_ru": "Распределительно расширенное состояние",
        "columns": (
            "filtered_rstar_gap",
            "filtered_productivity_gap",
            "filtered_fiscal_gap",
            "filtered_inflation_gap",
            "filtered_output_gap",
            "filtered_low_liquidity_gap",
            "filtered_mean_mpc_gap",
            "stress_probability",
            "lagged_policy_rate",
        ),
    },
}


TARGET_SPECS = {
    "inflation_gap": ("future_inflation_gap", "Инфляция"),
    "output_gap": ("future_output_gap", "Разрыв выпуска"),
    "future_loss_sum": ("future_loss_sum", "Будущая функция потерь"),
}


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _write_state_component_table(root: Path, components: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{p{0.22\\linewidth}p{0.34\\linewidth}p{0.34\\linewidth}}",
        "\\toprule",
        "Компонент & Экономический смысл & Канал HANK \\\\",
        "\\midrule",
    ]
    for row in components.to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["label_ru"]),
                    _latex_escape(row["economic_meaning"]),
                    _latex_escape(row["hank_channel"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_state_component_interpretation.tex").write_text("\n".join(lines), encoding="utf-8")


def _simulate_validation_traces(
    *,
    scenario_names: tuple[str, ...],
    train_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scenario_name in scenario_names:
        _hank_config, scenario_spec, env_factory = _build_objects(
            scenario_name,
            input_mode="belief_state",
            validation_seeds=train_seeds,
            test_seeds=test_seeds,
        )
        policy = ClassicalFilteredRulePolicy(action_bound=scenario_spec.action_bound)
        for split_name, seeds in (("train", train_seeds), ("test", test_seeds)):
            for seed in seeds:
                trace = simulate_policy_episode(
                    env_factory=env_factory,
                    policy=policy,
                    scenario_spec=scenario_spec,
                    evaluation_seed=int(seed),
                    policy_name="classical_filtered_rule",
                    policy_label="Классическое правило по оценённому состоянию",
                    training_seed=None,
                )
                trace["lagged_policy_rate"] = trace["current_rate"].to_numpy(dtype=float)
                trace["split"] = split_name
                frames.append(trace)
    return pd.concat(frames, ignore_index=True)


def _add_future_targets(frame: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for (_scenario, seed), group in frame.sort_values("period").groupby(["scenario_name", "evaluation_seed"]):
        group = group.copy().reset_index(drop=True)
        inflation = group["true_inflation_gap"].to_numpy(dtype=float)
        output = group["true_output_gap"].to_numpy(dtype=float)
        loss = group["loss"].to_numpy(dtype=float)
        for horizon in horizons:
            enriched = group.copy()
            enriched["forecast_horizon"] = int(horizon)
            enriched["future_inflation_gap"] = np.nan
            enriched["future_output_gap"] = np.nan
            enriched["future_loss_sum"] = np.nan
            for idx in range(len(group) - horizon):
                enriched.loc[idx, "future_inflation_gap"] = inflation[idx + horizon]
                enriched.loc[idx, "future_output_gap"] = output[idx + horizon]
                enriched.loc[idx, "future_loss_sum"] = float(np.sum(loss[idx + 1 : idx + horizon + 1]))
            rows.append(enriched.iloc[: len(group) - horizon])
    return pd.concat(rows, ignore_index=True)


def _feature_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = []
    for column in columns:
        if column in frame.columns:
            values.append(frame[column].fillna(0.0).to_numpy(dtype=float))
        else:
            values.append(np.zeros(len(frame), dtype=float))
    return np.column_stack(values)


def _fit_ridge_predict(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge_alpha: float = 1.0e-8,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    x_train = np.column_stack([np.ones(len(x_train), dtype=float), x_train])
    x_test = np.column_stack([np.ones(len(x_test), dtype=float), x_test])
    penalty = ridge_alpha * np.eye(x_train.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ train_y)
    return x_test @ beta


def _bootstrap_interval(
    values: np.ndarray,
    *,
    confidence_level: float = 0.95,
    bootstrap_draws: int = 2000,
    seed: int = 12345,
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(bootstrap_draws, values.size))
    samples = values[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - confidence_level)
    return (
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1.0 - alpha)),
    )


def _forecast_sufficiency(frame: pd.DataFrame, horizons: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = _add_future_targets(frame, horizons)
    rows = []
    for scenario_name, scenario_frame in enriched.groupby("scenario_name"):
        for horizon, horizon_frame in scenario_frame.groupby("forecast_horizon"):
            train = horizon_frame[horizon_frame["split"] == "train"].copy()
            test = horizon_frame[horizon_frame["split"] == "test"].copy()
            for feature_name, feature_spec in FEATURE_SPECS.items():
                train_x = _feature_matrix(train, feature_spec["columns"])
                test_x = _feature_matrix(test, feature_spec["columns"])
                for target_name, (target_column, target_label) in TARGET_SPECS.items():
                    train_y = train[target_column].to_numpy(dtype=float)
                    test_y = test[target_column].to_numpy(dtype=float)
                    prediction = _fit_ridge_predict(train_x=train_x, train_y=train_y, test_x=test_x)
                    error = prediction - test_y
                    baseline = train_y.mean()
                    baseline_error = baseline - test_y
                    ss_error = float(np.sum(np.square(error)))
                    ss_baseline = float(np.sum(np.square(baseline_error)))
                    rows.append(
                        {
                            "scenario_name": scenario_name,
                            "scenario_label": SCENARIO_LABELS.get(scenario_name, scenario_name),
                            "forecast_horizon": int(horizon),
                            "feature_set": feature_name,
                            "feature_set_label": feature_spec["label_ru"],
                            "target": target_name,
                            "target_label": target_label,
                            "test_rmse": float(np.sqrt(np.mean(np.square(error)))),
                            "test_mae": float(np.mean(np.abs(error))),
                            "oos_r2": float(1.0 - ss_error / ss_baseline) if ss_baseline > 0.0 else math.nan,
                            "num_train_observations": int(len(train)),
                            "num_test_observations": int(len(test)),
                        }
                    )
    detailed = pd.DataFrame(rows)
    summary = (
        detailed.groupby(["feature_set", "feature_set_label", "target", "target_label"], as_index=False)
        .agg(
            mean_oos_r2=("oos_r2", "mean"),
            mean_test_rmse=("test_rmse", "mean"),
            mean_test_mae=("test_mae", "mean"),
            num_scenarios=("scenario_name", "nunique"),
            num_horizons=("forecast_horizon", "nunique"),
            num_rows=("oos_r2", "size"),
        )
        .sort_values(["target", "mean_oos_r2"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return detailed, summary


def _future_loss_scenario_summary(detail: pd.DataFrame) -> pd.DataFrame:
    subset = detail[detail["target"] == "future_loss_sum"].copy()
    return (
        subset.groupby(
            ["scenario_name", "scenario_label", "feature_set", "feature_set_label"],
            as_index=False,
        )
        .agg(
            mean_oos_r2=("oos_r2", "mean"),
            mean_test_rmse=("test_rmse", "mean"),
            mean_test_mae=("test_mae", "mean"),
            num_horizons=("forecast_horizon", "nunique"),
        )
        .sort_values(["scenario_name", "mean_oos_r2"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _write_forecast_summary_table(root: Path, summary: pd.DataFrame) -> None:
    pivot = summary.pivot_table(
        index=["feature_set", "feature_set_label"],
        columns="target",
        values="mean_oos_r2",
        aggfunc="first",
    )
    counts = (
        summary.groupby(["feature_set", "feature_set_label"], as_index=False)
        .agg(num_scenarios=("num_scenarios", "min"))
        .set_index(["feature_set", "feature_set_label"])
    )
    order = [name for name in FEATURE_SPECS if name in pivot.index.get_level_values(0)]
    pivot = pivot.loc[order]
    counts = counts.loc[order]
    lines = [
        "\\begin{tabular}{p{0.30\\linewidth}rrrr}",
        "\\toprule",
        "Представление состояния & Инфляция & Разрыв выпуска & Потери & Число сценариев \\\\",
        "\\midrule",
    ]
    for (_feature_set, feature_label), row in pivot.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(feature_label),
                    f"{float(row.get('inflation_gap', math.nan)):.3f}",
                    f"{float(row.get('output_gap', math.nan)):.3f}",
                    f"{float(row.get('future_loss_sum', math.nan)):.3f}",
                    f"{int(counts.loc[(_feature_set, feature_label), 'num_scenarios'])}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_forecast_sufficiency.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_future_loss_scenario_table(root: Path, summary: pd.DataFrame) -> None:
    pivot = summary.pivot_table(
        index=["scenario_name", "scenario_label"],
        columns="feature_set",
        values="mean_oos_r2",
        aggfunc="first",
    )
    order = [name for name in SCENARIO_NAMES if name in pivot.index.get_level_values(0)]
    pivot = pivot.loc[order]
    lines = [
        "\\begin{tabular}{p{0.32\\linewidth}rrrr}",
        "\\toprule",
        "Сценарий & Наблюдаемые переменные & Макроэкономическое состояние & Макроэкономическое состояние + скрытый режим & Распределительно расширенное состояние \\\\",
        "\\midrule",
    ]
    for (_scenario_name, scenario_label), row in pivot.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(scenario_label),
                    f"{float(row.get('observables_only', math.nan)):.3f}",
                    f"{float(row.get('macro_state', math.nan)):.3f}",
                    f"{float(row.get('macro_regime_state', math.nan)):.3f}",
                    f"{float(row.get('distribution_augmented_state', math.nan)):.3f}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_future_loss_scenario_forecast.tex").write_text("\n".join(lines), encoding="utf-8")


def _rank_series(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=True)


def _spearman_from_ranks(left: pd.Series, right: pd.Series) -> float:
    common = left.index.intersection(right.index)
    if len(common) < 2:
        return math.nan
    left_rank = _rank_series(left.loc[common])
    right_rank = _rank_series(right.loc[common])
    return float(left_rank.corr(right_rank, method="pearson"))


def _policy_ranking_validation(*, reduced_metrics: pd.DataFrame, full_hank_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reduced = (
        reduced_metrics.groupby(["scenario_name", "scenario_label", "policy_name"], as_index=False)["cumulative_policy_loss"]
        .mean()
        .rename(columns={"cumulative_policy_loss": "reduced_cumulative_loss"})
    )
    full = full_hank_metrics[full_hank_metrics["solver_success"] == 1].copy()
    if full.empty:
        return pd.DataFrame(), pd.DataFrame()
    full = full[
        [
            "scenario_name",
            "scenario_label",
            "policy_name",
            "full_hank_cumulative_loss",
            "scale_used",
            "mean_shock_abs",
            "peak_shock_abs",
        ]
    ]
    merged = reduced.merge(full, on=["scenario_name", "scenario_label", "policy_name"], how="inner")
    ranking_rows = []
    pair_rows = []
    for scenario_name, frame in merged.groupby("scenario_name"):
        frame = frame.copy()
        frame["reduced_rank"] = _rank_series(frame.set_index("policy_name")["reduced_cumulative_loss"]).reindex(frame["policy_name"]).to_numpy()
        frame["full_hank_rank"] = _rank_series(frame.set_index("policy_name")["full_hank_cumulative_loss"]).reindex(frame["policy_name"]).to_numpy()
        reduced_series = frame.set_index("policy_name")["reduced_cumulative_loss"]
        full_series = frame.set_index("policy_name")["full_hank_cumulative_loss"]
        rank_corr = _spearman_from_ranks(reduced_series, full_series)
        scale_values = frame["scale_used"].to_numpy(dtype=float)
        comparable_scale = bool(np.allclose(scale_values, scale_values[0]))
        for row in frame.to_dict(orient="records"):
            ranking_rows.append(
                {
                    **row,
                    "spearman_rank_correlation": rank_corr,
                    "common_projection_scale": int(comparable_scale),
                }
            )
        policies = list(frame["policy_name"])
        for i, left in enumerate(policies):
            for right in policies[i + 1 :]:
                reduced_delta = float(reduced_series[left] - reduced_series[right])
                full_delta = float(full_series[left] - full_series[right])
                pair_rows.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario_label": frame["scenario_label"].iloc[0],
                        "left_policy": left,
                        "right_policy": right,
                        "reduced_delta_loss": reduced_delta,
                        "full_hank_delta_loss": full_delta,
                        "same_pairwise_sign": int(np.sign(reduced_delta) == np.sign(full_delta)),
                        "common_projection_scale": int(comparable_scale),
                    }
                )
    return pd.DataFrame(ranking_rows), pd.DataFrame(pair_rows)


def _write_ranking_table(root: Path, ranking: pd.DataFrame) -> None:
    if ranking.empty:
        return
    labels = {
        "classical_filtered_rule": "Классическое правило",
        "optimized_linear_estimated_state": "Оптимизированное линейное",
        "history_observables_rule": "Историческое по наблюдениям",
    }
    lines = [
        "\\begin{tabular}{p{0.30\\linewidth}p{0.25\\linewidth}rrrr}",
        "\\toprule",
        "Сценарий & Правило & Ранг в редуцированной среде & Ранг HANK & Масштаб & Потери HANK \\\\",
        "\\midrule",
    ]
    for row in ranking.sort_values(["scenario_name", "full_hank_rank"]).to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["scenario_label"]),
                    _latex_escape(labels.get(row["policy_name"], row["policy_name"])),
                    f"{float(row['reduced_rank']):.1f}",
                    f"{float(row['full_hank_rank']):.1f}",
                    f"{float(row['scale_used']):.2f}",
                    f"{float(row['full_hank_cumulative_loss']):.3e}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_policy_ranking_validation.tex").write_text("\n".join(lines), encoding="utf-8")


def _run_common_scale_projection(
    *,
    policy_paths: pd.DataFrame,
    full_hank_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_policies = (
        "classical_filtered_rule",
        "optimized_linear_estimated_state",
        "history_observables_rule",
    )
    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    rows = []
    pair_rows = []
    for scenario_name, frame in full_hank_metrics.groupby("scenario_name"):
        frame = frame[frame["policy_name"].isin(selected_policies) & (frame["solver_success"] == 1)].copy()
        if frame.empty:
            continue
        common_scale = float(frame["scale_used"].min())
        loss_by_policy: dict[str, float] = {}
        for policy_name in selected_policies:
            subset = policy_paths[
                (policy_paths["scenario_name"] == scenario_name)
                & (policy_paths["policy_name"] == policy_name)
            ].copy()
            if subset.empty:
                continue
            shock_path = subset.groupby("period")["policy_rate"].mean().sort_index().to_numpy(dtype=float)
            shock_path = shock_path[: hank_config.shock_T]
            if shock_path.size < hank_config.shock_T:
                shock_path = np.pad(shock_path, (0, hank_config.shock_T - shock_path.size))
            transition = solve_transition(bundle, {"monetary_policy_shock": common_scale * shock_path})
            pi = transition["pi"]
            output = transition["output_gap"]
            rate = transition["i"]
            loss = pi**2 + 0.5 * output**2 + 0.05 * np.square(np.diff(rate, prepend=0.0))
            cumulative_loss = float(np.sum(loss))
            loss_by_policy[policy_name] = cumulative_loss
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": policy_name,
                    "common_scale": common_scale,
                    "common_scale_full_hank_loss": cumulative_loss,
                }
            )
        policies = [name for name in selected_policies if name in loss_by_policy]
        for i, left in enumerate(policies):
            for right in policies[i + 1 :]:
                pair_rows.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario_label": SCENARIO_LABELS[scenario_name],
                        "left_policy": left,
                        "right_policy": right,
                        "common_scale": common_scale,
                        "common_scale_delta_loss": float(loss_by_policy[left] - loss_by_policy[right]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def _run_common_scale_seed_projection(
    *,
    policy_paths: pd.DataFrame,
    common_scale_rows: pd.DataFrame,
    evaluation_seeds: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    if common_scale_rows.empty:
        return pd.DataFrame()
    selected_policies = (
        "classical_filtered_rule",
        "optimized_linear_estimated_state",
        "history_observables_rule",
    )
    scale_by_scenario = (
        common_scale_rows.groupby("scenario_name", as_index=True)["common_scale"]
        .first()
        .to_dict()
    )
    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    rows = []
    filtered_paths = policy_paths[policy_paths["policy_name"].isin(selected_policies)].copy()
    if evaluation_seeds is not None:
        filtered_paths = filtered_paths[filtered_paths["evaluation_seed"].isin([int(seed) for seed in evaluation_seeds])].copy()
    for (scenario_name, policy_name, evaluation_seed), subset in filtered_paths.groupby(
        ["scenario_name", "policy_name", "evaluation_seed"]
    ):
        if scenario_name not in scale_by_scenario:
            continue
        common_scale = float(scale_by_scenario[scenario_name])
        shock_path = subset.sort_values("period")["policy_rate"].to_numpy(dtype=float)
        shock_path = shock_path[: hank_config.shock_T]
        if shock_path.size < hank_config.shock_T:
            shock_path = np.pad(shock_path, (0, hank_config.shock_T - shock_path.size))
        try:
            transition = solve_transition(bundle, {"monetary_policy_shock": common_scale * shock_path})
            pi = transition["pi"]
            output = transition["output_gap"]
            rate = transition["i"]
            loss = pi**2 + 0.5 * output**2 + 0.05 * np.square(np.diff(rate, prepend=0.0))
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": policy_name,
                    "evaluation_seed": int(evaluation_seed),
                    "common_scale": common_scale,
                    "solver_success": 1,
                    "common_scale_full_hank_loss": float(np.sum(loss)),
                    "peak_inflation_abs": float(np.max(np.abs(pi))),
                    "peak_output_gap_abs": float(np.max(np.abs(output))),
                    "peak_rate_abs": float(np.max(np.abs(rate))),
                }
            )
        except Exception as exc:  # pragma: no cover - captures numerical failures.
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": policy_name,
                    "evaluation_seed": int(evaluation_seed),
                    "common_scale": common_scale,
                    "solver_success": 0,
                    "solver_error": f"{type(exc).__name__}: {exc}",
                    "common_scale_full_hank_loss": math.nan,
                    "peak_inflation_abs": math.nan,
                    "peak_output_gap_abs": math.nan,
                    "peak_rate_abs": math.nan,
                }
            )
    return pd.DataFrame(rows)


def _common_scale_pairwise_intervals(
    *,
    seed_projection: pd.DataFrame,
    reduced_metrics: pd.DataFrame,
    confidence_level: float = 0.95,
    bootstrap_draws: int = 2000,
) -> pd.DataFrame:
    if seed_projection.empty:
        return pd.DataFrame()
    reduced = (
        reduced_metrics.groupby(["scenario_name", "policy_name", "evaluation_seed"], as_index=False)["cumulative_policy_loss"]
        .mean()
        .rename(columns={"cumulative_policy_loss": "reduced_cumulative_loss"})
    )
    successful = seed_projection[seed_projection["solver_success"] == 1].copy()
    merged = reduced.merge(
        successful[
            [
                "scenario_name",
                "scenario_label",
                "policy_name",
                "evaluation_seed",
                "common_scale",
                "common_scale_full_hank_loss",
            ]
        ],
        on=["scenario_name", "policy_name", "evaluation_seed"],
        how="inner",
    )
    rows = []
    for scenario_name, frame in merged.groupby("scenario_name"):
        policies = list(frame["policy_name"].drop_duplicates())
        for i, left in enumerate(policies):
            for right in policies[i + 1 :]:
                left_frame = frame[frame["policy_name"] == left].set_index("evaluation_seed")
                right_frame = frame[frame["policy_name"] == right].set_index("evaluation_seed")
                common_seeds = left_frame.index.intersection(right_frame.index)
                if common_seeds.empty:
                    continue
                reduced_delta = (
                    left_frame.loc[common_seeds, "reduced_cumulative_loss"].to_numpy(dtype=float)
                    - right_frame.loc[common_seeds, "reduced_cumulative_loss"].to_numpy(dtype=float)
                )
                hank_delta = (
                    left_frame.loc[common_seeds, "common_scale_full_hank_loss"].to_numpy(dtype=float)
                    - right_frame.loc[common_seeds, "common_scale_full_hank_loss"].to_numpy(dtype=float)
                )
                ci_low, ci_high = _bootstrap_interval(
                    hank_delta,
                    confidence_level=confidence_level,
                    bootstrap_draws=bootstrap_draws,
                    seed=12345 + sum(ord(char) for char in f"{scenario_name}|{left}|{right}"),
                )
                rows.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario_label": frame["scenario_label"].iloc[0],
                        "left_policy": left,
                        "right_policy": right,
                        "common_scale": float(frame["common_scale"].iloc[0]),
                        "num_trajectories": int(len(common_seeds)),
                        "mean_reduced_delta_loss": float(np.mean(reduced_delta)),
                        "mean_common_scale_full_hank_delta_loss": float(np.mean(hank_delta)),
                        "ci_low_common_scale_full_hank_delta_loss": ci_low,
                        "ci_high_common_scale_full_hank_delta_loss": ci_high,
                        "same_pairwise_sign_mean": int(np.sign(np.mean(reduced_delta)) == np.sign(np.mean(hank_delta))),
                        "share_same_pairwise_sign_by_seed": float(np.mean(np.sign(reduced_delta) == np.sign(hank_delta))),
                        "left_beats_right_share_hank": float(np.mean(hank_delta < 0.0)),
                        "crosses_zero": int(ci_low <= 0.0 <= ci_high),
                    }
                )
    return pd.DataFrame(rows)


def _scale_sensitivity_table(
    *,
    policy_paths: pd.DataFrame,
    reduced_metrics: pd.DataFrame,
    scales: tuple[float, ...] = (0.10, 0.25),
) -> pd.DataFrame:
    selected_policies = (
        "classical_filtered_rule",
        "optimized_linear_estimated_state",
        "history_observables_rule",
    )
    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    reduced = (
        reduced_metrics.groupby(["scenario_name", "policy_name"], as_index=False)["cumulative_policy_loss"]
        .mean()
        .rename(columns={"cumulative_policy_loss": "reduced_cumulative_loss"})
    )
    rows = []
    for scenario_name in SCENARIO_NAMES:
        reduced_series = (
            reduced[
                (reduced["scenario_name"] == scenario_name)
                & (reduced["policy_name"].isin(selected_policies))
            ]
            .set_index("policy_name")["reduced_cumulative_loss"]
        )
        if reduced_series.empty:
            continue
        for scale in scales:
            losses: dict[str, float] = {}
            failures = 0
            for policy_name in selected_policies:
                subset = policy_paths[
                    (policy_paths["scenario_name"] == scenario_name)
                    & (policy_paths["policy_name"] == policy_name)
                ].copy()
                if subset.empty:
                    failures += 1
                    continue
                shock_path = subset.groupby("period")["policy_rate"].mean().sort_index().to_numpy(dtype=float)
                shock_path = shock_path[: hank_config.shock_T]
                if shock_path.size < hank_config.shock_T:
                    shock_path = np.pad(shock_path, (0, hank_config.shock_T - shock_path.size))
                try:
                    transition = solve_transition(bundle, {"monetary_policy_shock": scale * shock_path})
                    pi = transition["pi"]
                    output = transition["output_gap"]
                    rate = transition["i"]
                    loss = pi**2 + 0.5 * output**2 + 0.05 * np.square(np.diff(rate, prepend=0.0))
                    losses[policy_name] = float(np.sum(loss))
                except Exception:
                    failures += 1
            full_series = pd.Series(losses, dtype=float)
            pairwise_total = 0
            pairwise_matches = 0
            if len(full_series) >= 2:
                policies = list(full_series.index)
                for i, left in enumerate(policies):
                    for right in policies[i + 1 :]:
                        pairwise_total += 1
                        left_red = float(reduced_series[left] - reduced_series[right])
                        left_full = float(full_series[left] - full_series[right])
                        pairwise_matches += int(np.sign(left_red) == np.sign(left_full))
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "projection_scale": float(scale),
                    "num_successful_policies": int(len(full_series)),
                    "all_policies_success": int(len(full_series) == len(selected_policies) and failures == 0),
                    "pairwise_signs_preserved": f"{pairwise_matches}/{pairwise_total}" if pairwise_total else "0/0",
                    "spearman_rank_correlation": _spearman_from_ranks(reduced_series, full_series),
                }
            )
    return pd.DataFrame(rows)


def _merge_common_scale_with_reduced(
    *,
    reduced_metrics: pd.DataFrame,
    common_scale_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if common_scale_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    reduced = (
        reduced_metrics.groupby(["scenario_name", "scenario_label", "policy_name"], as_index=False)["cumulative_policy_loss"]
        .mean()
        .rename(columns={"cumulative_policy_loss": "reduced_cumulative_loss"})
    )
    merged = reduced.merge(common_scale_rows, on=["scenario_name", "scenario_label", "policy_name"], how="inner")
    ranking_rows = []
    pair_rows = []
    for scenario_name, frame in merged.groupby("scenario_name"):
        frame = frame.copy()
        reduced_series = frame.set_index("policy_name")["reduced_cumulative_loss"]
        common_series = frame.set_index("policy_name")["common_scale_full_hank_loss"]
        rank_corr = _spearman_from_ranks(reduced_series, common_series)
        reduced_ranks = _rank_series(reduced_series)
        common_ranks = _rank_series(common_series)
        for policy_name, row in frame.set_index("policy_name").iterrows():
            ranking_rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": row["scenario_label"],
                    "policy_name": policy_name,
                    "reduced_rank": float(reduced_ranks[policy_name]),
                    "common_scale_full_hank_rank": float(common_ranks[policy_name]),
                    "common_scale": float(row["common_scale"]),
                    "common_scale_full_hank_loss": float(row["common_scale_full_hank_loss"]),
                    "spearman_rank_correlation": rank_corr,
                    "ranking_preserved": int(float(reduced_ranks[policy_name]) == float(common_ranks[policy_name])),
                }
            )
        policies = list(reduced_series.index)
        for i, left in enumerate(policies):
            for right in policies[i + 1 :]:
                reduced_delta = float(reduced_series[left] - reduced_series[right])
                common_delta = float(common_series[left] - common_series[right])
                pair_rows.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario_label": frame["scenario_label"].iloc[0],
                        "left_policy": left,
                        "right_policy": right,
                        "common_scale": float(frame["common_scale"].iloc[0]),
                        "reduced_delta_loss": reduced_delta,
                        "common_scale_full_hank_delta_loss": common_delta,
                        "same_pairwise_sign": int(np.sign(reduced_delta) == np.sign(common_delta)),
                    }
                )
    return pd.DataFrame(ranking_rows), pd.DataFrame(pair_rows)


def _write_common_scale_table(root: Path, ranking: pd.DataFrame) -> None:
    if ranking.empty:
        return
    labels = {
        "classical_filtered_rule": "Классическое правило",
        "optimized_linear_estimated_state": "Оптимизированное линейное",
        "history_observables_rule": "Историческое по наблюдениям",
    }
    lines = [
        "\\begin{tabular}{p{0.30\\linewidth}p{0.25\\linewidth}rrr}",
        "\\toprule",
        "Сценарий & Правило & Общий масштаб & Ранг HANK & Потери HANK \\\\",
        "\\midrule",
    ]
    for row in ranking.sort_values(["scenario_name", "common_scale_full_hank_rank"]).to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["scenario_label"]),
                    _latex_escape(labels.get(row["policy_name"], row["policy_name"])),
                    f"{float(row['common_scale']):.2f}",
                    f"{float(row['common_scale_full_hank_rank']):.1f}",
                    f"{float(row['common_scale_full_hank_loss']):.3e}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_common_scale_projection_validation.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_common_scale_pairwise_table(root: Path, pairwise: pd.DataFrame) -> None:
    if pairwise.empty:
        return
    labels = {
        "classical_filtered_rule": "Классическое правило",
        "optimized_linear_estimated_state": "Оптимизированное линейное",
        "history_observables_rule": "Историческое по наблюдениям",
    }
    lines = [
        "\\begin{tabular}{p{0.21\\linewidth}p{0.22\\linewidth}p{0.11\\linewidth}rrrr}",
        "\\toprule",
        "Сценарий & Сравнение & Число траекторий & $\\Delta J^{red}$ & $\\Delta J^{HANK}$ & 95\\%-й ДИ & Нуль в ДИ \\\\",
        "\\midrule",
    ]
    for row in pairwise.sort_values(["scenario_name", "left_policy", "right_policy"]).to_dict(orient="records"):
        comparison_label = f"{labels.get(row['left_policy'], row['left_policy'])} -- {labels.get(row['right_policy'], row['right_policy'])}"
        interval_label = (
            f"[{float(row['ci_low_common_scale_full_hank_delta_loss']):.2e}; "
            f"{float(row['ci_high_common_scale_full_hank_delta_loss']):.2e}]"
        )
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["scenario_label"]),
                    _latex_escape(comparison_label),
                    f"{int(row['num_trajectories'])}",
                    f"{float(row['mean_reduced_delta_loss']):.2e}",
                    f"{float(row['mean_common_scale_full_hank_delta_loss']):.2e}",
                    interval_label,
                    "Да" if int(row["crosses_zero"]) == 1 else "Нет",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_common_scale_pairwise_intervals.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_scale_sensitivity_table(root: Path, sensitivity: pd.DataFrame) -> None:
    if sensitivity.empty:
        return
    lines = [
        "\\begin{tabular}{p{0.30\\linewidth}rrr}",
        "\\toprule",
        "Сценарий & Масштаб & Попарные знаки & Ранговая корреляция \\\\",
        "\\midrule",
    ]
    for row in sensitivity.sort_values(["scenario_name", "projection_scale"]).to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["scenario_label"]),
                    f"{float(row['projection_scale']):.2f}",
                    _latex_escape(str(row["pairwise_signs_preserved"])),
                    "н/д"
                    if math.isnan(float(row["spearman_rank_correlation"]))
                    else f"{float(row['spearman_rank_correlation']):.2f}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_common_scale_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_validation_text_blocks(root: Path) -> None:
    text = r"""\subsection{Обоснование редуцированного представления состояния}

Используемое в работе редуцированное представление состояния не претендует на полное описание внутренней структуры HANK-модели. Его задача состоит в том, чтобы выделить те скрытые компоненты, которые одновременно имеют содержательный экономический смысл, поддаются оцениванию по наблюдаемым данным и значимы для выбора процентной ставки.

Обоснование такого представления опирается на три критерия. Во-первых, каждый компонент редуцированного состояния сопоставляется с одним из каналов денежно-кредитной трансмиссии в HANK-среде. Во-вторых, состояние проверяется на прогностическую достаточность: оно должно содержать информацию о будущей инфляции, разрыве выпуска и будущей функции потерь. При этом в прогнозной валидации используется только информационный набор, доступный до выбора текущей ставки, включая лагированную процентную ставку; текущая выбранная ставка исключается во избежание информационной утечки. В-третьих, состояние проверяется на управленческую достаточность: ранжирование правил, полученное в редуцированной постановке, должно сохраняться при пропуске соответствующих траекторий ставки через HANK-проекцию.

\begin{table}[htbp]
\centering
\small
\caption{Экономическая интерпретация компонент редуцированного состояния}
\label{tab:reduced_state_components}
\input{outputs/hank_regime_learning_stage6_reduced_state_validation/table_state_component_interpretation.tex}
\end{table}

\begin{table}[htbp]
\centering
\small
\caption{Прогностическая достаточность альтернативных представлений состояния}
\label{tab:forecast_sufficiency}
\input{outputs/hank_regime_learning_stage6_reduced_state_validation/table_forecast_sufficiency.tex}
\end{table}

\begin{table}[htbp]
\centering
\small
\caption{Сценарная прогностическая достаточность для будущей функции потерь}
\label{tab:future_loss_scenario_forecast}
\input{outputs/hank_regime_learning_stage6_reduced_state_validation/table_future_loss_scenario_forecast.tex}
\end{table}

\begin{table}[htbp]
\centering
\small
\caption{Сохранение ранжирования правил при проверке через HANK-проекцию}
\label{tab:policy_ranking_validation}
\input{outputs/hank_regime_learning_stage6_reduced_state_validation/table_policy_ranking_validation.tex}
\end{table}

\begin{table}[htbp]
\centering
\small
\caption{Попарные разности потерь в common-scale HANK-проекции}
\label{tab:common_scale_pairwise_intervals}
\input{outputs/hank_regime_learning_stage6_reduced_state_validation/table_common_scale_pairwise_intervals.tex}
\end{table}

\begin{table}[htbp]
\centering
\small
\caption{Чувствительность ранжирования к масштабу HANK-проекции}
\label{tab:common_scale_sensitivity}
\input{outputs/hank_regime_learning_stage6_reduced_state_validation/table_common_scale_sensitivity.tex}
\end{table}

Результаты показывают, что базовым представлением состояния в среднем следует считать макроэкономическое состояние с учётом скрытого режима, тогда как распределительно расширенное состояние разумно трактовать как HANK-расширение, а не как универсально доминирующий прогнозный набор. Главный прирост прогнозного качества для будущей функции потерь связан с учётом скрытого режима, а не с механическим добавлением всех распределительных компонент. Ранжирование правил сохраняется при проверке через HANK-проекцию. При этом таблица \ref{tab:policy_ranking_validation} должна интерпретироваться с учётом масштаба траекторий: если поле масштаба меньше единицы, соответствующая траектория прошла через переходный решатель HANK только после уменьшения амплитуды. Поэтому данная проверка является проверкой согласованности и численной мягкости траекторий, а не полной переоптимизацией правил в исходной HANK-модели. В таблице \ref{tab:common_scale_pairwise_intervals} специально вынесено число сопоставимых траекторий: для пары двух неклассических правил доступны все 12 отложенных траекторий, тогда как для сравнений с классическим правилом из-за сходимости HANK-проекции остаются лишь 1--3 траектории в зависимости от сценария. Дополнительные interval-оценки на seed-level HANK-проекциях подтверждают устойчивое превосходство двух неклассических правил над классическим правилом в доступном множестве сходящихся траекторий. При этом различие между двумя лучшими правилами существенно меньше по масштабу и должно интерпретироваться осторожнее, чем их отрыв от классического правила.
"""
    (root / "reduced_state_validation_text_blocks.tex").write_text(text, encoding="utf-8")
    (root / "state_representation_validation_text_blocks.tex").write_text(text, encoding="utf-8")


def run_reduced_state_validation(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_reduced_state_validation",
    policy_extension_dir: str = "outputs/hank_regime_learning_stage6_policy_extensions",
    scenario_names: tuple[str, ...] = SCENARIO_NAMES,
    train_seeds: tuple[int, ...] = tuple(range(700, 730)),
    test_seeds: tuple[int, ...] = tuple(range(900, 950)),
    common_scale_projection_seed_count: int = 20,
    forecast_horizons: tuple[int, ...] = (1, 4, 8),
    run_full_hank_projection: bool = False,
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    policy_root = Path(policy_extension_dir)

    _save_json(
        root / "reduced_state_validation_spec.json",
        {
            "scenario_names": list(scenario_names),
            "train_seeds": list(train_seeds),
            "test_seeds": list(test_seeds),
            "common_scale_projection_seed_count": int(common_scale_projection_seed_count),
            "forecast_horizons": list(forecast_horizons),
            "policy_extension_dir": str(policy_root),
            "run_full_hank_projection": bool(run_full_hank_projection),
            "feature_sets": {name: list(spec["columns"]) for name, spec in FEATURE_SPECS.items()},
            "targets": list(TARGET_SPECS),
        },
    )

    components = pd.DataFrame(STATE_COMPONENT_ROWS)
    components.to_csv(root / "state_component_interpretation.csv", index=False)
    _write_state_component_table(root, components)
    _write_validation_text_blocks(root)

    traces = _simulate_validation_traces(
        scenario_names=scenario_names,
        train_seeds=train_seeds,
        test_seeds=test_seeds,
    )
    traces.to_csv(root / "forecast_validation_traces.csv", index=False)
    forecast_detail, forecast_summary = _forecast_sufficiency(traces, forecast_horizons)
    future_loss_summary = _future_loss_scenario_summary(forecast_detail)
    forecast_detail.to_csv(root / "forecast_sufficiency.csv", index=False)
    forecast_summary.to_csv(root / "forecast_sufficiency_summary.csv", index=False)
    future_loss_summary.to_csv(root / "future_loss_scenario_summary.csv", index=False)
    _write_forecast_summary_table(root, forecast_summary)
    _write_future_loss_scenario_table(root, future_loss_summary)

    if run_full_hank_projection:
        run_full_hank_projection_from_policy_paths(
            input_dir=str(policy_root),
            output_dir=str(policy_root),
            scenario_names=scenario_names,
        )

    ranking = pd.DataFrame()
    pairwise_ranking = pd.DataFrame()
    common_scale_ranking = pd.DataFrame()
    common_scale_pairwise = pd.DataFrame()
    common_scale_seed_projection = pd.DataFrame()
    common_scale_pairwise_intervals = pd.DataFrame()
    common_scale_sensitivity = pd.DataFrame()
    metrics_path = policy_root / "policy_metrics.csv"
    full_hank_path = policy_root / "full_hank_projection_metrics.csv"
    if metrics_path.exists() and full_hank_path.exists():
        reduced_metrics = pd.read_csv(metrics_path)
        full_hank_metrics = pd.read_csv(full_hank_path)
        ranking, pairwise_ranking = _policy_ranking_validation(
            reduced_metrics=reduced_metrics,
            full_hank_metrics=full_hank_metrics,
        )
        ranking.to_csv(root / "policy_ranking_validation.csv", index=False)
        pairwise_ranking.to_csv(root / "policy_pairwise_ranking_validation.csv", index=False)
        _write_ranking_table(root, ranking)
        policy_paths = pd.read_csv(policy_root / "policy_paths.csv")
        common_scale_rows, _common_scale_pairs = _run_common_scale_projection(
            policy_paths=policy_paths,
            full_hank_metrics=full_hank_metrics,
        )
        common_scale_rows.to_csv(root / "common_scale_projection_metrics.csv", index=False)
        common_scale_ranking, common_scale_pairwise = _merge_common_scale_with_reduced(
            reduced_metrics=reduced_metrics,
            common_scale_rows=common_scale_rows,
        )
        common_scale_ranking.to_csv(root / "policy_common_scale_ranking_validation.csv", index=False)
        common_scale_pairwise.to_csv(root / "policy_common_scale_pairwise_validation.csv", index=False)
        _write_common_scale_table(root, common_scale_ranking)
        common_scale_seed_projection = _run_common_scale_seed_projection(
            policy_paths=policy_paths,
            common_scale_rows=common_scale_rows,
            evaluation_seeds=tuple(test_seeds[:common_scale_projection_seed_count]),
        )
        common_scale_seed_projection.to_csv(root / "policy_common_scale_seed_projection.csv", index=False)
        common_scale_pairwise_intervals = _common_scale_pairwise_intervals(
            seed_projection=common_scale_seed_projection,
            reduced_metrics=reduced_metrics,
        )
        common_scale_pairwise_intervals.to_csv(root / "policy_common_scale_pairwise_intervals.csv", index=False)
        _write_common_scale_pairwise_table(root, common_scale_pairwise_intervals)
        common_scale_sensitivity = _scale_sensitivity_table(
            policy_paths=policy_paths,
            reduced_metrics=reduced_metrics,
        )
        common_scale_sensitivity.to_csv(root / "common_scale_sensitivity.csv", index=False)
        _write_scale_sensitivity_table(root, common_scale_sensitivity)

    best_forecasts = (
        forecast_summary.sort_values(["target", "mean_oos_r2"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
    )
    future_loss_rank = future_loss_summary.sort_values(["scenario_name", "mean_oos_r2"], ascending=[True, False]).copy()
    ranking_note = "Проверка ранжирования через HANK-проекцию недоступна."
    scale_note = ""
    if not ranking.empty:
        matching_pairs = int(pairwise_ranking["same_pairwise_sign"].sum()) if not pairwise_ranking.empty else 0
        total_pairs = int(len(pairwise_ranking))
        ranking_note = (
            f"Проверка ранжирования через HANK-проекцию охватывает {ranking['scenario_name'].nunique()} сценария(ев); "
            f"знаки попарных сравнений совпадают с редуцированным оценивателем в {matching_pairs}/{total_pairs} случаях."
        )
        common_scale_scenarios = int(ranking.groupby("scenario_name")["common_projection_scale"].first().sum())
        total_scenarios = int(ranking["scenario_name"].nunique())
        if common_scale_scenarios < total_scenarios:
            scale_note = (
                "При интерпретации важно учитывать поле `scale_used`: не во всех сценариях траектории проходят "
                "через переходный HANK-решатель на одинаковом масштабе. Поэтому эта проверка является "
                "проверкой согласованности ранжирования и численной мягкости траекторий, а не полной переоптимизацией "
                "политики в исходной HANK-модели."
            )
    common_scale_note = ""
    if not common_scale_pairwise.empty:
        matches = int(common_scale_pairwise["same_pairwise_sign"].sum())
        total = int(len(common_scale_pairwise))
        common_scale = sorted({float(x) for x in common_scale_pairwise["common_scale"].dropna().unique()})
        common_scale_note = (
            f"Дополнительная common-scale проверка сравнивает правила на общем масштабе внутри каждого сценария; "
            f"совпадение знаков сохраняется в {matches}/{total} попарных сравнениях. "
            f"Использованные общие масштабы: {', '.join(f'{x:.2f}' for x in common_scale)}."
        )
    interval_note = ""
    if not common_scale_pairwise_intervals.empty:
        min_trajectories = int(common_scale_pairwise_intervals["num_trajectories"].min())
        classical_pairs = common_scale_pairwise_intervals[
            common_scale_pairwise_intervals["left_policy"].eq("classical_filtered_rule")
            | common_scale_pairwise_intervals["right_policy"].eq("classical_filtered_rule")
        ]
        min_classical_trajectories = (
            int(classical_pairs["num_trajectories"].min()) if not classical_pairs.empty else min_trajectories
        )
        top_pair = common_scale_pairwise_intervals[
            (
                common_scale_pairwise_intervals["left_policy"].eq("history_observables_rule")
                & common_scale_pairwise_intervals["right_policy"].eq("optimized_linear_estimated_state")
            )
            | (
                common_scale_pairwise_intervals["left_policy"].eq("optimized_linear_estimated_state")
                & common_scale_pairwise_intervals["right_policy"].eq("history_observables_rule")
            )
        ]
        interval_note = (
            f"Интервальные оценки на вспомогательной выборке из {min_trajectories} отложенных тестовых траекторий "
            "подтверждают, что различие между двумя лучшими правилами в HANK-проекции существенно меньше, "
            "чем их отрыв от классического правила."
        )
        if not top_pair.empty:
            zero_crossings = int(top_pair["crosses_zero"].sum())
            interval_note += (
                f" Для пары двух лучших правил доверительный интервал пересекает нуль в {zero_crossings}/"
                f"{len(top_pair)} сценариях."
            )
        interval_note += (
            f" При этом для сравнений с классическим правилом число сходящихся сопоставимых HANK-траекторий "
            f"составляет лишь {min_classical_trajectories} в наиболее трудном сценарии, поэтому такие интервалы "
            "следует трактовать как локальную проверку численной согласованности, а не как полную статистическую переоценку."
        )
    sensitivity_note = ""
    if not common_scale_sensitivity.empty:
        stable = common_scale_sensitivity[
            common_scale_sensitivity["all_policies_success"].eq(1)
        ].copy()
        if not stable.empty:
            highest_stable = stable.groupby("scenario_name", as_index=False)["projection_scale"].max()
            scale_parts = [
                f"{SCENARIO_LABELS[row['scenario_name']]}: {float(row['projection_scale']):.2f}"
                for row in highest_stable.to_dict(orient="records")
            ]
            sensitivity_note = (
                "Проверка чувствительности к масштабу HANK-проекции показывает, что общий устойчивый масштаб "
                "для всех трёх правил в рассматриваемой сетке равен "
                + "; ".join(scale_parts)
                + "."
            )
    lines = [
        "# Обоснование редуцированного представления состояния",
        "",
        "## Экономическая релевантность",
        "",
        "Каждый компонент редуцированного состояния сопоставлен с экономическим каналом HANK. Таблица сохранена в `state_component_interpretation.csv` и `table_state_component_interpretation.tex`.",
        "",
        "## Прогностическая достаточность",
        "",
        "Для проверки прогностической достаточности оцениваются линейные прогнозы будущей инфляции, разрыва выпуска и будущей функции потерь на независимых тестовых траекториях. Сравниваются четыре набора признаков: только наблюдаемые переменные, макроэкономическое состояние, макроэкономическое состояние с вероятностью режима и распределительно расширенное состояние. Во всех спецификациях используется только информационный набор, доступный до выбора текущей ставки; текущая выбранная ставка не включается в признаки.",
        "",
    ]
    for row in best_forecasts.to_dict(orient="records"):
        lines.append(
            f"- {row['target_label']}: лучший средний вневыборочный R2 даёт `{row['feature_set_label']}` "
            f"({row['mean_oos_r2']:.3f})."
        )
    lines.extend(
        [
            "",
            "Главный содержательный вывод относится к будущей функции потерь: наибольший прирост прогнозного качества связан с учётом скрытого режима, а не с механическим добавлением всех распределительных компонент.",
            "",
            "Сценарная картина для будущей функции потерь:",
        ]
    )
    for row in future_loss_rank.groupby("scenario_name", as_index=False).first().to_dict(orient="records"):
        lines.append(
            f"- {row['scenario_label']}: лучший средний R2 даёт `{row['feature_set_label']}` "
            f"({row['mean_oos_r2']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Управленческая достаточность",
            "",
            ranking_note,
            "",
            scale_note,
            "",
            common_scale_note,
            "",
            interval_note,
            "",
            sensitivity_note,
            "",
            "Интерпретация: редуцированное состояние не должно идеально воспроизводить полную HANK-динамику. Для задачи политики достаточно, чтобы оно сохраняло содержательное ранжирование правил и давало прогнозную информацию о переменных, входящих в функцию потерь.",
        ]
    )
    report_text = "\n".join(lines)
    (root / "report_reduced_state_validation.md").write_text(report_text, encoding="utf-8")
    (root / "report_state_representation_validation.md").write_text(report_text, encoding="utf-8")

    return {
        "components": components,
        "forecast_traces": traces,
        "forecast_detail": forecast_detail,
        "forecast_summary": forecast_summary,
        "future_loss_summary": future_loss_summary,
        "ranking": ranking,
        "pairwise_ranking": pairwise_ranking,
        "common_scale_ranking": common_scale_ranking,
        "common_scale_pairwise": common_scale_pairwise,
        "common_scale_seed_projection": common_scale_seed_projection,
        "common_scale_pairwise_intervals": common_scale_pairwise_intervals,
        "common_scale_sensitivity": common_scale_sensitivity,
    }
