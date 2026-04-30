from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.distribution import household_path_levels, path_distribution_statistics
from hank_full_baseline.household_solver import compute_mpc, compute_mpc_path
from hank_full_baseline.irfs import aggregate_paths_frame, group_consumption_irfs, group_paths_frame
from hank_full_baseline.steady_state import solve_steady_state, steady_state_aggregates
from hank_full_baseline.transition import solve_transition

from .config import HANKPartialInfoConfig, OBSERVATION_LABELS, default_partial_info_config
from .evaluation import (
    build_distributional_summary,
    compare_distributional_groups,
    evaluate_filter_metrics,
    evaluate_policy_metrics,
    scenario_metric_frame,
)
from .plots import (
    plot_aggregate_comparison,
    plot_distribution_error_heatmap,
    plot_distributional_comparison,
    plot_filter_rmse,
    plot_hidden_state_estimates,
    plot_policy_losses,
    plot_rate_gap,
    plot_scenario_summary,
)
from .state_space import (
    fit_reduced_state_space,
    scenario_observation_frame,
    scenario_state_frames,
    simulate_information_scenario,
)
from .tables import (
    distributional_consequences_table,
    filter_quality_table,
    information_scenarios_table,
    policy_quality_table,
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported type: {type(value)!r}")


def _save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


def _save_table(table: pd.DataFrame, basepath: Path):
    table.to_csv(basepath.with_suffix(".csv"), index=False)
    basepath.with_suffix(".tex").write_text(table.to_latex(index=False, escape=False))


def _with_scenario_columns(frame: pd.DataFrame, scenario_name: str, scenario_label: str) -> pd.DataFrame:
    enriched = frame.copy()
    enriched.insert(0, "scenario_label", scenario_label)
    enriched.insert(0, "scenario", scenario_name)
    return enriched


def _distribution_and_group_paths(ss, transition, hank_config, scenario_name: str, scenario_label: str):
    mpc = compute_mpc(ss)
    path_levels = household_path_levels(ss, transition)
    mpc_path = compute_mpc_path(path_levels)
    distribution_stats = _with_scenario_columns(
        path_distribution_statistics(ss, path_levels, hank_config, mpc_path),
        scenario_name,
        scenario_label,
    )
    group_paths = group_paths_frame(ss, transition, hank_config, mpc, scenario_name, scenario_label)
    group_consumption = group_consumption_irfs(ss, transition, hank_config, mpc, scenario_name, scenario_label)
    return distribution_stats, group_paths, group_consumption, path_levels


def _policy_rate_error_distribution(
    full_path_levels,
    filtered_path_levels,
    period: int,
) -> pd.DataFrame:
    full_joint = full_path_levels["D"][period].sum(axis=0)
    filtered_joint = filtered_path_levels["D"][period].sum(axis=0)
    b_grid = filtered_path_levels["b_grid"][period]
    a_grid = filtered_path_levels["a_grid"][period]
    rows = []
    for b_index, b_value in enumerate(b_grid):
        for a_index, a_value in enumerate(a_grid):
            rows.append({
                "period": period,
                "b": float(b_value),
                "a": float(a_value),
                "delta_mass": float(filtered_joint[b_index, a_index] - full_joint[b_index, a_index]),
            })
    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    config: HANKPartialInfoConfig,
    filter_metrics: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    distribution_summary: pd.DataFrame,
):
    best_filter = filter_metrics.loc[filter_metrics["mean_state_rmse"].idxmin()]
    hardest_filter = filter_metrics.loc[filter_metrics["mean_state_rmse"].idxmax()]
    worst_policy = policy_metrics.loc[policy_metrics["mean_policy_loss"].idxmax()]
    largest_rate_gap = policy_metrics.loc[policy_metrics["policy_rate_rmse"].idxmax()]
    best_distribution = distribution_summary.loc[distribution_summary["peak_mean_mpc_difference"].idxmin()]
    distribution_augmented = filter_metrics.loc[filter_metrics["scenario"] == "distribution_augmented"].iloc[0]
    full_macro = filter_metrics.loc[filter_metrics["scenario"] == "full_macro"].iloc[0]
    distribution_factor_gain = full_macro["distribution_factor_rmse"] - distribution_augmented["distribution_factor_rmse"]

    lines = [
        "# Этап 3. Классическая денежно-кредитная политика при неполной информации в полной HANK",
        "",
        "## Постановка базового эксперимента",
        "",
        "- Используется низкоразмерное локально-линейное представление полной двухактивной HANK-модели, а не попытка фильтровать полное распределение домохозяйств напрямую.",
        "- Скрытое состояние сочетает агрегатные структурные компоненты и низкоразмерные распределительные компоненты, значимые для политики.",
        "- Состояние: `rstar_gap`, `productivity_gap`, `fiscal_gap`, `inflation_gap`, `output_gap`, `low_liquidity_gap`, `mean_mpc_gap`.",
        "- Скрытость задаётся не только через шоки, но и через распределительные компоненты, влияющие на агрегированную динамику.",
        "",
        "## Информационные сценарии",
        "",
        "- В качестве верхней границы отдельно используется режим полной информации: правило строится по истинному скрытому состоянию, но потери считаются по тем же реализованным траекториям.",
        "- Основные наблюдаемые режимы разделены на базовые макроэкономические сигналы и макроэкономические сигналы с шумной распределительной статистикой.",
        "",
    ]
    for scenario in config.scenario_specs():
        observed = ", ".join(OBSERVATION_LABELS[name] for name in scenario["noisy_observations"])
        distribution_note = (
            "распределительные сигналы отсутствуют"
            if not scenario["includes_distribution_stats"]
            else "распределительные сигналы наблюдаются с шумом"
        )
        lines.append(
            f"- {scenario['label']}: режим `{scenario['information_regime_label']}`, "
            f"шумные наблюдения `{observed}`, {distribution_note}, "
            f"ставка известна точно, масштаб шума `{scenario['noise_scale']}`."
        )

    lines.extend([
        "",
        "## Блок фильтрации",
        "",
        "- Используется классический фильтр Калмана с точно известной ставкой и гауссовым шумом измерения.",
        "- Правило политики получает только наблюдаемые сигналы или их фильтрованную оценку, а функция потерь считается по истинным реализованным значениям инфляции и разрыва выпуска.",
        "- Цель шага состоит в проверке цепочки `скрытое состояние -> шумные наблюдения -> оценённое состояние` в HANK-среде.",
        "",
        "## Качество фильтрации",
        "",
        f"- Лучшее качество фильтрации даёт сценарий `{best_filter['scenario_label']}`: средний RMSE состояния `{best_filter['mean_state_rmse']:.4e}`, RMSE распределительного фактора `{best_filter['distribution_factor_rmse']:.4e}`.",
        f"- Наиболее сложным для фильтрации оказывается сценарий `{hardest_filter['scenario_label']}`: средний RMSE состояния `{hardest_filter['mean_state_rmse']:.4e}`.",
        f"- Добавление шумных распределительных сигналов улучшает восстановление распределительного фактора относительно базового макроэкономического режима на `{distribution_factor_gain:.4e}`.",
        f"- Логарифм правдоподобия в лучшем сценарии: `{best_filter['log_likelihood']:.2f}`.",
        "",
        "## Качество классического правила",
        "",
        f"- Наибольшую среднюю квадратичную потерю среди фильтрованных сценариев даёт `{worst_policy['scenario_label']}`: `{worst_policy['mean_policy_loss']:.4e}`.",
        f"- Наибольшее RMSE ставки относительно полной информации наблюдается в сценарии `{largest_rate_gap['scenario_label']}`: `{largest_rate_gap['policy_rate_rmse']:.4e}`.",
        f"- Для наиболее затратного по потерям сценария среднее абсолютное отклонение ставки равно `{worst_policy['mean_abs_rate_gap']:.4e}`.",
        "- Сравнение ведётся не с оптимальной политикой, а с реализацией того же правила при полной информации.",
        "- Поэтому дополнительную потерю здесь нужно понимать как диагностическую цену неполной наблюдаемости внутри одного и того же класса правил.",
        "",
        "## Распределительные последствия",
        "",
        f"- Наименьшее распределительное искажение даёт сценарий `{best_distribution['scenario_label']}`: пик разницы по средней MPC `{best_distribution['peak_mean_mpc_difference']:.4e}`.",
        f"- Пик отклика потребления нижнего квантиля в этом сценарии: `{best_distribution['peak_consumption_q1_filtered']:.4e}`.",
        "",
        "## Вывод",
        "",
        "Базовая постановка с фильтрацией показывает, что в полной HANK неполная наблюдаемость ухудшает качество денежно-кредитной политики не только через ошибки в восстановлении макросостояния, но и через различия в распределительных траекториях. Это даёт содержательный ориентир для следующих сравнений правил.",
    ])
    (output_dir / "report_stage3_partial_information_hank.md").write_text("\n".join(lines))


def run_pipeline(config: HANKPartialInfoConfig | None = None, output_dir: str | None = None):
    config = default_partial_info_config() if config is None else config
    if output_dir is not None:
        config = HANKPartialInfoConfig(**{**config.to_dict(), "output_dir": output_dir})

    root = Path(config.output_dir)
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    ss = bundle["ss"]
    reduced_model = fit_reduced_state_space(bundle, hank_config, config)

    scenario_state_frames_list = []
    filtered_state_frames_list = []
    observation_frames = []
    filter_metrics_records = []
    filter_state_metric_frames = []
    policy_metric_records = []
    policy_path_frames = []
    distribution_summary_records = []
    aggregate_frames = []
    distribution_frames = []
    group_frames = []
    group_comparison_frames = []

    scenario_runs = [
        simulate_information_scenario(
            reduced_model,
            config,
            scenario,
            seed=config.random_seed,
            phi_pi=hank_config.phi_pi,
            phi_y=hank_config.phi_y,
            rho_i=hank_config.rho_i,
        )
        for scenario in config.scenario_specs()
    ]

    reference_run = scenario_runs[0]
    full_info_inputs = {
        "rstar": reference_run.exogenous_paths["rstar"],
        "Z": reference_run.exogenous_paths["Z"],
        "G": reference_run.exogenous_paths["G"],
        "monetary_policy_shock": reference_run.base_policy_shock,
    }
    full_transition = solve_transition(bundle, full_info_inputs)
    full_aggregate = aggregate_paths_frame(ss, full_transition, "full_information", "Полная информация")
    full_distribution, full_group_paths, _, full_path_levels = _distribution_and_group_paths(
        ss,
        full_transition,
        hank_config,
        "full_information",
        "Полная информация",
    )

    aggregate_frames.append(full_aggregate)
    distribution_frames.append(full_distribution)
    group_frames.append(full_group_paths)
    full_policy_path = pd.DataFrame({
        "scenario": "full_information",
        "scenario_label": "Полная информация",
        "period": full_aggregate["period"],
        "full_information_rate": full_aggregate["i_deviation"],
        "filtered_rate": full_aggregate["i_deviation"],
        "rate_gap": np.zeros(len(full_aggregate), dtype=float),
        "abs_rate_gap": np.zeros(len(full_aggregate), dtype=float),
        "full_loss": (
            np.square(full_aggregate["pi_deviation"])
            + config.lambda_y * np.square(full_aggregate["output_gap_deviation"])
            + config.lambda_i * np.square(np.diff(full_aggregate["i_deviation"], prepend=0.0))
        ),
        "filtered_loss": (
            np.square(full_aggregate["pi_deviation"])
            + config.lambda_y * np.square(full_aggregate["output_gap_deviation"])
            + config.lambda_i * np.square(np.diff(full_aggregate["i_deviation"], prepend=0.0))
        ),
        "excess_loss": np.zeros(len(full_aggregate), dtype=float),
    })
    policy_path_frames.append(full_policy_path)

    heatmap_frame = None
    for simulation, scenario in zip(scenario_runs, config.scenario_specs()):
        true_state_frame, filtered_state_frame = scenario_state_frames(
            simulation,
            reduced_model,
            config.confidence_scale,
        )
        scenario_state_frames_list.append(true_state_frame)
        filtered_state_frames_list.append(filtered_state_frame)
        observation_frames.append(scenario_observation_frame(simulation))

        filter_metrics, filter_state_frame = evaluate_filter_metrics(
            simulation,
            reduced_model.state_names,
            distribution_state_names=("low_liquidity_gap", "mean_mpc_gap"),
            confidence_scale=config.confidence_scale,
        )
        filter_metrics_records.append(filter_metrics)
        filter_state_metric_frames.append(filter_state_frame)

        scenario_inputs = {
            "rstar": simulation.exogenous_paths["rstar"],
            "Z": simulation.exogenous_paths["Z"],
            "G": simulation.exogenous_paths["G"],
            "monetary_policy_shock": simulation.total_policy_shock,
        }
        filtered_transition = solve_transition(bundle, scenario_inputs)
        filtered_aggregate = aggregate_paths_frame(ss, filtered_transition, simulation.scenario_name, simulation.scenario_label)
        filtered_distribution, filtered_group_paths, _, filtered_path_levels = _distribution_and_group_paths(
            ss,
            filtered_transition,
            hank_config,
            simulation.scenario_name,
            simulation.scenario_label,
        )

        aggregate_frames.append(filtered_aggregate)
        distribution_frames.append(filtered_distribution)
        group_frames.append(filtered_group_paths)

        policy_metrics, policy_path = evaluate_policy_metrics(
            scenario_name=simulation.scenario_name,
            scenario_label=simulation.scenario_label,
            full_aggregate_paths=full_aggregate,
            filtered_aggregate_paths=filtered_aggregate,
            full_distribution_stats=full_distribution,
            filtered_distribution_stats=filtered_distribution,
            lambda_y=config.lambda_y,
            lambda_i=config.lambda_i,
        )
        policy_metric_records.append(policy_metrics)
        policy_path_frames.append(policy_path)

        group_comparison = compare_distributional_groups(
            scenario_name=simulation.scenario_name,
            scenario_label=simulation.scenario_label,
            full_group_paths=full_group_paths,
            filtered_group_paths=filtered_group_paths,
        )
        group_comparison_frames.append(group_comparison)
        distribution_summary_records.append(
            build_distributional_summary(
                simulation.scenario_name,
                simulation.scenario_label,
                group_comparison,
                full_distribution,
                filtered_distribution,
            )
        )

        if simulation.scenario_name == "macro_core":
            peak_period = int(np.argmax(np.abs(policy_path["rate_gap"].to_numpy(dtype=float))))
            heatmap_frame = _policy_rate_error_distribution(full_path_levels, filtered_path_levels, peak_period)

    true_state_paths = pd.concat(scenario_state_frames_list, ignore_index=True)
    filtered_state_paths = pd.concat(filtered_state_frames_list, ignore_index=True)
    observations_frame = pd.concat(observation_frames, ignore_index=True)
    filter_metrics_frame = scenario_metric_frame(filter_metrics_records)
    filter_state_metrics_frame = pd.concat(filter_state_metric_frames, ignore_index=True)
    policy_metrics_frame = scenario_metric_frame(policy_metric_records)
    policy_paths_frame = pd.concat(policy_path_frames, ignore_index=True)
    distribution_summary_frame = scenario_metric_frame(distribution_summary_records)
    aggregate_paths = pd.concat(aggregate_frames, ignore_index=True)
    distribution_stats = pd.concat(distribution_frames, ignore_index=True)
    group_paths = pd.concat(group_frames, ignore_index=True)
    group_comparison = pd.concat(group_comparison_frames, ignore_index=True)

    _save_json(root / "model_spec.json", config.model_spec_payload(bundle["model"].name))
    _save_json(root / "filter_spec.json", config.filter_spec_payload())
    _save_json(root / "policy_spec.json", config.policy_spec_payload(hank_config.phi_pi, hank_config.phi_y, hank_config.rho_i))
    _save_json(root / "scenario_spec.json", config.scenario_specs())
    _save_json(root / "information_regime_spec.json", config.article_information_regimes_payload())
    _save_json(root / "steady_state_aggregates.json", steady_state_aggregates(ss))
    _save_json(root / "reduced_state_space.json", {
        "state_names": list(reduced_model.state_names),
        "observation_names": list(reduced_model.observation_names),
        "transition_matrix": reduced_model.transition_matrix,
        "control_loadings": reduced_model.control_loadings,
        "process_noise_cov": reduced_model.process_noise_cov,
        "observation_matrix": reduced_model.observation_matrix,
        "observation_fit_rmse": reduced_model.observation_fit_rmse,
        "training_summary": reduced_model.training_summary,
        "steady_state_statistics": reduced_model.steady_state_statistics,
    })

    true_state_paths.to_csv(root / "true_state_paths.csv", index=False)
    filtered_state_paths.to_csv(root / "filtered_state_paths.csv", index=False)
    observations_frame.to_csv(root / "observations.csv", index=False)
    aggregate_paths.to_csv(root / "aggregate_paths.csv", index=False)
    group_paths.to_csv(root / "group_paths.csv", index=False)
    distribution_stats.to_csv(root / "distribution_stats.csv", index=False)
    policy_metrics_frame.to_csv(root / "policy_metrics.csv", index=False)
    filter_metrics_frame.to_csv(root / "filter_metrics.csv", index=False)
    policy_paths_frame.to_csv(root / "policy_path_diagnostics.csv", index=False)
    filter_state_metrics_frame.to_csv(root / "filter_state_metrics.csv", index=False)
    group_comparison.to_csv(root / "group_comparison.csv", index=False)
    distribution_summary_frame.to_csv(root / "distributional_summary.csv", index=False)
    if heatmap_frame is not None:
        heatmap_frame.to_csv(root / "distribution_error_heatmap.csv", index=False)

    info_table = information_scenarios_table(config.scenario_table_payload())
    filter_table = filter_quality_table(filter_metrics_frame)
    policy_table = policy_quality_table(policy_metrics_frame)
    distribution_table = distributional_consequences_table(distribution_summary_frame)

    _save_table(info_table, tables_dir / "table_01_information_scenarios")
    _save_table(filter_table, tables_dir / "table_02_filter_quality")
    _save_table(policy_table, tables_dir / "table_03_policy_quality")
    _save_table(distribution_table, tables_dir / "table_04_distributional_consequences")

    plot_aggregate_comparison(aggregate_paths, figures_dir)
    plot_hidden_state_estimates(filtered_state_paths, figures_dir)
    plot_filter_rmse(filter_state_metrics_frame, figures_dir)
    plot_rate_gap(policy_paths_frame, figures_dir)
    plot_policy_losses(policy_paths_frame, figures_dir)
    plot_distributional_comparison(distribution_stats, group_comparison, figures_dir)
    if heatmap_frame is not None:
        plot_distribution_error_heatmap(heatmap_frame, figures_dir)
    plot_scenario_summary(filter_metrics_frame, policy_metrics_frame, figures_dir)

    _write_report(root, config, filter_metrics_frame, policy_metrics_frame, distribution_summary_frame)

    diagnostics = {
        "reduced_model_training": reduced_model.training_summary,
        "steady_state_statistics": reduced_model.steady_state_statistics,
        "filter_metrics": filter_metrics_frame.to_dict(orient="records"),
        "policy_metrics": policy_metrics_frame.to_dict(orient="records"),
    }
    _save_json(root / "diagnostics_summary.json", diagnostics)

    return {
        "reduced_model": reduced_model,
        "true_state_paths": true_state_paths,
        "filtered_state_paths": filtered_state_paths,
        "observations": observations_frame,
        "aggregate_paths": aggregate_paths,
        "distribution_stats": distribution_stats,
        "group_paths": group_paths,
        "policy_metrics": policy_metrics_frame,
        "filter_metrics": filter_metrics_frame,
    }
