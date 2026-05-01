from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import HANKCalibration, default_calibration
from .distribution import (
    build_group_masks,
    distribution_snapshots,
    household_path_levels,
    household_levels,
    joint_distribution_shift,
    path_distribution_statistics,
    stationary_distribution,
)
from .experiments import monetary_policy_experiment, policy_scenarios
from .household_solver import (
    aggregate_consistency,
    compute_mpc,
    compute_mpc_path,
    compute_transfer_mpc,
    household_budget_residual,
)
from .irfs import (
    aggregate_income_channels_frame,
    aggregate_irf_frame,
    aggregate_paths_frame,
    channel_decomposition_frame,
    group_consumption_irfs,
    group_contribution_frame,
    group_income_irfs,
    group_paths_frame,
    steady_state_group_statistics,
)
from .plots import (
    plot_aggregate_irfs,
    plot_channels,
    plot_distribution_dynamics,
    plot_group_irfs,
    plot_group_contributions,
    plot_household_robustness,
    plot_low_liquidity_shares,
    plot_mpc_measure_comparison,
    plot_policy_functions,
    plot_policy_scenario_comparison,
    plot_reference_alignment,
    plot_stationary_distributions,
    plot_wealthy_htm_sensitivity,
    plot_wealth_quantiles,
)
from .robustness import (
    group_profile_frame,
    household_robustness_frames,
    mpc_moments_frame,
    reference_alignment_frames,
    transfer_mpc_moments_frame,
    wealthy_htm_sensitivity_frame,
)
from .sjacobian import solve_sequence_space_jacobian
from .steady_state import solve_steady_state, steady_state_aggregates
from .tables import (
    calibration_table,
    channel_summary_table,
    group_differences_table,
    policy_rule_table,
    shock_effects_table,
    steady_state_moments_table,
)
from .transition import channel_decomposition, solve_transition


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


def _checks(bundle, transition, config, mpc):
    ss = bundle["ss"]
    D = stationary_distribution(ss)
    budget_residual = household_budget_residual(ss)
    aggregates = aggregate_consistency(ss)

    grid_alt = solve_steady_state(replace(config, nB=config.nB + 2, nA=config.nA + 4))["ss"]

    return {
        "distribution_mass_sum": float(D.sum()),
        "distribution_min_mass": float(D.min()),
        "budget_residual_max_abs": float(np.max(np.abs(budget_residual))),
        "asset_consistency_abs": float(abs(aggregates["A_from_distribution"] - ss["A"])),
        "liquid_consistency_abs": float(abs(aggregates["B_from_distribution"] - ss["B"])),
        "consumption_consistency_abs": float(abs(aggregates["C_from_distribution"] - ss["C"])),
        "asset_market_residual_abs": float(abs(ss["asset_mkt"])),
        "goods_market_residual_abs": float(abs(ss["goods_mkt"])),
        "irf_is_finite": bool(np.isfinite(transition["C"]).all() and np.isfinite(transition["Y"]).all()),
        "max_abs_irf_output_pct": float(np.max(np.abs(100.0 * transition["Y"] / ss["Y"]))),
        "mean_mpc": float(np.sum(D * mpc)),
        "grid_sensitivity_C_abs": float(abs(grid_alt["C"] - ss["C"])),
        "grid_sensitivity_Y_abs": float(abs(grid_alt["Y"] - ss["Y"])),
        "grid_sensitivity_A_abs": float(abs(grid_alt["A"] - ss["A"])),
    }


def _append_scenario_column(df, scenario_name, scenario_label):
    df = df.copy()
    df.insert(0, "scenario_label", scenario_label)
    df.insert(0, "scenario", scenario_name)
    return df


def _write_report(
    output_dir: Path,
    baseline_config: HANKCalibration,
    shock_effects,
    group_table,
    channel_table,
    scenario_summary,
    mpc_validation,
    transfer_mpc_validation,
    robustness_summary,
    wealthy_htm_sensitivity,
    group_thresholds,
    reference_summary,
):
    top_group = group_table.iloc[np.argmin(group_table["минимум отклика потребления"].to_numpy())] if not group_table.empty else None
    top_channel = channel_table.iloc[np.argmax(np.abs(channel_table["вклад в пик отклика потребления"].to_numpy()))] if not channel_table.empty else None
    high_mpc_share = mpc_validation.loc[mpc_validation["момент"] == "Доля MPC > 0.2", "значение"].iloc[0]
    mean_mpc = mpc_validation.loc[mpc_validation["момент"] == "Средняя MPC", "значение"].iloc[0]
    median_mpc = mpc_validation.loc[mpc_validation["момент"] == "Медианная MPC", "значение"].iloc[0]
    transfer_high_mpc_share = transfer_mpc_validation.loc[transfer_mpc_validation["момент"] == "Доля MPC > 0.2", "значение"].iloc[0]
    transfer_mean_mpc = transfer_mpc_validation.loc[transfer_mpc_validation["момент"] == "Средняя MPC", "значение"].iloc[0]
    transfer_median_mpc = transfer_mpc_validation.loc[transfer_mpc_validation["момент"] == "Медианная MPC", "значение"].iloc[0]
    most_sensitive_robustness = robustness_summary.iloc[np.argmax(np.abs(robustness_summary["peak_consumption_response"].to_numpy()))]
    sensitivity_span = wealthy_htm_sensitivity["share_wealthy_htm"].agg(["min", "max"])

    lines = [
        "# Этап 2. Классическая денежно-кредитная политика в полной HANK",
        "",
        "## Спецификация правила ставки",
        "",
        f"- Используемое правило: `i_t = rho_i i_(t-1) + (1-rho_i)(r* + phi_pi*pi_t + phi_y*y_gap_t) + eps_t^i`.",
        f"- Базовая калибровка: `phi_pi = {baseline_config.phi_pi}`, `phi_y = {baseline_config.phi_y}`, `rho_i = {baseline_config.rho_i}`.",
        f"- Размер policy shock: `{baseline_config.mp_shock_size}`.",
        f"- Персистентность policy shock: `{baseline_config.mp_shock_persistence}`.",
        f"- В текущем baseline monetary shock подаётся как неожиданный импульс в период `{baseline_config.shock_period}`.",
        f"- Численная сетка специально уплотнена у constrained region: `nB = {baseline_config.nB}`, `nA = {baseline_config.nA}`, `nK = {baseline_config.nK}`, с дополнительной концентрацией узлов для `b <= {baseline_config.b_dense_region_max}`, `a <= {baseline_config.a_dense_region_max}`, `k <= {baseline_config.k_dense_region_max}`.",
        "",
        "## Валидация HANK-неоднородности",
        "",
        f"- Средняя MPC в базовой калибровке: `{mean_mpc:.4f}`, медианная MPC: `{median_mpc:.4f}`.",
        f"- Доля домохозяйств с MPC выше 0.2: `{high_mpc_share:.4f}`.",
        "- Текущая MPC-валидация построена как локальная slope-based мера по liquid asset grid, а не как полноценная MPC из транзиторного доходного или трансфертного шока.",
        "- Поэтому слабый MPC-tail здесь следует читать как ограничение текущей reduced-grid baseline и способа измерения, а не как окончательный количественный вывод о cash-flow MPC в полной HANK.",
        f"- Дополнительная validation по одноразовому равномерному трансферту размера `{baseline_config.mpc_transfer_shock_size}` даёт среднюю MPC `{transfer_mean_mpc:.4f}`, медианную MPC `{transfer_median_mpc:.4f}` и долю `MPC > 0.2` `{transfer_high_mpc_share:.4f}`.",
        "- Именно эта transfer-based мера ближе к эмпирической WHtM-интерпретации, потому что измеряет отклик потребления на транзиторное изменение располагаемого дохода, а не только локальный наклон policy function по liquid asset.",
        "- Текущая калибровка специально усилена по ликвидностным фрикциям и доходному риску относительно раннего baseline, но всё ещё остаётся умеренной по сравнению с наиболее агрессивными HANK-калибровками из литературы.",
        f"- Операционное определение WHtM: `b <= {group_thresholds['low_liquid_cutoff']:.2f}` и `a >= {group_thresholds['wealthy_htm_a_cutoff']:.2f}`.",
        f"- В sensitivity по порогам доля WHtM меняется в диапазоне `[{sensitivity_span['min']:.4f}, {sensitivity_span['max']:.4f}]`.",
        "- Поэтому WHtM в baseline следует интерпретировать как операционную классификацию, а не как полностью структурно-инвариантный объект.",
        "",
        "## Связь с sequence-jacobian tutorial",
        "",
    ]
    reference_project = reference_summary[reference_summary["specification"] == "project_baseline"].iloc[0]
    reference_tutorial = reference_summary[reference_summary["specification"] == "sequence_jacobian_tutorial"].iloc[0]
    lines.extend([
        f"- Для reference alignment используется `two_asset.ipynb` из sequence-jacobian: tutorial calibration оценивается на той же reduced-grid архитектуре, что и проектный baseline.",
        f"- Средняя MPC в project baseline `{reference_project['mean_mpc']:.4f}` против `{reference_tutorial['mean_mpc']:.4f}` в tutorial calibration.",
        f"- Пик отклика потребления: `{reference_project['peak_consumption_response']:.4f}` против `{reference_tutorial['peak_consumption_response']:.4f}`.",
        "",
        "## Основные количественные результаты",
        "",
    ])
    for _, row in shock_effects.iterrows():
        lines.append(
            f"- {row['переменная']}: impact `{row['отклик при ударе']:.4f}`, "
            f"минимум `{row['минимум отклика']:.4f}` в период `{int(row['период минимума'])}`, "
            f"максимум `{row['максимум отклика']:.4f}` в период `{int(row['период максимума'])}`."
        )

    lines.extend([
        "",
        "## Наиболее чувствительные группы",
        "",
        "- Группы `низкая ликвидность`, `WHtM` и `высоколиквидные` в response-сравнениях остаются аналитическими и пересекающимися.",
        "- Вклад в совокупное потребление теперь строится отдельно по взаимоисключающему разбиению: `низкая ликвидность без WHtM`, `WHtM`, `высоколиквидные`, `остальные домохозяйства`.",
    ])
    if top_group is not None:
        lines.append(
            f"- Наиболее глубокий отрицательный отклик потребления в сводной таблице даёт группа `{top_group['группа']}`: "
            f"минимум `{top_group['минимум отклика потребления']:.4f}`, "
            f"интегральный отклик `{top_group['интегральный отклик потребления']:.4f}`."
        )
    for _, row in group_table.iterrows():
        lines.append(
            f"- {row['группа']}: impact `{row['отклик при ударе']:.4f}`, "
            f"минимум `{row['минимум отклика потребления']:.4f}`, максимум `{row['максимум отклика потребления']:.4f}`, "
            f"пик дохода `{row['пик отклика дохода']:.4f}`, "
            f"изменение liquid wealth к периоду минимума `{row['изменение ликвидного богатства к периоду минимума']:.4f}`."
        )

    lines.extend([
        "",
        "## Основные каналы",
        "",
        "- Разложение по каналам ниже является accounting decomposition household-блока, а не полным общерыночным welfare decomposition.",
        "- Остаточный redistribution/liquidity компонент в текущей калибровке мал, поэтому сильные выводы о доминировании перераспределительного канала здесь делать нельзя.",
    ])
    if top_channel is not None:
        lines.append(
            f"- Наибольший вклад в пик отклика потребления даёт канал `{top_channel['канал']}`: "
            f"`{top_channel['вклад в пик отклика потребления']:.4f}`."
        )
    for _, row in channel_table.iterrows():
        lines.append(
            f"- {row['канал']}: вклад в пик `{row['вклад в пик отклика потребления']:.4f}`, "
            f"интегральный вклад `{row['вклад в интегральный отклик']:.4f}`."
        )

    lines.extend([
        "",
        "## Проверки по блоку домохозяйств",
        "",
        f"- Наиболее сильный отклик потребления среди дополнительных сценариев даёт `{most_sensitive_robustness['scenario_label']}`: `{most_sensitive_robustness['peak_consumption_response']:.4f}`.",
        f"- В этом сценарии средняя MPC равна `{most_sensitive_robustness['mean_mpc']:.4f}`, а доля WHtM `{most_sensitive_robustness['share_wealthy_htm']:.4f}`.",
        "",
        "## Сценарное сравнение",
        "",
        "Ниже приведены абсолютные пиковые отклонения по базовым агрегатам в разных конфигурациях правила ставки.",
        "",
    ])
    for entry in scenario_summary:
        lines.append(
            f"- {entry['scenario_label']}: пик выпуска `{entry['peak_output']:.4f}`, "
            f"пик инфляции `{entry['peak_inflation']:.4f}`, "
            f"пик потребления `{entry['peak_consumption']:.4f}`."
        )

    lines.extend([
        "",
        "## Вывод",
        "",
        "Даже при стандартном правиле ставки в полной HANK денежно-кредитная политика создаёт не только агрегатные, но и выраженные распределительные эффекты. Проверочный блок показывает, что эти выводы зависят не только от средних агрегатов, но и от калибровки ликвидностных фрикций, доходного риска и операционного определения состоятельных домохозяйств с низкой ликвидностью.",
    ])

    (output_dir / "report_hank_core.md").write_text("\n".join(lines))


def run_pipeline(config: HANKCalibration | None = None, output_dir: str | None = None):
    config = default_calibration() if config is None else config
    if output_dir is not None:
        config = replace(config, output_dir=output_dir)

    root = Path(config.output_dir)
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    data_dir = root / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    scenarios = policy_scenarios(config)

    aggregate_paths_all = []
    aggregate_irf_all = []
    distribution_paths_all = []
    group_paths_all = []
    group_consumption_all = []
    group_income_all = []
    aggregate_income_channels_all = []
    group_contributions_all = []
    channel_decomposition_all = []
    diagnostics = {}
    scenario_summary = []

    baseline_bundle = None
    baseline_ss = None
    baseline_mpc = None
    baseline_transition = None
    baseline_full_path_levels = None
    baseline_group_stats = None
    baseline_experiment = None
    baseline_transfer_mpc = None
    baseline_transfer_mpc_diagnostics = None
    jacobian_df = None
    liquid_snapshots = None
    illiquid_snapshots = None
    joint_shift = None
    baseline_group_thresholds = None

    for scenario in scenarios:
        scenario_name = scenario["name"]
        scenario_label = scenario["label"]
        scenario_config = scenario["config"]

        bundle = solve_steady_state(scenario_config)
        ss = bundle["ss"]
        mpc = compute_mpc(ss)
        experiment = monetary_policy_experiment(scenario_config)
        transition = solve_transition(bundle, experiment["inputs"])
        full_path_levels = household_path_levels(ss, transition)
        mpc_path = compute_mpc_path(full_path_levels)

        aggregate_paths = aggregate_paths_frame(ss, transition, scenario_name, scenario_label)
        aggregate_irf = aggregate_irf_frame(ss, transition, scenario_name, scenario_label)
        distribution_paths = _append_scenario_column(
            path_distribution_statistics(ss, full_path_levels, scenario_config, mpc_path),
            scenario_name,
            scenario_label,
        )
        group_paths = group_paths_frame(ss, transition, scenario_config, mpc, scenario_name, scenario_label)
        group_consumption = group_consumption_irfs(ss, transition, scenario_config, mpc, scenario_name, scenario_label)
        group_income = group_income_irfs(ss, transition, scenario_config, mpc, scenario_name, scenario_label)
        aggregate_income_channels = aggregate_income_channels_frame(ss, transition, scenario_name, scenario_label)
        channels = channel_decomposition(bundle, transition)
        channel_df = channel_decomposition_frame(ss, transition, channels, scenario_name, scenario_label)
        group_contributions = group_contribution_frame(ss, transition, scenario_config, mpc, scenario_name, scenario_label)

        aggregate_paths_all.append(aggregate_paths)
        aggregate_irf_all.append(aggregate_irf)
        distribution_paths_all.append(distribution_paths)
        group_paths_all.append(group_paths)
        group_consumption_all.append(group_consumption)
        group_income_all.append(group_income)
        aggregate_income_channels_all.append(aggregate_income_channels)
        group_contributions_all.append(group_contributions)
        channel_decomposition_all.append(channel_df)

        diagnostics[scenario_name] = _checks(bundle, transition, scenario_config, mpc)

        subset = aggregate_irf.sort_values("period")
        scenario_summary.append({
            "scenario": scenario_name,
            "scenario_label": scenario_label,
            "peak_output": float(subset[subset["variable"] == "Y"]["value"].abs().max()),
            "peak_inflation": float(subset[subset["variable"] == "pi"]["value"].abs().max()),
            "peak_consumption": float(subset[subset["variable"] == "C"]["value"].abs().max()),
        })

        if scenario_name == "baseline":
            baseline_bundle = bundle
            baseline_ss = ss
            baseline_mpc = mpc
            baseline_transition = transition
            baseline_full_path_levels = full_path_levels
            baseline_group_stats = steady_state_group_statistics(ss, mpc, scenario_config)
            baseline_group_thresholds = build_group_masks(ss, scenario_config, mpc=mpc)["thresholds"]
            baseline_experiment = experiment
            jacobian_df = solve_sequence_space_jacobian(bundle, scenario_config.shock_T)
            liquid_snapshots, illiquid_snapshots = distribution_snapshots(ss, full_path_levels, experiment["horizons"])
            liquid_snapshots = _append_scenario_column(liquid_snapshots, scenario_name, scenario_label)
            illiquid_snapshots = _append_scenario_column(illiquid_snapshots, scenario_name, scenario_label)
            joint_shift = _append_scenario_column(
                joint_distribution_shift(
                    ss,
                    full_path_levels,
                    experiment["horizons"][2] if len(experiment["horizons"]) > 2 else experiment["horizons"][-1],
                ),
                scenario_name,
                scenario_label,
            )

    aggregate_paths_df = pd.concat(aggregate_paths_all, ignore_index=True)
    aggregate_irf_df = pd.concat(aggregate_irf_all, ignore_index=True)
    distribution_paths_df = pd.concat(distribution_paths_all, ignore_index=True)
    group_paths_df = pd.concat(group_paths_all, ignore_index=True)
    group_consumption_df = pd.concat(group_consumption_all, ignore_index=True)
    group_income_df = pd.concat(group_income_all, ignore_index=True)
    aggregate_income_channels_df = pd.concat(aggregate_income_channels_all, ignore_index=True)
    group_contributions_df = pd.concat(group_contributions_all, ignore_index=True)
    channel_decomposition_df = pd.concat(channel_decomposition_all, ignore_index=True)

    calibration_df = calibration_table(config, {"beta": baseline_ss["beta"], "chi1": baseline_ss["chi1"]})
    policy_df = policy_rule_table(scenarios)
    moments_df = steady_state_moments_table(baseline_ss, baseline_mpc, config)
    mpc_validation_df = mpc_moments_frame(baseline_ss, baseline_mpc, config)
    baseline_transfer_mpc_diagnostics = compute_transfer_mpc(
        baseline_bundle,
        config.mpc_transfer_shock_size,
        horizon=config.mpc_transfer_horizon,
    )
    baseline_transfer_mpc = baseline_transfer_mpc_diagnostics["mpc"]
    transfer_mpc_validation_df = transfer_mpc_moments_frame(baseline_ss, baseline_transfer_mpc, config)
    group_profiles_df = group_profile_frame(baseline_ss, baseline_mpc, config)
    wealthy_htm_sensitivity_df = wealthy_htm_sensitivity_frame(baseline_ss, baseline_transition, baseline_mpc, config)
    reference_parameter_df, reference_summary_df = reference_alignment_frames(
        config,
        precomputed_baseline={
            "bundle": baseline_bundle,
            "ss": baseline_ss,
            "mpc": baseline_mpc,
            "transition": baseline_transition,
        },
    )
    household_robustness_df, household_robustness_groups_df = household_robustness_frames(
        config,
        precomputed_baseline={
            "bundle": baseline_bundle,
            "ss": baseline_ss,
            "mpc": baseline_mpc,
            "transition": baseline_transition,
        },
    )
    shock_df = shock_effects_table(aggregate_irf_df, scenario_name="baseline")
    group_table_df = group_differences_table(baseline_group_stats, group_paths_df, scenario_name="baseline")
    channel_table_df = channel_summary_table(channel_decomposition_df, scenario_name="baseline")

    plot_stationary_distributions(baseline_ss, baseline_mpc, config, figures_dir)
    plot_mpc_measure_comparison(baseline_ss, baseline_mpc, baseline_transfer_mpc, config, figures_dir)
    plot_policy_functions(baseline_ss, figures_dir)
    plot_aggregate_irfs(aggregate_irf_df, figures_dir, scenario_name="baseline")
    plot_wealth_quantiles(distribution_paths_df, figures_dir, scenario_name="baseline")
    plot_low_liquidity_shares(distribution_paths_df, figures_dir, scenario_name="baseline")
    plot_group_irfs(group_consumption_df, group_income_df, figures_dir, scenario_name="baseline")
    plot_distribution_dynamics(liquid_snapshots, illiquid_snapshots, joint_shift, figures_dir, scenario_name="baseline")
    plot_channels(channel_decomposition_df, aggregate_income_channels_df, figures_dir, scenario_name="baseline")
    plot_group_contributions(group_contributions_df, figures_dir, scenario_name="baseline")
    plot_wealthy_htm_sensitivity(wealthy_htm_sensitivity_df, figures_dir)
    plot_household_robustness(household_robustness_df, figures_dir)
    plot_policy_scenario_comparison(aggregate_irf_df, figures_dir)
    plot_reference_alignment(reference_summary_df, figures_dir)

    _save_json(root / "model_spec.json", {
        "model_name": "Two-Asset HANK",
        "stage": "HANK-ядро для новой работы о ценности распределительной информации",
        "solution_method": "стационарное состояние + якобиан последовательностей + нелинейный переход",
        "household_block": "два актива, индивидуальный доходный риск, издержки ребалансировки",
        "fiscal_mode": config.fiscal_mode,
        "policy_rule": "i_t = rho_i i_{t-1} + (1-rho_i)(rstar + phi_pi pi_t + phi_y y_gap_t) + eps_t^i",
        "output_measure_in_rule": "output_gap",
    })
    _save_json(root / "policy_config.json", config.rule_spec())
    _save_json(root / "scenario_config.json", [
        {
            "name": scenario["name"],
            "label": scenario["label"],
            "description": scenario["description"],
            "rule": scenario["config"].rule_spec(),
        }
        for scenario in scenarios
    ])
    _save_json(root / "calibration.json", asdict(config))
    _save_json(root / "steady_state_aggregates.json", steady_state_aggregates(baseline_ss))
    diagnostics["mpc_validation"] = mpc_validation_df.to_dict(orient="records")
    diagnostics["transfer_mpc_validation"] = transfer_mpc_validation_df.to_dict(orient="records")
    diagnostics["transfer_mpc_spec"] = {
        "transfer_shock_size": baseline_transfer_mpc_diagnostics["shock_size"],
        "transfer_shock_horizon": baseline_transfer_mpc_diagnostics["horizon"],
        "aggregate_mpc_impact": baseline_transfer_mpc_diagnostics["aggregate_mpc"],
    }
    diagnostics["household_robustness"] = household_robustness_df.to_dict(orient="records")
    diagnostics["sequence_jacobian_reference"] = reference_summary_df.to_dict(orient="records")
    _save_json(root / "diagnostics_summary.json", diagnostics)
    _save_json(root / "experiment_spec.json", baseline_experiment)
    _save_json(root / "group_definition_spec.json", {
        "operational_definition": "wealthy_htm := low liquid wealth below threshold and illiquid wealth above threshold quantile",
        "baseline_thresholds": baseline_group_thresholds,
        "low_liquidity_threshold": config.low_liquidity_threshold,
        "wealthy_htm_a_quantile": config.wealthy_htm_a_quantile,
    })
    _save_json(root / "reference_spec.json", {
        "reference_source": "sequence-jacobian Tutorial 4 / two_asset.ipynb",
        "reference_notebook_path": "/Users/polinazosimova/Downloads/two_asset.ipynb",
        "comparison_design": "tutorial economic calibration evaluated on the project's reduced grid",
    })
    _save_json(root / "mpc_measure_spec.json", {
        "local_mpc_measure": "finite-difference slope of the household consumption policy with respect to the liquid asset grid",
        "cashflow_mpc_measure": "impact response of household consumption to a one-period uniform transfer shock in disposable income",
        "transfer_shock_size": config.mpc_transfer_shock_size,
        "transfer_shock_horizon": config.mpc_transfer_horizon,
    })

    np.savez_compressed(
        data_dir / "steady_state_distribution.npz",
        D=baseline_ss.internals["hh"]["D"],
        Dbeg=baseline_ss.internals["hh"]["Dbeg"],
        b_grid=baseline_ss.internals["hh"]["b_grid"],
        a_grid=baseline_ss.internals["hh"]["a_grid"],
        z_grid=baseline_ss.internals["hh"]["z_grid"],
    )
    np.savez_compressed(
        data_dir / "steady_state_policies.npz",
        a=baseline_ss.internals["hh"]["a"],
        b=baseline_ss.internals["hh"]["b"],
        c=baseline_ss.internals["hh"]["c"],
        Va=baseline_ss.internals["hh"]["Va"],
        Vb=baseline_ss.internals["hh"]["Vb"],
        chi=baseline_ss.internals["hh"]["chi"],
        mpc=baseline_mpc,
        transfer_mpc=baseline_transfer_mpc,
    )
    np.savez_compressed(
        data_dir / "transition_household_paths.npz",
        D=baseline_full_path_levels["D"],
        a=baseline_full_path_levels["a"],
        b=baseline_full_path_levels["b"],
        c=baseline_full_path_levels["c"],
        z_grid=baseline_full_path_levels["z_grid"],
    )

    aggregate_paths_df.to_csv(root / "aggregate_paths.csv", index=False)
    distribution_paths_df.to_csv(root / "distribution_paths.csv", index=False)
    group_paths_df.to_csv(root / "group_paths.csv", index=False)
    channel_decomposition_df.to_csv(root / "channel_decomposition.csv", index=False)
    aggregate_irf_df.to_csv(root / "aggregate_irfs.csv", index=False)
    group_consumption_df.to_csv(root / "group_consumption_irfs.csv", index=False)
    group_income_df.to_csv(root / "group_income_irfs.csv", index=False)
    aggregate_income_channels_df.to_csv(root / "aggregate_income_channels.csv", index=False)
    group_contributions_df.to_csv(root / "group_contributions.csv", index=False)
    liquid_snapshots.to_csv(root / "liquid_distribution_snapshots.csv", index=False)
    illiquid_snapshots.to_csv(root / "illiquid_distribution_snapshots.csv", index=False)
    joint_shift.to_csv(root / "joint_distribution_shift.csv", index=False)
    jacobian_df.to_csv(root / "jacobian_summary.csv", index=False)
    baseline_group_stats.to_csv(root / "group_statistics.csv", index=False)
    mpc_validation_df.to_csv(root / "mpc_validation.csv", index=False)
    transfer_mpc_validation_df.to_csv(root / "transfer_mpc_validation.csv", index=False)
    group_profiles_df.to_csv(root / "group_profiles.csv", index=False)
    wealthy_htm_sensitivity_df.to_csv(root / "wealthy_htm_sensitivity.csv", index=False)
    reference_parameter_df.to_csv(root / "sequence_jacobian_reference_parameters.csv", index=False)
    reference_summary_df.to_csv(root / "sequence_jacobian_reference_summary.csv", index=False)
    household_robustness_df.to_csv(root / "household_robustness_summary.csv", index=False)
    household_robustness_groups_df.to_csv(root / "household_robustness_group_peaks.csv", index=False)

    _save_table(calibration_df, tables_dir / "table_00_calibration")
    _save_table(moments_df, tables_dir / "table_00_steady_state_moments")
    _save_table(mpc_validation_df, tables_dir / "table_00b_mpc_validation")
    _save_table(transfer_mpc_validation_df, tables_dir / "table_00c_transfer_mpc_validation")
    _save_table(policy_df, tables_dir / "table_01_policy_rule")
    _save_table(shock_df, tables_dir / "table_02_aggregate_peak_responses")
    _save_table(group_table_df, tables_dir / "table_03_group_effects")
    _save_table(group_profiles_df, tables_dir / "table_03b_group_profiles")
    _save_table(channel_table_df, tables_dir / "table_04_channel_decomposition")
    _save_table(household_robustness_df, tables_dir / "table_05_household_robustness")
    _save_table(wealthy_htm_sensitivity_df, tables_dir / "table_06_wealthy_htm_sensitivity")
    _save_table(reference_parameter_df, tables_dir / "table_07_reference_parameters")
    _save_table(reference_summary_df, tables_dir / "table_07b_reference_summary")

    _write_report(
        root,
        config,
        shock_df,
        group_table_df,
        channel_table_df,
        scenario_summary,
        mpc_validation_df,
        transfer_mpc_validation_df,
        household_robustness_df,
        wealthy_htm_sensitivity_df,
        baseline_group_thresholds,
        reference_summary_df,
    )

    return {
        "output_dir": str(root),
        "diagnostics": diagnostics,
        "steady_state_aggregates": steady_state_aggregates(baseline_ss),
        "shock_effects": shock_df.to_dict(orient="records"),
        "scenario_summary": scenario_summary,
    }
