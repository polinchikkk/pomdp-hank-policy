from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style_axis(axis, ylabel: str, xlabel: str = "Периоды после шока"):
    axis.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2, linewidth=0.5)


def _save_figure(figure: plt.Figure, basepath: Path):
    figure.tight_layout()
    figure.savefig(basepath.with_suffix(".png"), dpi=250)
    figure.savefig(basepath.with_suffix(".pdf"))
    plt.close(figure)


def plot_aggregate_comparison(aggregate_paths: pd.DataFrame, figures_dir: Path):
    variables = [
        ("pi_deviation", 100.0, "Инфляция", "п.п. отклонения"),
        ("output_gap_deviation", 100.0, "Выпуск", "% отклонения"),
        ("C_deviation", 100.0 / aggregate_paths["C_level"].iloc[0], "Потребление", "% отклонения"),
        ("i_deviation", 100.0, "Номинальная ставка", "п.п. отклонения"),
        ("r_deviation", 100.0, "Реальная ставка", "п.п. отклонения"),
        ("N_deviation", 100.0 / aggregate_paths["N_level"].iloc[0], "Занятость", "% отклонения"),
    ]
    focus = aggregate_paths[aggregate_paths["scenario"].isin(["full_information", "macro_core", "thin_information"])].copy()
    labels = {
        "full_information": "Полная информация",
        "macro_core": "Фильтрация: инфляция, выпуск и ставка",
        "thin_information": "Фильтрация: инфляция и ставка",
    }
    styles = {
        "full_information": ("black", "-"),
        "macro_core": ("#1f77b4", "-."),
        "thin_information": ("#d62728", ":"),
    }

    figure, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    axes = axes.ravel()
    for axis, (column, scale, title, ylabel_suffix) in zip(axes, variables):
        for scenario_name, group in focus.groupby("scenario"):
            axis.plot(
                group["period"],
                scale * group[column],
                label=labels.get(scenario_name, scenario_name),
                color=styles.get(scenario_name, ("0.3", "-"))[0],
                linestyle=styles.get(scenario_name, ("0.3", "-"))[1],
                linewidth=2.0,
            )
        axis.set_title(title)
        _style_axis(axis, ylabel_suffix)
    axes[0].legend(frameon=False, loc="best")
    _save_figure(figure, figures_dir / "fig_01_aggregate_irf_full_vs_filter")


def plot_hidden_state_estimates(filtered_states: pd.DataFrame, figures_dir: Path):
    focus_states = [
        "rstar_gap",
        "inflation_gap",
        "output_gap",
        "low_liquidity_gap",
        "mean_mpc_gap",
    ]
    subset = filtered_states[filtered_states["scenario"] == "macro_core"].copy()
    figure, axes = plt.subplots(len(focus_states), 1, figsize=(10, 12), sharex=True)
    for axis, state_name in zip(axes, focus_states):
        axis.plot(subset["period"], subset[f"true_{state_name}"], label="Истинное состояние", color="black", linewidth=1.8)
        axis.plot(subset["period"], subset[f"filtered_{state_name}"], label="Оценка фильтра", color="#1f77b4", linewidth=1.6)
        axis.fill_between(
            subset["period"],
            subset[f"lower_{state_name}"],
            subset[f"upper_{state_name}"],
            color="#1f77b4",
            alpha=0.18,
            label="Доверительный интервал" if state_name == focus_states[0] else None,
        )
        axis.set_title(state_name)
        _style_axis(axis, "Значение состояния", xlabel="Периоды")
    axes[0].legend(frameon=False, loc="best")
    _save_figure(figure, figures_dir / "fig_02_hidden_state_estimates")


def plot_filter_rmse(filter_state_metrics: pd.DataFrame, figures_dir: Path):
    pivot = filter_state_metrics.pivot(index="state", columns="scenario", values="rmse")
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=axis, width=0.85)
    axis.set_title("Средняя ошибка фильтрации по компонентам состояния")
    axis.set_xlabel("Компонента состояния")
    axis.set_ylabel("RMSE")
    axis.legend(frameon=False, title="Сценарий")
    axis.grid(alpha=0.2, linewidth=0.5, axis="y")
    _save_figure(figure, figures_dir / "fig_03_filter_rmse_by_state")


def plot_rate_gap(policy_paths: pd.DataFrame, figures_dir: Path):
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for scenario_name, group in policy_paths.groupby("scenario"):
        if scenario_name == "full_information":
            continue
        axis.plot(
            group["period"],
            100.0 * group["rate_gap"],
            linewidth=2.0,
            label=group["scenario_label"].iloc[0],
        )
    axis.set_title("Ошибка денежно-кредитной политики при неполной информации")
    _style_axis(axis, "Отклонение ставки от полной информации, п.п.")
    axis.legend(frameon=False, loc="best")
    _save_figure(figure, figures_dir / "fig_04_policy_rate_gap")


def plot_policy_losses(policy_paths: pd.DataFrame, figures_dir: Path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for scenario_name, group in policy_paths.groupby("scenario"):
        label = "Полная информация" if scenario_name == "full_information" else group["scenario_label"].iloc[0]
        series = group["full_loss"] if scenario_name == "full_information" else group["filtered_loss"]
        axes[0].plot(group["period"], series, linewidth=2.0, label=label)
        axes[1].plot(group["period"], np.cumsum(series), linewidth=2.0, label=label)
    axes[0].set_title("Потери регулятора по периодам")
    axes[1].set_title("Накопленные потери регулятора")
    _style_axis(axes[0], "Периодическая потеря")
    _style_axis(axes[1], "Накопленная потеря")
    axes[0].legend(frameon=False, loc="best")
    _save_figure(figure, figures_dir / "fig_05_policy_losses")


def plot_distributional_comparison(
    distribution_stats: pd.DataFrame,
    group_comparison: pd.DataFrame,
    figures_dir: Path,
):
    focus_scenarios = ["full_information", "macro_core", "thin_information"]
    dist_subset = distribution_stats[distribution_stats["scenario"].isin(focus_scenarios)]

    figure, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for scenario_name, group in dist_subset.groupby("scenario"):
        label = "Полная информация" if scenario_name == "full_information" else group["scenario_label"].iloc[0]
        axes[1].plot(group["period"], 100.0 * group["share_low_liquidity"], linewidth=2.0, label=label)
        axes[2].plot(group["period"], group["mean_mpc"], linewidth=2.0, label=label)

    liquid_q = group_comparison[
        (group_comparison["grouping"] == "liquid_wealth_quantile")
        & (group_comparison["group"].isin(["liquid_q1", "liquid_q5"]))
        & (group_comparison["scenario"] == "macro_core")
    ]
    for group_name, group in liquid_q.groupby("group"):
        axes[0].plot(
            group["period"],
            group["consumption_pct_deviation_full"],
            linewidth=2.0,
            linestyle="--",
            label=f"Полная информация: {group_name}",
        )
        axes[0].plot(
            group["period"],
            group["consumption_pct_deviation_filtered"],
            linewidth=2.0,
            linestyle="-",
            label=f"Фильтрация: {group_name}",
        )

    axes[0].set_title("Отклик потребления по группам: полная информация и фильтрация")
    axes[1].set_title("Доля домохозяйств с низкой ликвидностью")
    axes[2].set_title("Динамика средней предельной склонности к потреблению")
    _style_axis(axes[0], "Отклонение потребления, %")
    _style_axis(axes[1], "Доля домохозяйств, %")
    _style_axis(axes[2], "Средняя MPC", xlabel="Периоды")
    axes[0].legend(frameon=False, loc="best", ncol=2)
    axes[1].legend(frameon=False, loc="best")
    _save_figure(figure, figures_dir / "fig_06_distributional_comparison")


def plot_distribution_error_heatmap(distribution_error: pd.DataFrame, figures_dir: Path):
    pivot = distribution_error.pivot(index="a", columns="b", values="delta_mass")
    figure, axis = plt.subplots(figsize=(8, 6))
    mesh = axis.pcolormesh(
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
        cmap="coolwarm",
        shading="auto",
    )
    axis.set_title("Искажение распределения домохозяйств из-за неполной информации")
    axis.set_xlabel("Ликвидные активы")
    axis.set_ylabel("Неликвидные активы")
    figure.colorbar(mesh, ax=axis, label="Разность плотностей")
    _save_figure(figure, figures_dir / "fig_07_distribution_error_heatmap")


def plot_scenario_summary(filter_metrics: pd.DataFrame, policy_metrics: pd.DataFrame, figures_dir: Path):
    filter_plot = filter_metrics.copy()
    policy_plot = policy_metrics.copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(filter_plot["scenario_label"], filter_plot["mean_state_rmse"], color="#1f77b4")
    axes[0].set_title("Качество оценки состояния по сценариям")
    axes[0].set_ylabel("Средний RMSE состояния")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.2, linewidth=0.5, axis="y")

    axes[1].bar(policy_plot["scenario_label"], policy_plot["cumulative_excess_loss"], color="#d62728")
    axes[1].set_title("Дополнительная накопленная потеря по сценариям")
    axes[1].set_ylabel("Дополнительная потеря")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.2, linewidth=0.5, axis="y")

    _save_figure(figure, figures_dir / "fig_08_information_scenario_summary")
