from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_LABEL_ORDER = {
    "full_information_rule": "Полная информация",
    "classical_filtered_rule": "Classical: filter + fixed rule",
    "learning_policy": "Learning-based policy",
}

POLICY_STYLES = {
    "full_information_rule": ("black", "-"),
    "classical_filtered_rule": ("#1f77b4", "-."),
    "learning_policy": ("#d62728", "-"),
}


def _save_figure(figure: plt.Figure, basepath: Path):
    figure.tight_layout()
    figure.savefig(basepath.with_suffix(".png"), dpi=250)
    figure.savefig(basepath.with_suffix(".pdf"))
    plt.close(figure)


def _style_axis(axis, ylabel: str, xlabel: str = "Периоды"):
    axis.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2, linewidth=0.5)


def plot_policy_paths(policy_paths: pd.DataFrame, figures_dir: Path, variant_name: str):
    subset = policy_paths[policy_paths["variant_name"] == variant_name].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for policy_name, group in subset.groupby("policy_name"):
        color, linestyle = POLICY_STYLES.get(policy_name, ("0.4", "-"))
        label = group["policy_label"].iloc[0] if "policy_label" in group else POLICY_LABEL_ORDER.get(policy_name, policy_name)
        axes[0].plot(group["period"], 100.0 * group["policy_rate"], color=color, linestyle=linestyle, linewidth=2.0, label=label)
        axes[1].plot(group["period"], group["cumulative_policy_loss"], color=color, linestyle=linestyle, linewidth=2.0, label=label)
    axes[0].set_title("Траектория процентной ставки")
    axes[1].set_title("Накопленная потеря")
    _style_axis(axes[0], "Ставка, п.п. отклонения")
    _style_axis(axes[1], "Накопленная потеря")
    axes[0].legend(frameon=False, loc="best")
    _save_figure(figure, figures_dir / f"fig_01_policy_paths_{variant_name}")


def plot_macro_paths(aggregate_paths: pd.DataFrame, figures_dir: Path, variant_name: str):
    subset = aggregate_paths[aggregate_paths["scenario"] == variant_name].copy()
    variables = [
        ("pi_deviation", 100.0, "Инфляция", "п.п. отклонения"),
        ("output_gap_deviation", 100.0, "Разрыв выпуска", "% отклонения"),
        ("C_pct", 1.0, "Потребление", "% отклонения"),
        ("i_deviation", 100.0, "Номинальная ставка", "п.п. отклонения"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for axis, (column, scale, title, ylabel) in zip(axes, variables):
        for policy_name, group in subset.groupby("policy_name"):
            color, linestyle = POLICY_STYLES.get(policy_name, ("0.4", "-"))
            label = group["policy_label"].iloc[0] if "policy_label" in group else POLICY_LABEL_ORDER.get(policy_name, policy_name)
            axis.plot(group["period"], scale * group[column], color=color, linestyle=linestyle, linewidth=2.0, label=label)
        axis.set_title(title)
        _style_axis(axis, ylabel)
    axes[0].legend(frameon=False, loc="best")
    _save_figure(figure, figures_dir / f"fig_02_macro_paths_{variant_name}")


def plot_group_consumption(group_paths: pd.DataFrame, figures_dir: Path, variant_name: str):
    subset = group_paths[
        (group_paths["scenario"] == variant_name)
        & (group_paths["group"].isin(["liquid_q1", "liquid_q5", "wealthy_htm"]))
    ].copy()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, sharey=True)
    title_map = {
        "liquid_q1": "Нижний квантиль ликвидности",
        "liquid_q5": "Верхний квантиль ликвидности",
        "wealthy_htm": "Wealthy hand-to-mouth",
    }
    for axis, group_name in zip(axes, ["liquid_q1", "liquid_q5", "wealthy_htm"]):
        panel = subset[subset["group"] == group_name]
        for policy_name, frame in panel.groupby("policy_name"):
            color, linestyle = POLICY_STYLES.get(policy_name, ("0.4", "-"))
            label = frame["policy_label"].iloc[0] if "policy_label" in frame else POLICY_LABEL_ORDER.get(policy_name, policy_name)
            axis.plot(frame["period"], frame["consumption_pct_deviation"], color=color, linestyle=linestyle, linewidth=2.0, label=label)
        axis.set_title(title_map[group_name])
        _style_axis(axis, "Отклонение потребления, %")
    axes[0].legend(frameon=False, loc="best")
    _save_figure(figure, figures_dir / f"fig_03_group_consumption_{variant_name}")


def plot_scenario_performance(policy_comparison: pd.DataFrame, figures_dir: Path):
    main = policy_comparison.copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    positions = np.arange(len(main))
    width = 0.38
    axes[0].bar(positions - width / 2, main["classical_mean_policy_loss"], width=width, color="#1f77b4", label="Classical")
    axes[0].bar(positions + width / 2, main["rl_mean_policy_loss"], width=width, color="#d62728", label="RL")
    axes[0].set_xticks(positions, main["scenario_label"], rotation=18)
    axes[0].set_title("Средняя policy loss по сценариям")
    axes[0].set_ylabel("Средняя потеря")
    axes[0].grid(alpha=0.2, linewidth=0.5, axis="y")
    axes[0].legend(frameon=False)

    axes[1].bar(main["scenario_label"], main["delta_cumulative_policy_loss_rl_minus_classical"], color="#444444")
    axes[1].set_title("RL минус classical: накопленная потеря")
    axes[1].set_ylabel("Разница накопленной потери")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(alpha=0.2, linewidth=0.5, axis="y")

    _save_figure(figure, figures_dir / "fig_04_scenario_performance")


def plot_ablations(policy_comparison: pd.DataFrame, figures_dir: Path):
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(policy_comparison["scenario_label"], policy_comparison["delta_mean_policy_loss_rl_minus_classical"], color="#7f7f7f")
    axis.set_title("Абляции: RL минус classical по средней loss")
    axis.set_ylabel("Разница средней policy loss")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(alpha=0.2, linewidth=0.5, axis="y")
    _save_figure(figure, figures_dir / "fig_05_ablation_performance")


def plot_training_curve(training_history: pd.DataFrame, figures_dir: Path, variant_name: str):
    subset = training_history[training_history["label"] == variant_name].copy()
    figure, axis = plt.subplots(figsize=(10, 4.4))
    for training_seed, frame in subset.groupby("training_seed"):
        axis.plot(
            frame["iteration"],
            frame["best_validation_return"],
            linewidth=2.0,
            label=f"seed {training_seed}",
        )
    axis.set_title("Валидационный return во время обучения PPO")
    axis.set_xlabel("Итерация")
    axis.set_ylabel("Лучший validation return")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(frameon=False, loc="best")
    _save_figure(figure, figures_dir / f"fig_06_training_curve_{variant_name}")
