from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_ORDER = (
    "aggregate_only",
    "aggregate_history",
    "filtered_aggregates",
    "observed_distribution",
    "filtered_distribution",
    "full_information",
)

STATE_LABELS = {
    "aggregate_only": "Текущие\nагрегаты",
    "aggregate_history": "История\nагрегатов",
    "filtered_aggregates": "Фильтрованные\nагрегаты",
    "observed_distribution": "Наблюдаемые\nраспр. показатели",
    "filtered_distribution": "Фильтрованные\nраспр. показатели",
    "full_information": "Полная\nинформация",
}

PALETTE = {
    "aggregate": "#7c8a99",
    "filtered": "#276c8f",
    "distribution": "#c06c2d",
    "full": "#4f6f3f",
    "placebo": "#9a9a9a",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build article figures for the HANK/SSJ information experiment.")
    parser.add_argument("--main-voi-dir", default="outputs/ssj/stochastic/main_voi_joint_filter")
    parser.add_argument("--placebo-summary", default="outputs/ssj/stochastic/placebo/placebo_summary.csv")
    parser.add_argument(
        "--identification-summary",
        default="outputs/ssj/stochastic/identification_battery/identification_battery_summary.csv",
    )
    parser.add_argument("--noise-summary", default="outputs/ssj/stochastic/noise_sensitivity/noise_sensitivity_summary.csv")
    parser.add_argument(
        "--signal-strength-summary",
        default="outputs/ssj/stochastic/distributional_signal_strength/distributional_signal_strength_summary.csv",
    )
    parser.add_argument(
        "--loss-decomposition",
        default="outputs/ssj/stochastic/main_voi_joint_filter/loss_component_decomposition.csv",
    )
    parser.add_argument(
        "--income-risk-summary",
        default="outputs/ssj/income_risk_calibration/income_risk_calibration_summary.csv",
    )
    parser.add_argument(
        "--trajectory-count-summary",
        default="outputs/ssj/stochastic/trajectory_count_robustness/trajectory_count_robustness_summary.csv",
    )
    parser.add_argument(
        "--income-risk-shock-source-summary",
        default="outputs/ssj/stochastic/income_risk_shock_source/income_risk_shock_source_summary.csv",
    )
    parser.add_argument(
        "--liquid-wedge-summary",
        default="outputs/ssj/liquid_wedge_channel/liquid_wedge_channel_summary.csv",
    )
    parser.add_argument("--output-dir", default="article/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_main_losses(Path(args.main_voi_dir), output_dir)
    _plot_distributional_effect_evidence(
        Path(args.main_voi_dir),
        Path(args.placebo_summary),
        Path(args.identification_summary),
        output_dir,
    )
    _plot_noise_sensitivity(Path(args.noise_summary), output_dir)
    _plot_artificial_checks(Path(args.placebo_summary), output_dir)
    _plot_signal_strength(Path(args.signal_strength_summary), output_dir)
    _plot_loss_decomposition(Path(args.loss_decomposition), output_dir)
    if Path(args.income_risk_summary).exists():
        _plot_income_risk_calibration(Path(args.income_risk_summary), output_dir)
    if (
        Path(args.trajectory_count_summary).exists()
        and Path(args.income_risk_shock_source_summary).exists()
        and Path(args.liquid_wedge_summary).exists()
    ):
        _plot_additional_robustness(
            trajectory_count_summary_csv=Path(args.trajectory_count_summary),
            income_risk_shock_source_summary_csv=Path(args.income_risk_shock_source_summary),
            liquid_wedge_summary_csv=Path(args.liquid_wedge_summary),
            output_dir=output_dir,
        )

    print(f"Wrote figures to {output_dir}")


def _plot_main_losses(main_voi_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(main_voi_dir / "main_voi_summary.csv")
    frame = summary[summary["scenario"] == "all"].set_index("information_state").loc[list(STATE_ORDER)].reset_index()
    x = np.arange(len(frame))
    means = frame["mean_loss"].to_numpy(dtype=float)
    err_low = means - frame["ci_low"].to_numpy(dtype=float)
    err_high = frame["ci_high"].to_numpy(dtype=float) - means
    colors = [
        PALETTE["aggregate"],
        PALETTE["aggregate"],
        PALETTE["filtered"],
        PALETTE["distribution"],
        PALETTE["distribution"],
        PALETTE["full"],
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x, means, yerr=[err_low, err_high], capsize=4, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS[state] for state in frame["information_state"]], rotation=0, ha="center")
    ax.set_ylabel("Средние потери")
    ax.set_title("Потери политики по информационным состояниям")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_main_information_states.pdf")
    plt.close(fig)


def _plot_distributional_effect_evidence(
    main_voi_dir: Path,
    placebo_summary_csv: Path,
    identification_summary_csv: Path,
    output_dir: Path,
) -> None:
    losses = pd.read_csv(main_voi_dir / "trajectory_losses.csv")
    paired = (
        losses[
            losses["information_state"].isin(
                ["filtered_aggregates", "filtered_distribution"]
            )
        ]
        .pivot_table(
            index=["scenario", "observation_seed"],
            columns="information_state",
            values="total_loss",
            aggfunc="first",
        )
        .dropna()
    )
    reductions = (
        paired["filtered_aggregates"].to_numpy(dtype=float)
        - paired["filtered_distribution"].to_numpy(dtype=float)
    )
    reductions_sorted = np.sort(reductions)
    mean_reduction = float(np.mean(reductions))
    win_share = float(np.mean(reductions > 0.0))

    if identification_summary_csv.exists():
        frame = pd.read_csv(identification_summary_csv)
        order = [
            "actual_distribution",
            "residualized_distribution",
            "fake_matched_distribution",
            "permuted_by_scenario",
            "permuted_by_time",
            "lagged_distribution",
            "future_shifted_distribution",
        ]
        labels = {
            "actual_distribution": "Факт.",
            "residualized_distribution": "Остат.",
            "fake_matched_distribution": "Искус.",
            "permuted_by_scenario": "Сценар.",
            "permuted_by_time": "Время",
            "lagged_distribution": "Лаг",
            "future_shifted_distribution": "Сдвиг вперёд",
        }
        frame = frame.set_index("variant").loc[order].reset_index()
        x = np.arange(len(frame))
        means = frame["loss_reduction"].to_numpy(dtype=float)
        ci_low_reduction = -frame["ci_high"].to_numpy(dtype=float)
        ci_high_reduction = -frame["ci_low"].to_numpy(dtype=float)
        err_low = np.maximum(means - ci_low_reduction, 0.0)
        err_high = np.maximum(ci_high_reduction - means, 0.0)
        bar_labels = [labels[name] for name in frame["variant"]]
        colors = [
            PALETTE["distribution"],
            "#8a4f9a",
            PALETTE["placebo"],
            PALETTE["placebo"],
            PALETTE["placebo"],
            "#b86b6b",
            "#b86b6b",
        ]
        right_title = "Идентификационные проверки"
    else:
        placebo = pd.read_csv(placebo_summary_csv)
        labels = {
            "actual": "Фактические\nряды",
            "permuted": "Перемешанные\nряды",
            "fake": "Искусственные\nряды",
        }
        frame = placebo.set_index("run").loc[["actual", "permuted", "fake"]].reset_index()
        x = np.arange(len(frame))
        means = frame["loss_reduction"].to_numpy(dtype=float)
        ci_low_reduction = -frame["ci_high"].to_numpy(dtype=float)
        ci_high_reduction = -frame["ci_low"].to_numpy(dtype=float)
        err_low = np.maximum(means - ci_low_reduction, 0.0)
        err_high = np.maximum(ci_high_reduction - means, 0.0)
        bar_labels = [labels[name] for name in frame["run"]]
        colors = [PALETTE["distribution"], PALETTE["placebo"], PALETTE["placebo"]]
        right_title = "Фактический сигнал против искусственного"

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].axhline(mean_reduction, color=PALETTE["distribution"], linestyle="--", linewidth=1.6)
    axes[0].scatter(
        np.arange(1, reductions_sorted.size + 1),
        reductions_sorted,
        s=14,
        color=PALETTE["distribution"],
        alpha=0.72,
        edgecolor="none",
    )
    axes[0].set_title("Парные снижения потерь")
    axes[0].set_xlabel("Тестовая траектория, отсортировано")
    axes[0].set_ylabel("Снижение потерь")
    axes[0].text(
        0.04,
        0.93,
        f"N = {reductions.size}\nсреднее = {mean_reduction:.6f}\nдоля выигрышей = {win_share:.3f}",
        transform=axes[0].transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dddddd"},
    )

    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].bar(x, means, yerr=[err_low, err_high], capsize=4, color=colors, edgecolor="#222222", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bar_labels, rotation=25, ha="right")
    axes[1].set_ylabel("Среднее снижение потерь")
    axes[1].set_title(right_title)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Статистическая проверка ценности распределительной информации")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_distributional_effect_evidence.pdf")
    plt.close(fig)


def _plot_noise_sensitivity(noise_summary_csv: Path, output_dir: Path) -> None:
    summary = pd.read_csv(noise_summary_csv)
    baseline = summary[summary["case"] == "baseline"].iloc[0]
    aggregate_cases = (
        summary[summary["axis"].isin(["aggregate", "both"])]
        .sort_values("aggregate_noise_scale")
        .drop_duplicates("aggregate_noise_scale")
    )
    distribution_cases = (
        summary[summary["axis"].isin(["distribution", "both"])]
        .sort_values("distribution_noise_scale")
        .drop_duplicates("distribution_noise_scale")
    )
    if baseline["aggregate_noise_scale"] not in aggregate_cases["aggregate_noise_scale"].to_numpy(dtype=float):
        aggregate_cases = pd.concat([aggregate_cases, baseline.to_frame().T], ignore_index=True)
    if baseline["distribution_noise_scale"] not in distribution_cases["distribution_noise_scale"].to_numpy(dtype=float):
        distribution_cases = pd.concat([distribution_cases, baseline.to_frame().T], ignore_index=True)
    aggregate_cases = aggregate_cases.sort_values("aggregate_noise_scale")
    distribution_cases = distribution_cases.sort_values("distribution_noise_scale")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    axes[0].plot(
        aggregate_cases["aggregate_noise_scale"],
        aggregate_cases["mvoi_dist"],
        marker="o",
        color=PALETTE["distribution"],
        linewidth=2.0,
    )
    axes[0].set_title("Шум агрегатов")
    axes[0].set_xlabel("Множитель шума")
    axes[0].set_ylabel("Предельная ценность распределительной информации")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        distribution_cases["distribution_noise_scale"],
        distribution_cases["mvoi_dist"],
        marker="o",
        color=PALETTE["filtered"],
        linewidth=2.0,
    )
    axes[1].set_title("Шум распределительных показателей")
    axes[1].set_xlabel("Множитель шума")
    axes[1].grid(alpha=0.25)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Когда распределительная информация становится ценнее")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_noise_sensitivity.pdf")
    plt.close(fig)


def _plot_artificial_checks(placebo_summary_csv: Path, output_dir: Path) -> None:
    summary = pd.read_csv(placebo_summary_csv)
    labels = {
        "actual": "Фактические\nряды",
        "permuted": "Перемешанные\nряды",
        "fake": "Искусственные\nряды",
    }
    frame = summary.set_index("run").loc[["actual", "permuted", "fake"]].reset_index()
    x = np.arange(len(frame))
    means = frame["loss_reduction"].to_numpy(dtype=float)
    ci_low_reduction = -frame["ci_high"].to_numpy(dtype=float)
    ci_high_reduction = -frame["ci_low"].to_numpy(dtype=float)
    err_low = np.maximum(means - ci_low_reduction, 0.0)
    err_high = np.maximum(ci_high_reduction - means, 0.0)
    colors = [PALETTE["distribution"], PALETTE["placebo"], PALETTE["placebo"]]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.bar(x, means, yerr=[err_low, err_high], capsize=4, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[name] for name in frame["run"]])
    ax.set_ylabel("Снижение потерь")
    ax.set_title("Проверки с искусственными распределительными статистиками")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_artificial_distribution_checks.pdf")
    plt.close(fig)


def _plot_signal_strength(signal_strength_csv: Path, output_dir: Path) -> None:
    summary = pd.read_csv(signal_strength_csv).sort_values("distributional_signal_factor")
    x = summary["distributional_signal_factor"].to_numpy(dtype=float)
    means = summary["mvoi_dist"].to_numpy(dtype=float)
    ci_low_reduction = -summary["ci_high"].to_numpy(dtype=float)
    ci_high_reduction = -summary["ci_low"].to_numpy(dtype=float)
    err_low = np.maximum(means - ci_low_reduction, 0.0)
    err_high = np.maximum(ci_high_reduction - means, 0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.errorbar(
        x,
        means,
        yerr=[err_low, err_high],
        marker="o",
        color=PALETTE["distribution"],
        linewidth=2.0,
        capsize=4,
    )
    ax.set_xlabel("Сила распределительного сигнала")
    ax.set_ylabel("Предельная ценность распределительной информации")
    ax.set_title("Ценность информации растёт с силой распределительного сигнала")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_distributional_signal_strength.pdf")
    plt.close(fig)


def _plot_loss_decomposition(loss_decomposition_csv: Path, output_dir: Path) -> None:
    frame = pd.read_csv(loss_decomposition_csv)
    comparisons = [
        ("filtered_aggregates_minus_aggregate_only", "Фильтрованные\nагрегаты"),
        ("filtered_distribution_minus_filtered_aggregates", "Распределительная\nинформация"),
    ]
    components = [
        ("inflation_loss", "Инфляция", "#3f6c9f"),
        ("output_gap_loss", "Разрыв выпуска", "#c06c2d"),
        ("consumption_loss", "Потребление", "#5f8f4e"),
        ("rate_smoothing_loss", "Ставка", "#8f5f8f"),
    ]
    overall = frame[frame["scenario"] == "all"]
    rows = []
    for comparison, label in comparisons:
        subset = overall[overall["comparison"] == comparison].set_index("component")
        row = {"label": label}
        for component, _, _ in components:
            row[component] = float(subset.loc[component, "mean_reduction"])
        rows.append(row)
    plot_frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(plot_frame))
    bottom = np.zeros(len(plot_frame), dtype=float)
    for component, label, color in components:
        values = plot_frame[component].to_numpy(dtype=float)
        positive = np.maximum(values, 0.0)
        ax.bar(x, positive, bottom=bottom, color=color, edgecolor="#222222", linewidth=0.5, label=label)
        bottom += positive
        negative = np.minimum(values, 0.0)
        if np.any(np.abs(negative) > 0):
            ax.bar(x, negative, color=color, edgecolor="#222222", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_frame["label"])
    ax.set_ylabel("Снижение потерь")
    ax.set_title("Через какие компоненты снижаются потери")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_loss_component_decomposition.pdf")
    plt.close(fig)


def _plot_income_risk_calibration(income_risk_summary_csv: Path, output_dir: Path) -> None:
    summary = pd.read_csv(income_risk_summary_csv).sort_values("sigma_z")
    x = summary["sigma_z"].to_numpy(dtype=float)
    means = summary["mvoi_dist"].to_numpy(dtype=float)
    ci_low_reduction = -summary["ci_high"].to_numpy(dtype=float)
    ci_high_reduction = -summary["ci_low"].to_numpy(dtype=float)
    err_low = np.maximum(means - ci_low_reduction, 0.0)
    err_high = np.maximum(ci_high_reduction - means, 0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.errorbar(
        x,
        means,
        yerr=[err_low, err_high],
        marker="o",
        color=PALETTE["distribution"],
        linewidth=2.0,
        capsize=4,
    )
    ax.set_xlabel(r"Доходный риск $\sigma_z$")
    ax.set_ylabel("Предельная ценность распределительной информации")
    ax.set_title("Чувствительность к калибровке доходного риска HANK")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_income_risk_calibration.pdf")
    plt.close(fig)


def _plot_additional_robustness(
    *,
    trajectory_count_summary_csv: Path,
    income_risk_shock_source_summary_csv: Path,
    liquid_wedge_summary_csv: Path,
    output_dir: Path,
) -> None:
    trajectory = pd.read_csv(trajectory_count_summary_csv).sort_values("num_hank_paths")
    shock = pd.read_csv(income_risk_shock_source_summary_csv)
    wedge = pd.read_csv(liquid_wedge_summary_csv).sort_values("omega")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    _errorbar_from_delta_ci(
        ax=axes[0],
        x=trajectory["num_hank_paths"].to_numpy(dtype=float),
        means=trajectory["mvoi_dist"].to_numpy(dtype=float),
        ci_low_delta=trajectory["ci_low"].to_numpy(dtype=float),
        ci_high_delta=trajectory["ci_high"].to_numpy(dtype=float),
        color=PALETTE["distribution"],
        marker="o",
    )
    axes[0].set_title("Число HANK/SSJ-путей")
    axes[0].set_xlabel("Пути")
    axes[0].set_ylabel("Предельная ценность")

    shock_labels = ["Базовые\nшоки", "+ шок\nдоходного риска"]
    _errorbar_from_delta_ci(
        ax=axes[1],
        x=np.arange(len(shock)),
        means=shock["mvoi_dist"].to_numpy(dtype=float),
        ci_low_delta=shock["ci_low"].to_numpy(dtype=float),
        ci_high_delta=shock["ci_high"].to_numpy(dtype=float),
        color=PALETTE["filtered"],
        marker="o",
    )
    axes[1].set_title("Источник доходного риска")
    axes[1].set_xticks(np.arange(len(shock)))
    axes[1].set_xticklabels(shock_labels)

    _errorbar_from_delta_ci(
        ax=axes[2],
        x=wedge["omega"].to_numpy(dtype=float),
        means=wedge["mvoi_dist"].to_numpy(dtype=float),
        ci_low_delta=wedge["ci_low"].to_numpy(dtype=float),
        ci_high_delta=wedge["ci_high"].to_numpy(dtype=float),
        color=PALETTE["distribution"],
        marker="o",
    )
    axes[2].set_title("Клин ликвидной доходности")
    axes[2].set_xlabel(r"\(\omega\)")

    for ax in axes:
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Дополнительные проверки предельной ценности распределительной информации")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_additional_robustness.pdf")
    plt.close(fig)


def _errorbar_from_delta_ci(
    *,
    ax,
    x: np.ndarray,
    means: np.ndarray,
    ci_low_delta: np.ndarray,
    ci_high_delta: np.ndarray,
    color: str,
    marker: str,
) -> None:
    ci_low_reduction = -ci_high_delta
    ci_high_reduction = -ci_low_delta
    err_low = np.maximum(means - ci_low_reduction, 0.0)
    err_high = np.maximum(ci_high_reduction - means, 0.0)
    ax.errorbar(
        x,
        means,
        yerr=[err_low, err_high],
        marker=marker,
        color=color,
        linewidth=2.0,
        capsize=4,
    )


if __name__ == "__main__":
    main()
