from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .distribution import build_group_masks, central_ranges_for_assets, marginal_distributions, stationary_distribution
from .labels import pretty_group_label, pretty_robustness_label


def apply_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.figsize": (7.0, 4.2),
        "lines.linewidth": 2.0,
    })


def save_figure(fig, figures_dir: Path, basename: str):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{basename}.png", dpi=250, bbox_inches="tight")
    fig.savefig(figures_dir / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_stationary_distributions(ss, mpc, config, figures_dir: Path):
    apply_style()
    marginals = marginal_distributions(ss)
    ranges = central_ranges_for_assets(ss, config)
    D = stationary_distribution(ss)
    groups = build_group_masks(ss, config, mpc=mpc)["groups"]

    fig, ax = plt.subplots()
    ax.bar(marginals["b_grid"], marginals["b"], width=np.diff(np.r_[marginals["b_grid"], marginals["b_grid"][-1] + 1]), alpha=0.55, color="0.55", label="Гистограмма")
    ax.plot(marginals["b_grid"], marginals["b"], color="black", label="Плотность")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Ликвидное богатство, b")
    ax.set_ylabel("Плотность распределения")
    ax.set_title("Распределение домохозяйств по ликвидному богатству")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_01_liquid_wealth_distribution")

    fig, ax = plt.subplots()
    ax.bar(marginals["b_grid"], marginals["b"], width=np.diff(np.r_[marginals["b_grid"], marginals["b_grid"][-1] + 1]), alpha=0.55, color="0.55")
    ax.plot(marginals["b_grid"], marginals["b"], color="black")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(ranges["b"])
    ax.set_xlabel("Ликвидное богатство, b")
    ax.set_ylabel("Плотность распределения")
    ax.set_title("Распределение домохозяйств по ликвидному богатству, центральные 95%")
    save_figure(fig, figures_dir, "fig_01b_liquid_wealth_distribution_central95")

    fig, ax = plt.subplots()
    ax.bar(marginals["a_grid"], marginals["a"], width=np.diff(np.r_[marginals["a_grid"], marginals["a_grid"][-1] + 1]), alpha=0.55, color="0.45")
    ax.plot(marginals["a_grid"], marginals["a"], color="black")
    ax.set_xlabel("Неликвидное богатство, a")
    ax.set_ylabel("Плотность распределения")
    ax.set_title("Распределение домохозяйств по неликвидному богатству")
    save_figure(fig, figures_dir, "fig_02_illiquid_wealth_distribution")

    fig, ax = plt.subplots()
    ax.hist(mpc.reshape(-1), bins=np.linspace(0, 1.2, 25), weights=D.reshape(-1), color="0.45", edgecolor="black", alpha=0.75)
    ax.set_xlabel("Предельная склонность к потреблению")
    ax.set_ylabel("Плотность распределения")
    ax.set_title("Распределение предельных склонностей к потреблению")
    save_figure(fig, figures_dir, "fig_04_mpc_distribution")

    fig, ax = plt.subplots()
    means = []
    for i in range(1, 6):
        weight = np.sum(D * groups[f"liquid_q{i}"])
        means.append(0.0 if abs(weight) < 1e-12 else np.sum(D * mpc * groups[f"liquid_q{i}"]) / weight)
    ax.plot(range(1, 6), means, marker="o", color="black")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Квантиль ликвидного богатства")
    ax.set_ylabel("Средняя предельная склонность к потреблению")
    ax.set_title("Предельная склонность к потреблению по группам ликвидного богатства")
    save_figure(fig, figures_dir, "fig_05_mpc_by_liquid_quantile")


def plot_mpc_measure_comparison(ss, slope_mpc, transfer_mpc, config, figures_dir: Path):
    apply_style()
    D = stationary_distribution(ss)
    groups = build_group_masks(ss, config)["groups"]

    slope_upper = np.quantile(slope_mpc.reshape(-1), 0.995)
    transfer_upper = np.quantile(transfer_mpc.reshape(-1), 0.995)
    upper = max(0.4, float(slope_upper), float(transfer_upper))
    bins = np.linspace(0.0, upper, 28)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].hist(
        slope_mpc.reshape(-1),
        bins=bins,
        weights=D.reshape(-1),
        color="0.75",
        edgecolor="black",
        alpha=0.75,
        label="Локальная MPC по liquid grid",
    )
    axes[0].hist(
        transfer_mpc.reshape(-1),
        bins=bins,
        weights=D.reshape(-1),
        histtype="step",
        linewidth=2.0,
        color="black",
        label="MPC из одноразового трансферта",
    )
    axes[0].set_xlabel("Предельная склонность к потреблению")
    axes[0].set_ylabel("Плотность распределения")
    axes[0].set_title("Сравнение измерений MPC")
    axes[0].legend(frameon=False)

    slope_means = []
    transfer_means = []
    for idx in range(1, 6):
        mask = groups[f"liquid_q{idx}"]
        mass = np.sum(D * mask)
        slope_means.append(0.0 if mass <= 1e-12 else float(np.sum(D * slope_mpc * mask) / mass))
        transfer_means.append(0.0 if mass <= 1e-12 else float(np.sum(D * transfer_mpc * mask) / mass))
    x = np.arange(1, 6)
    axes[1].plot(x, slope_means, marker="o", color="0.45", linestyle="-", label="Локальная MPC по liquid grid")
    axes[1].plot(x, transfer_means, marker="s", color="black", linestyle="--", label="MPC из одноразового трансферта")
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Квантиль ликвидного богатства")
    axes[1].set_ylabel("Средняя MPC")
    axes[1].set_title("MPC по квантилям ликвидного богатства")
    axes[1].legend(frameon=False)
    save_figure(fig, figures_dir, "fig_04b_mpc_measure_comparison")


def plot_policy_functions(ss, figures_dir: Path):
    apply_style()
    hh = ss.internals["hh"]
    a_idx = len(hh["a_grid"]) // 2
    b_idx = len(hh["b_grid"]) // 2
    z_indices = [0, len(hh["z_grid"]) // 2, len(hh["z_grid"]) - 1]
    labels = ["Низкий доход", "Средний доход", "Высокий доход"]

    fig, ax = plt.subplots()
    for idx, label in zip(z_indices, labels):
        ax.plot(hh["b_grid"], hh["c"][idx, :, a_idx], label=label)
    ax.set_xlabel("Ликвидное богатство, b")
    ax.set_ylabel("Потребление, c")
    ax.set_title("Функция политики потребления")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_06_consumption_policy")

    fig, ax = plt.subplots()
    for idx, label in zip(z_indices, labels):
        ax.plot(hh["b_grid"], hh["b"][idx, :, a_idx], label=label)
    ax.plot(hh["b_grid"], hh["b_grid"], linestyle="--", color="black", linewidth=1.0, label="45 градусов")
    ax.set_xlabel("Ликвидное богатство в начале периода, b_t")
    ax.set_ylabel("Ликвидное богатство в следующем периоде, b_{t+1}")
    ax.set_title("Функция политики по ликвидному активу")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_07_liquid_asset_policy")

    fig, ax = plt.subplots()
    for idx, label in zip(z_indices, labels):
        ax.plot(hh["a_grid"], hh["a"][idx, b_idx, :], label=label)
    ax.plot(hh["a_grid"], hh["a_grid"], linestyle="--", color="black", linewidth=1.0, label="45 градусов")
    ax.set_xlabel("Неликвидное богатство в начале периода, a_t")
    ax.set_ylabel("Неликвидное богатство в следующем периоде, a_{t+1}")
    ax.set_title("Функция политики по неликвидному активу")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_08_illiquid_asset_policy")


def plot_aggregate_irfs(aggregate_irf, figures_dir: Path, scenario_name="baseline"):
    apply_style()
    subset = aggregate_irf[aggregate_irf["scenario"] == scenario_name]
    specs = [
        ("i", "Номинальная ставка, п.п.", "Номинальная ставка"),
        ("r", "Реальная ставка, п.п.", "Реальная ставка"),
        ("pi", "Инфляция, п.п.", "Инфляция"),
        ("Y", "Отклонение от стационарного уровня, %", "Выпуск"),
        ("C", "Отклонение от стационарного уровня, %", "Потребление"),
        ("N", "Отклонение от стационарного уровня, %", "Занятость"),
        ("w", "Отклонение от стационарного уровня, %", "Реальная заработная плата"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(10, 11))
    axes = axes.flatten()
    for ax, (variable, ylabel, title) in zip(axes, specs):
        series = subset[subset["variable"] == variable]
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.plot(series["period"], series["value"], color="black")
        ax.set_title(title)
        ax.set_xlabel("Периоды после шока")
        ax.set_ylabel(ylabel)
    axes[-1].axis("off")
    save_figure(fig, figures_dir, "fig_09_15_aggregate_irf_panels")


def plot_wealth_quantiles(distribution_paths, figures_dir: Path, scenario_name="baseline"):
    apply_style()
    subset = distribution_paths[distribution_paths["scenario"] == scenario_name]

    fig, ax = plt.subplots()
    for column, label, style in [
        ("p10_liquid_wealth", "10-й перцентиль", "-"),
        ("p25_liquid_wealth", "25-й перцентиль", "--"),
        ("median_liquid_wealth", "Медиана", "-."),
        ("p75_liquid_wealth", "75-й перцентиль", ":"),
        ("p90_liquid_wealth", "90-й перцентиль", (0, (3, 1, 1, 1))),
    ]:
        ax.plot(subset["period"], subset[column], linestyle=style, label=label)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Ликвидное богатство")
    ax.set_title("Квантили ликвидного богатства")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_16_liquid_wealth_quantiles")

    fig, ax = plt.subplots()
    for column, label, style in [
        ("p10_illiquid_wealth", "10-й перцентиль", "-"),
        ("p25_illiquid_wealth", "25-й перцентиль", "--"),
        ("median_illiquid_wealth", "Медиана", "-."),
        ("p75_illiquid_wealth", "75-й перцентиль", ":"),
        ("p90_illiquid_wealth", "90-й перцентиль", (0, (3, 1, 1, 1))),
    ]:
        ax.plot(subset["period"], subset[column], linestyle=style, label=label)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Неликвидное богатство")
    ax.set_title("Квантили неликвидного богатства")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_17_illiquid_wealth_quantiles")


def plot_low_liquidity_shares(distribution_paths, figures_dir: Path, scenario_name="baseline"):
    apply_style()
    subset = distribution_paths[distribution_paths["scenario"] == scenario_name]
    fig, ax = plt.subplots()
    ax.plot(subset["period"], 100.0 * subset["share_low_liquidity"], label="Низколиквидные домохозяйства")
    ax.plot(subset["period"], 100.0 * subset["share_wealthy_htm"], label="Состоятельные с низкой ликвидностью (WHtM)", linestyle="--")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Доля домохозяйств, %")
    ax.set_title("Доля домохозяйств с низкой ликвидностью")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_18_low_liquidity_shares")


def plot_group_irfs(group_consumption_irf, group_income_irf, figures_dir: Path, scenario_name="baseline"):
    apply_style()
    q_colors = plt.cm.Greys(np.linspace(0.3, 0.9, 5))
    subset = group_consumption_irf[group_consumption_irf["scenario"] == scenario_name]

    fig, ax = plt.subplots()
    for i, color in zip(range(1, 6), q_colors):
        group = f"liquid_q{i}"
        values = subset[subset["group"] == group]
        ax.plot(values["period"], values["value"], color=color, label=f"{i}-й квантиль")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Отклонение потребления от стационарного уровня, %")
    ax.set_title("Отклик потребления по квантилям ликвидного богатства")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_19_consumption_by_liquid_quantiles")

    fig, ax = plt.subplots()
    for group, label, style in [
        ("mpc_high", "Высокая MPC", "-"),
        ("mpc_mid", "Средняя MPC", "--"),
        ("mpc_low", "Низкая MPC", "-."),
    ]:
        values = subset[subset["group"] == group]
        ax.plot(values["period"], values["value"], linestyle=style, label=label, color="black")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Отклонение потребления от стационарного уровня, %")
    ax.set_title("Отклик потребления по группам предельной склонности к потреблению")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_20_consumption_by_mpc_groups")

    fig, ax = plt.subplots()
    for group_name, label, style in [
        ("low_liquid", "Низкая ликвидность", "-"),
        ("wealthy_htm", "Состоятельные с низкой ликвидностью (WHtM)", "--"),
        ("high_liquid", "Высоколиквидные", "-."),
    ]:
        values = subset[subset["group"] == group_name]
        ax.plot(values["period"], values["value"], linestyle=style, label=label, color="black")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Отклонение потребления от стационарного уровня, %")
    ax.set_title("Отклик потребления по структуре богатства")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_20b_consumption_by_wealth_structure")

    income_subset = group_income_irf[group_income_irf["scenario"] == scenario_name]
    fig, ax = plt.subplots()
    for variable, label, style in [
        ("labor_income", "Трудовой доход", "-"),
        ("financial_income", "Финансовый доход", "--"),
        ("disposable_income", "Располагаемый доход", "-."),
    ]:
        values = income_subset[(income_subset["group"] == "low_liquid") & (income_subset["variable"] == variable)]
        ax.plot(values["period"], values["value"], linestyle=style, label=label, color="black")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Отклонение от стационарного уровня, %")
    ax.set_title("Отклики дохода по каналам у низколиквидных домохозяйств")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_21_income_channels_low_liquid")


def plot_distribution_dynamics(liquid_snapshots, illiquid_snapshots, figures_dir: Path, scenario_name="baseline"):
    apply_style()
    liquid_snapshots = liquid_snapshots[liquid_snapshots["scenario"] == scenario_name]
    illiquid_snapshots = illiquid_snapshots[illiquid_snapshots["scenario"] == scenario_name]
    horizons = sorted(liquid_snapshots["period"].unique())
    greys = plt.cm.Greys(np.linspace(0.2, 0.9, len(horizons)))

    fig, ax = plt.subplots()
    for horizon, color in zip(horizons, greys):
        subset = liquid_snapshots[liquid_snapshots["period"] == horizon]
        label = "До шока" if horizon == 0 else f"Через {horizon} периодов"
        ax.plot(subset["b"], subset["mass"], color=color, label=label)
    ax.set_xlabel("Ликвидное богатство, b")
    ax.set_ylabel("Плотность распределения")
    ax.set_title("Динамика распределения ликвидного богатства после шока")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_22_liquid_distribution_dynamics")

    fig, ax = plt.subplots()
    for horizon, color in zip(horizons, greys):
        subset = illiquid_snapshots[illiquid_snapshots["period"] == horizon]
        label = "До шока" if horizon == 0 else f"Через {horizon} периодов"
        ax.plot(subset["a"], subset["mass"], color=color, label=label)
    ax.set_xlabel("Неликвидное богатство, a")
    ax.set_ylabel("Плотность распределения")
    ax.set_title("Динамика распределения неликвидного богатства после шока")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_23_illiquid_distribution_dynamics")


def plot_channels(channel_decomposition, aggregate_income_channels, figures_dir: Path, scenario_name="baseline"):
    apply_style()
    subset = channel_decomposition[channel_decomposition["scenario"] == scenario_name]

    fig, ax = plt.subplots()
    for component in [
        "intertemporal_financial_channel",
        "labor_income_channel",
        "redistribution_liquidity_residual",
        "general_equilibrium_total",
    ]:
        values = subset[subset["component"] == component]
        label = values["component_label"].iloc[0] if not values.empty else component
        ax.plot(values["period"], values["value"], label=label)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Отклик совокупного потребления, %")
    ax.set_title("Разложение отклика потребления по каналам трансмиссии")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_27_channel_decomposition")

    income_subset = aggregate_income_channels[aggregate_income_channels["scenario"] == scenario_name]
    fig, ax = plt.subplots()
    for channel, label, style in [
        ("labor_income", "Трудовой доход", "-"),
        ("financial_income", "Финансовый доход", "--"),
        ("disposable_income", "Располагаемый доход", "-."),
    ]:
        values = income_subset[income_subset["channel"] == channel]
        ax.plot(values["period"], values["value"], linestyle=style, label=label, color="black")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Отклонение от стационарного уровня, %")
    ax.set_title("Отклики дохода по каналам")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_28_income_channel_paths")


def plot_group_contributions(group_contributions, figures_dir: Path, scenario_name="baseline"):
    apply_style()
    subset = group_contributions[group_contributions["scenario"] == scenario_name]
    fig, ax = plt.subplots()
    for group_name, style, color in [
        ("low_liquid_non_whtm", "-", "0.15"),
        ("wealthy_htm", "--", "0.35"),
        ("high_liquid", "-.", "0.55"),
        ("other_households", ":", "0.7"),
        ("aggregate_total", "-", "black"),
    ]:
        values = subset[subset["group"] == group_name]
        if values.empty:
            continue
        linewidth = 2.4 if group_name == "aggregate_total" else 1.8
        ax.plot(
            values["period"],
            values["value"],
            linestyle=style,
            color=color,
            linewidth=linewidth,
            label=values["group_label"].iloc[0],
        )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Периоды после шока")
    ax.set_ylabel("Вклад в отклонение совокупного потребления, %")
    ax.set_title("Вклад взаимоисключающих групп в отклик потребления")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_29_group_contribution_components")


def plot_wealthy_htm_sensitivity(wealthy_htm_sensitivity, figures_dir: Path):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for a_q, linestyle in zip(sorted(wealthy_htm_sensitivity["wealthy_htm_a_quantile"].unique()), ["-", "--", "-."]):
        subset = wealthy_htm_sensitivity[wealthy_htm_sensitivity["wealthy_htm_a_quantile"] == a_q].sort_values("low_liquidity_threshold")
        label = f"Порог по a: {int(100 * a_q)}-й перцентиль"
        axes[0].plot(subset["low_liquidity_threshold"], 100.0 * subset["share_wealthy_htm"], linestyle=linestyle, color="black", label=label)
        axes[1].plot(subset["low_liquidity_threshold"], subset["mean_mpc_wealthy_htm"], linestyle=linestyle, color="black", label=label)
    axes[0].set_xlabel("Порог low-liquid по b")
    axes[0].set_ylabel("Доля WHtM, %")
    axes[0].set_title("Чувствительность доли WHtM")
    axes[1].set_xlabel("Порог low-liquid по b")
    axes[1].set_ylabel("Средняя MPC группы")
    axes[1].set_title("Чувствительность MPC у WHtM")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    save_figure(fig, figures_dir, "fig_30_wealthy_htm_sensitivity")


def plot_household_robustness(robustness_summary, figures_dir: Path):
    apply_style()
    ordered = robustness_summary.copy()
    ordered["scenario_label"] = ordered["scenario"].map(pretty_robustness_label)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].plot(ordered["scenario_label"], 100.0 * ordered["mean_mpc"], marker="o", color="black")
    axes[0].set_ylabel("Средняя MPC, %")
    axes[0].set_title("Средняя MPC по robustness-сценариям")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].plot(ordered["scenario_label"], ordered["peak_wealthy_htm_consumption"], marker="o", color="0.2", label="WHtM")
    axes[1].plot(ordered["scenario_label"], ordered["peak_high_liquid_consumption"], marker="s", color="0.55", label="Высоколиквидные")
    axes[1].set_ylabel("Пик отклика потребления, %")
    axes[1].set_title("Пиковые групповые отклики")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False)
    save_figure(fig, figures_dir, "fig_31_household_robustness")


def plot_policy_scenario_comparison(aggregate_irf, figures_dir: Path):
    apply_style()
    specs = [
        ("pi", "Инфляция, п.п.", "Инфляция при альтернативных правилах политики"),
        ("Y", "Выпуск, %", "Выпуск при альтернативных правилах политики"),
        ("C", "Потребление, %", "Потребление при альтернативных правилах политики"),
        ("i", "Ставка, п.п.", "Ставка при альтернативных правилах политики"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    styles = {
        "baseline": "-",
        "high_phi_pi": "--",
        "high_rho_i": "-.",
        "low_phi_y": ":",
        "persistent_shock": (0, (5, 1, 1, 1)),
    }
    for ax, (variable, ylabel, title) in zip(axes, specs):
        for scenario, linestyle in styles.items():
            subset = aggregate_irf[(aggregate_irf["scenario"] == scenario) & (aggregate_irf["variable"] == variable)]
            if subset.empty:
                continue
            label = subset["scenario_label"].iloc[0]
            ax.plot(subset["period"], subset["value"], linestyle=linestyle, label=label, color="black")
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Периоды после шока")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    save_figure(fig, figures_dir, "fig_32_policy_scenario_comparison")


def plot_reference_alignment(reference_summary, figures_dir: Path):
    apply_style()
    ordered = reference_summary.copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    axes[0].bar(ordered["label"], 100.0 * ordered["mean_mpc"], color=["0.15", "0.65"])
    axes[0].set_ylabel("Средняя MPC, %")
    axes[0].set_title("Сравнение с tutorial two_asset")
    axes[0].tick_params(axis="x", rotation=18)

    width = 0.35
    x = np.arange(len(ordered))
    axes[1].bar(x - width / 2, ordered["peak_consumption_response"], width=width, color="0.25", label="Пик отклика потребления")
    axes[1].bar(x + width / 2, ordered["peak_output_response"], width=width, color="0.65", label="Пик отклика выпуска")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ordered["label"], rotation=18)
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].set_ylabel("Отклонение от стационара, %")
    axes[1].set_title("Transmission relative to tutorial")
    axes[1].legend(frameon=False)

    save_figure(fig, figures_dir, "fig_33_reference_alignment")
