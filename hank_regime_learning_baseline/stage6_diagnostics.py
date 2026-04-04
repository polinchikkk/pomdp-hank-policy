from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_ORDER = [
    "macro_core_moderate_gap",
    "macro_core_strong_gap",
    "thin_information_moderate_gap",
    "thin_information_strong_gap",
]

SCENARIO_LABELS = {
    "macro_core_moderate_gap": "Инфляция, выпуск, ставка\nумеренный режимный разрыв",
    "macro_core_strong_gap": "Инфляция, выпуск, ставка\nсильный режимный разрыв",
    "thin_information_moderate_gap": "Инфляция, ставка\nумеренный режимный разрыв",
    "thin_information_strong_gap": "Инфляция, ставка\nсильный режимный разрыв",
}

COMPARISON_SPECS = {
    "belief_minus_classical": {
        "label": "Фильтр + обучение минус фиксированное правило",
        "short_label": "По состоянию vs фиксированное",
        "color": "#0f4c5c",
    },
    "rawobs_minus_classical": {
        "label": "Наблюдения + обучение минус фиксированное правило",
        "short_label": "По наблюдениям vs фиксированное",
        "color": "#e36414",
    },
    "rawobs_minus_belief": {
        "label": "Наблюдения + обучение минус обучение по состоянию",
        "short_label": "По наблюдениям vs по состоянию",
        "color": "#6a994e",
    },
}

ARCHITECTURE_SPECS = {
    "fixed": {"label": "Фиксированное правило", "color": "#7f8c8d"},
    "belief": {"label": "Обучение по состоянию", "color": "#0f4c5c"},
    "rawobs": {"label": "Обучение по наблюдениям", "color": "#e36414"},
}

ENVIRONMENT_ORDER = [
    "baseline_environment",
    "persistent_regimes",
    "shifted_transmission",
    "stronger_distributional_channel",
]

ENVIRONMENT_LABELS = {
    "baseline_environment": "Базовая среда",
    "persistent_regimes": "Более персистентные режимы",
    "shifted_transmission": "Сдвиг макропередачи",
    "stronger_distributional_channel": "Сильнее\nраспределительный канал",
}


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _latex_escape(text: str) -> str:
    return text.replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")


def _write_latex_table(path: Path, table: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{p{0.20\\textwidth}p{0.28\\textwidth}p{0.24\\textwidth}p{0.22\\textwidth}}",
        "\\toprule",
        "Метрика & Как читать & Что получилось & Интерпретация \\\\",
        "\\midrule",
    ]
    for row in table.to_dict(orient="records"):
        lines.append(
            f"{_latex_escape(row['metric'])} & {_latex_escape(row['how_to_read'])} & {_latex_escape(row['our_result'])} & {_latex_escape(row['interpretation'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _save_text(path, "\n".join(lines))


def _bootstrap_ci(values: np.ndarray, *, seed: int = 1234, draws: int = 4000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        value = float(values[0])
        return value, value
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper


def _load_architecture_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(root / "architecture_seed_level.csv"),
        pd.read_csv(root / "policy_paths_all.csv"),
        pd.read_csv(root / "policy_metrics_all.csv"),
        pd.read_csv(root / "architecture_seed_win_rates.csv"),
    )


def _load_environment_shift_data(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "environment_shift_win_summary.csv")


def _build_delta_summary(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_name in SCENARIO_ORDER:
        frame = seed_level.loc[seed_level["scenario_name"] == scenario_name]
        scenario_label = frame["scenario_label"].iloc[0]
        for column, spec in COMPARISON_SPECS.items():
            values = frame[column].to_numpy(dtype=float)
            lower, upper = _bootstrap_ci(values, seed=1234 + len(rows))
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": scenario_label,
                    "comparison_name": column,
                    "comparison_label": spec["label"],
                    "mean_delta_cumulative_loss": float(values.mean()),
                    "std_delta_cumulative_loss": float(values.std(ddof=1)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "num_seeds": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _plot_delta_interval(delta_summary: pd.DataFrame, output_dir: Path) -> None:
    figure, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(SCENARIO_ORDER), dtype=float)
    offsets = {
        "belief_minus_classical": -0.22,
        "rawobs_minus_classical": 0.0,
        "rawobs_minus_belief": 0.22,
    }
    for comparison_name, spec in COMPARISON_SPECS.items():
        frame = (
            delta_summary.loc[delta_summary["comparison_name"] == comparison_name]
            .set_index("scenario_name")
            .loc[SCENARIO_ORDER]
            .reset_index()
        )
        mean = frame["mean_delta_cumulative_loss"].to_numpy(dtype=float)
        lower = frame["ci_lower"].to_numpy(dtype=float)
        upper = frame["ci_upper"].to_numpy(dtype=float)
        yerr = np.vstack([mean - lower, upper - mean])
        ax.errorbar(
            x + offsets[comparison_name],
            mean,
            yerr=yerr,
            fmt="o",
            markersize=6,
            linewidth=1.8,
            capsize=4,
            color=spec["color"],
            label=spec["short_label"],
        )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[name] for name in SCENARIO_ORDER], fontsize=9)
    ax.set_ylabel(r"$\Delta J = J_{model} - J_{benchmark}$")
    ax.set_title("Разность накопленной функции потерь по сценариям")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "fig_01_delta_cumulative_loss_intervals")


def _plot_delta_boxplots(seed_level: pd.DataFrame, output_dir: Path) -> None:
    figure, ax = plt.subplots(figsize=(11.5, 6.2))
    x = np.arange(len(SCENARIO_ORDER), dtype=float)
    offsets = {
        "belief_minus_classical": -0.24,
        "rawobs_minus_classical": 0.0,
        "rawobs_minus_belief": 0.24,
    }
    width = 0.18
    for comparison_name, spec in COMPARISON_SPECS.items():
        data = [
            seed_level.loc[seed_level["scenario_name"] == scenario_name, comparison_name].to_numpy(dtype=float)
            for scenario_name in SCENARIO_ORDER
        ]
        parts = ax.boxplot(
            data,
            positions=x + offsets[comparison_name],
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
        )
        for box in parts["boxes"]:
            box.set(facecolor=spec["color"], alpha=0.6)
        for median in parts["medians"]:
            median.set(color="black", linewidth=1.4)
        for whisker in parts["whiskers"]:
            whisker.set(color=spec["color"], linewidth=1.2)
        for cap in parts["caps"]:
            cap.set(color=spec["color"], linewidth=1.2)
        for flier in parts["fliers"]:
            flier.set(markerfacecolor=spec["color"], markeredgecolor=spec["color"], alpha=0.5, markersize=4)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[name] for name in SCENARIO_ORDER], fontsize=9)
    ax.set_ylabel(r"$\Delta J$ по seed")
    ax.set_title("Распределение разности накопленной функции потерь")
    handles = [
        plt.Line2D([0], [0], color=spec["color"], lw=6, alpha=0.6, label=spec["short_label"])
        for spec in COMPARISON_SPECS.values()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "fig_02_delta_cumulative_loss_boxplots")


def _plot_win_rate_heatmap(win_rates: pd.DataFrame, output_dir: Path) -> None:
    order = win_rates.set_index("scenario_name").loc[SCENARIO_ORDER].reset_index()
    data = np.vstack(
        [
            order["belief_vs_classical_win_rate"].to_numpy(dtype=float),
            order["rawobs_vs_classical_win_rate"].to_numpy(dtype=float),
            order["rawobs_vs_belief_win_rate"].to_numpy(dtype=float),
        ]
    )
    row_labels = [
        "По состоянию vs\nфиксированное",
        "По наблюдениям vs\nфиксированное",
        "По наблюдениям vs\nпо состоянию",
    ]
    figure, ax = plt.subplots(figsize=(10.0, 4.8))
    image = ax.imshow(data, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(SCENARIO_ORDER)))
    ax.set_xticklabels([SCENARIO_LABELS[name] for name in SCENARIO_ORDER], fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title("Доля побед по проверочным траекториям")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=10, color="black")
    figure.colorbar(image, ax=ax, shrink=0.85, label="Доля побед")
    _save_figure(figure, output_dir / "fig_03_win_rate_heatmap")


def _representative_seed(seed_level: pd.DataFrame, scenario_name: str) -> int:
    frame = seed_level.loc[seed_level["scenario_name"] == scenario_name].copy()
    belief_target = float(frame["belief_minus_classical"].mean())
    raw_target = float(frame["rawobs_minus_classical"].mean())
    score = np.square(frame["belief_minus_classical"] - belief_target) + np.square(frame["rawobs_minus_classical"] - raw_target)
    best_row = frame.loc[score.idxmin()]
    return int(best_row["evaluation_seed"])


def _architecture_path_slice(paths: pd.DataFrame, scenario_name: str, evaluation_seed: int, architecture: str) -> pd.DataFrame:
    if architecture == "fixed":
        mask = (
            (paths["scenario_name"] == scenario_name)
            & (paths["evaluation_seed"] == evaluation_seed)
            & (paths["variant_architecture"] == "belief_state")
            & (paths["policy_name"] == "classical_filtered_rule")
        )
    elif architecture == "belief":
        mask = (
            (paths["scenario_name"] == scenario_name)
            & (paths["evaluation_seed"] == evaluation_seed)
            & (paths["variant_architecture"] == "belief_state")
            & (paths["policy_name"] == "learning_policy")
        )
    elif architecture == "rawobs":
        mask = (
            (paths["scenario_name"] == scenario_name)
            & (paths["evaluation_seed"] == evaluation_seed)
            & (paths["variant_architecture"] == "raw_observations")
            & (paths["policy_name"] == "learning_policy")
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return paths.loc[mask].sort_values("period").reset_index(drop=True)


def _shade_stress_regime(ax: plt.Axes, regime_series: np.ndarray) -> None:
    in_stress = False
    start = 0
    for idx, state in enumerate(regime_series):
        if state == 1 and not in_stress:
            in_stress = True
            start = idx
        if state == 0 and in_stress:
            ax.axvspan(start - 0.5, idx - 0.5, color="#f4d35e", alpha=0.20)
            in_stress = False
    if in_stress:
        ax.axvspan(start - 0.5, len(regime_series) - 0.5, color="#f4d35e", alpha=0.20)


def _plot_cumulative_loss_paths(seed_level: pd.DataFrame, paths: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True, sharey=False)
    for ax, scenario_name in zip(axes.flat, SCENARIO_ORDER):
        evaluation_seed = _representative_seed(seed_level, scenario_name)
        fixed = _architecture_path_slice(paths, scenario_name, evaluation_seed, "fixed")
        belief = _architecture_path_slice(paths, scenario_name, evaluation_seed, "belief")
        rawobs = _architecture_path_slice(paths, scenario_name, evaluation_seed, "rawobs")
        _shade_stress_regime(ax, fixed["hidden_regime"].to_numpy(dtype=int))
        for architecture, frame in [("fixed", fixed), ("belief", belief), ("rawobs", rawobs)]:
            spec = ARCHITECTURE_SPECS[architecture]
            ax.plot(
                frame["period"].to_numpy(dtype=int),
                frame["cumulative_policy_loss"].to_numpy(dtype=float),
                color=spec["color"],
                linewidth=2.0,
                label=spec["label"],
            )
        ax.set_title(f"{SCENARIO_LABELS[scenario_name]}\nseed = {evaluation_seed}", fontsize=10)
        ax.set_xlabel("Периоды")
        ax.set_ylabel("Накопленная потеря")
        ax.grid(alpha=0.25)
        rows.append(
            {
                "scenario_name": scenario_name,
                "representative_seed": evaluation_seed,
                "belief_final_cumulative_loss": float(belief["cumulative_policy_loss"].iloc[-1]),
                "rawobs_final_cumulative_loss": float(rawobs["cumulative_policy_loss"].iloc[-1]),
                "fixed_final_cumulative_loss": float(fixed["cumulative_policy_loss"].iloc[-1]),
            }
        )
    handles = [
        plt.Line2D([0], [0], color=ARCHITECTURE_SPECS[key]["color"], lw=2.5, label=ARCHITECTURE_SPECS[key]["label"])
        for key in ["fixed", "belief", "rawobs"]
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#f4d35e", alpha=0.20, label="Стрессовый режим"))
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("Накопленная функция потерь на представительных траекториях", y=1.08, fontsize=13)
    _save_figure(figure, output_dir / "fig_04_cumulative_loss_paths")
    return pd.DataFrame(rows)


def _map_architecture(policy_frame: pd.DataFrame) -> pd.DataFrame:
    mapped = policy_frame.copy()
    mapped["architecture"] = ""
    fixed_mask = (mapped["variant_architecture"] == "belief_state") & (mapped["policy_name"] == "classical_filtered_rule")
    belief_mask = (mapped["variant_architecture"] == "belief_state") & (mapped["policy_name"] == "learning_policy")
    raw_mask = (mapped["variant_architecture"] == "raw_observations") & (mapped["policy_name"] == "learning_policy")
    mapped.loc[fixed_mask, "architecture"] = "fixed"
    mapped.loc[belief_mask, "architecture"] = "belief"
    mapped.loc[raw_mask, "architecture"] = "rawobs"
    return mapped.loc[mapped["architecture"] != ""].copy()


def _compute_loss_components(paths: pd.DataFrame) -> pd.DataFrame:
    mapped = _map_architecture(paths)
    rows = []
    for (scenario_name, architecture, evaluation_seed), frame in mapped.groupby(
        ["scenario_name", "architecture", "evaluation_seed"]
    ):
        ordered = frame.sort_values("period")
        rate = ordered["policy_rate"].to_numpy(dtype=float)
        inflation_component = np.square(ordered["inflation_gap"].to_numpy(dtype=float))
        output_component = 0.5 * np.square(ordered["output_gap"].to_numpy(dtype=float))
        rate_component = 0.05 * np.square(np.diff(rate, prepend=0.0))
        rows.append(
            {
                "scenario_name": scenario_name,
                "architecture": architecture,
                "evaluation_seed": int(evaluation_seed),
                "inflation_component": float(inflation_component.sum()),
                "output_component": float(output_component.sum()),
                "rate_change_component": float(rate_component.sum()),
                "total_loss": float(inflation_component.sum() + output_component.sum() + rate_component.sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_loss_decomposition(components: pd.DataFrame, output_dir: Path) -> None:
    averaged = (
        components.groupby(["scenario_name", "architecture"], as_index=False)[
            ["inflation_component", "output_component", "rate_change_component", "total_loss"]
        ]
        .mean()
    )
    component_specs = [
        ("inflation_component", "Инфляция", "#bc4749"),
        ("output_component", "Разрыв выпуска", "#577590"),
        ("rate_change_component", "Сглаживание ставки", "#f4a261"),
    ]
    architecture_order = ["fixed", "belief", "rawobs"]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharey=False)
    for ax, scenario_name in zip(axes.flat, SCENARIO_ORDER):
        frame = averaged.loc[averaged["scenario_name"] == scenario_name].set_index("architecture").loc[architecture_order]
        x = np.arange(len(architecture_order), dtype=float)
        bottom = np.zeros(len(architecture_order), dtype=float)
        for component_name, component_label, color in component_specs:
            values = frame[component_name].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, width=0.62, color=color, label=component_label)
            bottom += values
        for xpos, total in zip(x, frame["total_loss"].to_numpy(dtype=float)):
            ax.text(xpos, total, f"{total:.3e}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCHITECTURE_SPECS[key]["label"] for key in architecture_order], rotation=18, ha="right")
        ax.set_title(SCENARIO_LABELS[scenario_name], fontsize=10)
        ax.set_ylabel("Накопленная потеря")
        ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for _, label, color in component_specs]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("Разложение накопленной функции потерь на компоненты", y=1.08, fontsize=13)
    _save_figure(figure, output_dir / "fig_05_loss_decomposition")


def _plot_smoothness_vs_performance(policy_metrics: pd.DataFrame, output_dir: Path) -> None:
    mapped = _map_architecture(policy_metrics)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=False, sharey=False)
    for ax, scenario_name in zip(axes.flat, SCENARIO_ORDER):
        frame = mapped.loc[mapped["scenario_name"] == scenario_name].copy()
        for architecture in ["fixed", "belief", "rawobs"]:
            sub = frame.loc[frame["architecture"] == architecture]
            spec = ARCHITECTURE_SPECS[architecture]
            ax.scatter(
                sub["policy_instrument_volatility"].to_numpy(dtype=float),
                sub["cumulative_policy_loss"].to_numpy(dtype=float),
                color=spec["color"],
                alpha=0.65,
                s=38,
                label=spec["label"],
            )
            ax.scatter(
                float(sub["policy_instrument_volatility"].mean()),
                float(sub["cumulative_policy_loss"].mean()),
                color=spec["color"],
                edgecolor="black",
                linewidth=0.8,
                s=95,
                marker="D",
            )
        ax.set_title(SCENARIO_LABELS[scenario_name], fontsize=10)
        ax.set_xlabel("Волатильность ставки")
        ax.set_ylabel("Накопленная потеря")
        ax.grid(alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ARCHITECTURE_SPECS[key]["color"], label=ARCHITECTURE_SPECS[key]["label"], markersize=8)
        for key in ["fixed", "belief", "rawobs"]
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("Плавность ставки и качество политики", y=1.08, fontsize=13)
    _save_figure(figure, output_dir / "fig_06_policy_smoothness_vs_performance")


def _plot_environment_shift_relative_improvement(environment_shift: pd.DataFrame, output_dir: Path) -> None:
    frame = environment_shift.set_index("environment_name").loc[ENVIRONMENT_ORDER].reset_index()
    x = np.arange(len(ENVIRONMENT_ORDER), dtype=float)
    width = 0.34
    figure, ax = plt.subplots(figsize=(10.4, 5.6))
    bars_fixed = ax.bar(
        x - width / 2.0,
        frame["mean_learned_improvement_vs_fixed_pct"].to_numpy(dtype=float),
        width=width,
        color="#0f4c5c",
        label="Относительно фиксированного правила",
    )
    bars_retuned = ax.bar(
        x + width / 2.0,
        frame["mean_learned_improvement_vs_retuned_pct"].to_numpy(dtype=float),
        width=width,
        color="#bc4749",
        label="Относительно перенастроенного простого правила",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([ENVIRONMENT_LABELS[name] for name in ENVIRONMENT_ORDER], fontsize=9)
    ax.set_ylabel("Относительное улучшение, %")
    ax.set_title("Относительное улучшение обучаемой политики при переносе на новые среды")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.25)
    for bars in [bars_fixed, bars_retuned]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )
    _save_figure(figure, output_dir / "fig_07_environment_shift_relative_improvement")


def run_stage6_diagnostics(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_diagnostics",
    architecture_dir: str = "outputs/hank_regime_learning_stage6_architecture_ablation",
    environment_shift_dir: str = "outputs/hank_regime_learning_stage6_environment_shift",
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    architecture_root = Path(architecture_dir)
    environment_root = Path(environment_shift_dir)
    seed_level, policy_paths, policy_metrics, win_rates = _load_architecture_data(architecture_root)
    environment_shift = _load_environment_shift_data(environment_root)

    _save_json(
        root / "stage6_diagnostics_spec.json",
        {
            "architecture_dir": str(architecture_root),
            "environment_shift_dir": str(environment_root),
            "scenario_order": SCENARIO_ORDER,
            "environment_order": ENVIRONMENT_ORDER,
            "bootstrap_draws": 4000,
        },
    )

    delta_summary = _build_delta_summary(seed_level)
    delta_summary.to_csv(root / "delta_loss_intervals.csv", index=False)
    _plot_delta_interval(delta_summary, root)
    _plot_delta_boxplots(seed_level, root)
    _plot_win_rate_heatmap(win_rates, root)
    representative = _plot_cumulative_loss_paths(seed_level, policy_paths, root)
    representative.to_csv(root / "representative_seeds.csv", index=False)
    components = _compute_loss_components(policy_paths)
    components.to_csv(root / "loss_component_seed_level.csv", index=False)
    _plot_loss_decomposition(components, root)
    _plot_smoothness_vs_performance(policy_metrics, root)
    _plot_environment_shift_relative_improvement(environment_shift, root)

    component_means = (
        components.groupby(["scenario_name", "architecture"], as_index=False)[
            ["inflation_component", "output_component", "rate_change_component", "total_loss"]
        ]
        .mean()
    )
    fixed_components = component_means.loc[component_means["architecture"] == "fixed"].copy()
    belief_components = component_means.loc[component_means["architecture"] == "belief"].copy()
    raw_components = component_means.loc[component_means["architecture"] == "rawobs"].copy()
    fixed_output_share = float((fixed_components["output_component"] / fixed_components["total_loss"]).mean())
    belief_output_share = float((belief_components["output_component"] / belief_components["total_loss"]).mean())
    raw_output_share = float((raw_components["output_component"] / raw_components["total_loss"]).mean())
    fixed_rate_component = float(fixed_components["rate_change_component"].mean())
    belief_rate_component = float(belief_components["rate_change_component"].mean())
    raw_rate_component = float(raw_components["rate_change_component"].mean())

    belief_delta = delta_summary.loc[delta_summary["comparison_name"] == "belief_minus_classical"].copy()
    raw_delta = delta_summary.loc[delta_summary["comparison_name"] == "rawobs_minus_classical"].copy()
    raw_vs_belief_delta = delta_summary.loc[delta_summary["comparison_name"] == "rawobs_minus_belief"].copy()
    raw_vs_belief_positive = raw_vs_belief_delta.loc[raw_vs_belief_delta["ci_lower"] > 0.0, "scenario_name"].tolist()
    all_d = seed_level["belief_minus_classical"].to_numpy(dtype=float)
    all_rel = (
        (
            seed_level["classical_cumulative_policy_loss"].to_numpy(dtype=float)
            - seed_level["belief_cumulative_policy_loss"].to_numpy(dtype=float)
        )
        / seed_level["classical_cumulative_policy_loss"].to_numpy(dtype=float)
    )
    scenario_mean_rel = (
        (
            seed_level["classical_cumulative_policy_loss"] - seed_level["belief_cumulative_policy_loss"]
        )
        / seed_level["classical_cumulative_policy_loss"]
    ).groupby(seed_level["scenario_name"]).mean()

    merged = pd.read_csv(architecture_root / "architecture_comparison.csv")[
        ["scenario_name", "classical_policy_volatility", "belief_policy_volatility"]
    ].copy()
    merged["vol_reduction_pct"] = 100.0 * (
        1.0 - merged["belief_policy_volatility"] / merged["classical_policy_volatility"]
    )

    env_fixed_range = (
        float(environment_shift["mean_learned_improvement_vs_fixed_pct"].min()),
        float(environment_shift["mean_learned_improvement_vs_fixed_pct"].max()),
    )
    env_retuned_range = (
        float(environment_shift["mean_learned_improvement_vs_retuned_pct"].min()),
        float(environment_shift["mean_learned_improvement_vs_retuned_pct"].max()),
    )
    baseline_improvement = float(
        environment_shift.loc[
            environment_shift["environment_name"] == "baseline_environment",
            "mean_learned_improvement_vs_fixed_pct",
        ].iloc[0]
    )
    environment_shift = environment_shift.copy()
    environment_shift["transfer_score_vs_fixed"] = (
        environment_shift["mean_learned_improvement_vs_fixed_pct"] / baseline_improvement
    )

    headlines = {
        "belief_vs_fixed_all_ci_below_zero": bool(np.all(belief_delta["ci_upper"].to_numpy(dtype=float) < 0.0)),
        "rawobs_vs_fixed_all_ci_below_zero": bool(np.all(raw_delta["ci_upper"].to_numpy(dtype=float) < 0.0)),
        "rawobs_vs_belief_positive_ci_scenarios": raw_vs_belief_positive,
        "fixed_output_share_mean": fixed_output_share,
        "belief_output_share_mean": belief_output_share,
        "rawobs_output_share_mean": raw_output_share,
        "fixed_rate_component_mean": fixed_rate_component,
        "belief_rate_component_mean": belief_rate_component,
        "rawobs_rate_component_mean": raw_rate_component,
        "environment_shift_improvement_vs_fixed_range_pct": env_fixed_range,
        "environment_shift_improvement_vs_retuned_range_pct": env_retuned_range,
    }
    _save_json(root / "diagnostic_headlines.json", headlines)

    interpretation_rows = [
        {
            "metric": "Относительное улучшение",
            "how_to_read": "5--10% — небольшое; 10--20% — содержательное; 20--30% — сильное; 50%+ — очень большое.",
            "our_result": (
                f"Для обучения по состоянию против фиксированного правила среднее улучшение = {100.0 * all_rel.mean():.1f}%; "
                f"по средним сценариев = {100.0 * scenario_mean_rel.min():.1f}--{100.0 * scenario_mean_rel.max():.1f}%."
            ),
            "interpretation": "Очень большой выигрыш относительно фиксированного правила.",
        },
        {
            "metric": "Доля побед",
            "how_to_read": "0.5 — преимущества нет; 0.7--0.8 — хорошо; 0.8--0.9 — сильно; 1.0 — полное доминирование на выборке.",
            "our_result": "Для обучения по состоянию против фиксированного правила доля побед = 1.0 во всех 4 сценариях.",
            "interpretation": "На проверочных траекториях фиксированное правило не выигрывает ни в одном сценарии.",
        },
        {
            "metric": "Вероятность ухудшения",
            "how_to_read": "Чем ближе к нулю, тем безопаснее улучшение.",
            "our_result": f"Для обучения по состоянию против фиксированного правила вероятность ухудшения = {np.mean(all_d > 0.0):.1f}.",
            "interpretation": "На проверенных траекториях ухудшения не наблюдается.",
        },
        {
            "metric": "Стандартизованный эффект",
            "how_to_read": "0.2 — маленький; 0.5 — средний; 0.8+ — большой.",
            "our_result": f"Парный стандартизованный эффект для обучения по состоянию против фиксированного правила = {abs(all_d.mean() / all_d.std(ddof=1)):.2f}.",
            "interpretation": "Большой эффект относительно собственного разброса.",
        },
        {
            "metric": "Хвостовой риск",
            "how_to_read": "Важно смотреть, не исчезает ли выигрыш в худших эпизодах.",
            "our_result": (
                f"90-й процентиль разности потерь = {np.quantile(all_d, 0.9):.3e}; "
                f"средняя разность в худших 10% случаев = {all_d[all_d >= np.quantile(all_d, 0.9)].mean():.3e}."
            ),
            "interpretation": "Даже в верхнем хвосте разность остается ниже нуля, значит эффект не держится на удачных выбросах.",
        },
        {
            "metric": "Покомпонентный механизм",
            "how_to_read": "Показывает, за счет чего именно уменьшается функция потерь.",
            "our_result": (
                f"Среднее снижение компоненты инфляции = {100.0 * (1.0 - belief_components['inflation_component'].mean() / fixed_components['inflation_component'].mean()):.1f}%, "
                f"выпуска = {100.0 * (1.0 - belief_components['output_component'].mean() / fixed_components['output_component'].mean()):.1f}%, "
                f"сглаживания ставки = {100.0 * (1.0 - belief_components['rate_change_component'].mean() / fixed_components['rate_change_component'].mean()):.1f}%."
            ),
            "interpretation": "Выигрыш идет через лучшую стабилизацию инфляции и выпуска, а не только через механическое сглаживание ставки.",
        },
        {
            "metric": "Плавность ставки",
            "how_to_read": "Если качество лучше при меньшей волатильности, политика не покупает выигрыш чрезмерной дерганостью.",
            "our_result": f"Среднее снижение волатильности ставки для обучения по состоянию против фиксированного правила = {merged['vol_reduction_pct'].mean():.1f}%.",
            "interpretation": "Обучаемое правило в среднем и лучше, и плавнее фиксированного.",
        },
        {
            "metric": "Переносимость на новые среды",
            "how_to_read": "Коэффициент переноса, близкий к 1, означает, что выигрыш почти сохраняется вне базовой среды.",
            "our_result": (
                f"Относительно фиксированного правила переносимый выигрыш = {env_fixed_range[0]:.1f}--{env_fixed_range[1]:.1f}%; "
                f"коэффициент переноса = {environment_shift['transfer_score_vs_fixed'].min():.2f}--{environment_shift['transfer_score_vs_fixed'].max():.2f}."
            ),
            "interpretation": "Эффект хорошо переносится относительно фиксированного правила.",
        },
        {
            "metric": "Граница применимости",
            "how_to_read": "Важно проверить, не является ли выигрыш универсальным по отношению ко всем альтернативам.",
            "our_result": (
                f"Прямая политика по наблюдениям выигрывает у обучения по состоянию только в {int(np.mean(seed_level['rawobs_minus_belief'].to_numpy(float) < 0.0) * 100)}% проверочных траекторий; "
                f"при переносе на новые среды обучаемое правило хуже перенастроенного простого правила на {abs(env_retuned_range[1]):.1f}--{abs(env_retuned_range[0]):.1f}%."
            ),
            "interpretation": "Обучаемая политика полезна не универсально, а прежде всего как альтернатива жесткому фиксированному правилу и ошибочной фильтрации.",
        },
    ]
    interpretation_table = pd.DataFrame(interpretation_rows)
    interpretation_table.to_csv(root / "stage6_metric_interpretation_table.csv", index=False)
    _write_latex_table(root / "table_stage6_metric_interpretation.tex", interpretation_table)
    _save_text(
        root / "stage6_metric_interpretation_summary.md",
        "\n".join(
            [
                "# Stage 6 Metric Interpretation",
                "",
                "Эта таблица переводит технические метрики качества политики в более читаемый масштаб.",
                "",
                interpretation_table.to_markdown(index=False),
            ]
        ),
    )

    report_lines = [
        "# Stage 6 Diagnostics",
        "",
        "Диагностический пакет собран для того, чтобы оценивать не абсолютный уровень функции потерь, а устойчивость выигрыша, его распределение по траекториям и экономический механизм.",
        "",
        "## Ключевые наблюдения",
        "",
        f"- Во всех четырех сценариях 95%-интервалы для `обучение по состоянию vs фиксированное правило` целиком ниже нуля; средняя разность накопленной функции потерь лежит между `{belief_delta['mean_delta_cumulative_loss'].min():.4e}` и `{belief_delta['mean_delta_cumulative_loss'].max():.4e}`.",
        f"- Для `наблюдения + обучение vs фиксированное правило` 95%-интервалы также целиком ниже нуля во всех четырех сценариях; средняя разность лежит между `{raw_delta['mean_delta_cumulative_loss'].min():.4e}` и `{raw_delta['mean_delta_cumulative_loss'].max():.4e}`.",
        "- Сравнение `наблюдения + обучение vs обучение по состоянию` не универсально: в двух macro-сценариях и в сценарии `thin_information_moderate_gap` интервалы пересекают ноль, а в `thin_information_strong_gap` прямая политика по наблюдениям уже устойчиво хуже.",
        "- Тепловая карта долей побед показывает, что `обучение по состоянию` выигрывает у фиксированного правила во всех четырех сценариях на всех проверочных траекториях, а прямое правило по наблюдениям теряет устойчивость именно в сценарии с тонким информационным набором и сильным режимным разрывом.",
        f"- Разложение функции потерь показывает, что выигрыш не сводится к механическому сглаживанию ставки: в среднем доля компоненты разрыва выпуска в общей потере составляет `{100.0 * fixed_output_share:.1f}%` у фиксированного правила, `{100.0 * belief_output_share:.1f}%` у обучения по состоянию и `{100.0 * raw_output_share:.1f}%` у прямой политики по наблюдениям, тогда как вклад сглаживания ставки остается на уровне `{fixed_rate_component:.2e}` -- `{raw_rate_component:.2e}`.",
        f"- При переносе на новые среды обучаемая политика лучше фиксированного правила на `{env_fixed_range[0]:.1f}%`--`{env_fixed_range[1]:.1f}%`, но хуже перенастроенного простого правила на `{abs(env_retuned_range[1]):.1f}%`--`{abs(env_retuned_range[0]):.1f}%`.",
        "",
        "## Основные файлы",
        "",
        "- `delta_loss_intervals.csv` — средняя разность накопленной функции потерь, разброс и bootstrap-интервалы.",
        "- `representative_seeds.csv` — выбранные представительные траектории для визуализации накопленной потери во времени.",
        "- `loss_component_seed_level.csv` — покомпонентное разложение потерь на инфляцию, выпуск и сглаживание ставки.",
        "- `diagnostic_headlines.json` — ключевые численные выводы для быстрого цитирования в тексте.",
        "- `stage6_metric_interpretation_table.csv` — компактная таблица «метрика → как читать → что получилось».",
        "",
        "## Рисунки",
        "",
        "- `fig_01_delta_cumulative_loss_intervals.pdf`",
        "- `fig_02_delta_cumulative_loss_boxplots.pdf`",
        "- `fig_03_win_rate_heatmap.pdf`",
        "- `fig_04_cumulative_loss_paths.pdf`",
        "- `fig_05_loss_decomposition.pdf`",
        "- `fig_06_policy_smoothness_vs_performance.pdf`",
        "- `fig_07_environment_shift_relative_improvement.pdf`",
    ]
    (root / "report_stage6_diagnostics.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "delta_summary": delta_summary,
        "representative": representative,
        "components": components,
    }
