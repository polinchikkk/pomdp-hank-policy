from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "macro_core_moderate_gap": "Базовый макроэкономический набор × умеренная разделимость режимов",
    "macro_core_strong_gap": "Базовый макроэкономический набор × высокая разделимость режимов",
    "thin_information_moderate_gap": "Ограниченный информационный набор × умеренная разделимость режимов",
    "thin_information_strong_gap": "Ограниченный информационный набор × высокая разделимость режимов",
}

POLICY_ORDER = [
    "full_information_classical",
    "filtered_classical",
    "filtered_selected",
    "raw_selected",
]

POLICY_LABELS = {
    "full_information_classical": "Правило при полной информации",
    "filtered_classical": "Классическое правило по оценённому состоянию",
    "filtered_selected": "Отобранное правило по оценённому состоянию",
    "raw_selected": "Отобранное правило по наблюдаемым переменным",
}

COMPARISON_LABELS = {
    "filtered_selected_minus_filtered_classical": "Отобранное правило по оценённому состоянию минус классическое правило по оценённому состоянию",
    "filtered_selected_minus_raw_selected": "Отобранное правило по оценённому состоянию минус отобранное правило по наблюдаемым переменным",
    "filtered_selected_minus_full_information_classical": "Отобранное правило по оценённому состоянию минус правило при полной информации",
    "raw_selected_minus_full_information_classical": "Отобранное правило по наблюдаемым переменным минус правило при полной информации",
    "filtered_classical_minus_full_information_classical": "Классическое правило по оценённому состоянию минус правило при полной информации",
}


def _save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _bootstrap_ci(values: np.ndarray, *, seed: int = 1234, draws: int = 4000) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _policy_rows_for_variant(metrics: pd.DataFrame, *, scenario_name: str) -> list[dict[str, float | str]]:
    rows = []
    mapping = {
        "full_information_classical": "full_information_rule",
        "filtered_classical": "classical_filtered_rule",
        "filtered_selected": "learning_policy",
    }
    for output_policy_name, internal_name in mapping.items():
        frame = metrics[metrics["policy_name"] == internal_name].copy()
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "policy_class": output_policy_name,
                "policy_label": POLICY_LABELS[output_policy_name],
                "mean_policy_loss": float(frame["mean_policy_loss"].mean()),
                "cumulative_policy_loss": float(frame["cumulative_policy_loss"].mean()),
                "policy_rate_rmse": float(frame["policy_rate_rmse"].mean()),
                "policy_instrument_volatility": float(frame["policy_instrument_volatility"].mean()),
                "unstable": int(frame["unstable"].max()),
            }
        )
    return rows


def _seed_loss_frame(
    belief_metrics: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    *,
    scenario_name: str,
) -> pd.DataFrame:
    belief = belief_metrics.pivot_table(
        index="evaluation_seed",
        columns="policy_name",
        values="cumulative_policy_loss",
        aggfunc="first",
    )
    raw = raw_metrics.pivot_table(
        index="evaluation_seed",
        columns="policy_name",
        values="cumulative_policy_loss",
        aggfunc="first",
    )
    frame = pd.DataFrame(
        {
            "scenario_name": scenario_name,
            "scenario_label": SCENARIO_LABELS[scenario_name],
            "evaluation_seed": belief.index.to_numpy(dtype=int),
            "full_information_classical": belief["full_information_rule"].to_numpy(dtype=float),
            "filtered_classical": belief["classical_filtered_rule"].to_numpy(dtype=float),
            "filtered_selected": belief["learning_policy"].to_numpy(dtype=float),
            "raw_selected": raw["learning_policy"].to_numpy(dtype=float),
        }
    )
    return frame


def _comparison_rows(seed_frame: pd.DataFrame) -> list[dict[str, float | str]]:
    comparisons = {
        "filtered_selected_minus_filtered_classical": (
            seed_frame["filtered_selected"].to_numpy(dtype=float)
            - seed_frame["filtered_classical"].to_numpy(dtype=float)
        ),
        "filtered_selected_minus_raw_selected": (
            seed_frame["filtered_selected"].to_numpy(dtype=float)
            - seed_frame["raw_selected"].to_numpy(dtype=float)
        ),
        "filtered_selected_minus_full_information_classical": (
            seed_frame["filtered_selected"].to_numpy(dtype=float)
            - seed_frame["full_information_classical"].to_numpy(dtype=float)
        ),
        "raw_selected_minus_full_information_classical": (
            seed_frame["raw_selected"].to_numpy(dtype=float)
            - seed_frame["full_information_classical"].to_numpy(dtype=float)
        ),
        "filtered_classical_minus_full_information_classical": (
            seed_frame["filtered_classical"].to_numpy(dtype=float)
            - seed_frame["full_information_classical"].to_numpy(dtype=float)
        ),
    }
    rows = []
    for comparison_name, deltas in comparisons.items():
        if "minus_filtered_classical" in comparison_name:
            benchmark = seed_frame["filtered_classical"].to_numpy(dtype=float)
        elif "minus_raw_selected" in comparison_name:
            benchmark = seed_frame["raw_selected"].to_numpy(dtype=float)
        else:
            benchmark = seed_frame["full_information_classical"].to_numpy(dtype=float)
        ci_lower, ci_upper = _bootstrap_ci(deltas)
        rows.append(
            {
                "scenario_name": seed_frame["scenario_name"].iloc[0],
                "scenario_label": seed_frame["scenario_label"].iloc[0],
                "comparison_name": comparison_name,
                "comparison_label": COMPARISON_LABELS[comparison_name],
                "mean_delta_cumulative_loss": float(deltas.mean()),
                "std_delta_cumulative_loss": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "win_rate": float(np.mean(deltas < 0.0)),
                "probability_of_degradation": float(np.mean(deltas > 0.0)),
                "mean_relative_improvement_pct": float(100.0 * np.mean(-deltas / benchmark)),
                "num_seeds": int(deltas.size),
            }
        )
    return rows


def _component_rows(
    belief_paths: pd.DataFrame,
    raw_paths: pd.DataFrame,
    *,
    scenario_name: str,
) -> list[dict[str, float | str]]:
    def with_components(frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        enriched["inflation_loss"] = np.square(enriched["inflation_gap"].to_numpy(dtype=float))
        enriched["output_gap_loss"] = 0.5 * np.square(enriched["output_gap"].to_numpy(dtype=float))
        enriched["rate_change_loss"] = (
            enriched["policy_loss"].to_numpy(dtype=float)
            - enriched["inflation_loss"].to_numpy(dtype=float)
            - enriched["output_gap_loss"].to_numpy(dtype=float)
        )
        return enriched

    belief_paths = with_components(belief_paths)
    raw_paths = with_components(raw_paths)

    belief = (
        belief_paths.groupby(["evaluation_seed", "policy_name"])[["inflation_loss", "output_gap_loss", "rate_change_loss"]]
        .sum()
        .reset_index()
    )
    raw = (
        raw_paths.groupby(["evaluation_seed", "policy_name"])[["inflation_loss", "output_gap_loss", "rate_change_loss"]]
        .sum()
        .reset_index()
    )

    def extract(frame: pd.DataFrame, policy_name: str) -> pd.DataFrame:
        sub = frame[frame["policy_name"] == policy_name].copy()
        return sub.rename(
            columns={
                "inflation_loss": f"{policy_name}_inflation_loss",
                "output_gap_loss": f"{policy_name}_output_gap_loss",
                "rate_change_loss": f"{policy_name}_rate_change_loss",
            }
        ).drop(columns="policy_name")

    merged = extract(belief, "learning_policy")
    merged = merged.merge(extract(belief, "classical_filtered_rule"), on="evaluation_seed")
    merged = merged.merge(extract(belief, "full_information_rule"), on="evaluation_seed")
    merged = merged.merge(extract(raw, "learning_policy"), on="evaluation_seed", suffixes=("", "_raw"))
    merged = merged.rename(
        columns={
            "learning_policy_inflation_loss_raw": "raw_learning_policy_inflation_loss",
            "learning_policy_output_gap_loss_raw": "raw_learning_policy_output_gap_loss",
            "learning_policy_rate_change_loss_raw": "raw_learning_policy_rate_change_loss",
        }
    )

    comparisons = {
        "filtered_selected_vs_filtered_classical": (
            "learning_policy",
            "classical_filtered_rule",
        ),
        "filtered_selected_vs_raw_selected": (
            "learning_policy",
            "raw_learning_policy",
        ),
        "filtered_selected_vs_full_information_classical": (
            "learning_policy",
            "full_information_rule",
        ),
    }
    rows = []
    for comparison_name, (left, right) in comparisons.items():
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "comparison_name": comparison_name,
                "delta_inflation_loss": float((merged[f"{left}_inflation_loss"] - merged[f"{right}_inflation_loss"]).mean()),
                "delta_output_gap_loss": float((merged[f"{left}_output_gap_loss"] - merged[f"{right}_output_gap_loss"]).mean()),
                "delta_rate_change_loss": float((merged[f"{left}_rate_change_loss"] - merged[f"{right}_rate_change_loss"]).mean()),
            }
        )
    return rows


def _latex_escape(text: str) -> str:
    return text.replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")


def _write_latex_table(path: Path, table: pd.DataFrame, columns: list[str], headers: list[str]) -> None:
    spec = "l" + "c" * (len(columns) - 1)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                cells.append(_latex_escape(value))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _save_text(path, "\n".join(lines))


def _plot_value_of_filtering(comparisons: pd.DataFrame, path: Path) -> None:
    data = comparisons[comparisons["comparison_name"] == "filtered_selected_minus_raw_selected"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    positions = np.arange(len(data))
    means = -data["mean_delta_cumulative_loss"].to_numpy(dtype=float)
    lower = means - (-data["ci_upper"].to_numpy(dtype=float))
    upper = (-data["ci_lower"].to_numpy(dtype=float)) - means
    ax.errorbar(positions, means, yerr=np.vstack([lower, upper]), fmt="o", capsize=4, color="#0b6e4f")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            "Базовый\nмакронабор,\nумеренная",
            "Базовый\nмакронабор,\nвысокая",
            "Ограниченный\nнабор,\nумеренная",
            "Ограниченный\nнабор,\nвысокая",
        ]
    )
    ax.set_ylabel("Снижение накопленной потери")
    ax.set_title("Выигрыш от оценивания скрытого состояния")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_gap_to_oracle(main_table: pd.DataFrame, path: Path) -> None:
    pivot = main_table.pivot(index="scenario_name", columns="policy_class", values="gap_to_oracle")
    order = list(SCENARIO_LABELS)
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(order))
    width = 0.22
    series = [
        ("filtered_classical", "#bb3e03"),
        ("filtered_selected", "#0b6e4f"),
        ("raw_selected", "#3a86ff"),
    ]
    for idx, (policy_class, color) in enumerate(series):
        ax.bar(x + (idx - 1) * width, pivot[policy_class].to_numpy(dtype=float), width=width, label=POLICY_LABELS[policy_class], color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(["Базовый\nмакронабор,\nумеренная", "Базовый\nмакронабор,\nвысокая", "Ограниченный\nнабор,\nумеренная", "Ограниченный\nнабор,\nвысокая"])
    ax.set_ylabel("Разрыв до полной информации")
    ax.set_title("Потери из-за неполной наблюдаемости")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_pairwise_scatter(seed_losses: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0), sharex=False, sharey=False)
    for ax, scenario_name in zip(axes.ravel(), SCENARIO_LABELS):
        frame = seed_losses[seed_losses["scenario_name"] == scenario_name].copy()
        x = frame["raw_selected"].to_numpy(dtype=float)
        y = frame["filtered_selected"].to_numpy(dtype=float)
        bound = max(x.max(), y.max()) * 1.05
        ax.scatter(x, y, color="#0b6e4f", alpha=0.85)
        ax.plot([0.0, bound], [0.0, bound], color="black", linewidth=0.8, linestyle="--")
        ax.set_title(
            {
                "macro_core_moderate_gap": "Базовый набор,\nумеренная разделимость",
                "macro_core_strong_gap": "Базовый набор,\nвысокая разделимость",
                "thin_information_moderate_gap": "Ограниченный набор,\nумеренная разделимость",
                "thin_information_strong_gap": "Ограниченный набор,\nвысокая разделимость",
            }[scenario_name]
        )
        ax.set_xlabel("Потери: правило по наблюдаемым переменным")
        ax.set_ylabel("Потери: правило по оценённому состоянию")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _build_value_summary_table(wide: pd.DataFrame) -> pd.DataFrame:
    table = wide[
        [
            "scenario_label",
            "filtered_selected_vs_raw_pct",
            "filtered_selected_vs_classical_pct",
            "full_information_gap_filtered_selected",
        ]
    ].copy()
    table = table.rename(
        columns={
            "scenario_label": "Сценарий",
            "filtered_selected_vs_raw_pct": "Выигрыш от оценивания скрытого состояния, %",
            "filtered_selected_vs_classical_pct": "Выигрыш от гибкости правила, %",
            "full_information_gap_filtered_selected": "Разрыв до правила при полной информации",
        }
    )
    return table


def _build_text_block(
    main_table: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    fs_vs_raw = comparisons[comparisons["comparison_name"] == "filtered_selected_minus_raw_selected"].copy()

    classical_range = (
        100.0
        * (
            main_table[main_table["policy_class"] == "filtered_classical"]["cumulative_policy_loss"].to_numpy(dtype=float)
            - main_table[main_table["policy_class"] == "filtered_selected"]["cumulative_policy_loss"].to_numpy(dtype=float)
        )
        / main_table[main_table["policy_class"] == "filtered_classical"]["cumulative_policy_loss"].to_numpy(dtype=float)
    )
    filtering_range = fs_vs_raw["mean_relative_improvement_pct"].to_numpy(dtype=float)

    return rf"""
\section{{Основная матрица сравнений}}

Центральная постановка работы строится вокруг четырех чисто монетарных вариантов денежно-кредитного правила: правила при полной информации; классического правила по оценённому состоянию; отобранного правила по оценённому состоянию; и отобранного правила по наблюдаемым переменным без явного этапа фильтрации. Такая матрица позволяет отдельно оценить ценность информации, выигрыш от оценивания скрытого состояния и выигрыш от более гибкой формы правила.

По четырем базовым сценариям --- базовый макроэкономический набор и ограниченный информационный набор, каждый в сочетании с умеренной или высокой разделимостью режимов --- отобранное правило по оценённому состоянию уменьшает накопленную функцию потерь относительно классического правила по оценённому состоянию на {classical_range.min():.1f}--{classical_range.max():.1f}\%. При этом выигрыш от оценивания скрытого состояния зависит от информационного режима. В сценариях с базовым макроэкономическим набором правило по наблюдаемым переменным и правило по оценённому состоянию почти не различаются, а при высокой разделимости режимов прямое правило по наблюдаемым данным даже слегка лучше. Напротив, при ограниченном информационном наборе преимущество правила по оценённому состоянию становится выраженным: относительный выигрыш по сравнению с правилом, использующим только наблюдаемые переменные, достигает {filtering_range.max():.1f}\%, а в сценарии ``ограниченный информационный набор × высокая разделимость режимов'' доверительный интервал для разницы потерь целиком лежит ниже нуля в пользу фильтрации.

Правило при полной информации используется здесь не как глобальная верхняя граница, а как внутренний ориентир внутри одной и той же семьи правил. Сопоставление с ним показывает, насколько дорого обходится потеря информации и насколько этот разрыв удается сократить за счет более качественного представления состояния. Тем самым главный результат этапа 6 связан не с усложнением пространства действий, а с тем, какую информацию регулятор действительно получает и как она переводится в решение по ставке.

\begin{{table}}[htbp]
\centering
\small
\caption{{Основная матрица сравнений денежно-кредитных правил}}
\label{{tab:stage6_core_main_matrix}}
\input{{outputs/hank_regime_learning_stage6_core_matrix/table_stage6_core_main_results.tex}}
\end{{table}}

\begin{{table}}[htbp]
\centering
\small
\caption{{Выигрыш от оценивания скрытого состояния, гибкости правила и разрыв до полной информации}}
\label{{tab:stage6_core_values}}
\input{{outputs/hank_regime_learning_stage6_core_matrix/table_stage6_core_value_summary.tex}}
\end{{table}}
""".strip()


def run_core_matrix(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_core_matrix",
    architecture_dir: str = "outputs/hank_regime_learning_stage6_architecture_ablation",
) -> dict[str, pd.DataFrame | str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    architecture_root = Path(architecture_dir)
    summary_rows: list[dict[str, float | str]] = []
    seed_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, float | str]] = []
    component_rows: list[dict[str, float | str]] = []

    for scenario_name in SCENARIO_LABELS:
        belief_variant = architecture_root / f"{scenario_name}_belief_state"
        raw_variant = architecture_root / f"{scenario_name}_raw_observations"
        belief_metrics = pd.read_csv(belief_variant / "policy_metrics.csv")
        raw_metrics = pd.read_csv(raw_variant / "policy_metrics.csv")
        belief_paths = pd.read_csv(belief_variant / "policy_paths.csv")
        raw_paths = pd.read_csv(raw_variant / "policy_paths.csv")

        summary_rows.extend(_policy_rows_for_variant(belief_metrics, scenario_name=scenario_name))
        raw_selected = raw_metrics[raw_metrics["policy_name"] == "learning_policy"].copy()
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "policy_class": "raw_selected",
                "policy_label": POLICY_LABELS["raw_selected"],
                "mean_policy_loss": float(raw_selected["mean_policy_loss"].mean()),
                "cumulative_policy_loss": float(raw_selected["cumulative_policy_loss"].mean()),
                "policy_rate_rmse": float(raw_selected["policy_rate_rmse"].mean()),
                "policy_instrument_volatility": float(raw_selected["policy_instrument_volatility"].mean()),
                "unstable": int(raw_selected["unstable"].max()),
            }
        )

        seed_frame = _seed_loss_frame(belief_metrics, raw_metrics, scenario_name=scenario_name)
        seed_frames.append(seed_frame)
        comparison_rows.extend(_comparison_rows(seed_frame))
        component_rows.extend(_component_rows(belief_paths, raw_paths, scenario_name=scenario_name))

    main_table = pd.DataFrame(summary_rows).sort_values(["scenario_name", "policy_class"]).reset_index(drop=True)
    seed_losses = pd.concat(seed_frames, ignore_index=True)
    comparisons = pd.DataFrame(comparison_rows).sort_values(["scenario_name", "comparison_name"]).reset_index(drop=True)
    components = pd.DataFrame(component_rows).sort_values(["scenario_name", "comparison_name"]).reset_index(drop=True)

    oracle_lookup = main_table[main_table["policy_class"] == "full_information_classical"][["scenario_name", "cumulative_policy_loss"]].rename(
        columns={"cumulative_policy_loss": "full_information_classical_cumulative_policy_loss"}
    )
    main_table = main_table.merge(oracle_lookup, on="scenario_name", how="left")
    main_table["gap_to_oracle"] = (
        main_table["cumulative_policy_loss"] - main_table["full_information_classical_cumulative_policy_loss"]
    )

    main_table.to_csv(root / "main_policy_matrix.csv", index=False)
    seed_losses.to_csv(root / "trajectory_level_losses.csv", index=False)
    comparisons.to_csv(root / "core_comparisons.csv", index=False)
    components.to_csv(root / "loss_component_decomposition.csv", index=False)

    wide = main_table.pivot(index="scenario_name", columns="policy_class", values="cumulative_policy_loss").reset_index()
    wide["scenario_label"] = wide["scenario_name"].map(SCENARIO_LABELS)
    wide["filtered_selected_vs_classical_pct"] = 100.0 * (wide["filtered_classical"] - wide["filtered_selected"]) / wide["filtered_classical"]
    wide["filtered_selected_vs_raw_pct"] = 100.0 * (wide["raw_selected"] - wide["filtered_selected"]) / wide["raw_selected"]
    wide["full_information_gap_filtered_selected"] = wide["filtered_selected"] - wide["full_information_classical"]
    wide["full_information_gap_raw_selected"] = wide["raw_selected"] - wide["full_information_classical"]
    wide.to_csv(root / "core_headline_table.csv", index=False)
    value_summary = _build_value_summary_table(wide)
    value_summary.to_csv(root / "core_value_summary_table.csv", index=False)

    latex_table = wide[
        [
            "scenario_label",
            "full_information_classical",
            "filtered_classical",
            "filtered_selected",
            "raw_selected",
        ]
    ].copy()
    _write_latex_table(
        root / "table_stage6_core_main_results.tex",
        latex_table,
        columns=["scenario_label", "full_information_classical", "filtered_classical", "filtered_selected", "raw_selected"],
        headers=[
            "Сценарий",
            "Полная информация",
            "Классическое по оценённому состоянию",
            "Отобранное по оценённому состоянию",
            "Отобранное по наблюдаемым переменным",
        ],
    )
    _write_latex_table(
        root / "table_stage6_core_value_summary.tex",
        value_summary,
        columns=[
            "Сценарий",
            "Выигрыш от оценивания скрытого состояния, %",
            "Выигрыш от гибкости правила, %",
            "Разрыв до правила при полной информации",
        ],
        headers=[
            "Сценарий",
            "Выигрыш от оценивания скрытого состояния, \\%",
            "Выигрыш от гибкости правила, \\%",
            "Разрыв до полной информации",
        ],
    )

    comparison_table = comparisons[
        comparisons["comparison_name"].isin(
            [
                "filtered_selected_minus_filtered_classical",
                "filtered_selected_minus_raw_selected",
                "filtered_selected_minus_full_information_classical",
            ]
        )
    ][
        ["scenario_label", "comparison_label", "mean_delta_cumulative_loss", "ci_lower", "ci_upper", "win_rate"]
    ].copy()
    _write_latex_table(
        root / "table_stage6_core_comparisons.tex",
        comparison_table,
        columns=["scenario_label", "comparison_label", "mean_delta_cumulative_loss", "ci_lower", "ci_upper", "win_rate"],
        headers=["Сценарий", "Сравнение", "Средняя разность", "Нижняя граница ДИ", "Верхняя граница ДИ", "Доля выигрышных траекторий"],
    )

    _plot_value_of_filtering(comparisons, figures_dir / "fig_01_value_of_filtering")
    _plot_gap_to_oracle(main_table, figures_dir / "fig_02_gap_to_oracle")
    _plot_pairwise_scatter(seed_losses, figures_dir / "fig_03_filtered_vs_raw_scatter")

    text_block = _build_text_block(main_table, comparisons)
    _save_text(root / "stage6_core_text_blocks.tex", text_block + "\n")
    _save_text(
        root / "report_stage6_core_matrix.md",
        "\n".join(
            [
                "# Основная матрица сравнений этапа 6",
                "",
                "## Главная рамка",
                "",
                "Основной текст теперь центрируется на чистой матрице монетарных сравнений: правило при полной информации, классическое правило по оценённому состоянию, отобранное правило по оценённому состоянию и отобранное правило по наблюдаемым переменным.",
                "",
                "## Главный вывод",
                "",
                text_block,
            ]
        ),
    )

    return {
        "main_table": main_table,
        "seed_losses": seed_losses,
        "comparisons": comparisons,
        "components": components,
        "text_block": text_block,
    }
