from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCENARIO_LABELS = {
    "baseline": "Базовый сценарий",
    "high_aggregate_noise": "Высокий шум агрегатов",
    "high_heterogeneity": "Высокая неоднородность",
    "noisy_distributional_data": "Шумные распределительные данные",
}


def build_article_outputs(*, outputs_dir: Path, article_dir: Path) -> None:
    tables_dir = article_dir / "tables"
    figures_dir = article_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenario_summary = pd.read_csv(outputs_dir / "exp02_distributional_value" / "scenario_distributional_value.csv")
    sufficiency = pd.read_csv(outputs_dir / "exp03_sufficiency_gap" / "sufficiency_gap_summary.csv")
    individual_baseline = pd.read_csv(outputs_dir / "exp04_individual_stats" / "baseline" / "individual_stat_value.csv")
    grid_path = outputs_dir / "exp06_distributional_value_grid" / "distributional_value_grid.csv"
    grid = pd.read_csv(grid_path) if grid_path.exists() else None
    policy_robustness_path = outputs_dir / "exp05_policy_class_robustness" / "policy_class_robustness_summary.csv"
    policy_robustness = pd.read_csv(policy_robustness_path) if policy_robustness_path.exists() else None

    _write_main_tables(scenario_summary, sufficiency, individual_baseline, tables_dir)
    if policy_robustness is not None:
        _write_policy_class_table(policy_robustness, tables_dir)
    _plot_distributional_value(scenario_summary, figures_dir)
    _plot_gap_closed(sufficiency, figures_dir)
    _plot_individual_stats(individual_baseline, figures_dir)
    if grid is not None:
        _write_grid_table(grid, tables_dir)
        _plot_grid_heatmap(grid, figures_dir)
    _write_results_block(
        scenario_summary,
        sufficiency,
        individual_baseline,
        grid,
        policy_robustness,
        article_dir / "sections" / "results.tex",
    )


def _write_main_tables(
    scenario_summary: pd.DataFrame,
    sufficiency: pd.DataFrame,
    individual_baseline: pd.DataFrame,
    tables_dir: Path,
) -> None:
    main_table = scenario_summary[
        [
            "scenario",
            "distributional_value_vs_aggregate",
            "distributional_value_vs_aggregate_ci_low",
            "distributional_value_vs_aggregate_ci_high",
            "distributional_win_rate_vs_aggregate",
            "share_of_full_information_gap_closed",
        ]
    ].copy()
    main_table["scenario"] = main_table["scenario"].map(SCENARIO_LABELS)
    main_table = main_table.rename(
        columns={
            "scenario": "Сценарий",
            "distributional_value_vs_aggregate": "Снижение потерь",
            "distributional_value_vs_aggregate_ci_low": "ДИ, нижняя граница",
            "distributional_value_vs_aggregate_ci_high": "ДИ, верхняя граница",
            "distributional_win_rate_vs_aggregate": "Доля выигрышей",
            "share_of_full_information_gap_closed": "Закрытая доля разрыва",
        }
    )
    _write_table(
        main_table,
        tables_dir / "table_main_distributional_value.tex",
        float_format="%.6f",
        caption="Ценность распределительной информации по сценариям",
        label="tab:main_distributional_value",
    )

    sufficiency_table = sufficiency[
        [
            "scenario",
            "sufficiency_gap_pct_of_aggregate",
            "share_of_gap_closed_by_distribution",
        ]
    ].copy()
    sufficiency_table["scenario"] = sufficiency_table["scenario"].map(SCENARIO_LABELS)
    sufficiency_table = sufficiency_table.rename(
        columns={
            "scenario": "Сценарий",
            "sufficiency_gap_pct_of_aggregate": "Разрыв к полной информации",
            "share_of_gap_closed_by_distribution": "Закрытая доля разрыва",
        }
    )
    _write_table(
        sufficiency_table,
        tables_dir / "table_sufficiency_gap.tex",
        float_format="%.6f",
        caption="Разрыв до полной информации и его сокращение",
        label="tab:sufficiency_gap",
    )

    individual_table = individual_baseline[
        [
            "comparison_label",
            "loss_reduction",
            "loss_reduction_ci_low",
            "loss_reduction_ci_high",
            "win_rate",
        ]
    ].copy()
    individual_label_map = {
        "Средняя MPC против оценённых агрегатов": "Средняя предельная склонность к потреблению против оценённых агрегатов",
        "Доля низколиквидных против оценённых агрегатов": "Доля низколиквидных домохозяйств против оценённых агрегатов",
        "MPC и доля низколиквидных против оценённых агрегатов": "Две распределительные статистики против оценённых агрегатов",
    }
    individual_table["comparison_label"] = individual_table["comparison_label"].replace(individual_label_map)
    individual_table = individual_table.rename(
        columns={
            "comparison_label": "Сравнение",
            "loss_reduction": "Снижение потерь",
            "loss_reduction_ci_low": "ДИ, нижняя граница",
            "loss_reduction_ci_high": "ДИ, верхняя граница",
            "win_rate": "Доля выигрышей",
        }
    )
    _write_table(
        individual_table,
        tables_dir / "table_individual_distributional_stats.tex",
        float_format="%.6f",
        caption="Отдельная роль распределительных статистик в базовом сценарии",
        label="tab:individual_distributional_stats",
    )


def _write_policy_class_table(policy_robustness: pd.DataFrame, tables_dir: Path) -> None:
    table = policy_robustness[
        [
            "scenario",
            "rule_class",
            "distributional_value_vs_aggregate",
            "distributional_value_vs_aggregate_ci_low",
            "distributional_value_vs_aggregate_ci_high",
            "distributional_win_rate_vs_aggregate",
            "distributional_value_vs_filtered",
        ]
    ].copy()
    table["scenario"] = table["scenario"].map(SCENARIO_LABELS)
    table["rule_class"] = table["rule_class"].replace(
        {
            "linear": "Линейное",
            "quadratic": "Квадратичное",
        }
    )
    table = table.rename(
        columns={
            "scenario": "Сценарий",
            "rule_class": "Правило",
            "distributional_value_vs_aggregate": "Снижение к агрегатам",
            "distributional_value_vs_aggregate_ci_low": "ДИ, нижняя граница",
            "distributional_value_vs_aggregate_ci_high": "ДИ, верхняя граница",
            "distributional_win_rate_vs_aggregate": "Доля выигрышей",
            "distributional_value_vs_filtered": "Снижение к оценённым агрегатам",
        }
    )
    _write_table(
        table,
        tables_dir / "table_policy_class_robustness.tex",
        float_format="%.6f",
        caption="Проверка устойчивости к классу правила",
        label="tab:policy_class_robustness",
    )


def _plot_distributional_value(summary: pd.DataFrame, figures_dir: Path) -> None:
    frame = summary.copy()
    frame["label"] = frame["scenario"].map(SCENARIO_LABELS)
    x = range(len(frame))
    values = frame["distributional_value_vs_aggregate"]
    errors_low = values - frame["distributional_value_vs_aggregate_ci_low"]
    errors_high = frame["distributional_value_vs_aggregate_ci_high"] - values

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, values, color="#2f6f73", alpha=0.85)
    ax.errorbar(x, values, yerr=[errors_low, errors_high], fmt="none", color="black", capsize=4, linewidth=1)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["label"], rotation=20, ha="right")
    ax.set_ylabel("Снижение потерь")
    ax.set_title("Ценность распределительной информации")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_distributional_value_by_scenario.pdf")
    plt.close(fig)


def _plot_gap_closed(sufficiency: pd.DataFrame, figures_dir: Path) -> None:
    frame = sufficiency.copy()
    frame["label"] = frame["scenario"].map(SCENARIO_LABELS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(frame)), 100.0 * frame["share_of_gap_closed_by_distribution"], color="#8a6a2a", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(frame["label"], rotation=20, ha="right")
    ax.set_ylabel("Доля закрытого разрыва, %")
    ax.set_title("Сколько разрыва до полной информации закрывает распределительный блок")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_gap_closed_by_distribution.pdf")
    plt.close(fig)


def _plot_individual_stats(individual: pd.DataFrame, figures_dir: Path) -> None:
    frame = individual.copy()
    labels = ["Предельная склонность", "Низкая ликвидность", "Обе статистики"]
    values = frame["loss_reduction"]
    errors_low = values - frame["loss_reduction_ci_low"]
    errors_high = frame["loss_reduction_ci_high"] - values

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(frame)), values, color="#5d5a88", alpha=0.85)
    ax.errorbar(range(len(frame)), values, yerr=[errors_low, errors_high], fmt="none", color="black", capsize=4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Снижение потерь")
    ax.set_title("Отдельная роль распределительных статистик")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_individual_distributional_stats.pdf")
    plt.close(fig)


def _write_grid_table(grid: pd.DataFrame, tables_dir: Path) -> None:
    table = grid[
        [
            "aggregate_observation_noise",
            "mpc_channel_strength",
            "distributional_value_vs_aggregate",
            "win_rate",
            "share_of_full_information_gap_closed",
        ]
    ].copy()
    table = table.rename(
        columns={
            "aggregate_observation_noise": "Шум агрегатов",
            "mpc_channel_strength": "Сила распределительного канала",
            "distributional_value_vs_aggregate": "Снижение потерь",
            "win_rate": "Доля выигрышей",
            "share_of_full_information_gap_closed": "Закрытая доля разрыва",
        }
    )
    _write_table(
        table,
        tables_dir / "table_distributional_value_grid.tex",
        float_format="%.6f",
        caption="Сетка ценности распределительной информации",
        label="tab:distributional_value_grid",
    )


def _write_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    float_format: str,
    caption: str,
    label: str,
) -> None:
    latex = frame.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
    )
    latex = latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
    latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.write_text(latex, encoding="utf-8")


def _plot_grid_heatmap(grid: pd.DataFrame, figures_dir: Path) -> None:
    pivot = grid.pivot(
        index="mpc_channel_strength",
        columns="aggregate_observation_noise",
        values="distributional_value_vs_aggregate",
    ).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="BrBG")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}" for value in pivot.index])
    ax.set_xlabel("Шум агрегатных наблюдений")
    ax.set_ylabel("Сила распределительного канала")
    ax.set_title("Ценность распределительной информации")
    fig.colorbar(image, ax=ax, label="Снижение потерь")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_distributional_value_grid.pdf")
    plt.close(fig)


def _write_results_block(
    scenario_summary: pd.DataFrame,
    sufficiency: pd.DataFrame,
    individual_baseline: pd.DataFrame,
    grid: pd.DataFrame | None,
    policy_robustness: pd.DataFrame | None,
    path: Path,
) -> None:
    scenario_index = scenario_summary.set_index("scenario")
    sufficiency_index = sufficiency.set_index("scenario")
    high_heterogeneity = scenario_index.loc["high_heterogeneity"]
    high_noise = scenario_index.loc["high_aggregate_noise"]
    noisy_distribution = scenario_index.loc["noisy_distributional_data"]
    baseline = scenario_index.loc["baseline"]
    suff_high = sufficiency_index.loc["high_heterogeneity"]
    suff_noise = sufficiency_index.loc["high_aggregate_noise"]
    liquidity = individual_baseline.set_index("left").loc["distributional_liquidity"]
    mpc = individual_baseline.set_index("left").loc["distributional_mpc"]

    lines = [
        "\\section{Результаты}",
        "",
        "\\subsection{Основной эффект распределительной информации}",
        "",
        "В базовом сценарии добавление распределительных статистик снижает средние потери относительно правила по агрегатной информации. "
        f"Среднее снижение потерь равно {baseline['distributional_value_vs_aggregate']:.6f}; "
        f"доверительный интервал составляет "
        f"[{baseline['distributional_value_vs_aggregate_ci_low']:.6f}; {baseline['distributional_value_vs_aggregate_ci_high']:.6f}], "
        f"а доля выигрышных траекторий равна {baseline['distributional_win_rate_vs_aggregate']:.2f}. "
        "Это означает, что распределительный блок имеет положительную ценность уже в базовой среде, хотя эффект остаётся умеренным.",
        "",
        "Наиболее выраженный результат возникает при высокой неоднородности домохозяйств. "
        f"В этом сценарии снижение потерь достигает {high_heterogeneity['distributional_value_vs_aggregate']:.6f}, "
        f"доля выигрышных траекторий возрастает до {high_heterogeneity['distributional_win_rate_vs_aggregate']:.2f}, "
        f"а распределительный блок закрывает {100 * suff_high['share_of_gap_closed_by_distribution']:.1f}\\% "
        "разрыва между агрегатной информацией и полной информацией.",
        "",
        "При высоком шуме агрегатных наблюдений средний эффект остаётся положительным, но доверительный интервал включает ноль. "
        f"Снижение потерь равно {high_noise['distributional_value_vs_aggregate']:.6f}, "
        f"а закрытая доля разрыва до полной информации составляет только {100 * high_noise['share_of_full_information_gap_closed']:.1f}\\%. "
        "Когда сами распределительные сигналы становятся более шумными, эффект также ослабевает: "
        f"снижение потерь равно {noisy_distribution['distributional_value_vs_aggregate']:.6f}, "
        f"доля выигрышных траекторий составляет {noisy_distribution['distributional_win_rate_vs_aggregate']:.2f}.",
        "",
        "\\input{tables/table_main_distributional_value}",
        "",
        "\\begin{figure}[ht]",
        "\\centering",
        "\\includegraphics[width=0.9\\textwidth]{figures/fig_distributional_value_by_scenario.pdf}",
        "\\caption{Ценность распределительной информации по сценариям}",
        "\\end{figure}",
        "",
        "\\subsection{Разрыв до полной информации}",
        "",
        "Полная информация используется как верхняя граница качества, поэтому важна не только абсолютная разность потерь, но и то, какую часть информационного разрыва закрывает распределительный блок. "
        f"В сценарии высокой неоднородности эта доля равна {100 * suff_high['share_of_gap_closed_by_distribution']:.1f}\\%. "
        f"Напротив, при высоком шуме агрегатов общий разрыв до полной информации возрастает до {100 * suff_noise['sufficiency_gap_pct_of_aggregate']:.1f}\\% "
        "от потерь агрегатного правила, но распределительный блок закрывает лишь небольшую часть этого разрыва. "
        "Это говорит о том, что распределительная информация полезна не механически, а тогда, когда она действительно помогает описать будущую трансмиссию ставки.",
        "",
        "\\input{tables/table_sufficiency_gap}",
        "",
        "\\begin{figure}[ht]",
        "\\centering",
        "\\includegraphics[width=0.9\\textwidth]{figures/fig_gap_closed_by_distribution.pdf}",
        "\\caption{Доля разрыва до полной информации, закрытая распределительным блоком}",
        "\\end{figure}",
        "",
        "\\subsection{Какая распределительная статистика важнее}",
        "",
        "Отдельная проверка распределительных статистик показывает, что в базовом сценарии более устойчивый сигнал даёт доля низколиквидных домохозяйств. "
        f"Снижение потерь для этой статистики равно {liquidity['loss_reduction']:.6f}, "
        f"доля выигрышных траекторий равна {liquidity['win_rate']:.2f}. "
        "Средняя предельная склонность к потреблению также даёт положительное среднее снижение потерь, "
        f"но оно меньше: {mpc['loss_reduction']:.6f}, при доле выигрышных траекторий {mpc['win_rate']:.2f}. "
        "В данной калибровке совместное добавление двух статистик не даёт заметного выигрыша сверх сигнала низкой ликвидности.",
        "",
        "\\input{tables/table_individual_distributional_stats}",
    ]
    if grid is not None:
        best = grid.loc[grid["distributional_value_vs_aggregate"].idxmax()]
        lines.extend(
            [
                "",
                "\\subsection{Когда эффект усиливается}",
                "",
                "Дополнительная сетка по шуму агрегатных наблюдений и силе распределительного канала показывает, что ценность распределительной информации меняется по параметрам среды. "
                f"Наибольшее снижение потерь в этой сетке равно {best['distributional_value_vs_aggregate']:.6f}; "
                f"оно достигается при шуме агрегатных наблюдений {best['aggregate_observation_noise']:.1f} "
                f"и силе распределительного канала {best['mpc_channel_strength']:.2f}. "
                "Это поддерживает основную интерпретацию: распределительные статистики особенно полезны, когда агрегатные индикаторы хуже отражают будущую силу трансмиссии ставки.",
                "",
                "\\input{tables/table_distributional_value_grid}",
                "",
                "\\begin{figure}[ht]",
                "\\centering",
                "\\includegraphics[width=0.75\\textwidth]{figures/fig_distributional_value_grid.pdf}",
                "\\caption{Сетка ценности распределительной информации}",
                "\\end{figure}",
            ]
        )
    if policy_robustness is not None:
        quadratic = policy_robustness[policy_robustness["rule_class"] == "quadratic"]
        if not quadratic.empty:
            base_quad = quadratic.set_index("scenario").loc["baseline"]
            high_quad = quadratic.set_index("scenario").loc["high_heterogeneity"]
            lines.extend(
                [
                    "",
                    "\\subsection{Проверка класса правила}",
                    "",
                    "Проверка с квадратичным правилом используется только как проверка устойчивости, а не как новая основная спецификация. "
                    "В этой проверке положительный знак ценности распределительной информации относительно одних агрегатов сохраняется, "
                    "но сам эффект становится намного меньше. "
                    f"В базовом сценарии квадратичное правило даёт снижение потерь {base_quad['distributional_value_vs_aggregate']:.6f}, "
                    f"а в сценарии высокой неоднородности -- {high_quad['distributional_value_vs_aggregate']:.6f}. "
                    "Добавочный выигрыш относительно оценённых агрегатов становится практически нулевым.",
                    "",
                    "Следовательно, вывод о распределительной информации следует формулировать осторожно. "
                    "Основной результат состоит не в универсальном доминировании распределительного состояния, а в том, что его ценность проявляется в средах и классах правил, где агрегатные показатели не исчерпывают информацию о трансмиссии ставки.",
                    "",
                    "\\input{tables/table_policy_class_robustness}",
                ]
            )
    lines.extend(
        [
            "",
            "\\subsection{Итоговая интерпретация}",
            "",
            "Полученные результаты в целом поддерживают центральную гипотезу работы в условной форме. "
            "Распределительная информация имеет положительную ценность тогда, когда она помогает предсказывать силу будущей трансмиссии ставки. "
            "Эта ценность наиболее заметна при высокой неоднородности домохозяйств и слабее при зашумлённых распределительных данных или при более гибком классе правила.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build article tables and figures from experiment outputs.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--article-dir", default="article")
    args = parser.parse_args()
    build_article_outputs(outputs_dir=Path(args.outputs_dir), article_dir=Path(args.article_dir))


if __name__ == "__main__":
    main()
