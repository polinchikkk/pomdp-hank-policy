from __future__ import annotations

from pathlib import Path

import pandas as pd


def _range_text(series: pd.Series) -> str:
    values = series.astype(float)
    return f"{values.min():.1f}--{values.max():.1f}"


def _save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _build_stage6_table(
    architecture: pd.DataFrame,
    misspec_win: pd.DataFrame,
    env_shift: pd.DataFrame,
) -> pd.DataFrame:
    belief_impr = 100.0 * (
        architecture["classical_mean_cumulative_loss"] - architecture["belief_mean_cumulative_loss"]
    ) / architecture["classical_mean_cumulative_loss"]
    rawobs_impr = 100.0 * (
        architecture["classical_mean_cumulative_loss"] - architecture["rawobs_mean_cumulative_loss"]
    ) / architecture["classical_mean_cumulative_loss"]
    rawobs_vs_belief = 100.0 * (
        architecture["belief_mean_cumulative_loss"] - architecture["rawobs_mean_cumulative_loss"]
    ) / architecture["belief_mean_cumulative_loss"]

    misspec_positive = misspec_win[misspec_win["mean_relative_improvement_pct"] > 0.0].copy()

    rows = [
        {
            "block": "Сопоставление архитектур",
            "main_result": (
                f"Обучаемое правило по оцененному состоянию улучшает результат относительно жестко заданного правила "
                f"на {_range_text(belief_impr)}% по четырем сценариям; прямая политика по наблюдениям улучшает его "
                f"на {_range_text(rawobs_impr)}%."
            ),
            "boundary": (
                "Прямая политика по наблюдениям полезнее только в сценариях с более богатым набором информации; "
                "при тонком информационном наборе лучше работает обучаемое правило по отфильтрованному состоянию."
            ),
        },
        {
            "block": "Карта архитектурных ошибок",
            "main_result": (
                f"Лучшее обучаемое правило превосходит классические схемы с ошибками фильтрации и задания вероятностей режима "
                f"в среднем на {_range_text(misspec_positive['mean_relative_improvement_pct'])}%; "
                "доля выигрыша по сценариям и по проверочным траекториям равна единице."
            ),
            "boundary": (
                "Преимущество не является универсальным: против удачного простого правила, реагирующего только на инфляцию, "
                "обучаемое правило уже не выигрывает."
            ),
        },
        {
            "block": "Перенос на новые среды",
            "main_result": (
                f"При переносе на новые среды обучаемое правило остается лучше фиксированного правила "
                f"в среднем на {_range_text(env_shift['mean_learned_improvement_vs_fixed_pct'])}%; "
                "выигрыш наблюдается во всех средах."
            ),
            "boundary": (
                "Однако перенастроенное простое правило переносится лучше: обучаемое правило проигрывает ему во всех средах "
                "и теряет больше качества при переносе."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _latex_escape(text: str) -> str:
    return (
        text.replace("%", "\\%")
        .replace("_", "\\_")
        .replace("&", "\\&")
    )


def _write_latex_table(path: Path, table: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{p{0.20\\textwidth}p{0.36\\textwidth}p{0.36\\textwidth}}",
        "\\toprule",
        "Блок & Основной количественный результат & Граница применимости \\\\",
        "\\midrule",
    ]
    for row in table.to_dict(orient="records"):
        lines.append(
            f"{_latex_escape(row['block'])} & {_latex_escape(row['main_result'])} & {_latex_escape(row['boundary'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _save_text(path, "\n".join(lines))


def _main_thesis() -> str:
    return (
        "Итог этапа 6 состоит в том, что обучаемое правило полезно не как универсальная замена классическому правилу, "
        "а как более гибкий способ выбора ставки в среде со скрытыми режимами и несовершенной информацией. "
        "Оно устойчиво превосходит жестко заданное правило и особенно выигрывает там, где классическая схема ошибается "
        "в фильтрации и формировании оценки состояния, однако не дает общего преимущества над хорошо перенастроенным "
        "простым правилом."
    )


def _main_result_section() -> str:
    return r"""
\section{Основной результат}

Итог этапа 6 состоит в том, что обучаемое правило полезно не как универсальная замена классическому правилу, а как более гибкий способ выбора процентной ставки в среде со скрытыми режимами и несовершенной информацией. В двухактивной HANK-модели со скрытым переключением режимов обучаемая политика устойчиво улучшает результат по сравнению с жестко заданным правилом, причем этот выигрыш проявляется как при прямом сопоставлении различных способов построения политики, так и при специально введенных ошибках в классической схеме фильтрации. В серии сопоставления архитектур обучаемое правило по отфильтрованному состоянию уменьшает накопленную функцию потерь на 56.7--71.1\% относительно схемы <<фильтрация + фиксированное правило>>, а прямая политика по наблюдаемым данным дает улучшение на 24.7--73.6\%. В серии карты архитектурных ошибок лучшее обучаемое правило превосходит классические схемы с ошибками фильтрации, завышенным шумом наблюдения и неверной персистентностью режимов в среднем на 62.9--66.6\%, причем этот результат устойчив по сценариям и по проверочным траекториям. Наконец, в серии переноса на новые среды обучаемая политика, настроенная на базовой среде, остается лучше фиксированного правила и при переносе на слегка измененные среды; средний выигрыш по накопленной функции потерь составляет 51.9--61.6\%.

\begin{table}[htbp]
\centering
\small
\caption{Итоговые результаты этапа 6}
\label{tab:stage6_summary}
\input{outputs/hank_regime_learning_stage6_summary/table_stage6_summary.tex}
\end{table}
"""


def _limits_section() -> str:
    return r"""
\section{Границы применимости обучаемой политики}

Полученный результат имеет четкие границы применимости. Во-первых, преимущество обучаемого правила не является универсальным по отношению ко всем классическим альтернативам. В серии сопоставления архитектур оказалось, что прямая политика по наблюдаемым переменным полезнее только тогда, когда информационный набор достаточно богат; при тонком информационном наборе лучше работает обучаемое правило, опирающееся на отфильтрованное состояние. Во-вторых, серия карты архитектурных ошибок показала, что выигрыш обучаемой политики связан прежде всего с устойчивостью к ошибкам в устройстве классической схемы принятия решений. Если классическая политика использует ошибочную фильтрацию или неверно задает вероятности режимов, обучаемое правило имеет значительное преимущество. Однако против удачного простого правила, которое реагирует только на инфляцию, это преимущество исчезает.

В-третьих, серия переноса на новые среды показывает, что обучаемое правило не превосходит хорошо перенастроенное простое правило по способности переноситься на новые среды. Хотя обучаемая политика сохраняет уверенный выигрыш относительно фиксированного правила, перенастроенное простое правило оказывается сильнее и по уровню накопленной потери, и по устойчивости к смене среды. Поэтому общий вывод этапа 6 состоит не в том, что обучаемая политика всегда лучше классической, а в более узком и содержательном утверждении: обучаемое правило особенно полезно тогда, когда регулятор сталкивается со скрытыми режимами и ошибками в архитектуре фильтрации и построения решения, тогда как в хорошо настроенной и корректно специфицированной простой схеме преимущество обучения уже не является гарантированным.
"""


def run_stage6_summary(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_summary",
    architecture_dir: str = "outputs/hank_regime_learning_stage6_architecture_ablation",
    misspecification_dir: str = "outputs/hank_regime_learning_stage6_misspecification_map",
    environment_shift_dir: str = "outputs/hank_regime_learning_stage6_environment_shift",
) -> dict[str, pd.DataFrame | str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    architecture = pd.read_csv(Path(architecture_dir) / "architecture_comparison.csv")
    misspec_win = pd.read_csv(Path(misspecification_dir) / "misspecification_win_summary.csv")
    environment_shift = pd.read_csv(Path(environment_shift_dir) / "environment_shift_win_summary.csv")

    table = _build_stage6_table(architecture, misspec_win, environment_shift)
    table.to_csv(root / "stage6_summary_table.csv", index=False)
    _write_latex_table(root / "table_stage6_summary.tex", table)

    thesis = _main_thesis()
    main_section = _main_result_section().strip()
    limits_section = _limits_section().strip()

    _save_text(root / "stage6_thesis.txt", thesis + "\n")
    _save_text(root / "stage6_text_blocks.tex", main_section + "\n\n" + limits_section + "\n")
    _save_text(
        root / "stage6_summary_report.md",
        "\n".join(
            [
                "# Stage 6 Summary",
                "",
                thesis,
                "",
                "## Основной результат",
                "",
                main_section,
                "",
                "## Границы применимости обучаемой политики",
                "",
                limits_section,
            ]
        ),
    )

    return {
        "table": table,
        "thesis": thesis,
        "main_section": main_section,
        "limits_section": limits_section,
    }
