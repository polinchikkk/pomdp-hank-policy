from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PALETTE = {
    "helpful": "#c06c2d",
    "not_helpful": "#8a8f98",
    "channel": "#276c8f",
    "loss": "#4f6f3f",
    "zero": "#222222",
}


@dataclass(frozen=True)
class RobustnessComparativeStaticsSpec:
    main_pairwise: str
    noise_summary: str
    signal_strength_summary: str
    no_distributional_signal_summary: str
    income_risk_summary: str
    income_risk_shock_summary: str
    liquid_wedge_summary: str
    loss_decomposition: str
    identification_summary: str
    output_dir: str
    figure_dir: str
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a comparative-statics robustness layer.")
    parser.add_argument("--main-pairwise", default="outputs/ssj/stochastic/main_voi_joint_filter/pairwise_value_of_information.csv")
    parser.add_argument("--noise-summary", default="outputs/ssj/stochastic/noise_sensitivity/noise_sensitivity_summary.csv")
    parser.add_argument("--signal-strength-summary", default="outputs/ssj/stochastic/distributional_signal_strength/distributional_signal_strength_summary.csv")
    parser.add_argument("--no-distributional-signal-summary", default="outputs/ssj/stochastic/no_distributional_signal/no_distributional_signal_summary.csv")
    parser.add_argument("--income-risk-summary", default="outputs/ssj/income_risk_calibration/income_risk_calibration_summary.csv")
    parser.add_argument("--income-risk-shock-summary", default="outputs/ssj/stochastic/income_risk_shock_source/income_risk_shock_source_summary.csv")
    parser.add_argument("--liquid-wedge-summary", default="outputs/ssj/liquid_wedge_channel/liquid_wedge_channel_summary.csv")
    parser.add_argument("--loss-decomposition", default="outputs/ssj/stochastic/main_voi_joint_filter/loss_component_decomposition.csv")
    parser.add_argument("--identification-summary", default="outputs/ssj/stochastic/identification_battery/identification_battery_summary.csv")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/robustness_comparative_statics")
    parser.add_argument("--figure-dir", default="article/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    comparative = _build_comparative_statics(args)
    question_summary = _build_question_summary(comparative)
    loss_channels = _loss_channel_summary(Path(args.loss_decomposition))

    comparative.to_csv(output_dir / "robustness_comparative_statics.csv", index=False)
    question_summary.to_csv(output_dir / "robustness_questions_summary.csv", index=False)
    loss_channels.to_csv(output_dir / "loss_channel_summary.csv", index=False)
    _write_report(question_summary, loss_channels, output_dir / "report_robustness_comparative_statics.md")
    _plot_comparative_statics(comparative, loss_channels, figure_dir / "fig_robustness_comparative_statics.pdf")

    spec = RobustnessComparativeStaticsSpec(
        main_pairwise=args.main_pairwise,
        noise_summary=args.noise_summary,
        signal_strength_summary=args.signal_strength_summary,
        no_distributional_signal_summary=args.no_distributional_signal_summary,
        income_risk_summary=args.income_risk_summary,
        income_risk_shock_summary=args.income_risk_shock_summary,
        liquid_wedge_summary=args.liquid_wedge_summary,
        loss_decomposition=args.loss_decomposition,
        identification_summary=args.identification_summary,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        note=(
            "Сводный слой робастности группирует существующие проверки вокруг трёх экономических вопросов: "
            "когда распределительная информация полезна, когда она не полезна и через какой компонент функции "
            "потерь проходит выигрыш. Часть sensitivity-прогонов относится к старой скалярной фильтрации и "
            "используется как направленная сравнительная статика."
        ),
    )
    (output_dir / "robustness_comparative_statics_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'robustness_comparative_statics.csv'}")
    print(f"Wrote {figure_dir / 'fig_robustness_comparative_statics.pdf'}")


def _build_comparative_statics(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_main_rows(Path(args.main_pairwise)))
    rows.extend(_noise_rows(Path(args.noise_summary)))
    rows.extend(_signal_strength_rows(Path(args.signal_strength_summary)))
    rows.extend(_no_distributional_signal_rows(Path(args.no_distributional_signal_summary)))
    rows.extend(_income_risk_rows(Path(args.income_risk_summary)))
    rows.extend(_income_risk_shock_rows(Path(args.income_risk_shock_summary)))
    rows.extend(_liquid_wedge_rows(Path(args.liquid_wedge_summary)))
    rows.extend(_identification_rows(Path(args.identification_summary)))
    return pd.DataFrame(rows)


def _main_rows(path: Path) -> list[dict[str, object]]:
    pairwise = pd.read_csv(path)
    row = pairwise[
        (pairwise["scenario"] == "all")
        & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
    ].iloc[0]
    return [
        _row(
            question="Когда полезна?",
            block="Основная оценка",
            case="joint_filter_continuous",
            label="Основная спецификация",
            x_value=1.0,
            mvoi=float(row["loss_reduction"]),
            ci_low=float(row["ci_low"]),
            ci_high=float(row["ci_high"]),
            win_rate=float(row["win_rate"]),
            source="joint_filter_continuous",
            interpretation="Положительная предельная ценность сверх фильтрованных агрегатов.",
        )
    ]


def _noise_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for _, item in frame.iterrows():
        if item["axis"] == "aggregate":
            block = "Шум агрегатов"
            label = f"шум агрегатов ×{item['aggregate_noise_scale']:g}"
            x_value = float(item["aggregate_noise_scale"])
            interpretation = "Распределительная информация полезнее, когда агрегатные сигналы менее точны."
        elif item["axis"] == "distribution":
            block = "Шум распределительных сигналов"
            label = f"шум распределения ×{item['distribution_noise_scale']:g}"
            x_value = float(item["distribution_noise_scale"])
            interpretation = "Ценность снижается, когда сами распределительные сигналы становятся шумнее."
        else:
            block = "Шум наблюдений"
            label = "базовый шум"
            x_value = 1.0
            interpretation = "Базовая точка старой sensitivity-спецификации."
        rows.append(
            _row(
                question="Когда полезна?",
                block=block,
                case=str(item["case"]),
                label=label,
                x_value=x_value,
                mvoi=float(item["mvoi_dist"]),
                ci_low=float(item["ci_low"]),
                ci_high=float(item["ci_high"]),
                win_rate=float(item["win_rate"]),
                source="scalar_filter_sensitivity",
                interpretation=interpretation,
            )
        )
    return rows


def _signal_strength_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [
        _row(
            question="Когда полезна?",
            block="Сила распределительного сигнала",
            case=f"signal_{item['distributional_signal_factor']:g}",
            label=f"сигнал ×{item['distributional_signal_factor']:g}",
            x_value=float(item["distributional_signal_factor"]),
            mvoi=float(item["mvoi_dist"]),
            ci_low=float(item["ci_low"]),
            ci_high=float(item["ci_high"]),
            win_rate=float(item["win_rate"]),
            source="scalar_filter_sensitivity",
            interpretation="Ценность растёт вместе с силой распределительного сигнала.",
        )
        for _, item in frame.iterrows()
    ]


def _no_distributional_signal_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [
        _row(
            question="Когда не полезна?",
            block="Выключение распределительного сигнала",
            case=str(item["case"]),
            label=str(item["description"]),
            x_value=float(index),
            mvoi=float(item["mvoi_dist"]),
            ci_low=float(item["ci_low"]),
            ci_high=float(item["ci_high"]),
            win_rate=float(item["win_rate"]),
            source="scalar_filter_sensitivity",
            interpretation="При нулевой распределительной динамике ценность исчезает.",
        )
        for index, (_, item) in enumerate(frame.iterrows())
    ]


def _income_risk_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [
        _row(
            question="Когда полезна?",
            block="Доходный риск",
            case=f"sigma_z_{item['sigma_z']:g}",
            label=f"sigma_z={item['sigma_z']:g}",
            x_value=float(item["sigma_z"]),
            mvoi=float(item["mvoi_dist"]),
            ci_low=float(item["ci_low"]),
            ci_high=float(item["ci_high"]),
            win_rate=float(item["win_rate"]),
            source="hank_recomputed_sensitivity",
            interpretation="Проверка устойчивости к калибровке доходного риска.",
        )
        for _, item in frame.iterrows()
    ]


def _income_risk_shock_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [
        _row(
            question="Когда полезна?",
            block="Шок доходного риска",
            case=str(item["case"]),
            label=str(item["description"]),
            x_value=float(index),
            mvoi=float(item["mvoi_dist"]),
            ci_low=float(item["ci_low"]),
            ci_high=float(item["ci_high"]),
            win_rate=float(item["win_rate"]),
            source="scalar_filter_sensitivity",
            interpretation="Добавление шока доходного риска не разрушает результат.",
        )
        for index, (_, item) in enumerate(frame.iterrows())
    ]


def _liquid_wedge_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [
        _row(
            question="Когда не полезна?",
            block="Клин ликвидной доходности",
            case=f"omega_{item['omega']:g}",
            label=f"omega={item['omega']:g}",
            x_value=float(item["omega"]),
            mvoi=float(item["mvoi_dist"]),
            ci_low=float(item["ci_low"]),
            ci_high=float(item["ci_high"]),
            win_rate=float(item["win_rate"]),
            source="hank_recomputed_sensitivity",
            interpretation="При слабом распределительном канале ценность мала или отрицательна; при высоком omega становится положительной.",
        )
        for _, item in frame.iterrows()
    ]


def _identification_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [
        _row(
            question="Когда не полезна?",
            block="Ложные распределительные признаки",
            case=str(item["variant"]),
            label=str(item["variant_ru"]),
            x_value=float(index),
            mvoi=float(item["loss_reduction"]),
            ci_low=float(item["ci_low"]),
            ci_high=float(item["ci_high"]),
            win_rate=float(item["win_rate"]),
            source="joint_filter_identification",
            interpretation="Ложные и временно смещённые признаки не должны воспроизводить содержательный эффект.",
        )
        for index, (_, item) in enumerate(frame.iterrows())
    ]


def _loss_channel_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    subset = frame[
        (frame["scenario"] == "all")
        & (frame["comparison"] == "filtered_distribution_minus_filtered_aggregates")
        & (frame["component"].isin(["inflation_loss", "output_gap_loss", "consumption_loss", "rate_smoothing_loss"]))
    ].copy()
    subset["positive_share"] = subset["mean_reduction"] / subset["mean_reduction"].abs().sum()
    return subset[
        [
            "component",
            "component_ru",
            "mean_reduction",
            "ci_low",
            "ci_high",
            "share_of_total_reduction",
            "positive_share",
        ]
    ]


def _build_question_summary(comparative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (question, block), frame in comparative.groupby(["question", "block"], sort=False):
        rows.append(
            {
                "question": question,
                "block": block,
                "num_cases": int(len(frame)),
                "min_mvoi": float(frame["mvoi_dist"].min()),
                "max_mvoi": float(frame["mvoi_dist"].max()),
                "mean_mvoi": float(frame["mvoi_dist"].mean()),
                "positive_share": float((frame["mvoi_dist"] > 0.0).mean()),
                "main_interpretation": str(frame["interpretation"].iloc[0]),
                "sources": ", ".join(sorted(set(frame["source"]))),
            }
        )
    return pd.DataFrame(rows)


def _plot_comparative_statics(comparative: pd.DataFrame, loss_channels: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4))
    _plot_line_block(
        axes[0, 0],
        comparative,
        block="Шум агрегатов",
        title="Когда агрегатные сигналы шумнее",
        xlabel="Множитель шума агрегатов",
        color=PALETTE["helpful"],
    )
    _plot_line_block(
        axes[0, 1],
        comparative,
        block="Шум распределительных сигналов",
        title="Когда распределительные сигналы шумнее",
        xlabel="Множитель шума распределения",
        color=PALETTE["channel"],
    )
    _plot_two_lines(
        axes[1, 0],
        comparative,
        title="Сила распределительного канала",
    )
    _plot_loss_channels(axes[1, 1], loss_channels)
    for ax in axes.ravel():
        ax.axhline(0.0, color=PALETTE["zero"], linewidth=0.8)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Сравнительная статика ценности распределительной информации")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_line_block(ax, comparative: pd.DataFrame, *, block: str, title: str, xlabel: str, color: str) -> None:
    frame = comparative[comparative["block"] == block].sort_values("x_value")
    if frame.empty:
        ax.set_title(title)
        return
    x = frame["x_value"].to_numpy(dtype=float)
    y = frame["mvoi_dist"].to_numpy(dtype=float)
    lower = frame["mvoi_ci_low"].to_numpy(dtype=float)
    upper = frame["mvoi_ci_high"].to_numpy(dtype=float)
    err_low = np.maximum(y - lower, 0.0)
    err_high = np.maximum(upper - y, 0.0)
    ax.errorbar(x, y, yerr=[err_low, err_high], marker="o", linewidth=2.0, capsize=4, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MVOI")


def _plot_two_lines(ax, comparative: pd.DataFrame, *, title: str) -> None:
    signal = comparative[comparative["block"] == "Сила распределительного сигнала"].sort_values("x_value")
    wedge = comparative[comparative["block"] == "Клин ликвидной доходности"].sort_values("x_value")
    if not signal.empty:
        ax.plot(
            signal["x_value"],
            signal["mvoi_dist"],
            marker="o",
            linewidth=2.0,
            color=PALETTE["helpful"],
            label="сила сигнала",
        )
    if not wedge.empty:
        ax.plot(
            wedge["x_value"],
            wedge["mvoi_dist"],
            marker="s",
            linewidth=2.0,
            color=PALETTE["channel"],
            label="ликвидный клин",
        )
    ax.set_title(title)
    ax.set_xlabel("Параметр сравнительной статики")
    ax.set_ylabel("MVOI")
    ax.legend(frameon=False)


def _plot_loss_channels(ax, loss_channels: pd.DataFrame) -> None:
    labels = {
        "inflation_loss": "инфляция",
        "output_gap_loss": "выпуск",
        "consumption_loss": "потребление",
        "rate_smoothing_loss": "ставка",
    }
    frame = loss_channels.copy()
    x = np.arange(len(frame))
    y = frame["mean_reduction"].to_numpy(dtype=float)
    ax.bar(x, y, color=PALETTE["loss"], edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[item] for item in frame["component"]], rotation=20, ha="right")
    ax.set_title("Через какой компонент идут выигрыши")
    ax.set_ylabel("Снижение потерь")


def _write_report(question_summary: pd.DataFrame, loss_channels: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Сравнительная статика робастности",
        "",
        "Проверки сгруппированы вокруг трёх вопросов: когда распределительная информация полезна, когда она не полезна и через какой компонент функции потерь проходит выигрыш.",
        "",
        "## Вопрос 1. Когда полезна",
        "",
        "- При росте шума агрегатных сигналов предельная ценность распределительной информации растёт.",
        "- При снижении шума распределительных сигналов предельная ценность также выше.",
        "- При усилении распределительного сигнала MVOI растёт от нуля к положительным значениям.",
        "",
        "## Вопрос 2. Когда не полезна",
        "",
        "- При выключенной распределительной динамике MVOI исчезает.",
        "- При слабом клине ликвидной доходности MVOI отрицателен или близок к нулю.",
        "- Искусственные и временно сдвинутые признаки не воспроизводят эффект фактической распределительной информации.",
        "",
        "## Вопрос 3. Канал выигрыша",
        "",
    ]
    for _, row in loss_channels.iterrows():
        lines.append(
            f"- {row['component_ru']}: снижение {row['mean_reduction']:.6g}, "
            f"доля общего эффекта {row['share_of_total_reduction']:.3g}."
        )
    lines.extend(
        [
            "",
            "Оговорка: часть sensitivity-прогонов была рассчитана в старой скалярной фильтровой спецификации. "
            "Они используются как направленная сравнительная статика; основная оценка статьи остаётся совместной фильтровой спецификацией.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(
    *,
    question: str,
    block: str,
    case: str,
    label: str,
    x_value: float,
    mvoi: float,
    ci_low: float,
    ci_high: float,
    win_rate: float,
    source: str,
    interpretation: str,
) -> dict[str, object]:
    return {
        "question": question,
        "block": block,
        "case": case,
        "label": label,
        "x_value": float(x_value),
        "mvoi_dist": float(mvoi),
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "mvoi_ci_low": float(-ci_high),
        "mvoi_ci_high": float(-ci_low),
        "win_rate": float(win_rate),
        "source": source,
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    main()
