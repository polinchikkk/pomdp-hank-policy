from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FRONTIER_LEVELS = (
    (0, "aggregate_only", "Текущие агрегаты"),
    (1, "aggregate_history", "История агрегатов"),
    (2, "filtered_aggregates", "Фильтрованные агрегаты"),
    (3, "best_single_distribution", "Один распределительный сигнал"),
    (4, "filtered_distribution", "Все распределительные сигналы"),
    (5, "full_information", "Полная информация"),
)

SINGLE_DISTRIBUTION_STATES = (
    "filtered_distribution_mpc",
    "filtered_distribution_liquidity",
    "filtered_distribution_exposure",
)

SINGLE_LABEL_RU = {
    "filtered_distribution_mpc": "MPC",
    "filtered_distribution_liquidity": "низкая ликвидность",
    "filtered_distribution_exposure": "процентная экспозиция",
}

RANKING_LEVELS = {0, 1, 2, 3, 4}


@dataclass(frozen=True)
class FrontierSource:
    family: str
    variant: str
    label_ru: str
    summary_path: str
    reference: str
    note: str


@dataclass(frozen=True)
class InformationFrontierSpec:
    output_dir: str
    figure_dir: str
    sources: tuple[dict[str, str], ...]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Build information frontier tables and figures.")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/information_frontier")
    parser.add_argument("--figure-dir", default="article/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    sources = _available_sources()
    frontier = _build_frontier(sources)
    marginal = _build_marginal_values(frontier)
    ranking = _build_ranking_details(frontier)
    stability = _build_ranking_stability(frontier)
    seed_stability = _build_seed_stability()
    if not seed_stability.empty:
        stability = pd.concat([seed_stability, stability], ignore_index=True)

    frontier.to_csv(output_dir / "information_frontier.csv", index=False)
    marginal.to_csv(output_dir / "information_frontier_marginal_values.csv", index=False)
    ranking.to_csv(output_dir / "information_state_ranking.csv", index=False)
    stability.to_csv(output_dir / "information_state_ranking_stability.csv", index=False)
    _write_ranking_table(stability, output_dir / "table_information_state_ranking.tex")
    _write_marginal_table(marginal, output_dir / "table_information_marginal_values.tex")
    _write_report(frontier, marginal, stability, output_dir / "report_information_frontier.md")
    _plot_frontier(frontier, figure_dir / "fig_information_frontier.pdf")
    _plot_marginal_values(marginal, figure_dir / "fig_information_marginal_values.pdf")

    spec = InformationFrontierSpec(
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        sources=tuple(asdict(item) for item in sources),
        note=(
            "Information frontier over six information levels. Level 3 is the best single "
            "distributional signal among MPC, low-liquidity share and interest exposure. "
            "Joint-filter results are the main specification; scalar-filter sensitivity cases "
            "are used as comparative statics."
        ),
    )
    (output_dir / "information_frontier_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'information_frontier.csv'}")
    print(f"Wrote {figure_dir / 'fig_information_frontier.pdf'}")


def _available_sources() -> list[FrontierSource]:
    candidates: list[FrontierSource] = [
        FrontierSource(
            family="main",
            variant="joint_filter",
            label_ru="Основная спецификация: совместный фильтр",
            summary_path="outputs/ssj/stochastic/main_voi_joint_filter/main_voi_summary.csv",
            reference="joint_filter",
            note="Основная оценка статьи.",
        ),
        FrontierSource(
            family="main",
            variant="scalar_filter",
            label_ru="Скалярный фильтр",
            summary_path="outputs/ssj/stochastic/main_voi/main_voi_summary.csv",
            reference="scalar_filter",
            note="Техническая проверка.",
        ),
        FrontierSource(
            family="path_count",
            variant="paths_100",
            label_ru="100 HANK/SSJ-траекторий",
            summary_path="outputs/ssj/stochastic/trajectory_count_robustness/paths_100/main_voi/main_voi_summary.csv",
            reference="scalar_filter",
            note="Проверка чувствительности к числу траекторий.",
        ),
        FrontierSource(
            family="noise_aggregate",
            variant="aggregate_noise_0p5",
            label_ru="Шум агрегатов ×0.5",
            summary_path="outputs/ssj/stochastic/noise_sensitivity/aggregate_noise_0p5/main_voi/main_voi_summary.csv",
            reference="scalar_filter",
            note="Сравнительная статика шума наблюдений.",
        ),
        FrontierSource(
            family="noise_aggregate",
            variant="aggregate_noise_2",
            label_ru="Шум агрегатов ×2",
            summary_path="outputs/ssj/stochastic/noise_sensitivity/aggregate_noise_2/main_voi/main_voi_summary.csv",
            reference="scalar_filter",
            note="Сравнительная статика шума наблюдений.",
        ),
        FrontierSource(
            family="noise_distribution",
            variant="distribution_noise_0p5",
            label_ru="Шум распределительных сигналов ×0.5",
            summary_path="outputs/ssj/stochastic/noise_sensitivity/distribution_noise_0p5/main_voi/main_voi_summary.csv",
            reference="scalar_filter",
            note="Сравнительная статика шума распределительных сигналов.",
        ),
        FrontierSource(
            family="noise_distribution",
            variant="distribution_noise_2",
            label_ru="Шум распределительных сигналов ×2",
            summary_path="outputs/ssj/stochastic/noise_sensitivity/distribution_noise_2/main_voi/main_voi_summary.csv",
            reference="scalar_filter",
            note="Сравнительная статика шума распределительных сигналов.",
        ),
    ]
    for factor in ("0", "0p5", "1", "2"):
        candidates.append(
            FrontierSource(
                family="distribution_signal",
                variant=f"signal_{factor}",
                label_ru=f"Сила распределительного сигнала {factor.replace('p', '.')}",
                summary_path=f"outputs/ssj/stochastic/distributional_signal_strength/signal_{factor}/main_voi/main_voi_summary.csv",
                reference="scalar_filter",
                note="Сравнительная статика силы распределительного канала.",
            )
        )
    for omega in ("0p005", "0p01", "0p015", "0p02"):
        candidates.append(
            FrontierSource(
                family="omega",
                variant=f"omega_{omega}",
                label_ru=f"Клин ликвидной доходности {omega.replace('p', '.')}",
                summary_path=f"outputs/ssj/liquid_wedge_channel/omega_{omega}/main_voi/main_voi_summary.csv",
                reference="scalar_filter",
                note="Сравнительная статика клина ликвидной доходности.",
            )
        )
    for sigma in ("0p84", "0p88", "0p92"):
        candidates.append(
            FrontierSource(
                family="income_risk",
                variant=f"sigma_z_{sigma}",
                label_ru=f"Доходный риск {sigma.replace('p', '.')}",
                summary_path=f"outputs/ssj/income_risk_calibration/sigma_z_{sigma}/main_voi/main_voi_summary.csv",
                reference="scalar_filter",
                note="Сравнительная статика доходного риска.",
            )
        )
    candidates.append(
        FrontierSource(
            family="income_risk",
            variant="with_income_risk_shock",
            label_ru="Шок доходного риска",
            summary_path="outputs/ssj/stochastic/income_risk_shock_source/main_voi/main_voi_summary.csv",
            reference="scalar_filter",
            note="Отдельный источник распределительной динамики.",
        )
    )
    return [item for item in candidates if Path(item.summary_path).exists()]


def _build_frontier(sources: list[FrontierSource]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in sources:
        summary = _overall_summary(source.summary_path)
        if summary.empty:
            continue
        for level, state, level_label in FRONTIER_LEVELS:
            if state == "best_single_distribution":
                row, chosen_state = _best_single_distribution(summary)
                information_state = chosen_state
                information_state_ru = f"Один сигнал: {SINGLE_LABEL_RU.get(chosen_state, chosen_state)}"
            else:
                if state not in summary.index:
                    continue
                row = summary.loc[state]
                information_state = state
                information_state_ru = str(row.get("information_state_ru", level_label))
            rows.append(
                {
                    "family": source.family,
                    "variant": source.variant,
                    "line_label_ru": source.label_ru,
                    "reference": source.reference,
                    "source_path": source.summary_path,
                    "note": source.note,
                    "information_level": int(level),
                    "level_label_ru": level_label,
                    "information_state": information_state,
                    "information_state_ru": information_state_ru,
                    "mean_loss": float(row["mean_loss"]),
                    "ci_low": float(row.get("ci_low", np.nan)),
                    "ci_high": float(row.get("ci_high", np.nan)),
                    "num_trajectories": int(row.get("num_trajectories", 0)),
                }
            )
    return pd.DataFrame(rows)


def _overall_summary(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame.columns and "all" in set(frame["scenario"]):
        frame = frame[frame["scenario"] == "all"].copy()
    else:
        numeric = frame.select_dtypes(include=[np.number]).columns
        frame = frame.groupby("information_state", as_index=False)[numeric].mean()
    return frame.set_index("information_state", drop=False)


def _best_single_distribution(summary: pd.DataFrame) -> tuple[pd.Series, str]:
    available = [state for state in SINGLE_DISTRIBUTION_STATES if state in summary.index]
    if not available:
        raise KeyError("No single distributional states in summary.")
    losses = summary.loc[available, "mean_loss"].astype(float)
    chosen_state = str(losses.idxmin())
    return summary.loc[chosen_state], chosen_state


def _build_marginal_values(frontier: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, variant), group in frontier.groupby(["family", "variant"], sort=False):
        ordered = group.sort_values("information_level")
        for current, next_row in zip(ordered.iloc[:-1].to_dict("records"), ordered.iloc[1:].to_dict("records")):
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "line_label_ru": current["line_label_ru"],
                    "from_level": int(current["information_level"]),
                    "to_level": int(next_row["information_level"]),
                    "transition_ru": f"{current['level_label_ru']} → {next_row['level_label_ru']}",
                    "loss_before": float(current["mean_loss"]),
                    "loss_after": float(next_row["mean_loss"]),
                    "marginal_value": float(current["mean_loss"]) - float(next_row["mean_loss"]),
                }
            )
    return pd.DataFrame(rows)


def _build_ranking_details(frontier: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, variant), group in frontier.groupby(["family", "variant"], sort=False):
        ordered = group.sort_values("mean_loss").reset_index(drop=True)
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "line_label_ru": row["line_label_ru"],
                    "rank": rank,
                    "information_level": int(row["information_level"]),
                    "level_label_ru": row["level_label_ru"],
                    "information_state": row["information_state"],
                    "mean_loss": float(row["mean_loss"]),
                }
            )
    return pd.DataFrame(rows)


def _build_ranking_stability(frontier: pd.DataFrame) -> pd.DataFrame:
    references = {
        "joint_filter": _rank_vector(frontier, "main", "joint_filter"),
        "scalar_filter": _rank_vector(frontier, "main", "scalar_filter"),
    }
    checks = [
        ("joint_vs_scalar_filter", "Joint filter vs scalar filter", "scalar_filter", [("main", "joint_filter")]),
        ("path_count", "Число траекторий", "scalar_filter", [("path_count", "paths_100")]),
        ("noise_scale", "Шум наблюдений", "scalar_filter", _variant_keys(frontier, ("noise_aggregate", "noise_distribution"))),
        ("omega", "Клин ликвидной доходности", "scalar_filter", _variant_keys(frontier, ("omega",))),
        ("income_risk", "Доходный риск", "scalar_filter", _variant_keys(frontier, ("income_risk",))),
        ("distribution_signal", "Сила распределительного сигнала", "scalar_filter", _variant_keys(frontier, ("distribution_signal",))),
    ]
    rows: list[dict[str, object]] = []
    for check_id, label, reference_name, keys in checks:
        reference = references.get(reference_name)
        if reference is None or not keys:
            continue
        correlations = []
        same_best = []
        variants = []
        for family, variant in keys:
            vector = _rank_vector(frontier, family, variant)
            if vector is None:
                continue
            correlations.append(_spearman(reference, vector))
            same_best.append(_best_level(reference) == _best_level(vector))
            variants.append(f"{family}:{variant}")
        if not correlations:
            continue
        rows.append(
            {
                "check_id": check_id,
                "check_ru": label,
                "reference": reference_name,
                "num_variants": len(correlations),
                "mean_spearman": float(np.nanmean(correlations)),
                "min_spearman": float(np.nanmin(correlations)),
                "same_best_share": float(np.mean(same_best)),
                "variants": "; ".join(variants),
            }
        )
    return pd.DataFrame(rows)


def _build_seed_stability() -> pd.DataFrame:
    path = Path("outputs/ssj/stochastic/main_voi_joint_filter/trajectory_losses.csv")
    if not path.exists():
        return pd.DataFrame()
    losses = pd.read_csv(path)
    baseline = _frontier_from_loss_frame(losses)
    reference = _rank_from_losses(baseline)
    correlations = []
    same_best = []
    for seed, seed_frame in losses.groupby("observation_seed"):
        seed_frontier = _frontier_from_loss_frame(seed_frame)
        vector = _rank_from_losses(seed_frontier)
        if reference is None or vector is None:
            continue
        correlations.append(_spearman(reference, vector))
        same_best.append(_best_level(reference) == _best_level(vector))
    if not correlations:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "check_id": "test_seeds",
                "check_ru": "Тестовые seed",
                "reference": "joint_filter",
                "num_variants": len(correlations),
                "mean_spearman": float(np.nanmean(correlations)),
                "min_spearman": float(np.nanmin(correlations)),
                "same_best_share": float(np.mean(same_best)),
                "variants": "observation_seed",
            }
        ]
    )


def _frontier_from_loss_frame(losses: pd.DataFrame) -> pd.DataFrame:
    summary = (
        losses.groupby("information_state", as_index=False)["total_loss"]
        .mean()
        .rename(columns={"total_loss": "mean_loss"})
        .set_index("information_state", drop=False)
    )
    rows = []
    for level, state, label in FRONTIER_LEVELS:
        if state == "best_single_distribution":
            try:
                row, chosen = _best_single_distribution(summary)
            except KeyError:
                continue
            rows.append({"information_level": level, "state": chosen, "mean_loss": float(row["mean_loss"])})
        elif state in summary.index:
            rows.append({"information_level": level, "state": state, "mean_loss": float(summary.loc[state, "mean_loss"])})
    return pd.DataFrame(rows)


def _rank_from_losses(frontier: pd.DataFrame) -> dict[int, float] | None:
    if frontier.empty:
        return None
    frame = frontier[frontier["information_level"].isin(RANKING_LEVELS)].copy()
    if frame.empty:
        return None
    frame["rank"] = frame["mean_loss"].rank(method="average", ascending=True)
    return {int(row.information_level): float(row.rank) for row in frame.itertuples()}


def _variant_keys(frontier: pd.DataFrame, families: tuple[str, ...]) -> list[tuple[str, str]]:
    frame = frontier[frontier["family"].isin(families)]
    return list(frame[["family", "variant"]].drop_duplicates().itertuples(index=False, name=None))


def _rank_vector(frontier: pd.DataFrame, family: str, variant: str) -> dict[int, float] | None:
    frame = frontier[(frontier["family"] == family) & (frontier["variant"] == variant)].copy()
    if frame.empty:
        return None
    frame = frame[frame["information_level"].isin(RANKING_LEVELS)].copy()
    if frame.empty:
        return None
    frame["rank"] = frame["mean_loss"].rank(method="average", ascending=True)
    return {int(row.information_level): float(row.rank) for row in frame.itertuples()}


def _spearman(left: dict[int, float], right: dict[int, float]) -> float:
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        return float("nan")
    x = np.asarray([left[key] for key in common], dtype=float)
    y = np.asarray([right[key] for key in common], dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _best_level(rank_vector: dict[int, float]) -> int:
    return int(min(rank_vector, key=lambda key: rank_vector[key]))


def _write_ranking_table(stability: pd.DataFrame, path: Path) -> None:
    display = stability[
        ["check_ru", "reference", "num_variants", "mean_spearman", "min_spearman", "same_best_share"]
    ].copy()
    display = display.rename(
        columns={
            "check_ru": "Проверка",
            "reference": "Ориентир",
            "num_variants": "Число вариантов",
            "mean_spearman": "Средняя ранговая корреляция",
            "min_spearman": "Минимальная ранговая корреляция",
            "same_best_share": "Доля совпадения лучшего уровня",
        }
    )
    for column in ("Средняя ранговая корреляция", "Минимальная ранговая корреляция", "Доля совпадения лучшего уровня"):
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_marginal_table(marginal: pd.DataFrame, path: Path) -> None:
    baseline = marginal[(marginal["family"] == "main") & (marginal["variant"] == "joint_filter")].copy()
    display = baseline[["transition_ru", "marginal_value"]].rename(
        columns={"transition_ru": "Переход", "marginal_value": "Предельное снижение потерь"}
    )
    display["Предельное снижение потерь"] = display["Предельное снижение потерь"].map(lambda value: f"{value:.6f}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(frontier: pd.DataFrame, marginal: pd.DataFrame, stability: pd.DataFrame, path: Path) -> None:
    baseline = frontier[(frontier["family"] == "main") & (frontier["variant"] == "joint_filter")].sort_values("information_level")
    marginal_base = marginal[(marginal["family"] == "main") & (marginal["variant"] == "joint_filter")]
    lines = [
        "# Информационная граница",
        "",
        "Граница строится по шести уровням информации: текущие агрегаты, история агрегатов, фильтрованные агрегаты, один распределительный сигнал, все распределительные сигналы и полная информация.",
        "",
        "## Основная спецификация",
        "",
    ]
    for row in baseline.itertuples():
        lines.append(f"- уровень {row.information_level}: {row.level_label_ru}, потери {row.mean_loss:.6f}.")
    lines.extend(["", "## Предельные выигрыши", ""])
    for row in marginal_base.itertuples():
        lines.append(f"- {row.transition_ru}: {row.marginal_value:.6f}.")
    lines.extend(
        [
            "",
            "## Устойчивость рангов",
            "",
            "Ранговая устойчивость считается по реализуемым информационным состояниям 0--4; полная информация остаётся ориентиром, но не включается в проверку рангов старых sensitivity-прогонов.",
            "",
        ]
    )
    for row in stability.itertuples():
        lines.append(
            f"- {row.check_ru}: средняя ранговая корреляция {row.mean_spearman:.3f}, "
            f"минимальная {row.min_spearman:.3f}, доля совпадения лучшего уровня {row.same_best_share:.3f}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_frontier(frontier: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), sharex=True)
    panels = [
        (axes[0, 0], "main", "Основная граница и проверки"),
        (axes[0, 1], ("noise_aggregate", "noise_distribution"), "Шум наблюдений"),
        (axes[1, 0], "omega", "Клин ликвидной доходности"),
        (axes[1, 1], "income_risk", "Доходный риск"),
    ]
    for ax, families, title in panels:
        if isinstance(families, str):
            panel = frontier[frontier["family"] == families]
        else:
            panel = frontier[frontier["family"].isin(families)]
        if title == "Основная граница и проверки":
            extra = frontier[frontier["family"] == "path_count"]
            panel = pd.concat([panel, extra], ignore_index=True)
        for (_, variant), group in panel.groupby(["family", "variant"], sort=False):
            ordered = group.sort_values("information_level")
            ax.plot(
                ordered["information_level"],
                ordered["mean_loss"],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=str(ordered["line_label_ru"].iloc[0]),
            )
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    x_labels = [label for _, _, label in FRONTIER_LEVELS]
    for ax in axes.ravel():
        ax.set_xticks([level for level, _, _ in FRONTIER_LEVELS])
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Потери J")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_marginal_values(marginal: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    baseline = marginal[(marginal["family"] == "main") & (marginal["variant"] == "joint_filter")].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#c06c2d" if value >= 0 else "#8a8f98" for value in baseline["marginal_value"]]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.bar(np.arange(len(baseline)), baseline["marginal_value"], color=colors, edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(np.arange(len(baseline)))
    ax.set_xticklabels(baseline["transition_ru"], rotation=25, ha="right")
    ax.set_ylabel("Предельное снижение потерь")
    ax.set_title("Предельная ценность следующего уровня информации")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
