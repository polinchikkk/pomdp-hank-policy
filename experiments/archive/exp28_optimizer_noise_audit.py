from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.archive.exp08_main_voi import INFORMATION_STATES, STATE_LABEL_RU, _supervised_candidates
from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights
from policy.fit_linear_rules import fit_linear_rule
from policy.linear_rules import LinearRule, coefficient_vector, rule_spec_for_information_state
from policy.optimize_linear_rules import (
    LinearRuleOptimizationBounds,
    fit_linear_rule_continuous,
)


@dataclass(frozen=True)
class OptimizerNoiseAuditSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    optimizer_seeds: tuple[int, ...]
    num_starts_list: tuple[int, ...]
    maxiter_list: tuple[int, ...]
    continuous_methods: tuple[str, ...]
    information_states: tuple[str, ...]
    warmup_candidates: int
    intercept_bound: float
    standardized_coefficient_bound: float
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether the value-of-information effect exceeds optimizer noise.")
    parser.add_argument(
        "--information-inputs",
        default="outputs/ssj/stochastic/state_space/information_inputs/information_state_inputs_long.csv",
    )
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/optimizer_noise_audit")
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--optimizer-seeds", default="1000:1004")
    parser.add_argument("--num-starts-list", default="1,5")
    parser.add_argument("--maxiter-list", default="12,50")
    parser.add_argument("--continuous-methods", default="L-BFGS-B")
    parser.add_argument("--information-states", default=",".join(INFORMATION_STATES))
    parser.add_argument("--warmup-candidates", type=int, default=64)
    parser.add_argument("--intercept-bound", type=float, default=0.01)
    parser.add_argument("--standardized-coefficient-bound", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_seeds = _parse_seed_range(args.validation_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)
    optimizer_seeds = _parse_seed_range(args.optimizer_seeds)
    num_starts_list = tuple(int(item) for item in args.num_starts_list.split(",") if item.strip())
    maxiter_list = tuple(int(item) for item in args.maxiter_list.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.continuous_methods.split(",") if item.strip())
    information_states = tuple(item.strip() for item in args.information_states.split(",") if item.strip())

    environment = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=Path(args.information_inputs),
        hank_observables_csv=Path(args.hank_observables),
        jacobians_npz=Path(args.jacobians),
        loss_weights=PolicyLossWeights(),
    )
    bounds = LinearRuleOptimizationBounds(
        intercept_abs_bound=args.intercept_bound,
        standardized_coefficient_abs_bound=args.standardized_coefficient_bound,
    )
    extra_candidates = {
        state: _supervised_candidates(
            environment=environment,
            information_state=state,
            validation_seeds=validation_seeds,
        )
        for state in information_states
    }

    run_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    mvoi_rows: list[dict[str, object]] = []
    total_configs = len(num_starts_list) * len(maxiter_list) * len(optimizer_seeds)
    config_index = 0
    for num_starts in num_starts_list:
        for maxiter in maxiter_list:
            for optimizer_seed in optimizer_seeds:
                config_index += 1
                print(
                    f"Audit config {config_index}/{total_configs}: "
                    f"seed={optimizer_seed}, starts={num_starts}, maxiter={maxiter}",
                    flush=True,
                )
                fits: dict[str, dict[str, object]] = {}
                for state_index, information_state in enumerate(information_states):
                    fit = _fit_one_state(
                        environment=environment,
                        information_state=information_state,
                        validation_seeds=validation_seeds,
                        test_seeds=test_seeds,
                        optimizer_seed=optimizer_seed + 10_000 * state_index,
                        num_starts=num_starts,
                        maxiter=maxiter,
                        methods=methods,
                        bounds=bounds,
                        warmup_candidates=args.warmup_candidates,
                        extra_candidates=extra_candidates[information_state],
                    )
                    fits[information_state] = fit
                    run_rows.append(
                        {
                            "optimizer_seed": int(optimizer_seed),
                            "num_starts": int(num_starts),
                            "maxiter": int(maxiter),
                            "methods": ",".join(methods),
                            "information_state": information_state,
                            "information_state_ru": STATE_LABEL_RU.get(information_state, information_state),
                            "validation_loss": fit["validation_loss"],
                            "test_loss": fit["test_loss"],
                            "best_method": fit["best_method"],
                            "best_start_index": fit["best_start_index"],
                            "converged": fit["converged"],
                            "num_function_evaluations": fit["num_function_evaluations"],
                            "message": fit["message"],
                        }
                    )
                    coefficient_rows.extend(
                        _coefficient_rows(
                            information_state=information_state,
                            optimizer_seed=optimizer_seed,
                            num_starts=num_starts,
                            maxiter=maxiter,
                            methods=methods,
                            rule=fit["rule"],
                        )
                    )
                best_state = min(fits, key=lambda state: float(fits[state]["test_loss"]))
                best_rows.append(
                    {
                        "optimizer_seed": int(optimizer_seed),
                        "num_starts": int(num_starts),
                        "maxiter": int(maxiter),
                        "methods": ",".join(methods),
                        "best_information_state": best_state,
                        "best_information_state_ru": STATE_LABEL_RU.get(best_state, best_state),
                        "best_test_loss": float(fits[best_state]["test_loss"]),
                    }
                )
                if "filtered_aggregates" in fits and "filtered_distribution" in fits:
                    mvoi = float(fits["filtered_aggregates"]["test_loss"]) - float(fits["filtered_distribution"]["test_loss"])
                    mvoi_rows.append(
                        {
                            "optimizer_seed": int(optimizer_seed),
                            "num_starts": int(num_starts),
                            "maxiter": int(maxiter),
                            "methods": ",".join(methods),
                            "loss_filtered_aggregates": float(fits["filtered_aggregates"]["test_loss"]),
                            "loss_filtered_distribution": float(fits["filtered_distribution"]["test_loss"]),
                            "mvoi_dist": mvoi,
                            "mvoi_positive": bool(mvoi > 0.0),
                        }
                    )

    runs = pd.DataFrame(run_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    best = pd.DataFrame(best_rows)
    mvoi = pd.DataFrame(mvoi_rows)
    state_summary = _state_summary(runs, coefficients, best)
    mvoi_summary = _mvoi_summary(mvoi)
    protection = _protection_summary(mvoi_summary)

    runs.to_csv(output_dir / "optimizer_noise_runs.csv", index=False)
    coefficients.to_csv(output_dir / "optimizer_noise_coefficients.csv", index=False)
    best.to_csv(output_dir / "optimizer_noise_best_state.csv", index=False)
    mvoi.to_csv(output_dir / "optimizer_noise_mvoi_distribution.csv", index=False)
    state_summary.to_csv(output_dir / "optimizer_noise_state_summary.csv", index=False)
    mvoi_summary.to_csv(output_dir / "optimizer_noise_mvoi_summary.csv", index=False)
    protection.to_csv(output_dir / "optimizer_noise_protection_summary.csv", index=False)
    _write_state_table(state_summary, output_dir / "table_optimizer_noise_state_summary.tex")
    _write_mvoi_table(mvoi_summary, output_dir / "table_optimizer_noise_mvoi_summary.tex")
    _write_report(mvoi_summary, protection, output_dir / "report_optimizer_noise_audit.md")
    _plot_mvoi(mvoi, mvoi_summary, Path("article/figures/fig_optimizer_noise_mvoi.pdf"))

    spec = OptimizerNoiseAuditSpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        optimizer_seeds=tuple(optimizer_seeds),
        num_starts_list=tuple(num_starts_list),
        maxiter_list=tuple(maxiter_list),
        continuous_methods=tuple(methods),
        information_states=tuple(information_states),
        warmup_candidates=int(args.warmup_candidates),
        intercept_bound=float(args.intercept_bound),
        standardized_coefficient_bound=float(args.standardized_coefficient_bound),
        note=(
            "Audit of optimizer-induced variation. The protected-result criterion compares "
            "mean MVOI to the standard deviation of MVOI across optimizer seeds."
        ),
    )
    (output_dir / "optimizer_noise_audit_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'optimizer_noise_mvoi_summary.csv'}")
    print("Wrote article/figures/fig_optimizer_noise_mvoi.pdf")


def _fit_one_state(
    *,
    environment: HankSSJPolicyEnvironment,
    information_state: str,
    validation_seeds: list[int],
    test_seeds: list[int],
    optimizer_seed: int,
    num_starts: int,
    maxiter: int,
    methods: tuple[str, ...],
    bounds: LinearRuleOptimizationBounds,
    warmup_candidates: int,
    extra_candidates: list[LinearRule],
) -> dict[str, object]:
    warmup = fit_linear_rule(
        environment=environment,
        information_state=information_state,
        validation_seeds=validation_seeds,
        num_candidates=warmup_candidates,
        seed=optimizer_seed,
        extra_candidates=extra_candidates,
    )
    initial_rules = [warmup.rule, *extra_candidates]
    fit = fit_linear_rule_continuous(
        environment=environment,
        information_state=information_state,
        validation_seeds=validation_seeds,
        feature_scales=warmup.feature_scales,
        initial_rules=initial_rules,
        seed=optimizer_seed + 20_000,
        num_starts=num_starts,
        methods=methods,
        bounds=bounds,
        maxiter=maxiter,
    )
    test_losses = [
        environment.simulate(policy=fit.rule, information_state=information_state, seed=seed).total_loss
        for seed in test_seeds
    ]
    return {
        "rule": fit.rule,
        "validation_loss": float(fit.validation_loss),
        "test_loss": float(np.mean(test_losses)),
        "best_method": fit.best_method,
        "best_start_index": int(fit.best_start_index),
        "converged": bool(fit.converged),
        "num_function_evaluations": int(fit.num_function_evaluations),
        "message": fit.message,
    }


def _coefficient_rows(
    *,
    information_state: str,
    optimizer_seed: int,
    num_starts: int,
    maxiter: int,
    methods: tuple[str, ...],
    rule: LinearRule,
) -> list[dict[str, object]]:
    vector = coefficient_vector(rule)
    terms = ("intercept", *rule.spec.feature_names, "lagged_rate")
    return [
        {
            "optimizer_seed": int(optimizer_seed),
            "num_starts": int(num_starts),
            "maxiter": int(maxiter),
            "methods": ",".join(methods),
            "information_state": information_state,
            "information_state_ru": STATE_LABEL_RU.get(information_state, information_state),
            "term": term,
            "coefficient": float(coefficient),
        }
        for term, coefficient in zip(terms, vector)
    ]


def _state_summary(runs: pd.DataFrame, coefficients: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_keys = ["num_starts", "maxiter", "methods", "information_state", "information_state_ru"]
    best_freq = (
        best.groupby(["num_starts", "maxiter", "methods", "best_information_state"])
        .size()
        .rename("best_count")
        .reset_index()
    )
    config_counts = best.groupby(["num_starts", "maxiter", "methods"]).size().rename("num_optimizer_runs").reset_index()
    best_freq = best_freq.merge(config_counts, on=["num_starts", "maxiter", "methods"], how="left")
    best_freq["frequency_same_best_state"] = best_freq["best_count"] / best_freq["num_optimizer_runs"]
    for keys, frame in runs.groupby(group_keys, sort=False):
        num_starts, maxiter, methods, information_state, information_state_ru = keys
        coeff_frame = coefficients[
            (coefficients["num_starts"] == num_starts)
            & (coefficients["maxiter"] == maxiter)
            & (coefficients["methods"] == methods)
            & (coefficients["information_state"] == information_state)
        ]
        coefficient_std = coeff_frame.groupby("term")["coefficient"].std(ddof=1).fillna(0.0)
        freq_row = best_freq[
            (best_freq["num_starts"] == num_starts)
            & (best_freq["maxiter"] == maxiter)
            & (best_freq["methods"] == methods)
            & (best_freq["best_information_state"] == information_state)
        ]
        frequency = float(freq_row["frequency_same_best_state"].iloc[0]) if not freq_row.empty else 0.0
        rows.append(
            {
                "num_starts": int(num_starts),
                "maxiter": int(maxiter),
                "methods": methods,
                "information_state": information_state,
                "information_state_ru": information_state_ru,
                "num_optimizer_runs": int(len(frame)),
                "mean_validation_loss": float(frame["validation_loss"].mean()),
                "mean_test_loss": float(frame["test_loss"].mean()),
                "std_test_loss_across_optimizer_seeds": float(frame["test_loss"].std(ddof=1)) if len(frame) > 1 else 0.0,
                "mean_abs_coefficient_std": float(coefficient_std.abs().mean()) if not coefficient_std.empty else 0.0,
                "coefficient_std_l2": float(np.sqrt(np.sum(coefficient_std.to_numpy(dtype=float) ** 2))) if not coefficient_std.empty else 0.0,
                "frequency_same_best_state": frequency,
                "convergence_rate": float(frame["converged"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _mvoi_summary(mvoi: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, frame in mvoi.groupby(["num_starts", "maxiter", "methods"], sort=False):
        num_starts, maxiter, methods = keys
        values = frame["mvoi_dist"].to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        rows.append(
            {
                "num_starts": int(num_starts),
                "maxiter": int(maxiter),
                "methods": methods,
                "num_optimizer_runs": int(values.size),
                "mean_mvoi": mean,
                "std_mvoi_across_optimizer_seeds": std,
                "min_mvoi": float(np.min(values)),
                "p10_mvoi": float(np.quantile(values, 0.10)),
                "median_mvoi": float(np.median(values)),
                "p90_mvoi": float(np.quantile(values, 0.90)),
                "max_mvoi": float(np.max(values)),
                "positive_share": float(np.mean(values > 0.0)),
                "effect_to_optimizer_noise_ratio": mean / std if std > 1e-14 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def _protection_summary(mvoi_summary: pd.DataFrame) -> pd.DataFrame:
    if mvoi_summary.empty:
        return pd.DataFrame()
    frame = mvoi_summary.copy()
    frame["protected_5x"] = frame["effect_to_optimizer_noise_ratio"] >= 5.0
    frame["protected_10x"] = frame["effect_to_optimizer_noise_ratio"] >= 10.0
    frame["positive_in_all_optimizer_runs"] = frame["positive_share"] >= 1.0
    return frame[
        [
            "num_starts",
            "maxiter",
            "methods",
            "mean_mvoi",
            "std_mvoi_across_optimizer_seeds",
            "effect_to_optimizer_noise_ratio",
            "positive_share",
            "protected_5x",
            "protected_10x",
            "positive_in_all_optimizer_runs",
        ]
    ]


def _write_state_table(summary: pd.DataFrame, path: Path) -> None:
    display = summary[
        [
            "num_starts",
            "maxiter",
            "information_state_ru",
            "mean_test_loss",
            "std_test_loss_across_optimizer_seeds",
            "frequency_same_best_state",
            "convergence_rate",
        ]
    ].copy()
    display = display.rename(
        columns={
            "num_starts": "Стартов",
            "maxiter": "Итераций",
            "information_state_ru": "Информационное состояние",
            "mean_test_loss": "Средние тестовые потери",
            "std_test_loss_across_optimizer_seeds": "Разброс по seed оптимизатора",
            "frequency_same_best_state": "Частота лучшего состояния",
            "convergence_rate": "Доля сходимости",
        }
    )
    for column in ("Средние тестовые потери", "Разброс по seed оптимизатора"):
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    for column in ("Частота лучшего состояния", "Доля сходимости"):
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_mvoi_table(summary: pd.DataFrame, path: Path) -> None:
    display = summary[
        [
            "num_starts",
            "maxiter",
            "mean_mvoi",
            "std_mvoi_across_optimizer_seeds",
            "min_mvoi",
            "max_mvoi",
            "positive_share",
            "effect_to_optimizer_noise_ratio",
        ]
    ].copy()
    display = display.rename(
        columns={
            "num_starts": "Стартов",
            "maxiter": "Итераций",
            "mean_mvoi": "Средний MVOI",
            "std_mvoi_across_optimizer_seeds": "Разброс MVOI",
            "min_mvoi": "Минимум",
            "max_mvoi": "Максимум",
            "positive_share": "Доля положительных",
            "effect_to_optimizer_noise_ratio": "Эффект / шум",
        }
    )
    for column in ("Средний MVOI", "Разброс MVOI", "Минимум", "Максимум"):
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    display["Доля положительных"] = display["Доля положительных"].map(lambda value: f"{value:.3f}")
    display["Эффект / шум"] = display["Эффект / шум"].map(lambda value: "inf" if np.isinf(value) else f"{value:.2f}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(mvoi_summary: pd.DataFrame, protection: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Аудит численного шума оптимизации",
        "",
        "Проверяется, насколько оценка MVOI_dist зависит от seed оптимизатора, числа стартов и числа итераций.",
        "",
    ]
    for row in mvoi_summary.itertuples():
        ratio = "inf" if np.isinf(row.effect_to_optimizer_noise_ratio) else f"{row.effect_to_optimizer_noise_ratio:.2f}"
        lines.append(
            f"- starts={row.num_starts}, maxiter={row.maxiter}: "
            f"mean MVOI={row.mean_mvoi:.6f}, std={row.std_mvoi_across_optimizer_seeds:.6f}, "
            f"range=[{row.min_mvoi:.6f}, {row.max_mvoi:.6f}], positive share={row.positive_share:.3f}, "
            f"effect/noise={ratio}."
        )
    if not protection.empty:
        best = protection.sort_values("effect_to_optimizer_noise_ratio", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                "## Краткий вывод",
                "",
                (
                    f"Наиболее защищённая конфигурация: starts={int(best['num_starts'])}, "
                    f"maxiter={int(best['maxiter'])}, effect/noise="
                    f"{best['effect_to_optimizer_noise_ratio']:.2f}."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_mvoi(mvoi: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if mvoi.empty:
        return
    frame = mvoi.copy()
    frame["config"] = frame["num_starts"].astype(str) + " starts, " + frame["maxiter"].astype(str) + " iter"
    configs = list(dict.fromkeys(frame["config"].tolist()))
    data = [frame[frame["config"] == config]["mvoi_dist"].to_numpy(dtype=float) for config in configs]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.boxplot(data, tick_labels=configs, showmeans=True)
    ax.set_ylabel("MVOI распределительной информации")
    ax.set_title("Разброс MVOI по seed оптимизатора")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
