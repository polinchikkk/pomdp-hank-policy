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

from experiments.archive.exp23_distributional_identification_battery import (  # noqa: E402
    DISTRIBUTIONAL_FEATURES,
    _align_distribution_aggregates_to_filtered_aggregates,
    _fake_matched_values,
    _fit_rule,
    _replace_distribution_features,
    _rule_rows,
)
from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights  # noqa: E402
from policy.optimize_rules import compare_paired_losses  # noqa: E402


@dataclass(frozen=True)
class NullDistributionChannelSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    replications: int
    null_seed: int
    num_candidates: int
    candidate_seed: int
    continuous_methods: tuple[str, ...]
    num_starts: int
    maxiter: int
    alpha: float
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Negative control: distributional series exist but carry no transmission signal.")
    parser.add_argument("--information-inputs", default="outputs/ssj/stochastic/state_space/information_inputs/information_state_inputs_long.csv")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/null_distribution_channel")
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--null-seed", type=int, default=7301)
    parser.add_argument("--num-candidates", type=int, default=90)
    parser.add_argument("--candidate-seed", type=int, default=7401)
    parser.add_argument("--continuous-methods", default="L-BFGS-B")
    parser.add_argument("--num-starts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_seeds = _parse_seed_range(args.validation_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)
    continuous_methods = tuple(part.strip() for part in args.continuous_methods.split(",") if part.strip())

    source = pd.read_csv(args.information_inputs)
    observables = pd.read_csv(args.hank_observables)
    with np.load(args.jacobians) as bundle:
        jacobians = {key: np.asarray(bundle[key], dtype=float) for key in bundle.files if key.startswith("J_")}

    controlled_source = _align_distribution_aggregates_to_filtered_aggregates(source)
    baseline_environment = HankSSJPolicyEnvironment(
        information_inputs=controlled_source,
        observables=observables,
        jacobians=jacobians,
        loss_weights=PolicyLossWeights(),
    )
    aggregate_fit = _fit_rule(
        environment=baseline_environment,
        information_state="filtered_aggregates",
        validation_seeds=validation_seeds,
        num_candidates=args.num_candidates,
        candidate_seed=args.candidate_seed,
        continuous_methods=continuous_methods,
        num_starts=args.num_starts,
        maxiter=args.maxiter,
    )

    replication_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    rule_rows.extend(_rule_rows("filtered_aggregates", "null_baseline", aggregate_fit.rule, aggregate_fit.validation_loss))

    for replication in range(int(args.replications)):
        print(f"Null replication {replication + 1}/{args.replications}", flush=True)
        fake_values = _fake_matched_values(
            controlled_source,
            validation_seeds=validation_seeds,
            seed=int(args.null_seed) + replication,
        )
        null_inputs = _replace_distribution_features(controlled_source, fake_values)
        environment = HankSSJPolicyEnvironment(
            information_inputs=null_inputs,
            observables=observables,
            jacobians=jacobians,
            loss_weights=PolicyLossWeights(),
        )
        distribution_fit = _fit_rule(
            environment=environment,
            information_state="filtered_distribution",
            validation_seeds=validation_seeds,
            num_candidates=args.num_candidates,
            candidate_seed=int(args.candidate_seed) + 1_000 + replication,
            continuous_methods=continuous_methods,
            num_starts=args.num_starts,
            maxiter=args.maxiter,
        )
        rule_rows.extend(
            _rule_rows(
                "filtered_distribution",
                f"null_replication_{replication:03d}",
                distribution_fit.rule,
                distribution_fit.validation_loss,
            )
        )
        losses = _evaluate_pair(
            environment=environment,
            aggregate_rule=aggregate_fit.rule,
            distribution_rule=distribution_fit.rule,
            replication=replication,
            test_seeds=test_seeds,
        )
        loss_rows.extend(losses.to_dict(orient="records"))
        comparison = compare_paired_losses(
            left_name="filtered_distribution",
            right_name="filtered_aggregates",
            left_losses=losses["loss_filtered_distribution"].to_numpy(dtype=float),
            right_losses=losses["loss_filtered_aggregates"].to_numpy(dtype=float),
            tie_eps=1e-10,
        )
        replication_rows.append(
            {
                "replication": int(replication),
                "loss_filtered_aggregates": float(losses["loss_filtered_aggregates"].mean()),
                "loss_filtered_distribution": float(losses["loss_filtered_distribution"].mean()),
                "mean_delta": comparison.mean_delta,
                "median_delta": comparison.median_delta,
                "loss_reduction": -comparison.mean_delta,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "permutation_p_value": comparison.permutation_p_value,
                "sign_flip_p_value": comparison.sign_flip_p_value,
                "win_rate": comparison.win_rate,
                "tie_rate": comparison.tie_rate,
                "loss_rate": comparison.loss_rate,
                "num_trajectories": comparison.num_trajectories,
                "validation_loss_filtered_aggregates": float(aggregate_fit.validation_loss),
                "validation_loss_filtered_distribution": float(distribution_fit.validation_loss),
                "distribution_optimization_converged": bool(distribution_fit.converged),
                "distribution_optimization_message": distribution_fit.message,
                "bootstrap_false_positive": bool(comparison.ci_high < 0.0),
                "sign_flip_false_positive": bool((comparison.mean_delta < 0.0) and (comparison.sign_flip_p_value < args.alpha)),
                "permutation_false_positive": bool((comparison.mean_delta < 0.0) and (comparison.permutation_p_value < args.alpha)),
            }
        )
        diagnostic_rows.extend(_null_feature_diagnostics(controlled_source, null_inputs, replication))

    replications = pd.DataFrame(replication_rows)
    losses_all = pd.DataFrame(loss_rows)
    rules = pd.DataFrame(rule_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = _summary(replications, alpha=float(args.alpha))

    replications.to_csv(output_dir / "null_distribution_channel_replications.csv", index=False)
    losses_all.to_csv(output_dir / "null_distribution_channel_trajectory_losses.csv", index=False)
    rules.to_csv(output_dir / "null_distribution_channel_fitted_rules.csv", index=False)
    diagnostics.to_csv(output_dir / "null_distribution_feature_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "null_distribution_channel_summary.csv", index=False)
    _write_latex(summary, output_dir / "table_null_distribution_channel.tex")
    _write_report(summary, replications, output_dir / "report_null_distribution_channel.md")

    spec = NullDistributionChannelSpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        replications=int(args.replications),
        null_seed=int(args.null_seed),
        num_candidates=int(args.num_candidates),
        candidate_seed=int(args.candidate_seed),
        continuous_methods=continuous_methods,
        num_starts=int(args.num_starts),
        maxiter=int(args.maxiter),
        alpha=float(args.alpha),
        note=(
            "Negative control: распределительные признаки заменяются искусственными рядами с похожей "
            "дисперсией, авторегрессией и корреляцией с агрегатами. Агрегатный блок filtered_distribution "
            "фиксируется равным filtered_aggregates, поэтому в null-мире меняются только распределительные "
            "признаки, не связанные с HANK/SSJ-трансмиссией."
        ),
    )
    (output_dir / "null_distribution_channel_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'null_distribution_channel_summary.csv'}")
    print(f"Wrote {output_dir / 'report_null_distribution_channel.md'}")


def _evaluate_pair(
    *,
    environment: HankSSJPolicyEnvironment,
    aggregate_rule,
    distribution_rule,
    replication: int,
    test_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in environment.scenarios:
        for seed in test_seeds:
            aggregate_loss = environment.simulate_scenario(
                policy=aggregate_rule,
                information_state="filtered_aggregates",
                scenario=scenario,
                seed=seed,
            )
            distribution_loss = environment.simulate_scenario(
                policy=distribution_rule,
                information_state="filtered_distribution",
                scenario=scenario,
                seed=seed,
            )
            rows.append(
                {
                    "replication": int(replication),
                    "scenario": scenario,
                    "observation_seed": int(seed),
                    "loss_filtered_aggregates": aggregate_loss.total_loss,
                    "loss_filtered_distribution": distribution_loss.total_loss,
                    "delta_distribution_minus_aggregates": distribution_loss.total_loss - aggregate_loss.total_loss,
                    "inflation_delta": distribution_loss.inflation_loss - aggregate_loss.inflation_loss,
                    "output_gap_delta": distribution_loss.output_gap_loss - aggregate_loss.output_gap_loss,
                    "consumption_delta": distribution_loss.consumption_loss - aggregate_loss.consumption_loss,
                    "rate_smoothing_delta": distribution_loss.rate_smoothing_loss - aggregate_loss.rate_smoothing_loss,
                }
            )
    return pd.DataFrame(rows)


def _summary(replications: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "num_replications": int(len(replications)),
                "mean_mvoi": float(replications["loss_reduction"].mean()),
                "median_mvoi": float(replications["loss_reduction"].median()),
                "std_mvoi": float(replications["loss_reduction"].std(ddof=0)),
                "positive_mvoi_share": float((replications["loss_reduction"] > 0).mean()),
                "bootstrap_false_positive_rate": float(replications["bootstrap_false_positive"].mean()),
                "sign_flip_false_positive_rate": float(replications["sign_flip_false_positive"].mean()),
                "permutation_false_positive_rate": float(replications["permutation_false_positive"].mean()),
                "target_false_positive_rate": float(alpha),
                "min_mvoi": float(replications["loss_reduction"].min()),
                "max_mvoi": float(replications["loss_reduction"].max()),
            }
        ]
    )


def _null_feature_diagnostics(source: pd.DataFrame, null_inputs: pd.DataFrame, replication: int) -> list[dict[str, object]]:
    source_wide = _distribution_wide(source)
    null_wide = _distribution_wide(null_inputs)
    rows: list[dict[str, object]] = []
    for feature in DISTRIBUTIONAL_FEATURES:
        source_values = source_wide[feature].to_numpy(dtype=float)
        null_values = null_wide[feature].to_numpy(dtype=float)
        rows.append(
            {
                "replication": int(replication),
                "feature": feature,
                "source_mean": float(np.mean(source_values)),
                "null_mean": float(np.mean(null_values)),
                "source_std": float(np.std(source_values, ddof=0)),
                "null_std": float(np.std(null_values, ddof=0)),
                "corr_with_source": _safe_corr(source_values, null_values),
            }
        )
    return rows


def _distribution_wide(source: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "scenario_label", "period", "observation_seed"]
    mask = source["information_state"].eq("filtered_distribution")
    return (
        source.loc[mask]
        .pivot_table(index=keys, columns="feature_name", values="value", aggfunc="first")
        .reset_index()
        .sort_values(["scenario", "observation_seed", "period"])
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) <= 1e-14 or np.std(right[mask]) <= 1e-14:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _write_latex(summary: pd.DataFrame, path: Path) -> None:
    display = summary.rename(
        columns={
            "num_replications": "Число повторов",
            "mean_mvoi": "Средний MVOI",
            "std_mvoi": "Ст. откл. MVOI",
            "sign_flip_false_positive_rate": "Ложные срабатывания",
            "target_false_positive_rate": "Целевой уровень",
        }
    )
    columns = ["Число повторов", "Средний MVOI", "Ст. откл. MVOI", "Ложные срабатывания", "Целевой уровень"]
    path.write_text(display[columns].to_latex(index=False, float_format="%.6g", escape=False), encoding="utf-8")


def _write_report(summary: pd.DataFrame, replications: pd.DataFrame, path: Path) -> None:
    row = summary.iloc[0]
    lines = [
        "# Negative control: нулевой распределительный канал",
        "",
        "В этом контроле распределительные признаки существуют как шумные ряды с похожей статистикой,",
        "но не связаны с HANK/SSJ-трансмиссией.",
        "",
        f"- число повторов: {int(row['num_replications'])};",
        f"- средний MVOI: {row['mean_mvoi']:.6g};",
        f"- стандартное отклонение MVOI: {row['std_mvoi']:.6g};",
        f"- доля положительных MVOI: {row['positive_mvoi_share']:.3g};",
        f"- false positive rate по sign-flip: {row['sign_flip_false_positive_rate']:.3g};",
        f"- целевой уровень: {row['target_false_positive_rate']:.3g}.",
        "",
        "Первые повторы:",
        "",
        replications.head(10)[["replication", "loss_reduction", "sign_flip_p_value", "bootstrap_false_positive"]].to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
