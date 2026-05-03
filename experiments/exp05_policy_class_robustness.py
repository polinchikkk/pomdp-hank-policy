from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.nonlinear_rules_optional import (
    fit_quadratic_rule,
    project_quadratic_rule_to_information_state,
)
from policy.optimize_rules import compare_paired_losses
from state_space import LocalHANKInformationEnvironment, scenario_config


INFORMATION_STATES = (
    "aggregate_only",
    "filtered_aggregates",
    "distributional",
    "full_information",
)

SCENARIOS = ("baseline", "high_heterogeneity")


def run_policy_class_robustness(
    *,
    output_dir: Path,
    scenarios: tuple[str, ...],
    horizon: int,
    validation_count: int,
    test_count: int,
    num_candidates: int,
    linear_reference_dir: Path | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in scenarios:
        scenario_dir = output_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        result = _run_quadratic_scenario(
            scenario=scenario,
            output_dir=scenario_dir,
            horizon=horizon,
            validation_count=validation_count,
            test_count=test_count,
            num_candidates=num_candidates,
        )
        summary = _scenario_summary(
            scenario=scenario,
            quadratic_losses=result["losses"],
            linear_reference_dir=linear_reference_dir,
        )
        rows.extend(summary.to_dict(orient="records"))

    final = pd.DataFrame(rows)
    final.to_csv(output_dir / "policy_class_robustness_summary.csv", index=False)
    (output_dir / "table_policy_class_robustness.tex").write_text(
        final.to_latex(index=False, float_format="%.6f"),
        encoding="utf-8",
    )
    _write_report(output_dir, final)
    return final


def _run_quadratic_scenario(
    *,
    scenario: str,
    output_dir: Path,
    horizon: int,
    validation_count: int,
    test_count: int,
    num_candidates: int,
) -> dict[str, pd.DataFrame]:
    validation_seeds = list(range(500, 500 + validation_count))
    test_seeds = list(range(900, 900 + test_count))
    environment = LocalHANKInformationEnvironment(scenario_config(scenario, horizon=horizon))

    fitted = {}
    for index, information_state in enumerate(INFORMATION_STATES):
        extra_candidates = []
        if information_state == "distributional" and "filtered_aggregates" in fitted:
            extra_candidates.append(
                project_quadratic_rule_to_information_state(fitted["filtered_aggregates"].rule, "distributional")
            )
        if information_state == "full_information" and "distributional" in fitted:
            extra_candidates.append(
                project_quadratic_rule_to_information_state(fitted["distributional"].rule, "full_information")
            )
        fitted[information_state] = fit_quadratic_rule(
            environment=environment,
            information_state=information_state,
            validation_seeds=validation_seeds,
            num_candidates=num_candidates,
            seed=9101 + 29 * index,
            extra_candidates=extra_candidates,
        )

    losses = _losses_frame(environment, fitted, test_seeds)
    selected = _selected_frame(fitted)
    pairwise = _pairwise_frame(losses)
    losses.to_csv(output_dir / "quadratic_test_losses.csv", index=False)
    selected.to_csv(output_dir / "quadratic_selected_rules.csv", index=False)
    pairwise.to_csv(output_dir / "quadratic_pairwise_value.csv", index=False)
    return {"losses": losses, "selected": selected, "pairwise": pairwise}


def _losses_frame(environment, fitted, test_seeds: list[int]) -> pd.DataFrame:
    rows = []
    for information_state, result in fitted.items():
        for seed in test_seeds:
            sim = environment.simulate(policy=result.rule, information_state=information_state, seed=seed)
            rows.append(
                {
                    "seed": seed,
                    "information_state": information_state,
                    "total_loss": sim.total_loss,
                    "inflation_loss": sim.inflation_loss,
                    "output_loss": sim.output_loss,
                    "rate_loss": sim.rate_loss,
                }
            )
    return pd.DataFrame(rows)


def _selected_frame(fitted) -> pd.DataFrame:
    rows = []
    for information_state, result in fitted.items():
        rows.append(
            {
                "information_state": information_state,
                "validation_loss": result.validation_loss,
                "num_candidates": result.num_candidates,
                "feature_names": list(result.rule.spec.feature_names),
                "linear_coefficients": list(result.rule.linear_coefficients),
                "squared_coefficients": list(result.rule.squared_coefficients),
                "lagged_rate_weight": result.rule.lagged_rate_weight,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_frame(losses: pd.DataFrame) -> pd.DataFrame:
    pivot = losses.pivot(index="seed", columns="information_state", values="total_loss")
    rows = []
    for left, right in (
        ("distributional", "aggregate_only"),
        ("distributional", "filtered_aggregates"),
        ("full_information", "aggregate_only"),
    ):
        comparison = compare_paired_losses(
            left_name=left,
            right_name=right,
            left_losses=pivot[left].to_numpy(dtype=float),
            right_losses=pivot[right].to_numpy(dtype=float),
            tie_eps=1e-12,
        )
        row = asdict(comparison)
        row["loss_reduction"] = -row["mean_delta"]
        row["loss_reduction_ci_low"] = -row["ci_high"]
        row["loss_reduction_ci_high"] = -row["ci_low"]
        rows.append(row)
    return pd.DataFrame(rows)


def _scenario_summary(
    *,
    scenario: str,
    quadratic_losses: pd.DataFrame,
    linear_reference_dir: Path | None,
) -> pd.DataFrame:
    rows = [_class_summary(scenario, "quadratic", quadratic_losses)]
    if linear_reference_dir is not None:
        reference = linear_reference_dir / scenario / "test_losses.csv"
        if reference.exists():
            rows.append(_class_summary(scenario, "linear", pd.read_csv(reference)))
    frame = pd.DataFrame(rows)
    frame["value_vs_aggregate_sign_preserved_vs_linear"] = np.nan
    frame["value_vs_filtered_sign_preserved_vs_linear"] = np.nan
    if set(frame["rule_class"]) == {"linear", "quadratic"}:
        linear = frame[frame["rule_class"] == "linear"].iloc[0]
        quadratic = frame[frame["rule_class"] == "quadratic"].iloc[0]
        aggregate_sign_preserved = np.sign(linear["distributional_value_vs_aggregate"]) == np.sign(
            quadratic["distributional_value_vs_aggregate"]
        )
        filtered_sign_preserved = np.sign(linear["distributional_value_vs_filtered"]) == np.sign(
            quadratic["distributional_value_vs_filtered"]
        )
        frame["value_vs_aggregate_sign_preserved_vs_linear"] = bool(aggregate_sign_preserved)
        frame["value_vs_filtered_sign_preserved_vs_linear"] = bool(filtered_sign_preserved)
    return frame


def _class_summary(scenario: str, rule_class: str, losses: pd.DataFrame) -> dict[str, float | str]:
    pivot = losses.pivot(index="seed", columns="information_state", values="total_loss")
    means = pivot.mean(axis=0)
    dist_vs_aggregate = compare_paired_losses(
        left_name="distributional",
        right_name="aggregate_only",
        left_losses=pivot["distributional"].to_numpy(dtype=float),
        right_losses=pivot["aggregate_only"].to_numpy(dtype=float),
        tie_eps=1e-12,
    )
    dist_vs_filtered = compare_paired_losses(
        left_name="distributional",
        right_name="filtered_aggregates",
        left_losses=pivot["distributional"].to_numpy(dtype=float),
        right_losses=pivot["filtered_aggregates"].to_numpy(dtype=float),
        tie_eps=1e-12,
    )
    full_gap = means["aggregate_only"] - means["full_information"]
    share_closed = np.nan if abs(full_gap) <= 1e-14 else (means["aggregate_only"] - means["distributional"]) / full_gap
    return {
        "scenario": scenario,
        "rule_class": rule_class,
        "aggregate_loss": _clean_zero(float(means["aggregate_only"])),
        "filtered_aggregate_loss": _clean_zero(float(means["filtered_aggregates"])),
        "distributional_loss": _clean_zero(float(means["distributional"])),
        "full_information_loss": _clean_zero(float(means["full_information"])),
        "distributional_value_vs_aggregate": _clean_zero(-float(dist_vs_aggregate.mean_delta)),
        "distributional_value_vs_aggregate_ci_low": _clean_zero(-float(dist_vs_aggregate.ci_high)),
        "distributional_value_vs_aggregate_ci_high": _clean_zero(-float(dist_vs_aggregate.ci_low)),
        "distributional_win_rate_vs_aggregate": _clean_zero(float(dist_vs_aggregate.win_rate)),
        "distributional_value_vs_filtered": _clean_zero(-float(dist_vs_filtered.mean_delta)),
        "distributional_win_rate_vs_filtered": _clean_zero(float(dist_vs_filtered.win_rate)),
        "share_of_full_information_gap_closed": _clean_zero(float(share_closed)),
    }


def _clean_zero(value: float, *, eps: float = 5e-13) -> float:
    if abs(value) <= eps:
        return 0.0
    return value


def _write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Эксперимент 5. Проверка класса правил",
        "",
        "Проверяется, сохраняется ли вывод о ценности распределительной информации при переходе от линейного правила к простому квадратичному правилу.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['scenario']}`, `{row['rule_class']}`: ценность распределительной информации "
            f"`{row['distributional_value_vs_aggregate']:.6f}`, "
            f"доля выигрышных траекторий `{row['distributional_win_rate_vs_aggregate']:.3f}`, "
            f"закрытая доля разрыва `{row['share_of_full_information_gap_closed']:.3f}`."
        )
    lines.extend(
        [
            "",
            "Ключевая проверка здесь не в том, что квадратичное правило должно стать новым основным правилом, а в том, сохраняется ли знак ценности распределительной информации при расширении класса правил.",
        ]
    )
    (output_dir / "report_exp05_policy_class_robustness.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run policy-class robustness checks.")
    parser.add_argument("--output-dir", default="outputs/exp05_policy_class_robustness")
    parser.add_argument("--linear-reference-dir", default="outputs/exp02_distributional_value")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--validation-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=220)
    args = parser.parse_args()

    run_policy_class_robustness(
        output_dir=Path(args.output_dir),
        scenarios=SCENARIOS,
        horizon=args.horizon,
        validation_count=args.validation_count,
        test_count=args.test_count,
        num_candidates=args.num_candidates,
        linear_reference_dir=Path(args.linear_reference_dir),
    )


if __name__ == "__main__":
    main()
