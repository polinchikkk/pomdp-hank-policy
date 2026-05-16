from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.inference import summarize_paired_inference  # noqa: E402


MAIN_COMPARISON = "filtered_distribution_minus_filtered_aggregates"
LQG_SIMPLE_COMPARISON = "simple_filtered_distribution_minus_simple_filtered_aggregates"
LQG_ORACLE_COMPARISON = "lqg_distribution_observations_minus_lqg_aggregate_observations"
LQG_AGGREGATE_GAP_COMPARISON = "lqg_aggregate_observations_minus_simple_filtered_aggregates"
LQG_DISTRIBUTION_GAP_COMPARISON = "lqg_distribution_observations_minus_simple_filtered_distribution"
LQG_CONTROLLER_ORDER = (
    "simple_filtered_aggregates",
    "simple_filtered_distribution",
    "lqg_aggregate_observations",
    "lqg_distribution_observations",
    "lqr_full_state",
)
LQG_CONTROLLER_LABELS = {
    "simple_filtered_aggregates": "simple filtered aggregates",
    "simple_filtered_distribution": "simple filtered distribution",
    "lqg_aggregate_observations": "LQG aggregate observations",
    "lqg_distribution_observations": "LQG aggregate + distribution observations",
    "lqr_full_state": "LQR full information",
}


@dataclass(frozen=True)
class FinalVOIProtocolSpec:
    open_loop_dir: str
    closed_loop_dir: str
    lqg_oracle_dir: str
    output_dir: str
    comparison: str
    closed_loop_mode: str
    scenario: str
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final open-loop vs closed-loop VOI protocol tables.")
    parser.add_argument("--open-loop-dir", default="outputs/ssj/stochastic/large_sample")
    parser.add_argument("--closed-loop-dir", default="outputs/ssj/stochastic/closed_loop_distributional_ssj")
    parser.add_argument("--lqg-oracle-dir", default="outputs/ssj/stochastic/lqg_oracle")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/final_voi_protocol")
    parser.add_argument("--comparison", default=MAIN_COMPARISON)
    parser.add_argument("--closed-loop-mode", default="closed_loop_local_projection")
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--allow-missing-closed-loop-gate", action="store_true")
    parser.add_argument("--allow-missing-lqg-oracle", action="store_true")
    parser.add_argument("--lqg-near-zero-tol", type=float, default=1e-12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    open_loop_dir = Path(args.open_loop_dir)
    closed_loop_dir = Path(args.closed_loop_dir)
    lqg_oracle_dir = Path(args.lqg_oracle_dir)
    _require_file(open_loop_dir / "trajectory_losses.csv")
    _require_file(closed_loop_dir / "trajectory_losses_closed_loop.csv")
    _require_file(closed_loop_dir / "convergence_diagnostics.csv")

    open_loop_losses = pd.read_csv(open_loop_dir / "trajectory_losses.csv")
    closed_loop_losses = pd.read_csv(closed_loop_dir / "trajectory_losses_closed_loop.csv")
    closed_loop_diagnostics = pd.read_csv(closed_loop_dir / "convergence_diagnostics.csv")
    lqg_summary, lqg_pairwise = _load_lqg_oracle(
        lqg_oracle_dir,
        allow_missing=bool(args.allow_missing_lqg_oracle),
    )

    hierarchical_open_row = _load_hierarchical_inference_row(
        open_loop_dir / "hierarchical_inference.csv",
        comparison=args.comparison,
    )
    open_row = _evaluation_row(
        evaluation="open_loop_fixed_path",
        losses=open_loop_losses,
        diagnostics=None,
        comparison=args.comparison,
        scenario=args.scenario,
        mode="fixed_path",
        hierarchical_inference=hierarchical_open_row,
    )
    closed_row = _evaluation_row(
        evaluation="closed_loop_local_projection",
        losses=closed_loop_losses[closed_loop_losses["mode"].eq(args.closed_loop_mode)].copy(),
        diagnostics=closed_loop_diagnostics[closed_loop_diagnostics["mode"].eq(args.closed_loop_mode)].copy(),
        comparison=args.comparison,
        scenario=args.scenario,
        mode=args.closed_loop_mode,
        hierarchical_inference=None,
    )
    comparison = pd.DataFrame([open_row, closed_row])
    side_by_side = _side_by_side(comparison)
    lqg_benchmark = _lqg_benchmark_table(lqg_summary)
    lqg_diagnostics = _lqg_oracle_diagnostics(
        lqg_pairwise=lqg_pairwise,
        open_row=open_row,
        near_zero_tol=float(args.lqg_near_zero_tol),
    )
    gate = _final_gate(
        open_row=open_row,
        closed_row=closed_row,
        closed_loop_gate_json=closed_loop_dir / "final_protocol_gate.json",
        allow_missing_closed_loop_gate=bool(args.allow_missing_closed_loop_gate),
        allow_missing_lqg_oracle=bool(args.allow_missing_lqg_oracle),
        lqg_diagnostics=lqg_diagnostics,
    )

    comparison.to_csv(output_dir / "open_loop_vs_closed_loop_main_pair.csv", index=False)
    side_by_side.to_csv(output_dir / "table_open_loop_vs_closed_loop_side_by_side.csv", index=False)
    lqg_benchmark.to_csv(output_dir / "table_lqg_oracle_benchmark.csv", index=False)
    pd.DataFrame([lqg_diagnostics]).to_csv(output_dir / "lqg_oracle_diagnostics.csv", index=False)
    (output_dir / "final_voi_protocol_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    spec = FinalVOIProtocolSpec(
        open_loop_dir=str(open_loop_dir),
        closed_loop_dir=str(closed_loop_dir),
        lqg_oracle_dir=str(lqg_oracle_dir),
        output_dir=str(output_dir),
        comparison=args.comparison,
        closed_loop_mode=args.closed_loop_mode,
        scenario=args.scenario,
        note=(
            "Large-sample open-loop/fixed-path VOI is treated as the primary estimate. "
            "Closed-loop local projection is the main credibility check because the rule feeds back "
            "through local HANK/SSJ dynamics, observations, filters, and the next policy rate. "
            "The LQG/Riccati oracle is mandatory as a methodological anchor for the same linear "
            "state-space information problem."
        ),
    )
    (output_dir / "final_voi_protocol_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_latex_side_by_side(side_by_side, output_dir / "table_final_open_closed_loop_side_by_side.tex")
    _write_latex_lqg_benchmark(lqg_benchmark, output_dir / "table_lqg_oracle_benchmark.tex")
    _write_report(
        comparison=comparison,
        side_by_side=side_by_side,
        lqg_benchmark=lqg_benchmark,
        lqg_diagnostics=lqg_diagnostics,
        gate=gate,
        output_path=output_dir / "report_final_voi_protocol.md",
    )
    print(f"Wrote {output_dir / 'open_loop_vs_closed_loop_main_pair.csv'}")
    print(f"Wrote {output_dir / 'table_lqg_oracle_benchmark.csv'}")
    print(f"Wrote {output_dir / 'final_voi_protocol_gate.json'}")
    print(f"Wrote {output_dir / 'report_final_voi_protocol.md'}")
    if not gate["passed"]:
        failed = ", ".join(check["name"] for check in gate["checks"] if not check["passed"])
        raise RuntimeError(f"Final VOI protocol failed: {failed}")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required final protocol artifact is missing: {path}")


def _load_lqg_oracle(lqg_oracle_dir: Path, *, allow_missing: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = lqg_oracle_dir / "lqg_oracle_summary.csv"
    pairwise_path = lqg_oracle_dir / "lqg_oracle_pairwise.csv"
    missing = [path for path in (summary_path, pairwise_path) if not path.exists()]
    if missing:
        if allow_missing:
            return pd.DataFrame(), pd.DataFrame()
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required LQG/Riccati oracle artifact is missing: {missing_list}")
    return pd.read_csv(summary_path), pd.read_csv(pairwise_path)


def _evaluation_row(
    *,
    evaluation: str,
    losses: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    comparison: str,
    scenario: str,
    mode: str,
    hierarchical_inference: pd.Series | None,
) -> dict[str, object]:
    left, right = _parse_comparison(comparison)
    frame = losses.copy()
    if scenario != "all":
        frame = frame[frame["scenario"].eq(scenario)].copy()
    if frame.empty:
        raise ValueError(f"No losses for evaluation={evaluation}, scenario={scenario}.")
    pivot_index = ["scenario", "observation_seed"]
    if "mode" in frame.columns:
        pivot_index = ["mode", *pivot_index]
    pivot = frame.pivot_table(
        index=pivot_index,
        columns="information_state",
        values="total_loss",
        aggfunc="first",
    ).reset_index()
    missing = [state for state in (left, right) if state not in pivot.columns]
    if missing:
        raise ValueError(f"Loss table for {evaluation} is missing states: {missing}")
    delta = pivot[left].to_numpy(dtype=float) - pivot[right].to_numpy(dtype=float)
    cluster_id = np.asarray([_shock_seed_from_scenario(value) for value in pivot["scenario"]], dtype=int)
    inference = summarize_paired_inference(delta, cluster_id=cluster_id, n_boot=4_000, n_perm=4_000, tie_eps=1e-10)
    stability_penalty = _pair_stability_penalty(frame, states=(left, right))
    diagnostic_metrics = _diagnostic_metrics(diagnostics, states=(left, right), scenario=scenario)
    cluster_ci_low = float(inference.clustered_ci_low)
    cluster_ci_high = float(inference.clustered_ci_high)
    inference_source = "clustered_shock_seed"
    if hierarchical_inference is not None:
        inference_source = str(hierarchical_inference.get("primary_inference", "hierarchical_inference"))
        mean_delta = float(hierarchical_inference["mean_delta"])
        cluster_ci_low = float(hierarchical_inference["hierarchical_ci_low"])
        cluster_ci_high = float(hierarchical_inference["hierarchical_ci_high"])
        sign_flip_p_value = float(hierarchical_inference["sign_flip_p_value"])
        win_rate = float(hierarchical_inference["cluster_win_share"])
        num_shock_clusters = int(hierarchical_inference["num_shock_clusters"])
        num_trajectories = int(hierarchical_inference["num_observation_subclusters"])
    else:
        mean_delta = float(inference.mean_delta)
        sign_flip_p_value = float(inference.sign_flip_p_value)
        win_rate = float(inference.win_rate)
        num_shock_clusters = int(inference.num_clusters)
        num_trajectories = int(inference.num_observations)
    return {
        "evaluation": evaluation,
        "mode": mode,
        "scenario": scenario,
        "comparison": comparison,
        "inference_source": inference_source,
        "left": left,
        "right": right,
        "num_trajectories": num_trajectories,
        "num_shock_clusters": num_shock_clusters,
        "mean_delta": mean_delta,
        "loss_reduction": float(-mean_delta),
        "cluster_ci_low": cluster_ci_low,
        "cluster_ci_high": cluster_ci_high,
        "loss_reduction_cluster_ci_low": float(-cluster_ci_high),
        "loss_reduction_cluster_ci_high": float(-cluster_ci_low),
        "sign_flip_p_value": sign_flip_p_value,
        "win_rate": win_rate,
        "tie_rate": float(inference.tie_rate),
        "loss_rate": float(inference.loss_rate),
        "convergence_failure_rate": diagnostic_metrics["convergence_failure_rate"],
        "max_spectral_radius_local_loop": diagnostic_metrics["max_spectral_radius_local_loop"],
        "mean_stability_penalty": float(stability_penalty),
        "diagnostic_stability_penalty": diagnostic_metrics["mean_stability_penalty"],
    }


def _load_hierarchical_inference_row(path: Path, *, comparison: str) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    rows = frame[frame["comparison"].eq(comparison)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _parse_comparison(comparison: str) -> tuple[str, str]:
    marker = "_minus_"
    if marker not in comparison:
        raise ValueError(f"Comparison must have '<left>_minus_<right>' form: {comparison}")
    left, right = comparison.split(marker, maxsplit=1)
    return left, right


def _pair_stability_penalty(losses: pd.DataFrame, *, states: tuple[str, str]) -> float:
    frame = losses[losses["information_state"].isin(states)]
    if frame.empty or "stability_penalty" not in frame.columns:
        return float("nan")
    return float(frame["stability_penalty"].mean())


def _diagnostic_metrics(
    diagnostics: pd.DataFrame | None,
    *,
    states: tuple[str, str],
    scenario: str,
) -> dict[str, float | None]:
    if diagnostics is None or diagnostics.empty:
        return {
            "convergence_failure_rate": None,
            "max_spectral_radius_local_loop": None,
            "mean_stability_penalty": None,
        }
    frame = diagnostics[diagnostics["information_state"].isin(states)].copy()
    if scenario != "all":
        frame = frame[frame["scenario"].eq(scenario)].copy()
    if frame.empty:
        return {
            "convergence_failure_rate": None,
            "max_spectral_radius_local_loop": None,
            "mean_stability_penalty": None,
        }
    return {
        "convergence_failure_rate": float(1.0 - frame["converged"].mean()),
        "max_spectral_radius_local_loop": float(frame["spectral_radius_local_loop"].max()),
        "mean_stability_penalty": float(frame["stability_penalty"].mean()),
    }


def _side_by_side(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("mean_delta", "Mean delta"),
        ("cluster_ci_low", "Cluster CI low"),
        ("cluster_ci_high", "Cluster CI high"),
        ("sign_flip_p_value", "Sign-flip p-value"),
        ("win_rate", "Win rate"),
        ("convergence_failure_rate", "Convergence failure rate"),
        ("max_spectral_radius_local_loop", "Max spectral radius local loop"),
        ("mean_stability_penalty", "Mean stability penalty"),
    ]
    open_row = comparison[comparison["evaluation"].eq("open_loop_fixed_path")].iloc[0]
    closed_row = comparison[comparison["evaluation"].eq("closed_loop_local_projection")].iloc[0]
    return pd.DataFrame(
        [
            {
                "metric": key,
                "metric_label": label,
                "A_open_loop_fixed_path": open_row[key],
                "B_closed_loop_local_projection": closed_row[key],
            }
            for key, label in rows
        ]
    )


def _lqg_benchmark_table(lqg_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "controller",
        "controller_label",
        "controller_ru",
        "num_trajectories",
        "mean_loss",
        "ci_low",
        "ci_high",
        "median_loss",
        "mean_abs_rate",
        "max_abs_rate",
    ]
    if lqg_summary.empty:
        return pd.DataFrame(columns=columns)
    order = {controller: index for index, controller in enumerate(LQG_CONTROLLER_ORDER)}
    frame = lqg_summary[lqg_summary["controller"].isin(order)].copy()
    missing = [controller for controller in LQG_CONTROLLER_ORDER if controller not in set(frame["controller"])]
    if missing:
        raise ValueError(f"LQG oracle summary is missing controllers required by final protocol: {missing}")
    frame["controller_label"] = frame["controller"].map(LQG_CONTROLLER_LABELS)
    frame = frame.sort_values("controller", key=lambda col: col.map(order)).reset_index(drop=True)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def _lqg_oracle_diagnostics(
    *,
    lqg_pairwise: pd.DataFrame,
    open_row: dict[str, object],
    near_zero_tol: float,
) -> dict[str, object]:
    if lqg_pairwise.empty:
        return {
            "lqg_oracle_available": False,
            "benchmark_interpretation": "lqg_oracle_missing",
        }

    simple_pair = _pairwise_row(lqg_pairwise, LQG_SIMPLE_COMPARISON)
    lqg_pair = _pairwise_row(lqg_pairwise, LQG_ORACLE_COMPARISON)
    aggregate_gap_pair = _pairwise_row(lqg_pairwise, LQG_AGGREGATE_GAP_COMPARISON)
    distribution_gap_pair = _pairwise_row(lqg_pairwise, LQG_DISTRIBUTION_GAP_COMPARISON)

    simple_final_mvoi = float(open_row["loss_reduction"])
    simple_final_ci_includes_zero = _delta_ci_includes_zero(open_row)
    simple_lqg_mvoi = float(simple_pair["loss_reduction"])
    simple_lqg_ci_includes_zero = _delta_ci_includes_zero(simple_pair)
    lqg_mvoi = float(lqg_pair["loss_reduction"])
    lqg_ci_includes_zero = _delta_ci_includes_zero(lqg_pair)
    aggregate_gap = float(aggregate_gap_pair["loss_reduction"])
    distribution_gap = float(distribution_gap_pair["loss_reduction"])

    simple_final_sign = _sign(simple_final_mvoi, tol=near_zero_tol)
    simple_lqg_sign = _sign(simple_lqg_mvoi, tol=near_zero_tol)
    lqg_sign = _sign(lqg_mvoi, tol=near_zero_tol)
    simple_any_positive = simple_final_sign > 0 or simple_lqg_sign > 0
    lqg_positive = lqg_sign > 0
    lqg_near_zero = bool(abs(lqg_mvoi) <= near_zero_tol or lqg_ci_includes_zero)
    simple_rule_sign_stable = bool(
        simple_final_sign == simple_lqg_sign
        and simple_final_sign != 0
        and not simple_final_ci_includes_zero
        and not simple_lqg_ci_includes_zero
    )

    if simple_any_positive and lqg_near_zero:
        interpretation = "simple_rule_positive_lqg_near_zero_rule_or_optimization_problem"
    elif lqg_positive and not simple_rule_sign_stable:
        interpretation = "lqg_positive_simple_rule_unstable_rule_class_problem_information_value_survives"
    elif lqg_positive and simple_rule_sign_stable and simple_final_sign > 0:
        interpretation = "simple_and_lqg_positive_strongest_case"
    elif lqg_positive:
        interpretation = "lqg_positive_information_value_survives"
    else:
        interpretation = "no_positive_lqg_distribution_value"

    return {
        "lqg_oracle_available": True,
        "simple_rule_mvoi_final_protocol": simple_final_mvoi,
        "simple_rule_mvoi_final_protocol_ci_includes_zero": simple_final_ci_includes_zero,
        "simple_rule_mvoi_in_lqg_system": simple_lqg_mvoi,
        "simple_rule_mvoi_in_lqg_system_ci_includes_zero": simple_lqg_ci_includes_zero,
        "lqg_distribution_observation_mvoi": lqg_mvoi,
        "lqg_distribution_observation_ci_includes_zero": lqg_ci_includes_zero,
        "lqg_distribution_observation_value_positive": lqg_positive,
        "lqg_distribution_observation_value_precise": bool(lqg_positive and not lqg_ci_includes_zero),
        "simple_rule_sign_stable_across_final_and_lqg_system": simple_rule_sign_stable,
        "simple_aggregate_gap_to_lqg": aggregate_gap,
        "simple_distribution_gap_to_lqg": distribution_gap,
        "simple_linear_rule_far_from_lqg": bool(max(aggregate_gap, distribution_gap) > near_zero_tol),
        "benchmark_interpretation": interpretation,
        "near_zero_tolerance": float(near_zero_tol),
    }


def _pairwise_row(pairwise: pd.DataFrame, comparison: str) -> pd.Series:
    rows = pairwise[pairwise["comparison"].eq(comparison)]
    if rows.empty:
        raise ValueError(f"LQG oracle pairwise table is missing comparison required by final protocol: {comparison}")
    return rows.iloc[0]


def _delta_ci_includes_zero(row: pd.Series | dict[str, object]) -> bool:
    return bool(float(row["cluster_ci_low"]) <= 0.0 <= float(row["cluster_ci_high"]))


def _final_gate(
    *,
    open_row: dict[str, object],
    closed_row: dict[str, object],
    closed_loop_gate_json: Path,
    allow_missing_closed_loop_gate: bool,
    allow_missing_lqg_oracle: bool,
    lqg_diagnostics: dict[str, object],
) -> dict[str, object]:
    open_sign = _sign(float(open_row["loss_reduction"]))
    closed_sign = _sign(float(closed_row["loss_reduction"]))
    open_ci_includes_zero = bool(float(open_row["cluster_ci_low"]) <= 0.0 <= float(open_row["cluster_ci_high"]))
    closed_ci_includes_zero = bool(float(closed_row["cluster_ci_low"]) <= 0.0 <= float(closed_row["cluster_ci_high"]))
    direction_survives = open_sign == closed_sign and closed_sign > 0
    if closed_sign <= 0 or open_sign != closed_sign:
        interpretation = "closed_loop_sign_flips_diagnose_environment"
    elif closed_ci_includes_zero:
        interpretation = "effect direction survives, precision weaker"
    else:
        interpretation = "closed_loop_direction_and_precision_survive"

    checks: list[dict[str, object]] = [
        {
            "name": "closed_loop_artifacts_required",
            "passed": True,
            "value": str(closed_loop_gate_json.parent),
            "threshold": "closed-loop trajectory losses and diagnostics exist",
        },
        {
            "name": "closed_loop_direction_matches_open_loop",
            "passed": bool(direction_survives),
            "value": {
                "open_loop_loss_reduction": float(open_row["loss_reduction"]),
                "closed_loop_loss_reduction": float(closed_row["loss_reduction"]),
            },
            "threshold": "same positive MVOI sign",
        },
    ]
    if closed_loop_gate_json.exists():
        closed_gate = json.loads(closed_loop_gate_json.read_text(encoding="utf-8"))
        checks.append(
            {
                "name": "closed_loop_internal_gate_passed",
                "passed": bool(closed_gate.get("passed", False)),
                "value": closed_gate.get("passed", False),
                "threshold": "true",
            }
        )
    else:
        checks.append(
            {
                "name": "closed_loop_internal_gate_passed",
                "passed": bool(allow_missing_closed_loop_gate),
                "value": f"missing: {closed_loop_gate_json}",
                "threshold": "true",
            }
        )
    checks.append(
        {
            "name": "large_sample_primary_precision_status_recorded",
            "passed": True,
            "value": "primary estimate imprecise" if open_ci_includes_zero else "primary estimate precise",
            "threshold": "record whether large-sample hierarchical CI includes zero",
        }
    )
    checks.append(
        {
            "name": "closed_loop_precision_status_recorded",
            "passed": True,
            "value": interpretation,
            "threshold": "must report whether closed-loop CI includes zero",
        }
    )
    lqg_available = bool(lqg_diagnostics.get("lqg_oracle_available", False))
    lqg_interpretation = str(lqg_diagnostics.get("benchmark_interpretation", "lqg_oracle_missing"))
    checks.extend(
        [
            {
                "name": "lqg_riccati_oracle_required",
                "passed": bool(lqg_available or allow_missing_lqg_oracle),
                "value": lqg_interpretation,
                "threshold": "lqg_oracle_summary.csv and lqg_oracle_pairwise.csv exist and contain required rows",
            },
            {
                "name": "lqg_distribution_value_question_recorded",
                "passed": bool(lqg_available or allow_missing_lqg_oracle),
                "value": {
                    "lqg_distribution_observation_mvoi": lqg_diagnostics.get("lqg_distribution_observation_mvoi"),
                    "lqg_distribution_observation_value_precise": lqg_diagnostics.get(
                        "lqg_distribution_observation_value_precise"
                    ),
                },
                "threshold": "report whether distributional observations have value inside LQG",
            },
            {
                "name": "simple_rule_distance_to_lqg_recorded",
                "passed": bool(lqg_available or allow_missing_lqg_oracle),
                "value": {
                    "simple_aggregate_gap_to_lqg": lqg_diagnostics.get("simple_aggregate_gap_to_lqg"),
                    "simple_distribution_gap_to_lqg": lqg_diagnostics.get("simple_distribution_gap_to_lqg"),
                },
                "threshold": "report distance of simple linear rules to LQG",
            },
            {
                "name": "lqg_anchor_consistent_with_simple_rule_claim",
                "passed": bool(
                    allow_missing_lqg_oracle
                    or lqg_interpretation != "simple_rule_positive_lqg_near_zero_rule_or_optimization_problem"
                ),
                "value": lqg_interpretation,
                "threshold": "simple-rule MVOI must not be positive while LQG MVOI is near zero",
            },
        ]
    )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "interpretation": interpretation,
        "lqg_benchmark_interpretation": lqg_interpretation,
        "lqg_oracle_diagnostics": lqg_diagnostics,
        "large_sample_primary_ci_includes_zero": open_ci_includes_zero,
        "closed_loop_cluster_ci_includes_zero": closed_ci_includes_zero,
        "checks": checks,
    }


def _sign(value: float, *, tol: float = 1e-12) -> int:
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


def _shock_seed_from_scenario(scenario: str) -> int:
    try:
        return int(str(scenario).split("_")[-1])
    except ValueError:
        return abs(hash(str(scenario))) % (2**31)


def _format_value(value: object) -> str:
    if value is None:
        return "--"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "--"
    if abs(numeric) >= 1e-3 and abs(numeric) < 1e3:
        return f"{numeric:.4f}"
    return f"{numeric:.3g}"


def _write_latex_side_by_side(side_by_side: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & A. Open-loop / fixed path & B. Closed-loop local projection \\\\",
        "\\midrule",
    ]
    for _, row in side_by_side.iterrows():
        lines.append(
            f"{row['metric_label']} & {_format_value(row['A_open_loop_fixed_path'])} "
            f"& {_format_value(row['B_closed_loop_local_projection'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_lqg_benchmark(lqg_benchmark: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Controller & Mean loss & 95\\% CI \\\\",
        "\\midrule",
    ]
    if lqg_benchmark.empty:
        lines.append("LQG/Riccati oracle missing & -- & -- \\\\")
    else:
        for _, row in lqg_benchmark.iterrows():
            ci = f"[{_format_value(row['ci_low'])}, {_format_value(row['ci_high'])}]"
            lines.append(f"{row['controller_label']} & {_format_value(row['mean_loss'])} & {ci} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    *,
    comparison: pd.DataFrame,
    side_by_side: pd.DataFrame,
    lqg_benchmark: pd.DataFrame,
    lqg_diagnostics: dict[str, object],
    gate: dict[str, object],
    output_path: Path,
) -> None:
    open_row = comparison[comparison["evaluation"].eq("open_loop_fixed_path")].iloc[0]
    closed_row = comparison[comparison["evaluation"].eq("closed_loop_local_projection")].iloc[0]
    lines = [
        "# Final VOI Protocol",
        "",
        "Open-loop / fixed-path evaluation is reported as the primary estimate.",
        "Closed-loop local projection is the main credibility check: the frozen rule feeds back through",
        "local HANK/SSJ dynamics, noisy observations, filtered information states, and the next rate path.",
        "",
        f"Final status: {'PASS' if gate['passed'] else 'FAIL'}.",
        f"Interpretation: {gate['interpretation']}.",
        "",
        "## A/B Table",
        "",
        side_by_side.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## LQG / Riccati Oracle",
        "",
        "The final protocol treats the LQG/Riccati oracle as a mandatory methodological anchor for the same linear state-space information problem.",
        "",
        lqg_benchmark.to_markdown(index=False, floatfmt=".4g") if not lqg_benchmark.empty else "LQG oracle artifacts are missing.",
        "",
        "## LQG Diagnostics",
        "",
        (
            "- Value of distributional observations inside LQG: "
            f"{_format_value(lqg_diagnostics.get('lqg_distribution_observation_mvoi'))}; "
            f"precise={lqg_diagnostics.get('lqg_distribution_observation_value_precise')}."
        ),
        (
            "- Simple rule MVOI in the final protocol: "
            f"{_format_value(lqg_diagnostics.get('simple_rule_mvoi_final_protocol'))}; "
            "simple rule MVOI inside the LQG linear system: "
            f"{_format_value(lqg_diagnostics.get('simple_rule_mvoi_in_lqg_system'))}."
        ),
        (
            "- Distance to LQG: aggregate rule gap "
            f"{_format_value(lqg_diagnostics.get('simple_aggregate_gap_to_lqg'))}; "
            "distribution rule gap "
            f"{_format_value(lqg_diagnostics.get('simple_distribution_gap_to_lqg'))}."
        ),
        f"- Benchmark interpretation: {lqg_diagnostics.get('benchmark_interpretation')}.",
        "",
        "## Main Pair",
        "",
        (
            f"- Open-loop loss reduction: {open_row['loss_reduction']:.6g}; "
            f"cluster CI for mean delta [{open_row['cluster_ci_low']:.6g}, {open_row['cluster_ci_high']:.6g}]."
        ),
        (
            f"- Closed-loop loss reduction: {closed_row['loss_reduction']:.6g}; "
            f"cluster CI for mean delta [{closed_row['cluster_ci_low']:.6g}, {closed_row['cluster_ci_high']:.6g}]."
        ),
        "",
    ]
    if gate["interpretation"] == "effect direction survives, precision weaker":
        lines.append("Closed-loop direction survives, but the confidence interval includes zero; precision is weaker.")
        lines.append("")
    if gate.get("large_sample_primary_ci_includes_zero"):
        lines.append("Large-sample primary estimate has a confidence interval that includes zero; report it as imprecise.")
        lines.append("")
    if gate["interpretation"] == "closed_loop_sign_flips_diagnose_environment":
        lines.append(
            "Closed-loop sign changes relative to open-loop. The final protocol does not rescue this with other robustness checks; diagnose the closed-loop environment."
        )
        lines.append("")
    lines.extend(["## Gate Checks", ""])
    for check in gate["checks"]:
        lines.append(f"- {check['name']}: {'PASS' if check['passed'] else 'FAIL'}; value={check['value']}.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
