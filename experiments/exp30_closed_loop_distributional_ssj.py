from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp22_closed_loop_evaluation import (  # noqa: E402
    PAIRWISE_COMPARISONS,
    STATE_LABEL_RU,
    _load_fitted_rules,
    _parse_seed_range,
    _write_latex,
)
from hank_ssj import (  # noqa: E402
    ClosedLoopSSJEnvironment,
    PolicyLossWeights,
    augment_jacobians_with_distributional_policy_responses,
    has_direct_distributional_jacobians,
    required_distributional_jacobian_keys,
)
from hank_ssj.closed_loop_environment import diagnostics_to_row  # noqa: E402
from policy.inference import bh_adjust_pvalues, summarize_paired_inference  # noqa: E402
from policy.optimize_rules import bootstrap_interval  # noqa: E402


@dataclass(frozen=True)
class ClosedLoopDistributionalSSJSpec:
    hank_observables: str
    hank_observations: str
    base_jacobians: str
    augmented_jacobians: str
    state_space_spec: str
    fitted_policy_rules: str
    output_dir: str
    modes: tuple[str, ...]
    test_seeds: tuple[int, ...]
    max_iterations: int
    min_iterations: int
    tolerance: float
    damping: float
    force_rebuild_distributional_jacobians: bool
    require_direct_distributional_jacobians: bool
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop local SSJ evaluation with direct distributional policy-response matrices."
    )
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--hank-observations", default="outputs/ssj/stochastic/hank_observations.csv")
    parser.add_argument("--base-jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument(
        "--augmented-jacobians",
        default="outputs/ssj/stochastic/closed_loop_distributional_ssj/jacobians_distributional_augmented.npz",
    )
    parser.add_argument("--state-space-spec", default="outputs/ssj/stochastic/state_space/state_space_spec.json")
    parser.add_argument("--fitted-policy-rules", default="outputs/ssj/stochastic/main_voi_joint_filter/fitted_policy_rules.csv")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/closed_loop_distributional_ssj")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--modes", default="closed_loop_local_projection")
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--min-iterations", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--force-rebuild-distributional-jacobians", action="store_true")
    parser.add_argument("--allow-fallback-distributional-jacobians", action="store_true")
    parser.add_argument("--distributional-jacobian-horizon", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_jacobians = Path(args.augmented_jacobians)
    _ensure_distributional_jacobians(
        base_jacobians=Path(args.base_jacobians),
        augmented_jacobians=augmented_jacobians,
        output_dir=output_dir,
        force=bool(args.force_rebuild_distributional_jacobians),
        horizon=args.distributional_jacobian_horizon,
    )

    require_direct = not bool(args.allow_fallback_distributional_jacobians)
    if require_direct and not has_direct_distributional_jacobians(augmented_jacobians):
        missing = ", ".join(required_distributional_jacobian_keys())
        raise RuntimeError(f"Missing direct distributional Jacobians: {missing}")

    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    test_seeds = _parse_seed_range(args.test_seeds)
    policies = _load_fitted_rules(Path(args.fitted_policy_rules))
    environment = ClosedLoopSSJEnvironment.from_files(
        hank_observables_csv=Path(args.hank_observables),
        hank_observations_csv=Path(args.hank_observations),
        jacobians_npz=augmented_jacobians,
        state_space_spec_json=Path(args.state_space_spec),
        loss_weights=PolicyLossWeights(),
    )
    if require_direct and environment.missing_direct_jacobians:
        raise RuntimeError(
            "Closed-loop environment still uses fallback effects for: "
            + ", ".join(environment.missing_direct_jacobians)
        )

    loss_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for mode in modes:
        for scenario in environment.scenarios:
            for seed in test_seeds:
                for state, policy in policies.items():
                    result = environment.simulate_scenario(
                        policy=policy,
                        information_state=state,
                        scenario=scenario,
                        seed=seed,
                        mode=mode,
                        max_iterations=args.max_iterations,
                        min_iterations=args.min_iterations,
                        tolerance=args.tolerance,
                        damping=args.damping,
                    )
                    loss_rows.append(
                        {
                            "mode": mode,
                            "scenario": scenario,
                            "shock_seed": _shock_seed_from_scenario(scenario),
                            "observation_seed": int(seed),
                            "information_state": state,
                            "information_state_ru": STATE_LABEL_RU[state],
                            **asdict(result.loss),
                        }
                    )
                    diagnostic_rows.append(diagnostics_to_row(result.diagnostics))

    losses = pd.DataFrame(loss_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = _summary_table(losses)
    pairwise = _pairwise_table(losses)
    convergence = _convergence_summary(diagnostics)
    jacobian_diagnostics = _jacobian_diagnostics(augmented_jacobians)

    losses.to_csv(output_dir / "trajectory_losses_closed_loop.csv", index=False)
    diagnostics.to_csv(output_dir / "convergence_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "main_voi_closed_loop_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_closed_loop_value_of_information.csv", index=False)
    convergence.to_csv(output_dir / "convergence_summary.csv", index=False)
    jacobian_diagnostics.to_csv(output_dir / "direct_distributional_jacobian_diagnostics.csv", index=False)
    _write_latex(summary, output_dir / "table_main_voi_closed_loop_summary.tex")
    _write_latex(pairwise, output_dir / "table_pairwise_closed_loop_value_of_information.tex")
    _write_latex(convergence, output_dir / "table_convergence_summary.tex")
    _write_latex(jacobian_diagnostics, output_dir / "table_direct_distributional_jacobian_diagnostics.tex")

    spec = ClosedLoopDistributionalSSJSpec(
        hank_observables=args.hank_observables,
        hank_observations=args.hank_observations,
        base_jacobians=args.base_jacobians,
        augmented_jacobians=str(augmented_jacobians),
        state_space_spec=args.state_space_spec,
        fitted_policy_rules=args.fitted_policy_rules,
        output_dir=args.output_dir,
        modes=modes,
        test_seeds=tuple(test_seeds),
        max_iterations=int(args.max_iterations),
        min_iterations=int(args.min_iterations),
        tolerance=float(args.tolerance),
        damping=float(args.damping),
        force_rebuild_distributional_jacobians=bool(args.force_rebuild_distributional_jacobians),
        require_direct_distributional_jacobians=require_direct,
        note=(
            "Контрфактическая ставка меняет агрегаты и распределительные статистики через локальные "
            "HANK/SSJ-матрицы. После этого заново строятся наблюдения, фильтр и ставка правила."
        ),
    )
    (output_dir / "closed_loop_distributional_ssj_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(summary, pairwise, convergence, jacobian_diagnostics, output_dir / "report_closed_loop.md")
    print(f"Wrote {output_dir / 'main_voi_closed_loop_summary.csv'}")
    print(f"Wrote {output_dir / 'convergence_diagnostics.csv'}")
    print(f"Wrote {output_dir / 'report_closed_loop.md'}")


def _ensure_distributional_jacobians(
    *,
    base_jacobians: Path,
    augmented_jacobians: Path,
    output_dir: Path,
    force: bool,
    horizon: int | None,
) -> None:
    if augmented_jacobians.exists() and has_direct_distributional_jacobians(augmented_jacobians) and not force:
        return
    augment_jacobians_with_distributional_policy_responses(
        base_jacobians_npz=base_jacobians,
        output_npz=augmented_jacobians,
        output_long_csv=output_dir / "distributional_policy_jacobians_long.csv",
        output_spec_json=output_dir / "distributional_policy_jacobians_spec.json",
        horizon=horizon,
        suppress_solver_output=True,
    )


def _summary_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (mode, scenario), frame in _mode_scenario_groups(losses):
        for state, state_frame in frame.groupby("information_state", sort=False):
            values = state_frame["total_loss"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_interval(values)
            rows.append(
                {
                    "mode": mode,
                    "scenario": scenario,
                    "information_state": state,
                    "information_state_ru": STATE_LABEL_RU[state],
                    "num_trajectories": int(values.size),
                    "mean_loss": float(np.mean(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "inflation_loss": float(state_frame["inflation_loss"].mean()),
                    "output_gap_loss": float(state_frame["output_gap_loss"].mean()),
                    "consumption_loss": float(state_frame["consumption_loss"].mean()),
                    "rate_smoothing_loss": float(state_frame["rate_smoothing_loss"].mean()),
                    "stability_penalty": float(state_frame["stability_penalty"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (mode, scenario), frame in _mode_scenario_groups(losses):
        pivot = frame.pivot_table(
            index=["mode", "scenario", "shock_seed", "observation_seed"],
            columns="information_state",
            values="total_loss",
            aggfunc="first",
        ).reset_index()
        cluster_id = pivot["shock_seed"].to_numpy(dtype=int)
        for left, right, label in PAIRWISE_COMPARISONS:
            if left not in pivot.columns or right not in pivot.columns:
                continue
            delta = pivot[left].to_numpy(dtype=float) - pivot[right].to_numpy(dtype=float)
            inference = summarize_paired_inference(delta, cluster_id=cluster_id, n_boot=4_000, n_perm=4_000, tie_eps=1e-10)
            rows.append(
                {
                    "mode": mode,
                    "scenario": scenario,
                    "comparison": f"{left}_minus_{right}",
                    "comparison_ru": label,
                    "left": left,
                    "right": right,
                    "num_trajectories": inference.num_observations,
                    "num_shock_clusters": inference.num_clusters,
                    "mean_delta": inference.mean_delta,
                    "median_delta": inference.median_delta,
                    "loss_reduction": -inference.mean_delta,
                    "bootstrap_ci_low": inference.bootstrap_ci_low,
                    "bootstrap_ci_high": inference.bootstrap_ci_high,
                    "cluster_ci_low": inference.clustered_ci_low,
                    "cluster_ci_high": inference.clustered_ci_high,
                    "wild_ci_low": inference.wild_ci_low,
                    "wild_ci_high": inference.wild_ci_high,
                    "permutation_p_value": inference.permutation_p_value,
                    "sign_flip_p_value": inference.sign_flip_p_value,
                    "win_rate": inference.win_rate,
                    "tie_rate": inference.tie_rate,
                    "loss_rate": inference.loss_rate,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_permutation_p_value"] = np.nan
        result["bh_sign_flip_p_value"] = np.nan
        for _, index in result.groupby(["mode", "scenario"]).groups.items():
            result.loc[index, "bh_permutation_p_value"] = bh_adjust_pvalues(result.loc[index, "permutation_p_value"])
            result.loc[index, "bh_sign_flip_p_value"] = bh_adjust_pvalues(result.loc[index, "sign_flip_p_value"])
    return result


def _convergence_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (mode, state), frame in diagnostics.groupby(["mode", "information_state"], sort=False):
        rows.append(
            {
                "mode": mode,
                "information_state": state,
                "information_state_ru": STATE_LABEL_RU[state],
                "num_trajectories": int(frame.shape[0]),
                "convergence_failure_rate": float(1.0 - frame["converged"].mean()),
                "mean_iterations": float(frame["iterations"].mean()),
                "mean_rate_update_norm": float(frame["rate_update_norm"].mean()),
                "mean_state_update_norm": float(frame["state_update_norm"].mean()),
                "mean_distribution_update_norm": float(frame["distribution_update_norm"].mean()),
                "max_spectral_radius_local_loop": float(frame["spectral_radius_local_loop"].max()),
                "mean_stability_penalty": float(frame["stability_penalty"].mean()),
                "mean_convergence_penalty": float(frame["convergence_penalty"].mean()),
                "fallback_effects": ",".join(sorted(set(",".join(frame["fallback_effects"]).split(",")) - {""})),
            }
        )
    return pd.DataFrame(rows)


def _jacobian_diagnostics(jacobians_npz: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with np.load(jacobians_npz, allow_pickle=True) as bundle:
        for key in required_distributional_jacobian_keys():
            if key in bundle.files:
                matrix = np.asarray(bundle[key], dtype=float)
                rows.append(
                    {
                        "jacobian_key": key,
                        "available": True,
                        "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                        "frobenius_norm": float(np.linalg.norm(matrix)),
                        "max_abs": float(np.max(np.abs(matrix))),
                    }
                )
            else:
                rows.append(
                    {
                        "jacobian_key": key,
                        "available": False,
                        "shape": "",
                        "frobenius_norm": np.nan,
                        "max_abs": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _mode_scenario_groups(losses: pd.DataFrame):
    for key, frame in losses.groupby(["mode", "scenario"], sort=False):
        yield key, frame
    for mode, frame in losses.groupby("mode", sort=False):
        yield (mode, "all"), frame


def _shock_seed_from_scenario(scenario: str) -> int:
    try:
        return int(str(scenario).split("_")[-1])
    except ValueError:
        return abs(hash(str(scenario))) % (2**31)


def _write_report(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    convergence: pd.DataFrame,
    jacobian_diagnostics: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Closed-loop с прямыми распределительными SSJ-откликами",
        "",
        "Правила заморожены после основного прогона с совместным фильтром.",
        "Контрфактическая ставка меняет агрегаты и распределительные статистики через локальные HANK/SSJ-матрицы;",
        "после этого заново строятся наблюдения, фильтрованные состояния и ставка правила.",
        "",
        "## Прямые распределительные отклики",
        "",
    ]
    for _, row in jacobian_diagnostics.iterrows():
        status = "есть" if row["available"] else "нет"
        lines.append(f"- {row['jacobian_key']}: {status}, max abs {row['max_abs']:.6g}.")

    for mode in summary["mode"].drop_duplicates():
        block = summary[(summary["mode"] == mode) & (summary["scenario"] == "all")].sort_values("mean_loss")
        lines.extend(["", f"## {mode}", ""])
        for _, row in block.head(6).iterrows():
            lines.append(f"- {row['information_state_ru']}: {row['mean_loss']:.6g}")
        main_pair = pairwise[
            (pairwise["mode"] == mode)
            & (pairwise["scenario"] == "all")
            & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
        ]
        if not main_pair.empty:
            row = main_pair.iloc[0]
            lines.append(
                f"- Все распределительные статистики против агрегатов: снижение потерь {-row['mean_delta']:.6g}, "
                f"кластерный интервал для разности [{row['cluster_ci_low']:.6g}, {row['cluster_ci_high']:.6g}], "
                f"sign-flip p-value {row['sign_flip_p_value']:.3g}."
            )

    lines.extend(["", "## Сходимость", ""])
    for _, row in convergence.iterrows():
        if row["information_state"] == "filtered_distribution":
            lines.append(
                f"- {row['mode']}, {row['information_state_ru']}: доля несходимости "
                f"{row['convergence_failure_rate']:.3g}, средняя норма обновления распределительных статистик "
                f"{row['mean_distribution_update_norm']:.3g}."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
