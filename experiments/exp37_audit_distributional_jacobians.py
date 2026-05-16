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

from experiments.exp22_closed_loop_evaluation import (  # noqa: E402
    STATE_LABEL_RU,
    _load_fitted_rules,
    _parse_seed_range,
)
from experiments.exp30_closed_loop_distributional_ssj import (  # noqa: E402
    _convergence_summary,
    _pairwise_table,
    _shock_seed_from_scenario,
    _summary_table,
)
from hank_ssj import (  # noqa: E402
    ClosedLoopSSJEnvironment,
    PolicyLossWeights,
    augment_jacobians_with_distributional_policy_responses,
)
from hank_ssj.closed_loop_environment import diagnostics_to_row  # noqa: E402
from hank_ssj.distributional_jacobians import (  # noqa: E402
    DISTRIBUTIONAL_JACOBIAN_OUTPUTS,
    transition_diagnostics_to_frame,
)


@dataclass(frozen=True)
class DistributionalJacobianAuditSpec:
    base_jacobians: str
    output_dir: str
    shock_name: str
    horizon: int
    eps_grid: tuple[float, ...]
    difference_method: str
    run_closed_loop_eps_grid: bool
    mvoi_comparison: str
    mvoi_scenario: str
    mass_convergence_failure_threshold: float
    magnitude_order_tolerance: float
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit direct distributional HANK/SSJ policy Jacobians.")
    parser.add_argument("--base-jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/distributional_jacobian_audit")
    parser.add_argument("--shock-name", default="monetary_policy_shock")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--eps-grid", default="0.00025,0.0005,0.001,0.002")
    parser.add_argument("--difference-method", choices=("central", "forward"), default="central")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-closed-loop-eps-grid", action="store_true")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--hank-observations", default="outputs/ssj/stochastic/hank_observations.csv")
    parser.add_argument("--state-space-spec", default="outputs/ssj/stochastic/state_space/state_space_spec.json")
    parser.add_argument("--fitted-policy-rules", default="outputs/ssj/stochastic/main_voi_joint_filter/fitted_policy_rules.csv")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--modes", default="closed_loop_local_projection")
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--min-iterations", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--mvoi-comparison", default="filtered_distribution_minus_filtered_aggregates")
    parser.add_argument("--mvoi-scenario", default="all")
    parser.add_argument("--mvoi-zero-tol", type=float, default=1e-12)
    parser.add_argument("--mass-convergence-failure-threshold", type=float, default=0.05)
    parser.add_argument("--magnitude-order-tolerance", type=float, default=1.0)
    parser.add_argument("--allow-failed-final-gate", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eps_grid = tuple(float(value) for value in _parse_grid(args.eps_grid))
    horizon = _resolve_horizon(Path(args.base_jacobians), args.horizon)

    transition_rows: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    signed_rows: list[dict[str, object]] = []
    response_rows: list[pd.DataFrame] = []
    matrices_by_eps: dict[float, dict[str, np.ndarray]] = {}
    jacobians_by_eps: dict[float, Path] = {}

    for eps in eps_grid:
        eps_dir = output_dir / f"eps_{_eps_label(eps)}"
        eps_dir.mkdir(parents=True, exist_ok=True)
        augmented_jacobians = eps_dir / "jacobians_distributional_augmented.npz"
        spec_json = eps_dir / "distributional_policy_jacobians_spec.json"
        diagnostics_csv = eps_dir / "transition_solver_diagnostics.csv"
        signed_csv = eps_dir / "signed_epsilon_comparison.csv"
        if (
            args.force_rebuild
            or not augmented_jacobians.exists()
            or not spec_json.exists()
            or not diagnostics_csv.exists()
            or not signed_csv.exists()
        ):
            result = augment_jacobians_with_distributional_policy_responses(
                base_jacobians_npz=Path(args.base_jacobians),
                output_npz=augmented_jacobians,
                output_long_csv=eps_dir / "distributional_policy_jacobians_long.csv",
                output_spec_json=spec_json,
                output_diagnostics_csv=diagnostics_csv,
                horizon=horizon,
                shock_name=args.shock_name,
                shock_size=eps,
                difference_method=args.difference_method,
                strict_mode=False,
                suppress_solver_output=True,
            )
        else:
            result = None
        jacobians_by_eps[eps] = augmented_jacobians
        spec = _load_json(spec_json)
        fallback_periods = tuple(int(period) for period in spec.get("failed_shifted_transition_periods", []))
        fallback_rows.append(
            {
                "eps": eps,
                "fallback_period_count": len(fallback_periods),
                "failed_shifted_transition_periods": ",".join(str(period) for period in fallback_periods),
                "difference_method": spec.get("difference_method", args.difference_method),
            }
        )
        diagnostics = _transition_diagnostics_frame(result, diagnostics_csv)
        if not diagnostics.empty:
            diagnostics.insert(0, "eps", eps)
            transition_rows.append(diagnostics)
        matrices = _load_distributional_matrices(augmented_jacobians, shock_name=args.shock_name, horizon=horizon)
        matrices_by_eps[eps] = matrices
        matrix_rows.extend(_matrix_diagnostics(matrices, eps=eps, shock_name=args.shock_name))
        if result is not None:
            signed_rows.extend(_signed_epsilon_comparison(result.signed_derivative_matrices, result.matrices, eps=eps))
        elif signed_csv.exists():
            signed_rows.extend(pd.read_csv(signed_csv).to_dict("records"))
        response_rows.append(_response_long(matrices, eps=eps))
        if result is not None:
            signed_frame = pd.DataFrame(_signed_epsilon_comparison(result.signed_derivative_matrices, result.matrices, eps=eps))
            signed_frame.to_csv(signed_csv, index=False)

    transition = pd.concat(transition_rows, ignore_index=True) if transition_rows else pd.DataFrame()
    period_convergence = _period_convergence_summary(transition)
    fallback = pd.DataFrame(fallback_rows)
    matrix_diagnostics = pd.DataFrame(matrix_rows)
    signed_comparison = pd.DataFrame(signed_rows)
    responses = pd.concat(response_rows, ignore_index=True) if response_rows else pd.DataFrame()
    linearity = _local_linearity(matrices_by_eps)

    transition.to_csv(output_dir / "transition_solver_diagnostics.csv", index=False)
    period_convergence.to_csv(output_dir / "transition_convergence_by_shock_period.csv", index=False)
    fallback.to_csv(output_dir / "fallback_summary_by_eps.csv", index=False)
    matrix_diagnostics.to_csv(output_dir / "jacobian_block_diagnostics.csv", index=False)
    signed_comparison.to_csv(output_dir / "signed_epsilon_comparison.csv", index=False)
    responses.to_csv(output_dir / "distributional_jacobian_responses_long.csv", index=False)
    linearity.to_csv(output_dir / "local_linearity_by_eps.csv", index=False)
    _plot_response_by_shock_period(responses, output_dir / "fig_response_by_shock_period.pdf")
    _plot_response_by_shock_period(responses, output_dir / "fig_response_by_shock_period.png")

    run_closed_loop = not bool(args.skip_closed_loop_eps_grid)
    mvoi_grid = pd.DataFrame()
    closed_loop_convergence = pd.DataFrame()
    if run_closed_loop:
        mvoi_grid, closed_loop_convergence = _run_closed_loop_eps_grid(
            jacobians_by_eps=jacobians_by_eps,
            args=args,
            output_dir=output_dir,
        )
        mvoi_grid.to_csv(output_dir / "mvoi_eps_grid.csv", index=False)
        closed_loop_convergence.to_csv(output_dir / "closed_loop_eps_grid_convergence_summary.csv", index=False)

    gate = _final_protocol_gate(
        fallback=fallback,
        mvoi_grid=mvoi_grid,
        closed_loop_convergence=closed_loop_convergence,
        mvoi_comparison=args.mvoi_comparison,
        mvoi_scenario=args.mvoi_scenario,
        mvoi_zero_tol=float(args.mvoi_zero_tol),
        mass_convergence_failure_threshold=float(args.mass_convergence_failure_threshold),
        magnitude_order_tolerance=float(args.magnitude_order_tolerance),
        run_closed_loop=run_closed_loop,
    )
    (output_dir / "final_protocol_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    spec = DistributionalJacobianAuditSpec(
        base_jacobians=args.base_jacobians,
        output_dir=str(output_dir),
        shock_name=args.shock_name,
        horizon=horizon,
        eps_grid=eps_grid,
        difference_method=args.difference_method,
        run_closed_loop_eps_grid=run_closed_loop,
        mvoi_comparison=args.mvoi_comparison,
        mvoi_scenario=args.mvoi_scenario,
        mass_convergence_failure_threshold=float(args.mass_convergence_failure_threshold),
        magnitude_order_tolerance=float(args.magnitude_order_tolerance),
        note=(
            "Аудит проверяет локальную производную распределительных статистик по eps-grid. "
            "Финальный gate проходит только без Toeplitz fallback, со стабильным знаком MVOI, "
            "устойчивым порядком величины MVOI и без массовой несходимости closed-loop."
        ),
    )
    (output_dir / "distributional_jacobian_audit_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        fallback=fallback,
        period_convergence=period_convergence,
        matrix_diagnostics=matrix_diagnostics,
        signed_comparison=signed_comparison,
        linearity=linearity,
        mvoi_grid=mvoi_grid,
        gate=gate,
    )
    print(f"Wrote {output_dir / 'final_protocol_gate.json'}")
    print(f"Wrote {output_dir / 'report_distributional_jacobian_audit.md'}")
    if not gate["passed"] and not args.allow_failed_final_gate:
        failed = ", ".join(check["name"] for check in gate["checks"] if not check["passed"])
        raise RuntimeError(f"Distributional Jacobian final protocol gate failed: {failed}")


def _parse_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("eps-grid must contain at least one value.")
    if any(value <= 0.0 for value in values):
        raise ValueError("eps-grid values must be positive.")
    return values


def _eps_label(eps: float) -> str:
    return f"{eps:.8g}".replace("-", "m").replace(".", "p")


def _resolve_horizon(base_jacobians: Path, requested: int | None) -> int:
    with np.load(base_jacobians, allow_pickle=True) as bundle:
        shapes = [np.asarray(bundle[key]).shape for key in bundle.files if key.startswith("J_") and np.asarray(bundle[key]).ndim == 2]
    if not shapes:
        raise ValueError(f"Base Jacobian archive {base_jacobians} does not contain matrix keys.")
    inferred = int(min(shape[0] for shape in shapes))
    return inferred if requested is None else min(int(requested), inferred)


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _transition_diagnostics_frame(result, diagnostics_csv: Path) -> pd.DataFrame:
    if result is not None:
        return transition_diagnostics_to_frame(result)
    if diagnostics_csv.exists():
        return pd.read_csv(diagnostics_csv)
    return pd.DataFrame()


def _load_distributional_matrices(jacobians_npz: Path, *, shock_name: str, horizon: int) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    with np.load(jacobians_npz, allow_pickle=True) as bundle:
        for variable in DISTRIBUTIONAL_JACOBIAN_OUTPUTS:
            key = f"J_{shock_name}_{variable}"
            if key not in bundle.files:
                continue
            matrices[variable] = np.asarray(bundle[key], dtype=float)[:horizon, :horizon]
    return matrices


def _matrix_diagnostics(matrices: dict[str, np.ndarray], *, eps: float, shock_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variable, matrix in matrices.items():
        try:
            condition_number = float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError:
            condition_number = math.inf
        rows.append(
            {
                "eps": eps,
                "jacobian_key": f"J_{shock_name}_{variable}",
                "variable": variable,
                "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                "frobenius_norm": float(np.linalg.norm(matrix, ord="fro")),
                "max_abs": float(np.max(np.abs(matrix))),
                "condition_number": condition_number,
                "finite_share": float(np.isfinite(matrix).mean()),
            }
        )
    return rows


def _period_convergence_summary(transition: pd.DataFrame) -> pd.DataFrame:
    if transition.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (eps, shock_period), frame in transition.groupby(["eps", "shock_period"], sort=True):
        failed = frame[~frame["converged"].astype(bool)]
        rows.append(
            {
                "eps": float(eps),
                "shock_period": int(shock_period),
                "num_direction_solves": int(frame.shape[0]),
                "all_directions_converged": bool(failed.empty),
                "max_residual": float(frame["max_residual"].max()),
                "max_residual_any_iteration": float(frame["max_residual_any_iteration"].max()),
                "failed_directions": ",".join(failed["direction"].astype(str).tolist()),
                "error_messages": " | ".join(str(value) for value in failed["error_message"].dropna().unique() if str(value)),
            }
        )
    return pd.DataFrame(rows)


def _signed_epsilon_comparison(
    signed_matrices: dict[str, dict[str, np.ndarray]],
    central_matrices: dict[str, np.ndarray],
    *,
    eps: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    plus = signed_matrices.get("plus", {})
    minus = signed_matrices.get("minus", {})
    for variable, central in central_matrices.items():
        plus_matrix = plus.get(variable)
        minus_matrix = minus.get(variable)
        if plus_matrix is None or minus_matrix is None:
            continue
        mask = np.isfinite(plus_matrix) & np.isfinite(minus_matrix)
        if not np.any(mask):
            continue
        plus_values = plus_matrix[mask]
        minus_values = minus_matrix[mask]
        central_values = central[mask]
        gap = plus_values - minus_values
        plus_norm = float(np.linalg.norm(plus_values))
        minus_norm = float(np.linalg.norm(minus_values))
        central_norm = float(np.linalg.norm(central_values))
        denominator = max(central_norm, 1e-14)
        cosine_denominator = max(plus_norm * minus_norm, 1e-14)
        rows.append(
            {
                "eps": eps,
                "variable": variable,
                "num_finite_entries": int(mask.sum()),
                "plus_forward_norm": plus_norm,
                "minus_forward_norm": minus_norm,
                "central_norm": central_norm,
                "relative_plus_minus_gap": float(np.linalg.norm(gap) / denominator),
                "max_abs_plus_minus_gap": float(np.max(np.abs(gap))),
                "plus_minus_cosine": float(np.dot(plus_values, minus_values) / cosine_denominator),
            }
        )
    return rows


def _response_long(matrices: dict[str, np.ndarray], *, eps: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable, matrix in matrices.items():
        for response_period in range(matrix.shape[0]):
            for shock_period in range(matrix.shape[1]):
                rows.append(
                    {
                        "eps": eps,
                        "variable": variable,
                        "response_period": int(response_period),
                        "shock_period": int(shock_period),
                        "central_derivative": float(matrix[response_period, shock_period]),
                    }
                )
    return pd.DataFrame(rows)


def _local_linearity(matrices_by_eps: dict[float, dict[str, np.ndarray]]) -> pd.DataFrame:
    if not matrices_by_eps:
        return pd.DataFrame()
    eps_values = sorted(matrices_by_eps)
    reference_eps = 0.001 if 0.001 in matrices_by_eps else eps_values[len(eps_values) // 2]
    rows: list[dict[str, object]] = []
    for variable in DISTRIBUTIONAL_JACOBIAN_OUTPUTS:
        reference = matrices_by_eps[reference_eps].get(variable)
        if reference is None:
            continue
        reference_norm = float(np.linalg.norm(reference, ord="fro"))
        for eps in eps_values:
            matrix = matrices_by_eps[eps].get(variable)
            if matrix is None:
                continue
            norm = float(np.linalg.norm(matrix, ord="fro"))
            rows.append(
                {
                    "eps": eps,
                    "reference_eps": reference_eps,
                    "variable": variable,
                    "frobenius_norm": norm,
                    "reference_frobenius_norm": reference_norm,
                    "norm_ratio_to_reference": float(norm / max(reference_norm, 1e-14)),
                    "relative_matrix_gap_to_reference": float(np.linalg.norm(matrix - reference, ord="fro") / max(reference_norm, 1e-14)),
                    "max_abs_gap_to_reference": float(np.max(np.abs(matrix - reference))),
                }
            )
    return pd.DataFrame(rows)


def _run_closed_loop_eps_grid(*, jacobians_by_eps: dict[float, Path], args, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = _load_fitted_rules(Path(args.fitted_policy_rules))
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    test_seeds = _parse_seed_range(args.test_seeds)
    pairwise_rows: list[pd.DataFrame] = []
    convergence_rows: list[pd.DataFrame] = []
    for eps, jacobians_npz in sorted(jacobians_by_eps.items()):
        eps_dir = output_dir / f"eps_{_eps_label(eps)}"
        environment = ClosedLoopSSJEnvironment.from_files(
            hank_observables_csv=Path(args.hank_observables),
            hank_observations_csv=Path(args.hank_observations),
            jacobians_npz=jacobians_npz,
            state_space_spec_json=Path(args.state_space_spec),
            loss_weights=PolicyLossWeights(),
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
                                "eps": eps,
                                "mode": mode,
                                "scenario": scenario,
                                "shock_seed": _shock_seed_from_scenario(scenario),
                                "observation_seed": int(seed),
                                "information_state": state,
                                "information_state_ru": STATE_LABEL_RU[state],
                                **asdict(result.loss),
                            }
                        )
                        row = diagnostics_to_row(result.diagnostics)
                        row["eps"] = eps
                        diagnostic_rows.append(row)
        losses = pd.DataFrame(loss_rows)
        diagnostics = pd.DataFrame(diagnostic_rows)
        summary = _summary_table(losses.drop(columns=["eps"]))
        pairwise = _pairwise_table(losses.drop(columns=["eps"]))
        convergence = _convergence_summary(diagnostics.drop(columns=["eps"]))
        summary.insert(0, "eps", eps)
        pairwise.insert(0, "eps", eps)
        convergence.insert(0, "eps", eps)
        losses.to_csv(eps_dir / "trajectory_losses_closed_loop.csv", index=False)
        diagnostics.to_csv(eps_dir / "convergence_diagnostics.csv", index=False)
        summary.to_csv(eps_dir / "main_voi_closed_loop_summary.csv", index=False)
        pairwise.to_csv(eps_dir / "pairwise_closed_loop_value_of_information.csv", index=False)
        convergence.to_csv(eps_dir / "convergence_summary.csv", index=False)
        pairwise_rows.append(pairwise)
        convergence_rows.append(convergence)
    mvoi_grid = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()
    closed_loop_convergence = pd.concat(convergence_rows, ignore_index=True) if convergence_rows else pd.DataFrame()
    return mvoi_grid, closed_loop_convergence


def _final_protocol_gate(
    *,
    fallback: pd.DataFrame,
    mvoi_grid: pd.DataFrame,
    closed_loop_convergence: pd.DataFrame,
    mvoi_comparison: str,
    mvoi_scenario: str,
    mvoi_zero_tol: float,
    mass_convergence_failure_threshold: float,
    magnitude_order_tolerance: float,
    run_closed_loop: bool,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    fallback_count = int(fallback["fallback_period_count"].sum()) if not fallback.empty else None
    checks.append(
        {
            "name": "no_shifted_transition_fallback",
            "passed": bool(fallback_count == 0),
            "value": fallback_count,
            "threshold": 0,
        }
    )
    mvoi_frame = _main_mvoi_frame(mvoi_grid, comparison=mvoi_comparison, scenario=mvoi_scenario)
    sign_stable, magnitude_stable, mvoi_rows = _mvoi_stability(
        mvoi_frame,
        zero_tol=mvoi_zero_tol,
        magnitude_order_tolerance=magnitude_order_tolerance,
    )
    checks.append(
        {
            "name": "mvoi_sign_stable_eps_grid",
            "passed": bool(run_closed_loop and sign_stable),
            "value": mvoi_rows,
            "threshold": "same nonzero sign across eps-grid",
        }
    )
    checks.append(
        {
            "name": "mvoi_magnitude_order_stable_eps_grid",
            "passed": bool(run_closed_loop and magnitude_stable),
            "value": mvoi_rows,
            "threshold": f"log10 span <= {magnitude_order_tolerance}",
        }
    )
    max_failure_rate = None
    if run_closed_loop and not closed_loop_convergence.empty:
        max_failure_rate = float(closed_loop_convergence["convergence_failure_rate"].max())
    checks.append(
        {
            "name": "closed_loop_no_mass_nonconvergence",
            "passed": bool(max_failure_rate is not None and max_failure_rate <= mass_convergence_failure_threshold),
            "value": max_failure_rate,
            "threshold": mass_convergence_failure_threshold,
        }
    )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
    }


def _main_mvoi_frame(mvoi_grid: pd.DataFrame, *, comparison: str, scenario: str) -> pd.DataFrame:
    if mvoi_grid.empty:
        return pd.DataFrame()
    frame = mvoi_grid[
        mvoi_grid["comparison"].eq(comparison)
        & mvoi_grid["scenario"].eq(scenario)
    ].copy()
    if frame.empty and scenario == "all":
        frame = mvoi_grid[mvoi_grid["comparison"].eq(comparison)].copy()
    return frame.sort_values("eps")


def _mvoi_stability(
    frame: pd.DataFrame,
    *,
    zero_tol: float,
    magnitude_order_tolerance: float,
) -> tuple[bool, bool, list[dict[str, object]]]:
    if frame.empty:
        return False, False, []
    rows: list[dict[str, object]] = []
    signs: list[int] = []
    orders: list[float] = []
    for _, row in frame.iterrows():
        value = float(row["loss_reduction"])
        if abs(value) <= zero_tol:
            sign = 0
            order = None
        else:
            sign = 1 if value > 0.0 else -1
            order = math.log10(abs(value))
        signs.append(sign)
        if order is not None:
            orders.append(order)
        rows.append(
            {
                "eps": float(row["eps"]),
                "scenario": row["scenario"],
                "loss_reduction": value,
                "sign": sign,
                "log10_abs_loss_reduction": order,
            }
        )
    nonzero_signs = {sign for sign in signs if sign != 0}
    sign_stable = len(nonzero_signs) == 1 and all(sign != 0 for sign in signs)
    magnitude_stable = bool(orders) and (max(orders) - min(orders) <= magnitude_order_tolerance)
    return sign_stable, magnitude_stable, rows


def _plot_response_by_shock_period(responses: pd.DataFrame, figure_path: Path) -> None:
    if responses.empty:
        return
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    eps_values = sorted(responses["eps"].unique())
    reference_eps = 0.001 if 0.001 in eps_values else eps_values[len(eps_values) // 2]
    frame = responses[responses["eps"].eq(reference_eps)]
    variables = list(DISTRIBUTIONAL_JACOBIAN_OUTPUTS)
    fig, axes = plt.subplots(len(variables), 1, figsize=(8.5, 8.6), sharex=True)
    if len(variables) == 1:
        axes = [axes]
    max_period = int(frame["shock_period"].max())
    shock_periods = sorted(set(int(round(value)) for value in np.linspace(0, max_period, min(6, max_period + 1))))
    for axis, variable in zip(axes, variables):
        variable_frame = frame[frame["variable"].eq(variable)]
        for shock_period in shock_periods:
            line = variable_frame[variable_frame["shock_period"].eq(shock_period)].sort_values("response_period")
            axis.plot(
                line["response_period"],
                line["central_derivative"],
                linewidth=1.4,
                label=f"shock t={shock_period}",
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        axis.set_title(variable)
        axis.set_ylabel("derivative")
    axes[-1].set_xlabel("response period")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Distributional Jacobian responses by shock period, eps={reference_eps:g}", fontsize=12)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    *,
    output_dir: Path,
    fallback: pd.DataFrame,
    period_convergence: pd.DataFrame,
    matrix_diagnostics: pd.DataFrame,
    signed_comparison: pd.DataFrame,
    linearity: pd.DataFrame,
    mvoi_grid: pd.DataFrame,
    gate: dict[str, object],
) -> None:
    lines = [
        "# Аудит распределительных SSJ-якобианов",
        "",
        f"Final protocol gate: {'PASS' if gate['passed'] else 'FAIL'}.",
        "",
        "## Fallback",
        "",
        fallback.to_markdown(index=False) if not fallback.empty else "Нет строк fallback-аудита.",
        "",
        "## Сходимость по shock_period",
        "",
        period_convergence.head(20).to_markdown(index=False, floatfmt=".4g")
        if not period_convergence.empty
        else "Нет строк transition diagnostics.",
        "",
        "## Нормы и conditioning",
        "",
    ]
    if not matrix_diagnostics.empty:
        display = matrix_diagnostics[["eps", "variable", "frobenius_norm", "max_abs", "condition_number"]]
        lines.append(display.to_markdown(index=False, floatfmt=".4g"))
    if not signed_comparison.empty:
        lines.extend(
            [
                "",
                "## +eps против -eps",
                "",
                signed_comparison[
                    ["eps", "variable", "relative_plus_minus_gap", "max_abs_plus_minus_gap", "plus_minus_cosine"]
                ].to_markdown(index=False, floatfmt=".4g"),
            ]
        )
    if not linearity.empty:
        lines.extend(
            [
                "",
                "## EPS-grid локальная линейность",
                "",
                linearity[
                    ["eps", "variable", "norm_ratio_to_reference", "relative_matrix_gap_to_reference", "max_abs_gap_to_reference"]
                ].to_markdown(index=False, floatfmt=".4g"),
            ]
        )
    if not mvoi_grid.empty:
        main = mvoi_grid[
            mvoi_grid["comparison"].eq("filtered_distribution_minus_filtered_aggregates")
            & mvoi_grid["scenario"].eq("all")
        ]
        if not main.empty:
            lines.extend(
                [
                    "",
                    "## MVOI eps-grid",
                    "",
                    main[["eps", "mode", "scenario", "loss_reduction", "sign_flip_p_value"]].to_markdown(index=False, floatfmt=".4g"),
                ]
            )
    lines.extend(["", "## Gate checks", ""])
    for check in gate["checks"]:
        lines.append(f"- {check['name']}: {'PASS' if check['passed'] else 'FAIL'}; value={check['value']}.")
    (output_dir / "report_distributional_jacobian_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
