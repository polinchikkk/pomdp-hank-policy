from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .calibration import HANKCalibration, default_calibration
from .experiments import _ar1_shock_path
from .steady_state import solve_steady_state
from .transition import DEFAULT_TRANSITION_OUTPUTS, solve_transition


DEFAULT_SHOCK_TYPES = ("monetary_policy_shock", "rstar", "Z", "G")
SOLVER_TARGET_RESIDUALS = ("asset_mkt", "fisher", "wnkpc")
TRANSITION_RESIDUAL_TARGETS = (*SOLVER_TARGET_RESIDUALS, "goods_mkt")


def load_calibration_from_core(core_dir: Path) -> HANKCalibration:
    path = Path(core_dir) / "calibration.json"
    if not path.exists():
        return default_calibration()
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = set(HANKCalibration.__dataclass_fields__)
    return HANKCalibration(**{key: value for key, value in payload.items() if key in valid})


def audit_steady_state_artifacts(core_dir: Path, *, config: HANKCalibration | None = None) -> dict[str, object]:
    core_dir = Path(core_dir)
    config = load_calibration_from_core(core_dir) if config is None else config
    distribution_path = core_dir / "data" / "steady_state_distribution.npz"
    policies_path = core_dir / "data" / "steady_state_policies.npz"
    transition_path = core_dir / "data" / "transition_household_paths.npz"
    aggregates_path = core_dir / "steady_state_aggregates.json"
    for path in (distribution_path, policies_path, aggregates_path):
        if not path.exists():
            raise FileNotFoundError(f"Required HANK core artifact is missing: {path}")

    aggregates = json.loads(aggregates_path.read_text(encoding="utf-8"))
    with np.load(distribution_path) as distribution, np.load(policies_path) as policies:
        D = np.asarray(distribution["D"], dtype=float)
        Dbeg = np.asarray(distribution["Dbeg"], dtype=float)
        b_grid = np.asarray(distribution["b_grid"], dtype=float)
        a_grid = np.asarray(distribution["a_grid"], dtype=float)
        z_grid = np.asarray(distribution["z_grid"], dtype=float)
        policy_a = np.asarray(policies["a"], dtype=float)
        policy_b = np.asarray(policies["b"], dtype=float)
        c = np.asarray(policies["c"], dtype=float)
        chi = np.asarray(policies["chi"], dtype=float)
        Va = np.asarray(policies["Va"], dtype=float)
        Vb = np.asarray(policies["Vb"], dtype=float)
        mpc = np.asarray(policies["mpc"], dtype=float) if "mpc" in policies else np.full_like(D, np.nan)
        transfer_mpc = (
            np.asarray(policies["transfer_mpc"], dtype=float)
            if "transfer_mpc" in policies
            else np.full_like(D, np.nan)
        )

    mesh_z = z_grid[:, None, None]
    mesh_b = b_grid[None, :, None]
    mesh_a = a_grid[None, None, :]
    rb = float(aggregates.get("rb", np.nan))
    ra = float(aggregates.get("ra", np.nan))
    budget_residual = mesh_z + (1.0 + rb) * mesh_b + (1.0 + ra) * mesh_a - chi - policy_a - policy_b - c
    marginal_utility = np.maximum(c, 1e-300) ** (-1.0 / float(config.eis))
    euler_proxy_log_gap = np.log(np.maximum(Vb, 1e-300)) - np.log(marginal_utility)

    A_from_distribution = float(np.sum(D * policy_a))
    B_from_distribution = float(np.sum(D * policy_b))
    C_from_distribution = float(np.sum(D * c))
    liquid_borrowing_share = float(np.sum(D[:, b_grid <= b_grid[0] + 1e-12, :]))
    illiquid_lower_bound_share = float(np.sum(D[:, :, a_grid <= a_grid[0] + 1e-12]))
    low_liquidity_share = float(np.sum(D[:, b_grid <= float(config.low_liquidity_threshold) + 1e-12, :]))

    transition_distribution = {}
    if transition_path.exists():
        with np.load(transition_path) as transition:
            D_path = np.asarray(transition["D"], dtype=float)
            mass_by_period = D_path.reshape(D_path.shape[0], -1).sum(axis=1)
            transition_distribution = {
                "available": True,
                "num_periods": int(D_path.shape[0]),
                "max_mass_error": float(np.max(np.abs(mass_by_period - 1.0))),
                "min_mass": float(np.min(mass_by_period)),
                "max_mass": float(np.max(mass_by_period)),
                "min_distribution": float(np.min(D_path)),
                "negative_entries": int(np.sum(D_path < -1e-12)),
                "nonfinite_entries": int(np.sum(~np.isfinite(D_path))),
            }
    else:
        transition_distribution = {"available": False}

    market = {
        "goods_mkt": float(aggregates.get("goods_mkt", np.nan)),
        "asset_mkt": float(aggregates.get("asset_mkt", np.nan)),
        "max_abs_market_clearing_residual": float(
            np.nanmax(np.abs([aggregates.get("goods_mkt", np.nan), aggregates.get("asset_mkt", np.nan)]))
        ),
    }
    aggregate_consistency = {
        "A_from_distribution": A_from_distribution,
        "B_from_distribution": B_from_distribution,
        "C_from_distribution": C_from_distribution,
        "A_reported": float(aggregates.get("A", np.nan)),
        "B_reported": float(aggregates.get("B", np.nan)),
        "C_reported": float(aggregates.get("C", np.nan)),
        "A_gap": A_from_distribution - float(aggregates.get("A", np.nan)),
        "B_gap": B_from_distribution - float(aggregates.get("B", np.nan)),
        "C_gap": C_from_distribution - float(aggregates.get("C", np.nan)),
    }
    return {
        "core_dir": str(core_dir),
        "distribution": {
            "mass": float(np.sum(D)),
            "mass_error": float(abs(np.sum(D) - 1.0)),
            "beginning_of_period_mass": float(np.sum(Dbeg)),
            "beginning_of_period_mass_error": float(abs(np.sum(Dbeg) - 1.0)),
            "min": float(np.min(D)),
            "max": float(np.max(D)),
            "negative_entries": int(np.sum(D < -1e-12)),
            "negative_mass": float(np.sum(np.abs(D[D < 0.0]))),
            "nonfinite_entries": int(np.sum(~np.isfinite(D))),
            "shape": list(D.shape),
        },
        "borrowing_constraints": {
            "liquid_lower_bound": float(b_grid[0]),
            "illiquid_lower_bound": float(a_grid[0]),
            "liquid_borrowing_constraint_share": liquid_borrowing_share,
            "illiquid_lower_bound_share": illiquid_lower_bound_share,
            "low_liquidity_threshold": float(config.low_liquidity_threshold),
            "low_liquidity_share": low_liquidity_share,
            "policy_b_min": float(np.min(policy_b)),
            "policy_a_min": float(np.min(policy_a)),
            "policy_b_below_grid_min_count": int(np.sum(policy_b < b_grid[0] - 1e-10)),
            "policy_a_below_grid_min_count": int(np.sum(policy_a < a_grid[0] - 1e-10)),
        },
        "policy_residuals": {
            "budget_max_abs_residual": float(np.max(np.abs(budget_residual))),
            "budget_weighted_mean_abs_residual": float(np.sum(D * np.abs(budget_residual))),
            "euler_proxy_vb_log_gap_max_abs": float(np.max(np.abs(euler_proxy_log_gap))),
            "euler_proxy_vb_log_gap_weighted_mean_abs": float(np.sum(D * np.abs(euler_proxy_log_gap))),
            "min_consumption": float(np.min(c)),
            "min_Va": float(np.min(Va)),
            "min_Vb": float(np.min(Vb)),
            "nonfinite_policy_entries": int(
                np.sum(~np.isfinite(policy_a))
                + np.sum(~np.isfinite(policy_b))
                + np.sum(~np.isfinite(c))
                + np.sum(~np.isfinite(Va))
                + np.sum(~np.isfinite(Vb))
            ),
        },
        "distributional_statistics": {
            "mean_mpc": float(np.sum(D * mpc)),
            "mean_transfer_mpc": float(np.sum(D * transfer_mpc)),
            "share_mpc_above_0_2": float(np.sum(D[mpc > 0.2])),
            "share_transfer_mpc_above_0_2": float(np.sum(D[transfer_mpc > 0.2])),
            "interest_exposure": float(np.sum(D * mesh_b * mpc)),
        },
        "market_clearing": market,
        "aggregate_consistency": aggregate_consistency,
        "steady_state_aggregates": {key: float(value) for key, value in aggregates.items()},
        "stored_transition_distribution": transition_distribution,
    }


def audit_transition_solvers(
    config: HANKCalibration,
    *,
    shock_types: Iterable[str] = DEFAULT_SHOCK_TYPES,
    horizon: int | None = None,
    shock_size: float | None = None,
    residual_tolerance: float = 1e-6,
) -> pd.DataFrame:
    horizon = int(config.shock_T if horizon is None else horizon)
    config = replace(config, shock_T=horizon)
    bundle = solve_steady_state(config)
    rows: list[dict[str, object]] = []
    outputs = list(dict.fromkeys([*DEFAULT_TRANSITION_OUTPUTS, *TRANSITION_RESIDUAL_TARGETS]))
    for shock_type in shock_types:
        size = float(config.mp_shock_size if shock_size is None else shock_size)
        inputs = {str(shock_type): _ar1_shock_path(size=size, period=config.shock_period, persistence=0.0, horizon=horizon)}
        row: dict[str, object] = {
            "shock_type": str(shock_type),
            "shock_size": size,
            "horizon": horizon,
            "solver_converged": False,
            "residual_converged": False,
            "max_solver_target_residual": np.nan,
            "max_transition_residual": np.nan,
            "max_distribution_mass_error": np.nan,
            "min_transition_distribution": np.nan,
            "exception": "",
        }
        try:
            transition = solve_transition(bundle, inputs, outputs=outputs)
            residuals: dict[str, float] = {}
            for target in TRANSITION_RESIDUAL_TARGETS:
                if target in transition:
                    values = np.asarray(transition[target], dtype=float)
                    residuals[target] = float(np.nanmax(np.abs(values)))
                    row[f"max_{target}_residual"] = residuals[target]
                else:
                    row[f"max_{target}_residual"] = np.nan
            if transition.internals and "hh" in transition.internals and "D" in transition.internals["hh"]:
                D_path = np.asarray(transition.internals["hh"]["D"], dtype=float) + np.asarray(
                    bundle["ss"].internals["hh"]["D"],
                    dtype=float,
                )[None, ...]
                mass_by_period = D_path.reshape(D_path.shape[0], -1).sum(axis=1)
                row["max_distribution_mass_error"] = float(np.nanmax(np.abs(mass_by_period - 1.0)))
                row["min_transition_distribution"] = float(np.nanmin(D_path))
            target_residuals = [
                value
                for target, value in residuals.items()
                if target in SOLVER_TARGET_RESIDUALS and np.isfinite(value)
            ]
            max_solver_target_residual = float(np.nanmax(target_residuals)) if target_residuals else np.nan
            max_residual = float(np.nanmax(list(residuals.values()))) if residuals else np.nan
            row["max_solver_target_residual"] = max_solver_target_residual
            row["max_transition_residual"] = max_residual
            row["solver_converged"] = True
            row["residual_converged"] = bool(
                np.isfinite(max_solver_target_residual)
                and max_solver_target_residual <= float(residual_tolerance)
            )
        except Exception as exc:  # noqa: BLE001 - audit rows should record solver failures.
            row["exception"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_hank_core_audit(
    *,
    core_dir: Path,
    output_dir: Path,
    config: HANKCalibration | None = None,
    shock_types: Iterable[str] = DEFAULT_SHOCK_TYPES,
    transition_horizon: int | None = None,
    skip_transition_solves: bool = False,
    residual_tolerance: float = 1e-6,
) -> tuple[dict[str, object], pd.DataFrame]:
    core_dir = Path(core_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_calibration_from_core(core_dir) if config is None else config
    steady = audit_steady_state_artifacts(core_dir, config=config)
    if skip_transition_solves:
        transitions = pd.DataFrame(
            [
                {
                    "shock_type": "transition_solves_skipped",
                    "solver_converged": False,
                    "residual_converged": False,
                    "exception": "transition solves skipped by CLI flag",
                }
            ]
        )
    else:
        transitions = audit_transition_solvers(
            config,
            shock_types=shock_types,
            horizon=transition_horizon,
            residual_tolerance=residual_tolerance,
        )
    (output_dir / "steady_state_audit.json").write_text(
        json.dumps(steady, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    transitions.to_csv(output_dir / "transition_audit.csv", index=False)
    _write_report(steady=steady, transitions=transitions, output_path=output_dir / "report_hank_core_audit.md")
    return steady, transitions


def _write_report(*, steady: dict[str, object], transitions: pd.DataFrame, output_path: Path) -> None:
    distribution = steady["distribution"]
    policy = steady["policy_residuals"]
    market = steady["market_clearing"]
    constraints = steady["borrowing_constraints"]
    aggregates = steady["steady_state_aggregates"]
    failed = transitions[~transitions.get("residual_converged", False).astype(bool)] if not transitions.empty else transitions
    lines = [
        "# HANK Core Audit",
        "",
        "This audit makes the HANK source of trajectories visible before the SSJ/VOI layer uses it.",
        "",
        "## Steady State",
        "",
        f"- Distribution mass: {distribution['mass']:.12g}; mass error {distribution['mass_error']:.3g}.",
        f"- Minimum distribution entry: {distribution['min']:.3g}; negative entries: {distribution['negative_entries']}.",
        (
            f"- Liquid borrowing-constraint share: {constraints['liquid_borrowing_constraint_share']:.6g}; "
            f"low-liquidity share: {constraints['low_liquidity_share']:.6g}."
        ),
        f"- Budget policy residual max abs: {policy['budget_max_abs_residual']:.3g}.",
        f"- Euler proxy Vb/u'(c) log-gap max abs: {policy['euler_proxy_vb_log_gap_max_abs']:.3g}.",
        f"- Market clearing max abs residual: {market['max_abs_market_clearing_residual']:.3g}.",
        "",
        "## Aggregates",
        "",
    ]
    for key in ("Y", "C", "I", "N", "pi", "i", "r", "A", "B", "wealth"):
        if key in aggregates:
            lines.append(f"- {key}: {aggregates[key]:.12g}.")
    lines.extend(["", "## Transition Solves", ""])
    if transitions.empty:
        lines.append("No transition audit rows were produced.")
    else:
        lines.append(transitions.to_markdown(index=False, floatfmt=".6g"))
    lines.extend(["", "## Status", ""])
    if failed.empty:
        lines.append("All audited transition solves satisfy the residual convergence threshold.")
    else:
        lines.append("Some transition solves failed the residual convergence threshold or were skipped:")
        for _, row in failed.iterrows():
            lines.append(
                f"- {row.get('shock_type')}: max residual {row.get('max_transition_residual')}; "
                f"exception `{row.get('exception', '')}`."
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return asdict(value)
    raise TypeError(f"Unsupported type: {type(value)!r}")
