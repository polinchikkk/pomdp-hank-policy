from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .model import NKParameters, SHOCK_NAMES, STATE_NAMES, model_spec_payload
from .solver import LinearNKSolution, determinacy_diagnostics, solve_linear_nk_model

PLOT_SCALE = 100.0
POLICY_SERIES = (
    ("x", "Output gap", "Percent"),
    ("pi", "Inflation", "Percentage points"),
    ("i", "Nominal rate", "Percentage points"),
)
SHOCK_SERIES = (
    ("r_n", "Natural-rate shock", "Percentage points"),
    ("u", "Cost-push shock", "Percentage points"),
    ("nu", "Monetary shock", "Percentage points"),
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _serialize_complex_vector(values: np.ndarray) -> list[dict[str, float | bool | None]]:
    payload: list[dict[str, float | bool | None]] = []
    for value in np.asarray(values).ravel():
        if not np.isfinite(value):
            payload.append({"real": None, "imag": None, "is_infinite": True})
            continue
        payload.append(
            {
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
                "is_infinite": False,
            }
        )
    return payload


def _observables_from_state_and_control(state: np.ndarray, control: np.ndarray) -> dict[str, float]:
    return {
        "r_n": float(state[0]),
        "u": float(state[1]),
        "nu": float(state[2]),
        "x": float(control[0]),
        "pi": float(control[1]),
        "i": float(control[2]),
    }


def _expected_residuals(solution: LinearNKSolution, state: np.ndarray) -> np.ndarray:
    control = solution.controls(state)
    expected_state_tp1 = solution.transition_matrix @ state
    expected_control_tp1 = solution.controls(expected_state_tp1)
    matrices = solution.system_matrices
    return (
        matrices["f_state_t"] @ state
        + matrices["f_control_t"] @ control
        + matrices["f_state_tp1"] @ expected_state_tp1
        + matrices["f_control_tp1"] @ expected_control_tp1
    )


def simulate_paths(
    solution: LinearNKSolution,
    periods: int,
    seed: int,
    burn_in: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    total_periods = periods + burn_in
    states = np.zeros((total_periods, len(STATE_NAMES)), dtype=float)
    realized_innovations = np.zeros((total_periods, len(SHOCK_NAMES)), dtype=float)

    rows: list[dict[str, float | int]] = []
    for period in range(total_periods):
        state = states[period]
        control = solution.controls(state)
        residuals = _expected_residuals(solution=solution, state=state)
        row: dict[str, float | int] = {
            "t": period - burn_in,
            "eps_demand": float(realized_innovations[period, 0]),
            "eps_costpush": float(realized_innovations[period, 1]),
            "eps_monetary": float(realized_innovations[period, 2]),
            "internal_linear_residual_max_abs": float(np.max(np.abs(residuals))),
        }
        row.update(_observables_from_state_and_control(state=state, control=control))
        rows.append(row)

        if period + 1 < total_periods:
            innovation = rng.standard_normal(len(SHOCK_NAMES))
            realized_innovations[period + 1] = innovation
            states[period + 1] = solution.next_state(state, innovation)

    frame = pd.DataFrame(rows)
    if burn_in > 0:
        frame = frame.loc[frame["t"] >= 0].reset_index(drop=True)
    return frame


def compute_irf(
    solution: LinearNKSolution,
    shock_name: str,
    horizon: int,
    shock_size: float = 1.0,
) -> pd.DataFrame:
    shock_index = SHOCK_NAMES.index(shock_name)
    state = solution.shock_matrix[:, shock_index] * shock_size

    rows: list[dict[str, float | int | str]] = []
    for period in range(horizon):
        control = solution.controls(state)
        row: dict[str, float | int | str] = {
            "t": period,
            "shock_name": shock_name,
            "epsilon_t": shock_size if period == 0 else 0.0,
        }
        row.update(_observables_from_state_and_control(state=state, control=control))
        rows.append(row)
        state = solution.transition_matrix @ state

    return pd.DataFrame(rows)


def _determinacy_map(
    base_params: NKParameters,
    phi_pi_grid: np.ndarray,
    phi_x_grid: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | None]] = []
    for phi_x in phi_x_grid:
        for phi_pi in phi_pi_grid:
            params = NKParameters(
                beta=base_params.beta,
                sigma=base_params.sigma,
                kappa=base_params.kappa,
                phi_pi=float(phi_pi),
                phi_x=float(phi_x),
                rho_r=base_params.rho_r,
                rho_u=base_params.rho_u,
                rho_nu=base_params.rho_nu,
                sigma_r=base_params.sigma_r,
                sigma_u=base_params.sigma_u,
                sigma_nu=base_params.sigma_nu,
            )
            diagnostics = determinacy_diagnostics(params)
            rows.append(
                {
                    "phi_pi": float(phi_pi),
                    "phi_x": float(phi_x),
                    "determinate": bool(diagnostics["determinate"]),
                    "stable_root_count": int(diagnostics["stable_root_count"]),
                    "roots_outside_unit_circle_count": int(
                        diagnostics["roots_outside_unit_circle_count"]
                    ),
                    "infinite_root_count": int(diagnostics["infinite_root_count"]),
                    "spectral_radius": diagnostics["spectral_radius"],
                }
            )
    return pd.DataFrame(rows)


def _plot_three_panel(
    data: pd.DataFrame,
    series: tuple[tuple[str, str, str], ...],
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 8.5), sharex=True)
    for axis, (column, label, ylabel) in zip(axes, series, strict=True):
        axis.axhline(0.0, color="#9aa5b1", linewidth=0.8)
        axis.plot(data["t"], PLOT_SCALE * data[column], color="#355070", linewidth=1.6)
        axis.set_title(label)
        axis.set_ylabel(ylabel)
    axes[-1].set_xlabel(x_label)
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_irf_policy(irf: pd.DataFrame, output_path: Path, title: str) -> None:
    _plot_three_panel(data=irf, series=POLICY_SERIES, output_path=output_path, title=title, x_label="Horizon")


def _plot_irf_shocks(irf: pd.DataFrame, output_path: Path, title: str) -> None:
    _plot_three_panel(
        data=irf,
        series=SHOCK_SERIES,
        output_path=output_path,
        title=title,
        x_label="Horizon",
    )


def _plot_simulated_policy_paths(simulation: pd.DataFrame, output_path: Path) -> None:
    _plot_three_panel(
        data=simulation,
        series=POLICY_SERIES,
        output_path=output_path,
        title="Stage 2 NK Simulation: Policy Variables",
        x_label="Period",
    )


def _plot_simulated_shock_paths(simulation: pd.DataFrame, output_path: Path) -> None:
    _plot_three_panel(
        data=simulation,
        series=SHOCK_SERIES,
        output_path=output_path,
        title="Stage 2 NK Simulation: Exogenous Shocks",
        x_label="Period",
    )


def _plot_determinacy_map(
    determinacy_map: pd.DataFrame,
    phi_pi_grid: np.ndarray,
    phi_x_grid: np.ndarray,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    pivot = (
        determinacy_map.pivot(index="phi_x", columns="phi_pi", values="determinate")
        .reindex(index=phi_x_grid, columns=phi_pi_grid)
        .astype(float)
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    mesh = axis.pcolormesh(phi_pi_grid, phi_x_grid, pivot.to_numpy(), shading="nearest", cmap="RdYlBu")
    cbar = figure.colorbar(mesh, ax=axis)
    cbar.set_label("Determinate region")
    axis.set_xlabel("phi_pi")
    axis.set_ylabel("phi_x")
    axis.set_title("Determinacy Map")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_stage2_report(
    output_path: Path,
    params: NKParameters,
    diagnostics_summary: dict,
) -> None:
    report = f"""# Stage 2 Report: Small NK Policy Baseline

## Model

- Baseline environment: canonical linear New Keynesian model in gap variables around the zero steady state.
- The model is written directly in linearized deviation variables, so the steady-state reference point is normalized to zero.
- Endogenous policy variables: output gap `x_t`, inflation `pi_t`, nominal rate `i_t`.
- Exogenous shocks: natural-rate shock `r^n_t`, cost-push shock `u_t`, monetary-policy shock `nu_t`.
- Policy rule: `i_t = phi_pi * pi_t + phi_x * x_t + nu_t`.

## Calibration

- beta = {params.beta:.3f}
- sigma = {params.sigma:.3f}
- kappa = {params.kappa:.3f}
- phi_pi = {params.phi_pi:.3f}
- phi_x = {params.phi_x:.3f}
- rho_r = {params.rho_r:.3f}
- rho_u = {params.rho_u:.3f}
- rho_nu = {params.rho_nu:.3f}

## Solution

- Solver: {diagnostics_summary['solver_name']}
- Stable roots: {diagnostics_summary['stable_root_count']}
- Roots outside the unit circle: {diagnostics_summary['roots_outside_unit_circle_count']}
- Infinite generalized eigenvalues: {diagnostics_summary['infinite_root_count']}
- Blanchard-Kahn condition satisfied: {diagnostics_summary['bk_condition_satisfied']}
- Spectral radius of the transition matrix: {diagnostics_summary['spectral_radius']:.4f}
- Linear-system residual max: {diagnostics_summary['linear_solution_residual_max']:.3e}
- Runtime: {diagnostics_summary['runtime_seconds']:.3f} seconds

## IRF Sanity Checks

- Positive demand shock: `x`, `pi`, and `i` all rise on impact -> {diagnostics_summary['demand_irf_signs']}
- Positive cost-push shock: inflation rises and output falls on impact -> {diagnostics_summary['costpush_irf_signs']}
- Positive monetary tightening shock: output gap and inflation fall while the nominal rate rises on impact -> {diagnostics_summary['monetary_irf_signs']}

## Baseline Policy Diagnostics

- Determinacy map computed on a grid over `(phi_pi, phi_x)`.
- Determinate parameter combinations in the grid: {diagnostics_summary['determinate_share']:.1%}
- For `phi_pi > 1`, the entire scanned grid is determinate: {diagnostics_summary['determinate_share_phi_pi_above_one']:.1%}
- For `phi_pi <= 1`, determinacy weakens sharply: {diagnostics_summary['determinate_share_phi_pi_at_or_below_one']:.1%}
- This pattern is consistent with the Taylor principle in the scanned parameter region.
- Simulated paths remain finite: {diagnostics_summary['finite_paths']}
- Internal linear-equation consistency check: {diagnostics_summary['max_abs_internal_linear_residual']:.3e}
- This residual metric confirms that the computed policy functions solve the linearized system; the main economic validation still comes from IRF logic and determinacy.

## Not Implemented Yet

- Hidden states or signal extraction.
- Regime switching or time variation in the policy rule.
- RL or adaptive policy design.
- Heterogeneous agents and real-data estimation.
"""
    (output_path / "stage2_report.md").write_text(report, encoding="utf-8")


def run_stage2_pipeline(
    output_dir: str | Path,
    periods: int = 240,
    burn_in: int = 80,
    irf_horizon: int = 24,
    seed: int = 123,
) -> dict:
    start_time = time.perf_counter()
    output_path = Path(output_dir)
    figure_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    params = NKParameters()
    solution = solve_linear_nk_model(params=params)

    irf_demand = compute_irf(solution=solution, shock_name="demand", horizon=irf_horizon)
    irf_costpush = compute_irf(solution=solution, shock_name="costpush", horizon=irf_horizon)
    irf_monetary = compute_irf(solution=solution, shock_name="monetary", horizon=irf_horizon)
    simulation = simulate_paths(solution=solution, periods=periods, burn_in=burn_in, seed=seed)

    phi_pi_grid = np.linspace(0.5, 2.5, 41)
    phi_x_grid = np.linspace(0.0, 1.0, 26)
    determinacy_map = _determinacy_map(
        base_params=params,
        phi_pi_grid=phi_pi_grid,
        phi_x_grid=phi_x_grid,
    )

    runtime_seconds = time.perf_counter() - start_time
    diagnostics_summary = {
        "solver_name": solution.solver_name,
        "linear_solution_residual_max": solution.residual_norm,
        "spectral_radius": solution.spectral_radius,
        "stable_root_count": solution.stable_root_count,
        "roots_outside_unit_circle_count": solution.roots_outside_unit_circle_count,
        "infinite_root_count": solution.infinite_root_count,
        "bk_condition_satisfied": solution.bk_condition_satisfied,
        "invariant_block_condition_number": solution.invariant_block_condition_number,
        "runtime_seconds": runtime_seconds,
        "simulation_seed": seed,
        "simulation_periods": periods,
        "burn_in": burn_in,
        "irf_horizon": irf_horizon,
        "finite_paths": bool(np.isfinite(simulation.to_numpy()).all()),
        "max_abs_internal_linear_residual": float(
            np.max(np.abs(simulation["internal_linear_residual_max_abs"].to_numpy()))
        ),
        "mean_abs_internal_linear_residual": float(
            np.mean(np.abs(simulation["internal_linear_residual_max_abs"].to_numpy()))
        ),
        "demand_irf_signs": {
            "x_positive": bool(irf_demand.iloc[0]["x"] > 0.0),
            "pi_positive": bool(irf_demand.iloc[0]["pi"] > 0.0),
            "i_positive": bool(irf_demand.iloc[0]["i"] > 0.0),
        },
        "costpush_irf_signs": {
            "x_negative": bool(irf_costpush.iloc[0]["x"] < 0.0),
            "pi_positive": bool(irf_costpush.iloc[0]["pi"] > 0.0),
            "i_positive": bool(irf_costpush.iloc[0]["i"] > 0.0),
        },
        "monetary_irf_signs": {
            "x_negative": bool(irf_monetary.iloc[0]["x"] < 0.0),
            "pi_negative": bool(irf_monetary.iloc[0]["pi"] < 0.0),
            "i_positive": bool(irf_monetary.iloc[0]["i"] > 0.0),
        },
        "determinate_share": float(determinacy_map["determinate"].mean()),
        "determinate_share_phi_pi_above_one": float(
            determinacy_map.loc[determinacy_map["phi_pi"] > 1.0, "determinate"].mean()
        ),
        "determinate_share_phi_pi_at_or_below_one": float(
            determinacy_map.loc[determinacy_map["phi_pi"] <= 1.0, "determinate"].mean()
        ),
    }

    solution_payload = {
        "solver_name": solution.solver_name,
        "policy_matrix": solution.policy_matrix.tolist(),
        "transition_matrix": solution.transition_matrix.tolist(),
        "shock_matrix": solution.shock_matrix.tolist(),
        "spectral_radius": solution.spectral_radius,
        "residual_norm": solution.residual_norm,
        "stable_root_count": solution.stable_root_count,
        "roots_outside_unit_circle_count": solution.roots_outside_unit_circle_count,
        "infinite_root_count": solution.infinite_root_count,
        "bk_condition_satisfied": solution.bk_condition_satisfied,
        "invariant_block_condition_number": solution.invariant_block_condition_number,
        "generalized_eigenvalues": _serialize_complex_vector(solution.generalized_eigenvalues),
    }

    _plot_irf_policy(
        irf=irf_demand,
        output_path=figure_dir / "irf_demand.png",
        title="IRF to a Demand Shock: Policy Variables",
    )
    _plot_irf_shocks(
        irf=irf_demand,
        output_path=figure_dir / "irf_demand_shocks.png",
        title="IRF to a Demand Shock: Exogenous States",
    )
    _plot_irf_policy(
        irf=irf_costpush,
        output_path=figure_dir / "irf_costpush.png",
        title="IRF to a Cost-Push Shock: Policy Variables",
    )
    _plot_irf_shocks(
        irf=irf_costpush,
        output_path=figure_dir / "irf_costpush_shocks.png",
        title="IRF to a Cost-Push Shock: Exogenous States",
    )
    _plot_irf_policy(
        irf=irf_monetary,
        output_path=figure_dir / "irf_monetary.png",
        title="IRF to a Monetary Tightening Shock: Policy Variables",
    )
    _plot_irf_shocks(
        irf=irf_monetary,
        output_path=figure_dir / "irf_monetary_shocks.png",
        title="IRF to a Monetary Tightening Shock: Exogenous States",
    )
    _plot_simulated_policy_paths(
        simulation=simulation.iloc[:120],
        output_path=figure_dir / "simulated_paths.png",
    )
    _plot_simulated_shock_paths(
        simulation=simulation.iloc[:120],
        output_path=figure_dir / "simulated_shocks.png",
    )
    _plot_determinacy_map(
        determinacy_map=determinacy_map,
        phi_pi_grid=phi_pi_grid,
        phi_x_grid=phi_x_grid,
        output_path=figure_dir / "determinacy_map.png",
    )

    _write_json(output_path / "model_spec.json", model_spec_payload(params))
    _write_json(output_path / "solution.json", solution_payload)
    _write_json(output_path / "diagnostics_summary.json", diagnostics_summary)
    irf_demand.to_csv(output_path / "irf_demand.csv", index=False)
    irf_costpush.to_csv(output_path / "irf_costpush.csv", index=False)
    irf_monetary.to_csv(output_path / "irf_monetary.csv", index=False)
    simulation.to_csv(output_path / "simulated_paths.csv", index=False)
    determinacy_map.to_csv(output_path / "determinacy_map.csv", index=False)
    _write_stage2_report(output_path=output_path, params=params, diagnostics_summary=diagnostics_summary)

    return {
        "params": params,
        "solution": solution,
        "irf_demand": irf_demand,
        "irf_costpush": irf_costpush,
        "irf_monetary": irf_monetary,
        "simulation": simulation,
        "determinacy_map": determinacy_map,
        "diagnostics_summary": diagnostics_summary,
    }
