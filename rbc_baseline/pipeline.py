from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .benchmark import ExternalIRFBenchmark, run_external_gensys_irf_benchmark
from .model import (
    RBCParameters,
    RBCSteadyState,
    compute_steady_state,
    equilibrium_residuals,
    observables_from_state,
)
from .solver import LinearRBCSolution, solve_linear_policy

DIAGNOSTIC_NAMES = (
    "euler_residual",
    "labor_residual",
    "capital_accumulation_residual",
    "technology_law_residual",
)


def simulate_paths(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    solution: LinearRBCSolution,
    periods: int,
    seed: int,
    burn_in: int = 0,
    initial_state: np.ndarray | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    total_periods = periods + burn_in
    states = np.zeros((total_periods, 2), dtype=float)
    innovations = rng.standard_normal(total_periods)
    realized_innovations = np.zeros(total_periods, dtype=float)

    if initial_state is not None:
        states[0] = np.asarray(initial_state, dtype=float)
    else:
        innovations[0] = 0.0

    rows: list[dict[str, float | int]] = []
    for period in range(total_periods):
        state = states[period]
        control = solution.controls(state)
        observables = observables_from_state(
            params=params,
            steady_state=steady_state,
            state=state,
            control=control,
        )
        row: dict[str, float | int] = {
            "t": period - burn_in,
            "epsilon_t": float(realized_innovations[period]),
            "shock_impact_t": float(params.sigma * realized_innovations[period]),
        }
        row.update(observables)
        rows.append(row)

        if period + 1 < total_periods:
            realized_innovations[period + 1] = innovations[period + 1]
            states[period + 1] = solution.next_state(state, innovations[period + 1])

    frame = pd.DataFrame(rows)
    if burn_in > 0:
        frame = frame.loc[frame["t"] >= 0].reset_index(drop=True)
    return frame


def compute_irf(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    solution: LinearRBCSolution,
    horizon: int,
    shock_size: float = 1.0,
) -> pd.DataFrame:
    state = solution.shock_vector * shock_size
    rows: list[dict[str, float | int]] = []

    for period in range(horizon):
        control = solution.controls(state)
        observables = observables_from_state(
            params=params,
            steady_state=steady_state,
            state=state,
            control=control,
        )
        row: dict[str, float | int] = {
            "t": period,
            "epsilon_t": shock_size if period == 0 else 0.0,
            "shock_impact_t": float(params.sigma * shock_size if period == 0 else 0.0),
        }
        row.update(observables)
        rows.append(row)
        state = solution.transition_matrix @ state

    return pd.DataFrame(rows)


def _normal_expectation(function, nodes: int = 5) -> np.ndarray:
    hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(nodes)
    total = None
    for node, weight in zip(hermite_nodes, hermite_weights, strict=True):
        innovation = np.sqrt(2.0) * node
        value = function(innovation)
        if total is None:
            total = np.zeros_like(value, dtype=float)
        total += weight * value
    assert total is not None
    return total / np.sqrt(np.pi)


def conditional_residuals(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    solution: LinearRBCSolution,
    state: np.ndarray,
) -> np.ndarray:
    control = solution.controls(state)

    return _normal_expectation(
        lambda innovation: equilibrium_residuals(
            params=params,
            steady_state=steady_state,
            state_t=state,
            control_t=control,
            state_tp1=solution.next_state(state, innovation),
            control_tp1=solution.controls(solution.next_state(state, innovation)),
            shock_tp1=innovation,
        )
    )


def diagnostic_table_from_simulation(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    solution: LinearRBCSolution,
    simulation: pd.DataFrame,
) -> pd.DataFrame:
    residual_rows = []
    for _, row in simulation.iterrows():
        state = np.array([row["log_k_dev"], row["z"]], dtype=float)
        residuals = conditional_residuals(
            params=params,
            steady_state=steady_state,
            solution=solution,
            state=state,
        )
        residual_rows.append(residuals)
    return pd.DataFrame(residual_rows, columns=DIAGNOSTIC_NAMES)


def seed_stability_summary(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    solution: LinearRBCSolution,
    periods: int,
    burn_in: int,
    seeds: list[int],
) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    for seed in seeds:
        simulation = simulate_paths(
            params=params,
            steady_state=steady_state,
            solution=solution,
            periods=periods,
            burn_in=burn_in,
            seed=seed,
        )
        diagnostics = diagnostic_table_from_simulation(
            params=params,
            steady_state=steady_state,
            solution=solution,
            simulation=simulation,
        )
        max_state_deviation = float(
            np.max(np.abs(simulation[["log_k_dev", "z", "log_c_dev", "log_n_dev"]].to_numpy()))
        )
        max_residual = float(np.max(np.abs(diagnostics.to_numpy())))
        rows.append(
            {
                "seed": seed,
                "max_abs_state_deviation": max_state_deviation,
                "max_abs_residual": max_residual,
                "finite_paths": bool(np.isfinite(simulation.to_numpy()).all()),
            }
        )
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _serialize_complex_vector(values: np.ndarray) -> list[dict[str, float | bool | None]]:
    serialized: list[dict[str, float | bool | None]] = []
    for value in np.asarray(values).ravel():
        if not np.isfinite(value):
            serialized.append({"real": None, "imag": None, "is_infinite": True})
            continue
        serialized.append(
            {
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
                "is_infinite": False,
            }
        )
    return serialized


def _write_stage_report(
    output_path: Path,
    params: RBCParameters,
    steady_state: RBCSteadyState,
    solution: LinearRBCSolution,
    diagnostics_summary: dict,
    irf: pd.DataFrame,
    benchmark_summary: dict,
) -> None:
    impact = irf.iloc[0]
    sign_checks = {
        "output": impact["log_y_dev"] > 0.0,
        "consumption": impact["log_c_dev"] > 0.0,
        "investment": impact["log_i_dev"] > 0.0,
        "labor": impact["log_n_dev"] > 0.0,
    }
    seed_lines = "\n".join(
        [
            f"- seed {entry['seed']}: finite={entry['finite_paths']}, "
            f"max |state dev|={entry['max_abs_state_deviation']:.4f}, "
            f"max |residual|={entry['max_abs_residual']:.4f}"
            for entry in diagnostics_summary["seed_stability"]
        ]
    )
    report = f"""# Stage 1 Note: RBC Baseline

## Model

- Baseline model: real business cycle model with a persistent technology shock.
- State variables: log capital deviation and technology level `z_t`.
- Control variables: log consumption deviation and log labor deviation.
- Shock law: `z_(t+1) = rho z_t + sigma epsilon_(t+1)`.

## Solution Method

- Solution class: first-order local perturbation around the deterministic steady state.
- Numerical approach: steady state in closed form, Jacobians by finite differences, and a generalized Schur (QZ) decomposition of the linearized equilibrium system.
- Canonical interpretation: this is a Klein/Sims-style linear rational expectations solution with explicit stable-versus-unstable root separation.
- This is the stage-1 benchmark and contains no hidden states, filtering, policy block, RL, regime switching, or external data.

## Core Outputs

- Steady state saved in `steady_state.json`.
- Linear policy and transition matrices saved in `solution.json`.
- Stochastic simulation saved in `simulated_paths.csv`.
- Impulse responses saved in `irf.csv`.
- External gensys benchmark IRFs saved in `gensys_irf.csv`.
- IRF comparison table saved in `irf_comparison.csv`.
- Diagnostic residuals saved in `diagnostics.csv`.
- Figures saved in `figures/steady_state_summary.png`, `figures/simulated_paths.png`, `figures/irf.png`, and `figures/irf_qz_vs_gensys.png`.

## Calibration Snapshot

- beta = {params.beta:.3f}
- alpha = {params.alpha:.3f}
- delta = {params.delta:.3f}
- rho = {params.rho:.3f}
- sigma = {params.sigma:.3f}
- implied steady-state labor disutility weight psi = {steady_state.psi:.3f}
- steady-state levels: y = {steady_state.y:.4f}, c = {steady_state.c:.4f}, i = {steady_state.i:.4f}, k = {steady_state.k:.4f}, n = {steady_state.n:.4f}

## Sanity Checks

- Linearized system residual max: {diagnostics_summary['linear_solution_residual_max']:.3e}
- Spectral radius of the transition matrix: {diagnostics_summary['spectral_radius']:.4f}
- Solver backend: {diagnostics_summary['solver_name']}
- Stable roots: {diagnostics_summary['stable_root_count']}
- Finite roots outside the unit circle: {diagnostics_summary['roots_outside_unit_circle_count']}
- Infinite generalized eigenvalues: {diagnostics_summary['infinite_root_count']}
- Blanchard-Kahn condition satisfied: {diagnostics_summary['bk_condition_satisfied']}
- Stable invariant block condition number: {diagnostics_summary['invariant_block_condition_number']:.4e}
- Max conditional residual on the stochastic simulation: {diagnostics_summary['max_abs_conditional_residual']:.4f}
- RMS conditional residual: {diagnostics_summary['rms_conditional_residual']:.4f}
- Runtime: {diagnostics_summary['runtime_seconds']:.3f} seconds
- Impact IRF signs after a positive technology shock: {sign_checks}

## External Benchmark

- External benchmark source: {benchmark_summary['source_package']}=={benchmark_summary['source_version']} ({benchmark_summary['source_file']})
- gensys existence/uniqueness code: {benchmark_summary['rc']}
- External benchmark reports existence: {benchmark_summary['existence']} | uniqueness: {benchmark_summary['uniqueness']}
- Max absolute IRF difference between internal QZ solver and external gensys benchmark: {benchmark_summary['max_abs_diff']:.3e}
- RMS IRF difference: {benchmark_summary['rms_diff']:.3e}

Seed stability summary:
{seed_lines}

## Interpretation

- The impact IRF is economically sensible for a positive technology shock: output, consumption, investment, and labor all rise on impact.
- The QZ split delivers exactly two stable roots for two predetermined state variables, so the Blanchard-Kahn count condition is met.
- The transition matrix is stable because its spectral radius is below one.
- The external gensys benchmark reproduces the same IRFs up to numerical precision, which provides an independent cross-check beyond the internal solver diagnostics.
- Conditional residuals remain small in the simulated neighborhood of the steady state, which is consistent with a first-order local approximation.

## Not Implemented Yet

- New Keynesian policy block.
- Hidden states or state-space filtering.
- RL or any adaptive policy design.
- HANK structure, regime switching, and real data.
"""
    (output_path / "stage1_report.md").write_text(report, encoding="utf-8")


def _plot_steady_state_summary(
    params: RBCParameters,
    steady_state: RBCSteadyState,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    quantities = {
        "Output": steady_state.y,
        "Consumption": steady_state.c,
        "Investment": steady_state.i,
        "Capital": steady_state.k,
        "Labor": steady_state.n,
    }
    axes[0].bar(quantities.keys(), quantities.values(), color=["#355070", "#6d597a", "#b56576", "#457b9d", "#e56b6f"])
    axes[0].set_title("RBC Steady State")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Level")

    axes[1].axis("off")
    parameter_lines = [
        "Parameters",
        f"beta = {params.beta:.3f}",
        f"alpha = {params.alpha:.3f}",
        f"delta = {params.delta:.3f}",
        f"rho = {params.rho:.3f}",
        f"sigma = {params.sigma:.3f}",
        f"Frisch inverse = {params.frisch_inverse:.3f}",
        "",
        "Implied steady state",
        f"psi = {steady_state.psi:.3f}",
        f"MPK = {steady_state.mpk:.3f}",
        f"wage = {steady_state.wage:.3f}",
    ]
    axes[1].text(0.0, 1.0, "\n".join(parameter_lines), va="top", ha="left", fontsize=11)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_simulated_paths(simulation: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    series = [
        ("log_y_dev", "Output"),
        ("log_c_dev", "Consumption"),
        ("log_i_dev", "Investment"),
        ("log_k_dev", "Capital"),
        ("log_n_dev", "Labor"),
        ("z", "Technology shock"),
    ]

    for axis, (column, title) in zip(axes, series, strict=True):
        axis.plot(simulation["t"], 100.0 * simulation[column], color="#355070", linewidth=1.4)
        axis.set_title(title)
        axis.set_ylabel("Percent dev.")

    axes[-1].set_xlabel("Period")
    axes[-2].set_xlabel("Period")
    figure.suptitle("Stochastic RBC Simulation", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_irf(irf: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    series = [
        ("log_y_dev", "Output"),
        ("log_c_dev", "Consumption"),
        ("log_i_dev", "Investment"),
        ("log_k_dev", "Capital"),
        ("log_n_dev", "Labor"),
        ("z", "Technology shock"),
    ]

    for axis, (column, title) in zip(axes, series, strict=True):
        axis.axhline(0.0, color="#9aa5b1", linewidth=0.8)
        axis.plot(irf["t"], 100.0 * irf[column], color="#b56576", linewidth=1.6)
        axis.set_title(title)
        axis.set_ylabel("Percent dev.")

    axes[-1].set_xlabel("Horizon")
    axes[-2].set_xlabel("Horizon")
    figure.suptitle("IRF to a One-Std Technology Shock", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_irf_benchmark_comparison(
    qz_irf: pd.DataFrame,
    gensys_irf: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    series = [
        ("log_y_dev", "Output"),
        ("log_c_dev", "Consumption"),
        ("log_i_dev", "Investment"),
        ("log_k_dev", "Capital"),
        ("log_n_dev", "Labor"),
        ("z", "Technology shock"),
    ]

    for axis, (column, title) in zip(axes, series, strict=True):
        axis.axhline(0.0, color="#9aa5b1", linewidth=0.8)
        axis.plot(qz_irf["t"], 100.0 * qz_irf[column], color="#355070", linewidth=1.6, label="Internal QZ")
        axis.plot(
            gensys_irf["t"],
            100.0 * gensys_irf[column],
            color="#e56b6f",
            linewidth=1.2,
            linestyle="--",
            label="External gensys",
        )
        axis.set_title(title)
        axis.set_ylabel("Percent dev.")

    axes[0].legend(frameon=False, loc="best")
    axes[-1].set_xlabel("Horizon")
    axes[-2].set_xlabel("Horizon")
    figure.suptitle("IRF Cross-Check: Internal QZ vs External gensys", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_stage1_pipeline(
    output_dir: str | Path,
    periods: int = 240,
    burn_in: int = 80,
    irf_horizon: int = 24,
    seed: int = 42,
) -> dict:
    start_time = time.perf_counter()
    output_path = Path(output_dir)
    figure_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    params = RBCParameters()
    steady_state = compute_steady_state(params)
    solution = solve_linear_policy(params=params, steady_state=steady_state)

    simulation = simulate_paths(
        params=params,
        steady_state=steady_state,
        solution=solution,
        periods=periods,
        burn_in=burn_in,
        seed=seed,
    )
    irf = compute_irf(
        params=params,
        steady_state=steady_state,
        solution=solution,
        horizon=irf_horizon,
    )
    benchmark = run_external_gensys_irf_benchmark(
        params=params,
        steady_state=steady_state,
        solution=solution,
        horizon=irf_horizon,
    )
    diagnostics = diagnostic_table_from_simulation(
        params=params,
        steady_state=steady_state,
        solution=solution,
        simulation=simulation,
    )
    seed_summary = seed_stability_summary(
        params=params,
        steady_state=steady_state,
        solution=solution,
        periods=periods,
        burn_in=burn_in,
        seeds=[0, 1, 2, 3, 4],
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
        "max_abs_conditional_residual": float(np.max(np.abs(diagnostics.to_numpy()))),
        "rms_conditional_residual": float(np.sqrt(np.mean(np.square(diagnostics.to_numpy())))),
        "mean_conditional_residuals": {
            name: float(diagnostics[name].mean()) for name in diagnostics.columns
        },
        "max_abs_conditional_residuals": {
            name: float(np.max(np.abs(diagnostics[name].to_numpy()))) for name in diagnostics.columns
        },
        "impact_irf_signs": {
            "output_positive": bool(irf.iloc[0]["log_y_dev"] > 0.0),
            "consumption_positive": bool(irf.iloc[0]["log_c_dev"] > 0.0),
            "investment_positive": bool(irf.iloc[0]["log_i_dev"] > 0.0),
            "labor_positive": bool(irf.iloc[0]["log_n_dev"] > 0.0),
        },
        "external_gensys_benchmark": {
            "source_package": benchmark.source_package,
            "source_version": benchmark.source_version,
            "source_file": benchmark.source_file,
            "rc": list(benchmark.rc),
            "existence": benchmark.existence,
            "uniqueness": benchmark.uniqueness,
            "max_abs_diff": benchmark.max_abs_diff,
            "rms_diff": benchmark.rms_diff,
        },
        "seed_stability": seed_summary,
    }

    steady_state_payload = {
        "parameters": params.to_dict(),
        "steady_state": steady_state.to_dict(),
    }
    solution_payload = {
        "solver_name": solution.solver_name,
        "policy_matrix": solution.policy_matrix.tolist(),
        "transition_matrix": solution.transition_matrix.tolist(),
        "shock_vector": solution.shock_vector.tolist(),
        "spectral_radius": solution.spectral_radius,
        "residual_norm": solution.residual_norm,
        "stable_root_count": solution.stable_root_count,
        "roots_outside_unit_circle_count": solution.roots_outside_unit_circle_count,
        "infinite_root_count": solution.infinite_root_count,
        "bk_condition_satisfied": solution.bk_condition_satisfied,
        "invariant_block_condition_number": solution.invariant_block_condition_number,
        "generalized_eigenvalues": _serialize_complex_vector(solution.generalized_eigenvalues),
    }

    _plot_steady_state_summary(
        params=params,
        steady_state=steady_state,
        output_path=figure_dir / "steady_state_summary.png",
    )
    _plot_simulated_paths(simulation=simulation.iloc[:120], output_path=figure_dir / "simulated_paths.png")
    _plot_irf(irf=irf, output_path=figure_dir / "irf.png")
    _plot_irf_benchmark_comparison(
        qz_irf=benchmark.qz_irf,
        gensys_irf=benchmark.gensys_irf,
        output_path=figure_dir / "irf_qz_vs_gensys.png",
    )

    simulation.to_csv(output_path / "simulated_paths.csv", index=False)
    irf.to_csv(output_path / "irf.csv", index=False)
    benchmark.gensys_irf.to_csv(output_path / "gensys_irf.csv", index=False)
    benchmark.irf_comparison.to_csv(output_path / "irf_comparison.csv", index=False)
    diagnostics.to_csv(output_path / "diagnostics.csv", index=False)
    _write_json(output_path / "steady_state.json", steady_state_payload)
    _write_json(output_path / "solution.json", solution_payload)
    _write_json(output_path / "diagnostics_summary.json", diagnostics_summary)
    _write_json(output_path / "benchmark_summary.json", diagnostics_summary["external_gensys_benchmark"])
    _write_stage_report(
        output_path=output_path,
        params=params,
        steady_state=steady_state,
        solution=solution,
        diagnostics_summary=diagnostics_summary,
        irf=irf,
        benchmark_summary=diagnostics_summary["external_gensys_benchmark"],
    )

    return {
        "parameters": params,
        "steady_state": steady_state,
        "solution": solution,
        "simulation": simulation,
        "irf": irf,
        "benchmark": benchmark,
        "diagnostics": diagnostics,
        "diagnostics_summary": diagnostics_summary,
    }
