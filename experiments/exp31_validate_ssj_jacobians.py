from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank.calibration import default_calibration  # noqa: E402
from hank.distribution import household_path_levels, path_distribution_statistics, stationary_distribution  # noqa: E402
from hank.grids import state_mesh  # noqa: E402
from hank.household_solver import compute_mpc, compute_mpc_path  # noqa: E402
from hank.steady_state import solve_steady_state  # noqa: E402
from hank.transition import solve_transition  # noqa: E402


SHOCK_INPUT_MAP = {
    "monetary_policy_shock": "monetary_policy_shock",
    "income_risk_shock": "sigma_z",
    "liquid_wedge_shock": "omega",
    "aggregate_demand_shock": "rstar",
    "aggregate_supply_shock": "Z",
}

DEFAULT_SHOCK_SIZES = {
    "monetary_policy_shock": 0.001,
    "income_risk_shock": 0.001,
    "liquid_wedge_shock": 0.001,
    "aggregate_demand_shock": 0.001,
    "aggregate_supply_shock": 0.001,
}

VALIDATION_VARIABLES = (
    "pi",
    "Y",
    "output_gap",
    "C",
    "i",
    "mean_mpc_centered",
    "share_low_liquidity_centered",
    "interest_exposure_centered",
)

DISTRIBUTIONAL_COLUMNS = {
    "mean_mpc_centered": "mean_mpc",
    "share_low_liquidity_centered": "share_low_liquidity",
    "interest_exposure_centered": "interest_exposure",
}


@dataclass(frozen=True)
class JacobianValidationSpec:
    output_dir: str
    figure_path: str
    jacobians_npz: str
    horizon: int
    shocks: tuple[str, ...]
    amplitudes: tuple[float, ...]
    base_shock_sizes: dict[str, float]
    variables: tuple[str, ...]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-difference validation of local HANK/SSJ Jacobian responses.")
    parser.add_argument(
        "--jacobians-npz",
        default="outputs/ssj/stochastic/closed_loop_distributional_ssj/jacobians_distributional_augmented.npz",
        help="SSJ matrix archive. Monetary-policy shock uses exported matrices when available.",
    )
    parser.add_argument("--output-dir", default="outputs/ssj/jacobian_validation")
    parser.add_argument("--figure-path", default="article/figures/fig_jacobian_validation.pdf")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--amplitudes", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--shocks", default=",".join(SHOCK_INPUT_MAP))
    parser.add_argument("--tiny-response-tol", type=float, default=1e-10)
    parser.add_argument("--relative-error-threshold", type=float, default=0.20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    shocks = tuple(part.strip() for part in args.shocks.split(",") if part.strip())
    unknown = sorted(set(shocks) - set(SHOCK_INPUT_MAP))
    if unknown:
        raise ValueError(f"Unknown shock labels: {unknown}. Available: {sorted(SHOCK_INPUT_MAP)}")
    amplitudes = tuple(float(part) for part in args.amplitudes.split(",") if part.strip())

    config = default_calibration()
    horizon = min(int(args.horizon), int(config.shock_T))
    with contextlib.redirect_stdout(io.StringIO()):
        bundle = solve_steady_state(config)
    steady_distribution = _steady_distributional_values(bundle["ss"], config)
    jacobian_archive = _load_jacobians(Path(args.jacobians_npz))

    rows: list[dict[str, object]] = []
    response_rows: list[dict[str, object]] = []
    reference_responses: dict[str, np.ndarray] = {}

    for shock_label in shocks:
        input_name = SHOCK_INPUT_MAP[shock_label]
        base_size = float(DEFAULT_SHOCK_SIZES[shock_label])
        nonlinear_by_amplitude: dict[float, dict[str, np.ndarray]] = {}
        for amplitude in amplitudes:
            shock_size = base_size * float(amplitude)
            try:
                nonlinear_by_amplitude[float(amplitude)] = _nonlinear_transition_response(
                    bundle=bundle,
                    config=config,
                    steady_distribution=steady_distribution,
                    input_name=input_name,
                    shock_size=shock_size,
                    horizon=horizon,
                )
            except Exception as exc:
                rows.extend(
                    _failed_rows(
                        shock_label=shock_label,
                        input_name=input_name,
                        amplitude=float(amplitude),
                        shock_size=shock_size,
                        horizon=horizon,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )

        if shock_label == "monetary_policy_shock":
            linear_source = "exported_ssj_jacobian"
            reference = _exported_linear_response(
                jacobian_archive=jacobian_archive,
                shock_label=shock_label,
                horizon=horizon,
            )
        else:
            linear_source = "finite_difference_unit_response"
            if 1.0 not in nonlinear_by_amplitude:
                continue
            reference = {
                variable: values / base_size
                for variable, values in nonlinear_by_amplitude[1.0].items()
            }
        for variable, values in reference.items():
            reference_responses[f"{shock_label}__{variable}"] = values

        for amplitude, nonlinear_response in nonlinear_by_amplitude.items():
            shock_size = base_size * float(amplitude)
            for variable in VALIDATION_VARIABLES:
                if variable not in nonlinear_response or variable not in reference:
                    rows.append(
                        _metric_row(
                            shock_label=shock_label,
                            input_name=input_name,
                            amplitude=amplitude,
                            shock_size=shock_size,
                            variable=variable,
                            linear_source=linear_source,
                            status="missing_linear_response",
                            horizon=horizon,
                            error_message="No linear response for this variable.",
                        )
                    )
                    continue
                nonlinear = nonlinear_response[variable]
                linear = reference[variable] * shock_size
                row = _metric_row(
                    shock_label=shock_label,
                    input_name=input_name,
                    amplitude=amplitude,
                    shock_size=shock_size,
                    variable=variable,
                    linear_source=linear_source,
                    status="ok",
                    horizon=horizon,
                    nonlinear=nonlinear,
                    linear=linear,
                    tiny_response_tol=float(args.tiny_response_tol),
                    relative_error_threshold=float(args.relative_error_threshold),
                )
                rows.append(row)
                response_rows.extend(_response_long_rows(shock_label, input_name, amplitude, shock_size, variable, nonlinear, linear))

    summary = pd.DataFrame(rows)
    by_shock = _aggregate_summary(summary)
    response_long = pd.DataFrame(response_rows)

    summary.to_csv(output_dir / "jacobian_validation_summary.csv", index=False)
    by_shock.to_csv(output_dir / "jacobian_validation_by_shock.csv", index=False)
    response_long.to_csv(output_dir / "jacobian_validation_responses_long.csv", index=False)
    if reference_responses:
        np.savez_compressed(output_dir / "finite_difference_reference_responses.npz", **reference_responses)

    spec = JacobianValidationSpec(
        output_dir=str(output_dir),
        figure_path=str(figure_path),
        jacobians_npz=str(args.jacobians_npz),
        horizon=horizon,
        shocks=shocks,
        amplitudes=amplitudes,
        base_shock_sizes={shock: float(DEFAULT_SHOCK_SIZES[shock]) for shock in shocks},
        variables=VALIDATION_VARIABLES,
        note=(
            "Для денежного шока линейный отклик берётся из экспортированного SSJ-якобиана. "
            "Для остальных шоков текущий архив не содержит готовых SSJ-матриц, поэтому проверяется "
            "локальная линейность HANK transition solver относительно конечной разностной реакции при "
            "базовой амплитуде. Это верифицирует область применимости локальной аппроксимации, но не "
            "является отдельным экспортом всех SSJ-матриц по неполитическим шокам."
        ),
    )
    (output_dir / "jacobian_validation_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(summary, by_shock, output_dir / "report_jacobian_validation.md")
    _plot_validation(by_shock, figure_path)
    print(f"Wrote {output_dir / 'jacobian_validation_summary.csv'}")
    print(f"Wrote {output_dir / 'jacobian_validation_by_shock.csv'}")
    print(f"Wrote {figure_path}")


def _load_jacobians(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files if key.startswith("J_")}


def _steady_distributional_values(ss, config) -> dict[str, float]:
    distribution = stationary_distribution(ss)
    mpc = compute_mpc(ss)
    mesh = state_mesh(ss)
    return {
        "mean_mpc": float(np.sum(distribution * mpc)),
        "share_low_liquidity": float(np.sum(distribution * (mesh["b"] <= config.low_liquidity_threshold))),
        "interest_exposure": float(np.sum(distribution * mesh["b"] * mpc)),
    }


def _nonlinear_transition_response(
    *,
    bundle,
    config,
    steady_distribution: dict[str, float],
    input_name: str,
    shock_size: float,
    horizon: int,
) -> dict[str, np.ndarray]:
    shock_path = np.zeros(int(config.shock_T), dtype=float)
    shock_path[0] = float(shock_size)
    with contextlib.redirect_stdout(io.StringIO()):
        transition = solve_transition(bundle, {input_name: shock_path})
    response: dict[str, np.ndarray] = {}
    for variable in ("pi", "Y", "output_gap", "C", "i"):
        response[variable] = np.asarray(transition[variable], dtype=float)[:horizon]

    full_path_levels = household_path_levels(bundle["ss"], transition)
    mpc_path = compute_mpc_path(full_path_levels)
    distribution = path_distribution_statistics(
        bundle["ss"],
        full_path_levels,
        config,
        mpc_path=mpc_path,
    ).sort_values("period")
    for variable, source_column in DISTRIBUTIONAL_COLUMNS.items():
        response[variable] = distribution[source_column].to_numpy(dtype=float)[:horizon] - steady_distribution[source_column]
    return response


def _exported_linear_response(
    *,
    jacobian_archive: dict[str, np.ndarray],
    shock_label: str,
    horizon: int,
) -> dict[str, np.ndarray]:
    response: dict[str, np.ndarray] = {}
    for variable in VALIDATION_VARIABLES:
        key = f"J_{shock_label}_{variable}"
        if key not in jacobian_archive:
            continue
        matrix = np.asarray(jacobian_archive[key], dtype=float)
        response[variable] = matrix[:horizon, 0]
    return response


def _metric_row(
    *,
    shock_label: str,
    input_name: str,
    amplitude: float,
    shock_size: float,
    variable: str,
    linear_source: str,
    status: str,
    horizon: int,
    nonlinear: np.ndarray | None = None,
    linear: np.ndarray | None = None,
    tiny_response_tol: float = 1e-10,
    relative_error_threshold: float = 0.20,
    error_message: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "shock": shock_label,
        "model_input": input_name,
        "amplitude_multiplier": float(amplitude),
        "shock_size": float(shock_size),
        "variable": variable,
        "linear_source": linear_source,
        "status": status,
        "horizon": int(horizon),
        "error_message": error_message,
    }
    if nonlinear is None or linear is None:
        row.update(
            {
                "relative_error": np.nan,
                "rank_correlation": np.nan,
                "max_abs_error": np.nan,
                "rmse_error": np.nan,
                "max_abs_nonlinear": np.nan,
                "max_abs_linear": np.nan,
                "nonlinear_norm": np.nan,
                "linear_norm": np.nan,
                "tiny_response": np.nan,
                "passes_relative_error_threshold": False,
            }
        )
        return row
    error = nonlinear - linear
    nonlinear_norm = float(np.linalg.norm(nonlinear))
    linear_norm = float(np.linalg.norm(linear))
    denominator = max(nonlinear_norm, float(tiny_response_tol))
    max_abs_nonlinear = float(np.max(np.abs(nonlinear)))
    relative_error = float(np.linalg.norm(error) / denominator)
    tiny_response = bool(max_abs_nonlinear < tiny_response_tol)
    row.update(
        {
            "relative_error": relative_error,
            "rank_correlation": _rank_correlation(nonlinear, linear),
            "max_abs_error": float(np.max(np.abs(error))),
            "rmse_error": float(np.sqrt(np.mean(error**2))),
            "max_abs_nonlinear": max_abs_nonlinear,
            "max_abs_linear": float(np.max(np.abs(linear))),
            "nonlinear_norm": nonlinear_norm,
            "linear_norm": linear_norm,
            "tiny_response": tiny_response,
            "passes_relative_error_threshold": bool((relative_error <= relative_error_threshold) or tiny_response),
        }
    )
    return row


def _failed_rows(
    *,
    shock_label: str,
    input_name: str,
    amplitude: float,
    shock_size: float,
    horizon: int,
    error_message: str,
) -> list[dict[str, object]]:
    return [
        _metric_row(
            shock_label=shock_label,
            input_name=input_name,
            amplitude=amplitude,
            shock_size=shock_size,
            variable=variable,
            linear_source="not_computed",
            status="transition_failed",
            horizon=horizon,
            error_message=error_message,
        )
        for variable in VALIDATION_VARIABLES
    ]


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy(dtype=float)
    if np.std(left_rank) <= 1e-14 or np.std(right_rank) <= 1e-14:
        return np.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _response_long_rows(
    shock_label: str,
    input_name: str,
    amplitude: float,
    shock_size: float,
    variable: str,
    nonlinear: np.ndarray,
    linear: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "shock": shock_label,
            "model_input": input_name,
            "amplitude_multiplier": float(amplitude),
            "shock_size": float(shock_size),
            "variable": variable,
            "period": int(period),
            "nonlinear_transition_response": float(nonlinear[period]),
            "ssj_linear_response": float(linear[period]),
            "error": float(nonlinear[period] - linear[period]),
        }
        for period in range(len(nonlinear))
    ]


def _aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary[summary["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (shock, amplitude, linear_source), frame in ok.groupby(["shock", "amplitude_multiplier", "linear_source"], sort=False):
        non_tiny = frame[~frame["tiny_response"].fillna(False)]
        metric_frame = non_tiny if len(non_tiny) else frame
        rows.append(
            {
                "shock": shock,
                "amplitude_multiplier": float(amplitude),
                "linear_source": linear_source,
                "num_variables": int(len(frame)),
                "num_non_tiny_variables": int(len(non_tiny)),
                "mean_relative_error": float(metric_frame["relative_error"].mean()),
                "median_relative_error": float(metric_frame["relative_error"].median()),
                "max_relative_error": float(metric_frame["relative_error"].max()),
                "median_rank_correlation": float(metric_frame["rank_correlation"].median(skipna=True)),
                "max_abs_error": float(metric_frame["max_abs_error"].max()),
                "mean_rmse_error": float(metric_frame["rmse_error"].mean()),
                "pass_share": float(frame["passes_relative_error_threshold"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_validation(by_shock: pd.DataFrame, figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    if by_shock.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    shocks = list(dict.fromkeys(by_shock["shock"].tolist()))
    for shock in shocks:
        frame = by_shock[by_shock["shock"].eq(shock)].sort_values("amplitude_multiplier")
        axes[0].plot(frame["amplitude_multiplier"], frame["mean_relative_error"], marker="o", linewidth=1.8, label=shock)
        axes[1].plot(frame["amplitude_multiplier"], frame["pass_share"], marker="o", linewidth=1.8, label=shock)
    axes[0].axhline(0.20, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    axes[0].set_title("Mean relative error")
    axes[0].set_xlabel("Shock amplitude, relative to baseline")
    axes[0].set_ylabel("Relative error")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Share passing 20% threshold")
    axes[1].set_xlabel("Shock amplitude, relative to baseline")
    axes[1].set_ylabel("Pass share")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Finite-difference validation of local HANK/SSJ responses", fontsize=12)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: pd.DataFrame, by_shock: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Проверка локальной линейной аппроксимации HANK/SSJ",
        "",
        "Таблица сравнивает нелинейный переходный отклик HANK и локальный линейный отклик.",
        "Для денежного шока используется экспортированный SSJ-якобиан. Для остальных шоков используется",
        "конечная разностная реакция при базовой амплитуде как локальный линейный ориентир.",
        "",
    ]
    if not by_shock.empty:
        display = by_shock[
            [
                "shock",
                "amplitude_multiplier",
                "linear_source",
                "num_non_tiny_variables",
                "mean_relative_error",
                "median_relative_error",
                "pass_share",
            ]
        ].copy()
        lines.append(display.to_markdown(index=False, floatfmt=".4g"))
        lines.append("")
    failures = summary[~summary["status"].eq("ok")]
    if not failures.empty:
        lines.extend(
            [
                "## Ошибки или отсутствующие матрицы",
                "",
                failures[["shock", "amplitude_multiplier", "variable", "status", "error_message"]]
                .drop_duplicates()
                .to_markdown(index=False),
                "",
            ]
        )
    lines.extend(
        [
            "## Интерпретация",
            "",
            "Если ошибка растёт при амплитуде 2x, это не опровергает HANK/SSJ-подход, но сужает",
            "область интерпретации: результаты следует читать как локальные около стационарного состояния.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
