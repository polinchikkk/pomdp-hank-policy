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

from hank_ssj.closed_loop_environment import _rate_to_shock_inversion  # noqa: E402


@dataclass(frozen=True)
class RateToShockInversionAuditSpec:
    jacobians_npz: str
    hank_observables: str
    output_dir: str
    horizons: tuple[int, ...]
    ridge_grid: tuple[float, ...]
    num_random_paths: int
    random_seed: int
    bad_relative_residual_threshold: float
    include_basis_paths: bool
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the rate-path to policy-shock inversion used in closed-loop SSJ.")
    parser.add_argument(
        "--jacobians-npz",
        default="outputs/ssj/stochastic/closed_loop_distributional_ssj/jacobians_distributional_augmented.npz",
    )
    parser.add_argument("--fallback-jacobians-npz", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/rate_to_shock_inversion_audit")
    parser.add_argument("--horizons", default="all", help="Comma-separated horizons, or 'all' for 1..max horizon.")
    parser.add_argument("--ridge-grid", default="1e-12,1e-10,1e-8,1e-6")
    parser.add_argument("--reference-ridge", type=float, default=1e-10)
    parser.add_argument("--num-random-paths", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=3701)
    parser.add_argument("--bad-relative-residual-threshold", type=float, default=1e-3)
    parser.add_argument("--include-basis-paths", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jacobians_npz = _existing_jacobian_path(Path(args.jacobians_npz), Path(args.fallback_jacobians_npz))
    jacobians = _load_jacobians(jacobians_npz)
    max_horizon = int(jacobians["J_monetary_policy_shock_i"].shape[0])
    horizons = _parse_horizons(args.horizons, max_horizon=max_horizon)
    ridge_grid = tuple(float(value) for value in _parse_float_grid(args.ridge_grid))
    observables = pd.read_csv(args.hank_observables)
    paths = _rate_path_catalog(
        observables=observables,
        max_horizon=max(horizons),
        num_random_paths=int(args.num_random_paths),
        seed=int(args.random_seed),
        include_basis_paths=bool(args.include_basis_paths),
    )

    summary_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    operators_by_horizon: dict[int, dict[float, object]] = {}

    for horizon in horizons:
        operators_by_horizon[horizon] = {}
        for ridge in ridge_grid:
            inversion = _rate_to_shock_inversion(jacobians=jacobians, periods=horizon, ridge=ridge)
            operators_by_horizon[horizon][ridge] = inversion
            horizon_paths = paths[paths["horizon"].eq(horizon)]
            residuals = _path_residuals(
                paths=horizon_paths,
                reconstruction_operator=inversion.reconstruction_operator,
                horizon=horizon,
                ridge=ridge,
                condition_number=float(inversion.condition_number),
                bad_relative_residual_threshold=float(args.bad_relative_residual_threshold),
            )
            path_rows.extend(residuals)
            summary_rows.extend(
                _residual_summary(
                    residuals=residuals,
                    rate_response=inversion.rate_response,
                    reconstruction_operator=inversion.reconstruction_operator,
                    horizon=horizon,
                    ridge=ridge,
                    condition_number=float(inversion.condition_number),
                    threshold=float(args.bad_relative_residual_threshold),
                )
            )
        reference_ridge = _reference_ridge(ridge_grid, requested=float(args.reference_ridge))
        reference = operators_by_horizon[horizon][reference_ridge]
        for ridge in ridge_grid:
            current = operators_by_horizon[horizon][ridge]
            sensitivity_rows.append(
                {
                    "horizon": int(horizon),
                    "ridge": float(ridge),
                    "reference_ridge": float(reference_ridge),
                    "condition_number": float(current.condition_number),
                    "shock_from_rate_norm": float(np.linalg.norm(current.shock_from_rate, ord="fro")),
                    "reconstruction_operator_norm": float(np.linalg.norm(current.reconstruction_operator, ord="fro")),
                    "relative_shock_from_rate_gap_to_reference": _relative_gap(
                        current.shock_from_rate,
                        reference.shock_from_rate,
                    ),
                    "relative_reconstruction_gap_to_reference": _relative_gap(
                        current.reconstruction_operator,
                        reference.reconstruction_operator,
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    path_residuals = pd.DataFrame(path_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    summary.to_csv(output_dir / "rate_inversion_horizon_ridge_summary.csv", index=False)
    path_residuals.to_csv(output_dir / "rate_inversion_path_residuals.csv", index=False)
    sensitivity.to_csv(output_dir / "rate_inversion_ridge_sensitivity.csv", index=False)
    _plot_audit(summary, output_dir / "fig_rate_to_shock_inversion_audit.pdf")
    _plot_audit(summary, output_dir / "fig_rate_to_shock_inversion_audit.png")

    spec = RateToShockInversionAuditSpec(
        jacobians_npz=str(jacobians_npz),
        hank_observables=args.hank_observables,
        output_dir=str(output_dir),
        horizons=tuple(int(horizon) for horizon in horizons),
        ridge_grid=tuple(float(ridge) for ridge in ridge_grid),
        num_random_paths=int(args.num_random_paths),
        random_seed=int(args.random_seed),
        bad_relative_residual_threshold=float(args.bad_relative_residual_threshold),
        include_basis_paths=bool(args.include_basis_paths),
        note=(
            "This audit checks the regularized least-squares bridge from closed-loop rate deviations "
            "to monetary-policy shock paths: shock = (J_i'J_i + ridge I)^(-1) J_i' rate_dev."
        ),
    )
    (output_dir / "rate_to_shock_inversion_audit_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        sensitivity=sensitivity,
        threshold=float(args.bad_relative_residual_threshold),
    )
    print(f"Wrote {output_dir / 'rate_inversion_horizon_ridge_summary.csv'}")
    print(f"Wrote {output_dir / 'report_rate_to_shock_inversion_audit.md'}")


def _existing_jacobian_path(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Neither {primary} nor {fallback} exists.")


def _load_jacobians(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as bundle:
        jacobians = {key: np.asarray(bundle[key], dtype=float) for key in bundle.files if key.startswith("J_")}
    if "J_monetary_policy_shock_i" not in jacobians:
        raise ValueError(f"Jacobian archive {path} does not contain J_monetary_policy_shock_i.")
    return jacobians


def _parse_float_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Grid must contain at least one value.")
    if any(value <= 0.0 for value in values):
        raise ValueError("Grid values must be positive.")
    return values


def _parse_horizons(raw: str, *, max_horizon: int) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(1, max_horizon + 1))
    horizons = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not horizons:
        raise ValueError("At least one horizon is required.")
    invalid = [horizon for horizon in horizons if horizon <= 0 or horizon > max_horizon]
    if invalid:
        raise ValueError(f"Horizons must be in 1..{max_horizon}; invalid: {invalid}")
    return tuple(sorted(set(horizons)))


def _rate_path_catalog(
    *,
    observables: pd.DataFrame,
    max_horizon: int,
    num_random_paths: int,
    seed: int,
    include_basis_paths: bool,
) -> pd.DataFrame:
    required = {"scenario", "period", "i"}
    missing = required.difference(observables.columns)
    if missing:
        raise ValueError(f"HANK observables are missing columns: {sorted(missing)}")
    full_paths: list[dict[str, object]] = []
    observed_arrays: list[np.ndarray] = []
    for scenario, frame in observables.groupby("scenario", sort=False):
        values = frame.sort_values("period")["i"].to_numpy(dtype=float)[:max_horizon]
        if values.size < max_horizon:
            continue
        observed_arrays.append(values)
        full_paths.append({"path_source": "observed_hank_rate", "path_id": str(scenario), "values": values})
    observed_stack = np.vstack(observed_arrays) if observed_arrays else np.zeros((1, max_horizon))
    scale = float(np.sqrt(np.mean(observed_stack**2)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1e-3

    rng = np.random.default_rng(seed)
    for index in range(int(num_random_paths)):
        innovations = rng.normal(size=max_horizon)
        values = np.zeros(max_horizon, dtype=float)
        rho = 0.75
        for period in range(max_horizon):
            previous = values[period - 1] if period else 0.0
            values[period] = rho * previous + np.sqrt(1.0 - rho**2) * innovations[period]
        values = values / max(float(np.sqrt(np.mean(values**2))), 1e-14) * scale
        full_paths.append({"path_source": "random_smooth_rate", "path_id": f"random_{index:04d}", "values": values})

    if include_basis_paths:
        for period in range(max_horizon):
            values = np.zeros(max_horizon, dtype=float)
            values[period] = scale * np.sqrt(max_horizon)
            full_paths.append({"path_source": "unit_basis_rate", "path_id": f"basis_{period:03d}", "values": values})

    rows: list[dict[str, object]] = []
    for item in full_paths:
        values = np.asarray(item["values"], dtype=float)
        for horizon in range(1, max_horizon + 1):
            row = {
                "horizon": int(horizon),
                "path_source": item["path_source"],
                "path_id": item["path_id"],
            }
            for period, value in enumerate(values[:horizon]):
                row[f"r_{period}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def _path_matrix(paths: pd.DataFrame, horizon: int) -> np.ndarray:
    columns = [f"r_{period}" for period in range(horizon)]
    return paths.loc[:, columns].to_numpy(dtype=float)


def _path_residuals(
    *,
    paths: pd.DataFrame,
    reconstruction_operator: np.ndarray,
    horizon: int,
    ridge: float,
    condition_number: float,
    bad_relative_residual_threshold: float,
) -> list[dict[str, object]]:
    if paths.empty:
        return []
    values = _path_matrix(paths, horizon)
    reconstructed = values @ reconstruction_operator[:horizon, :horizon].T
    residual = reconstructed - values
    residual_norm = np.linalg.norm(residual, axis=1)
    path_norm = np.linalg.norm(values, axis=1)
    relative = residual_norm / np.maximum(path_norm, 1e-14)
    rows: list[dict[str, object]] = []
    for index, (_, row) in enumerate(paths.iterrows()):
        rows.append(
            {
                "horizon": int(horizon),
                "ridge": float(ridge),
                "condition_number": float(condition_number),
                "path_source": row["path_source"],
                "path_id": row["path_id"],
                "rate_path_norm": float(path_norm[index]),
                "rate_inversion_residual": float(residual_norm[index]),
                "relative_rate_inversion_residual": float(relative[index]),
                "bad_reconstruction": bool(relative[index] > bad_relative_residual_threshold),
            }
        )
    return rows


def _residual_summary(
    *,
    residuals: list[dict[str, object]],
    rate_response: np.ndarray,
    reconstruction_operator: np.ndarray,
    horizon: int,
    ridge: float,
    condition_number: float,
    threshold: float,
) -> list[dict[str, object]]:
    frame = pd.DataFrame(residuals)
    rows: list[dict[str, object]] = []
    identity_gap = reconstruction_operator - np.eye(horizon)
    singular_values = np.linalg.svd(rate_response, compute_uv=False)
    for source, source_frame in _source_groups(frame):
        rows.append(
            {
                "horizon": int(horizon),
                "ridge": float(ridge),
                "path_source": source,
                "num_rate_paths": int(source_frame.shape[0]),
                "condition_number": float(condition_number),
                "min_singular_value": float(np.min(singular_values)),
                "max_singular_value": float(np.max(singular_values)),
                "operator_identity_gap_frobenius": float(np.linalg.norm(identity_gap, ord="fro")),
                "operator_identity_gap_spectral": float(np.linalg.norm(identity_gap, ord=2)),
                "mean_rate_inversion_residual": float(source_frame["rate_inversion_residual"].mean()),
                "max_rate_inversion_residual": float(source_frame["rate_inversion_residual"].max()),
                "mean_relative_rate_inversion_residual": float(source_frame["relative_rate_inversion_residual"].mean()),
                "p95_relative_rate_inversion_residual": float(source_frame["relative_rate_inversion_residual"].quantile(0.95)),
                "max_relative_rate_inversion_residual": float(source_frame["relative_rate_inversion_residual"].max()),
                "bad_reconstruction_share": float(source_frame["bad_reconstruction"].mean()),
                "bad_relative_residual_threshold": float(threshold),
            }
        )
    return rows


def _source_groups(frame: pd.DataFrame):
    if frame.empty:
        return []
    groups = [(source, source_frame) for source, source_frame in frame.groupby("path_source", sort=False)]
    groups.append(("all", frame))
    return groups


def _reference_ridge(ridge_grid: tuple[float, ...], *, requested: float) -> float:
    if requested in ridge_grid:
        return requested
    return min(ridge_grid, key=lambda value: abs(np.log10(value) - np.log10(requested)))


def _relative_gap(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, ord="fro") / max(float(np.linalg.norm(right, ord="fro")), 1e-14))


def _plot_audit(summary: pd.DataFrame, figure_path: Path) -> None:
    if summary.empty:
        return
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = summary[summary["path_source"].eq("all")].copy()
    if all_rows.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for ridge, frame in all_rows.groupby("ridge", sort=True):
        ordered = frame.sort_values("horizon")
        axes[0].plot(ordered["horizon"], ordered["condition_number"], linewidth=1.6, label=f"ridge={ridge:g}")
        axes[1].plot(ordered["horizon"], ordered["bad_reconstruction_share"], linewidth=1.6, label=f"ridge={ridge:g}")
    axes[0].set_yscale("log")
    axes[0].set_title("cond(J_i)")
    axes[0].set_xlabel("Horizon")
    axes[0].set_ylabel("Condition number")
    axes[1].set_title("Bad reconstruction share")
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("Share")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Rate-to-policy-shock inversion audit", fontsize=12)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    threshold: float,
) -> None:
    lines = [
        "# Rate-to-shock inversion audit",
        "",
        "The closed-loop environment maps a rate deviation path into a monetary-policy shock path",
        "through a ridge-regularized inverse of `J_monetary_policy_shock_i`.",
        "",
        f"Bad reconstruction threshold: relative residual > {threshold:g}.",
        "",
    ]
    all_rows = summary[summary["path_source"].eq("all")].copy()
    if not all_rows.empty:
        selected = all_rows[
            all_rows["horizon"].isin(sorted(set(all_rows["horizon"]))[-5:])
        ][
            [
                "horizon",
                "ridge",
                "condition_number",
                "operator_identity_gap_frobenius",
                "mean_relative_rate_inversion_residual",
                "max_relative_rate_inversion_residual",
                "bad_reconstruction_share",
            ]
        ]
        lines.extend(["## Horizon/ridge summary", "", selected.to_markdown(index=False, floatfmt=".4g"), ""])
    worst = summary.sort_values("bad_reconstruction_share", ascending=False).head(10)
    if not worst.empty:
        lines.extend(
            [
                "## Worst reconstruction shares",
                "",
                worst[
                    [
                        "horizon",
                        "ridge",
                        "path_source",
                        "condition_number",
                        "max_relative_rate_inversion_residual",
                        "bad_reconstruction_share",
                    ]
                ].to_markdown(index=False, floatfmt=".4g"),
                "",
            ]
        )
    if not sensitivity.empty:
        tail = sensitivity[sensitivity["horizon"].eq(int(sensitivity["horizon"].max()))]
        lines.extend(
            [
                "## Ridge sensitivity at max horizon",
                "",
                tail[
                    [
                        "horizon",
                        "ridge",
                        "reference_ridge",
                        "relative_shock_from_rate_gap_to_reference",
                        "relative_reconstruction_gap_to_reference",
                    ]
                ].to_markdown(index=False, floatfmt=".4g"),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Low reconstruction residuals mean the closed-loop counterfactual is using a stable numerical bridge",
            "from policy-rate paths to monetary shock paths. Large ridge sensitivity or large bad-path shares",
            "would indicate that closed-loop effects may be dominated by pseudo-inversion artifacts.",
            "",
        ]
    )
    (output_dir / "report_rate_to_shock_inversion_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
