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
    _fit_rule,
    _rule_rows,
)
from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights  # noqa: E402
from policy.optimize_rules import compare_paired_losses  # noqa: E402


SIGNAL_COLUMNS = (
    "mean_mpc_centered",
    "share_low_liquidity_centered",
    "interest_exposure_centered",
)


@dataclass(frozen=True)
class KnownDistributionChannelSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    gammas: tuple[float, ...]
    lag: int
    signal_columns: tuple[str, ...]
    num_candidates: int
    candidate_seed: int
    continuous_methods: tuple[str, ...]
    num_starts: int
    maxiter: int
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Positive control: inject a known distributional channel into future output.")
    parser.add_argument("--information-inputs", default="outputs/ssj/stochastic/state_space/information_inputs/information_state_inputs_long.csv")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/known_distribution_channel")
    parser.add_argument("--figure-path", default="article/figures/fig_known_distribution_channel.pdf")
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--gammas", default="0,0.00025,0.0005,0.001,0.002")
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--candidate-seed", type=int, default=7601)
    parser.add_argument("--continuous-methods", default="L-BFGS-B")
    parser.add_argument("--num-starts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    validation_seeds = _parse_seed_range(args.validation_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)
    gammas = tuple(float(part) for part in args.gammas.split(",") if part.strip())
    continuous_methods = tuple(part.strip() for part in args.continuous_methods.split(",") if part.strip())

    source = pd.read_csv(args.information_inputs)
    observables = pd.read_csv(args.hank_observables)
    with np.load(args.jacobians) as bundle:
        jacobians = {key: np.asarray(bundle[key], dtype=float) for key in bundle.files if key.startswith("J_")}

    controlled_source = _align_distribution_aggregates_to_filtered_aggregates(source)
    channel_signal = _known_channel_signal(observables, lag=int(args.lag))

    summary_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []

    for index, gamma in enumerate(gammas):
        print(f"Known-channel gamma {gamma:g} ({index + 1}/{len(gammas)})", flush=True)
        control_observables = _inject_known_distribution_channel(
            observables=observables,
            signal=channel_signal,
            gamma=float(gamma),
        )
        environment = HankSSJPolicyEnvironment(
            information_inputs=controlled_source,
            observables=control_observables,
            jacobians=jacobians,
            loss_weights=PolicyLossWeights(),
        )
        aggregate_fit = _fit_rule(
            environment=environment,
            information_state="filtered_aggregates",
            validation_seeds=validation_seeds,
            num_candidates=args.num_candidates,
            candidate_seed=int(args.candidate_seed) + index * 10,
            continuous_methods=continuous_methods,
            num_starts=args.num_starts,
            maxiter=args.maxiter,
        )
        distribution_fit = _fit_rule(
            environment=environment,
            information_state="filtered_distribution",
            validation_seeds=validation_seeds,
            num_candidates=args.num_candidates,
            candidate_seed=int(args.candidate_seed) + index * 10 + 1,
            continuous_methods=continuous_methods,
            num_starts=args.num_starts,
            maxiter=args.maxiter,
        )
        rule_rows.extend(_rule_rows("filtered_aggregates", f"gamma_{gamma:g}", aggregate_fit.rule, aggregate_fit.validation_loss))
        rule_rows.extend(_rule_rows("filtered_distribution", f"gamma_{gamma:g}", distribution_fit.rule, distribution_fit.validation_loss))
        losses = _evaluate_pair(
            environment=environment,
            aggregate_rule=aggregate_fit.rule,
            distribution_rule=distribution_fit.rule,
            gamma=float(gamma),
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
        summary_rows.append(
            {
                "gamma": float(gamma),
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
            }
        )
        signal_rows.extend(_signal_diagnostics(control_observables, channel_signal, gamma=float(gamma)))

    summary = pd.DataFrame(summary_rows)
    monotonicity_tolerance = 1e-6
    summary["mvoi_monotone_non_decreasing_so_far"] = _monotone_prefix(
        summary["loss_reduction"].to_numpy(dtype=float),
        tolerance=monotonicity_tolerance,
    )
    losses_all = pd.DataFrame(loss_rows)
    rules = pd.DataFrame(rule_rows)
    signal_diagnostics = pd.DataFrame(signal_rows)
    monotonicity = _monotonicity_summary(summary, tolerance=monotonicity_tolerance)

    summary.to_csv(output_dir / "known_distribution_channel_summary.csv", index=False)
    losses_all.to_csv(output_dir / "known_distribution_channel_trajectory_losses.csv", index=False)
    rules.to_csv(output_dir / "known_distribution_channel_fitted_rules.csv", index=False)
    signal_diagnostics.to_csv(output_dir / "known_distribution_channel_signal_diagnostics.csv", index=False)
    monotonicity.to_csv(output_dir / "known_distribution_channel_monotonicity.csv", index=False)
    _write_latex(summary, output_dir / "table_known_distribution_channel.tex")
    _write_report(summary, monotonicity, output_dir / "report_known_distribution_channel.md")
    _plot(summary, figure_path)

    spec = KnownDistributionChannelSpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        gammas=gammas,
        lag=int(args.lag),
        signal_columns=SIGNAL_COLUMNS,
        num_candidates=int(args.num_candidates),
        candidate_seed=int(args.candidate_seed),
        continuous_methods=continuous_methods,
        num_starts=int(args.num_starts),
        maxiter=int(args.maxiter),
        note=(
            "Positive control: в истинный output gap добавляется известный будущий распределительный "
            "канал gamma * signal_{t-lag}. Информационные состояния и класс правила остаются прежними. "
            "Ожидаемый результат -- MVOI filtered_distribution над filtered_aggregates должен расти по gamma."
        ),
    )
    (output_dir / "known_distribution_channel_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'known_distribution_channel_summary.csv'}")
    print(f"Wrote {output_dir / 'report_known_distribution_channel.md'}")
    print(f"Wrote {figure_path}")


def _known_channel_signal(observables: pd.DataFrame, *, lag: int) -> pd.DataFrame:
    frame = observables[["scenario", "period", *SIGNAL_COLUMNS]].copy()
    signal = np.zeros(len(frame), dtype=float)
    for column in SIGNAL_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        scale = max(float(np.std(values, ddof=0)), 1e-12)
        signal += (values - float(np.mean(values))) / scale
    signal /= np.sqrt(len(SIGNAL_COLUMNS))
    frame["_raw_signal"] = signal
    parts = []
    for _, group in frame.groupby("scenario", sort=False):
        shifted = group["_raw_signal"].shift(lag)
        shifted = shifted.bfill().ffill()
        part = group[["scenario", "period"]].copy()
        part["known_distribution_signal"] = shifted.to_numpy(dtype=float)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _inject_known_distribution_channel(
    *,
    observables: pd.DataFrame,
    signal: pd.DataFrame,
    gamma: float,
) -> pd.DataFrame:
    result = observables.copy()
    signal_key = signal.set_index(["scenario", "period"])["known_distribution_signal"]
    values = [
        float(signal_key.loc[(row.scenario, row.period)])
        for row in result[["scenario", "period"]].itertuples(index=False)
    ]
    addition = float(gamma) * np.asarray(values, dtype=float)
    result["known_distribution_signal"] = values
    result["output_gap_original"] = result["output_gap"].to_numpy(dtype=float)
    result["Y_original"] = result["Y"].to_numpy(dtype=float)
    result["output_gap"] = result["output_gap"].to_numpy(dtype=float) + addition
    result["Y"] = result["Y"].to_numpy(dtype=float) + addition
    return result


def _evaluate_pair(
    *,
    environment: HankSSJPolicyEnvironment,
    aggregate_rule,
    distribution_rule,
    gamma: float,
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
                    "gamma": float(gamma),
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


def _signal_diagnostics(observables: pd.DataFrame, signal: pd.DataFrame, *, gamma: float) -> list[dict[str, object]]:
    merged = observables.merge(signal, on=["scenario", "period"], how="left", suffixes=("", "_source"))
    return [
        {
            "gamma": float(gamma),
            "signal_mean": float(merged["known_distribution_signal"].mean()),
            "signal_std": float(merged["known_distribution_signal"].std(ddof=0)),
            "output_gap_original_std": float(merged["output_gap_original"].std(ddof=0)),
            "output_gap_control_std": float(merged["output_gap"].std(ddof=0)),
            "corr_signal_output_gap_original": _safe_corr(
                merged["known_distribution_signal"].to_numpy(dtype=float),
                merged["output_gap_original"].to_numpy(dtype=float),
            ),
            "corr_signal_output_gap_control": _safe_corr(
                merged["known_distribution_signal"].to_numpy(dtype=float),
                merged["output_gap"].to_numpy(dtype=float),
            ),
        }
    ]


def _monotone_prefix(values: np.ndarray, *, tolerance: float) -> list[bool]:
    result: list[bool] = []
    current_max = -np.inf
    ok = True
    for value in values:
        if value + float(tolerance) < current_max:
            ok = False
        current_max = max(current_max, float(value))
        result.append(bool(ok))
    return result


def _monotonicity_summary(summary: pd.DataFrame, *, tolerance: float) -> pd.DataFrame:
    values = summary.sort_values("gamma")["loss_reduction"].to_numpy(dtype=float)
    diffs = np.diff(values)
    return pd.DataFrame(
        [
            {
                "num_gamma_values": int(len(values)),
                "monotone_non_decreasing": bool(np.all(diffs >= -float(tolerance))),
                "monotonicity_tolerance": float(tolerance),
                "min_increment": float(np.min(diffs)) if len(diffs) else float("nan"),
                "max_increment": float(np.max(diffs)) if len(diffs) else float("nan"),
                "mvoi_at_min_gamma": float(values[0]) if len(values) else float("nan"),
                "mvoi_at_max_gamma": float(values[-1]) if len(values) else float("nan"),
            }
        ]
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) <= 1e-14 or np.std(right[mask]) <= 1e-14:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _write_latex(summary: pd.DataFrame, path: Path) -> None:
    display = summary[
        [
            "gamma",
            "loss_reduction",
            "ci_low",
            "ci_high",
            "win_rate",
            "sign_flip_p_value",
            "num_trajectories",
        ]
    ].rename(
        columns={
            "gamma": "Gamma",
            "loss_reduction": "MVOI",
            "ci_low": "Нижняя граница",
            "ci_high": "Верхняя граница",
            "win_rate": "Доля выигрышей",
            "sign_flip_p_value": "p-value",
            "num_trajectories": "Число траекторий",
        }
    )
    path.write_text(display.to_latex(index=False, float_format="%.6g", escape=False), encoding="utf-8")


def _write_report(summary: pd.DataFrame, monotonicity: pd.DataFrame, path: Path) -> None:
    mono = monotonicity.iloc[0]
    lines = [
        "# Positive control: известный распределительный канал",
        "",
        "В этом контроле в истинный разрыв выпуска добавляется известный распределительный канал.",
        "Проверка показывает, растёт ли оцененный MVOI при усилении заданного канала.",
        "",
        summary[["gamma", "loss_reduction", "ci_low", "ci_high", "win_rate", "sign_flip_p_value"]].to_markdown(
            index=False,
            floatfmt=".6g",
        ),
        "",
        f"Монотонный рост MVOI с допуском {mono['monotonicity_tolerance']:.1e}: {bool(mono['monotone_non_decreasing'])}.",
        f"MVOI при минимальном gamma: {mono['mvoi_at_min_gamma']:.6g}.",
        f"MVOI при максимальном gamma: {mono['mvoi_at_max_gamma']:.6g}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(summary: pd.DataFrame, figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = summary.sort_values("gamma")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(frame["gamma"], frame["loss_reduction"], marker="o", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Known distributional channel strength")
    ax.set_ylabel("MVOI: loss reduction vs filtered aggregates")
    ax.set_title("Positive control: recovery of a known distributional channel")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
