from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.archive.exp08_main_voi import _supervised_candidates  # noqa: E402
from experiments.archive.exp23_distributional_identification_battery import (  # noqa: E402
    AGGREGATE_FEATURES,
    DISTRIBUTIONAL_FEATURES,
    _align_distribution_aggregates_to_filtered_aggregates,
    _fit_rule,
    _residualized_values,
    _replace_distribution_features,
)
from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights  # noqa: E402
from policy.fit_linear_rules import fit_linear_rule  # noqa: E402
from policy.linear_rules import LinearRule, coefficient_vector  # noqa: E402
from policy.optimize_linear_rules import (  # noqa: E402
    LinearRuleOptimizationBounds,
    fit_linear_rule_continuous,
    fitted_rule_as_continuous_like,
)
from policy.optimize_rules import compare_paired_losses  # noqa: E402


FEATURES = {
    "mpc": "E_mean_mpc",
    "liquidity": "E_low_liquidity_share",
    "exposure": "E_interest_exposure",
}

FEATURE_LABEL_RU = {
    "mpc": "MPC",
    "liquidity": "Доля низколиквидных",
    "exposure": "Процентная экспозиция",
    "all_distributional": "Все распределительные признаки",
}

COMPONENTS = (
    ("total_loss", "Итого"),
    ("inflation_loss", "Инфляция"),
    ("output_gap_loss", "Разрыв выпуска"),
    ("consumption_loss", "Потребление"),
    ("rate_smoothing_loss", "Сглаживание ставки"),
    ("stability_penalty", "Штраф устойчивости"),
)


@dataclass(frozen=True)
class FeatureDecompositionSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    features: tuple[str, ...]
    num_candidates: int
    candidate_seed: int
    continuous_methods: tuple[str, ...]
    num_starts: int
    maxiter: int
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose distributional value by individual distributional features.")
    parser.add_argument("--information-inputs", default="outputs/ssj/stochastic/state_space/information_inputs/information_state_inputs_long.csv")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/feature_decomposition")
    parser.add_argument("--figure-path", default="article/figures/fig_distributional_feature_decomposition.pdf")
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--num-candidates", type=int, default=120)
    parser.add_argument("--candidate-seed", type=int, default=7801)
    parser.add_argument("--continuous-methods", default="L-BFGS-B")
    parser.add_argument("--num-starts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    validation_seeds = _parse_seed_range(args.validation_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)
    continuous_methods = tuple(part.strip() for part in args.continuous_methods.split(",") if part.strip())

    source = pd.read_csv(args.information_inputs)
    observables = pd.read_csv(args.hank_observables)
    with np.load(args.jacobians) as bundle:
        jacobians = {key: np.asarray(bundle[key], dtype=float) for key in bundle.files if key.startswith("J_")}

    controlled_source = _align_distribution_aggregates_to_filtered_aggregates(source)
    residualized_source = _replace_distribution_features(
        controlled_source,
        _residualized_values(controlled_source, validation_seeds=validation_seeds),
    )

    coalition_specs = _coalition_specs()
    inputs = _build_coalition_inputs(controlled_source, coalition_specs, residualized=False)
    residualized_inputs = _build_coalition_inputs(residualized_source, coalition_specs, residualized=True)
    combined_inputs = pd.concat([inputs, residualized_inputs], ignore_index=True)

    environment = HankSSJPolicyEnvironment(
        information_inputs=combined_inputs,
        observables=observables,
        jacobians=jacobians,
        loss_weights=PolicyLossWeights(),
    )

    fit_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    fitted: dict[tuple[str, str], LinearRule] = {}
    for index, spec in enumerate(coalition_specs):
        print(f"Fitting coalition {index + 1}/{len(coalition_specs)}: {spec['state']}", flush=True)
        fit = _fit_dynamic_rule(
            environment=environment,
            information_state=spec["state"],
            validation_seeds=validation_seeds,
            num_candidates=args.num_candidates,
            candidate_seed=int(args.candidate_seed) + index,
            continuous_methods=continuous_methods,
            num_starts=args.num_starts,
            maxiter=args.maxiter,
        )
        fitted[("raw", spec["coalition"])] = fit.rule
        fit_rows.extend(_rule_rows(spec["coalition"], spec["state"], "raw", fit.rule, fit.validation_loss))
        loss_rows.extend(
            _evaluate_state(
                environment=environment,
                rule=fit.rule,
                information_state=spec["state"],
                coalition=spec["coalition"],
                feature_set=spec["feature_set"],
                feature_set_ru=spec["feature_set_ru"],
                variant="raw",
                test_seeds=test_seeds,
            ).to_dict(orient="records")
        )

        residualized_state = spec["residualized_state"]
        fit_residualized = _fit_dynamic_rule(
            environment=environment,
            information_state=residualized_state,
            validation_seeds=validation_seeds,
            num_candidates=args.num_candidates,
            candidate_seed=int(args.candidate_seed) + 10_000 + index,
            continuous_methods=continuous_methods,
            num_starts=args.num_starts,
            maxiter=args.maxiter,
        )
        fitted[("residualized", spec["coalition"])] = fit_residualized.rule
        fit_rows.extend(
            _rule_rows(spec["coalition"], residualized_state, "residualized", fit_residualized.rule, fit_residualized.validation_loss)
        )
        loss_rows.extend(
            _evaluate_state(
                environment=environment,
                rule=fit_residualized.rule,
                information_state=residualized_state,
                coalition=spec["coalition"],
                feature_set=spec["feature_set"],
                feature_set_ru=spec["feature_set_ru"],
                variant="residualized",
                test_seeds=test_seeds,
            ).to_dict(orient="records")
        )

    losses = pd.DataFrame(loss_rows)
    fits = pd.DataFrame(fit_rows)
    coalition_mvoi = _coalition_mvoi(losses)
    feature_mvoi = _feature_mvoi(coalition_mvoi)
    component_decomposition = _component_decomposition(losses)
    shapley = _shapley(coalition_mvoi, variant="raw")
    residualized_shapley = _shapley(coalition_mvoi, variant="residualized")
    shapley_all = pd.concat([shapley, residualized_shapley], ignore_index=True)

    losses.to_csv(output_dir / "feature_decomposition_trajectory_losses.csv", index=False)
    fits.to_csv(output_dir / "feature_decomposition_fitted_rules.csv", index=False)
    coalition_mvoi.to_csv(output_dir / "coalition_mvoi.csv", index=False)
    feature_mvoi.to_csv(output_dir / "feature_mvoi.csv", index=False)
    component_decomposition.to_csv(output_dir / "feature_loss_component_decomposition.csv", index=False)
    shapley_all.to_csv(output_dir / "feature_shapley_values.csv", index=False)
    _write_latex(feature_mvoi, output_dir / "table_feature_mvoi.tex")
    _write_latex(shapley_all, output_dir / "table_feature_shapley_values.tex")
    _write_report(feature_mvoi, component_decomposition, shapley_all, output_dir / "report_feature_decomposition.md")
    _plot(feature_mvoi, shapley_all, figure_path)

    spec = FeatureDecompositionSpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        features=tuple(FEATURES),
        num_candidates=int(args.num_candidates),
        candidate_seed=int(args.candidate_seed),
        continuous_methods=continuous_methods,
        num_starts=int(args.num_starts),
        maxiter=int(args.maxiter),
        note=(
            "Разложение предельной ценности распределительной информации по признакам: one-feature-at-a-time, "
            "leave-one-feature-out, Shapley по коалициям и residualized Shapley после очистки распределительных "
            "признаков от фильтрованных агрегатов."
        ),
    )
    (output_dir / "feature_decomposition_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'feature_mvoi.csv'}")
    print(f"Wrote {output_dir / 'feature_shapley_values.csv'}")
    print(f"Wrote {figure_path}")


def _coalition_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "coalition": "none",
            "state": "filtered_aggregates",
            "residualized_state": "filtered_aggregates",
            "feature_set": tuple(),
            "feature_set_ru": "Фильтрованные агрегаты",
        }
    ]
    feature_names = tuple(FEATURES)
    for size in range(1, len(feature_names) + 1):
        for combo in itertools.combinations(feature_names, size):
            coalition = "_".join(combo)
            specs.append(
                {
                    "coalition": coalition,
                    "state": f"feature_decomposition_{coalition}",
                    "residualized_state": f"feature_decomposition_residualized_{coalition}",
                    "feature_set": tuple(combo),
                    "feature_set_ru": _feature_set_label(combo),
                }
            )
    return specs


def _build_coalition_inputs(source: pd.DataFrame, coalition_specs: list[dict[str, object]], *, residualized: bool) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    aggregate = source[source["information_state"].eq("filtered_aggregates")].copy()
    if residualized:
        aggregate = aggregate.copy()
    for spec in coalition_specs:
        if spec["coalition"] == "none":
            frame = aggregate.copy()
            frame["information_state"] = "filtered_aggregates"
            rows.append(frame)
            continue
        state_name = spec["residualized_state"] if residualized else spec["state"]
        frame = aggregate.copy()
        frame["information_state"] = state_name
        extra = source[
            source["information_state"].eq("filtered_distribution")
            & source["feature_name"].isin([FEATURES[name] for name in spec["feature_set"]])
        ].copy()
        extra["information_state"] = state_name
        rows.append(pd.concat([frame, extra], ignore_index=True))
    return pd.concat(rows, ignore_index=True)


def _fit_dynamic_rule(
    *,
    environment: HankSSJPolicyEnvironment,
    information_state: str,
    validation_seeds: list[int],
    num_candidates: int,
    candidate_seed: int,
    continuous_methods: tuple[str, ...],
    num_starts: int,
    maxiter: int,
):
    extra_candidates = _supervised_candidates(
        environment=environment,
        information_state=information_state,
        validation_seeds=validation_seeds,
    )
    grid_fit = fit_linear_rule(
        environment=environment,
        information_state=information_state,
        validation_seeds=validation_seeds,
        num_candidates=num_candidates,
        seed=candidate_seed,
        extra_candidates=extra_candidates,
    )
    grid_like = fitted_rule_as_continuous_like(fit=grid_fit, seed=candidate_seed, mode="grid_random")
    return fit_linear_rule_continuous(
        environment=environment,
        information_state=information_state,
        validation_seeds=validation_seeds,
        feature_scales=grid_like.feature_scales,
        initial_rules=[grid_like.rule, *extra_candidates],
        seed=candidate_seed + 20_000,
        num_starts=num_starts,
        methods=continuous_methods,
        bounds=LinearRuleOptimizationBounds(),
        maxiter=maxiter,
    )


def _evaluate_state(
    *,
    environment: HankSSJPolicyEnvironment,
    rule: LinearRule,
    information_state: str,
    coalition: str,
    feature_set: tuple[str, ...],
    feature_set_ru: str,
    variant: str,
    test_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in environment.scenarios:
        for seed in test_seeds:
            loss = environment.simulate_scenario(
                policy=rule,
                information_state=information_state,
                scenario=scenario,
                seed=seed,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "observation_seed": int(seed),
                    "variant": variant,
                    "coalition": coalition,
                    "feature_set": ",".join(feature_set),
                    "feature_set_ru": feature_set_ru,
                    "information_state": information_state,
                    **asdict(loss),
                }
            )
    return pd.DataFrame(rows)


def _coalition_mvoi(losses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, variant_frame in losses.groupby("variant", sort=False):
        baseline = variant_frame[variant_frame["coalition"].eq("none")]
        for coalition, frame in variant_frame.groupby("coalition", sort=False):
            merged = frame.merge(
                baseline[["scenario", "observation_seed", *[component for component, _ in COMPONENTS]]],
                on=["scenario", "observation_seed"],
                suffixes=("", "_baseline"),
                how="inner",
                validate="one_to_one",
            )
            comparison = compare_paired_losses(
                left_name=str(coalition),
                right_name="filtered_aggregates",
                left_losses=merged["total_loss"].to_numpy(dtype=float),
                right_losses=merged["total_loss_baseline"].to_numpy(dtype=float),
                tie_eps=1e-10,
            )
            row = {
                "variant": variant,
                "coalition": coalition,
                "feature_set": str(frame["feature_set"].iloc[0]),
                "feature_set_ru": str(frame["feature_set_ru"].iloc[0]),
                "num_features": 0 if coalition == "none" else len(str(frame["feature_set"].iloc[0]).split(",")),
                "num_trajectories": comparison.num_trajectories,
                "mean_delta": comparison.mean_delta,
                "loss_reduction": -comparison.mean_delta,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "win_rate": comparison.win_rate,
                "tie_rate": comparison.tie_rate,
                "loss_rate": comparison.loss_rate,
            }
            for component, _ in COMPONENTS:
                row[f"{component}_reduction"] = float(
                    merged[f"{component}_baseline"].mean() - merged[component].mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_mvoi(coalition_mvoi: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    raw = coalition_mvoi[coalition_mvoi["variant"].eq("raw")].set_index("coalition")
    full_key = "mpc_liquidity_exposure"
    full_mvoi = float(raw.loc[full_key, "loss_reduction"])
    baseline = float(raw.loc["none", "loss_reduction"])
    del baseline
    for feature in FEATURES:
        one = feature
        without = "_".join(name for name in FEATURES if name != feature)
        rows.append(
            {
                "feature": feature,
                "feature_ru": FEATURE_LABEL_RU[feature],
                "method": "one_feature_at_a_time",
                "method_ru": "Один признак",
                "mvoi": float(raw.loc[one, "loss_reduction"]),
                "share_of_full_mvoi": float(raw.loc[one, "loss_reduction"] / full_mvoi) if abs(full_mvoi) > 1e-14 else np.nan,
            }
        )
        rows.append(
            {
                "feature": feature,
                "feature_ru": FEATURE_LABEL_RU[feature],
                "method": "leave_one_feature_out",
                "method_ru": "Исключение признака",
                "mvoi": float(full_mvoi - raw.loc[without, "loss_reduction"]),
                "share_of_full_mvoi": float((full_mvoi - raw.loc[without, "loss_reduction"]) / full_mvoi) if abs(full_mvoi) > 1e-14 else np.nan,
            }
        )
    rows.append(
        {
            "feature": "all_distributional",
            "feature_ru": FEATURE_LABEL_RU["all_distributional"],
            "method": "full_block",
            "method_ru": "Полный блок",
            "mvoi": full_mvoi,
            "share_of_full_mvoi": 1.0,
        }
    )
    return pd.DataFrame(rows)


def _component_decomposition(losses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline = losses[losses["coalition"].eq("none")]
    for (variant, coalition), frame in losses.groupby(["variant", "coalition"], sort=False):
        merged = frame.merge(
            baseline[baseline["variant"].eq(variant)][["scenario", "observation_seed", *[component for component, _ in COMPONENTS]]],
            on=["scenario", "observation_seed"],
            suffixes=("", "_baseline"),
            how="inner",
            validate="one_to_one",
        )
        total_reduction = float(merged["total_loss_baseline"].mean() - merged["total_loss"].mean())
        for component, component_ru in COMPONENTS:
            reduction = float(merged[f"{component}_baseline"].mean() - merged[component].mean())
            rows.append(
                {
                    "variant": variant,
                    "coalition": coalition,
                    "feature_set_ru": str(frame["feature_set_ru"].iloc[0]),
                    "component": component,
                    "component_ru": component_ru,
                    "mean_reduction": reduction,
                    "share_of_total_reduction": reduction / total_reduction if abs(total_reduction) > 1e-14 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _shapley(coalition_mvoi: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    values = coalition_mvoi[coalition_mvoi["variant"].eq(variant)].set_index("coalition")["loss_reduction"].to_dict()
    features = tuple(FEATURES)
    rows: list[dict[str, object]] = []
    n = len(features)
    for feature in features:
        value = 0.0
        others = tuple(name for name in features if name != feature)
        for size in range(0, n):
            for subset in itertools.combinations(others, size):
                subset_key = _coalition_key(subset)
                with_feature_key = _coalition_key((*subset, feature))
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                value += weight * (float(values[with_feature_key]) - float(values[subset_key]))
        rows.append(
            {
                "variant": variant,
                "feature": feature,
                "feature_ru": FEATURE_LABEL_RU[feature],
                "shapley_value": float(value),
            }
        )
    total = sum(row["shapley_value"] for row in rows)
    for row in rows:
        row["share_of_shapley_total"] = row["shapley_value"] / total if abs(total) > 1e-14 else np.nan
    return pd.DataFrame(rows)


def _coalition_key(features: tuple[str, ...]) -> str:
    if not features:
        return "none"
    ordered = [feature for feature in FEATURES if feature in set(features)]
    return "_".join(ordered)


def _feature_set_label(features: tuple[str, ...]) -> str:
    if not features:
        return "Фильтрованные агрегаты"
    return " + ".join(FEATURE_LABEL_RU[feature] for feature in features)


def _rule_rows(coalition: str, information_state: str, variant: str, rule: LinearRule, validation_loss: float) -> list[dict[str, object]]:
    rows = [
        {
            "variant": variant,
            "coalition": coalition,
            "information_state": information_state,
            "term": "intercept",
            "coefficient": rule.intercept,
            "validation_loss": float(validation_loss),
        },
        {
            "variant": variant,
            "coalition": coalition,
            "information_state": information_state,
            "term": "lagged_rate",
            "coefficient": rule.lagged_rate_weight,
            "validation_loss": float(validation_loss),
        },
    ]
    for name, coefficient in zip(rule.spec.feature_names, rule.coefficients):
        rows.append(
            {
                "variant": variant,
                "coalition": coalition,
                "information_state": information_state,
                "term": name,
                "coefficient": coefficient,
                "validation_loss": float(validation_loss),
            }
        )
    rows[0]["coefficient_vector"] = coefficient_vector(rule).tolist()
    return rows


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    for column in numeric:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(feature_mvoi: pd.DataFrame, component_decomposition: pd.DataFrame, shapley: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Разложение распределительной информации по признакам",
        "",
        "Разложение показывает, какой распределительный признак несёт основной вклад и через какой компонент функции потерь он работает.",
        "",
        "## One-feature и leave-one-out",
        "",
    ]
    for _, row in feature_mvoi.iterrows():
        lines.append(f"- {row['method_ru']}, {row['feature_ru']}: MVOI {row['mvoi']:.6g}.")
    lines.extend(["", "## Shapley", ""])
    for _, row in shapley[shapley["variant"].eq("raw")].iterrows():
        lines.append(f"- {row['feature_ru']}: {row['shapley_value']:.6g} ({row['share_of_shapley_total']:.3g}).")
    lines.extend(["", "## Компоненты потерь для полного распределительного блока", ""])
    block = component_decomposition[
        (component_decomposition["variant"].eq("raw"))
        & (component_decomposition["coalition"].eq("mpc_liquidity_exposure"))
        & (~component_decomposition["component"].eq("total_loss"))
    ]
    for _, row in block.iterrows():
        lines.append(
            f"- {row['component_ru']}: снижение {row['mean_reduction']:.6g}, "
            f"доля {row['share_of_total_reduction']:.3g}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(feature_mvoi: pd.DataFrame, shapley: pd.DataFrame, figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    one = feature_mvoi[feature_mvoi["method"].eq("one_feature_at_a_time")]
    leave = feature_mvoi[feature_mvoi["method"].eq("leave_one_feature_out")]
    x = np.arange(len(one))
    width = 0.36
    axes[0].bar(x - width / 2, one["mvoi"], width=width, label="One feature")
    axes[0].bar(x + width / 2, leave["mvoi"], width=width, label="Leave-one-out")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(one["feature_ru"], rotation=15, ha="right")
    axes[0].set_ylabel("MVOI")
    axes[0].set_title("Feature-level value")
    axes[0].legend()

    shp = shapley[shapley["variant"].eq("raw")]
    x_shapley = np.arange(len(shp))
    axes[1].bar(x_shapley, shp["shapley_value"])
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x_shapley)
    axes[1].set_xticklabels(shp["feature_ru"], rotation=15, ha="right")
    axes[1].set_ylabel("Shapley value")
    axes[1].set_title("Shapley decomposition")
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
