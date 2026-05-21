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

from experiments.archive.exp08_main_voi import _supervised_candidates
from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights
from policy.fit_linear_rules import fit_linear_rule
from policy.linear_rules import LinearRule, coefficient_vector
from policy.optimize_linear_rules import (
    LinearRuleOptimizationBounds,
    fit_linear_rule_continuous,
    fitted_rule_as_continuous_like,
)
from policy.optimize_rules import compare_paired_losses


AGGREGATE_FEATURES = ("E_pi", "E_Y", "E_C")
DISTRIBUTIONAL_FEATURES = ("E_mean_mpc", "E_low_liquidity_share", "E_interest_exposure")

VARIANT_LABEL_RU = {
    "actual_distribution": "Фактическая распределительная информация",
    "fake_matched_distribution": "Искусственные признаки с похожей статистикой",
    "permuted_by_scenario": "Перемешивание между сценариями",
    "permuted_by_time": "Перемешивание времени внутри сценария",
    "lagged_distribution": "Запаздывающие распределительные признаки",
    "future_shifted_distribution": "Будущие сдвинутые распределительные признаки",
    "residualized_distribution": "Остаточная распределительная информация",
}


@dataclass(frozen=True)
class IdentificationBatterySpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    variants: tuple[str, ...]
    num_candidates: int
    candidate_seed: int
    continuous_methods: tuple[str, ...]
    num_starts: int
    maxiter: int
    shift: int
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run identification checks for the value of distributional information.")
    parser.add_argument("--information-inputs", default="outputs/ssj/stochastic/state_space/information_inputs/information_state_inputs_long.csv")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/identification_battery")
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--num-candidates", type=int, default=180)
    parser.add_argument("--candidate-seed", type=int, default=6027)
    parser.add_argument("--continuous-methods", default="L-BFGS-B")
    parser.add_argument("--num-starts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--shift", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_seeds = _parse_seed_range(args.validation_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)
    continuous_methods = tuple(part.strip() for part in args.continuous_methods.split(",") if part.strip())

    source = pd.read_csv(args.information_inputs)
    observables = pd.read_csv(args.hank_observables)
    with np.load(args.jacobians) as bundle:
        jacobians = {key: np.asarray(bundle[key], dtype=float) for key in bundle.files if key.startswith("J_")}

    controlled_source = _align_distribution_aggregates_to_filtered_aggregates(source)

    baseline_environment = HankSSJPolicyEnvironment(
        information_inputs=controlled_source,
        observables=observables,
        jacobians=jacobians,
        loss_weights=PolicyLossWeights(),
    )
    aggregate_fit = _fit_rule(
        environment=baseline_environment,
        information_state="filtered_aggregates",
        validation_seeds=validation_seeds,
        num_candidates=args.num_candidates,
        candidate_seed=args.candidate_seed,
        continuous_methods=continuous_methods,
        num_starts=args.num_starts,
        maxiter=args.maxiter,
    )

    variants = _build_variants(
        controlled_source,
        validation_seeds=validation_seeds,
        seed=args.candidate_seed + 100,
        shift=args.shift,
    )
    summary_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    rule_rows.extend(_rule_rows("filtered_aggregates", "baseline", aggregate_fit.rule, aggregate_fit.validation_loss))
    for index, (variant, variant_inputs) in enumerate(variants.items()):
        print(f"Identification variant {variant} ({index + 1}/{len(variants)})", flush=True)
        environment = HankSSJPolicyEnvironment(
            information_inputs=variant_inputs,
            observables=observables,
            jacobians=jacobians,
            loss_weights=PolicyLossWeights(),
        )
        distribution_fit = _fit_rule(
            environment=environment,
            information_state="filtered_distribution",
            validation_seeds=validation_seeds,
            num_candidates=args.num_candidates,
            candidate_seed=args.candidate_seed + index + 1,
            continuous_methods=continuous_methods,
            num_starts=args.num_starts,
            maxiter=args.maxiter,
        )
        rule_rows.extend(_rule_rows("filtered_distribution", variant, distribution_fit.rule, distribution_fit.validation_loss))
        variant_losses = _evaluate_variant(
            environment=environment,
            aggregate_rule=aggregate_fit.rule,
            distribution_rule=distribution_fit.rule,
            variant=variant,
            test_seeds=test_seeds,
        )
        loss_rows.extend(variant_losses.to_dict(orient="records"))
        summary_rows.append(
            _summary_row(
                variant=variant,
                losses=variant_losses,
                validation_loss_aggregate=aggregate_fit.validation_loss,
                validation_loss_distribution=distribution_fit.validation_loss,
                distribution_converged=distribution_fit.converged,
                distribution_message=distribution_fit.message,
            )
        )
        diagnostic_rows.extend(_feature_diagnostics(controlled_source, variant_inputs, variant))

    summary = pd.DataFrame(summary_rows)
    losses = pd.DataFrame(loss_rows)
    rules = pd.DataFrame(rule_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    summary.to_csv(output_dir / "identification_battery_summary.csv", index=False)
    losses.to_csv(output_dir / "identification_battery_trajectory_losses.csv", index=False)
    rules.to_csv(output_dir / "identification_battery_fitted_rules.csv", index=False)
    diagnostics.to_csv(output_dir / "identification_feature_diagnostics.csv", index=False)
    _write_latex(summary, output_dir / "table_identification_battery.tex")
    _write_report(summary, output_dir / "report_identification_battery.md")

    spec = IdentificationBatterySpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        variants=tuple(variants),
        num_candidates=int(args.num_candidates),
        candidate_seed=int(args.candidate_seed),
        continuous_methods=continuous_methods,
        num_starts=int(args.num_starts),
        maxiter=int(args.maxiter),
        shift=int(args.shift),
        note=(
            "Проверка отделяет содержательную распределительную информацию от механического увеличения числа признаков. "
            "Для каждого варианта заново настраивается одно и то же линейное правило filtered_distribution. "
            "Агрегатные признаки E_pi, E_Y и E_C в filtered_distribution принудительно приравнены к filtered_aggregates, "
            "поэтому в проверке меняются только распределительные признаки."
        ),
    )
    (output_dir / "identification_battery_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'identification_battery_summary.csv'}")
    print(f"Wrote {output_dir / 'report_identification_battery.md'}")


def _fit_rule(
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


def _evaluate_variant(
    *,
    environment: HankSSJPolicyEnvironment,
    aggregate_rule: LinearRule,
    distribution_rule: LinearRule,
    variant: str,
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
                    "variant": variant,
                    "variant_ru": VARIANT_LABEL_RU[variant],
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


def _summary_row(
    *,
    variant: str,
    losses: pd.DataFrame,
    validation_loss_aggregate: float,
    validation_loss_distribution: float,
    distribution_converged: bool,
    distribution_message: str,
) -> dict[str, object]:
    delta = losses["delta_distribution_minus_aggregates"].to_numpy(dtype=float)
    comparison = compare_paired_losses(
        left_name="filtered_distribution",
        right_name="filtered_aggregates",
        left_losses=losses["loss_filtered_distribution"].to_numpy(dtype=float),
        right_losses=losses["loss_filtered_aggregates"].to_numpy(dtype=float),
        tie_eps=1e-10,
    )
    return {
        "variant": variant,
        "variant_ru": VARIANT_LABEL_RU[variant],
        "loss_filtered_aggregates": float(losses["loss_filtered_aggregates"].mean()),
        "loss_filtered_distribution": float(losses["loss_filtered_distribution"].mean()),
        "mean_delta": comparison.mean_delta,
        "loss_reduction": -comparison.mean_delta,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "win_rate": comparison.win_rate,
        "tie_rate": comparison.tie_rate,
        "loss_rate": comparison.loss_rate,
        "num_trajectories": comparison.num_trajectories,
        "validation_loss_filtered_aggregates": float(validation_loss_aggregate),
        "validation_loss_filtered_distribution": float(validation_loss_distribution),
        "distribution_optimization_converged": bool(distribution_converged),
        "distribution_optimization_message": distribution_message,
        "output_gap_delta": float(losses["output_gap_delta"].mean()),
        "inflation_delta": float(losses["inflation_delta"].mean()),
        "consumption_delta": float(losses["consumption_delta"].mean()),
        "rate_smoothing_delta": float(losses["rate_smoothing_delta"].mean()),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
    }


def _build_variants(
    source: pd.DataFrame,
    *,
    validation_seeds: list[int],
    seed: int,
    shift: int,
) -> dict[str, pd.DataFrame]:
    variants = {
        "actual_distribution": source.copy(),
        "fake_matched_distribution": _replace_distribution_features(
            source,
            _fake_matched_values(source, validation_seeds=validation_seeds, seed=seed),
        ),
        "permuted_by_scenario": _replace_distribution_features(
            source,
            _scenario_permuted_values(source, seed=seed + 1),
        ),
        "permuted_by_time": _replace_distribution_features(
            source,
            _time_permuted_values(source, seed=seed + 2),
        ),
        "lagged_distribution": _replace_distribution_features(
            source,
            _shifted_values(source, periods=shift),
        ),
        "future_shifted_distribution": _replace_distribution_features(
            source,
            _shifted_values(source, periods=-shift),
        ),
        "residualized_distribution": _replace_distribution_features(
            source,
            _residualized_values(source, validation_seeds=validation_seeds),
        ),
    }
    return variants


def _align_distribution_aggregates_to_filtered_aggregates(source: pd.DataFrame) -> pd.DataFrame:
    """Keep the aggregate block fixed when testing distributional add-ons."""

    result = source.copy()
    aggregate = _aggregate_wide(source)
    aggregate_long = aggregate.melt(
        id_vars=["scenario", "scenario_label", "period", "observation_seed"],
        value_vars=list(AGGREGATE_FEATURES),
        var_name="feature_name",
        value_name="aggregate_value",
    )
    keyed = aggregate_long.set_index(["scenario", "period", "observation_seed", "feature_name"])["aggregate_value"]
    mask = result["information_state"].eq("filtered_distribution") & result["feature_name"].isin(AGGREGATE_FEATURES)
    keys = list(zip(result.loc[mask, "scenario"], result.loc[mask, "period"], result.loc[mask, "observation_seed"], result.loc[mask, "feature_name"]))
    result.loc[mask, "value"] = [float(keyed.loc[key]) for key in keys]
    return result


def _distribution_wide(source: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "scenario_label", "period", "observation_seed"]
    mask = source["information_state"].eq("filtered_distribution")
    wide = (
        source.loc[mask]
        .pivot_table(index=keys, columns="feature_name", values="value", aggfunc="first")
        .reset_index()
        .sort_values(["scenario", "observation_seed", "period"])
    )
    return wide


def _aggregate_wide(source: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "scenario_label", "period", "observation_seed"]
    mask = source["information_state"].eq("filtered_aggregates")
    wide = (
        source.loc[mask]
        .pivot_table(index=keys, columns="feature_name", values="value", aggfunc="first")
        .reset_index()
        .sort_values(["scenario", "observation_seed", "period"])
    )
    return wide


def _replace_distribution_features(source: pd.DataFrame, replacement: pd.DataFrame) -> pd.DataFrame:
    result = source.copy()
    replacement_long = replacement.melt(
        id_vars=["scenario", "scenario_label", "period", "observation_seed"],
        value_vars=list(DISTRIBUTIONAL_FEATURES),
        var_name="feature_name",
        value_name="replacement_value",
    )
    keyed = replacement_long.set_index(["scenario", "period", "observation_seed", "feature_name"])["replacement_value"]
    mask = result["information_state"].eq("filtered_distribution") & result["feature_name"].isin(DISTRIBUTIONAL_FEATURES)
    keys = list(zip(result.loc[mask, "scenario"], result.loc[mask, "period"], result.loc[mask, "observation_seed"], result.loc[mask, "feature_name"]))
    result.loc[mask, "value"] = [float(keyed.loc[key]) for key in keys]
    return result


def _fake_matched_values(source: pd.DataFrame, *, validation_seeds: list[int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wide = _distribution_wide(source)
    aggregate = _aggregate_wide(source)
    merged = wide.merge(
        aggregate[["scenario", "period", "observation_seed", *AGGREGATE_FEATURES]],
        on=["scenario", "period", "observation_seed"],
        suffixes=("", "_agg"),
        how="left",
    )
    train = merged["observation_seed"].isin(validation_seeds)
    x = np.column_stack([np.ones(len(merged)), merged.loc[:, AGGREGATE_FEATURES].to_numpy(dtype=float)])
    x_train = x[train.to_numpy()]
    result = wide.copy()
    for feature in DISTRIBUTIONAL_FEATURES:
        y = merged[feature].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x_train, y[train.to_numpy()], rcond=None)[0]
        fitted = x @ beta
        residual = y - fitted
        rho = _estimate_ar1_from_wide(merged.assign(_residual=residual), "_residual")
        scale = float(np.std(residual[train.to_numpy()], ddof=0))
        fake_residual = _fake_residual_paths(
            merged,
            rng=rng,
            scale=scale,
            rho=rho,
        )
        fake = fitted + fake_residual
        fake = _match_mean_std(fake, y)
        result[feature] = fake
    return result


def _scenario_permuted_values(source: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wide = _distribution_wide(source)
    result = wide.copy()
    for feature in DISTRIBUTIONAL_FEATURES:
        for seed_value, seed_frame in wide.groupby("observation_seed", sort=False):
            scenarios = np.asarray(sorted(seed_frame["scenario"].unique()), dtype=object)
            permuted = scenarios.copy()
            rng.shuffle(permuted)
            mapping = dict(zip(scenarios, permuted))
            source_index = seed_frame.set_index(["scenario", "period"])[feature]
            mask = result["observation_seed"].eq(seed_value)
            values = []
            for row in result.loc[mask, ["scenario", "period"]].itertuples(index=False):
                values.append(float(source_index.loc[(mapping[row.scenario], row.period)]))
            result.loc[mask, feature] = values
    return result


def _time_permuted_values(source: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wide = _distribution_wide(source)
    result = wide.copy()
    for feature in DISTRIBUTIONAL_FEATURES:
        for _, group in wide.groupby(["scenario", "observation_seed"], sort=False):
            values = group[feature].to_numpy(dtype=float).copy()
            rng.shuffle(values)
            result.loc[group.index, feature] = values
    return result


def _shifted_values(source: pd.DataFrame, *, periods: int) -> pd.DataFrame:
    wide = _distribution_wide(source)
    result = wide.copy()
    for feature in DISTRIBUTIONAL_FEATURES:
        shifted_parts = []
        for _, group in wide.groupby(["scenario", "observation_seed"], sort=False):
            shifted = group[feature].shift(periods)
            shifted = shifted.bfill().ffill()
            shifted_parts.append(shifted)
        result[feature] = pd.concat(shifted_parts).sort_index().to_numpy(dtype=float)
    return result


def _residualized_values(source: pd.DataFrame, *, validation_seeds: list[int]) -> pd.DataFrame:
    wide = _distribution_wide(source)
    aggregate = _aggregate_wide(source)
    merged = wide.merge(
        aggregate[["scenario", "period", "observation_seed", *AGGREGATE_FEATURES]],
        on=["scenario", "period", "observation_seed"],
        suffixes=("", "_agg"),
        how="left",
    )
    train = merged["observation_seed"].isin(validation_seeds)
    x = np.column_stack([np.ones(len(merged)), merged.loc[:, AGGREGATE_FEATURES].to_numpy(dtype=float)])
    x_train = x[train.to_numpy()]
    result = wide.copy()
    for feature in DISTRIBUTIONAL_FEATURES:
        y = merged[feature].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x_train, y[train.to_numpy()], rcond=None)[0]
        residual = y - x @ beta
        result[feature] = residual
    return result


def _fake_residual_paths(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    scale: float,
    rho: float,
) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    if scale <= 1e-14:
        return result
    innovation_scale = scale * np.sqrt(max(1.0 - rho**2, 1e-8))
    for _, group in frame.groupby(["scenario", "observation_seed"], sort=False):
        values = np.zeros(len(group), dtype=float)
        values[0] = rng.normal(0.0, scale)
        for index in range(1, len(group)):
            values[index] = rho * values[index - 1] + rng.normal(0.0, innovation_scale)
        result[group.index.to_numpy()] = values
    return result


def _estimate_ar1_from_wide(frame: pd.DataFrame, column: str) -> float:
    lagged: list[float] = []
    current: list[float] = []
    for _, group in frame.groupby(["scenario", "observation_seed"], sort=False):
        values = group[column].to_numpy(dtype=float)
        if len(values) > 1:
            lagged.extend(values[:-1])
            current.extend(values[1:])
    if len(lagged) < 3 or np.std(lagged) <= 1e-14 or np.std(current) <= 1e-14:
        return 0.0
    return float(np.clip(np.corrcoef(lagged, current)[0, 1], -0.95, 0.95))


def _match_mean_std(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    centered = values - np.mean(values)
    scale = np.std(centered, ddof=0)
    target_scale = np.std(reference, ddof=0)
    if scale <= 1e-14:
        return np.full_like(values, np.mean(reference), dtype=float)
    return np.mean(reference) + centered * target_scale / scale


def _feature_diagnostics(source: pd.DataFrame, variant_inputs: pd.DataFrame, variant: str) -> list[dict[str, object]]:
    source_wide = _distribution_wide(source)
    variant_wide = _distribution_wide(variant_inputs)
    aggregate_wide = _aggregate_wide(variant_inputs)
    rows: list[dict[str, object]] = []
    for feature in DISTRIBUTIONAL_FEATURES:
        values = variant_wide[feature].to_numpy(dtype=float)
        source_values = source_wide[feature].to_numpy(dtype=float)
        rows.append(
            {
                "variant": variant,
                "variant_ru": VARIANT_LABEL_RU[variant],
                "feature": feature,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "source_mean": float(np.mean(source_values)),
                "source_std": float(np.std(source_values, ddof=0)),
                "ar1": _estimate_ar1_from_wide(variant_wide, feature),
                "source_ar1": _estimate_ar1_from_wide(source_wide, feature),
                "corr_with_E_pi": _safe_corr(values, aggregate_wide["E_pi"].to_numpy(dtype=float)),
                "corr_with_E_Y": _safe_corr(values, aggregate_wide["E_Y"].to_numpy(dtype=float)),
                "corr_with_E_C": _safe_corr(values, aggregate_wide["E_C"].to_numpy(dtype=float)),
                "corr_with_source_feature": _safe_corr(values, source_values),
                "missing_rate": float(np.mean(~np.isfinite(values))),
            }
        )
    return rows


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) <= 1e-14 or np.std(right[mask]) <= 1e-14:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _rule_rows(
    information_state: str,
    variant: str,
    rule: LinearRule,
    validation_loss: float,
) -> list[dict[str, object]]:
    rows = [
        {
            "variant": variant,
            "information_state": information_state,
            "term": "intercept",
            "coefficient": rule.intercept,
            "validation_loss": validation_loss,
        },
        {
            "variant": variant,
            "information_state": information_state,
            "term": "lagged_rate",
            "coefficient": rule.lagged_rate_weight,
            "validation_loss": validation_loss,
        },
    ]
    for name, coefficient in zip(rule.spec.feature_names, rule.coefficients):
        rows.append(
            {
                "variant": variant,
                "information_state": information_state,
                "term": name,
                "coefficient": coefficient,
                "validation_loss": validation_loss,
            }
        )
    rows[0]["coefficient_vector"] = coefficient_vector(rule).tolist()
    return rows


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "variant_ru",
        "loss_reduction",
        "ci_low",
        "ci_high",
        "win_rate",
        "loss_rate",
        "num_trajectories",
    ]
    display = frame.loc[:, columns].copy()
    display = display.rename(
        columns={
            "variant_ru": "Проверка",
            "loss_reduction": "Снижение потерь",
            "ci_low": "Нижняя граница",
            "ci_high": "Верхняя граница",
            "win_rate": "Доля выигрышей",
            "loss_rate": "Доля ухудшений",
            "num_trajectories": "Число траекторий",
        }
    )
    numeric = display.select_dtypes(include=[np.number]).columns
    for column in numeric:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Идентификационные проверки распределительной информации",
        "",
        "Проверки отделяют содержательную ценность распределительных признаков от механического расширения числа входов правила.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['variant_ru']}: снижение потерь {row['loss_reduction']:.6g}, "
            f"интервал [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"доля выигрышей {row['win_rate']:.3g}."
        )
    lines.extend(
        [
            "",
            "Ключевая проверка -- остаточная распределительная информация. Она оставляет только ту часть распределительных признаков, "
            "которая не объясняется фильтрованными агрегатами на валидационной выборке.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
