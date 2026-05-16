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

from experiments.exp08_main_voi import (
    INFORMATION_STATES,
    PAIRWISE_COMPARISONS,
    STATE_LABEL_RU,
    _fit_policy_under_modes,
    _fit_policy_final,
    _gap_table,
    _load_policy_optimization_config,
    _policy_optimization_budget_table,
    _projected_rule_test,
    _rule_rows,
    _scenario_groups,
    _summary_table,
    _supervised_candidates,
    _write_latex,
)
from hank_ssj import (
    HankSSJPolicyEnvironment,
    PolicyLossWeights,
    build_information_state_inputs,
    build_joint_kalman_filtered_states,
)
from hank_ssj.observations import build_noisy_observations
from hank_ssj.shock_library import generate_stochastic_hank_paths
from policy.optimize_linear_rules import LinearRuleOptimizationBounds
from policy.inference import bh_adjust_pvalues, sign_flip_test, summarize_paired_inference
from policy.optimize_rules import compare_paired_losses


@dataclass(frozen=True)
class LargeSampleSpec:
    shock_library: str
    steady_values: str
    jacobians: str
    output_dir: str
    train_shock_seeds: tuple[int, ...]
    validation_shock_seeds: tuple[int, ...]
    test_shock_seeds: tuple[int, ...]
    observation_seeds_validation: tuple[int, ...]
    observation_seeds_test: tuple[int, ...]
    optimizer_seed: int
    optimization_modes: tuple[str, ...]
    primary_optimization_mode: str
    continuous_methods: tuple[str, ...]
    num_starts: int
    maxiter: int
    cluster_bootstrap_reps: int
    hierarchical_bootstrap_reps: int
    primary_inference: str
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-sample joint-filter evaluation with separate shock and observation seeds.")
    parser.add_argument("--shock-library", default="outputs/ssj/stochastic/shock_response_library.csv")
    parser.add_argument("--steady-values", default="outputs/ssj/stochastic/steady_distributional_values.json")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/large_sample")
    parser.add_argument("--train-shock-seeds", default="0:199")
    parser.add_argument("--validation-shock-seeds", default="200:399")
    parser.add_argument("--test-shock-seeds", default="400:899")
    parser.add_argument("--observation-seeds-train", default="900:929")
    parser.add_argument("--observation-seeds-validation", default="930:959")
    parser.add_argument("--observation-seeds-test", default="960:999")
    parser.add_argument("--optimizer-seed", type=int, default=5027)
    parser.add_argument("--num-candidates", type=int, default=220)
    parser.add_argument("--optimization-modes", default="random_candidates,grid_random,continuous")
    parser.add_argument("--primary-optimization-mode", default="continuous")
    parser.add_argument("--continuous-methods", default="L-BFGS-B")
    parser.add_argument("--num-starts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--intercept-bound", type=float, default=0.01)
    parser.add_argument("--standardized-coefficient-bound", type=float, default=0.05)
    parser.add_argument("--policy-optimization-config", default="config/final_policy_optimization.yaml")
    parser.add_argument("--cluster-bootstrap-reps", type=int, default=2000)
    parser.add_argument("--hierarchical-bootstrap-reps", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.train_shock_seeds = "0:49"
        args.validation_shock_seeds = "200:219"
        args.test_shock_seeds = "400:499"
        args.observation_seeds_validation = "930:934"
        args.observation_seeds_test = "960:969"
        args.policy_optimization_config = ""
        args.num_candidates = min(int(args.num_candidates), 120)
        args.num_starts = 1
        args.maxiter = min(int(args.maxiter), 12)
        args.cluster_bootstrap_reps = min(int(args.cluster_bootstrap_reps), 1000)
        args.hierarchical_bootstrap_reps = min(int(args.hierarchical_bootstrap_reps or 1000), 1000)
        if args.output_dir == "outputs/ssj/stochastic/large_sample":
            args.output_dir = "outputs/ssj/stochastic/large_sample_smoke"

    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    validation_dir = output_dir / "validation"
    test_dir = output_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_shock_seeds = _parse_seed_range(args.train_shock_seeds)
    validation_shock_seeds = _parse_seed_range(args.validation_shock_seeds)
    test_shock_seeds = _parse_seed_range(args.test_shock_seeds)
    observation_seeds_train = _parse_seed_range(args.observation_seeds_train)
    observation_seeds_validation = _parse_seed_range(args.observation_seeds_validation)
    observation_seeds_test = _parse_seed_range(args.observation_seeds_test)
    del observation_seeds_train  # Noise seeds for training are documented in the split but not needed by the current filter.

    optimization_modes = tuple(item.strip() for item in args.optimization_modes.split(",") if item.strip())
    continuous_methods = tuple(item.strip() for item in args.continuous_methods.split(",") if item.strip())
    final_optimization_config = _load_policy_optimization_config(args.policy_optimization_config)
    if final_optimization_config is not None:
        primary_optimization_mode = "final_continuous"
        optimization_modes = ("final_continuous",)
        continuous_methods = tuple(final_optimization_config["methods"])
    else:
        primary_optimization_mode = args.primary_optimization_mode
    if primary_optimization_mode not in optimization_modes:
        raise ValueError("--primary-optimization-mode must be included in --optimization-modes.")

    hierarchical_bootstrap_reps = (
        int(args.cluster_bootstrap_reps)
        if args.hierarchical_bootstrap_reps is None
        else int(args.hierarchical_bootstrap_reps)
    )

    if not args.skip_build:
        _build_large_sample_inputs(
            shock_library=Path(args.shock_library),
            steady_values=Path(args.steady_values),
            train_dir=train_dir,
            validation_dir=validation_dir,
            test_dir=test_dir,
            train_shock_seeds=train_shock_seeds,
            validation_shock_seeds=validation_shock_seeds,
            test_shock_seeds=test_shock_seeds,
            observation_seeds_validation=observation_seeds_validation,
            observation_seeds_test=observation_seeds_test,
        )

    validation_env = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=validation_dir / "information_inputs" / "information_state_inputs_long.csv",
        hank_observables_csv=validation_dir / "hank_observables.csv",
        jacobians_npz=Path(args.jacobians),
        loss_weights=PolicyLossWeights(),
    )
    test_env = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=test_dir / "information_inputs" / "information_state_inputs_long.csv",
        hank_observables_csv=test_dir / "hank_observables.csv",
        jacobians_npz=Path(args.jacobians),
        loss_weights=PolicyLossWeights(),
    )

    bounds = LinearRuleOptimizationBounds(
        intercept_abs_bound=args.intercept_bound,
        standardized_coefficient_abs_bound=args.standardized_coefficient_bound,
    )
    fitted = {}
    fit_results = {}
    rule_rows = []
    all_rule_rows = []
    for index, information_state in enumerate(INFORMATION_STATES):
        print(f"Fitting {information_state} ({index + 1}/{len(INFORMATION_STATES)})", flush=True)
        extra_candidates = _supervised_candidates(
            environment=validation_env,
            information_state=information_state,
            validation_seeds=observation_seeds_validation,
        )
        if final_optimization_config is None:
            results = _fit_policy_under_modes(
                environment=validation_env,
                information_state=information_state,
                validation_seeds=observation_seeds_validation,
                num_candidates=args.num_candidates,
                candidate_seed=args.optimizer_seed + index,
                optimization_modes=optimization_modes,
                extra_candidates=extra_candidates,
                continuous_methods=continuous_methods,
                num_starts=args.num_starts,
                maxiter=args.maxiter,
                bounds=bounds,
            )
        else:
            results = _fit_policy_final(
                environment=validation_env,
                information_state=information_state,
                validation_seeds=observation_seeds_validation,
                candidate_seed=args.optimizer_seed + index,
                extra_candidates=extra_candidates,
                config=final_optimization_config,
            )
        fit_results[information_state] = results
        primary = results[primary_optimization_mode]
        fitted[information_state] = primary.rule
        rule_rows.extend(_rule_rows(primary, optimization_mode=primary_optimization_mode))
        for mode, fit in results.items():
            all_rule_rows.extend(_rule_rows(fit, optimization_mode=mode))

    pd.DataFrame(rule_rows).to_csv(output_dir / "fitted_policy_rules.csv", index=False)
    pd.DataFrame(all_rule_rows).to_csv(output_dir / "fitted_policy_rules_all_optimization_modes.csv", index=False)
    _policy_optimization_budget_table(
        information_states=INFORMATION_STATES,
        final_config=final_optimization_config,
        fallback_args=args,
        optimization_modes=optimization_modes,
        continuous_methods=continuous_methods,
    ).to_csv(output_dir / "policy_optimization_budget.csv", index=False)

    losses = _evaluate_rules_large(test_env, fitted, observation_seeds_test)
    losses.to_csv(output_dir / "trajectory_losses.csv", index=False)
    summary = _summary_table(losses)
    summary_all = summary[summary["scenario"] == "all"].copy()
    summary_by_shock = summary[summary["scenario"] != "all"].copy()
    summary_all.to_csv(output_dir / "main_voi_summary.csv", index=False)
    summary_by_shock.to_csv(output_dir / "main_voi_by_shock_cluster.csv", index=False)
    _write_latex(summary_all, output_dir / "table_main_voi_summary.tex")
    clustered = _clustered_inference(losses, bootstrap_reps=args.cluster_bootstrap_reps, seed=args.optimizer_seed + 99_000)
    hierarchical = _hierarchical_inference(
        losses,
        bootstrap_reps=hierarchical_bootstrap_reps,
        seed=args.optimizer_seed + 119_000,
    )
    pairwise = _pairwise_table_large(losses)
    pairwise_all = _attach_clustered_inference(pairwise[pairwise["scenario"] == "all"].copy(), clustered)
    pairwise_all = _attach_hierarchical_inference(pairwise_all, hierarchical)
    pairwise_by_shock = pairwise[pairwise["scenario"] != "all"].copy()
    pairwise_all.to_csv(output_dir / "pairwise_value_of_information.csv", index=False)
    pairwise_by_shock.to_csv(output_dir / "pairwise_value_of_information_by_shock_cluster.csv", index=False)
    _write_latex(pairwise_all, output_dir / "table_pairwise_value_of_information.tex")
    clustered.to_csv(output_dir / "clustered_inference.csv", index=False)
    hierarchical.to_csv(output_dir / "hierarchical_inference.csv", index=False)
    hierarchical.to_csv(output_dir / "main_inference.csv", index=False)
    _write_latex(clustered, output_dir / "table_clustered_inference.tex")
    _write_latex(hierarchical, output_dir / "table_hierarchical_inference.tex")
    gap = _gap_table(summary_all)
    gap.to_csv(output_dir / "full_information_gap.csv", index=False)
    _write_latex(gap, output_dir / "table_full_information_gap.tex")
    projected = _projected_rule_test(test_env, fitted, observation_seeds_test)
    projected.to_csv(output_dir / "projected_rule_test.csv", index=False)
    _write_report(summary_all, pairwise_all, clustered, hierarchical, output_dir / "report_large_sample.md")

    spec = LargeSampleSpec(
        shock_library=args.shock_library,
        steady_values=args.steady_values,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        train_shock_seeds=tuple(train_shock_seeds),
        validation_shock_seeds=tuple(validation_shock_seeds),
        test_shock_seeds=tuple(test_shock_seeds),
        observation_seeds_validation=tuple(observation_seeds_validation),
        observation_seeds_test=tuple(observation_seeds_test),
        optimizer_seed=int(args.optimizer_seed),
        optimization_modes=optimization_modes,
        primary_optimization_mode=primary_optimization_mode,
        continuous_methods=continuous_methods,
        num_starts=int(final_optimization_config["num_starts"] if final_optimization_config else args.num_starts),
        maxiter=int(final_optimization_config["maxiter"] if final_optimization_config else args.maxiter),
        cluster_bootstrap_reps=int(args.cluster_bootstrap_reps),
        hierarchical_bootstrap_reps=int(hierarchical_bootstrap_reps),
        primary_inference="hierarchical_shock_seed_outer_observation_seed_nested",
        note=(
            "Train shock paths estimate the joint state transition. Validation shock paths tune rules. "
            "Test shock paths evaluate rules. Primary inference is hierarchical: shock_seed is the outer "
            "economic cluster and observation_seed is nested measurement noise within each shock."
        ),
    )
    (output_dir / "large_sample_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'main_voi_summary.csv'}")
    print(f"Wrote {output_dir / 'hierarchical_inference.csv'}")


def _build_large_sample_inputs(
    *,
    shock_library: Path,
    steady_values: Path,
    train_dir: Path,
    validation_dir: Path,
    test_dir: Path,
    train_shock_seeds: list[int],
    validation_shock_seeds: list[int],
    test_shock_seeds: list[int],
    observation_seeds_validation: list[int],
    observation_seeds_test: list[int],
) -> None:
    train = generate_stochastic_hank_paths(
        shock_library_csv=shock_library,
        steady_distributional_values_json=steady_values,
        output_dir=train_dir,
        trajectory_seeds=tuple(train_shock_seeds),
    )
    print(f"Train shock paths: {train['scenario'].nunique()}")
    _build_policy_split(
        split_dir=validation_dir,
        shock_library=shock_library,
        steady_values=steady_values,
        shock_seeds=validation_shock_seeds,
        observation_seeds=observation_seeds_validation,
        transition_observables_csv=train_dir / "hank_observables.csv",
    )
    _build_policy_split(
        split_dir=test_dir,
        shock_library=shock_library,
        steady_values=steady_values,
        shock_seeds=test_shock_seeds,
        observation_seeds=observation_seeds_test,
        transition_observables_csv=train_dir / "hank_observables.csv",
    )


def _build_policy_split(
    *,
    split_dir: Path,
    shock_library: Path,
    steady_values: Path,
    shock_seeds: list[int],
    observation_seeds: list[int],
    transition_observables_csv: Path,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = generate_stochastic_hank_paths(
        shock_library_csv=shock_library,
        steady_distributional_values_json=steady_values,
        output_dir=split_dir,
        trajectory_seeds=tuple(shock_seeds),
    )
    print(f"{split_dir.name} shock paths: {paths['scenario'].nunique()}")
    observations = build_noisy_observations(
        observables_csv=split_dir / "hank_observables.csv",
        output_dir=split_dir,
        seeds=tuple(observation_seeds),
    )
    print(f"{split_dir.name} observations: {len(observations)} rows")
    state_space_dir = split_dir / "state_space"
    filtered = build_joint_kalman_filtered_states(
        observables_csv=split_dir / "hank_observables.csv",
        observations_csv=split_dir / "hank_observations.csv",
        observations_spec_json=split_dir / "hank_observations_spec.json",
        transition_observables_csv=transition_observables_csv,
        output_dir=state_space_dir,
    )
    print(f"{split_dir.name} filtered rows: {len(filtered)}")
    inputs = build_information_state_inputs(
        observables_csv=split_dir / "hank_observables.csv",
        observations_csv=split_dir / "hank_observations.csv",
        filtered_states_csv=state_space_dir / "kalman_filtered_states.csv",
        output_dir=split_dir / "information_inputs",
    )
    print(f"{split_dir.name} information input rows: {len(inputs)}")


def _evaluate_rules_large(environment: HankSSJPolicyEnvironment, fitted, test_observation_seeds: list[int]) -> pd.DataFrame:
    rows = []
    for scenario in environment.scenarios:
        shock_seed = _shock_seed_from_scenario(scenario)
        for observation_seed in test_observation_seeds:
            for state, rule in fitted.items():
                loss = environment.simulate_scenario(
                    policy=rule,
                    information_state=state,
                    scenario=scenario,
                    seed=observation_seed,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "shock_seed": shock_seed,
                        "observation_seed": int(observation_seed),
                        "information_state": state,
                        "information_state_ru": STATE_LABEL_RU[state],
                        **asdict(loss),
                    }
                )
    return pd.DataFrame(rows)


def _pairwise_table_large(losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, frame in _scenario_groups(losses):
        index_cols = ["scenario", "observation_seed"] if scenario != "all" else ["shock_seed", "observation_seed"]
        pivot = frame.pivot_table(
            index=index_cols,
            columns="information_state",
            values="total_loss",
            aggfunc="first",
        )
        for left, right, label in PAIRWISE_COMPARISONS:
            comparison = compare_paired_losses(
                left_name=left,
                right_name=right,
                left_losses=pivot[left].to_numpy(dtype=float),
                right_losses=pivot[right].to_numpy(dtype=float),
                tie_eps=1e-10,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "comparison": f"{left}_minus_{right}",
                    "comparison_ru": label,
                    "left": left,
                    "right": right,
                    "left_ru": STATE_LABEL_RU[left],
                    "right_ru": STATE_LABEL_RU[right],
                    "num_trajectories": comparison.num_trajectories,
                    "mean_delta": comparison.mean_delta,
                    "median_delta": comparison.median_delta,
                    "loss_reduction": -comparison.mean_delta,
                    "bootstrap_ci_low": comparison.ci_low,
                    "bootstrap_ci_high": comparison.ci_high,
                    "ci_low": comparison.ci_low,
                    "ci_high": comparison.ci_high,
                    "permutation_p_value": comparison.permutation_p_value,
                    "pair_sign_flip_p_value": comparison.sign_flip_p_value,
                    "win_rate": comparison.win_rate,
                    "tie_rate": comparison.tie_rate,
                    "loss_rate": comparison.loss_rate,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_permutation_p_value"] = np.nan
        for scenario, index in result.groupby("scenario").groups.items():
            result.loc[index, "bh_permutation_p_value"] = bh_adjust_pvalues(result.loc[index, "permutation_p_value"])
    return result


def _clustered_inference(losses: pd.DataFrame, *, bootstrap_reps: int, seed: int) -> pd.DataFrame:
    pivot = losses.pivot_table(
        index=["shock_seed", "observation_seed"],
        columns="information_state",
        values="total_loss",
        aggfunc="first",
    ).reset_index()
    rng = np.random.default_rng(seed)
    shock_seeds = np.asarray(sorted(pivot["shock_seed"].unique()), dtype=int)
    rows = []
    for left, right, label in PAIRWISE_COMPARISONS:
        if left not in pivot.columns or right not in pivot.columns:
            continue
        frame = pivot[["shock_seed", "observation_seed", left, right]].copy()
        frame["delta"] = frame[left] - frame[right]
        cluster_values = frame.groupby("shock_seed", sort=True)["delta"].mean().to_numpy(dtype=float)
        inference = summarize_paired_inference(
            frame["delta"].to_numpy(dtype=float),
            cluster_id=frame["shock_seed"].to_numpy(dtype=int),
            n_boot=bootstrap_reps,
            n_perm=bootstrap_reps,
            seed=int(rng.integers(0, 2**31 - 1)),
            tie_eps=1e-10,
        )
        rows.append(
            {
                "comparison": f"{left}_minus_{right}",
                "comparison_ru": label,
                "left": left,
                "right": right,
                "num_shock_clusters": inference.num_clusters,
                "num_observation_subclusters": int(frame[["shock_seed", "observation_seed"]].drop_duplicates().shape[0]),
                "num_pair_observations": int(len(frame)),
                "mean_delta": inference.mean_delta,
                "median_delta": inference.median_delta,
                "bootstrap_ci_low": inference.bootstrap_ci_low,
                "bootstrap_ci_high": inference.bootstrap_ci_high,
                "cluster_mean_delta": float(cluster_values.mean()),
                "clustered_se": inference.clustered_se,
                "cluster_ci_low": inference.clustered_ci_low,
                "cluster_ci_high": inference.clustered_ci_high,
                "wild_ci_low": inference.wild_ci_low,
                "wild_ci_high": inference.wild_ci_high,
                "permutation_p_value": inference.permutation_p_value,
                "sign_flip_p_value": inference.sign_flip_p_value,
                "cluster_loss_reduction": float(-cluster_values.mean()),
                "cluster_win_share": float(np.mean(cluster_values < -1e-10)),
                "pair_win_rate": inference.win_rate,
                "tie_rate": inference.tie_rate,
                "loss_rate": inference.loss_rate,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_permutation_p_value"] = bh_adjust_pvalues(result["permutation_p_value"])
        result["bh_sign_flip_p_value"] = bh_adjust_pvalues(result["sign_flip_p_value"])
    return result


def _hierarchical_inference(losses: pd.DataFrame, *, bootstrap_reps: int, seed: int) -> pd.DataFrame:
    pivot = losses.pivot_table(
        index=["shock_seed", "observation_seed"],
        columns="information_state",
        values="total_loss",
        aggfunc="first",
    ).reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for left, right, label in PAIRWISE_COMPARISONS:
        if left not in pivot.columns or right not in pivot.columns:
            continue
        frame = pivot[["shock_seed", "observation_seed", left, right]].copy()
        frame["delta"] = frame[left] - frame[right]
        shock_groups = [
            group["delta"].to_numpy(dtype=float)
            for _, group in frame.groupby("shock_seed", sort=True)
        ]
        shock_means = np.asarray([values.mean() for values in shock_groups], dtype=float)
        observation_counts = np.asarray([values.size for values in shock_groups], dtype=int)
        hierarchical_low, hierarchical_high = _hierarchical_bootstrap_ci(
            shock_groups,
            n_boot=bootstrap_reps,
            rng=rng,
        )
        shock_low, shock_high = _shock_mean_bootstrap_ci(
            shock_means,
            n_boot=bootstrap_reps,
            rng=rng,
        )
        wild_low, wild_high = _shock_wild_bootstrap_ci(
            shock_means,
            n_boot=bootstrap_reps,
            rng=rng,
        )
        mean_delta = float(shock_means.mean())
        rows.append(
            {
                "comparison": f"{left}_minus_{right}",
                "comparison_ru": label,
                "left": left,
                "right": right,
                "primary_inference": "hierarchical_shock_seed_outer_observation_seed_nested",
                "num_shock_clusters": int(len(shock_groups)),
                "num_observation_subclusters": int(frame[["shock_seed", "observation_seed"]].drop_duplicates().shape[0]),
                "min_observation_seeds_per_shock": int(observation_counts.min()),
                "median_observation_seeds_per_shock": float(np.median(observation_counts)),
                "max_observation_seeds_per_shock": int(observation_counts.max()),
                "mean_delta": mean_delta,
                "loss_reduction": float(-mean_delta),
                "hierarchical_ci_low": hierarchical_low,
                "hierarchical_ci_high": hierarchical_high,
                "shock_mean_ci_low": shock_low,
                "shock_mean_ci_high": shock_high,
                "wild_ci_low": wild_low,
                "wild_ci_high": wild_high,
                "sign_flip_p_value": sign_flip_test(
                    shock_means,
                    n_perm=bootstrap_reps,
                    seed=int(rng.integers(0, 2**31 - 1)),
                ),
                "cluster_win_share": float(np.mean(shock_means < -1e-10)),
                "pair_win_rate": float(np.mean(frame["delta"].to_numpy(dtype=float) < -1e-10)),
                "pair_loss_rate": float(np.mean(frame["delta"].to_numpy(dtype=float) > 1e-10)),
                "between_shock_std": float(np.std(shock_means, ddof=1)) if shock_means.size > 1 else 0.0,
                "mean_within_shock_std": float(np.mean([np.std(values, ddof=1) if values.size > 1 else 0.0 for values in shock_groups])),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_sign_flip_p_value"] = bh_adjust_pvalues(result["sign_flip_p_value"])
    return result


def _hierarchical_bootstrap_ci(
    shock_groups: list[np.ndarray],
    *,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not shock_groups:
        return float("nan"), float("nan")
    if len(shock_groups) == 1:
        value = float(np.mean(shock_groups[0]))
        return value, value
    draws = np.empty(int(n_boot), dtype=float)
    num_shocks = len(shock_groups)
    for draw_index in range(int(n_boot)):
        sampled_shocks = rng.integers(0, num_shocks, size=num_shocks)
        sampled_means = []
        for shock_index in sampled_shocks:
            values = shock_groups[int(shock_index)]
            obs_draw = rng.integers(0, values.size, size=values.size)
            sampled_means.append(float(values[obs_draw].mean()))
        draws[draw_index] = float(np.mean(sampled_means))
    return _quantile_interval(draws, alpha=alpha)


def _shock_mean_bootstrap_ci(
    shock_means: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(shock_means, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    return _quantile_interval(values[draws].mean(axis=1), alpha=alpha)


def _shock_wild_bootstrap_ci(
    shock_means: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(shock_means, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    observed = float(values.mean())
    centered = values - observed
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_boot), values.size), replace=True)
    return _quantile_interval(observed + (signs * centered).mean(axis=1), alpha=alpha)


def _quantile_interval(values: np.ndarray, *, alpha: float = 0.05) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(clean, alpha / 2.0)),
        float(np.quantile(clean, 1.0 - alpha / 2.0)),
    )


def _attach_clustered_inference(pairwise: pd.DataFrame, clustered: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "comparison",
        "num_shock_clusters",
        "num_observation_subclusters",
        "clustered_se",
        "cluster_ci_low",
        "cluster_ci_high",
        "wild_ci_low",
        "wild_ci_high",
        "sign_flip_p_value",
        "bh_sign_flip_p_value",
        "cluster_win_share",
    ]
    available = [column for column in columns if column in clustered.columns]
    if pairwise.empty or not available:
        return pairwise
    return pairwise.merge(clustered[available], on="comparison", how="left")


def _attach_hierarchical_inference(pairwise: pd.DataFrame, hierarchical: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "comparison",
        "primary_inference",
        "hierarchical_ci_low",
        "hierarchical_ci_high",
        "shock_mean_ci_low",
        "shock_mean_ci_high",
        "cluster_win_share",
        "pair_win_rate",
        "between_shock_std",
        "mean_within_shock_std",
    ]
    available = [column for column in columns if column in hierarchical.columns]
    if pairwise.empty or not available:
        return pairwise
    return pairwise.merge(hierarchical[available], on="comparison", how="left", suffixes=("", "_hierarchical"))


def _write_report(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    clustered: pd.DataFrame,
    hierarchical: pd.DataFrame,
    path: Path,
) -> None:
    overall = summary.set_index("information_state")
    cluster_key = clustered[clustered["comparison"] == "filtered_distribution_minus_filtered_aggregates"]
    hierarchical_key = hierarchical[hierarchical["comparison"] == "filtered_distribution_minus_filtered_aggregates"]
    lines = [
        "# Основная large-sample статистика",
        "",
        "В финальном прогоне отдельно задаются shock_seed для HANK/SSJ-траекторий, observation_seed для шума наблюдений и optimizer_seed для подбора правила.",
        "Основная статистика является иерархической: shock_seed -- внешний экономический кластер, observation_seed -- вложенный измерительный шум.",
        "",
        "## Основные потери",
        "",
    ]
    for state in ("aggregate_only", "filtered_aggregates", "filtered_distribution", "full_information"):
        if state in overall.index:
            lines.append(f"- {STATE_LABEL_RU[state]}: {overall.loc[state, 'mean_loss']:.6f}.")
    if not hierarchical_key.empty:
        row = hierarchical_key.iloc[0]
        lines.extend(
            [
                "",
                "## Иерархическая проверка главного сравнения",
                "",
                (
                    f"Фильтрованные распределительные показатели минус фильтрованные агрегаты: "
                    f"средняя разность={row['mean_delta']:.6f}, "
                    f"иерархический интервал=[{row['hierarchical_ci_low']:.6f}, {row['hierarchical_ci_high']:.6f}], "
                    f"число HANK/SSJ-кластеров={int(row['num_shock_clusters'])}, "
                    f"число парных наблюдений={int(row['num_observation_subclusters'])}."
                ),
            ]
        )
    elif not cluster_key.empty:
        row = cluster_key.iloc[0]
        lines.extend(
            [
                "",
                "## Кластерная проверка главного сравнения",
                "",
                (
                    f"Фильтрованные распределительные показатели минус фильтрованные агрегаты: "
                    f"средняя разность={row['cluster_mean_delta']:.6f}, "
                    f"кластерный интервал=[{row['cluster_ci_low']:.6f}, {row['cluster_ci_high']:.6f}], "
                    f"число HANK/SSJ-кластеров={int(row['num_shock_clusters'])}, "
                    f"число парных наблюдений={int(row['num_observation_subclusters'])}."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shock_seed_from_scenario(scenario: str) -> int:
    return int(str(scenario).split("_")[-1])


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
