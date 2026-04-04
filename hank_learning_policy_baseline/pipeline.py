from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.household_solver import compute_mpc
from hank_full_baseline.steady_state import solve_steady_state
from hank_partial_info_baseline.state_space import fit_reduced_state_space

from .config import Stage4Config, default_stage4_config
from .environment import HANKPolicyEnvironment, build_scenario_spec
from .evaluation import (
    build_policy_comparison,
    evaluate_episode_on_full_hank,
    evaluate_policy_run,
    simulate_policy_episode,
    summarize_training_history,
)
from .plots import (
    plot_ablations,
    plot_group_consumption,
    plot_macro_paths,
    plot_policy_paths,
    plot_scenario_performance,
    plot_training_curve,
)
from .policies import ClassicalFilteredRulePolicy, FullInformationRulePolicy
from .ppo import train_ppo_policy
from .tables import (
    ablation_table,
    distributional_summary_table,
    macro_summary_table,
    policy_performance_table,
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported type: {type(value)!r}")


def _save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


def _save_table(table: pd.DataFrame, basepath: Path):
    table.to_csv(basepath.with_suffix(".csv"), index=False)
    basepath.with_suffix(".tex").write_text(table.to_latex(index=False, escape=False))


def _reduced_state_payload(reduced_model) -> dict:
    return {
        "state_names": list(reduced_model.state_names),
        "observation_names": list(reduced_model.observation_names),
        "transition_matrix": reduced_model.transition_matrix,
        "control_loadings": reduced_model.control_loadings,
        "observation_matrix": reduced_model.observation_matrix,
        "observation_fit_rmse": reduced_model.observation_fit_rmse,
        "steady_state_statistics": reduced_model.steady_state_statistics,
        "training_summary": reduced_model.training_summary,
    }


def _stage4_policy_spec(config: Stage4Config, hank_config) -> dict:
    return {
        "policy_family": "residual_ppo_over_filtered_classical_rule",
        "classical_reference_rule": {
            "phi_pi": float(hank_config.phi_pi),
            "phi_y": float(hank_config.phi_y),
            "rho_i": float(hank_config.rho_i),
        },
        "rl_action": "continuous residual around filtered classical rule",
        "action_bound": float(config.action_bound),
        "rate_bounds": [-2.5 * float(config.action_bound), 2.5 * float(config.action_bound)],
        "observation_modes": {
            "filtered_state": "(x_hat_t, i_{t-1})",
            "filtered_state_uncertainty": "(x_hat_t, diag(P_t), i_{t-1})",
            "raw_observations": "(y_t, i_{t-1})",
        },
        "loss_function": "L_t = pi_t^2 + lambda_y y_gap_t^2 + lambda_i (i_t - i_{t-1})^2",
        "lambda_y": float(config.lambda_y),
        "lambda_i": float(config.lambda_i),
        "gamma": float(config.gamma),
        "selection_metric": config.selection_metric,
        "selection_seeds": list(config.selection_seeds),
        "ppo": config.ppo.to_dict(),
    }


def _selection_key(selection_metric: str, summary: dict[str, float | int]) -> tuple[float, float, int, int]:
    if selection_metric == "full_hank_cumulative_loss":
        return (
            float(summary["selection_objective"]),
            float(summary["selection_policy_rate_rmse"]),
            int(summary["selection_unstable_episodes"]),
            int(summary["checkpoint_iteration"]),
        )
    return (
        -float(summary["best_validation_return"]),
        float(summary["selection_policy_rate_rmse"]),
        int(summary["selection_unstable_episodes"]),
        int(summary["checkpoint_iteration"]),
    )


def _evaluate_policy_selection(
    *,
    policy,
    env_factory,
    scenario_spec,
    config: Stage4Config,
    bundle,
    ss,
    hank_config,
    mpc_ss,
    reduced_model,
    full_information_policy,
    training_seed: int,
):
    cumulative_losses = []
    rate_rmses = []
    unstable_episodes = 0

    for selection_seed in config.selection_seeds:
        full_trace = simulate_policy_episode(
            env_factory=env_factory,
            policy=full_information_policy,
            scenario_spec=scenario_spec,
            evaluation_seed=int(selection_seed),
            policy_name="full_information_rule",
            policy_label="Полная информация",
            training_seed=None,
        )
        candidate_trace = simulate_policy_episode(
            env_factory=env_factory,
            policy=policy,
            scenario_spec=scenario_spec,
            evaluation_seed=int(selection_seed),
            policy_name="learning_policy",
            policy_label="Learning-based policy",
            training_seed=training_seed,
        )
        try:
            full_run = evaluate_episode_on_full_hank(
                bundle=bundle,
                ss=ss,
                hank_config=hank_config,
                mpc_ss=mpc_ss,
                policy_trace=full_trace,
                scenario_spec=scenario_spec,
                policy_name="full_information_rule",
                policy_label="Полная информация",
                training_seed=None,
                evaluation_seed=int(selection_seed),
            )
            candidate_run = evaluate_episode_on_full_hank(
                bundle=bundle,
                ss=ss,
                hank_config=hank_config,
                mpc_ss=mpc_ss,
                policy_trace=candidate_trace,
                scenario_spec=scenario_spec,
                policy_name="learning_policy",
                policy_label="Learning-based policy",
                training_seed=training_seed,
                evaluation_seed=int(selection_seed),
            )
            metrics, _, _ = evaluate_policy_run(
                run=candidate_run,
                reference_run=full_run,
                scenario_spec=scenario_spec,
                stage4_config=config,
                steady_state_statistics=reduced_model.steady_state_statistics,
            )
            cumulative_losses.append(float(metrics["cumulative_policy_loss"]))
            rate_rmses.append(float(metrics["policy_rate_rmse"]))
            unstable_episodes += int(metrics["unstable"])
        except Exception:
            cumulative_losses.append(1.0)
            rate_rmses.append(10.0 * float(config.action_bound))
            unstable_episodes += 1

    return {
        "selection_objective": float(np.mean(cumulative_losses)),
        "selection_policy_rate_rmse": float(np.mean(rate_rmses)),
        "selection_unstable_episodes": int(unstable_episodes),
    }


def _write_report(
    output_dir: Path,
    config: Stage4Config,
    training_seed_summary: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    policy_comparison: pd.DataFrame,
) -> None:
    main_metrics = policy_metrics[
        (policy_metrics["policy_name"].isin(["classical_filtered_rule", "learning_policy"]))
        & (policy_metrics["variant_name"].isin([variant.name for variant in config.main_variants()]))
    ].copy()
    best_main = policy_comparison.loc[policy_comparison["delta_cumulative_policy_loss_rl_minus_classical"].idxmin()]
    worst_main = policy_comparison.loc[policy_comparison["delta_cumulative_policy_loss_rl_minus_classical"].idxmax()]
    selected_seed = training_seed_summary.sort_values(
        ["variant_name", "selection_objective", "selection_policy_rate_rmse", "checkpoint_iteration"],
        ascending=[True, True, True, True],
    ).drop_duplicates("variant_name")

    lines = [
        "# Этап 4. Learning-based policy layer в полной HANK при неполной информации",
        "",
        "## Постановка",
        "",
        "- Структурная two-asset HANK-среда, reduced-state representation и Kalman filtering block сохранены без изменений относительно этапа 3.",
        "- Меняется только последнее звено: вместо fixed Taylor-type rule используется learning-based policy layer.",
        "- Практически baseline реализован как residual PPO: агент получает filtered policy-relevant state и учит непрерывную поправку к classical filtered rule.",
        "- Это сохраняет честное сравнение `filtering + fixed rule` против `filtering + learned policy` внутри одной и той же информационной среды.",
        "",
        "## Обучение",
        "",
        f"- PPO training seeds: {', '.join(str(seed) for seed in config.training_seeds)}.",
        f"- Selection seeds: {', '.join(str(seed) for seed in config.selection_seeds)}.",
        f"- Evaluation seeds: {', '.join(str(seed) for seed in config.evaluation_seeds)}.",
        f"- Горизонт эпизода: {config.horizon} периодов.",
        f"- Reward совпадает с classical loss: `-(pi^2 + {config.lambda_y:.2f} * y_gap^2 + {config.lambda_i:.2f} * Delta i^2)`.",
        f"- Отбор learning-based policy ведётся по метрике `{config.selection_metric}`.",
        "",
        "## Основные результаты",
        "",
        f"- Лучший основной сценарий для RL относительно classical: `{best_main['scenario_label']}`; разница накопленной потери RL минус classical `{best_main['delta_cumulative_policy_loss_rl_minus_classical']:.4e}`.",
        f"- Наименее благоприятный основной сценарий: `{worst_main['scenario_label']}`; разница накопленной потери `{worst_main['delta_cumulative_policy_loss_rl_minus_classical']:.4e}`.",
        "- RL сравнивается главным образом с classical filter-plus-rule; full-information rule используется только как reference layer.",
        "",
        "## Seeds и стабильность",
        "",
    ]
    if any(variant.classical_benchmark_scenario_name is not None for variant in config.main_variants()):
        lines.insert(
            8,
            "- В HANK-specific extension comparator дополнительно ужесточается: RL получает distribution-augmented information set, а classical benchmark остаётся macro-only filter-plus-rule architecture.",
        )
    for _, row in selected_seed.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: training seed `{int(row['training_seed'])}`, checkpoint `{int(row.get('checkpoint_iteration', -1))}`, selection objective `{row.get('selection_objective', float('nan')):.4e}`."
        )

    lines.extend([
        "",
        "## Интерпретация",
        "",
        "- Этот этап не заменяет structural model и не учит политику на полном распределении агентов напрямую.",
        "- Learning-based layer работает только поверх filtered reduced HANK state и поэтому изолирует именно policy-mapping problem.",
        "- Основной содержательный вопрос здесь: когда гибкая learned reaction function улучшает или не улучшает classical filtered Taylor-type rule при ограниченной информации.",
    ])
    (output_dir / "report_stage4_learning_policy_hank.md").write_text("\n".join(lines))


def run_pipeline(
    config: Stage4Config | None = None,
    output_dir: str | None = None,
    variant_names: list[str] | None = None,
):
    config = default_stage4_config() if config is None else config
    if output_dir is not None:
        config = replace(config, output_dir=output_dir)

    root = Path(config.output_dir)
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    ss = bundle["ss"]
    mpc_ss = compute_mpc(ss)
    reduced_model = fit_reduced_state_space(bundle, hank_config, config.partial_config)

    variants = config.all_variants()
    if variant_names is not None:
        variant_set = set(variant_names)
        variants = [variant for variant in variants if variant.name in variant_set]
    scenario_specs = {variant.name: build_scenario_spec(config, variant) for variant in variants}

    _save_json(root / "stage4_config.json", config.to_dict())
    _save_json(root / "policy_spec.json", _stage4_policy_spec(config, hank_config))
    _save_json(root / "filter_spec.json", config.partial_config.filter_spec_payload())
    _save_json(root / "model_spec.json", config.partial_config.model_spec_payload("full_two_asset_hank_baseline"))
    _save_json(root / "reduced_state_space.json", _reduced_state_payload(reduced_model))
    _save_json(root / "scenario_spec.json", [spec.to_dict() for spec in scenario_specs.values()])

    training_histories = []
    training_seed_rows = []
    selection_rows = []
    selected_policy_map = {}
    full_information_policy = FullInformationRulePolicy(action_bound=config.action_bound)

    for variant in variants:
        scenario_spec = scenario_specs[variant.name]
        classical_benchmark_spec = scenario_spec
        if variant.classical_benchmark_scenario_name is not None:
            scenario_labels = {
                spec["name"]: spec["label"]
                for spec in config.partial_config.scenario_specs()
            }
            benchmark_variant = replace(
                variant,
                scenario_name=variant.classical_benchmark_scenario_name,
                scenario_label=scenario_labels[variant.classical_benchmark_scenario_name],
                input_mode="filtered_state",
                include_distributional_state=False,
                classical_benchmark_scenario_name=None,
            )
            classical_benchmark_spec = build_scenario_spec(config, benchmark_variant)

        def env_factory(spec=scenario_spec):
            return HANKPolicyEnvironment(
                reduced_model=reduced_model,
                partial_config=config.partial_config,
                scenario_spec=spec,
                phi_pi=hank_config.phi_pi,
                phi_y=hank_config.phi_y,
                rho_i=hank_config.rho_i,
            )

        best_candidate_summary = None
        best_policy = None
        best_seed = None
        for training_seed in config.training_seeds:
            policy, history, checkpoints = train_ppo_policy(
                env_factory=env_factory,
                ppo_config=config.ppo,
                action_bound=config.action_bound,
                gamma=config.gamma,
                training_seed=training_seed,
                label=variant.name,
            )
            history.insert(0, "variant_name", variant.name)
            history.insert(1, "scenario_name", variant.scenario_name)
            history.insert(2, "scenario_label", variant.scenario_label)
            history.insert(3, "input_mode", variant.input_mode)
            training_histories.append(history)

            best_validation_seed = float(history["best_validation_return"].max())
            checkpoint_summaries = []
            for checkpoint in checkpoints:
                selection_summary = _evaluate_policy_selection(
                    policy=checkpoint.policy,
                    env_factory=env_factory,
                    scenario_spec=scenario_spec,
                    config=config,
                    bundle=bundle,
                    ss=ss,
                    hank_config=hank_config,
                    mpc_ss=mpc_ss,
                    reduced_model=reduced_model,
                    full_information_policy=full_information_policy,
                    training_seed=int(training_seed),
                )
                checkpoint_summary = {
                    "variant_name": variant.name,
                    "scenario_name": variant.scenario_name,
                    "scenario_label": variant.scenario_label,
                    "training_seed": int(training_seed),
                    "checkpoint_iteration": int(checkpoint.iteration),
                    "best_validation_return": best_validation_seed,
                    "checkpoint_validation_return": float(checkpoint.validation_return),
                    "checkpoint_mean_episode_return": float(checkpoint.mean_episode_return),
                    **selection_summary,
                }
                checkpoint_summaries.append(checkpoint_summary)
                selection_rows.append(checkpoint_summary)

            best_checkpoint_summary = min(
                checkpoint_summaries,
                key=lambda row: _selection_key(config.selection_metric, row),
            )
            training_seed_rows.append({
                "variant_name": variant.name,
                "scenario_name": variant.scenario_name,
                "scenario_label": variant.scenario_label,
                "training_seed": int(training_seed),
                "best_validation_return": best_validation_seed,
                "final_validation_return": float(history["validation_return"].iloc[-1]),
                "final_mean_episode_return": float(history["mean_episode_return"].iloc[-1]),
                "checkpoint_iteration": int(best_checkpoint_summary["checkpoint_iteration"]),
                "selection_objective": float(best_checkpoint_summary["selection_objective"]),
                "selection_policy_rate_rmse": float(best_checkpoint_summary["selection_policy_rate_rmse"]),
                "selection_unstable_episodes": int(best_checkpoint_summary["selection_unstable_episodes"]),
            })
            if best_candidate_summary is None or _selection_key(config.selection_metric, best_checkpoint_summary) < _selection_key(config.selection_metric, best_candidate_summary):
                best_candidate_summary = best_checkpoint_summary
                best_policy = next(
                    checkpoint.policy
                    for checkpoint in checkpoints
                    if int(checkpoint.iteration) == int(best_checkpoint_summary["checkpoint_iteration"])
                )
                best_seed = training_seed
        if best_policy is None or best_seed is None:
            raise RuntimeError(f"Failed to train PPO policy for variant {variant.name}.")
        selected_policy_map[variant.name] = {
            "policy": best_policy,
            "training_seed": best_seed,
            "best_validation_return": float(best_candidate_summary["best_validation_return"]),
            "checkpoint_iteration": int(best_candidate_summary["checkpoint_iteration"]),
            "selection_objective": float(best_candidate_summary["selection_objective"]),
            "selection_policy_rate_rmse": float(best_candidate_summary["selection_policy_rate_rmse"]),
        }

    training_history = pd.concat(training_histories, ignore_index=True) if training_histories else pd.DataFrame()
    training_seed_summary = pd.DataFrame(training_seed_rows)
    selection_summary = pd.DataFrame(selection_rows)
    _save_json(
        root / "selected_policy_summary.json",
        {
            variant_name: {
                "training_seed": int(payload["training_seed"]),
                "best_validation_return": float(payload["best_validation_return"]),
                "checkpoint_iteration": int(payload["checkpoint_iteration"]),
                "selection_objective": float(payload["selection_objective"]),
                "selection_policy_rate_rmse": float(payload["selection_policy_rate_rmse"]),
            }
            for variant_name, payload in selected_policy_map.items()
        },
    )
    training_history.to_csv(root / "training_history.csv", index=False)
    training_seed_summary.to_csv(root / "training_seed_summary.csv", index=False)
    selection_summary.to_csv(root / "selection_summary.csv", index=False)

    classical_policy = ClassicalFilteredRulePolicy(action_bound=config.action_bound)

    episode_results = []
    policy_metric_rows = []
    comparison_path_frames = []
    group_comparison_frames = []

    for variant in variants:
        scenario_spec = scenario_specs[variant.name]
        classical_benchmark_spec = scenario_spec
        if variant.classical_benchmark_scenario_name is not None:
            scenario_labels = {
                spec["name"]: spec["label"]
                for spec in config.partial_config.scenario_specs()
            }
            benchmark_variant = replace(
                variant,
                scenario_name=variant.classical_benchmark_scenario_name,
                scenario_label=scenario_labels[variant.classical_benchmark_scenario_name],
                input_mode="filtered_state",
                include_distributional_state=False,
                classical_benchmark_scenario_name=None,
            )
            classical_benchmark_spec = build_scenario_spec(config, benchmark_variant)

        def env_factory(spec=scenario_spec):
            return HANKPolicyEnvironment(
                reduced_model=reduced_model,
                partial_config=config.partial_config,
                scenario_spec=spec,
                phi_pi=hank_config.phi_pi,
                phi_y=hank_config.phi_y,
                rho_i=hank_config.rho_i,
            )

        def classical_env_factory(spec=classical_benchmark_spec):
            return HANKPolicyEnvironment(
                reduced_model=reduced_model,
                partial_config=config.partial_config,
                scenario_spec=spec,
                phi_pi=hank_config.phi_pi,
                phi_y=hank_config.phi_y,
                rho_i=hank_config.rho_i,
            )

        rl_policy = selected_policy_map[variant.name]["policy"]
        rl_training_seed = int(selected_policy_map[variant.name]["training_seed"])

        for evaluation_seed in config.evaluation_seeds:
            full_trace = simulate_policy_episode(
                env_factory=env_factory,
                policy=full_information_policy,
                scenario_spec=scenario_spec,
                evaluation_seed=evaluation_seed,
                policy_name="full_information_rule",
                policy_label="Полная информация",
                training_seed=None,
            )
            classical_trace = simulate_policy_episode(
                env_factory=classical_env_factory,
                policy=classical_policy,
                scenario_spec=classical_benchmark_spec,
                evaluation_seed=evaluation_seed,
                policy_name="classical_filtered_rule",
                policy_label=variant.classical_policy_label,
                training_seed=None,
            )
            rl_trace = simulate_policy_episode(
                env_factory=env_factory,
                policy=rl_policy,
                scenario_spec=scenario_spec,
                evaluation_seed=evaluation_seed,
                policy_name="learning_policy",
                policy_label="Learning-based policy",
                training_seed=rl_training_seed,
            )

            for trace, information_spec in (
                (full_trace, scenario_spec),
                (classical_trace, classical_benchmark_spec),
                (rl_trace, scenario_spec),
            ):
                trace["variant_name"] = scenario_spec.variant_name
                trace["scenario_name"] = scenario_spec.scenario_name
                trace["scenario_label"] = scenario_spec.scenario_label
                trace["information_scenario_name"] = information_spec.scenario_name
                trace["information_scenario_label"] = information_spec.scenario_label

            full_run = evaluate_episode_on_full_hank(
                bundle=bundle,
                ss=ss,
                hank_config=hank_config,
                mpc_ss=mpc_ss,
                policy_trace=full_trace,
                scenario_spec=scenario_spec,
                policy_name="full_information_rule",
                policy_label="Полная информация",
                training_seed=None,
                evaluation_seed=evaluation_seed,
            )
            classical_run = evaluate_episode_on_full_hank(
                bundle=bundle,
                ss=ss,
                hank_config=hank_config,
                mpc_ss=mpc_ss,
                policy_trace=classical_trace,
                scenario_spec=scenario_spec,
                policy_name="classical_filtered_rule",
                policy_label=variant.classical_policy_label,
                training_seed=None,
                evaluation_seed=evaluation_seed,
            )
            rl_run = evaluate_episode_on_full_hank(
                bundle=bundle,
                ss=ss,
                hank_config=hank_config,
                mpc_ss=mpc_ss,
                policy_trace=rl_trace,
                scenario_spec=scenario_spec,
                policy_name="learning_policy",
                policy_label="Learning-based policy",
                training_seed=rl_training_seed,
                evaluation_seed=evaluation_seed,
            )

            episode_results.extend([full_run, classical_run, rl_run])

            full_metrics, full_paths, full_group_comp = evaluate_policy_run(
                run=full_run,
                reference_run=full_run,
                scenario_spec=scenario_spec,
                stage4_config=config,
                steady_state_statistics=reduced_model.steady_state_statistics,
            )
            classical_metrics, classical_paths, classical_group_comp = evaluate_policy_run(
                run=classical_run,
                reference_run=full_run,
                scenario_spec=scenario_spec,
                stage4_config=config,
                steady_state_statistics=reduced_model.steady_state_statistics,
            )
            rl_metrics, rl_paths, rl_group_comp = evaluate_policy_run(
                run=rl_run,
                reference_run=full_run,
                scenario_spec=scenario_spec,
                stage4_config=config,
                steady_state_statistics=reduced_model.steady_state_statistics,
            )
            for metrics, paths, groups, information_spec, policy_information_label in (
                (full_metrics, full_paths, full_group_comp, scenario_spec, "Полная информация"),
                (classical_metrics, classical_paths, classical_group_comp, classical_benchmark_spec, variant.classical_policy_label),
                (rl_metrics, rl_paths, rl_group_comp, scenario_spec, "Learning-based policy"),
            ):
                metrics["information_scenario_name"] = information_spec.scenario_name
                metrics["information_scenario_label"] = information_spec.scenario_label
                metrics["policy_information_label"] = policy_information_label
                paths["information_scenario_name"] = information_spec.scenario_name
                paths["information_scenario_label"] = information_spec.scenario_label
                groups["information_scenario_name"] = information_spec.scenario_name
                groups["information_scenario_label"] = information_spec.scenario_label
            policy_metric_rows.extend([full_metrics, classical_metrics, rl_metrics])
            comparison_path_frames.extend([full_paths, classical_paths, rl_paths])
            group_comparison_frames.extend([full_group_comp, classical_group_comp, rl_group_comp])

    policy_metrics = pd.DataFrame(policy_metric_rows)
    policy_paths = pd.concat(comparison_path_frames, ignore_index=True) if comparison_path_frames else pd.DataFrame()
    group_comparisons = pd.concat(group_comparison_frames, ignore_index=True) if group_comparison_frames else pd.DataFrame()
    policy_traces = pd.concat([result.policy_trace for result in episode_results], ignore_index=True)
    aggregate_paths = pd.concat([result.aggregate_paths for result in episode_results], ignore_index=True)
    distribution_stats = pd.concat([result.distribution_stats for result in episode_results], ignore_index=True)
    group_paths = pd.concat([result.group_paths for result in episode_results], ignore_index=True)

    policy_metrics = (
        policy_metrics.sort_values(
            ["variant_name", "policy_name", "evaluation_seed", "training_seed"],
            ignore_index=True,
        )
    )
    policy_comparison = build_policy_comparison(
        policy_metrics[
            policy_metrics["variant_name"].isin([variant.name for variant in config.all_variants()])
        ]
        .sort_values(["variant_name", "policy_name"])
        .drop_duplicates(["variant_name", "policy_name"], keep="first")
    )

    main_variant_names = {variant.name for variant in config.main_variants()}
    ablation_variant_names = {variant.name for variant in config.ablation_variants()}
    main_policy_comparison = policy_comparison[policy_comparison["variant_name"].isin(main_variant_names)].copy()
    ablation_policy_comparison = policy_comparison[policy_comparison["variant_name"].isin(ablation_variant_names)].copy()

    policy_traces.to_csv(root / "policy_traces.csv", index=False)
    policy_paths.to_csv(root / "policy_paths.csv", index=False)
    aggregate_paths.to_csv(root / "aggregate_paths.csv", index=False)
    distribution_stats.to_csv(root / "distribution_stats.csv", index=False)
    group_paths.to_csv(root / "group_paths.csv", index=False)
    group_comparisons.to_csv(root / "group_comparisons.csv", index=False)
    policy_metrics.to_csv(root / "policy_metrics.csv", index=False)
    policy_comparison.to_csv(root / "policy_comparison.csv", index=False)
    main_policy_comparison.to_csv(root / "policy_performance_main.csv", index=False)
    ablation_policy_comparison.to_csv(root / "ablation_results.csv", index=False)

    _save_table(
        policy_performance_table(
            policy_metrics[
                policy_metrics["variant_name"].isin(main_variant_names)
                & policy_metrics["policy_name"].isin(["classical_filtered_rule", "learning_policy"])
            ]
        ),
        tables_dir / "table_01_policy_performance",
    )
    _save_table(
        macro_summary_table(
            policy_metrics[
                policy_metrics["variant_name"].isin(main_variant_names)
                & policy_metrics["policy_name"].isin(["classical_filtered_rule", "learning_policy", "full_information_rule"])
            ]
        ),
        tables_dir / "table_02_macro_summary",
    )
    _save_table(
        distributional_summary_table(
            policy_metrics[
                policy_metrics["variant_name"].isin(main_variant_names)
                & policy_metrics["policy_name"].isin(["classical_filtered_rule", "learning_policy"])
            ]
        ),
        tables_dir / "table_03_distributional_summary",
    )
    _save_table(
        ablation_table(ablation_policy_comparison),
        tables_dir / "table_04_ablations",
    )

    available_variants = set(policy_metrics["variant_name"].unique()) if not policy_metrics.empty else set()
    representative_variant = next(
        (
            name
            for name in (
                "distribution_sensitive_stress_filtered_state",
                "distribution_sensitive_filtered_state",
                "macro_core_filtered_state",
                "distribution_augmented_filtered_state",
            )
            if name in available_variants
        ),
        None,
    )
    if representative_variant is not None:
        plot_policy_paths(policy_paths, figures_dir, representative_variant)
        plot_macro_paths(aggregate_paths, figures_dir, representative_variant)
        plot_group_consumption(group_paths, figures_dir, representative_variant)
    if not main_policy_comparison.empty:
        plot_scenario_performance(main_policy_comparison, figures_dir)
    if not ablation_policy_comparison.empty:
        plot_ablations(ablation_policy_comparison, figures_dir)
    if not training_history.empty:
        training_variant = representative_variant or str(training_history["label"].iloc[0])
        plot_training_curve(training_history, figures_dir, training_variant)

    _write_report(root, config, training_seed_summary, policy_metrics, main_policy_comparison)

    return {
        "policy_metrics": policy_metrics,
        "policy_comparison": policy_comparison,
        "training_history": training_history,
        "training_seed_summary": training_seed_summary,
    }
