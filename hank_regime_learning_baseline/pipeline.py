from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_learning_policy_baseline.policies import (
    ClassicalFilteredRulePolicy,
    FullInformationRulePolicy,
)
from hank_learning_policy_baseline.ppo import train_ppo_policy
from hank_partial_info_baseline.state_space import fit_reduced_state_space
from regime_switching_baseline.regime_model import build_regime_switching_model, regime_model_spec_payload

from .config import RegimeLearningConfig, default_regime_learning_config
from .environment import RegimeSwitchingPolicyEnvironment, build_scenario_spec
from .evaluation import (
    build_policy_comparison,
    evaluate_policy_trace,
    simulate_policy_episode,
    summarize_training_history,
)
from .policies import MisspecifiedClassicalRulePolicy


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


def _selection_key(summary: dict[str, float | int]) -> tuple[float, float, int]:
    return (
        float(summary["selection_objective"]),
        float(summary["selection_rate_rmse"]),
        int(summary["selection_unstable_episodes"]),
    )


def _evaluate_checkpoint_selection(
    *,
    env_factory,
    policy,
    classical_policy,
    classical_label: str,
    scenario_spec,
    selection_seeds: tuple[int, ...],
) -> dict[str, float | int]:
    cumulative_losses = []
    rate_rmses = []
    unstable = 0
    for selection_seed in selection_seeds:
        classical_trace = simulate_policy_episode(
            env_factory=env_factory,
            policy=classical_policy,
            scenario_spec=scenario_spec,
            evaluation_seed=int(selection_seed),
            policy_name="classical_filtered_rule",
            policy_label=classical_label,
            training_seed=None,
        )
        candidate_trace = simulate_policy_episode(
            env_factory=env_factory,
            policy=policy,
            scenario_spec=scenario_spec,
            evaluation_seed=int(selection_seed),
            policy_name="learning_policy",
            policy_label="Learning-based policy",
            training_seed=None,
        )
        metrics, _ = evaluate_policy_trace(
            policy_trace=candidate_trace,
            reference_trace=classical_trace,
            scenario_spec=scenario_spec,
        )
        cumulative_losses.append(float(metrics["cumulative_policy_loss"]))
        rate_rmses.append(float(metrics["policy_rate_rmse"]))
        unstable += int(metrics["unstable"])
    return {
        "selection_objective": float(np.mean(cumulative_losses)),
        "selection_rate_rmse": float(np.mean(rate_rmses)),
        "selection_unstable_episodes": int(unstable),
    }


def _write_report(
    output_dir: Path,
    training_seed_summary: pd.DataFrame,
    policy_comparison: pd.DataFrame,
) -> None:
    best = policy_comparison.loc[policy_comparison["delta_cumulative_policy_loss_rl_minus_classical"].idxmin()]
    worst = policy_comparison.loc[policy_comparison["delta_cumulative_policy_loss_rl_minus_classical"].idxmax()]
    all_nonnegative = bool(np.all(policy_comparison["delta_cumulative_policy_loss_rl_minus_classical"].to_numpy(dtype=float) >= 0.0))
    lines = [
        "# Этап 6. Learning-based policy в regime-switching HANK",
        "",
        "## Постановка",
        "",
        "- Используется stage-5 regime-switching reduced-state HANK overlay с hidden regimes `normal` и `stress`.",
        "- Classical benchmark: `switching filter + fixed Taylor-type rule`.",
        "- Learning-based benchmark: residual PPO, который получает filtered state, belief о stress regime и лагированную ставку.",
        "",
        "## Главный результат",
        "",
        (
            f"- Наиболее благоприятный сценарий для RL: `{best['scenario_label']}` "
            f"с delta cumulative loss `{best['delta_cumulative_policy_loss_rl_minus_classical']:.4e}`."
            if not all_nonnegative
            else f"- Сценарий с минимальным ухудшением RL относительно classical: `{best['scenario_label']}` "
            f"с delta cumulative loss `{best['delta_cumulative_policy_loss_rl_minus_classical']:.4e}`."
        ),
        f"- Наименее благоприятный сценарий: `{worst['scenario_label']}` с delta cumulative loss `{worst['delta_cumulative_policy_loss_rl_minus_classical']:.4e}`.",
        "",
        "## Seeds",
        "",
    ]
    for _, row in training_seed_summary.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: training seed `{int(row['training_seed'])}`, best validation return `{row['best_validation_return']:.4e}`."
        )
    lines.extend([
        "",
        "## Ограничение",
        "",
        "- Это RL поверх regime-switching reduced-state HANK overlay, а не обучение на новой структурной full HANK solution с эндогенными режимами.",
    ])
    (output_dir / "report_stage6_regime_learning_hank.md").write_text("\n".join(lines))


def run_pipeline(
    config: RegimeLearningConfig | None = None,
    output_dir: str | None = None,
    scenario_names: list[str] | None = None,
):
    config = default_regime_learning_config() if config is None else config
    if output_dir is not None:
        config = replace(config, output_dir=output_dir)

    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, config.regime_config.partial_config)

    selected_variants = config.main_variants()
    if scenario_names is not None:
        names = set(scenario_names)
        selected_variants = [variant for variant in selected_variants if variant.name in names]

    _save_json(root / "stage6_config.json", config.to_dict())
    _save_json(root / "reduced_state_space.json", _reduced_state_payload(reduced_model))
    _save_json(root / "scenario_spec.json", [build_scenario_spec(config, variant).to_dict() for variant in selected_variants])

    training_histories = []
    training_seed_rows = []
    selected_policy_rows = []
    policy_metric_rows = []
    policy_path_frames = []
    model_specs = {}

    for variant in selected_variants:
        scenario_spec = build_scenario_spec(config, variant)
        regime_model = build_regime_switching_model(reduced_model, config.regime_config, scenario_spec.gap_scale)
        model_specs[variant.name] = regime_model_spec_payload(
            regime_model,
            {
                "name": scenario_spec.scenario_name,
                "label": scenario_spec.scenario_label,
                "gap_name": scenario_spec.gap_name,
                "gap_scale": scenario_spec.gap_scale,
                "info_scenario_name": scenario_spec.scenario_name.split("_", 1)[0],
                "noisy_observations": list(scenario_spec.noisy_observations),
            },
        )

        def env_factory():
            return RegimeSwitchingPolicyEnvironment(
                model=regime_model,
                regime_config=config.regime_config,
                scenario_spec=scenario_spec,
                phi_pi=hank_config.phi_pi,
                phi_y=hank_config.phi_y,
                rho_i=hank_config.rho_i,
            )

        if config.classical_policy_mode == "normal_only":
            classical_policy = MisspecifiedClassicalRulePolicy()
            classical_label = "Classical: normal-only filter + fixed rule"
        else:
            classical_policy = ClassicalFilteredRulePolicy(action_bound=config.action_bound)
            classical_label = "Classical: switching filter + fixed rule"
        full_information_policy = FullInformationRulePolicy(action_bound=config.action_bound)

        best_entry = None
        for training_seed in config.training_seeds:
            trained_policy, history, checkpoints = train_ppo_policy(
                env_factory=env_factory,
                ppo_config=config.ppo,
                action_bound=config.action_bound,
                gamma=config.gamma,
                training_seed=int(training_seed),
                label=variant.name,
            )
            training_histories.append(history)
            checkpoint_candidates = checkpoints or []
            checkpoint_candidates.append(
                type("FinalCheckpoint", (), {
                    "iteration": int(history["iteration"].iloc[-1]) if not history.empty else -1,
                    "policy": trained_policy,
                    "validation_return": float(history["validation_return"].iloc[-1]) if not history.empty else 0.0,
                    "mean_episode_return": float(history["mean_episode_return"].iloc[-1]) if not history.empty else 0.0,
                })()
            )
            for checkpoint in checkpoint_candidates:
                selection_summary = _evaluate_checkpoint_selection(
                    env_factory=env_factory,
                    policy=checkpoint.policy,
                    classical_policy=classical_policy,
                    classical_label=classical_label,
                    scenario_spec=scenario_spec,
                    selection_seeds=config.selection_seeds,
                )
                candidate = {
                    "variant_name": variant.name,
                    "scenario_name": scenario_spec.scenario_name,
                    "scenario_label": scenario_spec.scenario_label,
                    "training_seed": int(training_seed),
                    "checkpoint_iteration": int(checkpoint.iteration),
                    "validation_return": float(checkpoint.validation_return),
                    "mean_episode_return": float(checkpoint.mean_episode_return),
                    **selection_summary,
                    "policy": checkpoint.policy,
                }
                if best_entry is None or _selection_key(candidate) < _selection_key(best_entry):
                    best_entry = candidate

        assert best_entry is not None
        selected_policy = best_entry["policy"]
        selected_policy_rows.append({key: value for key, value in best_entry.items() if key != "policy"})
        training_seed_rows.append({
            "variant_name": variant.name,
            "training_seed": int(best_entry["training_seed"]),
            "best_validation_return": float(best_entry["validation_return"]),
            "final_validation_return": float(best_entry["validation_return"]),
            "final_mean_episode_return": float(best_entry["mean_episode_return"]),
        })

        for evaluation_seed in config.evaluation_seeds:
            classical_trace = simulate_policy_episode(
                env_factory=env_factory,
                policy=classical_policy,
                scenario_spec=scenario_spec,
                evaluation_seed=int(evaluation_seed),
                policy_name="classical_filtered_rule",
                policy_label=classical_label,
                training_seed=None,
            )
            rl_trace = simulate_policy_episode(
                env_factory=env_factory,
                policy=selected_policy,
                scenario_spec=scenario_spec,
                evaluation_seed=int(evaluation_seed),
                policy_name="learning_policy",
                policy_label="Learning-based policy",
                training_seed=int(best_entry["training_seed"]),
            )
            full_trace = simulate_policy_episode(
                env_factory=env_factory,
                policy=full_information_policy,
                scenario_spec=scenario_spec,
                evaluation_seed=int(evaluation_seed),
                policy_name="full_information_rule",
                policy_label="Полная информация",
                training_seed=None,
            )
            for trace, reference in (
                (classical_trace, full_trace),
                (rl_trace, classical_trace),
                (full_trace, full_trace),
            ):
                metrics, path_frame = evaluate_policy_trace(
                    policy_trace=trace,
                    reference_trace=reference,
                    scenario_spec=scenario_spec,
                )
                policy_metric_rows.append(metrics)
                policy_path_frames.append(path_frame)

    training_history = pd.concat(training_histories, ignore_index=True) if training_histories else pd.DataFrame()
    training_seed_summary = summarize_training_history(training_history)
    best_rows = pd.DataFrame(training_seed_rows)
    if not best_rows.empty:
        training_seed_summary = training_seed_summary.merge(
            best_rows,
            on=["variant_name", "training_seed"],
            how="outer",
            suffixes=("", "_selected"),
        )
    policy_metrics = pd.DataFrame(policy_metric_rows)
    policy_paths = pd.concat(policy_path_frames, ignore_index=True) if policy_path_frames else pd.DataFrame()
    policy_comparison = build_policy_comparison(policy_metrics)
    selected_policy_summary = pd.DataFrame(selected_policy_rows)

    _save_json(root / "policy_spec.json", {
        "policy_family": "residual_ppo_over_switching_filtered_rule",
        "classical_policy_mode": config.classical_policy_mode,
        "classical_reference_rule": {
            "phi_pi": float(hank_config.phi_pi),
            "phi_y": float(hank_config.phi_y),
            "rho_i": float(hank_config.rho_i),
        },
        "rl_input": "(filtered reduced HANK state, stress belief, lagged rate)",
        "reward": "-(pi^2 + lambda_y * y_gap^2 + lambda_i * Delta i^2)",
        "lambda_y": float(config.lambda_y),
        "lambda_i": float(config.lambda_i),
        "gamma": float(config.gamma),
        "action_bound": float(config.action_bound),
        "ppo": config.ppo.to_dict(),
    })
    _save_json(root / "regime_model_spec.json", model_specs)

    training_history.to_csv(root / "training_history.csv", index=False)
    training_seed_summary.to_csv(root / "training_seed_summary.csv", index=False)
    selected_policy_summary.to_csv(root / "selected_policy_summary.csv", index=False)
    policy_metrics.to_csv(root / "policy_metrics.csv", index=False)
    policy_paths.to_csv(root / "policy_paths.csv", index=False)
    policy_comparison.to_csv(root / "policy_comparison.csv", index=False)

    _write_report(root, training_seed_summary, policy_comparison)
    return {
        "training_history": training_history,
        "training_seed_summary": training_seed_summary,
        "selected_policy_summary": selected_policy_summary,
        "policy_metrics": policy_metrics,
        "policy_paths": policy_paths,
        "policy_comparison": policy_comparison,
    }
