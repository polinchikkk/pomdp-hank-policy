from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_learning_policy_baseline.config import PPOConfig
from hank_learning_policy_baseline.policies import FullInformationRulePolicy
from hank_learning_policy_baseline.ppo import train_ppo_policy
from hank_partial_info_baseline.state_space import fit_reduced_state_space
from regime_switching_baseline.regime_model import (
    RegimeSwitchingConfig,
    build_regime_switching_model,
    regime_model_spec_payload,
)

from .config import RegimeLearningConfig, RegimeLearningVariant
from .environment import RegimeSwitchingPolicyEnvironment, build_scenario_spec
from .evaluation import (
    build_policy_comparison,
    evaluate_policy_trace,
    simulate_policy_episode,
    summarize_training_history,
)
from .pipeline import _evaluate_checkpoint_selection
from .policies import MisspecifiedClassicalRulePolicy


@dataclass(frozen=True)
class UniversalTuningCandidate:
    name: str
    action_bound: float
    ppo: PPOConfig
    description: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["ppo"] = self.ppo.to_dict()
        return payload


def extreme_sticky_regime_config() -> RegimeSwitchingConfig:
    return RegimeSwitchingConfig(
        regime_transition=((0.975, 0.025), (0.02, 0.98)),
        moderate_gap_scale=1.0,
        strong_gap_scale=3.0,
        stress_inflation_row_factor=1.18,
        stress_output_row_factor=1.30,
        stress_low_liquidity_row_factor=1.45,
        stress_mean_mpc_row_factor=1.35,
        stress_inflation_control_factor=1.40,
        stress_output_control_factor=1.80,
        stress_low_liquidity_control_factor=2.50,
        stress_mean_mpc_control_factor=2.00,
        stress_macro_noise_factor=1.40,
        stress_distribution_noise_factor=2.30,
    )


def raw_observation_variants_2x2() -> list[RegimeLearningVariant]:
    return [
        RegimeLearningVariant(
            name="macro_core_moderate_gap_rawobs",
            scenario_name="macro_core_moderate_gap",
            scenario_label="Фильтрация: инфляция, выпуск и ставка × умеренный режимный разрыв",
            input_mode="raw_observations",
            include_distributional_state=True,
            description="Raw-observation RL with macro_core information.",
        ),
        RegimeLearningVariant(
            name="macro_core_strong_gap_rawobs",
            scenario_name="macro_core_strong_gap",
            scenario_label="Фильтрация: инфляция, выпуск и ставка × сильный режимный разрыв",
            input_mode="raw_observations",
            include_distributional_state=True,
            description="Raw-observation RL with macro_core information under strong regime gap.",
        ),
        RegimeLearningVariant(
            name="thin_information_moderate_gap_rawobs",
            scenario_name="thin_information_moderate_gap",
            scenario_label="Фильтрация: инфляция и ставка × умеренный режимный разрыв",
            input_mode="raw_observations",
            include_distributional_state=True,
            description="Raw-observation RL with thin information.",
        ),
        RegimeLearningVariant(
            name="thin_information_strong_gap_rawobs",
            scenario_name="thin_information_strong_gap",
            scenario_label="Фильтрация: инфляция и ставка × сильный режимный разрыв",
            input_mode="raw_observations",
            include_distributional_state=True,
            description="Raw-observation RL with thin information under strong regime gap.",
        ),
    ]


def default_universal_tuning_candidates() -> list[UniversalTuningCandidate]:
    baseline = PPOConfig()
    return [
        UniversalTuningCandidate(
            name="baseline_rawobs",
            action_bound=0.0030,
            ppo=baseline,
            description="Current raw-observation baseline.",
        ),
        UniversalTuningCandidate(
            name="wider_actions",
            action_bound=0.0040,
            ppo=replace(
                baseline,
                rollout_episodes=12,
                num_iterations=32,
                actor_learning_rate=4.0e-4,
                critic_learning_rate=7.0e-4,
                initial_log_std=-8.8,
            ),
            description="Slightly wider action space and longer training horizon.",
        ),
        UniversalTuningCandidate(
            name="robust_rollout",
            action_bound=0.0030,
            ppo=replace(
                baseline,
                rollout_episodes=16,
                num_iterations=32,
                num_epochs=10,
                minibatch_size=256,
                actor_learning_rate=3.0e-4,
                critic_learning_rate=6.0e-4,
                entropy_coefficient=5.0e-4,
                initial_log_std=-9.2,
            ),
            description="More data per update and more conservative optimization.",
        ),
        UniversalTuningCandidate(
            name="larger_network",
            action_bound=0.0035,
            ppo=replace(
                baseline,
                hidden_dim_1=48,
                hidden_dim_2=48,
                rollout_episodes=12,
                num_iterations=36,
                actor_learning_rate=4.0e-4,
                critic_learning_rate=7.0e-4,
                entropy_coefficient=1.5e-3,
                initial_log_std=-8.8,
            ),
            description="Wider network with slightly more exploration and training steps.",
        ),
        UniversalTuningCandidate(
            name="cautious_policy",
            action_bound=0.00225,
            ppo=replace(
                baseline,
                rollout_episodes=12,
                num_iterations=32,
                actor_learning_rate=3.0e-4,
                critic_learning_rate=6.0e-4,
                entropy_coefficient=5.0e-4,
                initial_log_std=-9.5,
            ),
            description="Smaller action range and tighter exploration for stable control.",
        ),
    ]


def default_universal_candidate_lookup() -> dict[str, UniversalTuningCandidate]:
    return {candidate.name: candidate for candidate in default_universal_tuning_candidates()}


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _selection_key(row: dict[str, float | int]) -> tuple[float, float, int]:
    return (
        float(row["selection_objective"]),
        float(row["selection_rate_rmse"]),
        int(row["selection_unstable_episodes"]),
    )


def _run_single_variant(
    *,
    root: Path,
    config: RegimeLearningConfig,
    variant: RegimeLearningVariant,
    reduced_model,
    hank_config,
) -> dict[str, pd.DataFrame]:
    scenario_spec = build_scenario_spec(config, variant)
    regime_model = build_regime_switching_model(reduced_model, config.regime_config, scenario_spec.gap_scale)

    def env_factory():
        return RegimeSwitchingPolicyEnvironment(
            model=regime_model,
            regime_config=config.regime_config,
            scenario_spec=scenario_spec,
            phi_pi=hank_config.phi_pi,
            phi_y=hank_config.phi_y,
            rho_i=hank_config.rho_i,
        )

    root.mkdir(parents=True, exist_ok=True)
    _save_json(root / "scenario_spec.json", scenario_spec.to_dict())
    _save_json(
        root / "regime_model_spec.json",
        regime_model_spec_payload(
            regime_model,
            {
                "name": scenario_spec.scenario_name,
                "label": scenario_spec.scenario_label,
                "gap_name": scenario_spec.gap_name,
                "gap_scale": scenario_spec.gap_scale,
                "info_scenario_name": scenario_spec.scenario_name.split("_", 1)[0],
                "noisy_observations": list(scenario_spec.noisy_observations),
            },
        ),
    )

    classical_policy = MisspecifiedClassicalRulePolicy()
    classical_label = "Classical: normal-only filter + fixed rule"
    full_information_policy = FullInformationRulePolicy(action_bound=config.action_bound)

    training_histories = []
    training_seed_rows = []
    selected_rows = []
    policy_metric_rows = []
    policy_path_frames = []

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
            type(
                "FinalCheckpoint",
                (),
                {
                    "iteration": int(history["iteration"].iloc[-1]) if not history.empty else -1,
                    "policy": trained_policy,
                    "validation_return": float(history["validation_return"].iloc[-1]) if not history.empty else 0.0,
                    "mean_episode_return": float(history["mean_episode_return"].iloc[-1]) if not history.empty else 0.0,
                },
            )()
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
    selected_rows.append({key: value for key, value in best_entry.items() if key != "policy"})
    training_seed_rows.append(
        {
            "variant_name": variant.name,
            "training_seed": int(best_entry["training_seed"]),
            "best_validation_return": float(best_entry["validation_return"]),
            "final_validation_return": float(best_entry["validation_return"]),
            "final_mean_episode_return": float(best_entry["mean_episode_return"]),
        }
    )

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

    training_history = pd.concat(training_histories, ignore_index=True)
    training_seed_summary = summarize_training_history(training_history)
    training_seed_summary = training_seed_summary.merge(
        pd.DataFrame(training_seed_rows),
        on=["variant_name", "training_seed"],
        how="outer",
        suffixes=("", "_selected"),
    )
    selected_policy_summary = pd.DataFrame(selected_rows)
    policy_metrics = pd.DataFrame(policy_metric_rows)
    policy_paths = pd.concat(policy_path_frames, ignore_index=True)
    policy_comparison = build_policy_comparison(policy_metrics)

    training_history.to_csv(root / "training_history.csv", index=False)
    training_seed_summary.to_csv(root / "training_seed_summary.csv", index=False)
    selected_policy_summary.to_csv(root / "selected_policy_summary.csv", index=False)
    policy_metrics.to_csv(root / "policy_metrics.csv", index=False)
    policy_paths.to_csv(root / "policy_paths.csv", index=False)
    policy_comparison.to_csv(root / "policy_comparison.csv", index=False)
    return {
        "training_history": training_history,
        "training_seed_summary": training_seed_summary,
        "selected_policy_summary": selected_policy_summary,
        "policy_metrics": policy_metrics,
        "policy_comparison": policy_comparison,
    }


def run_universal_rawobs_misspecified_tuning(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_universal_tuning",
    candidates: list[UniversalTuningCandidate] | None = None,
    regime_config: RegimeSwitchingConfig | None = None,
    variants: list[RegimeLearningVariant] | None = None,
    training_seeds: tuple[int, ...] = (11, 22),
    selection_seeds: tuple[int, ...] = (500, 501),
    evaluation_seeds: tuple[int, ...] = (700, 701, 702, 703, 704),
    horizon: int = 60,
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    regime_config = extreme_sticky_regime_config() if regime_config is None else regime_config
    candidates = default_universal_tuning_candidates() if candidates is None else candidates
    variants = raw_observation_variants_2x2() if variants is None else variants

    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, regime_config.partial_config)

    _save_json(root / "candidate_grid.json", {"candidates": [candidate.to_dict() for candidate in candidates]})

    all_rows = []
    for candidate in candidates:
        candidate_root = root / candidate.name
        candidate_root.mkdir(parents=True, exist_ok=True)
        candidate_config = RegimeLearningConfig(
            output_dir=str(candidate_root),
            horizon=horizon,
            action_bound=candidate.action_bound,
            classical_policy_mode="normal_only",
            training_seeds=training_seeds,
            selection_seeds=selection_seeds,
            evaluation_seeds=evaluation_seeds,
            regime_config=regime_config,
            ppo=candidate.ppo,
        )
        _save_json(root / candidate.name / "candidate_spec.json", candidate.to_dict())
        _save_json(root / candidate.name / "stage6_config.json", candidate_config.to_dict())

        for variant in variants:
            variant_root = candidate_root / variant.name
            results = _run_single_variant(
                root=variant_root,
                config=candidate_config,
                variant=variant,
                reduced_model=reduced_model,
                hank_config=hank_config,
            )
            comparison_row = results["policy_comparison"].iloc[0].to_dict()
            selected_row = results["selected_policy_summary"].iloc[0].to_dict()
            all_rows.append(
                {
                    "candidate_name": candidate.name,
                    "candidate_description": candidate.description,
                    "action_bound": candidate.action_bound,
                    "scenario_name": comparison_row["scenario_name"],
                    "scenario_label": comparison_row["scenario_label"],
                    "delta_cumulative_policy_loss_rl_minus_classical": comparison_row[
                        "delta_cumulative_policy_loss_rl_minus_classical"
                    ],
                    "delta_mean_policy_loss_rl_minus_classical": comparison_row[
                        "delta_mean_policy_loss_rl_minus_classical"
                    ],
                    "classical_mean_policy_loss": comparison_row["classical_mean_policy_loss"],
                    "rl_mean_policy_loss": comparison_row["rl_mean_policy_loss"],
                    "classical_policy_rate_rmse": comparison_row["classical_policy_rate_rmse"],
                    "rl_policy_rate_rmse": comparison_row["rl_policy_rate_rmse"],
                    "classical_regime_accuracy": comparison_row["classical_regime_accuracy"],
                    "rl_regime_accuracy": comparison_row["rl_regime_accuracy"],
                    "selected_training_seed": int(selected_row["training_seed"]),
                    "selected_checkpoint_iteration": int(selected_row["checkpoint_iteration"]),
                }
            )

    scenario_results = pd.DataFrame(all_rows).sort_values(["candidate_name", "scenario_name"]).reset_index(drop=True)
    candidate_summary_rows = []
    for candidate_name, frame in scenario_results.groupby("candidate_name"):
        deltas = frame["delta_cumulative_policy_loss_rl_minus_classical"].to_numpy(dtype=float)
        relative_improvement = 100.0 * (
            frame["classical_mean_policy_loss"].to_numpy(dtype=float)
            - frame["rl_mean_policy_loss"].to_numpy(dtype=float)
        ) / frame["classical_mean_policy_loss"].to_numpy(dtype=float)
        rate_rmse_improvement = 100.0 * (
            frame["classical_policy_rate_rmse"].to_numpy(dtype=float)
            - frame["rl_policy_rate_rmse"].to_numpy(dtype=float)
        ) / frame["classical_policy_rate_rmse"].to_numpy(dtype=float)
        candidate_summary_rows.append(
            {
                "candidate_name": candidate_name,
                "action_bound": float(frame["action_bound"].iloc[0]),
                "num_wins": int(np.sum(deltas < 0.0)),
                "mean_delta_cumulative_loss": float(np.mean(deltas)),
                "median_delta_cumulative_loss": float(np.median(deltas)),
                "worst_case_delta_cumulative_loss": float(np.max(deltas)),
                "best_case_delta_cumulative_loss": float(np.min(deltas)),
                "mean_relative_loss_improvement_pct": float(np.mean(relative_improvement)),
                "mean_rate_rmse_improvement_pct": float(np.mean(rate_rmse_improvement)),
                "nonzero_checkpoint_count": int(np.sum(frame["selected_checkpoint_iteration"].to_numpy(dtype=int) != 0)),
            }
        )
    candidate_summary = pd.DataFrame(candidate_summary_rows).sort_values(
        ["mean_delta_cumulative_loss", "worst_case_delta_cumulative_loss"]
    ).reset_index(drop=True)
    best_candidate = candidate_summary.iloc[0]["candidate_name"]
    best_map = scenario_results[scenario_results["candidate_name"] == best_candidate].copy()

    scenario_results.to_csv(root / "scenario_results.csv", index=False)
    candidate_summary.to_csv(root / "candidate_summary.csv", index=False)
    best_map.to_csv(root / "best_candidate_core_map.csv", index=False)
    best_map_with_axes = best_map.assign(
        info_regime=best_map["scenario_name"].str.replace(r"_(moderate|strong)_gap$", "", regex=True),
        structure_regime=best_map["scenario_name"].str.extract(r"_(moderate|strong)_gap")[0] + "_gap",
    )
    best_map_with_axes.pivot(
        index="info_regime",
        columns="structure_regime",
        values="delta_cumulative_policy_loss_rl_minus_classical",
    ).to_csv(root / "best_candidate_delta_loss_matrix.csv")

    report_lines = [
        "# Universal Tuning For Raw-Observation RL",
        "",
        "Search setup: same PPO/action configuration is applied to the full 2x2 regime-switching map.",
        "",
        f"- Candidates evaluated: {len(candidates)}",
        f"- Best candidate by mean delta cumulative loss: `{best_candidate}`",
        f"- Mean delta cumulative loss: {candidate_summary.iloc[0]['mean_delta_cumulative_loss']:.6e}",
        f"- Worst-case delta cumulative loss: {candidate_summary.iloc[0]['worst_case_delta_cumulative_loss']:.6e}",
        f"- Wins across 4 cells: {int(candidate_summary.iloc[0]['num_wins'])}",
    ]
    (root / "report_universal_tuning.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "scenario_results": scenario_results,
        "candidate_summary": candidate_summary,
        "best_candidate_core_map": best_map,
    }


def run_best_candidate_validation_suite(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_validation_suite",
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    candidate_lookup = default_universal_candidate_lookup()
    candidates = [candidate_lookup["baseline_rawobs"], candidate_lookup["larger_network"]]

    persistent_regime_config = replace(
        extreme_sticky_regime_config(),
        regime_transition=((0.985, 0.015), (0.01, 0.99)),
    )

    validation_runs = [
        {
            "name": "oos_seeds",
            "description": "Same 2x2 map, but with unseen evaluation seeds.",
            "kwargs": {
                "evaluation_seeds": (900, 901, 902, 903, 904, 905, 906, 907, 908, 909),
            },
        },
        {
            "name": "long_horizon",
            "description": "Same environment, but longer policy horizon.",
            "kwargs": {
                "horizon": 90,
                "evaluation_seeds": (920, 921, 922, 923, 924, 925, 926, 927, 928, 929),
            },
        },
        {
            "name": "persistent_regimes",
            "description": "More persistent hidden regimes with unseen evaluation seeds.",
            "kwargs": {
                "regime_config": persistent_regime_config,
                "evaluation_seeds": (940, 941, 942, 943, 944, 945, 946, 947, 948, 949),
            },
        },
    ]

    summary_rows: list[dict[str, object]] = []
    for run in validation_runs:
        run_root = root / run["name"]
        results = run_universal_rawobs_misspecified_tuning(
            output_dir=str(run_root),
            candidates=candidates,
            training_seeds=(11, 22),
            selection_seeds=(500, 501),
            **run["kwargs"],
        )
        candidate_summary = results["candidate_summary"]
        best_map = results["best_candidate_core_map"]
        best_row = candidate_summary.iloc[0]
        summary_rows.append(
            {
                "validation_name": run["name"],
                "description": run["description"],
                "best_candidate": str(best_row["candidate_name"]),
                "mean_delta_cumulative_loss": float(best_row["mean_delta_cumulative_loss"]),
                "worst_case_delta_cumulative_loss": float(best_row["worst_case_delta_cumulative_loss"]),
                "mean_relative_loss_improvement_pct": float(best_row["mean_relative_loss_improvement_pct"]),
                "num_wins": int(best_row["num_wins"]),
                "nonzero_checkpoint_count": int(best_row["nonzero_checkpoint_count"]),
                "macro_core_moderate_gap_delta": float(
                    best_map.loc[
                        best_map["scenario_name"] == "macro_core_moderate_gap",
                        "delta_cumulative_policy_loss_rl_minus_classical",
                    ].iloc[0]
                ),
                "macro_core_strong_gap_delta": float(
                    best_map.loc[
                        best_map["scenario_name"] == "macro_core_strong_gap",
                        "delta_cumulative_policy_loss_rl_minus_classical",
                    ].iloc[0]
                ),
                "thin_information_moderate_gap_delta": float(
                    best_map.loc[
                        best_map["scenario_name"] == "thin_information_moderate_gap",
                        "delta_cumulative_policy_loss_rl_minus_classical",
                    ].iloc[0]
                ),
                "thin_information_strong_gap_delta": float(
                    best_map.loc[
                        best_map["scenario_name"] == "thin_information_strong_gap",
                        "delta_cumulative_policy_loss_rl_minus_classical",
                    ].iloc[0]
                ),
            }
        )

    validation_summary = pd.DataFrame(summary_rows)
    validation_summary.to_csv(root / "validation_summary.csv", index=False)
    report_lines = [
        "# Stage 6 Validation Suite",
        "",
        "The suite re-runs the best stage-6 raw-observation setup against the misspecified classical benchmark",
        "under out-of-sample seeds and mild environment shifts.",
        "",
    ]
    for row in validation_summary.to_dict(orient="records"):
        report_lines.extend(
            [
                f"## {row['validation_name']}",
                f"- Best candidate: `{row['best_candidate']}`",
                f"- Mean delta cumulative loss: {row['mean_delta_cumulative_loss']:.6e}",
                f"- Worst-case delta cumulative loss: {row['worst_case_delta_cumulative_loss']:.6e}",
                f"- Mean relative loss improvement: {row['mean_relative_loss_improvement_pct']:.2f}%",
                f"- Wins across 4 cells: {int(row['num_wins'])}",
                "",
            ]
        )
    (root / "report_validation_suite.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {"validation_summary": validation_summary}
