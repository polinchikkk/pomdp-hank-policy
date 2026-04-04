from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_learning_policy_baseline.policies import BasePolicy, ClassicalFilteredRulePolicy
from hank_learning_policy_baseline.ppo import train_ppo_policy
from hank_partial_info_baseline.state_space import fit_reduced_state_space
from regime_switching_baseline.regime_model import RegimeSwitchingConfig, build_regime_switching_model

from .config import RegimeLearningConfig, RegimeLearningVariant
from .environment import RegimeSwitchingPolicyEnvironment, build_scenario_spec
from .evaluation import _is_unstable, simulate_policy_episode
from .pipeline import _evaluate_checkpoint_selection
from .tuning import default_universal_candidate_lookup, extreme_sticky_regime_config


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


@dataclass(frozen=True)
class EnvironmentShiftSpec:
    name: str
    label: str
    description: str
    regime_config: RegimeSwitchingConfig

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "regime_config": self.regime_config.to_dict(),
        }


class FilteredSimpleRulePolicy(BasePolicy):
    def __init__(self, *, phi_pi: float, phi_y: float, rho_i: float) -> None:
        self.phi_pi = float(phi_pi)
        self.phi_y = float(phi_y)
        self.rho_i = float(rho_i)

    def rate(self, observation: np.ndarray, info: dict) -> float:
        state = np.asarray(info["filtered_state"], dtype=float)
        state_names = tuple(info["state_names"])
        index = {name: idx for idx, name in enumerate(state_names)}
        prev_rate = float(info["current_rate"])
        lower, upper = info["rate_bounds"]
        rule_term = (
            state[index["rstar_gap"]]
            + self.phi_pi * state[index["inflation_gap"]]
            + self.phi_y * state[index["output_gap"]]
        )
        rate = self.rho_i * prev_rate + (1.0 - self.rho_i) * rule_term
        return float(np.clip(rate, lower, upper))


def _environment_shift_specs() -> list[EnvironmentShiftSpec]:
    baseline = extreme_sticky_regime_config()
    shifted_transmission = replace(
        baseline,
        stress_inflation_row_factor=1.24,
        stress_output_row_factor=1.42,
        stress_inflation_control_factor=1.55,
        stress_output_control_factor=2.15,
        output_to_inflation_link=0.045,
    )
    stronger_distributional_channel = replace(
        baseline,
        stress_low_liquidity_row_factor=1.70,
        stress_mean_mpc_row_factor=1.55,
        stress_low_liquidity_control_factor=3.10,
        stress_mean_mpc_control_factor=2.40,
        output_to_low_liquidity_link=-0.13,
        output_to_mean_mpc_link=-0.10,
        stress_distribution_noise_factor=2.80,
    )
    persistent_regimes = replace(
        baseline,
        regime_transition=((0.988, 0.012), (0.015, 0.985)),
    )
    return [
        EnvironmentShiftSpec(
            name="baseline_environment",
            label="Baseline regime environment",
            description="The same extreme-sticky regime-switching environment used for stage-6 architecture ablation.",
            regime_config=baseline,
        ),
        EnvironmentShiftSpec(
            name="shifted_transmission",
            label="Shifted macro transmission",
            description="Higher sensitivity of inflation and output dynamics to policy shocks in the stress regime.",
            regime_config=shifted_transmission,
        ),
        EnvironmentShiftSpec(
            name="stronger_distributional_channel",
            label="Stronger distributional channel",
            description="Amplified low-liquidity and mean-MPC transmission in the stress regime.",
            regime_config=stronger_distributional_channel,
        ),
        EnvironmentShiftSpec(
            name="persistent_regimes",
            label="More persistent hidden regimes",
            description="Regime switches become rarer and stress spells last longer.",
            regime_config=persistent_regimes,
        ),
    ]


def _scenario_architecture_lookup(architecture_dir: Path) -> dict[str, str]:
    comparison = pd.read_csv(architecture_dir / "architecture_comparison.csv")
    lookup = {}
    for row in comparison.to_dict(orient="records"):
        lookup[row["scenario_name"]] = (
            "raw_observations"
            if float(row["rawobs_mean_cumulative_loss"]) < float(row["belief_mean_cumulative_loss"])
            else "belief_state"
        )
    return lookup


def _variant_for_scenario(scenario_name: str, input_mode: str) -> RegimeLearningVariant:
    labels = {
        "macro_core_moderate_gap": "Инфляция, выпуск, ставка × умеренный режимный разрыв",
        "macro_core_strong_gap": "Инфляция, выпуск, ставка × сильный режимный разрыв",
        "thin_information_moderate_gap": "Инфляция, ставка × умеренный режимный разрыв",
        "thin_information_strong_gap": "Инфляция, ставка × сильный режимный разрыв",
    }
    return RegimeLearningVariant(
        name=f"{scenario_name}_{input_mode}",
        scenario_name=scenario_name,
        scenario_label=labels[scenario_name],
        input_mode=input_mode,
        include_distributional_state=True,
        description="Best learned architecture from stage-6 architecture ablation.",
    )


def _load_retuned_rule_lookup(retuned_csv: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(retuned_csv)
    return {
        row["scenario_name"]: {
            "phi_pi": float(row["phi_pi"]),
            "phi_y": float(row["phi_y"]),
            "rho_i": float(row["rho_i"]),
        }
        for row in frame.to_dict(orient="records")
    }


def _make_env_factory(
    *,
    reduced_model,
    hank_config,
    regime_config: RegimeSwitchingConfig,
    scenario_name: str,
    input_mode: str,
    action_bound: float,
):
    variant = _variant_for_scenario(scenario_name, input_mode)
    config = RegimeLearningConfig(
        action_bound=action_bound,
        classical_policy_mode="switching",
        regime_config=regime_config,
    )
    scenario_spec = build_scenario_spec(config, variant)
    model = build_regime_switching_model(reduced_model, regime_config, scenario_spec.gap_scale)

    def env_factory():
        return RegimeSwitchingPolicyEnvironment(
            model=model,
            regime_config=regime_config,
            scenario_spec=scenario_spec,
            phi_pi=hank_config.phi_pi,
            phi_y=hank_config.phi_y,
            rho_i=hank_config.rho_i,
        )

    return env_factory, scenario_spec


def _selection_key(row: dict[str, float | int]) -> tuple[float, float, int]:
    return (
        float(row["selection_objective"]),
        float(row["selection_rate_rmse"]),
        int(row["selection_unstable_episodes"]),
    )


def _train_baseline_learned_policies(
    *,
    root: Path,
    scenario_architecture: dict[str, str],
    baseline_environment: EnvironmentShiftSpec,
    candidate,
    training_seeds: tuple[int, ...],
    selection_seeds: tuple[int, ...],
) -> tuple[dict[str, BasePolicy], pd.DataFrame, pd.DataFrame]:
    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, baseline_environment.regime_config.partial_config)

    selected_policies: dict[str, BasePolicy] = {}
    selected_rows = []
    training_rows = []

    for scenario_name, input_mode in scenario_architecture.items():
        env_factory, scenario_spec = _make_env_factory(
            reduced_model=reduced_model,
            hank_config=hank_config,
            regime_config=baseline_environment.regime_config,
            scenario_name=scenario_name,
            input_mode=input_mode,
            action_bound=candidate.action_bound,
        )
        classical_policy = ClassicalFilteredRulePolicy(action_bound=candidate.action_bound)

        best_entry = None
        for training_seed in training_seeds:
            trained_policy, history, checkpoints = train_ppo_policy(
                env_factory=env_factory,
                ppo_config=candidate.ppo,
                action_bound=candidate.action_bound,
                gamma=0.99,
                training_seed=int(training_seed),
                label=f"{scenario_name}_{input_mode}",
            )
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
                    classical_label="Filter + fixed rule",
                    scenario_spec=scenario_spec,
                    selection_seeds=selection_seeds,
                )
                candidate_row = {
                    "scenario_name": scenario_name,
                    "input_mode": input_mode,
                    "training_seed": int(training_seed),
                    "checkpoint_iteration": int(checkpoint.iteration),
                    "validation_return": float(checkpoint.validation_return),
                    "mean_episode_return": float(checkpoint.mean_episode_return),
                    **selection_summary,
                    "policy": checkpoint.policy,
                }
                if best_entry is None or _selection_key(candidate_row) < _selection_key(best_entry):
                    best_entry = candidate_row
        assert best_entry is not None
        selected_policies[scenario_name] = best_entry["policy"]
        selected_rows.append({key: value for key, value in best_entry.items() if key != "policy"})
        training_rows.append(
            {
                "scenario_name": scenario_name,
                "input_mode": input_mode,
                "training_seed": int(best_entry["training_seed"]),
                "checkpoint_iteration": int(best_entry["checkpoint_iteration"]),
                "selection_objective": float(best_entry["selection_objective"]),
                "selection_rate_rmse": float(best_entry["selection_rate_rmse"]),
            }
        )

    selected_summary = pd.DataFrame(selected_rows).sort_values("scenario_name").reset_index(drop=True)
    training_summary = pd.DataFrame(training_rows).sort_values("scenario_name").reset_index(drop=True)
    selected_summary.to_csv(root / "selected_learned_policy_summary.csv", index=False)
    training_summary.to_csv(root / "selected_learned_training_summary.csv", index=False)
    return selected_policies, selected_summary, training_summary


def run_environment_shift(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_environment_shift",
    architecture_dir: str = "outputs/hank_regime_learning_stage6_architecture_ablation",
    retuned_csv: str = "outputs/hank_regime_learning_stage6_deep_validation/retuned_classical/retuned_classical_best.csv",
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    architecture_root = Path(architecture_dir)
    retuned_path = Path(retuned_csv)

    scenario_architecture = _scenario_architecture_lookup(architecture_root)
    candidate = default_universal_candidate_lookup()["larger_network"]
    environments = _environment_shift_specs()
    baseline_environment = environments[0]
    retuned_lookup = _load_retuned_rule_lookup(retuned_path)

    _save_json(
        root / "environment_shift_spec.json",
        {
            "candidate_name": candidate.name,
            "candidate_description": candidate.description,
            "action_bound": candidate.action_bound,
            "training_seeds": [11],
            "selection_seeds": [500, 501],
            "evaluation_seeds": list(range(900, 910)),
            "scenario_architecture": scenario_architecture,
            "environments": [env.to_dict() for env in environments],
            "retuned_rule_source": str(retuned_path),
            "retuned_rule_lookup": retuned_lookup,
        },
    )

    learned_policies, selected_summary, training_summary = _train_baseline_learned_policies(
        root=root,
        scenario_architecture=scenario_architecture,
        baseline_environment=baseline_environment,
        candidate=candidate,
        training_seeds=(11,),
        selection_seeds=(500, 501),
    )

    hank_config = default_calibration()
    rows = []
    for env_spec in environments:
        bundle = solve_steady_state(hank_config)
        reduced_model = fit_reduced_state_space(bundle, hank_config, env_spec.regime_config.partial_config)
        for scenario_name, input_mode in scenario_architecture.items():
            env_factory, scenario_spec = _make_env_factory(
                reduced_model=reduced_model,
                hank_config=hank_config,
                regime_config=env_spec.regime_config,
                scenario_name=scenario_name,
                input_mode=input_mode,
                action_bound=candidate.action_bound,
            )
            fixed_policy = ClassicalFilteredRulePolicy(action_bound=candidate.action_bound)
            tuned_params = retuned_lookup[scenario_name]
            tuned_policy = FilteredSimpleRulePolicy(
                phi_pi=tuned_params["phi_pi"],
                phi_y=tuned_params["phi_y"],
                rho_i=tuned_params["rho_i"],
            )
            learned_policy = learned_policies[scenario_name]

            for evaluation_seed in range(900, 910):
                for policy_name, policy_label, policy in (
                    ("fixed_classical_rule", "Filter + fixed rule", fixed_policy),
                    ("retuned_simple_rule", "Filter + retuned simple rule", tuned_policy),
                    ("learned_policy", f"Learned policy ({input_mode})", learned_policy),
                ):
                    trace = simulate_policy_episode(
                        env_factory=env_factory,
                        policy=policy,
                        scenario_spec=scenario_spec,
                        evaluation_seed=int(evaluation_seed),
                        policy_name=policy_name,
                        policy_label=policy_label,
                        training_seed=11 if policy_name == "learned_policy" else None,
                    )
                    rows.append(
                        {
                            "environment_name": env_spec.name,
                            "environment_label": env_spec.label,
                            "scenario_name": scenario_name,
                            "scenario_label": scenario_spec.scenario_label,
                            "input_mode": input_mode,
                            "evaluation_seed": int(evaluation_seed),
                            "policy_name": policy_name,
                            "policy_label": policy_label,
                            "cumulative_loss": float(trace["loss"].sum()),
                            "mean_loss": float(trace["loss"].mean()),
                            "policy_volatility": float(np.std(trace["policy_rate"].to_numpy(dtype=float))),
                            "corner_share": float(
                                np.mean(
                                    np.isclose(
                                        np.abs(trace["policy_rate"].to_numpy(dtype=float)),
                                        scenario_spec.rate_bounds[1],
                                        atol=1.0e-10,
                                    )
                                )
                            ),
                            "unstable": int(_is_unstable(trace)),
                        }
                    )

    seed_level = pd.DataFrame(rows).sort_values(
        ["environment_name", "scenario_name", "policy_name", "evaluation_seed"]
    ).reset_index(drop=True)
    seed_level.to_csv(root / "environment_shift_seed_level.csv", index=False)

    summary_rows = []
    for (env_name, scenario_name), frame in seed_level.groupby(["environment_name", "scenario_name"]):
        env_label = frame["environment_label"].iloc[0]
        scenario_label = frame["scenario_label"].iloc[0]
        input_mode = frame["input_mode"].iloc[0]

        pivot = frame.pivot_table(
            index="evaluation_seed",
            columns="policy_name",
            values="cumulative_loss",
            aggfunc="first",
        )
        fixed_mean = float(pivot["fixed_classical_rule"].mean())
        tuned_mean = float(pivot["retuned_simple_rule"].mean())
        learned_mean = float(pivot["learned_policy"].mean())

        fixed_metrics = frame[frame["policy_name"] == "fixed_classical_rule"]
        tuned_metrics = frame[frame["policy_name"] == "retuned_simple_rule"]
        learned_metrics = frame[frame["policy_name"] == "learned_policy"]

        summary_rows.append(
            {
                "environment_name": env_name,
                "environment_label": env_label,
                "scenario_name": scenario_name,
                "scenario_label": scenario_label,
                "best_learned_input_mode": input_mode,
                "fixed_mean_cumulative_loss": fixed_mean,
                "retuned_mean_cumulative_loss": tuned_mean,
                "learned_mean_cumulative_loss": learned_mean,
                "learned_minus_fixed": learned_mean - fixed_mean,
                "learned_minus_retuned": learned_mean - tuned_mean,
                "learned_rel_improvement_vs_fixed_pct": float(100.0 * (fixed_mean - learned_mean) / fixed_mean),
                "learned_rel_improvement_vs_retuned_pct": float(100.0 * (tuned_mean - learned_mean) / tuned_mean),
                "learned_win_rate_vs_fixed": float(np.mean(pivot["learned_policy"] < pivot["fixed_classical_rule"])),
                "learned_win_rate_vs_retuned": float(np.mean(pivot["learned_policy"] < pivot["retuned_simple_rule"])),
                "fixed_mean_policy_volatility": float(fixed_metrics["policy_volatility"].mean()),
                "retuned_mean_policy_volatility": float(tuned_metrics["policy_volatility"].mean()),
                "learned_mean_policy_volatility": float(learned_metrics["policy_volatility"].mean()),
                "learned_any_unstable": int(learned_metrics["unstable"].max()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["environment_name", "scenario_name"]).reset_index(drop=True)
    summary.to_csv(root / "environment_shift_results.csv", index=False)

    baseline_lookup = {}
    for row in summary[summary["environment_name"] == "baseline_environment"].to_dict(orient="records"):
        baseline_lookup[row["scenario_name"]] = row

    degradation_rows = []
    for row in summary.to_dict(orient="records"):
        baseline_row = baseline_lookup[row["scenario_name"]]
        degradation_rows.append(
            {
                "environment_name": row["environment_name"],
                "environment_label": row["environment_label"],
                "scenario_name": row["scenario_name"],
                "scenario_label": row["scenario_label"],
                "fixed_degradation_vs_baseline": float(row["fixed_mean_cumulative_loss"] - baseline_row["fixed_mean_cumulative_loss"]),
                "retuned_degradation_vs_baseline": float(row["retuned_mean_cumulative_loss"] - baseline_row["retuned_mean_cumulative_loss"]),
                "learned_degradation_vs_baseline": float(row["learned_mean_cumulative_loss"] - baseline_row["learned_mean_cumulative_loss"]),
            }
        )
    degradation = pd.DataFrame(degradation_rows).sort_values(["environment_name", "scenario_name"]).reset_index(drop=True)
    degradation.to_csv(root / "environment_shift_degradation.csv", index=False)

    win_rows = []
    for env_name, frame in summary.groupby("environment_name"):
        win_rows.append(
            {
                "environment_name": env_name,
                "environment_label": frame["environment_label"].iloc[0],
                "mean_learned_improvement_vs_fixed_pct": float(frame["learned_rel_improvement_vs_fixed_pct"].mean()),
                "mean_learned_improvement_vs_retuned_pct": float(frame["learned_rel_improvement_vs_retuned_pct"].mean()),
                "scenario_win_share_vs_fixed": float(np.mean(frame["learned_minus_fixed"] < 0.0)),
                "scenario_win_share_vs_retuned": float(np.mean(frame["learned_minus_retuned"] < 0.0)),
                "mean_seed_win_rate_vs_fixed": float(frame["learned_win_rate_vs_fixed"].mean()),
                "mean_seed_win_rate_vs_retuned": float(frame["learned_win_rate_vs_retuned"].mean()),
            }
        )
    win_summary = pd.DataFrame(win_rows).sort_values("environment_name").reset_index(drop=True)
    win_summary.to_csv(root / "environment_shift_win_summary.csv", index=False)

    heatmap_fixed = summary.pivot(index="environment_label", columns="scenario_label", values="learned_rel_improvement_vs_fixed_pct")
    heatmap_retuned = summary.pivot(index="environment_label", columns="scenario_label", values="learned_rel_improvement_vs_retuned_pct")
    heatmap_fixed.to_csv(root / "environment_shift_heatmap_vs_fixed.csv")
    heatmap_retuned.to_csv(root / "environment_shift_heatmap_vs_retuned.csv")

    for suffix, heatmap, title in (
        ("fixed", heatmap_fixed, "Преимущество learned policy над fixed rule, %"),
        ("retuned", heatmap_retuned, "Преимущество learned policy над retuned simple rule, %"),
    ):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        image = ax.imshow(heatmap.to_numpy(dtype=float), cmap="RdYlGn", aspect="auto")
        ax.set_xticks(np.arange(len(heatmap.columns)))
        ax.set_xticklabels(list(heatmap.columns), rotation=25, ha="right")
        ax.set_yticks(np.arange(len(heatmap.index)))
        ax.set_yticklabels(list(heatmap.index))
        ax.set_title(title)
        for i in range(len(heatmap.index)):
            for j in range(len(heatmap.columns)):
                value = float(heatmap.iloc[i, j])
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="#1f1f1f", fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.85, label="% улучшения по cumulative loss")
        fig.tight_layout()
        fig.savefig(root / f"fig_environment_shift_vs_{suffix}.png", dpi=220)
        fig.savefig(root / f"fig_environment_shift_vs_{suffix}.pdf")
        plt.close(fig)

    report_lines = [
        "# Stage 6 Environment Shift",
        "",
        "В этой серии learned policy и baseline-tuned simple rule настраиваются на baseline regime environment, а затем без перенастройки оцениваются на новых структурных средах.",
        "",
        "## Selected Learned Architectures",
        "",
    ]
    for row in selected_summary.to_dict(orient="records"):
        report_lines.append(
            f"- `{row['scenario_name']}`: input mode `{scenario_architecture[row['scenario_name']]}`, checkpoint `{int(row['checkpoint_iteration'])}`, selection objective `{row['selection_objective']:.6e}`."
        )
    report_lines.extend(["", "## Transfer Summary", ""])
    for row in win_summary.to_dict(orient="records"):
        report_lines.extend(
            [
                f"### {row['environment_label']}",
                f"- Learned vs fixed: `{row['mean_learned_improvement_vs_fixed_pct']:.2f}%` в среднем.",
                f"- Learned vs retuned simple rule: `{row['mean_learned_improvement_vs_retuned_pct']:.2f}%` в среднем.",
                f"- Scenario win share vs fixed: `{row['scenario_win_share_vs_fixed']:.2f}`.",
                f"- Scenario win share vs retuned: `{row['scenario_win_share_vs_retuned']:.2f}`.",
                "",
            ]
        )
    (root / "report_environment_shift.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "selected_summary": selected_summary,
        "training_summary": training_summary,
        "seed_level": seed_level,
        "summary": summary,
        "degradation": degradation,
        "win_summary": win_summary,
    }
