from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_learning_policy_baseline.policies import BasePolicy
from hank_partial_info_baseline.config import STATE_NAMES
from hank_partial_info_baseline.state_space import fit_reduced_state_space
from regime_switching_baseline.regime_model import build_regime_switching_model

from .config import RegimeLearningConfig
from .environment import RegimeSwitchingPolicyEnvironment, build_scenario_spec
from .evaluation import simulate_policy_episode
from .tuning import (
    UniversalTuningCandidate,
    default_universal_candidate_lookup,
    extreme_sticky_regime_config,
    raw_observation_variants_2x2,
    run_universal_rawobs_misspecified_tuning,
)


class RetunedMisspecifiedRulePolicy(BasePolicy):
    def __init__(self, *, phi_pi: float, phi_y: float, rho_i: float) -> None:
        self.phi_pi = float(phi_pi)
        self.phi_y = float(phi_y)
        self.rho_i = float(rho_i)
        self._index = {name: idx for idx, name in enumerate(STATE_NAMES)}

    def rate(self, observation: np.ndarray, info: dict) -> float:
        state = np.asarray(info["misspecified_filtered_state"], dtype=float)
        lower, upper = info["rate_bounds"]
        prev_rate = float(info["current_rate"])
        rule_term = (
            state[self._index["rstar_gap"]]
            + self.phi_pi * state[self._index["inflation_gap"]]
            + self.phi_y * state[self._index["output_gap"]]
        )
        rate = self.rho_i * prev_rate + (1.0 - self.rho_i) * rule_term
        return float(np.clip(rate, lower, upper))


def _load_oos_base(base_dir: str | Path) -> Path:
    return Path(base_dir)


def _scenario_subdir_name(scenario_name: str) -> str:
    return f"{scenario_name}_rawobs"


def _bootstrap_ci(values: np.ndarray, *, num_bootstrap: int = 2000, seed: int = 123) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = []
    n = len(values)
    for _ in range(num_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boots.append(float(np.mean(sample)))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_dispersion_and_ci(
    *,
    base_dir: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = _load_oos_base(base_dir)

    rows: list[dict[str, object]] = []
    box_rows: list[pd.DataFrame] = []
    for candidate_name in ("baseline_rawobs", "larger_network"):
        for scenario_dir in sorted((base / candidate_name).glob("*_rawobs")):
            policy_metrics = pd.read_csv(scenario_dir / "policy_metrics.csv")
            pivot = policy_metrics.pivot_table(
                index="evaluation_seed",
                columns="policy_name",
                values="cumulative_policy_loss",
                aggfunc="first",
            )
            deltas = (pivot["learning_policy"] - pivot["classical_filtered_rule"]).to_numpy(dtype=float)
            mean_delta = float(np.mean(deltas))
            std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
            ci_low, ci_high = _bootstrap_ci(deltas)
            scenario_name = scenario_dir.name.replace("_rawobs", "")
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "scenario_name": scenario_name,
                    "num_evaluation_seeds": int(len(deltas)),
                    "mean_delta_cumulative_loss": mean_delta,
                    "std_delta_cumulative_loss": std_delta,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "share_negative_delta": float(np.mean(deltas < 0.0)),
                }
            )
            box_rows.append(
                pd.DataFrame(
                    {
                        "candidate_name": candidate_name,
                        "scenario_name": scenario_name,
                        "delta_cumulative_loss": deltas,
                    }
                )
            )

    summary = pd.DataFrame(rows).sort_values(["candidate_name", "scenario_name"]).reset_index(drop=True)
    box_data = pd.concat(box_rows, ignore_index=True)
    summary.to_csv(root / "delta_loss_dispersion.csv", index=False)
    box_data.to_csv(root / "delta_loss_by_seed.csv", index=False)

    order = [
        "macro_core_moderate_gap",
        "macro_core_strong_gap",
        "thin_information_moderate_gap",
        "thin_information_strong_gap",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    labels = []
    plot_data = []
    current = 1
    for scenario_name in order:
        for candidate_name in ("baseline_rawobs", "larger_network"):
            subset = box_data[
                (box_data["scenario_name"] == scenario_name) & (box_data["candidate_name"] == candidate_name)
            ]["delta_cumulative_loss"].to_numpy(dtype=float)
            if subset.size == 0:
                continue
            positions.append(current)
            labels.append(f"{scenario_name}\n{candidate_name}")
            plot_data.append(subset)
            current += 1
        current += 0.5
    bp = ax.boxplot(plot_data, positions=positions, widths=0.6, patch_artist=True)
    colors = ["#9ecae1", "#f28e2b"] * 4
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("RL минус classical: накопленная loss")
    ax.set_title("Разброс выигрыша RL по evaluation seeds")
    fig.tight_layout()
    fig.savefig(root / "fig_delta_loss_boxplot.png", dpi=220)
    fig.savefig(root / "fig_delta_loss_boxplot.pdf")
    plt.close(fig)
    return summary


def _build_regime_eval_objects(regime_config):
    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, regime_config.partial_config)
    return hank_config, reduced_model


def _evaluate_rule_grid_for_scenario(
    *,
    scenario_name: str,
    evaluation_seeds: tuple[int, ...],
    regime_config,
    action_bound: float,
    grid_phi_pi: tuple[float, ...],
    grid_phi_y: tuple[float, ...],
    grid_rho_i: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hank_config, reduced_model = _build_regime_eval_objects(regime_config)
    variants = {variant.scenario_name: variant for variant in raw_observation_variants_2x2()}
    variant = variants[scenario_name]
    config = RegimeLearningConfig(
        action_bound=action_bound,
        classical_policy_mode="normal_only",
        training_seeds=(11,),
        selection_seeds=(500,),
        evaluation_seeds=evaluation_seeds,
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

    rows = []
    for phi_pi in grid_phi_pi:
        for phi_y in grid_phi_y:
            for rho_i in grid_rho_i:
                policy = RetunedMisspecifiedRulePolicy(phi_pi=phi_pi, phi_y=phi_y, rho_i=rho_i)
                cumulative_losses = []
                mean_losses = []
                volatilities = []
                for seed in evaluation_seeds:
                    trace = simulate_policy_episode(
                        env_factory=env_factory,
                        policy=policy,
                        scenario_spec=scenario_spec,
                        evaluation_seed=int(seed),
                        policy_name="retuned_classical_rule",
                        policy_label="Retuned classical rule",
                        training_seed=None,
                    )
                    cumulative_losses.append(float(trace["loss"].sum()))
                    mean_losses.append(float(trace["loss"].mean()))
                    volatilities.append(float(np.std(trace["policy_rate"].to_numpy(dtype=float))))
                rows.append(
                    {
                        "scenario_name": scenario_name,
                        "phi_pi": float(phi_pi),
                        "phi_y": float(phi_y),
                        "rho_i": float(rho_i),
                        "mean_cumulative_loss": float(np.mean(cumulative_losses)),
                        "std_cumulative_loss": float(np.std(cumulative_losses, ddof=1)) if len(cumulative_losses) > 1 else 0.0,
                        "mean_policy_loss": float(np.mean(mean_losses)),
                        "mean_policy_volatility": float(np.mean(volatilities)),
                    }
                )
    grid = pd.DataFrame(rows).sort_values(["mean_cumulative_loss", "mean_policy_loss"]).reset_index(drop=True)
    best = grid.iloc[[0]].copy()
    return grid, best


def run_retuned_classical_benchmark(
    *,
    base_dir: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = _load_oos_base(base_dir)
    evaluation_seeds = tuple(range(900, 910))
    regime_config = extreme_sticky_regime_config()
    action_bound = default_universal_candidate_lookup()["larger_network"].action_bound

    grid_frames = []
    best_frames = []
    for scenario_name in (
        "macro_core_moderate_gap",
        "macro_core_strong_gap",
        "thin_information_moderate_gap",
        "thin_information_strong_gap",
    ):
        grid, best = _evaluate_rule_grid_for_scenario(
            scenario_name=scenario_name,
            evaluation_seeds=evaluation_seeds,
            regime_config=regime_config,
            action_bound=action_bound,
            grid_phi_pi=(1.0, 1.5, 2.0, 2.5),
            grid_phi_y=(0.0, 0.125, 0.25),
            grid_rho_i=(0.3, 0.5, 0.7, 0.85),
        )
        rl_metrics = pd.read_csv(base / "larger_network" / _scenario_subdir_name(scenario_name) / "policy_metrics.csv")
        classical_metrics = pd.read_csv(base / "baseline_rawobs" / _scenario_subdir_name(scenario_name) / "policy_metrics.csv")
        rl_mean_cum_loss = float(
            rl_metrics.loc[rl_metrics["policy_name"] == "learning_policy", "cumulative_policy_loss"].mean()
        )
        misspecified_mean_cum_loss = float(
            classical_metrics.loc[classical_metrics["policy_name"] == "classical_filtered_rule", "cumulative_policy_loss"].mean()
        )
        best = best.assign(
            rl_mean_cumulative_loss=rl_mean_cum_loss,
            misspecified_classical_mean_cumulative_loss=misspecified_mean_cum_loss,
            delta_rl_minus_retuned=float(rl_mean_cum_loss - best["mean_cumulative_loss"].iloc[0]),
            delta_retuned_minus_misspecified=float(best["mean_cumulative_loss"].iloc[0] - misspecified_mean_cum_loss),
        )
        grid_frames.append(grid)
        best_frames.append(best)

    grid_results = pd.concat(grid_frames, ignore_index=True)
    best_results = pd.concat(best_frames, ignore_index=True)
    grid_results.to_csv(root / "retuned_classical_grid_results.csv", index=False)
    best_results.to_csv(root / "retuned_classical_best.csv", index=False)
    return grid_results, best_results


def summarize_policy_sanity(
    *,
    base_dir: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = _load_oos_base(base_dir)
    rows = []
    for scenario_name in (
        "macro_core_moderate_gap",
        "macro_core_strong_gap",
        "thin_information_moderate_gap",
        "thin_information_strong_gap",
    ):
        scenario_dir = base / "larger_network" / _scenario_subdir_name(scenario_name)
        paths = pd.read_csv(scenario_dir / "policy_paths.csv")
        scenario_spec = json.loads((scenario_dir / "scenario_spec.json").read_text())
        rate_bound = float(scenario_spec["rate_bounds"][1])
        for policy_name in ("classical_filtered_rule", "learning_policy"):
            sub = paths[paths["policy_name"] == policy_name].copy()
            sign_change_rates = []
            corner_shares = []
            impact_sign_matches = []
            for _, frame in sub.groupby("evaluation_seed"):
                rate = frame.sort_values("period")["policy_rate"].to_numpy(dtype=float)
                diff = np.diff(rate)
                nonzero = np.abs(diff) > 1.0e-8
                signs = np.sign(diff[nonzero])
                sign_changes = float(np.mean(signs[1:] != signs[:-1])) if signs.size > 1 else 0.0
                sign_change_rates.append(sign_changes)
                corner_shares.append(float(np.mean(np.abs(rate) >= 0.95 * rate_bound)))
            if policy_name == "learning_policy":
                by_seed_rl = {
                    int(seed): frame.sort_values("period")["policy_rate"].to_numpy(dtype=float)
                    for seed, frame in sub.groupby("evaluation_seed")
                }
                by_seed_classical = {
                    int(seed): frame.sort_values("period")["policy_rate"].to_numpy(dtype=float)
                    for seed, frame in paths[paths["policy_name"] == "classical_filtered_rule"].groupby("evaluation_seed")
                }
                for seed, rl_rate in by_seed_rl.items():
                    classical_rate = by_seed_classical[int(seed)]
                    impact_sign_matches.append(
                        float(np.sign(rl_rate[0]) == np.sign(classical_rate[0]))
                    )
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "policy_name": policy_name,
                    "mean_policy_rate_volatility": float(
                        sub.groupby("evaluation_seed")["policy_rate"].std().mean()
                    ),
                    "mean_sign_change_rate": float(np.mean(sign_change_rates)),
                    "mean_corner_share": float(np.mean(corner_shares)),
                    "rate_bound": rate_bound,
                    "impact_sign_match_share_vs_classical": (
                        float(np.mean(impact_sign_matches)) if impact_sign_matches else np.nan
                    ),
                }
            )
    sanity = pd.DataFrame(rows).sort_values(["scenario_name", "policy_name"]).reset_index(drop=True)
    sanity.to_csv(root / "policy_sanity_summary.csv", index=False)
    return sanity


def run_small_perturbation_checks(
    *,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lookup = default_universal_candidate_lookup()
    baseline = lookup["baseline_rawobs"]
    larger = lookup["larger_network"]

    perturbations = [
        (
            "higher_noise",
            replace(
                extreme_sticky_regime_config(),
                stress_macro_noise_factor=1.60,
                stress_distribution_noise_factor=2.60,
            ),
            [baseline, larger],
        ),
        (
            "persistent_regimes",
            replace(
                extreme_sticky_regime_config(),
                regime_transition=((0.985, 0.015), (0.01, 0.99)),
            ),
            [baseline, larger],
        ),
        (
            "action_bound_sensitivity",
            extreme_sticky_regime_config(),
            [
                UniversalTuningCandidate(
                    name="larger_network_bound_down",
                    action_bound=0.0030,
                    ppo=larger.ppo,
                    description="larger_network PPO with smaller action bound.",
                ),
                larger,
                UniversalTuningCandidate(
                    name="larger_network_bound_up",
                    action_bound=0.0040,
                    ppo=larger.ppo,
                    description="larger_network PPO with larger action bound.",
                ),
            ],
        ),
    ]

    rows = []
    candidate_summaries = []
    for name, regime_config, candidates in perturbations:
        run_root = root / name
        results = run_universal_rawobs_misspecified_tuning(
            output_dir=str(run_root),
            candidates=candidates,
            regime_config=regime_config,
            training_seeds=(11,),
            selection_seeds=(500,),
            evaluation_seeds=(960, 961, 962, 963, 964),
        )
        summary = results["candidate_summary"].copy()
        summary.insert(0, "perturbation_name", name)
        candidate_summaries.append(summary)
        best = summary.iloc[0]
        rows.append(
            {
                "perturbation_name": name,
                "best_candidate": str(best["candidate_name"]),
                "mean_delta_cumulative_loss": float(best["mean_delta_cumulative_loss"]),
                "worst_case_delta_cumulative_loss": float(best["worst_case_delta_cumulative_loss"]),
                "mean_relative_loss_improvement_pct": float(best["mean_relative_loss_improvement_pct"]),
                "num_wins": int(best["num_wins"]),
            }
        )
    perturbation_summary = pd.DataFrame(rows)
    candidate_summary = pd.concat(candidate_summaries, ignore_index=True)
    perturbation_summary.to_csv(root / "perturbation_summary.csv", index=False)
    candidate_summary.to_csv(root / "perturbation_candidate_summary.csv", index=False)
    return perturbation_summary, candidate_summary


def run_deep_validation(
    *,
    base_dir: str | Path = "outputs/hank_regime_learning_stage6_validation_suite/oos_seeds",
    output_dir: str | Path = "outputs/hank_regime_learning_stage6_deep_validation",
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    dispersion = compute_dispersion_and_ci(base_dir=base_dir, output_dir=root / "dispersion")
    _, retuned_best = run_retuned_classical_benchmark(base_dir=base_dir, output_dir=root / "retuned_classical")
    sanity = summarize_policy_sanity(base_dir=base_dir, output_dir=root / "policy_sanity")
    perturbation_summary, perturbation_candidates = run_small_perturbation_checks(output_dir=root / "perturbations")

    report_lines = [
        "# Stage 6 Deep Validation",
        "",
        "This validation layer adds dispersion, a stronger classical benchmark, small perturbation checks,",
        "and economic sanity diagnostics for the tuned raw-observation regime-learning result.",
        "",
        "## Dispersion and confidence intervals",
    ]
    for row in dispersion[dispersion["candidate_name"] == "larger_network"].to_dict(orient="records"):
        report_lines.append(
            f"- `{row['scenario_name']}`: mean delta `{row['mean_delta_cumulative_loss']:.6e}`, "
            f"95% bootstrap CI `[{row['bootstrap_ci_low']:.6e}, {row['bootstrap_ci_high']:.6e}]`, "
            f"seed win share `{row['share_negative_delta']:.2f}`."
        )
    report_lines.extend(["", "## Retuned classical benchmark"])
    for row in retuned_best.to_dict(orient="records"):
        report_lines.append(
            f"- `{row['scenario_name']}`: best retuned rule `(phi_pi={row['phi_pi']:.3f}, "
            f"phi_y={row['phi_y']:.3f}, rho_i={row['rho_i']:.3f})`, "
            f"RL minus retuned classical `{row['delta_rl_minus_retuned']:.6e}`."
        )
    report_lines.extend(["", "## Small perturbation checks"])
    for row in perturbation_summary.to_dict(orient="records"):
        report_lines.append(
            f"- `{row['perturbation_name']}`: best candidate `{row['best_candidate']}`, "
            f"mean delta `{row['mean_delta_cumulative_loss']:.6e}`, "
            f"worst-case delta `{row['worst_case_delta_cumulative_loss']:.6e}`."
        )
    report_lines.extend(["", "## Policy sanity"])
    for row in sanity[sanity["policy_name"] == "learning_policy"].to_dict(orient="records"):
        report_lines.append(
            f"- `{row['scenario_name']}`: volatility `{row['mean_policy_rate_volatility']:.6e}`, "
            f"corner share `{row['mean_corner_share']:.3f}`, sign-change rate `{row['mean_sign_change_rate']:.3f}`."
        )
    (root / "report_stage6_deep_validation.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "dispersion": dispersion,
        "retuned_classical_best": retuned_best,
        "policy_sanity": sanity,
        "perturbation_summary": perturbation_summary,
        "perturbation_candidate_summary": perturbation_candidates,
    }
