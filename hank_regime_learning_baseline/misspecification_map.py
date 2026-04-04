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
from hank_partial_info_baseline.state_space import fit_reduced_state_space
from regime_switching_baseline.regime_filter import _imm_update_step
from regime_switching_baseline.regime_model import RegimeSwitchingModel, build_regime_switching_model

from .config import RegimeLearningConfig
from .environment import RegimeSwitchingPolicyEnvironment, build_scenario_spec
from .evaluation import _is_unstable, simulate_policy_episode
from .tuning import default_universal_candidate_lookup, extreme_sticky_regime_config, raw_observation_variants_2x2


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _single_regime_kalman_update(
    *,
    predicted_mean: np.ndarray,
    predicted_covariance: np.ndarray,
    observation: np.ndarray,
    observation_matrix: np.ndarray,
    measurement_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_dim = len(predicted_mean)
    identity = np.eye(state_dim, dtype=float)
    innovation = observation - observation_matrix @ predicted_mean
    innovation_covariance = observation_matrix @ predicted_covariance @ observation_matrix.T + measurement_covariance
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
    jitter = 1.0e-12
    sign, _ = np.linalg.slogdet(innovation_covariance)
    while sign <= 0:
        innovation_covariance = innovation_covariance + jitter * np.eye(observation.shape[0], dtype=float)
        jitter *= 10.0
        sign, _ = np.linalg.slogdet(innovation_covariance)
        if jitter > 1.0:
            raise RuntimeError("Innovation covariance is not positive definite.")
    innovation_precision = np.linalg.inv(innovation_covariance)
    kalman_gain = predicted_covariance @ observation_matrix.T @ innovation_precision
    filtered_mean = predicted_mean + kalman_gain @ innovation
    filtered_covariance = (
        (identity - kalman_gain @ observation_matrix)
        @ predicted_covariance
        @ (identity - kalman_gain @ observation_matrix).T
        + kalman_gain @ measurement_covariance @ kalman_gain.T
    )
    filtered_covariance = 0.5 * (filtered_covariance + filtered_covariance.T)
    return filtered_mean, filtered_covariance


def _rule_rate(
    *,
    state: np.ndarray,
    state_index: dict[str, int],
    prev_rate: float,
    phi_pi: float,
    phi_y: float,
    rho_i: float,
    bounds: tuple[float, float],
) -> float:
    rule_term = (
        state[state_index["rstar_gap"]]
        + phi_pi * state[state_index["inflation_gap"]]
        + phi_y * state[state_index["output_gap"]]
    )
    rate = rho_i * prev_rate + (1.0 - rho_i) * rule_term
    return float(np.clip(rate, bounds[0], bounds[1]))


@dataclass(frozen=True)
class MisspecificationSpec:
    name: str
    label: str
    description: str
    filter_type: str
    measurement_noise_scale: float = 1.0
    phi_pi: float = 1.5
    phi_y: float = 0.125
    rho_i: float = 0.7
    regime_transition_override: tuple[tuple[float, float], tuple[float, float]] | None = None

    def to_dict(self) -> dict:
        payload = {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "filter_type": self.filter_type,
            "measurement_noise_scale": self.measurement_noise_scale,
            "phi_pi": self.phi_pi,
            "phi_y": self.phi_y,
            "rho_i": self.rho_i,
        }
        if self.regime_transition_override is not None:
            payload["regime_transition_override"] = [list(row) for row in self.regime_transition_override]
        return payload


class _SingleRegimeClassicalPolicy(BasePolicy):
    def __init__(
        self,
        *,
        model: RegimeSwitchingModel,
        observation_names: tuple[str, ...],
        measurement_covariance: np.ndarray,
        phi_pi: float,
        phi_y: float,
        rho_i: float,
    ) -> None:
        self.transition = np.asarray(model.transition_matrices[0], dtype=float)
        self.control = np.asarray(model.control_loadings[0], dtype=float)
        self.process_noise = np.asarray(model.process_noise_covariances[0], dtype=float)
        self.state_index = {name: idx for idx, name in enumerate(model.state_names)}
        self.observation_matrix = np.asarray(
            model.observation_matrix[[model.observation_index(name) for name in observation_names]],
            dtype=float,
        )
        self.measurement_covariance = np.asarray(measurement_covariance, dtype=float)
        self.initial_covariance = np.asarray(model.stationary_state_covariances()[0], dtype=float)
        self.phi_pi = float(phi_pi)
        self.phi_y = float(phi_y)
        self.rho_i = float(rho_i)
        self.filtered_mean: np.ndarray | None = None
        self.filtered_covariance: np.ndarray | None = None

    def reset(self) -> None:
        self.filtered_mean = None
        self.filtered_covariance = None

    def rate(self, observation: np.ndarray, info: dict) -> float:
        y = np.asarray(info["current_observations"], dtype=float)
        prev_control = float(info["policy_shock"])
        prev_rate = float(info["current_rate"])
        bounds = tuple(info["rate_bounds"])
        if self.filtered_mean is None or self.filtered_covariance is None:
            predicted_mean = np.zeros((len(self.state_index),), dtype=float)
            predicted_covariance = self.initial_covariance.copy()
        else:
            predicted_mean = self.transition @ self.filtered_mean + self.control * prev_control
            predicted_covariance = self.transition @ self.filtered_covariance @ self.transition.T + self.process_noise
            predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
        self.filtered_mean, self.filtered_covariance = _single_regime_kalman_update(
            predicted_mean=predicted_mean,
            predicted_covariance=predicted_covariance,
            observation=y,
            observation_matrix=self.observation_matrix,
            measurement_covariance=self.measurement_covariance,
        )
        return _rule_rate(
            state=self.filtered_mean,
            state_index=self.state_index,
            prev_rate=prev_rate,
            phi_pi=self.phi_pi,
            phi_y=self.phi_y,
            rho_i=self.rho_i,
            bounds=bounds,
        )


class _SwitchingClassicalPolicy(BasePolicy):
    def __init__(
        self,
        *,
        model: RegimeSwitchingModel,
        observation_names: tuple[str, ...],
        measurement_covariance: np.ndarray,
        phi_pi: float,
        phi_y: float,
        rho_i: float,
    ) -> None:
        self.model = model
        self.state_index = {name: idx for idx, name in enumerate(model.state_names)}
        self.observation_matrix = np.asarray(
            model.observation_matrix[[model.observation_index(name) for name in observation_names]],
            dtype=float,
        )
        self.measurement_covariance = np.asarray(measurement_covariance, dtype=float)
        self.initial_mode_probabilities = model.stationary_regime_distribution()
        self.initial_mode_means = np.repeat(model.initial_state_mean()[None, :], model.num_regimes(), axis=0)
        self.initial_mode_covariances = model.stationary_state_covariances()
        self.phi_pi = float(phi_pi)
        self.phi_y = float(phi_y)
        self.rho_i = float(rho_i)
        self.mode_probabilities: np.ndarray | None = None
        self.mode_means: np.ndarray | None = None
        self.mode_covariances: np.ndarray | None = None
        self.filtered_mean: np.ndarray | None = None

    def reset(self) -> None:
        self.mode_probabilities = None
        self.mode_means = None
        self.mode_covariances = None
        self.filtered_mean = None

    def rate(self, observation: np.ndarray, info: dict) -> float:
        y = np.asarray(info["current_observations"], dtype=float)
        prev_control = float(info["policy_shock"])
        prev_rate = float(info["current_rate"])
        bounds = tuple(info["rate_bounds"])
        if self.mode_probabilities is None or self.mode_means is None or self.mode_covariances is None:
            self.mode_probabilities = self.initial_mode_probabilities.copy()
            self.mode_means = self.initial_mode_means.copy()
            self.mode_covariances = self.initial_mode_covariances.copy()
        (
            _predicted_mode_probabilities,
            filtered_mode_probabilities,
            _predicted_means,
            _predicted_covariances,
            regime_conditioned_means,
            regime_conditioned_covariances,
            filtered_mean,
            _filtered_covariance,
            _innovations,
            _innovation_covariances,
            _increment,
        ) = _imm_update_step(
            model=self.model,
            previous_mode_probabilities=self.mode_probabilities,
            previous_mode_means=self.mode_means,
            previous_mode_covariances=self.mode_covariances,
            observation=y,
            observation_matrix=self.observation_matrix,
            measurement_covariance=self.measurement_covariance,
            previous_control=prev_control,
        )
        self.mode_probabilities = filtered_mode_probabilities
        self.mode_means = regime_conditioned_means
        self.mode_covariances = regime_conditioned_covariances
        self.filtered_mean = filtered_mean
        return _rule_rate(
            state=filtered_mean,
            state_index=self.state_index,
            prev_rate=prev_rate,
            phi_pi=self.phi_pi,
            phi_y=self.phi_y,
            rho_i=self.rho_i,
            bounds=bounds,
        )


def _scenario_labels() -> dict[str, str]:
    return {
        "macro_core_moderate_gap": "Инфляция, выпуск, ставка × умеренный режимный разрыв",
        "macro_core_strong_gap": "Инфляция, выпуск, ставка × сильный режимный разрыв",
        "thin_information_moderate_gap": "Инфляция, ставка × умеренный режимный разрыв",
        "thin_information_strong_gap": "Инфляция, ставка × сильный режимный разрыв",
    }


def _misspecification_specs() -> list[MisspecificationSpec]:
    return [
        MisspecificationSpec(
            name="normal_only_filter",
            label="Single-regime normal-only filter",
            description="Classical benchmark ignores hidden regime switching and filters as if only the normal regime existed.",
            filter_type="single_regime_normal_only",
        ),
        MisspecificationSpec(
            name="overstated_noise",
            label="Overstated measurement noise",
            description="Switching filter is retained, but measurement noise variance is overstated, making the classical rule excessively cautious.",
            filter_type="switching",
            measurement_noise_scale=2.0,
        ),
        MisspecificationSpec(
            name="wrong_persistence",
            label="Normal-biased regime persistence",
            description="Switching filter underestimates stress arrivals and overestimates stress exits.",
            filter_type="switching",
            regime_transition_override=((0.992, 0.008), (0.12, 0.88)),
        ),
        MisspecificationSpec(
            name="inflation_only_rule",
            label="Inflation-only simple rule",
            description="Filter is correctly switching, but the policy rule omits the output-gap term.",
            filter_type="switching",
            phi_y=0.0,
        ),
    ]


def _build_measurement_covariance(env: RegimeSwitchingPolicyEnvironment, scale: float) -> np.ndarray:
    base_std = [env.measurement_noise_std[name] for name in env.scenario_spec.noisy_observations]
    return np.diag([(float(std) * scale) ** 2 for std in base_std]).astype(float)


def _with_transition_override(model: RegimeSwitchingModel, transition_override: tuple[tuple[float, float], tuple[float, float]]) -> RegimeSwitchingModel:
    return RegimeSwitchingModel(
        state_names=model.state_names,
        observation_names=model.observation_names,
        regime_names=model.regime_names,
        regime_transition_matrix=np.asarray(transition_override, dtype=float),
        transition_matrices=np.asarray(model.transition_matrices, dtype=float).copy(),
        control_loadings=np.asarray(model.control_loadings, dtype=float).copy(),
        process_noise_covariances=np.asarray(model.process_noise_covariances, dtype=float).copy(),
        observation_matrix=np.asarray(model.observation_matrix, dtype=float).copy(),
        steady_state_statistics=dict(model.steady_state_statistics),
        base_model_training_summary=dict(model.base_model_training_summary),
        gap_scale=float(model.gap_scale),
    )


def _make_misspecified_policy(
    *,
    spec: MisspecificationSpec,
    env: RegimeSwitchingPolicyEnvironment,
) -> BasePolicy:
    measurement_covariance = _build_measurement_covariance(env, spec.measurement_noise_scale)
    if spec.filter_type == "single_regime_normal_only":
        return _SingleRegimeClassicalPolicy(
            model=env.model,
            observation_names=env.scenario_spec.noisy_observations,
            measurement_covariance=measurement_covariance,
            phi_pi=spec.phi_pi,
            phi_y=spec.phi_y,
            rho_i=spec.rho_i,
        )
    if spec.filter_type == "switching":
        model = env.model
        if spec.regime_transition_override is not None:
            model = _with_transition_override(model, spec.regime_transition_override)
        return _SwitchingClassicalPolicy(
            model=model,
            observation_names=env.scenario_spec.noisy_observations,
            measurement_covariance=measurement_covariance,
            phi_pi=spec.phi_pi,
            phi_y=spec.phi_y,
            rho_i=spec.rho_i,
        )
    raise ValueError(f"Unsupported misspecification filter type: {spec.filter_type}")


def _load_best_learned_selection(architecture_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison = pd.read_csv(architecture_dir / "architecture_comparison.csv")
    seed_level = pd.read_csv(architecture_dir / "architecture_seed_level.csv")
    policy_metrics = pd.read_csv(architecture_dir / "policy_metrics_all.csv")

    rows = []
    for row in comparison.to_dict(orient="records"):
        if float(row["rawobs_mean_cumulative_loss"]) < float(row["belief_mean_cumulative_loss"]):
            best_arch = "raw_observations"
            best_column = "rawobs_cumulative_policy_loss"
            best_mean = float(row["rawobs_mean_cumulative_loss"])
        else:
            best_arch = "belief_state"
            best_column = "belief_cumulative_policy_loss"
            best_mean = float(row["belief_mean_cumulative_loss"])
        rows.append({
            "scenario_name": row["scenario_name"],
            "scenario_label": row["scenario_label"],
            "best_learned_architecture": best_arch,
            "best_learned_mean_cumulative_loss": best_mean,
            "switching_classical_mean_cumulative_loss": float(row["classical_mean_cumulative_loss"]),
            "full_information_mean_cumulative_loss": float(
                seed_level.loc[seed_level["scenario_name"] == row["scenario_name"], "full_information_cumulative_policy_loss"].mean()
            ),
            "seed_loss_column": best_column,
        })
    selection = pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)

    seed_rows = []
    for row in selection.to_dict(orient="records"):
        scenario_seed = seed_level[seed_level["scenario_name"] == row["scenario_name"]].copy()
        for seed_row in scenario_seed.to_dict(orient="records"):
            seed_rows.append({
                "scenario_name": row["scenario_name"],
                "scenario_label": row["scenario_label"],
                "evaluation_seed": int(seed_row["evaluation_seed"]),
                "best_learned_architecture": row["best_learned_architecture"],
                "switching_classical_cumulative_loss": float(seed_row["classical_cumulative_policy_loss"]),
                "best_learned_cumulative_loss": float(seed_row[row["seed_loss_column"]]),
                "full_information_cumulative_loss": float(seed_row["full_information_cumulative_policy_loss"]),
            })
    learned_seed = pd.DataFrame(seed_rows).sort_values(["scenario_name", "evaluation_seed"]).reset_index(drop=True)
    return selection, learned_seed, policy_metrics


def run_misspecification_map(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_misspecification_map",
    architecture_dir: str = "outputs/hank_regime_learning_stage6_architecture_ablation",
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    architecture_root = Path(architecture_dir)

    candidate = default_universal_candidate_lookup()["larger_network"]
    regime_config = extreme_sticky_regime_config()
    config = RegimeLearningConfig(
        output_dir=str(root),
        horizon=60,
        gamma=0.99,
        lambda_y=0.5,
        lambda_i=0.05,
        action_bound=candidate.action_bound,
        classical_policy_mode="switching",
        training_seeds=(11,),
        selection_seeds=(500, 501),
        evaluation_seeds=tuple(range(900, 910)),
        regime_config=regime_config,
        ppo=candidate.ppo,
    )

    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, regime_config.partial_config)
    scenario_variants = {variant.scenario_name: variant for variant in raw_observation_variants_2x2()}
    selection, learned_seed, _policy_metrics = _load_best_learned_selection(architecture_root)
    misspec_specs = _misspecification_specs()

    _save_json(root / "misspecification_spec.json", {
        "source_architecture_dir": str(architecture_root),
        "candidate_name": candidate.name,
        "candidate_description": candidate.description,
        "action_bound": candidate.action_bound,
        "evaluation_seeds": list(config.evaluation_seeds),
        "misspecifications": [spec.to_dict() for spec in misspec_specs],
        "best_learned_selection": selection.to_dict(orient="records"),
    })

    scenario_rows = []
    seed_rows = []

    for scenario_name, variant in scenario_variants.items():
        scenario_spec = build_scenario_spec(config, variant)
        true_model = build_regime_switching_model(reduced_model, config.regime_config, scenario_spec.gap_scale)

        def env_factory():
            return RegimeSwitchingPolicyEnvironment(
                model=true_model,
                regime_config=config.regime_config,
                scenario_spec=scenario_spec,
                phi_pi=hank_config.phi_pi,
                phi_y=hank_config.phi_y,
                rho_i=hank_config.rho_i,
            )

        switching_policy = ClassicalFilteredRulePolicy(action_bound=config.action_bound)

        for evaluation_seed in config.evaluation_seeds:
            classical_trace = simulate_policy_episode(
                env_factory=env_factory,
                policy=switching_policy,
                scenario_spec=scenario_spec,
                evaluation_seed=int(evaluation_seed),
                policy_name="classical_filtered_rule",
                policy_label="Filter + fixed rule",
                training_seed=None,
            )
            switching_cumulative_loss = float(classical_trace["loss"].sum())
            switching_volatility = float(np.std(classical_trace["policy_rate"].to_numpy(dtype=float)))
            for spec in misspec_specs:
                policy = _make_misspecified_policy(spec=spec, env=env_factory())
                trace = simulate_policy_episode(
                    env_factory=env_factory,
                    policy=policy,
                    scenario_spec=scenario_spec,
                    evaluation_seed=int(evaluation_seed),
                    policy_name=spec.name,
                    policy_label=spec.label,
                    training_seed=None,
                )
                seed_rows.append({
                    "scenario_name": scenario_name,
                    "scenario_label": _scenario_labels()[scenario_name],
                    "misspecification_name": spec.name,
                    "misspecification_label": spec.label,
                    "evaluation_seed": int(evaluation_seed),
                    "switching_classical_cumulative_loss": switching_cumulative_loss,
                    "switching_classical_policy_volatility": switching_volatility,
                    "misspecified_cumulative_loss": float(trace["loss"].sum()),
                    "misspecified_mean_loss": float(trace["loss"].mean()),
                    "misspecified_policy_volatility": float(np.std(trace["policy_rate"].to_numpy(dtype=float))),
                    "misspecified_corner_share": float(
                        np.mean(
                            np.isclose(
                                np.abs(trace["policy_rate"].to_numpy(dtype=float)),
                                scenario_spec.rate_bounds[1],
                                atol=1.0e-10,
                            )
                        )
                    ),
                    "misspecified_unstable": int(_is_unstable(trace)),
                })

    seed_level = pd.DataFrame(seed_rows).sort_values(
        ["scenario_name", "misspecification_name", "evaluation_seed"]
    ).reset_index(drop=True)
    seed_level.to_csv(root / "misspecification_seed_level.csv", index=False)
    selection.to_csv(root / "best_learned_selection.csv", index=False)
    learned_seed.to_csv(root / "best_learned_seed_level.csv", index=False)

    merged = seed_level.merge(
        learned_seed,
        on=["scenario_name", "scenario_label", "evaluation_seed"],
        how="left",
        suffixes=("", "_learned"),
    )
    if "switching_classical_cumulative_loss_learned" in merged.columns:
        merged["switching_classical_alignment_gap"] = (
            merged["switching_classical_cumulative_loss"] - merged["switching_classical_cumulative_loss_learned"]
        )
        merged = merged.drop(columns=["switching_classical_cumulative_loss_learned"])
    merged["learned_minus_misspecified"] = (
        merged["best_learned_cumulative_loss"] - merged["misspecified_cumulative_loss"]
    )
    merged["misspecified_minus_switching"] = (
        merged["misspecified_cumulative_loss"] - merged["switching_classical_cumulative_loss"]
    )
    merged["learned_minus_switching"] = (
        merged["best_learned_cumulative_loss"] - merged["switching_classical_cumulative_loss"]
    )

    summary_rows = []
    for (scenario_name, misspec_name), frame in merged.groupby(["scenario_name", "misspecification_name"]):
        scenario_label = frame["scenario_label"].iloc[0]
        misspec_label = frame["misspecification_label"].iloc[0]
        best_arch = frame["best_learned_architecture"].iloc[0]
        summary_rows.append({
            "scenario_name": scenario_name,
            "scenario_label": scenario_label,
            "best_learned_architecture": best_arch,
            "misspecification_name": misspec_name,
            "misspecification_label": misspec_label,
            "switching_classical_mean_cumulative_loss": float(frame["switching_classical_cumulative_loss"].mean()),
            "best_learned_mean_cumulative_loss": float(frame["best_learned_cumulative_loss"].mean()),
            "misspecified_mean_cumulative_loss": float(frame["misspecified_cumulative_loss"].mean()),
            "best_learned_minus_misspecified": float(frame["learned_minus_misspecified"].mean()),
            "best_learned_relative_improvement_vs_misspecified_pct": float(
                100.0
                * (frame["misspecified_cumulative_loss"].mean() - frame["best_learned_cumulative_loss"].mean())
                / frame["misspecified_cumulative_loss"].mean()
            ),
            "misspecified_excess_loss_vs_switching": float(frame["misspecified_minus_switching"].mean()),
            "best_learned_excess_loss_vs_switching": float(frame["learned_minus_switching"].mean()),
            "best_learned_win_rate_vs_misspecified": float(np.mean(frame["learned_minus_misspecified"] < 0.0)),
            "misspecified_std_cumulative_loss": float(frame["misspecified_cumulative_loss"].std(ddof=1)),
            "misspecified_mean_policy_volatility": float(frame["misspecified_policy_volatility"].mean()),
            "misspecified_mean_corner_share": float(frame["misspecified_corner_share"].mean()),
            "misspecified_any_unstable": int(frame["misspecified_unstable"].max()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["misspecification_name", "scenario_name"]
    ).reset_index(drop=True)
    summary.to_csv(root / "misspecification_results.csv", index=False)

    win_rows = []
    for misspec_name, frame in summary.groupby("misspecification_name"):
        win_rows.append({
            "misspecification_name": misspec_name,
            "misspecification_label": frame["misspecification_label"].iloc[0],
            "mean_relative_improvement_pct": float(frame["best_learned_relative_improvement_vs_misspecified_pct"].mean()),
            "mean_excess_loss_vs_switching": float(frame["misspecified_excess_loss_vs_switching"].mean()),
            "four_of_four_wins": int(np.sum(frame["best_learned_minus_misspecified"] < 0.0) == 4),
            "scenario_win_share": float(np.mean(frame["best_learned_minus_misspecified"] < 0.0)),
            "mean_seed_win_rate": float(frame["best_learned_win_rate_vs_misspecified"].mean()),
        })
    win_summary = pd.DataFrame(win_rows).sort_values("mean_relative_improvement_pct", ascending=False).reset_index(drop=True)
    win_summary.to_csv(root / "misspecification_win_summary.csv", index=False)

    heatmap = summary.pivot(index="misspecification_label", columns="scenario_label", values="best_learned_relative_improvement_vs_misspecified_pct")
    heatmap.to_csv(root / "misspecification_heatmap.csv")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(heatmap.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels(list(heatmap.columns), rotation=25, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(list(heatmap.index))
    ax.set_title("Преимущество learned policy над misspecified classical, %")
    for i in range(len(heatmap.index)):
        for j in range(len(heatmap.columns)):
            value = float(heatmap.iloc[i, j])
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="#1f1f1f", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.85, label="% улучшения по cumulative loss")
    fig.tight_layout()
    fig.savefig(root / "fig_misspecification_heatmap.png", dpi=220)
    fig.savefig(root / "fig_misspecification_heatmap.pdf")
    plt.close(fig)

    report_lines = [
        "# Stage 6 Misspecification Map",
        "",
        "В этой серии лучший learned controller из architecture ablation сравнивается с несколькими misspecified classical architectures на той же 2x2 regime-switching карте.",
        "",
        "## Best Learned Selection",
        "",
    ]
    for row in selection.to_dict(orient="records"):
        report_lines.append(
            f"- `{row['scenario_name']}`: лучшая learned architecture = `{row['best_learned_architecture']}`, mean cumulative loss = `{row['best_learned_mean_cumulative_loss']:.6e}`."
        )
    report_lines.extend(["", "## Misspecification Summary", ""])
    for row in win_summary.to_dict(orient="records"):
        report_lines.extend(
            [
                f"### {row['misspecification_label']}",
                f"- Mean relative improvement of learned policy vs misspecified classical: `{row['mean_relative_improvement_pct']:.2f}%`.",
                f"- Mean excess loss of misspecified classical vs correctly specified switching rule: `{row['mean_excess_loss_vs_switching']:.6e}`.",
                f"- Scenario win share: `{row['scenario_win_share']:.2f}`.",
                f"- Mean seed win rate: `{row['mean_seed_win_rate']:.2f}`.",
                "",
            ]
        )
    (root / "report_misspecification_map.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "selection": selection,
        "seed_level": merged,
        "summary": summary,
        "win_summary": win_summary,
    }
