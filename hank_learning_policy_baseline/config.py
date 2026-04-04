from __future__ import annotations

from dataclasses import asdict, dataclass

from hank_partial_info_baseline.config import HANKPartialInfoConfig, default_partial_info_config


@dataclass(frozen=True)
class PPOConfig:
    hidden_dim_1: int = 32
    hidden_dim_2: int = 32
    rollout_episodes: int = 8
    num_iterations: int = 24
    num_epochs: int = 8
    minibatch_size: int = 128
    actor_learning_rate: float = 5.0e-4
    critic_learning_rate: float = 8.0e-4
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    entropy_coefficient: float = 1.0e-3
    max_grad_norm: float = 1.0
    initial_log_std: float = -9.0
    min_log_std: float = -10.5
    max_log_std: float = -7.0
    validation_episodes: int = 4
    normalization_episodes: int = 8
    normalization_clip: float = 6.0
    checkpoint_interval: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingVariant:
    name: str
    scenario_name: str
    scenario_label: str
    input_mode: str
    include_distributional_state: bool = True
    classical_benchmark_scenario_name: str | None = None
    classical_policy_label: str = "Classical: filter + fixed rule"
    innovation_scale: float = 1.0
    distributional_state_shock_scale: float = 0.0
    is_main: bool = True
    is_ablation: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Stage4Config:
    output_dir: str = "outputs/hank_learning_stage4"
    horizon: int = 60
    gamma: float = 0.99
    lambda_y: float = 0.5
    lambda_i: float = 0.05
    action_bound: float = 0.0015
    training_seeds: tuple[int, ...] = (11, 22)
    selection_seeds: tuple[int, ...] = (500, 501)
    evaluation_seeds: tuple[int, ...] = (700,)
    selection_metric: str = "full_hank_cumulative_loss"
    partial_config: HANKPartialInfoConfig = default_partial_info_config()
    ppo: PPOConfig = PPOConfig()

    def main_variants(self) -> list[TrainingVariant]:
        labels = {
            spec["name"]: spec["label"]
            for spec in self.partial_config.scenario_specs()
        }
        baseline_variants = [
            TrainingVariant(
                name=f"{scenario_name}_filtered_state",
                scenario_name=scenario_name,
                scenario_label=labels[scenario_name],
                input_mode="filtered_state",
                include_distributional_state=True,
                is_main=True,
                description="Основной learning-based policy на filtered reduced HANK state.",
            )
            for scenario_name in (
                "macro_core",
                "full_macro",
                "thin_information",
                "high_noise",
                "distribution_augmented",
            )
        ]
        baseline_variants.extend([
            TrainingVariant(
                name="macro_core_stress_filtered_state",
                scenario_name="macro_core",
                scenario_label="Фильтрация: инфляция, выпуск и ставка × distributional stress",
                input_mode="filtered_state",
                include_distributional_state=True,
                innovation_scale=1.35,
                distributional_state_shock_scale=2.5e-4,
                is_main=True,
                description=(
                    "2x2 pilot stress case: macro_core information with larger macro "
                    "disturbances and direct shocks to reduced distributional state."
                ),
            ),
            TrainingVariant(
                name="thin_information_stress_filtered_state",
                scenario_name="thin_information",
                scenario_label="Фильтрация: инфляция и ставка × distributional stress",
                input_mode="filtered_state",
                include_distributional_state=True,
                innovation_scale=1.35,
                distributional_state_shock_scale=2.5e-4,
                is_main=True,
                description=(
                    "2x2 pilot stress case: thin information with larger macro "
                    "disturbances and direct shocks to reduced distributional state."
                ),
            ),
        ])
        baseline_variants.append(
            TrainingVariant(
                name="distribution_sensitive_filtered_state",
                scenario_name="distribution_augmented",
                scenario_label="HANK-sensitive policy: RL с распределительной статистикой vs macro-only Taylor",
                input_mode="filtered_state",
                include_distributional_state=True,
                classical_benchmark_scenario_name="macro_core",
                classical_policy_label="Classical: macro-only filter + fixed rule",
                is_main=True,
                description=(
                    "HANK-specific extension: learning-based policy observes filtered "
                    "distributional state, while the classical benchmark remains a "
                    "macro-core filter-plus-rule architecture."
                ),
            )
        )
        baseline_variants.append(
            TrainingVariant(
                name="distribution_sensitive_stress_filtered_state",
                scenario_name="distribution_augmented",
                scenario_label="HANK-stress policy: RL с distribution state vs macro-only Taylor",
                input_mode="filtered_state",
                include_distributional_state=True,
                classical_benchmark_scenario_name="macro_core",
                classical_policy_label="Classical: macro-only filter + fixed rule",
                innovation_scale=1.35,
                distributional_state_shock_scale=2.5e-4,
                is_main=True,
                description=(
                    "Stress extension with larger macro disturbances and direct reduced-form "
                    "shocks to distributional state components."
                ),
            )
        )
        return baseline_variants

    def ablation_variants(self) -> list[TrainingVariant]:
        labels = {
            spec["name"]: spec["label"]
            for spec in self.partial_config.scenario_specs()
        }
        return [
            TrainingVariant(
                name="macro_core_filtered_state_uncertainty",
                scenario_name="macro_core",
                scenario_label=labels["macro_core"],
                input_mode="filtered_state_uncertainty",
                include_distributional_state=True,
                is_main=False,
                is_ablation=True,
                description="Ablation A: filtered state plus uncertainty.",
            ),
            TrainingVariant(
                name="macro_core_raw_observations",
                scenario_name="macro_core",
                scenario_label=labels["macro_core"],
                input_mode="raw_observations",
                include_distributional_state=True,
                is_main=False,
                is_ablation=True,
                description="Ablation A: raw noisy observations instead of explicit filtering input.",
            ),
            TrainingVariant(
                name="distribution_augmented_no_distribution_state",
                scenario_name="distribution_augmented",
                scenario_label=labels["distribution_augmented"],
                input_mode="filtered_state",
                include_distributional_state=False,
                is_main=False,
                is_ablation=True,
                description="Ablation B: remove filtered distributional state from the policy input.",
            ),
        ]

    def all_variants(self) -> list[TrainingVariant]:
        return self.main_variants() + self.ablation_variants()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["training_seeds"] = list(self.training_seeds)
        payload["selection_seeds"] = list(self.selection_seeds)
        payload["evaluation_seeds"] = list(self.evaluation_seeds)
        payload["main_variants"] = [variant.to_dict() for variant in self.main_variants()]
        payload["ablation_variants"] = [variant.to_dict() for variant in self.ablation_variants()]
        return payload


def default_stage4_config() -> Stage4Config:
    return Stage4Config()
