from __future__ import annotations

from dataclasses import asdict, dataclass, field

from hank_learning_policy_baseline.config import PPOConfig
from regime_switching_baseline.regime_model import RegimeSwitchingConfig


@dataclass(frozen=True)
class RegimeLearningVariant:
    name: str
    scenario_name: str
    scenario_label: str
    input_mode: str = "belief_state"
    include_distributional_state: bool = True
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RegimeLearningConfig:
    output_dir: str = "outputs/hank_regime_learning_stage6"
    horizon: int = 60
    gamma: float = 0.99
    lambda_y: float = 0.5
    lambda_i: float = 0.05
    action_bound: float = 0.0015
    classical_policy_mode: str = "switching"
    training_seeds: tuple[int, ...] = (11, 22)
    selection_seeds: tuple[int, ...] = (500, 501)
    evaluation_seeds: tuple[int, ...] = (700,)
    regime_config: RegimeSwitchingConfig = field(default_factory=RegimeSwitchingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def main_variants(self) -> list[RegimeLearningVariant]:
        return [
            RegimeLearningVariant(
                name=spec["name"],
                scenario_name=spec["name"],
                scenario_label=spec["label"],
                input_mode="belief_state",
                include_distributional_state=True,
                description=(
                    "Regime-switching HANK learning policy: filtered reduced state, "
                    "stress belief and lagged rate as policy input."
                ),
            )
            for spec in self.regime_config.scenario_specs()
        ]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["training_seeds"] = list(self.training_seeds)
        payload["selection_seeds"] = list(self.selection_seeds)
        payload["evaluation_seeds"] = list(self.evaluation_seeds)
        payload["main_variants"] = [variant.to_dict() for variant in self.main_variants()]
        return payload


def default_regime_learning_config() -> RegimeLearningConfig:
    return RegimeLearningConfig()
