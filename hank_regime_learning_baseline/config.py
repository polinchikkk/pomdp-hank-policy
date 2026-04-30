from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
    output_dir: str = "outputs/information_state_design_main"
    horizon: int = 60
    gamma: float = 0.99
    lambda_y: float = 0.5
    lambda_i: float = 0.05
    action_bound: float = 0.0015
    classical_policy_mode: str = "switching"
    training_seeds: tuple[int, ...] = ()
    selection_seeds: tuple[int, ...] = tuple(range(500, 510))
    evaluation_seeds: tuple[int, ...] = tuple(range(900, 950))
    regime_config: RegimeSwitchingConfig = field(default_factory=RegimeSwitchingConfig)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["training_seeds"] = list(self.training_seeds)
        payload["selection_seeds"] = list(self.selection_seeds)
        payload["evaluation_seeds"] = list(self.evaluation_seeds)
        return payload


def default_regime_learning_config() -> RegimeLearningConfig:
    return RegimeLearningConfig()
