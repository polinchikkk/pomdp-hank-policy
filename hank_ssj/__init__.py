"""Интерфейсный слой HANK/SSJ для проекта о распределительной информации.

Пакет :mod:`hank` содержит экономическое HANK-ядро. Этот пакет добавляет
тонкий слой для новой постановки: экспортированные якобианы, наблюдаемые
переменные, информационные состояния и будущие эксперименты с правилами
политики.
"""

from .artifacts import SSJArtifactSpec, export_long_jacobian_to_npz
from .filters import FilterBuildSpec, ScalarFilterParams, build_filtered_states
from .information_sets import InformationStateInputSpec, build_information_state_inputs
from .observables import HankObservableBuildSpec, build_hank_observable_panel
from .observations import ObservationNoiseSpec, build_noisy_observations
from .policy_evaluation import HankSSJPolicyEnvironment, PolicyLossWeights, SSJPolicyEvaluationSpec, TrajectoryLoss
from .shock_library import ShockLibrarySpec, StochasticPathSpec, build_shock_response_library, generate_stochastic_hank_paths

__all__ = [
    "FilterBuildSpec",
    "HankSSJPolicyEnvironment",
    "HankObservableBuildSpec",
    "InformationStateInputSpec",
    "ObservationNoiseSpec",
    "PolicyLossWeights",
    "ScalarFilterParams",
    "SSJPolicyEvaluationSpec",
    "SSJArtifactSpec",
    "ShockLibrarySpec",
    "StochasticPathSpec",
    "TrajectoryLoss",
    "build_shock_response_library",
    "build_filtered_states",
    "generate_stochastic_hank_paths",
    "build_information_state_inputs",
    "build_hank_observable_panel",
    "build_noisy_observations",
    "export_long_jacobian_to_npz",
]
