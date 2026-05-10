"""Интерфейсный слой HANK/SSJ для проекта о распределительной информации.

Пакет :mod:`hank` содержит экономическое HANK-ядро. Этот пакет добавляет
тонкий слой для новой постановки: экспортированные якобианы, наблюдаемые
переменные, информационные состояния и будущие эксперименты с правилами
политики.
"""

from .artifacts import SSJArtifactSpec, export_long_jacobian_to_npz
from .closed_loop_environment import ClosedLoopDiagnostics, ClosedLoopResult, ClosedLoopSSJEnvironment
from .distributional_jacobians import (
    DistributionalJacobianSpec,
    augment_jacobians_with_distributional_policy_responses,
    has_direct_distributional_jacobians,
    required_distributional_jacobian_keys,
)
from .filters import FilterBuildSpec, ScalarFilterParams, build_filtered_states
from .information_sets import InformationStateInputSpec, build_information_state_inputs
from .kalman_filter import (
    DEFAULT_STATE_NAMES,
    JointKalmanBuildSpec,
    build_joint_kalman_filtered_states,
    build_observation_matrix,
    fit_state_transition,
    run_kalman_filter,
)
from .observables import HankObservableBuildSpec, build_hank_observable_panel
from .observations import ObservationNoiseSpec, build_noisy_observations
from .policy_evaluation import HankSSJPolicyEnvironment, PolicyLossWeights, SSJPolicyEvaluationSpec, TrajectoryLoss
from .shock_library import ShockLibrarySpec, StochasticPathSpec, build_shock_response_library, generate_stochastic_hank_paths
from .state_space import StateSpaceSpec

__all__ = [
    "FilterBuildSpec",
    "ClosedLoopDiagnostics",
    "ClosedLoopResult",
    "ClosedLoopSSJEnvironment",
    "DEFAULT_STATE_NAMES",
    "DistributionalJacobianSpec",
    "HankSSJPolicyEnvironment",
    "HankObservableBuildSpec",
    "InformationStateInputSpec",
    "JointKalmanBuildSpec",
    "ObservationNoiseSpec",
    "PolicyLossWeights",
    "ScalarFilterParams",
    "SSJPolicyEvaluationSpec",
    "SSJArtifactSpec",
    "ShockLibrarySpec",
    "StateSpaceSpec",
    "StochasticPathSpec",
    "TrajectoryLoss",
    "build_shock_response_library",
    "augment_jacobians_with_distributional_policy_responses",
    "build_filtered_states",
    "build_joint_kalman_filtered_states",
    "generate_stochastic_hank_paths",
    "build_information_state_inputs",
    "build_hank_observable_panel",
    "build_observation_matrix",
    "build_noisy_observations",
    "export_long_jacobian_to_npz",
    "fit_state_transition",
    "has_direct_distributional_jacobians",
    "required_distributional_jacobian_keys",
    "run_kalman_filter",
]
