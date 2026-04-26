from .config import RegimeLearningConfig, default_regime_learning_config

__all__ = [
    "run_core_matrix",
    "run_architecture_ablation",
    "RegimeLearningConfig",
    "default_regime_learning_config",
    "run_environment_shift",
    "run_misspecification_map",
    "run_full_hank_projection_from_policy_paths",
    "run_policy_extension_experiments",
    "run_reduced_state_validation",
    "run_pipeline",
    "run_stage6_diagnostics",
    "run_stage6_summary",
    "run_best_candidate_validation_suite",
    "run_deep_validation",
    "run_universal_rawobs_misspecified_tuning",
]


def __getattr__(name: str):
    if name == "run_core_matrix":
        from .core_matrix import run_core_matrix

        return run_core_matrix
    if name == "run_architecture_ablation":
        from .architecture_ablation import run_architecture_ablation

        return run_architecture_ablation
    if name == "run_environment_shift":
        from .environment_shift import run_environment_shift

        return run_environment_shift
    if name == "run_misspecification_map":
        from .misspecification_map import run_misspecification_map

        return run_misspecification_map
    if name == "run_policy_extension_experiments":
        from .policy_extensions import run_policy_extension_experiments

        return run_policy_extension_experiments
    if name == "run_full_hank_projection_from_policy_paths":
        from .policy_extensions import run_full_hank_projection_from_policy_paths

        return run_full_hank_projection_from_policy_paths
    if name == "run_reduced_state_validation":
        from .reduced_state_validation import run_reduced_state_validation

        return run_reduced_state_validation
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    if name == "run_stage6_diagnostics":
        from .stage6_diagnostics import run_stage6_diagnostics

        return run_stage6_diagnostics
    if name == "run_stage6_summary":
        from .stage6_summary import run_stage6_summary

        return run_stage6_summary
    if name == "run_best_candidate_validation_suite":
        from .tuning import run_best_candidate_validation_suite

        return run_best_candidate_validation_suite
    if name == "run_universal_rawobs_misspecified_tuning":
        from .tuning import run_universal_rawobs_misspecified_tuning

        return run_universal_rawobs_misspecified_tuning
    if name == "run_deep_validation":
        from .validation import run_deep_validation

        return run_deep_validation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
