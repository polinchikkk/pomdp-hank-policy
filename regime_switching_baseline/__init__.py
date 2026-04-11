"""Regime-switching reduced-state HANK baseline."""

from .regime_filter import SwitchingKalmanFilterResults, run_switching_kalman_filter
from .regime_model import (
    RegimeSwitchingConfig,
    RegimeSwitchingModel,
    build_regime_switching_model,
    regime_model_spec_payload,
)

__all__ = [
    "RegimePolicyRun",
    "RegimeSwitchingConfig",
    "RegimeSwitchingModel",
    "SwitchingKalmanFilterResults",
    "build_regime_switching_model",
    "evaluate_policy_under_regime_uncertainty",
    "evaluate_regime_filter",
    "generate_regime_observations",
    "regime_model_spec_payload",
    "run_pipeline",
    "run_stage4_pipeline",
    "run_switching_kalman_filter",
    "simulate_filtered_policy",
    "simulate_full_information_policy",
    "simulate_hidden_regimes",
]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    if name == "run_stage4_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    if name == "evaluate_policy_under_regime_uncertainty":
        from .regime_evaluation import evaluate_policy_under_regime_uncertainty

        return evaluate_policy_under_regime_uncertainty
    if name == "evaluate_regime_filter":
        from .regime_evaluation import evaluate_regime_filter

        return evaluate_regime_filter
    if name == "RegimePolicyRun":
        from .regime_simulation import RegimePolicyRun

        return RegimePolicyRun
    if name == "generate_regime_observations":
        from .regime_simulation import generate_regime_observations

        return generate_regime_observations
    if name == "simulate_filtered_policy":
        from .regime_simulation import simulate_filtered_policy

        return simulate_filtered_policy
    if name == "simulate_full_information_policy":
        from .regime_simulation import simulate_full_information_policy

        return simulate_full_information_policy
    if name == "simulate_hidden_regimes":
        from .regime_simulation import simulate_hidden_regimes

        return simulate_hidden_regimes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
