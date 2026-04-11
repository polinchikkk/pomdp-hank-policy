"""Learning-based policy layer for partial-information HANK."""

from .config import Stage4Config, default_stage4_config

__all__ = ["Stage4Config", "default_stage4_config", "run_pipeline"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
