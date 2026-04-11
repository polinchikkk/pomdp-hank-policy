"""Partial-information HANK baseline built on top of the full HANK model."""

from .config import HANKPartialInfoConfig, default_partial_info_config

__all__ = ["HANKPartialInfoConfig", "default_partial_info_config", "run_pipeline"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
