"""HANK-ядро для анализа распределительной информации в правилах ставки."""

from .calibration import HANKCalibration, default_calibration

__all__ = ["HANKCalibration", "default_calibration", "run_pipeline"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
