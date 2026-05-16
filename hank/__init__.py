"""HANK-ядро для анализа распределительной информации в правилах ставки."""

from .calibration import HANKCalibration, default_calibration

__all__ = ["HANKCalibration", "default_calibration", "run_pipeline", "write_hank_core_audit"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    if name == "write_hank_core_audit":
        from .audit import write_hank_core_audit

        return write_hank_core_audit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
