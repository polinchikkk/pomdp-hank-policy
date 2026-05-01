from __future__ import annotations

"""Эксперимент 3: разрыв между агрегатной информацией и полной информацией."""


def sufficiency_gap(aggregate_loss: float, full_information_loss: float) -> float:
    return float(aggregate_loss - full_information_loss)
