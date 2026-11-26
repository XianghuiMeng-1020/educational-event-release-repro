"""Mastery / progression policy locked for G3 validation."""
from __future__ import annotations

from typing import Iterable


def mastery_decision(probability: float, threshold: float = 0.80) -> str:
    return "progress" if float(probability) >= float(threshold) else "review"


def mastery_decision_set(probabilities: Iterable[float], threshold: float = 0.80) -> list[str]:
    return sorted({mastery_decision(p, threshold) for p in probabilities})
