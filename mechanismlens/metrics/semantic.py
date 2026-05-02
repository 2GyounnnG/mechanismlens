"""Semantic consistency helpers."""

from __future__ import annotations

from collections.abc import Sequence


def label_agreement(expected: Sequence[str], observed: Sequence[str]) -> float:
    total = min(len(expected), len(observed))
    if total == 0:
        return 1.0
    return sum(1 for lhs, rhs in zip(expected, observed, strict=False) if lhs == rhs) / total
