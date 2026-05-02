"""Decision consistency helpers."""

from __future__ import annotations

from collections.abc import Sequence


def action_switch_rate(actions: Sequence[str]) -> float:
    if len(actions) < 2:
        return 0.0
    switches = sum(1 for prev, cur in zip(actions, actions[1:], strict=False) if prev != cur)
    return switches / (len(actions) - 1)
