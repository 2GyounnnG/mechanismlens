"""Causal examiner wrapper."""

from __future__ import annotations

from mechanismlens.metrics.causal import intervention_locality_score
from mechanismlens.schema import Trajectory


class CausalExaminer:
    def examine(
        self,
        predicted_base: Trajectory,
        predicted_intervened: Trajectory,
        affected_object_ids: set[str],
    ) -> dict[str, float]:
        return intervention_locality_score(predicted_base, predicted_intervened, affected_object_ids)
