from __future__ import annotations

import numpy as np

from zwm.core.hexagram import Hexagram
from zwm.moe.experts import (
    element_expert,
    narrative_expert,
    risk_expert,
    social_expert,
    space_expert,
    time_expert,
)
from zwm.moe.router import MoERouter
from zwm.self_field.palace_graph import LuoshuGrid


class SparseMoE:
    def __init__(self, top_k: int = 3) -> None:
        self._router = MoERouter()
        self._top_k = top_k
        self._expert_names = [
            "time", "space", "social",
            "element", "risk", "narrative",
        ]

    @property
    def router(self) -> MoERouter:
        return self._router

    @property
    def expert_names(self) -> list[str]:
        return list(self._expert_names)

    def evaluate(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
        target_palace: int,
        preference_weights: dict[str, float] | None = None,
    ) -> float:
        weights = self._router.route(h, grid, time_phase)
        if preference_weights is not None:
            pref = np.array(
                [preference_weights.get(name, 0.0) for name in self._expert_names],
                dtype=np.float32,
            )
            # Multiplicatively bias the learned router by the agent's
            # accumulated expert preferences, then renormalise.
            weights = weights * (1.0 + pref)
            total = weights.sum()
            if total > 1e-10:
                weights = weights / total
        threshold = np.sort(weights)[-self._top_k]
        mask = weights >= threshold

        scores = np.zeros(6, dtype=np.float32)
        if mask[0]:
            scores[0] = time_expert(h, time_phase)
        if mask[1]:
            scores[1] = space_expert(h, target_palace)
        if mask[2]:
            scores[2] = social_expert(h, grid, target_palace)
        if mask[3]:
            scores[3] = element_expert(h, h.lower_trigram.element)
        if mask[4]:
            scores[4] = risk_expert(h)
        if mask[5]:
            scores[5] = narrative_expert(h)

        active_weights = weights * mask.astype(np.float32)
        if active_weights.sum() < 1e-10:
            return float(np.mean(scores))
        return float(np.dot(active_weights, scores) / active_weights.sum())

    def active_experts(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
    ) -> list[str]:
        weights = self._router.route(h, grid, time_phase)
        threshold = np.sort(weights)[-self._top_k]
        return [
            self._expert_names[i]
            for i in range(6)
            if weights[i] >= threshold
        ]
