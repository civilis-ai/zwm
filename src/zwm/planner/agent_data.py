"""P2-1 (audit) — TickReport / TickPrediction 数据契约。

把它们从 ``agent.py`` 抽出,使 agent.py 体积可控,且其它模块可以
零依赖地引用 telemetry 容器。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from zwm.core.hexagram import Hexagram
from zwm.planner.loop import PlanResult
from zwm.scene_field.unified_field import UnifiedField


# Reward at/above this is treated as a "good" outcome worth consolidating
# and reinforcing through the learners.
GOOD_OUTCOME = 0.5


@dataclass(frozen=True, slots=True)
class TickReport:
    """Everything one OODA tick produced — returned for inspection/logging."""

    plan: PlanResult
    h_current: Hexagram
    h_next: Hexagram
    reward: float
    jepa_loss: float | None
    router_loss: float | None
    episode_id: int
    surprise: float = 0.0
    # P3: Mutation classification (e.g. "初爻变", "六爻全变") and codon
    # mapping for the current hexagram, wired from mutations.py + codon.py.
    mutation_class: str = ""
    codon: str = ""
    codon_aa: str = ""

    # Convenience pass-throughs so callers can treat a TickReport like a result.
    @property
    def top_mutation(self) -> int:
        return self.plan.top_mutation

    @property
    def top_score(self) -> float:
        return self.plan.top_score


class TickPrediction(NamedTuple):
    """Bundle returned by ``TrinityAgent._predict`` — feeds Phase 3 and 4.

    Attributes:
        z_world          106-dim world vector (77 joint + 29 unified field).
        z_pred           JEPA prediction at scale "short".
        z_var            Uncertainty estimate from variational JEPA, or None.
        world            UnifiedField snapshot (re-used by Phase 4).
        calendar_context Calendar + ganzhi time context (cached for reuse).
    """

    z_world: np.ndarray
    z_pred: np.ndarray
    z_var: np.ndarray | None
    world: UnifiedField
    calendar_context: dict
