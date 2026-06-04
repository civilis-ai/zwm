from __future__ import annotations

from dataclasses import dataclass, field

from zwm.core.hexagram import Hexagram
from zwm.langevin.sampler import LangevinSampler
from zwm.moe.sparse_activation import SparseMoE
from zwm.self_field.palace_graph import LuoshuGrid
from zwm.scene_field.five_hexagrams import FiveHexagramChain


@dataclass
class PlanResult:
    chain: FiveHexagramChain
    hexagram_scores: list[tuple[int, float]]
    top_mutation: int
    top_score: float
    moe_active_experts: list[str]
    trajectory: list[tuple[str, float]]


class TrinityPlanner:
    def __init__(self) -> None:
        self._sampler = LangevinSampler()
        self._moe = SparseMoE(top_k=3)

    def plan(
        self,
        h_current: Hexagram,
        grid: LuoshuGrid | None = None,
        time_phase: float = 0.0,
        top_k: int = 5,
    ) -> PlanResult:
        if grid is None:
            grid = LuoshuGrid()

        scores = self._sampler.top_k_mutations(h_current, k=top_k)
        top_h, top_mask, top_score = scores[0]

        chain = FiveHexagramChain.with_evolution(h_current, top_mask)

        active = self._moe.active_experts(h_current, grid, time_phase)

        trajectory: list[tuple[str, float]] = []
        for h, mask, score in scores[:3]:
            h_mut = h_current.mutate(mask)
            trajectory.append((h_mut.name, score))

        return PlanResult(
            chain=chain,
            hexagram_scores=[(s[1], s[2]) for s in scores],
            top_mutation=top_mask,
            top_score=top_score,
            moe_active_experts=active,
            trajectory=trajectory,
        )

    def observe_predict_evaluate_act(
        self,
        h_current: Hexagram,
        grid: LuoshuGrid | None = None,
        time_phase: float = 0.0,
    ) -> PlanResult:
        return self.plan(h_current, grid, time_phase)
