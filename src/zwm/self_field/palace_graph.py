from __future__ import annotations

from dataclasses import dataclass, field

from zwm.core.constants import (
    LUOSHU_CONFLICT_PAIRS,
    LUOSHU_DIRECTION_NAMES,
    LUOSHU_GENERATION_PAIRS,
    PALACE_POST_HEAVEN_BAGUA,
)


@dataclass(frozen=True, slots=True)
class PalaceNode:
    position: int
    luoshu_number: int
    bagua: str
    direction: str
    trigram_element: str | None = None

    @property
    def is_center(self) -> bool:
        return self.position == 5


@dataclass
class LuoshuGrid:
    nodes: dict[int, PalaceNode] = field(default_factory=dict)
    self_position: int = 5

    def __post_init__(self) -> None:
        if not self.nodes:
            self._build_palaces()

    def _build_palaces(self) -> None:
        for pos in range(1, 10):
            self.nodes[pos] = PalaceNode(
                position=pos,
                luoshu_number=pos,
                bagua=PALACE_POST_HEAVEN_BAGUA.get(pos, "中"),
                direction=LUOSHU_DIRECTION_NAMES.get(pos, "中"),
            )

    def is_generation_pair(self, p1: int, p2: int) -> bool:
        n1 = self.nodes[p1].luoshu_number
        n2 = self.nodes[p2].luoshu_number
        return (n1, n2) in LUOSHU_GENERATION_PAIRS

    def is_conflict_pair(self, p1: int, p2: int) -> bool:
        n1 = self.nodes[p1].luoshu_number
        n2 = self.nodes[p2].luoshu_number
        return (n1, n2) in LUOSHU_CONFLICT_PAIRS

    def generation_score(self, p1: int, p2: int) -> float:
        if p1 == p2:
            return 1.0
        if self.is_generation_pair(p1, p2):
            return 0.8
        if self.is_conflict_pair(p1, p2):
            return 0.1
        return 0.4

    def adjacent_palaces(self, position: int) -> list[int]:
        if position == 5:
            return [1, 2, 3, 4, 6, 7, 8, 9]
        adj_map: dict[int, list[int]] = {
            1: [5, 2, 8, 4], 2: [5, 1, 3, 9, 7, 8],
            3: [5, 2, 4, 6], 4: [5, 1, 3, 7],
            6: [5, 3, 7, 9], 7: [5, 4, 6, 8],
            8: [5, 1, 2, 7, 9, 6], 9: [5, 6, 8, 2],
        }
        return adj_map.get(position, [])

    def opposite_palace(self, position: int) -> int:
        return {1: 9, 9: 1, 2: 8, 8: 2, 3: 7, 7: 3, 4: 6, 6: 4, 5: 5}[position]

    def move_self(self, new_position: int) -> LuoshuGrid:
        new_grid = LuoshuGrid(nodes=self.nodes.copy(), self_position=new_position)
        return new_grid

    def directional_message(
        self,
        direction_palace: int,
        hexagram_resonance: float,
        time_potential: float,
    ) -> float:
        gen_score = self.generation_score(self.self_position, direction_palace)
        return gen_score * hexagram_resonance * time_potential
