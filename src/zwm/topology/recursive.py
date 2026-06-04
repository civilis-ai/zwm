"""Recursive nine-palace topology expansion (邵雍 九宫递归展开).

Implements hierarchical spatial decomposition:
  Level 0 → 1 palace  (中宫)
  Level 1 → 9 palaces (洛书网格 3×3)
  Level 2 → 81 palaces (each of 9 expanded into 9 sub-palaces)
  Level 3 → 729 palaces

Each node carries:
  - palace_position: 1-9 within its parent
  - luoshu_number: the Luoshu magic-square number (1-9)
  - absolute_path: tuple of positions from root to this node
  - children: 9 sub-palaces (or None at leaf level)

Uses:
  Strategic planning   → Level 2 (81-palace resolution)
  Tactical planning    → Level 1 (9-palace resolution)
  Operational planning → Level 0 (single-palace resolution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from zwm.core.constants import (
    LUOSHU_GENERATION_PAIRS,
    LUOSHU_CONFLICT_PAIRS,
    LUOSHU_DIRECTION_NAMES,
    PALACE_POST_HEAVEN_BAGUA,
)


@dataclass
class TopologyNode:
    palace_position: int                              # 1-9 within parent
    luoshu_number: int                                # 1-9 (洛书数)
    depth: int                                        # 0 = root, 1 = Level-1, etc.
    path: tuple[int, ...]                             # absolute path from root
    bagua: str                                        # 后天八卦 name
    direction: str                                    # compass direction
    children: list[TopologyNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def path_str(self) -> str:
        return ".".join(str(p) for p in self.path)

    def __repr__(self) -> str:
        return (
            f"TopologyNode(pos={self.palace_position}, "
            f"depth={self.depth}, bagua={self.bagua})"
        )


class RecursiveTopology:
    """邵雍-style recursive nine-palace topology.

    Generates a fractal spatial structure where each palace
    recursively contains 9 sub-palaces.
    """

    _LUOSHU_ORDER: ClassVar[list[int]] = [4, 9, 2, 3, 5, 7, 8, 1, 6]

    def __init__(self, max_depth: int = 3) -> None:
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        self._max_depth = max_depth
        self._root = self._build_node(
            palace_position=5,
            path=(),
            depth=0,
        )

    @property
    def root(self) -> TopologyNode:
        return self._root

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def _build_node(
        self,
        palace_position: int,
        path: tuple[int, ...],
        depth: int,
    ) -> TopologyNode:
        luoshu_num = palace_position
        bagua = PALACE_POST_HEAVEN_BAGUA.get(luoshu_num, "中")
        direction = LUOSHU_DIRECTION_NAMES.get(luoshu_num, "中")
        new_path = path + (palace_position,)

        node = TopologyNode(
            palace_position=palace_position,
            luoshu_number=luoshu_num,
            depth=depth,
            path=new_path,
            bagua=bagua,
            direction=direction,
        )

        if depth < self._max_depth:
            node.children = [
                self._build_node(
                    palace_position=pos,
                    path=new_path,
                    depth=depth + 1,
                )
                for pos in self._LUOSHU_ORDER
            ]

        return node

    def nodes_at_depth(self, depth: int) -> list[TopologyNode]:
        if depth == 0:
            return [self._root]
        result: list[TopologyNode] = []

        def _collect(n: TopologyNode) -> None:
            if n.depth == depth:
                result.append(n)
            else:
                for child in n.children:
                    _collect(child)

        _collect(self._root)
        return result

    def total_nodes(self) -> int:
        return sum(1 for _ in self.iter_nodes())

    def iter_nodes(self):
        stack = [self._root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def find_by_path(self, path: tuple[int, ...]) -> TopologyNode | None:
        node = self._root
        for pos in path:
            found = False
            for child in node.children:
                if child.palace_position == pos:
                    node = child
                    found = True
                    break
            if not found:
                return None
        return node

    def generation_pairs_at(self, depth: int) -> list[tuple[TopologyNode, TopologyNode]]:
        nodes = self.nodes_at_depth(depth)
        pairs: list[tuple[TopologyNode, TopologyNode]] = []
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                if (a.luoshu_number, b.luoshu_number) in LUOSHU_GENERATION_PAIRS:
                    pairs.append((a, b))
        return pairs

    def conflict_pairs_at(self, depth: int) -> list[tuple[TopologyNode, TopologyNode]]:
        nodes = self.nodes_at_depth(depth)
        pairs: list[tuple[TopologyNode, TopologyNode]] = []
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                if (a.luoshu_number, b.luoshu_number) in LUOSHU_CONFLICT_PAIRS:
                    pairs.append((a, b))
        return pairs

    def __repr__(self) -> str:
        return f"RecursiveTopology(max_depth={self._max_depth}, nodes={self.total_nodes()})"


def expand_topology(max_depth: int = 3) -> RecursiveTopology:
    """Create a recursive nine-palace topology.

    Args:
        max_depth: Maximum recursion depth (0=center only, 3=729 nodes).

    Returns:
        RecursiveTopology instance.
    """
    return RecursiveTopology(max_depth=max_depth)
