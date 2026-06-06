from __future__ import annotations

import numpy as np

from zwm.core.constants import (
    ELEMENT_CONTROL,
    ELEMENT_GENERATION,
    TRIGRAM_ELEMENTS,
)
from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid


def element_affinity(h: Hexagram, target_element: str) -> float:
    lower_elem = TRIGRAM_ELEMENTS[h.lower_trigram.index]
    upper_elem = TRIGRAM_ELEMENTS[h.upper_trigram.index]

    score = 0.0
    for elem in (lower_elem, upper_elem):
        if elem == target_element:
            score += 0.5
        elif ELEMENT_GENERATION.get(elem) == target_element:
            score += 0.3
        elif ELEMENT_CONTROL.get(elem) == target_element:
            score -= 0.2
    return max(-1.0, min(1.0, score))


def luoshu_harmony(
    h: Hexagram,
    grid: LuoshuGrid,
    target_palace: int,
) -> float:
    gen_score = grid.generation_score(grid.self_position, target_palace)
    elem_score = element_affinity(
        h,
        TRIGRAM_ELEMENTS.get(h.lower_trigram.index, "土"),
    )
    return 0.5 * gen_score + 0.5 * (elem_score + 1.0) / 2.0


def compute_self_field(
    h: Hexagram,
    grid: LuoshuGrid,
    time_potentials: dict[int, float],
) -> dict[int, float]:
    field: dict[int, float] = {}
    for pos in range(1, 10):
        if pos == grid.self_position:
            field[pos] = 1.0
            continue
        harmony = luoshu_harmony(h, grid, pos)
        time_pot = time_potentials.get(pos, 0.5)
        field[pos] = harmony * time_pot
    return field


def self_field_tensor(
    h: Hexagram,
    grid: LuoshuGrid,
    time_potentials: dict[int, float],
) -> np.ndarray:
    field = compute_self_field(h, grid, time_potentials)
    tensor = np.zeros(9, dtype=np.float32)
    for pos, val in field.items():
        tensor[pos - 1] = val
    return tensor
