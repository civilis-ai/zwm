from __future__ import annotations

from zwm.core.constants import TRIGRAM_ELEMENTS
from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid


def determine_six_relations(
    h: Hexagram,
    grid: LuoshuGrid,
    self_element: str | None = None,
) -> dict[int, str]:
    if self_element is None:
        self_element = TRIGRAM_ELEMENTS.get(h.lower_trigram.index, "土")

    from zwm.core.constants import (
        ELEMENT_CONTROL,
        ELEMENT_GENERATION,
        ELEMENT_REVERSE_CONTROL,
    )

    relations: dict[int, str] = {grid.self_position: "我"}

    for pos in range(1, 10):
        if pos == grid.self_position:
            continue
        palace_elem = TRIGRAM_ELEMENTS.get(pos % 8, "土")

        if palace_elem == self_element:
            relations[pos] = "兄弟"
        elif ELEMENT_GENERATION.get(palace_elem) == self_element:
            relations[pos] = "父母"
        elif ELEMENT_GENERATION.get(self_element) == palace_elem:
            relations[pos] = "子孙"
        elif ELEMENT_CONTROL.get(palace_elem) == self_element:
            relations[pos] = "官鬼"
        elif ELEMENT_CONTROL.get(self_element) == palace_elem:
            relations[pos] = "妻财"
        else:
            relations[pos] = "兄弟"

    return relations


def social_field_vector(
    h: Hexagram,
    grid: LuoshuGrid,
) -> dict[str, list[int]]:
    relations = determine_six_relations(h, grid)
    field: dict[str, list[int]] = {
        "我": [], "父母": [], "兄弟": [],
        "妻财": [], "官鬼": [], "子孙": [],
    }
    for pos, role in relations.items():
        field[role].append(pos)
    return field
