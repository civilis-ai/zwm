from __future__ import annotations

from zwm.core.constants import (
    ELEMENT_CONTROL,
    ELEMENT_GENERATION,
    ELEMENT_REVERSE_CONTROL,
    GAN_ELEMENT,
    PALACE_ELEMENT,
    _HEXAGRAM_TO_PALACE,
)
from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid


def self_element_from_day_gan(day_gan: str) -> str:
    """从日干获取五行'我'元素."""
    if day_gan not in GAN_ELEMENT:
        raise ValueError(
            f"Unknown day stem: '{day_gan}'. Must be one of: "
            f"{', '.join(GAN_ELEMENT.keys())}"
        )
    return GAN_ELEMENT[day_gan]


def hexagram_palace_index(h: Hexagram) -> int:
    """返回卦所属八宫的纯卦 trigram index."""
    return _HEXAGRAM_TO_PALACE[h.normal_order]


def hexagram_palace_element(h: Hexagram) -> str:
    """返回卦所属八宫的五行."""
    return PALACE_ELEMENT[hexagram_palace_index(h)]


def determine_six_relations(
    h: Hexagram,
    grid: LuoshuGrid,
    self_element: str | None = None,
    day_gan: str | None = None,
) -> dict[int, str]:
    """Determine 六亲 (six relations) for each Luoshu palace position.

    If `day_gan` is provided, it is used as the 太极点 (self reference):
    the self element is derived from the Day Heavenly Stem element.
    Otherwise, `self_element` can be provided directly.
    Falls back to the hexagram's palace element if neither is given.

    Args:
        h: Current hexagram.
        grid: Luoshu grid with self_position.
        self_element: Explicit self element name.
        day_gan: Day Heavenly Stem (日干) — primary mechanism for 太极点定位.

    Returns:
        dict mapping palace position → relation role.
    """
    if day_gan is not None:
        self_elem = self_element_from_day_gan(day_gan)
    elif self_element is not None:
        self_elem = self_element
    else:
        self_elem = hexagram_palace_element(h)

    relations: dict[int, str] = {grid.self_position: "我"}

    for pos in range(1, 10):
        if pos == grid.self_position:
            continue
        # 卦宫五行 determined by palace trigram index
        palace_trigram_idx = pos % 8
        palace_elem = PALACE_ELEMENT.get(palace_trigram_idx, "土")

        if palace_elem == self_elem:
            relations[pos] = "兄弟"
        elif ELEMENT_GENERATION.get(palace_elem) == self_elem:
            relations[pos] = "父母"
        elif ELEMENT_GENERATION.get(self_elem) == palace_elem:
            relations[pos] = "子孙"
        elif ELEMENT_CONTROL.get(palace_elem) == self_elem:
            relations[pos] = "官鬼"
        elif ELEMENT_CONTROL.get(self_elem) == palace_elem:
            relations[pos] = "妻财"
        elif ELEMENT_REVERSE_CONTROL.get(self_elem) == palace_elem:
            relations[pos] = "官鬼"
        else:
            relations[pos] = "兄弟"

    return relations


def social_field_vector(
    h: Hexagram,
    grid: LuoshuGrid,
    day_gan: str | None = None,
) -> dict[str, list[int]]:
    """Build social field vector grouping positions by relation type.

    Args:
        h: Current hexagram.
        grid: Luoshu grid.
        day_gan: Day Heavenly Stem for 太极点 self-positioning.

    Returns:
        dict mapping relation role → list of palace positions.
    """
    relations = determine_six_relations(h, grid, day_gan=day_gan)
    field: dict[str, list[int]] = {
        "我": [], "父母": [], "兄弟": [],
        "妻财": [], "官鬼": [], "子孙": [],
    }
    for pos, role in relations.items():
        field[role].append(pos)
    return field
