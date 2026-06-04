from __future__ import annotations

from zwm.core.hexagram import Hexagram


MUTATION_NAMES: dict[int, str] = {
    0x01: "初爻变", 0x02: "二爻变", 0x04: "三爻变",
    0x08: "四爻变", 0x10: "五爻变", 0x20: "上爻变",
    0x03: "初二爻变", 0x07: "初三爻变", 0x0F: "初四爻变",
    0x1F: "初五爻变", 0x3F: "六爻全变",
}


def all_mutations() -> list[int]:
    return list(range(64))


def single_yao_mutations() -> list[int]:
    return [0x01, 0x02, 0x04, 0x08, 0x10, 0x20]


def apply_mutation(h: Hexagram, mask: int) -> Hexagram:
    return h.mutate(mask)


def classify_mutation(mask: int) -> str:
    if mask == 0:
        return "不变"
    if mask == 0x3F:
        return "六爻全变 (错卦)"
    count = mask.bit_count()
    if count == 1:
        return MUTATION_NAMES.get(mask, f"{count}爻变")
    return f"{count}爻变"


def mutation_path(s0: Hexagram, masks: list[int]) -> list[Hexagram]:
    path = [s0]
    current = s0
    for mask in masks:
        current = current.mutate(mask)
        path.append(current)
    return path


def all_successors(h: Hexagram) -> dict[int, Hexagram]:
    return {mask: h.mutate(mask) for mask in range(64)}
