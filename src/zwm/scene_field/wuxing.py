from __future__ import annotations

from zwm.core.constants import (
    ELEMENT_CONTROL,
    ELEMENT_GENERATION,
    ELEMENT_REVERSE_CONTROL,
    TRIGRAM_ELEMENTS,
)
from zwm.core.hexagram import Hexagram


def hexagram_element_profile(h: Hexagram) -> dict[str, float]:
    lower_elem = TRIGRAM_ELEMENTS[h.lower_trigram.index]
    upper_elem = TRIGRAM_ELEMENTS[h.upper_trigram.index]

    profile: dict[str, float] = {"金": 0.0, "木": 0.0, "水": 0.0, "火": 0.0, "土": 0.0}
    profile[lower_elem] += 0.5
    profile[upper_elem] += 0.5
    return profile


def element_force(h1: Hexagram, h2: Hexagram) -> float:
    p1 = hexagram_element_profile(h1)
    p2 = hexagram_element_profile(h2)

    force = 0.0
    for e1, w1 in p1.items():
        if w1 == 0:
            continue
        for e2, w2 in p2.items():
            if w2 == 0:
                continue
            if e1 == e2:
                force += w1 * w2 * 0.5
            elif ELEMENT_GENERATION.get(e1) == e2:
                force += w1 * w2 * 1.0
            elif ELEMENT_CONTROL.get(e1) == e2:
                force -= w1 * w2 * 1.0
            elif ELEMENT_REVERSE_CONTROL.get(e1) == e2:
                force += w1 * w2 * 0.3
    return float(max(-1.0, min(1.0, force)))


def control_network() -> dict[str, str]:
    """P0 — 五行相克网: 4 步生克路径评分。

    之前是死代码; 现在接到 ``risk_expert`` 的"内势相克张力"项。
    返回 4 步相克路径 (e1→克e2→克e3→克e4) 评分, 用来度量
    卦象内部 5 行的克制环路强度 (0.0–1.0)。"""
    # 4 步相克路径 (e1→克e2→克e3→克e4)
    starts = list(ELEMENT_CONTROL.keys())
    network: dict[str, str] = {}
    for e in starts:
        nxt = ELEMENT_CONTROL.get(e, e)
        network[e] = nxt
    return network


def generation_chain(elem: str) -> list[str]:
    """P0 — 五行相生链: 4 步生路径, 接到 ``narrative_expert`` 的"势能推进"项。

    之前是死代码; 现在用来度量卦象上下卦五行之间的生助方向
    是否沿相生链推进 (木→火→土→金→水→木)。"""
    chain = [elem]
    current = elem
    for _ in range(4):
        current = ELEMENT_GENERATION.get(current, current)
        if current == elem:
            break
        chain.append(current)
    return chain
