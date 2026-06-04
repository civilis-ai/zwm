from __future__ import annotations

import math

import numpy as np

from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid


def pragmatic_value(
    h: Hexagram,
    grid: LuoshuGrid,
    target_palace: int,
    preference_temperature: float = 1.0,
) -> float:
    from zwm.self_field.harmony import luoshu_harmony

    harmony = luoshu_harmony(h, grid, target_palace)
    return float(harmony / preference_temperature)


def epistemic_value(
    h: Hexagram,
    grid: LuoshuGrid,
    visit_counts: dict[int, int],
    total_visits: int = 1,
) -> float:
    from zwm.core.hexagram import all_hexagrams
    from zwm.self_field.harmony import luoshu_harmony

    unknown_bonus = 0.0
    for pos in range(1, 10):
        if pos == grid.self_position:
            continue
        visits = visit_counts.get(pos, 0)
        if visits == 0:
            unknown_bonus += 0.1 * luoshu_harmony(h, grid, pos)

    try:
        h_bits = h.normal_order
        h_visits = visit_counts.get(h_bits, 0)
        novelty = 1.0 / (1.0 + h_visits / max(total_visits, 1))
    except (ZeroDivisionError, OverflowError):
        novelty = 1.0

    return float(unknown_bonus + 0.2 * novelty)


def expected_free_energy(
    h: Hexagram,
    grid: LuoshuGrid,
    target_palace: int,
    visit_counts: dict[int, int],
    total_visits: int = 1,
    beta_curiosity: float = 0.3,
) -> float:
    pragmatic = pragmatic_value(h, grid, target_palace)
    epistemic = epistemic_value(h, grid, visit_counts, total_visits)
    return float(pragmatic + beta_curiosity * epistemic)


def preferred_prior_distribution(
    grid: LuoshuGrid,
    target_palace: int,
) -> np.ndarray:
    probs = np.zeros(64, dtype=np.float32)
    from zwm.core.hexagram import all_hexagrams
    for h in all_hexagrams():
        harmony = pragmatic_value(h, grid, target_palace)
        probs[h.normal_order] = math.exp(harmony)
    if probs.sum() < 1e-10:
        return np.ones(64, dtype=np.float32) / 64.0
    return probs / probs.sum()
