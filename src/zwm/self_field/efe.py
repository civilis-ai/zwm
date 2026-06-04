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
    palace_visit_counts: dict[int, int] | None = None,
) -> float:
    """Information-seeking value: curiosity over states + unexplored palaces.

    ``visit_counts`` is keyed by hexagram identity (normal_order, 0-63) and
    drives the state-novelty term. ``palace_visit_counts`` is keyed by palace
    position (1-9) and drives the unexplored-palace bonus. These are kept in
    separate dicts on purpose — their key spaces overlap (1-9), so conflating
    them silently corrupts the palace term.
    """
    from zwm.self_field.harmony import luoshu_harmony

    palace_visit_counts = palace_visit_counts or {}

    unknown_bonus = 0.0
    for pos in range(1, 10):
        if pos == grid.self_position:
            continue
        if palace_visit_counts.get(pos, 0) == 0:
            unknown_bonus += 0.1 * luoshu_harmony(h, grid, pos)

    h_visits = visit_counts.get(h.normal_order, 0)
    novelty = 1.0 / (1.0 + h_visits / max(total_visits, 1))

    return float(unknown_bonus + 0.2 * novelty)


def expected_free_energy(
    h: Hexagram,
    grid: LuoshuGrid,
    target_palace: int,
    visit_counts: dict[int, int],
    total_visits: int = 1,
    beta_curiosity: float = 0.3,
    palace_visit_counts: dict[int, int] | None = None,
) -> float:
    pragmatic = pragmatic_value(h, grid, target_palace)
    epistemic = epistemic_value(
        h, grid, visit_counts, total_visits, palace_visit_counts
    )
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
