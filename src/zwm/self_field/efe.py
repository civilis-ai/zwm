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
    intrinsic_fn=None,
    log_evidence: float | None = None,
) -> float:
    """Expected Free Energy: pragmatic_value + beta * epistemic + intrinsic.

    ``intrinsic_fn`` is a caller-supplied world-model signal (e.g. JEPA
    surprise) that lets the search be steered by the predictive model — the
    intrinsic is a *positive* term, so a high intrinsic pulls the planner
    toward states the world model is curious about. When ``intrinsic_fn`` is
    None, the term is 0.0 and EFE reduces to its classical form.
    """
    pragmatic = pragmatic_value(h, grid, target_palace)

    # Strategic-1: Blend learned JEPA log-evidence with analytical harmony
    if log_evidence is not None:
        pragmatic = 0.6 * pragmatic + 0.4 * log_evidence
    epistemic = epistemic_value(
        h, grid, visit_counts, total_visits, palace_visit_counts
    )
    intrinsic = float(intrinsic_fn(h)) if intrinsic_fn is not None else 0.0
    return float(pragmatic + beta_curiosity * epistemic + intrinsic)


def preferred_prior_distribution(
    grid: LuoshuGrid,
    target_palace: int,
) -> np.ndarray:
    """P0 — 偏好先验分布: 基于 ``pragmatic_value`` 的 softmax(和谐度)。

    之前是死代码; 现在被 ``TrinityAgent._combined_priors`` 调用, 作为
    一类"基于当前空间-目标"的高层先验, 与 Hebbian 在线先验和 episodic
    memory 先验并联加权 — 这样 MCTS 不仅被"记忆"驱动, 也被"目标
    空间布局"驱动。"""
    probs = np.zeros(64, dtype=np.float32)
    from zwm.core.hexagram import all_hexagrams
    for h in all_hexagrams():
        harmony = pragmatic_value(h, grid, target_palace)
        probs[h.normal_order] = math.exp(harmony)
    if probs.sum() < 1e-10:
        return np.ones(64, dtype=np.float32) / 64.0
    return probs / probs.sum()
