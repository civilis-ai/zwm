"""P2-1 (audit) — 记忆先验 / 世界向量 / 日历上下文。

从 ``agent.py`` 抽出,封装"过去发生的事如何影响下一个动作"的所有
只读 / 缓存逻辑 — 不修改持久态,只提供查询/聚合。

包含:
  * memory_priors          — VSA + 情节 + 语义 + 持久化 buffer 的先验聚合
  * _combined_priors       — Hebbian + 记忆 + Preferred-Prior 拼接去重
  * _next_palace_to_explore— 最小访问宫位 (epistemic drive over space)
  * _world_vector          — 106 维联合向量 (77 joint + 29 unified)
  * _calendar_context      — 干支历法上下文 (天)
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from zwm.scene_field.calendar import GanzhiTime
from zwm.scene_field.unified_field import UnifiedField
from zwm.self_field.palace_graph import LuoshuGrid

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zwm.core.hexagram import Hexagram
    from zwm.planner.agent import TrinityAgent


# ================= 记忆先验 =================
def memory_priors(agent: "TrinityAgent", h_current: "Hexagram", k: int = 5) -> dict[int, float]:
    """Mask priors learned from past episodes with similar starting states.

    Returns {mutation_mask: cumulative_reward}. Uses VSA similarity over the
    stored episode fingerprints, then reconstructs the mask that was taken.
    Also incorporates SemanticStore frequency data to boost rare transitions.
    """
    query = agent.vsa.encode_hexagram(h_current.normal_order)
    similar = agent.store.query_similar_vector(query, limit=k)
    priors: dict[int, float] = {}
    for ep in similar:
        main = ep.get("main_hex_bits")
        evolved = ep.get("evolved_hex_bits")
        if main is None or evolved is None:
            continue
        mask = (main ^ evolved) & 0b111111
        if 1 <= mask <= 63:
            priors[mask] = priors.get(mask, 0.0) + float(ep.get("reward", 0.0))

    # P0-3: Consume SemanticStore frequency data — boost rare transitions.
    if agent.semantic is not None:
        for mask in list(priors.keys()):
            target_bits = h_current.normal_order ^ mask
            freq = agent.semantic.get_frequency(target_bits)
            # Rare targets get a novelty bonus; common ones are dampened.
            novelty = 1.0 / (1.0 + freq)
            priors[mask] *= (0.5 + 0.5 * novelty)

        # P1-3: Consume SemanticStore associations — boost transitions
        # with strong historical co-occurrence.
        for mask in list(priors.keys()):
            target_bits = h_current.normal_order ^ mask
            assoc = agent.semantic.get_association(h_current.normal_order, target_bits)
            if assoc > 0:
                priors[mask] += assoc * 0.1

    # P1-6: Consume VSAMemoryBuffer.query() for associative recall.
    try:
        query_vec = agent.vsa.encode_hexagram(h_current.normal_order)
        recalled = agent.vsa_buffer.query(query_vec, k=k)
        for ep, _sim in recalled:
            mask = (h_current.normal_order ^ int(ep.reward * 63)) & 0b111111
            if 1 <= mask <= 63:
                priors[mask] = priors.get(mask, 0.0) + 0.1
    except Exception as exc:
        # AUDIT-S3: surface VSA buffer failures — they would otherwise
        # silently shrink the prior set and degrade planning quality.
        _log.warning("VSA buffer query failed: %s", exc, exc_info=False)

    # P2: query_by_outcome — boost priors from episodes with same outcome
    try:
        outcome_label = "吉"  # we're looking for good outcomes to boost
        good_episodes = agent.store.query_by_outcome(outcome_label, limit=k)
        for ep in good_episodes:
            main = ep.get("main_hex_bits")
            evolved = ep.get("evolved_hex_bits")
            if main is None or evolved is None:
                continue
            mask = (main ^ evolved) & 0b111111
            if 1 <= mask <= 63:
                priors[mask] = priors.get(mask, 0.0) + 0.05
    except Exception as exc:
        _log.warning("query_by_outcome(%s) failed: %s", outcome_label, exc)

    # P2-7: Consume the durable consolidated memory.
    try:
        query_vec = agent.vsa.encode_hexagram(h_current.normal_order)
        durable = agent.vsa_buffer.recall_consolidated(query_vec, top_k=k)
        for _vec, sim in durable:
            if sim <= 0.0:
                continue
            idx = agent.vsa.decode_to_hexagram(_vec)
            m = (h_current.normal_order ^ idx) & 0b111111
            if 1 <= m <= 63:
                priors[m] = priors.get(m, 0.0) + 0.05 * float(sim)
    except Exception as exc:
        _log.warning("vsa_buffer.recall_consolidated failed: %s", exc)

    return priors


def _combined_priors(agent: "TrinityAgent", h_current: "Hexagram") -> list[int]:
    """聚合 Hebbian + 情节 + Preferred-Prior 三路先验,去重后返回。

    Preferred-Prior 是 EFE pragmatic 项的 softmax 分布 (见
    ``preferred_prior_distribution``),它将空间目标 (top-K 宫位)
    转换为动作 mask,与"我曾见过什么"互补。
    """
    cur = h_current.normal_order
    masks: list[int] = []

    # Hebbian: high-association successor hexagrams -> the mask reaching them.
    for h2, _strength in agent.hebbian.suggest_next(cur, top_k=5):
        mask = (cur ^ h2) & 0b111111
        if 1 <= mask <= 63:
            masks.append(mask)

    # Episodic memory: masks that paid off from similar states.
    mem = memory_priors(agent, h_current)
    masks.extend(
        m for m, _w in sorted(mem.items(), key=lambda x: x[1], reverse=True)
    )

    # P0 — Preferred-prior distribution (EFE pragmatic softmax) of the
    # TOP-K hexagrams the agent wants to reach in the current grid.
    try:
        from zwm.self_field.efe import preferred_prior_distribution
        probs = preferred_prior_distribution(
            agent.grid, _next_palace_to_explore(agent)
        )
        top_targets = np.argsort(-probs)[:3]
        for tgt in top_targets:
            tgt = int(tgt)
            if tgt == cur:
                continue
            m = (cur ^ tgt) & 0b111111
            if 1 <= m <= 63:
                masks.append(m)
    except Exception as exc:
        # AUDIT-S3: prefer-prior failures used to vanish; now they
        # log at WARNING so an off-grid agent is observable.
        _log.warning("preferred_prior_distribution failed: %s", exc)

    # De-duplicate, preserving priority order.
    seen: set[int] = set()
    ordered: list[int] = []
    for m in masks:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _next_palace_to_explore(agent: "TrinityAgent", grid: LuoshuGrid | None = None) -> int:
    """Least-visited palace from the recursive topology scaffold.

    Uses depth-1 (9-palace) for the primary exploration target.  When
    the agent has visited all 9 palaces at least once, it switches to
    depth-2 (81-palace) for fine-grained spatial exploration.

    P0: agent.self_state 作为"我"的单一来源."我"永远在中宫(5),
    八方关系固定, 探索基于 palace_visits。
    """
    ss = agent.self_state
    # Depth-1: 9-palace exploration (排除中宫5)
    candidates_d1 = [
        n.palace_position
        for n in agent.topology.nodes_at_depth(1)
        if n.palace_position != 5
    ]
    if not candidates_d1:
        return 5
    # Check if all depth-1 palaces have been visited
    all_d1_visited = all(
        ss.palace_visits.get(p, 0) > 0 for p in candidates_d1
    )
    if all_d1_visited:
        # Depth-2: 81-palace fine-grained exploration
        candidates_d2 = [
            n for n in agent.topology.nodes_at_depth(2)
            if n.palace_position != 5
        ]
        if candidates_d2:
            d2_visits = {
                n.path_str: ss.palace_visits.get(hash(n.path_str) % 10000, 0)
                for n in candidates_d2
            }
            best = min(
                candidates_d2,
                key=lambda n: (d2_visits[n.path_str], n.palace_position),
            )
            return best.path[-2] if len(best.path) >= 2 else best.palace_position
    return min(
        candidates_d1,
        key=lambda p: (ss.palace_visits.get(p, 0), p),
    )


# ================= 世界向量 =================
def _world_vector(
    agent: "TrinityAgent",
    h: "Hexagram",
    time_phase: float,
    grid: LuoshuGrid,
    day_gan: str | None = None,
    calendar_context: dict | None = None,
) -> tuple[np.ndarray, UnifiedField]:
    """Return (full_world_vector, world_snapshot) tuple.

    支持两种 joint 类型:
      - FieldSquareCircularJoint → 使用缓存的 hex_field
      - SquareCircularJoint (旧) → 使用单卦 h
    """
    from zwm.jepa.field_gnn import FieldSquareCircularJoint

    if isinstance(agent.joint, FieldSquareCircularJoint):
        # 场模式: 使用缓存的卦象场
        hex_field = getattr(agent, "_last_hex_field", None)
        if hex_field is None:
            hex_field = np.zeros((64, 6), dtype=np.float32)
        z_joint = agent.joint.encode(hex_field, time_phase)
    else:
        z_joint = agent.joint.encode(h, time_phase)

    world = UnifiedField.snapshot(
        h, grid, time_phase,
        calendar_context=calendar_context,
        day_gan=day_gan,
    )
    z_unified = np.asarray(world.to_tensor(), dtype=np.float32)
    # Pad z_joint if needed to match JEPA input_dim
    target_77 = 77
    if len(z_joint) < target_77:
        z_joint = np.pad(z_joint, (0, target_77 - len(z_joint)),
                         'constant').astype(np.float32)
    else:
        z_joint = z_joint[:target_77].astype(np.float32)
    return np.concatenate([z_joint, z_unified]).astype(np.float32), world


def _world_vector_with_field(
    agent: "TrinityAgent",
    z_field: np.ndarray,        # (77,) 来自 FieldSquareCircularJoint.encode
    h_current: "Hexagram",
    grid: "LuoshuGrid",
    day_gan: str | None = None,
    calendar_context: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """构建包含卦象场的世界向量.

    将 FieldGNN 输出的 77-dim z_sq+cp 与 UnifiedField 的 29-dim
    拼接为完整的 106-dim z_world。这保持了与旧管道的完全兼容。
    """
    from zwm.scene_field.unified_field import UnifiedField
    from zwm.core.constants import Z_WORLD_DIM

    # UnifiedField 提供 29 维上下文
    uf = UnifiedField()
    _, world = uf.encode(
        h_current, grid, calendar_context=calendar_context,
        day_gan=day_gan,
    )
    # world dict → 29 维向量
    unified_vec = _world_dict_to_vector(world)

    # z_field = 77 dim (64 + 13), unified_vec = 29 dim → 106 dim
    # 但如果 z_field 已经包含了一些上下文, unified 可能部分冗余
    # 进一步统一: z_field 替代旧的 z_sq+cp, unified 保持不变
    z_field_flat = np.asarray(z_field, dtype=np.float32).flatten()
    unified_flat = np.asarray(unified_vec, dtype=np.float32).flatten()

    # 确保总维度为 Z_WORLD_DIM (106)
    if len(z_field_flat) + len(unified_flat) < Z_WORLD_DIM:
        pad = np.zeros(Z_WORLD_DIM - len(z_field_flat) - len(unified_flat), dtype=np.float32)
        z_world = np.concatenate([z_field_flat, unified_flat, pad])
    else:
        z_world = np.concatenate([z_field_flat, unified_flat])[:Z_WORLD_DIM]

    return z_world.astype(np.float32), world


def _world_dict_to_vector(world: dict) -> np.ndarray:
    """将 world dict 转为固定维度的 numpy 向量 (29-dim, 用于拼接).

    Supports both the new ``UnifiedField.encode()`` format (which
    provides ``unified_vec``) and the legacy format (which provides
    ``wuxing`` / ``liuqin`` keys).
    """
    # Fast path: if encode() already gave us the 29-dim vector, use it.
    unified_vec = world.get("unified_vec")
    if unified_vec is not None:
        vec = np.asarray(unified_vec, dtype=np.float32).flatten()
        if len(vec) >= 29:
            return vec[:29]
        # Pad if shorter than 29.
        return np.concatenate([vec, np.zeros(29 - len(vec), dtype=np.float32)])
    # Legacy path: reconstruct from individual keys.
    vec = np.zeros(29, dtype=np.float32)
    # 五行 (5) — try element_profile first, then wuxing
    elem_profile = world.get("element_profile", world.get("wuxing", {}))
    for i, elem in enumerate(["金", "木", "水", "火", "土"]):
        vec[i] = float(elem_profile.get(elem, 0.0))
    # 六亲 (9 + 1) — try six_relations first, then liuqin
    liuqin_map = world.get("six_relations", world.get("liuqin", {}))
    for i in range(9):
        vec[5 + i] = float(liuqin_map.get(i + 1, 0.0))
    # 其他 (calendar, self, etc) → 剩余 14 维
    # 填充为占位值, 后续由 UnifiedField 的真实输出覆盖
    return vec


def _calendar_context(
    agent: "TrinityAgent",
    year: int = 1, month: int = 1, day: int = 1, hour: int = 0,
    cosmic_phases: dict[str, float] | None = None,
) -> dict[str, float]:
    """消费宇宙时间信号 — 天地人三才中"天"的部分。

    修复: 当 TimeContext 可用时, 注入元会运世/值年卦/节气/六亲信息,
    让 UnifiedField 获得完整的"天时"上下文。
    """
    ctx = agent.calendar.calendar_context(year=year, month=month, day=day, hour=hour,
                                           cosmic_phases=cosmic_phases)
    # P1-1: GanzhiTime 干支信号接入 — 60甲子周期
    try:
        gan_signal = agent.ganzhi.time_signal()
        if gan_signal != 0.0:
            ctx["干支"] = gan_signal
    except Exception as exc:
        _log.warning("GanzhiTime.time_signal failed: %s", exc)

    # TimeContext 增强 — 注入值年卦/节气/元会运世 indices
    tc = getattr(agent, "_time_context", None)
    if tc is not None:
        try:
            # 值年卦相位 (64卦圆图位置)
            ctx["值年卦"] = 2 * math.pi * tc.value_year_hex / 64.0
            # 节气相位
            ctx["节气"] = tc.solar_term_phase
            # 元会运世索引 (作为连续特征)
            ctx["元_idx"] = tc.yuan_index / 10.0
            ctx["会_idx"] = tc.hui_index / 12.0
            ctx["运_idx"] = tc.yun_index / 30.0
            ctx["世_idx"] = tc.shi_index / 12.0
            # 中宫六亲编码 (将六亲关系转为标量)
            if tc.six_relations:
                ctx["六亲_self"] = tc.six_relations.get(5, 0)  # 中宫=我=0
        except Exception as exc:
            _log.debug("TimeContext injection failed: %s", exc)
    return ctx
