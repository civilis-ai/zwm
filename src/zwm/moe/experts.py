from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from zwm.core.constants import ELEMENT_CONTROL, ELEMENT_GENERATION
from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid


def time_expert(h: Hexagram, time_phase: float) -> float:
    encoder = _get_circular_encoder()
    return encoder.time_potential(h, time_phase)


def space_expert(h: Hexagram, target_direction: int) -> float:
    from zwm.core.constants import LUOSHU_POSITIONS

    row, col = h.square_position()
    target_pos = LUOSHU_POSITIONS.get(target_direction, (1, 1))
    dist = ((row - target_pos[0] * 8 / 3) ** 2 + (col - target_pos[1] * 8 / 3) ** 2) ** 0.5
    return float(1.0 / (1.0 + dist / 4.0))


def social_expert(h: Hexagram, grid: LuoshuGrid, target_palace: int) -> float:
    from zwm.self_field.harmony import luoshu_harmony
    return luoshu_harmony(h, grid, target_palace)


def element_expert(h: Hexagram, context_element: str | None = None) -> float:
    lower_elem = h.lower_trigram.element
    upper_elem = h.upper_trigram.element

    score = 0.0
    if context_element:
        for elem in (lower_elem, upper_elem):
            if elem == context_element:
                score += 0.4
            elif ELEMENT_GENERATION.get(elem) == context_element:
                score += 0.3
            elif ELEMENT_CONTROL.get(elem) == context_element:
                score -= 0.2
    if lower_elem == upper_elem:
        score += 0.2
    return float(max(-1.0, min(1.0, score)))


def risk_expert(h: Hexagram) -> float:
    """基于互卦 (2/3/4 爻 → 下卦; 3/4/5 爻 → 上卦) 的"内势"风险。

    互卦代表事物"潜在趋势",与本卦之差 = 隐藏的结构性张力/破裂风险。
    传统易学术语: 互卦为本卦的"内忧外患"指标。P0 升级: 加入
    ``control_network`` 的五行相克张力项 — 上下卦五行如果构成
    4 步相克环路, 风险提升。"""
    from zwm.core.hexagram import hexagram_from_bits
    from zwm.spectrum.interference import compute_interference
    from zwm.spectrum.frequency import FrequencySpectrum
    from zwm.spectrum.complex_phase import HexagramPhaseVector
    from zwm.scene_field.wuxing import control_network

    # 互卦: 下卦取 2/3/4 爻; 上卦取 3/4/5 爻
    lower_tri = (
        (int(h.lines[1]) << 0)
        | (int(h.lines[2]) << 1)
        | (int(h.lines[3]) << 2)
    )
    upper_tri = (
        (int(h.lines[2]) << 0)
        | (int(h.lines[3]) << 1)
        | (int(h.lines[4]) << 2)
    )
    inter_bits = (upper_tri << 3) | lower_tri
    try:
        inter = hexagram_from_bits(inter_bits)
    except ValueError:
        return 0.0

    pv_main = HexagramPhaseVector.from_hexagram(h)
    pv_inter = HexagramPhaseVector.from_hexagram(inter)

    spec_main = FrequencySpectrum(pv_main)
    spec_inter = FrequencySpectrum(pv_inter)

    res_main = compute_interference(spec_main)
    res_inter = compute_interference(spec_inter)

    # 互卦吉度 (fortune) 与本卦吉度之差 = 隐藏张力
    fortune_gap = abs(res_main.fortune_index - res_inter.fortune_index)

    # 六爻相位差 (跨谱干涉) — 数值越大代表相变越剧烈
    main_pv = np.asarray([p.value.real for p in pv_main.phases], dtype=np.float32)
    inter_pv = np.asarray([p.value.real for p in pv_inter.phases], dtype=np.float32)
    phase_diss = float(np.mean(np.abs(main_pv - inter_pv)) / 2.0)

    # P0 — 五行相克张力项: 上下卦五行相克环路长度
    lower_elem = h.lower_trigram.element
    upper_elem = h.upper_trigram.element
    network = control_network()
    if network.get(lower_elem) == upper_elem:
        # 上卦克下卦 = 1 步相克, 高张力
        control_tension = 0.3
    elif network.get(upper_elem) == lower_elem:
        # 下卦克上卦, 同上
        control_tension = 0.3
    else:
        control_tension = 0.0

    # 风险 = 50% 隐藏吉度差距 + 35% 显性相位失谐 + 15% 相克张力
    risk = min(1.0, 0.50 * fortune_gap + 0.35 * phase_diss + 0.15 * control_tension)
    return float(risk)


def narrative_expert(h: Hexagram) -> float:
    """叙事弧 = 4 频率谱一致性 + 五行相生推进度。

    P0 升级: 加入 ``generation_chain`` 项 — 上下卦五行沿
    相生链 (木→火→土→金→水→木) 推进时, 叙事方向性更强。"""
    from zwm.spectrum.frequency import FrequencySpectrum, SceneSpectrum
    from zwm.spectrum.complex_phase import HexagramPhaseVector
    from zwm.scene_field.wuxing import generation_chain

    pv_main = HexagramPhaseVector.from_hexagram(h)
    main = FrequencySpectrum(pv_main)
    inter = FrequencySpectrum(pv_main.mutate(0b000110))
    evolved = FrequencySpectrum(pv_main.mutate(0b000001))
    reversed_ = FrequencySpectrum(pv_main.reverse())
    complement = FrequencySpectrum(pv_main.complement())

    scene = SceneSpectrum(main, inter, evolved, reversed_, complement)
    base_coh = scene.narrative_coherence()

    # P0 — 五行相生推进度
    lower_elem = h.lower_trigram.element
    upper_elem = h.upper_trigram.element
    chain = generation_chain(lower_elem)
    if upper_elem in chain[1:]:
        # 上卦在相生链上, 推进度 ≈ 1 - position
        pos = chain.index(upper_elem)
        gen_push = 1.0 / pos if pos > 0 else 0.0
    else:
        gen_push = 0.0

    # 0.75 基础一致性 + 0.25 相生推进度
    return float(max(0.0, min(1.0, 0.75 * base_coh + 0.25 * gen_push)))


_circular_encoder = None


def _get_circular_encoder():
    global _circular_encoder
    if _circular_encoder is None:
        from zwm.jepa.circular_encoder import CircularEncoder
        _circular_encoder = CircularEncoder()
    return _circular_encoder


class FineGrainedExpertNetwork(nn.Module):
    """Small expert network for fine-grained MoE: Linear(15→8→1) with GELU."""

    def __init__(self, feature_dim: int = 15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
