from __future__ import annotations

from zwm.core.constants import ELEMENT_CONTROL, ELEMENT_GENERATION
from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid


def time_expert(h: Hexagram, time_phase: float) -> float:
    from zwm.jepa.circular_encoder import CircularEncoder
    encoder = _get_circular_encoder()
    return encoder.time_potential(h, time_phase)


def space_expert(h: Hexagram, target_direction: int) -> float:
    from zwm.core.constants import LUOSHU_POSITIONS
    from zwm.jepa.square_encoder import FixedWeightSquareGNN

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
    complement = h.complement()
    from zwm.spectrum.interference import compute_interference
    from zwm.spectrum.frequency import FrequencySpectrum
    from zwm.spectrum.complex_phase import HexagramPhaseVector

    pv_comp = HexagramPhaseVector.from_hexagram(complement)
    spec_comp = FrequencySpectrum(pv_comp)
    result = compute_interference(spec_comp)
    return 1.0 - result.fortune_index


def narrative_expert(h: Hexagram) -> float:
    from zwm.spectrum.frequency import FrequencySpectrum, SceneSpectrum
    from zwm.spectrum.complex_phase import HexagramPhaseVector

    pv_main = HexagramPhaseVector.from_hexagram(h)
    main = FrequencySpectrum(pv_main)
    inter = FrequencySpectrum(pv_main.mutate(0b000110))
    evolved = FrequencySpectrum(pv_main.mutate(0b000001))
    reversed_ = FrequencySpectrum(pv_main.reverse())
    complement = FrequencySpectrum(pv_main.complement())

    scene = SceneSpectrum(main, inter, evolved, reversed_, complement)
    return scene.narrative_coherence()


_circular_encoder = None


def _get_circular_encoder():
    global _circular_encoder
    if _circular_encoder is None:
        from zwm.jepa.circular_encoder import CircularEncoder
        _circular_encoder = CircularEncoder()
    return _circular_encoder
