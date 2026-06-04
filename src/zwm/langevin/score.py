from __future__ import annotations

import math

import numpy as np

from zwm.core.hexagram import Hexagram
from zwm.spectrum.complex_phase import HexagramPhaseVector


def resonance_gradient(pv: HexagramPhaseVector) -> np.ndarray:
    grad = np.zeros(6, dtype=np.float32)
    weighted_sum = pv.weighted_sum()
    sign_sum = 1.0 if weighted_sum.real >= 0 else -1.0

    for k in range(6):
        phi_val = pv.phases[k].value
        weight = pv.weights[k]
        grad[k] = -weight * phi_val.imag * sign_sum

    return grad


def harmony_gradient(
    pv: HexagramPhaseVector,
    pv_target: HexagramPhaseVector,
) -> np.ndarray:
    grad = np.zeros(6, dtype=np.float32)

    for k in range(6):
        phi_self = pv.phases[k]
        phi_target = pv_target.phases[k]
        phase_diff = phi_self.phase - phi_target.phase
        weight = pv.weights[k]
        grad[k] = -weight * math.sin(phase_diff)

    return grad


def total_score_gradient(
    h: Hexagram,
    h_target: Hexagram | None = None,
    alpha_resonance: float = 0.3,
    beta_harmony: float = 0.4,
) -> np.ndarray:
    pv = HexagramPhaseVector.from_hexagram(h)
    grad = alpha_resonance * resonance_gradient(pv)

    if h_target is not None:
        pv_target = HexagramPhaseVector.from_hexagram(h_target)
        grad += beta_harmony * harmony_gradient(pv, pv_target)

    return grad


def score_surface(
    h: Hexagram,
    h_target: Hexagram | None = None,
) -> float:
    from zwm.spectrum.frequency import FrequencySpectrum
    from zwm.spectrum.interference import compute_interference

    pv = HexagramPhaseVector.from_hexagram(h)
    spec = FrequencySpectrum(pv)
    result = compute_interference(spec)

    score = result.fortune_index

    if h_target is not None:
        pv_t = HexagramPhaseVector.from_hexagram(h_target)
        harmony = 0.0
        for k in range(6):
            harmony += math.cos(pv.phases[k].phase - pv_t.phases[k].phase)
        harmony /= 6.0
        score = 0.6 * score + 0.4 * max(0.0, harmony)

    return float(score)
