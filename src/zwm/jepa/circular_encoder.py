from __future__ import annotations

import math

import numpy as np

from zwm.core.hexagram import Hexagram


class CircularEncoder:
    def __init__(self) -> None:
        self._phase_cache: dict[int, float] = {}
        for h in range(64):
            from zwm.core.hexagram import hexagram_from_bits
            from zwm.spectrum.complex_phase import HexagramPhaseVector
            pv = HexagramPhaseVector.from_bits(h)
            spec_sum = pv.weighted_sum()
            phase = math.atan2(spec_sum.imag, spec_sum.real)
            if phase < 0:
                phase += 2 * math.pi
            self._phase_cache[h] = phase

    def phase_of(self, hexagram: Hexagram) -> float:
        return self._phase_cache[hexagram.normal_order]

    def phase_gap(self, h1: Hexagram, h2: Hexagram) -> float:
        p1 = self.phase_of(h1)
        p2 = self.phase_of(h2)
        diff = abs(p1 - p2)
        return min(diff, 2 * math.pi - diff)

    def time_potential(self, hexagram: Hexagram, current_phase: float) -> float:
        h_phase = self.phase_of(hexagram)
        gap = abs(h_phase - current_phase)
        gap = min(gap, 2 * math.pi - gap)
        return float(1.0 - gap / math.pi)

    def solar_term_index(self, hexagram: Hexagram) -> int:
        phase = self.phase_of(hexagram)
        return int(phase / (2 * math.pi) * 24) % 24

    def arc_for_direction(self, direction: int, num_segments: int = 8) -> list[int]:
        segment_size = 64 // num_segments
        start = (direction - 1) * segment_size
        return [i % 64 for i in range(start, start + segment_size)]

    def encode(self, hexagram: Hexagram) -> np.ndarray:
        phase = self.phase_of(hexagram)
        return np.array([
            math.cos(phase),
            math.sin(phase),
            math.cos(2 * phase),
            math.sin(2 * phase),
            math.cos(3 * phase),
            math.sin(3 * phase),
            math.cos(4 * phase),
            math.sin(4 * phase),
            math.cos(5 * phase),
            math.sin(5 * phase),
            math.cos(6 * phase),
            math.sin(6 * phase),
            phase / (2 * math.pi),
        ], dtype=np.float32)
