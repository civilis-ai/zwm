from __future__ import annotations

import math
from dataclasses import dataclass, field

from zwm.spectrum.complex_phase import HexagramPhaseVector


@dataclass(frozen=True, slots=True)
class FrequencySpectrum:
    phase_vector: HexagramPhaseVector
    base_frequency: float = 1.0

    def evaluate(self, t: float) -> complex:
        result = 0 + 0j
        for n, (phase, weight) in enumerate(
            zip(self.phase_vector.phases, self.phase_vector.weights), start=1
        ):
            omega = n * self.base_frequency
            result += weight * phase.value * (math.cos(omega * t) + 1j * math.sin(omega * t))
        return result

    def amplitude(self, t: float) -> float:
        return abs(self.evaluate(t))

    def phase_angle(self, t: float) -> float:
        val = self.evaluate(t)
        return math.atan2(val.imag, val.real)

    def resonance(self) -> float:
        return abs(self.evaluate(0.0))

    def resonance_components(self) -> tuple[float, float]:
        total = 0 + 0j
        for n, (phase, weight) in enumerate(
            zip(self.phase_vector.phases, self.phase_vector.weights), start=1
        ):
            total += weight * phase.value
        return (total.real, total.imag)

    def harmonic_profile(self) -> list[tuple[int, float]]:
        return [
            (n, abs(w * p.value))
            for n, (p, w) in enumerate(
                zip(self.phase_vector.phases, self.phase_vector.weights), start=1
            )
        ]

    def cross_spectrum(self, other: FrequencySpectrum, t: float = 0.0) -> complex:
        f_self = self.evaluate(t)
        f_other = other.evaluate(t)
        return f_self * f_other.conjugate()

    def coherence(self, other: FrequencySpectrum) -> float:
        return abs(self.cross_spectrum(other, 0.0))

    def __repr__(self) -> str:
        return f"Spectrum(0b{self.phase_vector.to_bits():06b}, r={self.resonance():.3f})"


@dataclass(frozen=True, slots=True)
class SceneSpectrum:
    main: FrequencySpectrum
    inter: FrequencySpectrum
    evolved: FrequencySpectrum
    reversed_: FrequencySpectrum
    complement: FrequencySpectrum
    weights: tuple[float, float, float, float, float] = (0.35, 0.20, 0.25, 0.10, 0.10)

    def evaluate(self, t: float) -> complex:
        w_main, w_inter, w_evolv, w_revrs, w_compl = self.weights
        return (
            w_main * self.main.evaluate(t)
            + w_inter * self.inter.evaluate(t)
            + w_evolv * self.evolved.evaluate(t)
            + w_revrs * self.reversed_.evaluate(t)
            + w_compl * self.complement.evaluate(t)
        )

    def narrative_coherence(self) -> float:
        return 1.0 - abs(
            self.main.resonance()
            - self.complement.resonance()
        ) / max(self.main.resonance(), self.complement.resonance(), 1e-10)

    def __repr__(self) -> str:
        return f"SceneSpectrum(coherence={self.narrative_coherence():.3f})"
