from __future__ import annotations

import math
from dataclasses import dataclass

from zwm.spectrum.complex_phase import HexagramPhaseVector
from zwm.spectrum.frequency import FrequencySpectrum


@dataclass(frozen=True, slots=True)
class InterferenceResult:
    resonance: float
    constructive_ratio: float
    destructive_ratio: float
    dominant_harmonic: int
    phase_coherence: float

    @property
    def is_harmonious(self) -> bool:
        return self.resonance > 2.0 and self.constructive_ratio > 0.5

    @property
    def is_conflicted(self) -> bool:
        return self.resonance < 1.0 and self.destructive_ratio > 0.5

    @property
    def fortune_index(self) -> float:
        raw = (self.resonance + self.phase_coherence) / 4.0
        return max(0.0, min(1.0, raw))

    def __repr__(self) -> str:
        label = "吉" if self.is_harmonious else "凶" if self.is_conflicted else "平"
        return f"Interference({label}, r={self.resonance:.3f})"


def compute_interference(spectrum: FrequencySpectrum) -> InterferenceResult:
    components = spectrum.harmonic_profile()
    real_sum, imag_sum = spectrum.resonance_components()

    amplitude_values = [amp for _, amp in components]
    constructive_count = sum(
        1 for a, b in zip(amplitude_values, amplitude_values[1:])
        if a * b > 0
    )
    destructive_count = sum(
        1 for a, b in zip(amplitude_values, amplitude_values[1:])
        if a * b < 0
    )
    total_pairs = len(amplitude_values) - 1 or 1
    constructive_ratio = constructive_count / total_pairs
    destructive_ratio = destructive_count / total_pairs

    dominant_harmonic = max(components, key=lambda x: x[1])[0]

    phase_angles = [
        math.atan2(0, amp) if amp > 0 else math.atan2(0, -amp)
        for _, amp in components
    ]
    if len(phase_angles) >= 2:
        phase_diffs = [
            abs(phase_angles[i] - phase_angles[i - 1])
            for i in range(1, len(phase_angles))
        ]
        avg_diff = sum(phase_diffs) / len(phase_diffs)
        phase_coherence = 1.0 - (avg_diff / math.pi)
    else:
        phase_coherence = 1.0

    resonance = abs(complex(real_sum, imag_sum))

    return InterferenceResult(
        resonance=resonance,
        constructive_ratio=constructive_ratio,
        destructive_ratio=destructive_ratio,
        dominant_harmonic=dominant_harmonic,
        phase_coherence=phase_coherence,
    )


def cross_interference(
    spectrum_a: FrequencySpectrum,
    spectrum_b: FrequencySpectrum,
) -> float:
    coherence = spectrum_a.coherence(spectrum_b)
    resonance_a = spectrum_a.resonance()
    resonance_b = spectrum_b.resonance()
    if resonance_a < 1e-10 or resonance_b < 1e-10:
        return 0.0
    return coherence / math.sqrt(resonance_a * resonance_b)
