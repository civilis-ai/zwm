from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComplexPhase:
    value: complex

    @classmethod
    def from_yin_yang(cls, is_yang: bool) -> ComplexPhase:
        return cls(1 + 0j if is_yang else -1 + 0j)

    @classmethod
    def from_phase(cls, phi: float) -> ComplexPhase:
        return cls(math.cos(phi) + 1j * math.sin(phi))

    @classmethod
    def from_index(cls, idx: int) -> ComplexPhase:
        return cls(1 + 0j if idx == 0 else -1 + 0j)

    @property
    def is_yang(self) -> bool:
        return abs(self.value.real - 1.0) < 1e-10

    @property
    def phase(self) -> float:
        return 0.0 if self.is_yang else math.pi

    @property
    def binary(self) -> int:
        return 1 if self.is_yang else 0

    def flip(self) -> ComplexPhase:
        return ComplexPhase(-self.value)

    def __mul__(self, other: ComplexPhase) -> ComplexPhase:
        return ComplexPhase(self.value * other.value)

    def __add__(self, other: ComplexPhase) -> ComplexPhase:
        return ComplexPhase(self.value + other.value)

    def __neg__(self) -> ComplexPhase:
        return ComplexPhase(-self.value)

    def __repr__(self) -> str:
        return f"Phase({'YANG' if self.is_yang else 'YIN'})"

    def __str__(self) -> str:
        return "⚊" if self.is_yang else "⚋"


YANG_PHASE: ComplexPhase = ComplexPhase(1 + 0j)
YIN_PHASE: ComplexPhase = ComplexPhase(-1 + 0j)


@dataclass(frozen=True, slots=True)
class HexagramPhaseVector:
    phases: tuple[ComplexPhase, ComplexPhase, ComplexPhase, ComplexPhase, ComplexPhase, ComplexPhase]
    weights: tuple[float, float, float, float, float, float] = (1.0, 0.9, 0.7, 0.5, 0.3, 0.2)

    @classmethod
    def from_hexagram(cls, hexagram) -> HexagramPhaseVector:
        from zwm.core.hexagram import Hexagram
        if isinstance(hexagram, Hexagram):
            return cls(tuple(
                ComplexPhase(line.complex_phase) for line in hexagram.lines
            ))
        raise TypeError(f"Expected Hexagram, got {type(hexagram)}")

    @classmethod
    def from_bits(cls, bits: int) -> HexagramPhaseVector:
        phases = tuple(
            YANG_PHASE if (bits >> i) & 1 else YIN_PHASE
            for i in range(6)
        )
        return cls(phases)

    def to_bits(self) -> int:
        bits = 0
        for i, p in enumerate(self.phases):
            if p.is_yang:
                bits |= 1 << i
        return bits

    def mutate(self, mask: int) -> HexagramPhaseVector:
        new_phases = tuple(
            p.flip() if (mask >> i) & 1 else p
            for i, p in enumerate(self.phases)
        )
        return HexagramPhaseVector(new_phases, self.weights)

    def reverse(self) -> HexagramPhaseVector:
        return HexagramPhaseVector(tuple(reversed(self.phases)), self.weights)

    def complement(self) -> HexagramPhaseVector:
        return HexagramPhaseVector(tuple(p.flip() for p in self.phases), self.weights)

    def weighted_sum(self) -> complex:
        return sum(
            w * p.value
            for w, p in zip(self.weights, self.phases)
        )

    def cosine_similarity(self, other: HexagramPhaseVector) -> float:
        a = self.weighted_sum()
        b = other.weighted_sum()
        dot = (a.real * b.real + a.imag * b.imag)
        norm_a = abs(a)
        norm_b = abs(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    def __repr__(self) -> str:
        bits = self.to_bits()
        return f"PhaseVector(0b{bits:06b})"
