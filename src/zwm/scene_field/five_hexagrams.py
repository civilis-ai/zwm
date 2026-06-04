from __future__ import annotations

from dataclasses import dataclass

from zwm.core.hexagram import Hexagram


@dataclass(frozen=True, slots=True)
class FiveHexagramChain:
    main: Hexagram
    inter: Hexagram
    evolved: Hexagram
    reversed_: Hexagram
    complement: Hexagram

    @classmethod
    def from_current(
        cls,
        current: Hexagram,
        mutation_mask: int = 0,
    ) -> FiveHexagramChain:
        return cls(
            main=current,
            inter=current.interlock(),
            evolved=current.mutate(mutation_mask) if mutation_mask else current,
            reversed_=current.reverse(),
            complement=current.complement(),
        )

    @classmethod
    def with_evolution(
        cls,
        current: Hexagram,
        mutation_mask: int,
    ) -> FiveHexagramChain:
        evolved = current.mutate(mutation_mask)
        return cls(
            main=current,
            inter=current.interlock(),
            evolved=evolved,
            reversed_=evolved.reverse(),
            complement=evolved.complement(),
        )

    def narrative_coherence(self) -> float:
        from zwm.spectrum.frequency import FrequencySpectrum, SceneSpectrum
        from zwm.spectrum.complex_phase import HexagramPhaseVector

        pvs = [
            HexagramPhaseVector.from_hexagram(h)
            for h in (self.main, self.inter, self.evolved, self.reversed_, self.complement)
        ]
        specs = [FrequencySpectrum(pv) for pv in pvs]
        scene = SceneSpectrum(*specs)
        return scene.narrative_coherence()

    def to_dict(self) -> dict:
        return {
            "main": self.main.name,
            "inter": self.inter.name,
            "evolved": self.evolved.name,
            "reversed": self.reversed_.name,
            "complement": self.complement.name,
        }
