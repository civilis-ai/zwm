from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HebbianAssociator:
    dim: int = 10_000
    learning_rate: float = 0.01
    associations: dict[str, float] = field(default_factory=dict)

    def strengthen(
        self,
        h1: int,
        h2: int,
        reward: float,
    ) -> None:
        key = f"{h1}-{h2}"
        delta = self.learning_rate * reward
        self.associations[key] = self.associations.get(key, 0.0) + delta

    def weaken(
        self,
        h1: int,
        h2: int,
        penalty: float = 0.005,
    ) -> None:
        key = f"{h1}-{h2}"
        current = self.associations.get(key, 0.0)
        if current > 0:
            self.associations[key] = max(0.0, current - penalty)

    def get_strength(self, h1: int, h2: int) -> float:
        return self.associations.get(f"{h1}-{h2}", 0.0)

    def update_from_episode(
        self,
        hexagram_sequence: list[int],
        reward: float,
    ) -> None:
        if reward > 0.5:
            for i in range(len(hexagram_sequence) - 1):
                self.strengthen(
                    hexagram_sequence[i],
                    hexagram_sequence[i + 1],
                    reward,
                )
        else:
            for i in range(len(hexagram_sequence) - 1):
                self.weaken(hexagram_sequence[i], hexagram_sequence[i + 1])

    def suggest_next(
        self,
        current_hex: int,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        candidates: list[tuple[int, float]] = []
        for h2 in range(64):
            strength = self.get_strength(current_hex, h2)
            if strength > 0:
                candidates.append((h2, strength))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
