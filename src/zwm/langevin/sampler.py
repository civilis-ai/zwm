from __future__ import annotations

import math

import numpy as np

from zwm.core.hexagram import Hexagram, hexagram_from_bits
from zwm.langevin.score import score_surface, total_score_gradient


class LangevinSampler:
    def __init__(
        self,
        step_size: float = 0.1,
        noise_scale: float = 0.05,
        num_steps: int = 100,
        temperature_init: float = 1.0,
        cooling_rate: float = 0.95,
    ) -> None:
        self._epsilon = step_size
        self._noise_scale = noise_scale
        self._num_steps = num_steps
        self._temperature_init = temperature_init
        self._cooling_rate = cooling_rate

    def sample(
        self,
        h_current: Hexagram,
        h_target: Hexagram | None = None,
    ) -> list[tuple[Hexagram, float]]:
        rng = np.random.default_rng()
        pv = h_current.normal_order
        trajectory: list[tuple[Hexagram, float]] = []

        phase = np.array([
            h_current.lines[i].phase * math.pi for i in range(6)
        ], dtype=np.float32)

        temperature = self._temperature_init

        for step in range(self._num_steps):
            h_curr = hexagram_from_bits(self._continuous_to_bits(phase))

            grad = total_score_gradient(h_curr, h_target)

            noise = rng.normal(0, self._noise_scale * math.sqrt(temperature), 6)
            phase = phase + self._epsilon * grad + noise.astype(np.float32)

            if step % 20 == 0:
                best_h = hexagram_from_bits(self._continuous_to_bits(phase))
                best_score = score_surface(best_h, h_target)
                trajectory.append((best_h, best_score))

            temperature *= self._cooling_rate

        final_h = hexagram_from_bits(self._continuous_to_bits(phase))
        final_score = score_surface(final_h, h_target)
        trajectory.append((final_h, final_score))

        return sorted(trajectory, key=lambda x: x[1], reverse=True)

    def _continuous_to_bits(self, phase: np.ndarray) -> int:
        bits = 0
        for i in range(6):
            phi_mod = phase[i] % (2 * math.pi)
            if phi_mod < math.pi / 2 or phi_mod > 3 * math.pi / 2:
                bits |= 1 << (5 - i)
        return bits

    def top_k_mutations(
        self,
        h_current: Hexagram,
        k: int = 5,
    ) -> list[tuple[Hexagram, int, float]]:
        results: list[tuple[Hexagram, int, float]] = []
        for mask in range(1, 64):
            h_mutated = h_current.mutate(mask)
            score = score_surface(h_mutated)
            results.append((h_mutated, mask, score))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]
