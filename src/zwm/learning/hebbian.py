from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class HebbianAssociator:
    """Three-factor Hebbian associator with Oja normalisation and reward modulation.

    Update rule (per (h1, h2) edge):
        Δw = η · pre · post · mod · (1 - post²/w_max) - forget · w

    * pre / post = visit counts (h1, h2)
    * mod = reward modulation factor (clipped to [-1, 1])
    * (1 - post²/w_max) = Oja's normalisation (inhibits runaway growth)
    * forget · w = weight decay (prevents memory bloat)

    The 2025/2026 "three-factor" Hebbian rule (Fremaux & Gerstner 2015,
    used in modern continual learning systems) outperforms plain Hebbian
    by a wide margin on noisy / non-stationary streams. BCM-style
    multiplicative modulation here uses ``reward`` as the third factor.
    """

    dim: int = 10_000
    learning_rate: float = 0.01
    oja_w_max: float = 5.0
    forget: float = 0.001
    associations: dict[str, float] = field(default_factory=dict)
    # P1-3: Per-pair visit counters (pre/post activity). The Oja normalisation
    # is computed from these so that the rule becomes a true BCM-modulated
    # three-factor rule rather than a pure Oja rule.
    _pre_count: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _post_count: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def _key(self, h1: int, h2: int) -> str:
        return f"{h1}-{h2}"

    def strengthen(
        self,
        h1: int,
        h2: int,
        reward: float,
    ) -> None:
        """P1-3 — three-factor update: Δw = η · pre · post · mod · (1 - post²/w_max) − forget · w.

        - ``pre`` is the (clipped, ≥1) visit count of h1.
        - ``post`` is the visit count of h2.
        - ``mod`` is reward (clipped to [-1, 1]) — BCM-style modulation.
        - ``(1 - post²/w_max)`` is the Oja normalisation: when post is
          saturated, growth is suppressed. This is what stops the
          associator from diverging.
        - ``forget · w`` is the weight-decay term — keeps the dictionary
          from growing unbounded.
        """
        self._pre_count[h1] += 1
        self._post_count[h2] += 1
        key = self._key(h1, h2)
        pre = float(self._pre_count[h1])
        post = float(self._post_count[h2])
        w_old = self.associations.get(key, 0.0)
        mod = max(-1.0, min(1.0, float(reward)))
        oja = max(0.0, 1.0 - (post * post) / max(1e-6, self.oja_w_max))
        # Three-factor + Oja normalisation + decay.
        delta = self.learning_rate * pre * post * mod * oja
        # Decay existing weight slightly so dormant edges don't grow stale.
        new_w = (1.0 - self.forget) * w_old + delta
        # Floor at 0 (associations are non-negative by default).
        self.associations[key] = max(0.0, new_w)

    def weaken(
        self,
        h1: int,
        h2: int,
        penalty: float = 0.005,
    ) -> None:
        key = self._key(h1, h2)
        current = self.associations.get(key, 0.0)
        if current > 0:
            self.associations[key] = max(0.0, current - penalty)

    def get_strength(self, h1: int, h2: int) -> float:
        return self.associations.get(self._key(h1, h2), 0.0)

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
