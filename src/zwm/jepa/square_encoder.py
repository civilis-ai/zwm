from __future__ import annotations

from typing import Optional

import numpy as np

from zwm.core.constants import ELEMENT_CONTROL, ELEMENT_GENERATION, TRIGRAM_ELEMENTS
from zwm.core.hexagram import Hexagram, fuxi_square_hexagram


class FixedWeightSquareGNN:
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2) -> None:
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._weights = self._build_fixed_weights()

    def _build_fixed_weights(self) -> list[np.ndarray]:
        rng = np.random.default_rng(42)
        weights = []
        for layer in range(self._num_layers):
            in_dim = 12 if layer == 0 else self._hidden_dim
            out_dim = self._hidden_dim if layer < self._num_layers - 1 else 64
            w = rng.normal(0, 1.0 / np.sqrt(in_dim), (in_dim, out_dim)).astype(np.float32)
            weights.append(w)
        return weights

    def _hexagram_features(self, h: Hexagram) -> np.ndarray:
        lower_tri = h.lower_trigram
        upper_tri = h.upper_trigram
        lower_elem = TRIGRAM_ELEMENTS[lower_tri.index]
        upper_elem = TRIGRAM_ELEMENTS[upper_tri.index]

        features = np.zeros(12, dtype=np.float32)
        for i, line in enumerate(h.lines):
            features[i] = 1.0 if line.is_yang else -1.0
        features[6] = float(lower_tri.pre_heaven_order) / 8.0
        features[7] = float(upper_tri.pre_heaven_order) / 8.0
        features[8] = 1.0 if lower_elem == upper_elem else 0.0
        features[9] = 1.0 if ELEMENT_GENERATION.get(lower_elem) == upper_elem else 0.0
        features[10] = 1.0 if ELEMENT_CONTROL.get(lower_elem) == upper_elem else 0.0
        features[11] = float(lower_tri.index * 8 + upper_tri.index) / 64.0
        return features

    def encode_hexagram(self, h: Hexagram) -> np.ndarray:
        x = self._hexagram_features(h)
        for w in self._weights[:-1]:
            x = np.tanh(x @ w)
        return x @ self._weights[-1]

    def encode_grid(self) -> dict[tuple[int, int], np.ndarray]:
        embeddings: dict[tuple[int, int], np.ndarray] = {}
        for row in range(8):
            for col in range(8):
                h = fuxi_square_hexagram(row, col)
                embeddings[(row, col)] = self.encode_hexagram(h)
        return embeddings

    def message_passing(
        self,
        hexagram: Hexagram,
        direction: Optional[int] = None,
    ) -> np.ndarray:
        center_vec = self.encode_hexagram(hexagram)
        row, col = hexagram.square_position()

        neighbors: list[np.ndarray] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                nh = fuxi_square_hexagram(nr, nc)
                neighbors.append(self.encode_hexagram(nh))

        if not neighbors:
            return center_vec

        context = np.mean(neighbors, axis=0)
        return (center_vec + context) * 0.5


class SquareCircularJoint:
    def __init__(self, square_encoder: FixedWeightSquareGNN) -> None:
        self._square_encoder = square_encoder
        self._progression_angle: float = 0.0

    @property
    def progression_angle(self) -> float:
        return self._progression_angle

    def encode(
        self,
        hexagram: Hexagram,
        time_phase: float,
    ) -> np.ndarray:
        z_s = self._square_encoder.message_passing(hexagram)
        z_t = self._circular_phase_to_vector(time_phase)
        self._progression_angle = time_phase
        return np.concatenate([z_s, z_t])

    def _circular_phase_to_vector(self, phase: float) -> np.ndarray:
        return np.array([
            np.cos(phase),
            np.sin(phase),
            np.cos(2 * phase),
            np.sin(2 * phase),
            np.cos(3 * phase),
            np.sin(3 * phase),
            phase / (2 * np.pi),
            np.cos(4 * phase),
            np.sin(4 * phase),
            np.cos(5 * phase),
            np.sin(5 * phase),
            np.cos(6 * phase),
            np.sin(6 * phase),
        ], dtype=np.float32)
