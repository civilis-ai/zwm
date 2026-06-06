from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from zwm.core.constants import ELEMENT_CONTROL, ELEMENT_GENERATION, TRIGRAM_ELEMENTS
from zwm.core.hexagram import Hexagram, fuxi_square_hexagram


def hexagram_square_features(h: Hexagram) -> np.ndarray:
    """The 12 hand-engineered structural features of a hexagram.

    Shared by both the fixed-weight and the learnable square encoders so the two
    operate on an identical, deterministic featurisation of the hexagram. The
    learnable encoder then learns a projection of these from data, instead of
    relying on a frozen random projection.
    """
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


def circular_phase_vector(phase: float) -> np.ndarray:
    """13-dim deterministic harmonic encoding of a time phase (6 harmonics + raw)."""
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


class FixedWeightSquareGNN:
    """Frozen structured-feature encoder: 12 hand features -> 64-dim projection.

    Kept as a deterministic, non-learning baseline (random fixed projection of
    structured features — a reservoir-style front-end). For representation
    learning *from data*, use ``LearnableSquareGNN`` instead; it shares the same
    feature function and output shape but trains end-to-end with the JEPA.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2) -> None:
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._weights = self._build_fixed_weights()
        # P1-10: Cache the 8x8 grid embeddings — they are deterministic
        # (the weights are frozen), so recomputing on every call is
        # wasted work.  The first ``encode_grid`` call computes them;
        # subsequent calls return the cached dict.
        self._grid_cache: dict[tuple[int, int], np.ndarray] | None = None

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
        return hexagram_square_features(h)

    def encode_hexagram(self, h: Hexagram) -> np.ndarray:
        x = self._hexagram_features(h)
        for w in self._weights[:-1]:
            x = np.tanh(x @ w)
        return x @ self._weights[-1]

    def encode_grid(self) -> dict[tuple[int, int], np.ndarray]:
        if self._grid_cache is None:
            embeddings: dict[tuple[int, int], np.ndarray] = {}
            for row in range(8):
                for col in range(8):
                    h = fuxi_square_hexagram(row, col)
                    embeddings[(row, col)] = self.encode_hexagram(h)
            self._grid_cache = embeddings
        return self._grid_cache

    def message_passing(
        self,
        hexagram: Hexagram,
        direction: Optional[int] = None,  # noqa: ARG002 — reserved for directional GNN extension
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

    # Uniform interface shared with LearnableSquareGNN.
    def embed(self, hexagram: Hexagram) -> np.ndarray:
        return self.message_passing(hexagram)


class LearnableSquareGNN(nn.Module):
    """Learnable structured-feature encoder: 12 hand features -> 64-dim, trained.

    A real ``nn.Module`` whose parameters are registered in the JEPA optimiser
    and updated by backprop on the world-model loss. This is the end-to-end
    representation-learning path — the hexagram embedding is *learned from
    experience*, not a frozen random projection.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)

    def embed(self, hexagram: Hexagram) -> np.ndarray:
        """Eval-path embedding (no grad), numpy — matches FixedWeightSquareGNN."""
        feats = torch.from_numpy(hexagram_square_features(hexagram))
        device = next(self.net.parameters()).device
        with torch.no_grad():
            z = self.net(feats.to(device))
        return z.cpu().numpy().astype(np.float32)

    def embed_train(self, hexagram: Hexagram) -> torch.Tensor:
        """Training-path embedding: returns a Tensor with grad attached."""
        feats = torch.from_numpy(hexagram_square_features(hexagram))
        device = next(self.net.parameters()).device
        return self.net(feats.to(device))


class SquareCircularJoint:
    """Joins a 64-dim square embedding with a 13-dim circular phase -> 77-dim.

    The square encoder may be the frozen ``FixedWeightSquareGNN`` or the
    ``LearnableSquareGNN``; both expose ``embed(h) -> np[64]``, so the joint is
    agnostic and always returns a 77-dim world vector.
    """

    def __init__(self, square_encoder: FixedWeightSquareGNN | LearnableSquareGNN) -> None:
        self._square_encoder = square_encoder
        self._progression_angle: float = 0.0

    @property
    def progression_angle(self) -> float:
        return self._progression_angle

    @property
    def square_encoder(self) -> FixedWeightSquareGNN | LearnableSquareGNN:
        return self._square_encoder

    def encode(
        self,
        hexagram: Hexagram,
        time_phase: float,
    ) -> np.ndarray:
        z_s = self._square_encoder.embed(hexagram)
        z_t = self._circular_phase_to_vector(time_phase)
        self._progression_angle = time_phase
        return np.concatenate([z_s, z_t])

    def encode_train(
        self,
        hexagram: Hexagram,
        time_phase: float,
    ) -> torch.Tensor:
        """Training-path: returns a 77-dim torch Tensor with grad attached."""
        from torch import from_numpy, cat
        z_s = self._square_encoder.embed_train(hexagram)
        z_t = from_numpy(self._circular_phase_to_vector(time_phase))
        if z_s.device != z_t.device:
            z_t = z_t.to(z_s.device)
        self._progression_angle = time_phase
        return cat([z_s, z_t])

    # Raw feature components — used by the end-to-end training path so gradients
    # can flow into a LearnableSquareGNN (the 64-dim square part), while the
    # 13-dim circular part stays deterministic.
    @staticmethod
    def square_features(hexagram: Hexagram) -> np.ndarray:
        return hexagram_square_features(hexagram)

    def _circular_phase_to_vector(self, phase: float) -> np.ndarray:
        return circular_phase_vector(phase)
