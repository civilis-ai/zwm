from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field

import numpy as np


DEFAULT_DIM: int = 10_000


def generate_bipolar_vector(dim: int = DEFAULT_DIM) -> np.ndarray:
    rng = np.random.default_rng(secrets.randbits(128))
    return np.where(rng.random(dim) < 0.5, -1, 1).astype(np.int8)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.multiply(a, b, dtype=np.int8)


def bundle(*vectors: np.ndarray) -> np.ndarray:
    stacked = np.stack(vectors)
    summed = np.sum(stacked, axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return bind(a, b)


def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    return np.roll(v, shift)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.astype(np.float64), b.astype(np.float64))
           / (np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(b.astype(np.float64))))


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


class VSACodebook:
    def __init__(self, dim: int = DEFAULT_DIM, seed: int = 42) -> None:
        self._dim = dim
        rng = np.random.default_rng(seed)
        self._trigram_vectors: dict[int, np.ndarray] = {
            i: np.where(rng.random(dim) < 0.5, -1, 1).astype(np.int8)
            for i in range(8)
        }
        self._hexagram_vectors: dict[int, np.ndarray] = {}
        for upper_idx in range(8):
            for lower_idx in range(8):
                fuxi_idx = lower_idx * 8 + upper_idx
                upper_shifted = permute(self._trigram_vectors[upper_idx], shift=3)
                self._hexagram_vectors[fuxi_idx] = bind(
                    upper_shifted,
                    self._trigram_vectors[lower_idx],
                )
        self._codon_vectors: dict[str, np.ndarray] = {}
        codon_bases = {"U": 0, "C": 1, "A": 2, "G": 3}
        for codon_idx, codon in self._build_codon_map().items():
            self._codon_vectors[codon] = self._hexagram_vectors.get(
                codon_idx,
                np.where(rng.random(dim) < 0.5, -1, 1).astype(np.int8),
            )

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _build_codon_map() -> dict[int, str]:
        from zwm.core.constants import CODON_TABLE
        return CODON_TABLE

    def encode_hexagram(self, bits: int) -> np.ndarray:
        return self._hexagram_vectors[bits]

    def encode_trigram(self, index: int) -> np.ndarray:
        return self._trigram_vectors[index]

    def decode_to_hexagram(self, vector: np.ndarray) -> int:
        best_idx = 0
        best_sim = -2.0
        for idx, hv in self._hexagram_vectors.items():
            sim = cosine_similarity(vector, hv)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        return best_idx

    def decode_to_codon(self, vector: np.ndarray) -> str:
        best_codon = "UUU"
        best_sim = -2.0
        for codon, cv in self._codon_vectors.items():
            sim = cosine_similarity(vector, cv)
            if sim > best_sim:
                best_sim = sim
                best_codon = codon
        return best_codon


@dataclass
class VSAEpisode:
    hexagram_vector: np.ndarray
    context_vector: np.ndarray
    outcome_vector: np.ndarray
    reward: float = 0.0
    timestamp: float = 0.0

    @property
    def bundled_episode(self) -> np.ndarray:
        return bundle(
            self.hexagram_vector,
            self.context_vector,
            self.outcome_vector,
        )


@dataclass
class VSAMemoryBuffer:
    capacity: int = 1000
    episodes: list[VSAEpisode] = field(default_factory=list)
    consolidated: list[np.ndarray] = field(default_factory=list)

    def add(self, episode: VSAEpisode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def query(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[VSAEpisode, float]]:
        scored = [
            (ep, cosine_similarity(query_vector, ep.bundled_episode))
            for ep in self.episodes
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def consolidate(self, threshold: float = 0.5) -> None:
        for ep in self.episodes:
            if ep.reward >= threshold:
                self.consolidated.append(ep.bundled_episode)
