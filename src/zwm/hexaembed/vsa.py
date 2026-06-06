from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DIM: int = 10_000


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
                normal_order = upper_idx * 8 + lower_idx  # matches hexagram.py normal_order
                upper_shifted = permute(self._trigram_vectors[upper_idx], shift=3)
                self._hexagram_vectors[normal_order] = bind(
                    upper_shifted,
                    self._trigram_vectors[lower_idx],
                )

    @property
    def dim(self) -> int:
        return self._dim

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
    persist_path: str | None = None

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

    def consolidate(self, threshold: float = 0.5) -> int:
        """Move high-reward episodes into the durable consolidated store.

        Returns the number of episodes newly consolidated.  When
        ``persist_path`` is set, the consolidated store is flushed to disk
        after every call so the agent's long-term episodic memory survives
        process restarts.
        """
        added = 0
        for ep in self.episodes:
            if ep.reward >= threshold:
                self.consolidated.append(ep.bundled_episode)
                added += 1
        if added > 0:
            self._persist_consolidated()
        return added

    # ------------------------------------------------------------------
    # Persistence (P2-7): consolidate() now optionally flushes the long-term
    # memory to disk so it survives process restarts.  The on-disk format is
    # an ``.npz`` archive holding:
    #   * ``consolidated`` — stacked [N, dim] int8 array of bundled episodes
    #   * ``count``        — number of valid rows
    # The ``episodes`` short-term ring is intentionally NOT persisted —
    # it represents the most recent working memory, which the next
    # process rebuilds from the SQLite episodic store.
    # ------------------------------------------------------------------
    def _persist_consolidated(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            arr = np.stack(self.consolidated).astype(np.int8) if self.consolidated else np.zeros(
                (0, 0), dtype=np.int8
            )
            np.savez(
                self.persist_path,
                consolidated=arr,
                count=np.array([len(self.consolidated)], dtype=np.int64),
            )
        except Exception:
            # Persistence is best-effort; never break the OODA loop on IO.
            pass

    def load_persisted(self, path: str | None = None) -> int:
        """Load previously persisted consolidated episodes from disk.

        Returns the number of consolidated episodes loaded.  Safe to call
        even when no file exists yet (returns 0).
        """
        target = path if path is not None else self.persist_path
        if not target or not os.path.exists(target):
            return 0
        try:
            with np.load(target, allow_pickle=False) as data:
                arr = data["consolidated"]
                if arr.ndim == 2 and arr.shape[0] > 0:
                    # Don't duplicate on re-load
                    self.consolidated = [arr[i] for i in range(arr.shape[0])]
                    return arr.shape[0]
        except Exception:
            return 0
        return 0

    def recall_consolidated(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[np.ndarray, float]]:
        """Query the durable consolidated memory by cosine similarity."""
        if not self.consolidated:
            return []
        scored = [
            (vec, cosine_similarity(query_vector, vec))
            for vec in self.consolidated
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class TrainableVSACodebook:
    """Trainable VSA codebook using learnable embeddings (THDC-style).

    Replaces the static random bipolar vectors of VSACodebook with
    trainable torch embeddings that can be optimized end-to-end via
    backpropagation. This follows the THDC (Trainable Hyperdimensional
    Computing) approach: random initialization is replaced with learned
    embeddings, and a binary neural network optimizes class representations.

    Key advantages over static VSACodebook:
      - Representations adapt to the data distribution
      - Lower dimensionality sufficient (64 vs 10000) due to learned structure
      - End-to-end differentiable — integrates with JEPA/MoE training
    """

    def __init__(
        self,
        dim: int = 256,
        num_trigrams: int = 8,
        num_hexagrams: int = 64,
        seed: int = 42,
    ) -> None:
        self._dim = dim
        torch.manual_seed(seed)

        # Trainable embeddings (replaces random bipolar vectors)
        self._trigram_embed = nn.Embedding(num_trigrams, dim)
        self._hexagram_embed = nn.Embedding(num_hexagrams, dim)

        # Binary projection layer (THDC: one-layer BNN for class repr)
        self._binary_proj = nn.Linear(dim, dim)

        # Optimizer for end-to-end training
        self._opt = torch.optim.Adam(
            list(self._trigram_embed.parameters())
            + list(self._hexagram_embed.parameters())
            + list(self._binary_proj.parameters()),
            lr=1e-3,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def encode_hexagram(self, bits: int) -> np.ndarray:
        """Encode a hexagram as a trainable embedding vector."""
        with torch.no_grad():
            idx = torch.tensor(bits, dtype=torch.long)
            emb = self._hexagram_embed(idx)
            # Binary projection (sign activation for bipolar output)
            projected = self._binary_proj(emb)
            bipolar = torch.sign(projected)
            # Replace zeros with +1 (avoid degenerate dimensions)
            bipolar[bipolar == 0] = 1
            return bipolar.numpy().astype(np.int8)

    def encode_trigram(self, index: int) -> np.ndarray:
        """Encode a trigram as a trainable embedding vector."""
        with torch.no_grad():
            idx = torch.tensor(index, dtype=torch.long)
            emb = self._trigram_embed(idx)
            projected = self._binary_proj(emb)
            bipolar = torch.sign(projected)
            bipolar[bipolar == 0] = 1
            return bipolar.numpy().astype(np.int8)

    def decode_to_hexagram(self, vector: np.ndarray) -> int:
        """Decode a vector to the nearest hexagram by cosine similarity."""
        v = torch.from_numpy(vector.astype(np.float32))
        with torch.no_grad():
            all_embs = self._hexagram_embed.weight  # [64, dim]
            all_proj = torch.sign(self._binary_proj(all_embs))
            all_proj[all_proj == 0] = 1
            sims = torch.nn.functional.cosine_similarity(
                v.unsqueeze(0), all_proj.float()
            )
        return int(sims.argmax())

    def train_step(
        self,
        hexagram_bits: int,
        target_vector: np.ndarray,
        lr: float | None = None,
    ) -> float:
        """One gradient step to align a hexagram embedding with a target.

        Uses cosine similarity loss so the embedding moves toward the
        target direction while maintaining bipolar structure.
        """
        if lr is not None:
            for group in self._opt.param_groups:
                group["lr"] = lr

        idx = torch.tensor(hexagram_bits, dtype=torch.long)
        emb = self._hexagram_embed(idx)
        projected = self._binary_proj(emb)

        target = torch.from_numpy(target_vector.astype(np.float32))
        # Cosine similarity loss (maximize similarity = minimize 1 - cos_sim)
        loss = 1.0 - torch.nn.functional.cosine_similarity(
            projected.unsqueeze(0), target.unsqueeze(0)
        ).squeeze()

        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self._trigram_embed.parameters())
            + list(self._hexagram_embed.parameters())
            + list(self._binary_proj.parameters()),
            max_norm=1.0,
        )
        self._opt.step()
        return float(loss.detach())
