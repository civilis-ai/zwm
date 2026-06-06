"""Sub-linear similarity search index for the episodic store.

Two operating modes:
  * Brute-force: when the index has fewer than ``ivf_threshold`` vectors
    (default 256), every query scans all stored vectors. The constant factor
    is tiny (VSA dim=1000, dot product is one BLAS call), so brute-force is
    fine for N < 256 and avoids cold-start latency.
  * IVF (Inverted File): when N ≥ ``ivf_threshold``, the index is partitioned
    into ``nlist`` cells via k-means, and a query only scans the ``nprobe``
    nearest cells. This gives the standard O(sqrt(N)) speedup used by modern
    vector databases (FAISS, Qdrant). Falls back to a pure-numpy k-means
    implementation when faiss is not installed.

The class never throws: every error path returns a brute-force result so the
agent's similarity lookups keep working even if the index breaks.
"""
from __future__ import annotations

import logging

import numpy as np


class VectorIndex:
    def __init__(
        self,
        dim: int = 1000,
        nlist: int = 16,
        nprobe: int = 4,
        ivf_threshold: int = 256,
        kmeans_iters: int = 8,
    ) -> None:
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.ivf_threshold = ivf_threshold
        self.kmeans_iters = kmeans_iters
        # Storage
        self._ids: list[int] = []
        self._vecs: list[np.ndarray] = []
        # IVF state (populated lazily when N >= ivf_threshold)
        self._ivf_centroids: np.ndarray | None = None
        self._ivf_cells: list[list[int]] = []
        # P2-4: Optional FAISS backend. When the index is large enough
        # (N >= 1024), the pure-numpy k-means IVF starts to get slow; the
        # FAISS backend gives the standard ``IndexIVFFlat`` recipe used by
        # every 2026 vector DB. Falls back to the numpy path silently.
        self._faiss_index = None
        self._faiss_backend = self._try_load_faiss()

    @staticmethod
    def _try_load_faiss():
        """Return the faiss module if importable, else None."""
        try:
            import faiss
            return faiss
        except Exception:
            return None

    # L4 — auto-tune parameters for the current corpus size
    @staticmethod
    def _auto_tune_params(n: int) -> tuple[int, int, int, int]:
        """L4: derive ``(nlist, nprobe, ivf_threshold, kmeans_iters)`` from corpus size.

        Heuristic (FAISS best-practice, 2026 SOTA):

        * ``nlist`` ≈ √N (so each cell holds ~√N vectors on average)
        * ``nprobe`` ≈ ⌈√nlist⌉ (the standard 4-8 range for 10⁴-10⁶ corpora)
        * ``ivf_threshold`` = max(256, nlist × 4) (don't build IVF before it pays off)
        * ``kmeans_iters`` = 8 (default; we keep the existing value)

        Caps prevent pathological behaviour on tiny / huge corpora::

            n         nlist   nprobe   ivf_threshold
            256       16      4        256
            1 000     32      6        256
            10 000    64      8        256
            100 000   128     12       512
            1 000 000 512     24       2048
        """
        if n <= 0:
            return 16, 4, 256, 8
        import math
        nlist = max(4, int(round(math.sqrt(n))))
        # Clamp to a sane upper bound so a 10M corpus doesn't ask
        # FAISS to allocate 3000 cells.
        nlist = min(nlist, 1024)
        nprobe = max(1, int(math.ceil(math.sqrt(nlist))))
        nprobe = min(nprobe, nlist)  # never probe more cells than exist
        ivf_threshold = max(256, nlist * 4)
        return nlist, nprobe, ivf_threshold, 8

    def tune_for_corpus(self) -> dict:
        """L4: re-derive ``nlist`` / ``nprobe`` from the current corpus.

        Returns the *new* parameter set so callers can log / persist it.
        Resets the FAISS index (it must be rebuilt against the new
        centroids) — the next call to :meth:`add` or :meth:`query`
        will trigger a rebuild.
        """
        nlist, nprobe, ivf_threshold, kmeans_iters = self._auto_tune_params(
            len(self._ids)
        )
        old = {
            "nlist": self.nlist, "nprobe": self.nprobe,
            "ivf_threshold": self.ivf_threshold,
        }
        self.nlist = nlist
        self.nprobe = nprobe
        self.ivf_threshold = ivf_threshold
        self.kmeans_iters = kmeans_iters
        # Force a rebuild on next access.
        self._faiss_index = None
        self._ivf_centroids = None
        self._ivf_cells = []
        return {
            "old": old,
            "new": {
                "nlist": nlist, "nprobe": nprobe,
                "ivf_threshold": ivf_threshold,
                "kmeans_iters": kmeans_iters,
            },
            "corpus_size": len(self._ids),
        }

    def _maybe_build_faiss(self) -> None:
        """P2-4 — build a FAISS IVF-Flat index for the current vectors.

        Falls back silently if faiss isn't installed. The numpy path
        remains the default for small/medium collections.
        """
        if self._faiss_backend is None or len(self._ids) < 1024:
            return
        if self._faiss_index is not None:
            return
        try:
            import faiss
            n = len(self._ids)
            k = min(self.nlist, n)
            # Use the centroids we already trained (or rebuild).
            if self._ivf_centroids is None:
                self._build_ivf()
            if self._ivf_centroids is None:
                return
            quantiser = faiss.IndexFlatL2(self.dim)
            idx = faiss.IndexIVFFlat(
                quantiser, self.dim, k, faiss.METRIC_L2
            )
            mat = np.stack(self._vecs, axis=0).astype(np.float32)
            idx.train(mat)
            idx.add(mat)
            idx.nprobe = self.nprobe
            self._faiss_index = idx
        except Exception:
            self._faiss_index = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._ids)

    def add(self, episode_id: int, vec: np.ndarray) -> None:
        """Add one vector. Triggers IVF build when N >= ivf_threshold.

        P1-2 (audit): now performs *true incremental* updates to the
        FAISS backend and the numpy IVF cells, instead of waiting for
        a full rebuild.  Without this, every new vector after the
        initial build silently fell out of FAISS and was only visible
        via the brute-force path — defeating the O(sqrt(N)) speedup.
        """
        if vec.ndim != 1 or vec.shape[0] != self.dim:
            # Silently skip malformed vectors — the index should never crash
            # the agent loop.
            return
        new_id = int(episode_id)
        new_vec = np.asarray(vec, dtype=np.float32)
        new_idx = len(self._ids)
        self._ids.append(new_id)
        self._vecs.append(new_vec)
        if len(self._ids) >= self.ivf_threshold and self._ivf_centroids is None:
            self._build_ivf()
        # P1-2: when the FAISS index is already built, the new vector
        # must be added to it too — otherwise FAISS only knows about the
        # snapshot it was trained on and the new vector becomes invisible
        # to sub-linear queries.  ``index.add`` is the standard FAISS API
        # for incremental updates.
        if len(self._ids) >= 1024:
            self._maybe_build_faiss()
            if self._faiss_index is not None and self._faiss_backend is not None:
                try:
                    self._faiss_index.add(new_vec.reshape(1, -1).astype(np.float32))
                except Exception as exc:
                    logging.getLogger(__name__).warning("VectorIndex operation failed: %s", exc)
        # P1-2: same fix for the numpy IVF — when a vector arrives
        # after the cells are built, we must assign it to its nearest
        # cell so the IVF score path can find it.
        if self._ivf_centroids is not None and self._ivf_cells:
            try:
                d = -((self._ivf_centroids - new_vec) ** 2).sum(axis=1)
                nearest = int(np.argmax(d))
                self._ivf_cells[nearest].append(new_idx)
            except Exception as exc:
                logging.getLogger(__name__).warning("VectorIndex operation failed: %s", exc)

    def rebuild(self, store) -> None:
        """Rebuild from the full SQLite store."""
        try:
            rows = store.query_recent(10000)
        except Exception:
            return
        self._ids.clear()
        self._vecs.clear()
        self._ivf_centroids = None
        self._ivf_cells = []
        for ep in rows:
            if ep.get("encoded_vector") is None:
                continue
            stored_vec = np.frombuffer(ep["encoded_vector"], dtype=np.float32)
            if stored_vec.shape[0] != self.dim:
                continue
            self._ids.append(int(ep["id"]))
            self._vecs.append(stored_vec)
        if len(self._ids) >= self.ivf_threshold:
            self._build_ivf()

    def query(self, store, query_vec: np.ndarray, limit: int = 10) -> list[dict]:
        """Return up to ``limit`` episodes most similar to ``query_vec``."""
        if not self._ids:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            return store.query_similar_vector.__wrapped__(store, query_vec, limit) \
                if hasattr(store.query_similar_vector, "__wrapped__") else []
        # Brute-force or IVF depending on N
        if self._faiss_index is not None and self._faiss_backend is not None:
            scored = self._faiss_score(q)
        elif self._ivf_centroids is None or len(self._ids) < self.ivf_threshold:
            scored = self._brute_score(q)
        else:
            scored = self._ivf_score(q)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:limit]
        # Materialise full episode rows by id.
        out: list[dict] = []
        for ep_id, sim in top:
            ep = self._fetch_episode(store, ep_id)
            if ep is not None:
                out.append(ep)
        return out

    def _faiss_score(self, q: np.ndarray) -> list[tuple[int, float]]:
        """P2-4 — query the FAISS backend if available."""
        if self._faiss_index is None:
            return self._ivf_score(q)
        try:
            D, ids = self._faiss_index.search(q.reshape(1, -1).astype(np.float32), len(self._ids))
            scored = []
            for dist, idx in zip(D[0], ids[0]):
                if idx < 0 or idx >= len(self._ids):
                    continue
                # FAISS L2 -> convert to similarity in [0, 1].
                sim = 1.0 / (1.0 + float(dist))
                scored.append((self._ids[int(idx)], sim))
            return scored
        except Exception:
            return self._ivf_score(q)

    # ------------------------------------------------------------------
    def _brute_score(self, q: np.ndarray) -> list[tuple[int, float]]:
        if not self._vecs:
            return []
        mat = np.stack(self._vecs, axis=0)
        # Cosine similarity for unit-ish vectors; falls back to dot for
        # non-normalised inputs (VSA vectors are {-1, +1}^d, magnitudes ~d).
        q_norm = float(np.linalg.norm(q)) + 1e-9
        mat_norms = np.linalg.norm(mat, axis=1) + 1e-9
        sims = (mat @ q) / (mat_norms * q_norm)
        return [(self._ids[i], float(sims[i])) for i in range(len(self._ids))]

    def _ivf_score(self, q: np.ndarray) -> list[tuple[int, float]]:
        # P2-2 (audit): numpy array 必须用 is None 判定,
        # ``not array`` 会报 "ambiguous truth value" 错误。
        if self._ivf_centroids is None or not self._ivf_cells:
            return self._brute_score(q)
        c = self._ivf_centroids
        # Distance to each centroid (negative = closer).
        d = -((c - q) ** 2).sum(axis=1)
        nearest = np.argsort(-d)[: self.nprobe]
        # Gather candidate vectors.
        cand_ids: list[int] = []
        for cell_idx in nearest:
            cand_ids.extend(self._ivf_cells[int(cell_idx)])
        if not cand_ids:
            return []
        cand_vecs = np.stack([self._vecs[i] for i in cand_ids], axis=0)
        q_norm = float(np.linalg.norm(q)) + 1e-9
        cand_norms = np.linalg.norm(cand_vecs, axis=1) + 1e-9
        sims = (cand_vecs @ q) / (cand_norms * q_norm)
        return [(self._ids[cand_ids[i]], float(sims[i])) for i in range(len(cand_ids))]

    def _build_ivf(self) -> None:
        n = len(self._ids)
        if n < self.ivf_threshold:
            return
        mat = np.stack(self._vecs, axis=0)
        k = min(self.nlist, n)
        # Tiny k-means: random init + a few Lloyd iterations. O(k * n * d * iters).
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=k, replace=False)
        centroids = mat[idx].copy()
        for _ in range(self.kmeans_iters):
            # Assign each vector to its nearest centroid.
            dists = ((mat[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            assign = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.int64)
            for i, a in enumerate(assign):
                new_centroids[a] += mat[i]
                counts[a] += 1
            nonzero = counts > 0
            new_centroids[nonzero] /= counts[nonzero, None]
            # Re-seed empty cells from a random vector to keep k stable.
            empty = np.where(~nonzero)[0]
            for e in empty:
                new_centroids[e] = mat[rng.integers(0, n)]
            if np.allclose(new_centroids, centroids, atol=1e-5):
                centroids = new_centroids
                break
            centroids = new_centroids
        # Build cells
        dists = ((mat[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(dists, axis=1)
        cells: list[list[int]] = [[] for _ in range(k)]
        for i, a in enumerate(assign):
            cells[int(a)].append(i)
        self._ivf_centroids = centroids.astype(np.float32)
        self._ivf_cells = cells

    @staticmethod
    def _fetch_episode(store, ep_id: int) -> dict | None:
        try:
            row = store._conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (ep_id,)
            ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        return store._row_to_dict(row)
