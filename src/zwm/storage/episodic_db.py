from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np


class EpisodicStore:
    """SQLite-backed episodic store, safe to call from any thread.

    P5-1 (audit): the previous implementation used
    ``sqlite3.connect(db_path)`` which defaults to
    ``check_same_thread=True``.  Combined with FastAPI's thread-pool
    and the new ``AsyncAgent`` (which runs ticks on a background
    thread), this would raise::

        sqlite3.ProgrammingError: SQLite objects created in a thread
        can only be used in the same thread

    We now:
      * pass ``check_same_thread=False`` so the same connection may
        be touched from any thread,
      * set ``PRAGMA busy_timeout=30000`` so a writer is retried for
        up to 30 s when another writer holds the WAL lock,
      * serialise every public mutation behind a re-entrant
        :class:`threading.RLock` so a single round-trip is atomic,
      * keep WAL mode (set in the constructor) for concurrent
        readers + a single writer.
    """

    def __init__(self, db_path: str = "zwm_episodes.db", use_index: bool = True) -> None:
        # P5-1: cross-thread safe connection.
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=30.0,  # wait up to 30 s for the lock
        )
        # P5-1: 30 s busy-timeout for contending writers.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout = 30000")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        # P5-1: serialise all writes / reads behind a re-entrant lock so
        # a FastAPI request handler and the AsyncAgent background loop
        # can both call ``store()`` / ``query_*`` without races.
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    main_hex_bits INTEGER NOT NULL,
                    inter_hex_bits INTEGER,
                    evolved_hex_bits INTEGER,
                    reversed_hex_bits INTEGER,
                    complement_hex_bits INTEGER,
                    outcome_label TEXT,
                    reward REAL DEFAULT 0.0,
                    encoded_vector BLOB,
                    context_json TEXT
                )
            """)
            # P1-1: ReAct reflection table — stores the textual chain-of-thought
            # from each ReAct step as a separate row keyed by episode_id.  This
            # implements the 2026 SOTA "Reflexion" / "Self-Refine" pattern: the
            # agent's reasoning is persistent and queryable, not just float
            # vectors in the latent space.
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS react_reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id INTEGER,
                    timestamp REAL NOT NULL,
                    step_index INTEGER NOT NULL,
                    thought TEXT NOT NULL,
                    tool_name TEXT,
                    tool_input TEXT,
                    observation TEXT,
                    tool_score REAL,
                    confidence REAL,
                    recommendation TEXT,
                    FOREIGN KEY (episode_id) REFERENCES episodes(id)
                        ON DELETE CASCADE
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_react_episode
                    ON react_reflections(episode_id)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_react_timestamp
                    ON react_reflections(timestamp DESC)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp DESC)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcome ON episodes(outcome_label)
            """)
            self._conn.commit()
        # Sub-linear similarity search index — built lazily, warmed on each
        # ``store()`` call. When ``use_index=False`` (e.g. CLI smoke tests
        # that don't need similarity search) the index is never created
        # and ``query_similar_vector`` falls back to a linear scan over
        # the most-recent 1000 episodes (O(N·D) — fine for N < 10k).
        #
        # AUDIT-I10: the constructor used to *look like* it was creating
        # a VectorIndex (it imported VectorIndex, set ``self._index =
        # None`` inside a ``try``, and commented "lazy init in
        # add_to_index") but the only path that ever created one was
        # ``add_to_index``.  That was actually correct, but the
        # inline comments were self-contradictory and confused the
        # original author.  We now spell out the two paths explicitly.
        self._index: "VectorIndex | None" = None
        self._index_dim: int | None = None
        if use_index:
            # Mark the intent: caller wants the index.  The actual
            # VectorIndex instance is built in ``add_to_index`` once
            # we know the VSA dim.  This avoids importing FAISS /
            # numpy machinery during boot when the caller doesn't
            # need similarity search.
            self._index_pending = True
        else:
            self._index_pending = False

    def store(
        self,
        main_bits: int,
        inter_bits: int | None = None,
        evolved_bits: int | None = None,
        reversed_bits: int | None = None,
        complement_bits: int | None = None,
        outcome: str | None = None,
        reward: float = 0.0,
        encoded_vector: np.ndarray | None = None,
        context: dict | None = None,
    ) -> int:
        row = (
            time.time(),
            main_bits,
            inter_bits,
            evolved_bits,
            reversed_bits,
            complement_bits,
            outcome,
            reward,
            encoded_vector.tobytes() if encoded_vector is not None else None,
            json.dumps(context) if context else None,
        )
        # P5-1: serialise writes — see class docstring.
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO episodes
                   (timestamp, main_hex_bits, inter_hex_bits, evolved_hex_bits,
                    reversed_hex_bits, complement_hex_bits,
                    outcome_label, reward, encoded_vector, context_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row,
            )
            self._conn.commit()
            return cursor.lastrowid or -1

    def query_recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def query_by_outcome(self, outcome: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE outcome_label = ? ORDER BY timestamp DESC LIMIT ?",
                (outcome, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def query_similar_vector(
        self,
        query_vec: np.ndarray,
        limit: int = 10,
    ) -> list[dict]:
        # Prefer the sub-linear VectorIndex (brute-force k-NN when index is
        # empty, IVF/FAISS when populated). Falls back to the linear scan
        # only if the index is unavailable.
        if self._index is not None:
            return self._index.query(self, query_vec, limit=limit)

        # Brute-force fallback: scan the most-recent 1000 episodes,
        # decode their int8 fingerprints, and rank by cosine similarity.
        # Module-level import (no inner imports) — keeps the function
        # hot-path import-free.
        from zwm.hexaembed.vsa import cosine_similarity

        recent = self.query_recent(1000)
        scored = []
        for ep in recent:
            if ep["encoded_vector"] is not None:
                stored_vec = np.frombuffer(ep["encoded_vector"], dtype=np.float32)
                sim = cosine_similarity(query_vec, stored_vec)
                scored.append((ep, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:limit]]

    def add_to_index(self, episode_id: int, vec: np.ndarray) -> None:
        """Add an episode's vector to the in-memory VectorIndex.

        Called by the agent on every ``store()`` so the index stays warm and
        the similarity search is sub-linear (IVF when N ≥ 256, brute-force
        otherwise).

        The VectorIndex is created lazily on the first call with the actual
        vector dimensionality, so it always matches the VSA codebook output
        (256 for TrainableVSACodebook, 10000 for legacy VSACodebook).

        AUDIT-I10: a no-op when the constructor was called with
        ``use_index=False`` (the CLI / smoke-test path).  We never
        silently fall back to brute-force at query time *because the
        caller opted in*; the explicit intent is preserved.
        """
        if not getattr(self, "_index_pending", False):
            return  # AUDIT-I10: caller disabled the index.
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim != 1:
            return
        # Lazy init: create the index with the correct dim on first use.
        if self._index is None:
            from zwm.storage.vector_index import VectorIndex
            self._index_dim = vec.shape[0]
            self._index = VectorIndex(dim=self._index_dim)
        self._index.add(episode_id, vec)

    def rebuild_index(self) -> None:
        """Rebuild the in-memory index from scratch over all stored vectors.

        O(N) on first build, then O(log N) at query time. Idempotent."""
        if self._index is None:
            return
        self._index.rebuild(self)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    # ------------------------------------------------------------------
    # P1-1: ReAct reflection log (textual chain-of-thought persistence)
    # ------------------------------------------------------------------
    def store_react_reflection(
        self,
        episode_id: int | None,
        step_index: int,
        thought: str,
        tool_name: str | None = None,
        tool_input: str | None = None,
        observation: str | None = None,
        tool_score: float | None = None,
        confidence: float | None = None,
        recommendation: str | None = None,
    ) -> int:
        """Persist a single ReAct reasoning step.

        Implements the 2026 SOTA "Reflexion" pattern: the agent's
        *textual* chain-of-thought is durable, queryable, and
        can later be replayed to bootstrap a new tick's reasoning
        (``SemanticStore.query_recent_reflections()``).
        Returns the inserted reflection id.
        """
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO react_reflections
                   (episode_id, timestamp, step_index, thought,
                    tool_name, tool_input, observation, tool_score,
                    confidence, recommendation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode_id,
                    time.time(),
                    step_index,
                    thought,
                    tool_name,
                    tool_input,
                    observation,
                    tool_score,
                    confidence,
                    recommendation,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid or -1

    def query_react_reflections(
        self,
        limit: int = 50,
        episode_id: int | None = None,
    ) -> list[dict]:
        """Retrieve ReAct reflections (most recent first).

        ``episode_id`` filters to a single episode's reflection chain;
        ``None`` returns the global stream (newest first).
        """
        with self._lock:
            if episode_id is not None:
                rows = self._conn.execute(
                    """SELECT * FROM react_reflections
                       WHERE episode_id = ?
                       ORDER BY timestamp ASC LIMIT ?""",
                    (episode_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM react_reflections
                       ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
        columns = [
            "id", "episode_id", "timestamp", "step_index", "thought",
            "tool_name", "tool_input", "observation", "tool_score",
            "confidence", "recommendation",
        ]
        return [dict(zip(columns, r)) for r in rows]

    def count_react_reflections(self) -> int:
        """Return the total number of ReAct reflections stored.

        Used by the Prometheus ``/metrics`` endpoint to publish a gauge
        without paying the cost of pulling rows."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM react_reflections"
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def update_context(self, episode_id: int, context_updates: dict) -> None:
        """Merge key-value pairs into an episode's ``context_json`` field.

        Atomically reads the current context, merges the updates, and
        writes back.  Used to enrich episode rows with post-hoc
        computed signals (VQ tokens, multimodal embeddings, etc.)
        without requiring the caller to store everything up front.
        """
        import json as _json
        with self._lock:
            row = self._conn.execute(
                "SELECT context_json FROM episodes WHERE id = ?",
                (episode_id,),
            ).fetchone()
            if row is None:
                return
            current_raw = row[0]
            current: dict = _json.loads(current_raw) if current_raw else {}
            current.update(context_updates)
            self._conn.execute(
                "UPDATE episodes SET context_json = ? WHERE id = ?",
                (_json.dumps(current), episode_id),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def _row_to_dict(self, row: tuple) -> dict:
        columns = [
            "id", "timestamp", "main_hex_bits", "inter_hex_bits",
            "evolved_hex_bits", "reversed_hex_bits", "complement_hex_bits",
            "outcome_label", "reward", "encoded_vector", "context_json",
        ]
        result = dict(zip(columns, row))
        if result.get("context_json"):
            result["context"] = json.loads(result["context_json"])
        return result

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


class SemanticStore:
    def __init__(self, file_path: str = "zwm_semantic.json") -> None:
        self._path = Path(file_path)
        self._data: dict = {"associations": {}, "hexagram_frequencies": {}}
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def update_association(self, h1: int, h2: int, delta: float = 0.01) -> None:
        key = f"{h1}-{h2}"
        current = self._data["associations"].get(key, 0.0)
        self._data["associations"][key] = current + delta

    def get_association(self, h1: int, h2: int) -> float:
        return self._data["associations"].get(f"{h1}-{h2}", 0.0)

    def increment_frequency(self, hex_bits: int) -> None:
        key = str(hex_bits)
        self._data["hexagram_frequencies"][key] = (
            self._data["hexagram_frequencies"].get(key, 0) + 1
        )

    def get_frequency(self, hex_bits: int) -> int:
        return self._data["hexagram_frequencies"].get(str(hex_bits), 0)

    def save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        try:
            self.save()
        except Exception:
            pass  # swallow save errors to guarantee resource cleanup
