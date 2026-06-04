from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np


class EpisodicStore:
    def __init__(self, db_path: str = "zwm_episodes.db") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
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
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp DESC)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcome ON episodes(outcome_label)
        """)
        self._conn.commit()

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
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def query_by_outcome(self, outcome: str, limit: int = 50) -> list[dict]:
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
        from zwm.hexaembed.vsa import cosine_similarity

        recent = self.query_recent(1000)
        scored = []
        for ep in recent:
            if ep["encoded_vector"] is not None:
                stored_vec = np.frombuffer(ep["encoded_vector"], dtype=np.int8)
                sim = cosine_similarity(query_vec, stored_vec)
                scored.append((ep, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:limit]]

    def count(self) -> int:
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
        self._conn.close()


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
        self.save()
